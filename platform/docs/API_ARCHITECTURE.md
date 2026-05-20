# Sentinel — Backend & API Architecture

> The current v1.0 client ships with deterministic in-memory data so the UI runs
> standalone. The architecture below describes the productionization layer.

## 1. High-Level Architecture

```
              ┌────────────────────────────────────────────┐
              │                Web Client                  │
              │  Next.js 14 · App Router · React 18        │
              └──────────────┬───────────────┬─────────────┘
                             │ HTTPS/JSON    │ WebSocket
                             ▼               ▼
                       ┌──────────┐   ┌──────────────┐
                       │ Node.js  │   │ Stream Gateway│
                       │ BFF /    │   │ (Redis Pub/Sub│
                       │ tRPC API │   │  + SSE)       │
                       └────┬─────┘   └──────┬───────┘
                            │                 │
              ┌─────────────┼─────────────────┴──────────────┐
              ▼             ▼              ▼                 ▼
        ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌───────────┐
        │ FastAPI  │  │ Data ETL │  │ Agent Router │  │ Auth /    │
        │ Quant /  │  │ Workers  │  │ (LangChain)  │  │ Billing   │
        │ Scoring  │  │ (Cron +  │  │              │  │ NextAuth  │
        │ Service  │  │  Kafka)  │  │              │  │ + Stripe  │
        └────┬─────┘  └────┬─────┘  └──────┬───────┘  └───────────┘
             │             │               │
       ┌─────┴───────┐     │       ┌───────┴────────┐
       ▼             ▼     ▼       ▼                ▼
  Postgres     TimescaleDB  Pinecone           Claude API
  (users,      (candles,    (RAG vector       (claude-opus-4-7,
  journal)     ticks)        store)            claude-haiku for
                                               cheap classify)
```

## 2. External Data Providers

| Provider | Use | Plan |
|---|---|---|
| **Polygon.io** | Real-time + historical equities | Advanced ($199/mo) |
| **Finnhub** | Earnings calendar, sentiment | Premium |
| **Alpaca** | Optional paper-trade integration | Free tier |
| **FMP** | Fundamentals, news | Premium |
| **SEC EDGAR API** | 13F, 10-K/Q, 8-K | Free |
| **FRED** | Macro (CPI, unemployment, yields) | Free |

## 3. REST API (Pro tier)

```
GET  /api/v1/stocks                       → universe snapshot
GET  /api/v1/stocks/{symbol}              → enriched stock object
GET  /api/v1/stocks/{symbol}/candles      → ?range=1Y&interval=1D
GET  /api/v1/stocks/{symbol}/score        → composite + sub-scores
GET  /api/v1/stocks/{symbol}/setup        → AI trade setup
GET  /api/v1/macro                        → 8 macro signals + regime
GET  /api/v1/sectors                      → rotation matrix
GET  /api/v1/news?symbol=NVDA&limit=20    → filtered news
POST /api/v1/agent/chat                   → { query, context } → orchestrated response
GET  /api/v1/portfolio                    → user book (auth)
POST /api/v1/journal                      → create journal entry
GET  /api/v1/journal                      → list entries
```

All POST endpoints require Bearer JWT. Rate-limited at 60 rpm (Pro) / 1000 rpm (Institutional).

## 4. WebSocket Streams

```
wss://stream.sentinel.ai/v1?token=...

Channels:
  quotes.NVDA,AMD,...           → tick updates
  macro                          → regime / signal changes
  news                           → live news firehose (filtered by user prefs)
  alerts                         → user-defined trigger fires
  agents                         → AI insight push (e.g., regime change)
```

Message envelope:

```json
{ "channel": "quotes.NVDA", "ts": 1747915800000, "data": { "p": 142.18, "c": 2.84, "v": 12_345_678 } }
```

## 5. Database Schema (Postgres + Timescale)

```sql
-- Postgres
users(id, email, hashed_pw, plan, created_at, last_login)
subscriptions(id, user_id, stripe_id, plan, status, period_end)
watchlists(id, user_id, name, symbols[])
portfolios(id, user_id, name, base_currency)
positions(id, portfolio_id, symbol, qty, cost_basis, opened_at)
journal_entries(id, user_id, symbol, side, thesis, emotion, result, notes, ts)
alerts(id, user_id, type, payload, channel, created_at)
agent_logs(id, user_id, query, response, agents_used[], ts)   -- compliance audit

-- Timescale (hypertables)
candles(symbol, ts, o, h, l, c, v)               PRIMARY KEY (symbol, ts)
ticks(symbol, ts, p, v)                           PRIMARY KEY (symbol, ts)
macro_signals(key, ts, value, state)
sector_metrics(sector, ts, rs_rank, flow, momentum)
news(id, ts, source, title, body, symbols[], sentiment, impact)
```

## 6. AI Agent Service (FastAPI + LangChain)

```python
POST /agent/run
{
  "query": "Analyze NVDA from macro to valuation",
  "user_id": "u_abc",
  "context": { "watchlist": ["NVDA","AMD"], "portfolio_id": "p_xyz" }
}

→ Router classifies intent + extracts symbol
→ Fan-out (asyncio.gather) to specialist chains:
     Macro     → RAG over FRED + recent FOMC minutes
     Sector    → calls /api/v1/sectors + ETF flow
     Fund      → RAG over latest 10-K/Q + earnings transcript
     Technical → in-process candle math (ta-lib)
     News      → vector search Pinecone (last 24h)
     Risk      → portfolio service
→ Synthesizer chain (claude-opus-4-7) merges with structured output schema
→ Stream Server-Sent Events back to client
```

## 7. Caching & Performance

| Layer | TTL | Tool |
|---|---|---|
| Quote snapshots | 1s | Redis |
| Candles (intraday) | 30s | Redis |
| Macro signals | 60s | Redis |
| Stock score | 5 min | Redis |
| News list | 10s | Redis |
| AI agent response (same query) | 60s | Redis (hash query+ctx) |
| Static fundamentals | 12h | Postgres + Redis |

## 8. Deployment

- **Hosting:** Vercel (frontend) + Fly.io / Render (Python services) + Upstash (Redis) + Neon (Postgres) + Timescale Cloud
- **Observability:** Datadog APM, OpenTelemetry, Sentry
- **CI/CD:** GitHub Actions → Vercel + Fly auto-deploy
- **Secrets:** Doppler / Vercel env
- **Compliance:** SOC2 roadmap for Institutional plan, encryption at rest (AES-256), in transit (TLS 1.3)
