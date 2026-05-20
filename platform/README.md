# Sentinel AI — US Stock Intelligence Platform

> **Bloomberg Terminal × TradingView × OpenAI Agent × Hedge Fund Dashboard**
>
> An institutional-grade, AI-driven decision terminal for US equities. Built on the
> philosophy: **"Trend First. Entry Second."**

---

## 1. Run it locally (60 seconds)

```bash
cd platform
npm install
npm run dev
# open http://localhost:3000
```

Production build:

```bash
npm run build
npm run start
```

The app ships **fully self-contained** — synthetic but realistic market data, deterministic
AI agent logic, no external API keys required. Swap `lib/data.ts` for real Polygon /
Finnhub / Alpaca / FMP feeds in production.

---

## 2. What's inside

| Route | Purpose |
|---|---|
| `/` | Marketing landing — Hero, philosophy, agents, CTA |
| `/dashboard` | Command Center — macro KPIs, top movers, AI scoring, news feed |
| `/macro` | Macro Engine — SPY/QQQ/VIX/yields, regime gauge, AI narrative |
| `/sectors` | Sector Rotation — RS matrix, heat-map, flow detection |
| `/stocks` | AI Scoring Screener — filter universe by composite score |
| `/stocks/[symbol]` | Stock Detail — 5-dim score, AI trade setup, agent stack |
| `/agent` | AI Agent Terminal — multi-agent chat orchestrator |
| `/risk` | Risk Control Center — exposure, concentration, position risk |
| `/journal` | AI Trading Journal — behavioral analysis of your trades |
| `/pricing` | Free / Pro / Institutional tiers |

---

## 3. Architecture

```
platform/
├── app/                    # Next.js 14 App Router pages
├── components/
│   ├── AppShell.tsx        # Sidebar + topbar chrome
│   ├── TickerTape.tsx      # Marquee scroller
│   ├── charts/             # Sparkline, PriceChart, Gauge, Heatmap
│   ├── panels/             # AgentBlockCard
│   └── ui/                 # Panel, Stat primitives
├── lib/
│   ├── data.ts             # Stock universe, sectors, news, candle generator
│   ├── scoring.ts          # Trend / Fund / Val / Risk / Composite scorers
│   ├── ai.ts               # 7-agent orchestrator + chat runtime
│   └── utils.ts            # Formatters, seeded RNG
└── docs/                   # PRD, Sitemap, Design System, API
```

### AI Agent System

Seven specialist agents collaborate through an orchestrator:

| Agent | Domain |
|---|---|
| **Macro** | Liquidity, rates, vol regime, Fed |
| **Sector Rotation** | Relative strength, institutional flow |
| **Fundamentals** | Growth, margins, FCF, valuation |
| **Technical** | Trend structure, EMA alignment, momentum |
| **News** | Filings, earnings, sentiment |
| **Risk** | Sizing, stops, hedge |
| **Portfolio** | Allocation, rebalancing |

The chat orchestrator (`lib/ai.ts:runAgents`) routes user queries by keyword and ticker
detection, runs the matching agents in parallel, and synthesizes a structured response
with a one-line summary, per-agent blocks, and a disclaimer.

### Scoring Engine

`lib/scoring.ts` implements a transparent, trend-first composite:

```
composite = trend × 0.40 + fund × 0.28 + val × 0.18 + risk × 0.14
```

Each sub-score is clamped to `[0, 100]`. Trade setups (`buildTradeSetup`) are derived
from the composite and the live price, producing watch / entry / stop / target zones,
R:R, and a suggested position size.

---

## 4. Productionization checklist

When you're ready to ship live data:

- [ ] Replace `STOCKS` in `lib/data.ts` with a live universe loader (Polygon snapshot)
- [ ] Replace `generateSeries` with a candle API (`/aggs/ticker/.../range/1/day/...`)
- [ ] Wire `MACRO_SIGNALS` to FRED / Polygon index endpoints
- [ ] Wire `NEWS` to a sentiment-scored news API (Benzinga, FMP, or your own)
- [ ] Replace `lib/ai.ts` deterministic agents with `claude-opus-4-7` calls behind a
      LangChain / Vercel AI SDK router
- [ ] Add auth (NextAuth + Postgres) and stripe-gated `/pricing` upgrade flow
- [ ] Add Redis-backed quote stream + SSE/WebSocket for live tape

---

## 5. Disclaimer

> All content is for **educational and research purposes only** and does **not**
> constitute investment advice. AI outputs may contain errors. Users assume all
> investment risk. The platform does not execute trades or hold customer funds.
