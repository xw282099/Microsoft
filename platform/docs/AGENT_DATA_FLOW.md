# AI Agent Data Flow

## 1. Orchestrator

```
User query  ──▶  Orchestrator (lib/ai.ts:runAgents)
                    │
                    ├── tokenize / intent classify
                    │      ├─ symbol detection (regex against STOCKS)
                    │      └─ keyword routing (macro|sector|news|risk|portfolio)
                    │
                    ├── parallel fan-out ─▶  Macro Agent
                    │                        Sector Rotation Agent
                    │                        Fundamentals Agent (symbol-scoped)
                    │                        Technical Agent     (symbol-scoped)
                    │                        News Agent          (filterable)
                    │                        Risk Agent
                    │                        Portfolio Agent
                    │
                    └── synthesize: { summary, blocks[], disclaimer }
```

## 2. Per-Agent Inputs / Outputs

### Macro Agent
- **In:** `MACRO_SIGNALS[]` (SPY, QQQ, IWM, VIX, DXY, US10Y, HYG, FED)
- **Logic:** count bull / bear states → tone; canned narrative bullets
- **Out:** `{ agent, headline, bullets[], tone }`

### Sector Rotation Agent
- **In:** `SECTOR_ROTATION[]`
- **Logic:** sort by RS rank → top 3 leaders, bottom 2 laggards
- **Out:** narrative per leader + laggard avoidance

### Fundamentals Agent
- **In:** `Stock` (revGrowth, epsGrowth, grossMargin, fcf, pe, peg, catalysts)
- **Logic:** thresholded explanations (growth > 18% bull, FCF > 0 healthy, PEG < 1.5 reasonable)
- **Out:** 4 bullets + tone

### Technical Agent
- **In:** `Stock` (change %) + (optional) candle series for EMA computation
- **Logic:** uses `lib/scoring.ts:trendScore` internally; produces structure narrative
- **Out:** 4 bullets + tone

### News Agent
- **In:** `NEWS[]`, optionally filtered by symbol
- **Logic:** counts sentiment splits; lists top items
- **Out:** N bullets + net tone

### Risk Agent
- **In:** `Stock` (beta, shortInterest, insiderTrend, fcf) OR no stock (book-level)
- **Logic:** size suggestion, factor-risk narrative
- **Out:** 4 bullets + tone

### Portfolio Agent
- **In:** (live book — placeholder), heuristics
- **Logic:** canned allocation suggestions
- **Out:** 3 bullets + neutral tone

## 3. Composite Scoring (`lib/scoring.ts`)

```
trendScore(stock, candles)
   = 50
   + (last > EMA20) +10
   + (last > EMA50) +10
   + (last > EMA200) +10
   + (EMA20 > EMA50) +5
   + (EMA50 > EMA200) +5
   + clamp(20d momentum × 100, -15..15)
   + change% × 1.2

fundamentalScore(stock)
   = 40
   + min(25, revGrowth × 0.4)
   + min(15, epsGrowth × 0.12)
   + min(10, (grossMargin - 40) × 0.4)
   + (fcf > 0) +5
   + (fcf > 5B) +5

valuationScore(stock)
   = peg ? 80 - peg × 18 - max(0, (pe-40) × 0.4)
         : 40 + min(40, revGrowth × 0.4)

riskScore(stock)
   = 70
   - max(0, (beta - 1) × 18)
   - max(0, (shortInterest - 2) × 4)
   + (insider = buy +8 | sell -12 | neutral 0)
   + (fcf < 0) -15

composite = trend × 0.40 + fund × 0.28 + val × 0.18 + risk × 0.14
```

## 4. Trade Setup Generator

```
if composite < 45  →  AVOID  (no entry; show watch zone only)
if composite < 60  →  WAIT   (require pullback into 0.95–0.98 × price)
else                →  LONG  (staged entry near price; stop 1.8 ATR; targets 3/6 ATR)
position size:
   composite ≥ 80 → 5% of book
   composite ≥ 60 → 3% of book
   else            → 2% of book (WAIT) or 0% (AVOID)
```

## 5. Real-Data Migration Plan

Replace these stubs with live calls:

| Stub | Real source |
|---|---|
| `STOCKS` (universe) | Polygon `/v3/reference/tickers` + `/v2/snapshot/locale/us/markets/stocks/tickers` |
| `generateSeries()` | Polygon `/v2/aggs/ticker/{sym}/range/1/day/...` |
| `MACRO_SIGNALS` | FRED + Polygon indices |
| `NEWS` | Benzinga or FMP news endpoints + LLM sentiment classifier |
| `SECTOR_ROTATION` | Computed from sector ETF candles (XLK, XLE, XLI, …) and 13F flow data |
| All agents | Claude `claude-opus-4-7` calls with RAG over filings / earnings / news |
| Trade setup | Add live ATR from candles; risk size from user's portfolio value |
