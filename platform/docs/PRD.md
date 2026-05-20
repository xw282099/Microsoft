# Product Requirements Document
## Sentinel AI — US Stock Intelligence Platform

**Version:** 1.0
**Status:** Ready for engineering
**Owner:** Product Director, Sentinel
**Target launch:** Q3 2026

---

## 1. Vision

Build the trader's terminal of the AI era — Bloomberg-grade data density, TradingView-grade
charts, hedge-fund-grade workflow, with seven collaborating AI agents replacing what used
to require a research desk of analysts.

**One-liner:** *"AI-powered US stock intelligence — trend first, entry second."*

---

## 2. Target Users

| Persona | Need | What they want |
|---|---|---|
| **Growth stock investor** (NVDA, AMD, PLTR, META) | Conviction on entry timing | Trend regime + valuation context |
| **AI-infra investor** (GPU/DC/optical/power) | Theme-level rotation visibility | Sector flow + capex narrative |
| **Swing / Momentum trader** | High-probability setups | R:R, stop, target zones |
| **Time-constrained pro** | AI does the screening | Daily AI digest + alerts |
| **Hedge fund analyst** | Multi-agent research at scale | API + custom agents |

---

## 3. Core Philosophy (non-negotiable)

Every UX flow must reinforce the chain:

```
Macro Trend → Sector Rotation → Stock Selection → Timing → Risk Control
```

If macro is risk-off, *all* stock recommendations carry a "REGIME WARNING" badge.
If a stock's sector RS rank < 50, the AI Trade Setup defaults to `AVOID`.

---

## 4. Functional Scope (v1.0)

### 4.1 Macro Engine
- 8 macro signals (SPY, QQQ, IWM, VIX, DXY, US10Y, HY credit, Fed stance)
- Single market trend gauge (0-100) with regime label (Bull · Stage 1/2/3, Sideways, Bear)
- AI-generated daily/intraday narrative
- Regime cheat-sheet panel

### 4.2 Sector Rotation
- 11 sector universe (Semis, AI Infra, DC, Cloud SW, Cyber, Energy, Utilities, Networking, Optical, Nuclear, Storage)
- Columns: RS rank, institutional flow, momentum, 1W/1M/YTD return, AI narrative
- Heat-map view (tile size = market cap, color = % change)
- Theme generator — auto-grouped trade ideas

### 4.3 AI Stock Scoring
- Composite score 0-100, weighted: Trend 40% / Fund 28% / Val 18% / Risk 14%
- Sub-scores all visible and explainable
- Screener: filter by sector, min score, sort by any sub-score
- Sparklines per row

### 4.4 Stock Detail (the most-used page)
- 5 gauges: composite + 4 sub-scores
- 240-day price chart with EMA 20/50 overlays
- AI Summary (one paragraph)
- AI Trade Setup sidebar: bias (LONG/WAIT/AVOID), watch zone, entry zone, stop, targets, R:R, position size, rationale
- Per-agent analysis cards (Fundamentals, Technical, News, Risk)
- Fundamentals snapshot grid
- Catalysts list
- Symbol-filtered news

### 4.5 AI Agent Terminal
- Multi-agent chat orchestrator
- Auto-detects tickers and intent keywords → routes to matching agents
- Streams synthesized summary + per-agent blocks
- Suggested prompts panel
- Agent status indicators

### 4.6 Risk Control Center
- 5 portfolio-level gauges (stability, concentration, liquidity buffer, earnings tail, hedge coverage)
- Sector exposure bars with risk-level coloring
- Position-level table (weight, P&L, stop distance, days to earnings, status)
- AI risk + portfolio agent blocks
- Concentration alerts

### 4.7 AI Trading Journal
- CRUD entries (symbol, side, thesis, emotion, result, notes)
- KPIs: total entries, win rate, FOMO count, open positions
- AI behavioral analysis paragraph (auto-generated from entries)
- Trade log card list

### 4.8 Pricing / Plans
- Free, Pro ($49/mo), Institutional (custom)
- Feature comparison table

---

## 5. Non-Functional Requirements

| Requirement | Target |
|---|---|
| First Contentful Paint | < 1.2s |
| Time to Interactive (dashboard) | < 2.5s |
| Quote stream latency | < 50ms p50 |
| Mobile (≥ 375px) | Full feature parity except multi-pane charts |
| Accessibility | WCAG 2.1 AA; all charts have data-table fallback |
| Browser support | Last 2 versions of Chrome, Safari, Firefox, Edge |
| Uptime SLA (Pro) | 99.9% |

---

## 6. Out of Scope (v1.0)

- ❌ Order routing / brokerage integration
- ❌ Options chains and Greeks (planned v1.1)
- ❌ Crypto cross-market analysis (planned v2.0)
- ❌ Backtesting engine (planned v1.2)
- ❌ Mobile native apps (PWA only for v1.0)

---

## 7. Success Metrics

| Metric | 90-day target |
|---|---|
| Free → Pro conversion | ≥ 6% |
| Pro D30 retention | ≥ 70% |
| Daily Active Users / MAU | ≥ 45% |
| Median session duration | ≥ 12 min |
| AI Agent queries / user / day | ≥ 8 |
| NPS | ≥ 50 |

---

## 8. Compliance & Risk

- Every AI output ends with educational-use disclaimer.
- Trade setups are clearly labeled "AI-generated · not investment advice".
- No order execution, no fund custody — platform is research-only.
- Footer on every page reinforces disclaimer.
- Terms of Service explicitly waives investment advisory relationship.
- Audit log of all AI outputs retained for institutional plan (compliance requirement).

---

## 9. Roadmap

| Version | Highlights |
|---|---|
| **v1.0** (Q3 2026) | Core terminal — this PRD |
| **v1.1** | Options strategy AI; alerts engine; mobile PWA polish |
| **v1.2** | Backtesting engine; custom AI agent builder (Pro) |
| **v2.0** | AI Portfolio Manager; Crypto/US cross-market; institutional API GA |
| **v2.1** | AI Earnings Prediction Engine; AI Hedge Fund Dashboard |
| **v3.0** | Global macro engine; institutional flow tracking; Asia equities |
