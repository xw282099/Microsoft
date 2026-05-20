# Sentinel AI — US Stock Intelligence Platform

An institutional-grade, AI-driven decision terminal for US equities. Built on the
philosophy: **"Trend First. Entry Second."**

> 🚀 **The full Next.js application lives in [`./platform`](./platform).**

## Quick start

```bash
cd platform
npm install
npm run dev
# → http://localhost:3000
```

## What's inside

- 🟢 **Landing page** with hero, philosophy, agent architecture
- 📊 **Command Center** dashboard — macro gauges, top movers, news feed
- 🧭 **Macro Engine** — SPY/QQQ/VIX/yields with AI regime narrative
- 🌀 **Sector Rotation Matrix** + heat-map (11 sectors)
- 🔎 **AI Stock Scoring** screener (25 ticker universe, real growth/AI infra names)
- 📈 **Per-stock detail pages** with 5-dim score gauges + AI trade setup sidebar
- 🧠 **AI Agent Chat Terminal** orchestrating 7 specialist agents
- 🛡️ **Risk Control Center** with position-level monitoring
- 📓 **AI Trading Journal** with behavioral pattern detection
- 💳 **Pricing tiers** (Free / Pro / Institutional)

## Architecture & docs

| Document | Purpose |
|---|---|
| [`platform/docs/PRD.md`](./platform/docs/PRD.md) | Full product requirements document |
| [`platform/docs/SITEMAP.md`](./platform/docs/SITEMAP.md) | Sitemap + wireframes |
| [`platform/docs/DESIGN_SYSTEM.md`](./platform/docs/DESIGN_SYSTEM.md) | Color, typography, components, motion |
| [`platform/docs/AGENT_DATA_FLOW.md`](./platform/docs/AGENT_DATA_FLOW.md) | 7-agent orchestration and scoring math |
| [`platform/docs/API_ARCHITECTURE.md`](./platform/docs/API_ARCHITECTURE.md) | Backend, REST/WebSocket, DB schema |
| [`platform/docs/AI_OUTPUT_TEMPLATES.md`](./platform/docs/AI_OUTPUT_TEMPLATES.md) | Structured AI response templates |
| [`platform/docs/MOBILE_AND_INSTITUTIONAL.md`](./platform/docs/MOBILE_AND_INSTITUTIONAL.md) | Mobile PWA + institutional plan |

## Tech stack

- **Frontend:** Next.js 14 (App Router) · React 18 · TypeScript · Tailwind CSS · Framer Motion · Recharts · lucide-react
- **Productionization (planned):** FastAPI quant service · LangChain agent router · Claude synthesizer · Postgres + Timescale · Pinecone · Polygon / Finnhub / FMP / FRED / SEC EDGAR

## Disclaimer

For **educational and research purposes only**. Not investment advice. AI outputs may
contain errors. Users assume all investment risk. The platform does not execute trades
or hold customer funds.

---

> The legacy MSFT Streamlit demo (`app.py`, `msft.csv`) is preserved at the repo root
> for reference.
