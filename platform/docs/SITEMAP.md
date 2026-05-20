# Sentinel — Sitemap & Page Wireframes

## 1. Site Map

```
/                            Marketing landing
├── /dashboard               🟢 Command Center (post-login default)
├── /macro                   Macro Engine
├── /sectors                 Sector Rotation Matrix
├── /stocks                  AI Scoring Screener
│   └── /stocks/{symbol}     Stock Detail (NVDA, AMD, AVGO, …)
├── /agent                   AI Agent Terminal
├── /risk                    Risk Control Center
├── /journal                 AI Trading Journal
├── /pricing                 Plans (Free / Pro / Institutional)
└── /auth                    Sign in / Sign up   (planned)
    ├── /settings            Account, theme, alerts
    ├── /portfolio           My positions
    └── /watchlist           Custom watchlist
```

---

## 2. Page Wireframes (text-frame)

### 2.1 Landing (`/`)
```
┌─────────────────────────────────────────────────────────────────┐
│  LOGO   Features  Agents  Workflow  Pricing       Demo  Launch │
├─────────────────────────────────────────────────────────────────┤
│  AI-Powered US Stock Intelligence Platform        ┌──────────┐ │
│  Analyze trends, rotation, risk…                  │ LIVE     │ │
│  [Start Analysis] [Ask AI] [View Sectors]         │ MACRO    │ │
│  7 Agents · 120+ Indicators · 1.2M pts/day        │ Top movers│ │
│                                                    │ NVDA +2% │ │
│                                                    └──────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  PHILOSOPHY: Trend First. Entry Second.                         │
│  Macro → Sector → Stock → Timing → Risk (5 step chain)          │
├─────────────────────────────────────────────────────────────────┤
│  7 Feature Cards (Macro / Rotation / Scoring / Setup / News /…) │
├─────────────────────────────────────────────────────────────────┤
│  Agent Architecture · 7 agent tiles · CTA Open AI Terminal      │
├─────────────────────────────────────────────────────────────────┤
│  Live Sector Rotation preview · CTA Full Matrix                 │
├─────────────────────────────────────────────────────────────────┤
│  Final CTA · Trade with conviction · [Launch] [Pricing]         │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Command Center (`/dashboard`)
```
┌────────┬──────────────────────────────────────────────────────┐
│ SIDE   │  [Search] [MARKET OPEN] [LATENCY 38ms] [🔔] [Avatar]│
│ ┌────┐ │  TICKER TAPE marquee ─────────────────────────────── │
│ │NAV │ │ ┌─Trend Gauge┬─Liquidity┬─Confidence┬─Risk─────────┐│
│ │ 8  │ │ │   78       │   72     │    84     │   72         ││
│ │ items│ └────────────┴──────────┴───────────┴──────────────┘│
│ │    │ │ 8 macro signal cards (SPY QQQ IWM VIX DXY 10Y HYG…) │
│ │    │ │ ┌──── Narrative panel (2/3) ──┬─ Macro Agent ─────┐│
│ │    │ │ │ AI text + 3 stat tiles      │ Sector Agent      ││
│ │    │ │ │ Sector Leadership cards     │ Risk Pulse        ││
│ │    │ │ │ Top AI-Scored Stocks (tbl)  │ Live News Feed    ││
│ │    │ │ └─────────────────────────────┴────────────────────┘│
│ │PRO │ │                                                      │
│ └────┘ │                                                      │
└────────┴──────────────────────────────────────────────────────┘
```

### 2.3 Macro Engine (`/macro`)
```
[Trend Score Gauge 200px]    [8 Macro Signal cards grid 2x4]
[SPY chart][QQQ chart]
[VIX chart][US10Y chart]
[Macro Agent Block (2/3)]    [Regime Cheat-Sheet (1/3)]
```

### 2.4 Sector Rotation (`/sectors`)
```
[Rotation Matrix table — 11 rows × 8 cols, RS bar chart in cell]
[Sector Heatmap (2/3)]    [Sector Agent + Trade Themes (1/3)]
```

### 2.5 Stocks (Screener) (`/stocks`)
```
[Filter bar: search · sector · min score · sort]
[Big sortable table — 25 rows × 11 cols, sparkline last col]
```

### 2.6 Stock Detail (`/stocks/[symbol]`)
```
[Back] [SYMBOL + Name + Sector chip]    [Watchlist] [Journal]
[Price $XXX · ▲X.XX% · Mcap]
[5 Gauges: Composite | Trend | Fund | Val | Risk]
┌────── (2/3) ───────────────────┬──── (1/3) ────────────┐
│ [Price chart 360px + EMAs]     │ AI TRADE SETUP        │
│ [AI Summary card]              │  LONG · Stage Entry   │
│ [Fundamentals Agent][Technical]│  Watch ─ Entry ─ Stop │
│ [News Agent][Risk Agent]       │  Targets · R:R · Size │
│ [Fundamentals Snapshot 4x2]    │  Rationale text       │
│                                │ [Catalysts list]      │
│                                │ [Recent News stack]   │
└────────────────────────────────┴───────────────────────┘
```

### 2.7 AI Agent (`/agent`)
```
┌────── (3/4) ─────────────────────────────┬──── (1/4) ─────┐
│ AGENT TERMINAL (chat scroll)             │ Suggested      │
│  AI: Welcome…                            │ Prompts (8)    │
│  USER: Analyze NVDA…                     │                │
│  AI: Summary + per-agent blocks (4-6)    │ Active Agents  │
│                                          │  · Macro ready │
│ [Input ─────────────────] [Send]         │  · Sector ...  │
└──────────────────────────────────────────┴────────────────┘
```

### 2.8 Risk Center (`/risk`)
```
[5 Gauges: Stability · Concentration · Liquidity · Earnings · Hedge]
[Sector Exposure bars (2/3)] [Risk Agent + Portfolio Agent (1/3)]
[Position-Level Risk table: 7 cols, color-coded status chips]
```

### 2.9 Journal (`/journal`)
```
[KPIs: Entries · Win Rate · FOMO · Open]
[AI Behavioral Analysis panel]
[Trade Log — card list, chips for side/emotion/result]
[+ New Entry modal: symbol/side/thesis/emotion/notes]
```

### 2.10 Pricing (`/pricing`)
```
[Hero: title + subtitle]
[3 Tier cards side-by-side: Free | Pro★ | Institutional]
[Feature comparison table]
```

---

## 3. Component Hierarchy

```
AppShell
├── Sidebar (8 nav items, brand, pro upsell card)
├── TopBar
│   ├── Search input
│   ├── Status pills (Market, Latency)
│   ├── Bell
│   └── Avatar
└── TickerTape (marquee, all stocks)
   └── Page content (children)
       ├── Panel  ← all dashboards composed of Panel cards
       ├── Stat   ← KPI tiles
       ├── Gauge  ← semicircle 0-100
       ├── Sparkline, PriceChart, Heatmap
       └── AgentBlockCard ← bull/bear/neutral colored block
```
