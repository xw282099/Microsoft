# Sentinel — Design System

## 1. Brand Tone
Institutional · Quant · Restrained futurism · Bloomberg-dense.
No skeuomorphism, no glossy gradients, no playful illustrations. Data first.

## 2. Color Tokens (`tailwind.config.ts`)

| Token | Hex | Use |
|---|---|---|
| `bg.base` | `#05070d` | Page background |
| `bg.panel` | `#0a0e18` | Nav / topbar surface |
| `bg.card` | `#0f1422` | Card surface |
| `bg.elev` | `#141b2d` | Elevated tile, hover |
| `bg.line` | `#1d2538` | Borders / dividers |
| `ink` | `#e6ebf5` | Primary text |
| `ink.muted` | `#8a93a6` | Secondary text |
| `ink.dim` | `#5a6378` | Tertiary text / axis |
| `ink.faint` | `#3a4256` | Disabled |
| `accent.cyan` | `#00e5ff` | Brand / data highlight |
| `accent.blue` | `#3b82f6` | Charts secondary |
| `accent.neon` | `#39ff88` | Live indicators |
| `accent.ai` | `#7c5cff` | AI agents |
| `accent.gold` | `#f5b400` | EMA20, neutral signals |
| `risk.bull` | `#22d39a` | Positive, bullish |
| `risk.bear` | `#ff4d6d` | Negative, bearish |
| `risk.neutral` | `#f5b400` | Caution, sideways |

## 3. Typography

- **UI text:** Inter, system-ui
- **All numbers, tickers, percentages:** ui-monospace, SF Mono, Menlo, Consolas
- Label-xs: `text-[10px] uppercase tracking-widest text-ink-dim font-mono`
- Section headers: `text-2xl font-bold tracking-tight`
- Stat values: `text-xl font-mono font-semibold`

## 4. Spacing & Radius

- Base unit: 4px (Tailwind default)
- Card padding: `p-4` (16px) for content, `px-4 py-2.5` for headers
- Radii: `rounded-md` (6px) tiles, `rounded-xl` (12px) panels, `rounded-full` chips
- Grids: `gap-3` standard, `gap-4` between sections

## 5. Components

### Panel
The atomic surface — header (title + action) + body. Every dashboard tile is a `Panel`.
Optional `glow` prop adds cyan shadow for emphasis.

### Gauge
Half-circle SVG, 0-100 range, color-coded:
- 75+ green (bull)
- 55-74 cyan
- 40-54 amber
- <40 red

### AgentBlockCard
Color-toned card (bull/bear/neutral border + tint), icon, headline, bulleted analysis.

### Chip
- `chip-bull` green
- `chip-bear` red
- `chip-neutral` amber
- `chip-cyan` brand
- `chip-ai` purple

### Sparkline / PriceChart / Heatmap
All built on `recharts` for footprint; sparkline 28px height inline, full chart 320-360px.

## 6. Motion

- `animate-pulseSoft` — soft opacity pulse for live dots
- `animate-tickerSlide` — 40s linear marquee
- `animate-scan` — 2.6s vertical scanline overlay
- All transitions ≤ 200ms; no decorative bounce.

## 7. Iconography

`lucide-react` — single icon family throughout. Always 14-16px in UI chrome, 20px in headers, monochrome.

## 8. Density

- Topbar: 44px high
- Sidebar: 240px wide
- Table rows: 36-40px
- Mobile: sidebar collapses; top KPI grids become 2-col

## 9. Layout Patterns

1. **3-column grid** (`grid lg:grid-cols-3`) — main content 2/3, sidebar 1/3
2. **KPI strip** — 4-5 gauges in a row, `panel p-4` each
3. **Macro signal grid** — 8 small cards, `grid-cols-2 md:grid-cols-4 lg:grid-cols-8`
4. **Two-column charts** — daily price grids, `grid lg:grid-cols-2 gap-4`

## 10. Microcopy

- All AI outputs end with educational disclaimer
- Status labels in UPPERCASE monospace (LIVE, BULL, RISK)
- Sector & ticker references are always uppercase, monospace, no $ prefix in tables, $-prefix in chat
- Numbers always use tabular-nums
