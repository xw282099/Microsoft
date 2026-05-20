import { notFound } from "next/navigation";
import Link from "next/link";
import { ArrowLeft, Crosshair, Shield, Target, TrendingUp } from "lucide-react";
import { Panel } from "@/components/ui/Panel";
import { Stat } from "@/components/ui/Stat";
import { Gauge } from "@/components/charts/Gauge";
import { PriceChart } from "@/components/charts/PriceChart";
import { AgentBlockCard } from "@/components/panels/AgentBlockCard";
import { STOCKS, generateSeries, getStock, NEWS } from "@/lib/data";
import { compositeScore, buildTradeSetup } from "@/lib/scoring";
import { fundamentalsAgent, technicalAgent, newsAgent, riskAgent } from "@/lib/ai";
import { fmtPct, fmtUsd, cn } from "@/lib/utils";

export function generateStaticParams() {
  return STOCKS.map((s) => ({ symbol: s.symbol }));
}

export default function StockDetail({ params }: { params: { symbol: string } }) {
  const stock = getStock(params.symbol);
  if (!stock) return notFound();

  const candles = generateSeries(stock.symbol, 240, stock.price);
  const scores = compositeScore(stock, candles);
  const setup = buildTradeSetup(stock, scores.composite);
  const stockNews = NEWS.filter((n) => n.symbols.includes(stock.symbol));

  const setupBiasClass =
    setup.bias === "long" ? "border-risk-bull/40 bg-risk-bull/5 text-risk-bull"
    : setup.bias === "wait" ? "border-risk-neutral/40 bg-risk-neutral/5 text-risk-neutral"
    : "border-risk-bear/40 bg-risk-bear/5 text-risk-bear";

  return (
    <div className="p-4 lg:p-6 space-y-4">
      {/* Header */}
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <Link href="/stocks" className="inline-flex items-center gap-1.5 text-xs text-ink-dim hover:text-accent-cyan mb-2">
            <ArrowLeft className="h-3 w-3" /> All stocks
          </Link>
          <div className="flex items-baseline gap-3">
            <h1 className="text-3xl font-bold font-mono">{stock.symbol}</h1>
            <span className="text-ink-muted text-lg">{stock.name}</span>
            <span className="chip chip-cyan">{stock.sector}</span>
          </div>
          <div className="mt-2 flex items-baseline gap-4">
            <span className="text-3xl font-mono font-semibold">${stock.price.toFixed(2)}</span>
            <span className={"text-lg font-mono " + (stock.change >= 0 ? "text-risk-bull" : "text-risk-bear")}>
              {stock.change >= 0 ? "▲" : "▼"} {fmtPct(stock.change)}
            </span>
            <span className="text-xs font-mono text-ink-dim">Mcap {fmtUsd(stock.marketCap)}</span>
          </div>
        </div>
        <div className="flex gap-2">
          <button className="btn-ai text-xs">+ Watchlist</button>
          <button className="btn-primary text-xs">+ Journal Entry</button>
        </div>
      </div>

      {/* Score grid */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <div className="panel p-4 flex flex-col items-center">
          <Gauge value={scores.composite} label="AI COMPOSITE" />
          <div className="mt-1 text-[11px] text-ink-muted text-center">Trend-weighted master score</div>
        </div>
        <div className="panel p-4 flex flex-col items-center">
          <Gauge value={scores.trend} label="TREND" />
        </div>
        <div className="panel p-4 flex flex-col items-center">
          <Gauge value={scores.fund} label="FUNDAMENTAL" />
        </div>
        <div className="panel p-4 flex flex-col items-center">
          <Gauge value={scores.val} label="VALUATION" />
        </div>
        <div className="panel p-4 flex flex-col items-center">
          <Gauge value={scores.risk} label="RISK (HIGHER=SAFER)" />
        </div>
      </div>

      <div className="grid lg:grid-cols-3 gap-4">
        {/* Chart + analysis */}
        <div className="lg:col-span-2 space-y-4">
          <Panel title="Price · EMA Structure" subtitle="Daily · 240 days · EMA 20/50">
            <PriceChart data={candles} symbol={stock.symbol} height={360} />
          </Panel>

          <Panel title="AI Summary" glow>
            <div className="text-sm text-ink leading-relaxed">
              <strong className="text-accent-cyan">{stock.symbol}</strong> scores{" "}
              <strong className="text-accent-cyan">{scores.composite}/100</strong> on the Sentinel AI Composite.
              The technical structure is {scores.trend > 70 ? "decisively bullish" : scores.trend > 50 ? "constructive" : "weakening"};
              fundamentals are {scores.fund > 65 ? "strong" : "mixed"} with revenue growth of{" "}
              <strong>{stock.revGrowth.toFixed(1)}%</strong>; valuation is{" "}
              {scores.val > 60 ? "reasonable" : "stretched"} (PEG {stock.peg || "n/a"}).
              Institutional flow is currently <strong className="text-accent-cyan">{stock.institutionalFlow}</strong>.
              Active catalysts: {stock.catalysts.join("; ")}.
            </div>
          </Panel>

          <div className="grid md:grid-cols-2 gap-3">
            <AgentBlockCard block={fundamentalsAgent(stock)} />
            <AgentBlockCard block={technicalAgent(stock)} />
            <AgentBlockCard block={newsAgent(stock.symbol)} />
            <AgentBlockCard block={riskAgent(stock)} />
          </div>

          <Panel title="Fundamentals Snapshot">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              <Stat label="Revenue Growth" value={`${stock.revGrowth.toFixed(1)}%`} tone={stock.revGrowth > 20 ? "bull" : "neutral"} />
              <Stat label="EPS Growth" value={`${stock.epsGrowth.toFixed(1)}%`} tone={stock.epsGrowth > 20 ? "bull" : "neutral"} />
              <Stat label="Gross Margin" value={`${stock.grossMargin.toFixed(1)}%`} />
              <Stat label="Free Cash Flow" value={fmtUsd(stock.fcf)} tone={stock.fcf > 0 ? "bull" : "bear"} />
              <Stat label="Forward PE" value={stock.pe ? stock.pe.toFixed(1) : "n/a"} />
              <Stat label="PEG" value={stock.peg ? stock.peg.toFixed(2) : "n/a"} />
              <Stat label="Beta" value={stock.beta.toFixed(2)} />
              <Stat label="Short Interest" value={`${stock.shortInterest.toFixed(1)}%`} />
            </div>
          </Panel>
        </div>

        {/* Trade Setup sidebar */}
        <div className="space-y-4">
          <Panel
            title={<span className="flex items-center gap-2"><Target className="h-4 w-4 text-accent-cyan" />AI Trade Setup</span>}
            subtitle="Auto-generated · refresh on next bar close"
            glow
          >
            <div className={cn("rounded-md border px-3 py-2 mb-3 text-sm font-mono uppercase tracking-wider text-center", setupBiasClass)}>
              {setup.bias === "long" ? "LONG BIAS — Stage Entry" : setup.bias === "wait" ? "WAIT — Pullback Setup" : "AVOID — Below Quality Threshold"}
            </div>

            {setup.bias !== "avoid" && (
              <div className="space-y-2 text-sm">
                <div className="data-row">
                  <span className="flex items-center gap-1.5 text-ink-muted"><Crosshair className="h-3.5 w-3.5" />Watch Zone</span>
                  <span className="font-mono">${setup.watch[0]} – ${setup.watch[1]}</span>
                </div>
                <div className="data-row">
                  <span className="text-ink-muted">Entry Zone</span>
                  <span className="font-mono text-accent-cyan">${setup.entry[0]} – ${setup.entry[1]}</span>
                </div>
                <div className="data-row">
                  <span className="flex items-center gap-1.5 text-ink-muted"><Shield className="h-3.5 w-3.5" />Stop Loss</span>
                  <span className="font-mono text-risk-bear">${setup.stop}</span>
                </div>
                <div className="data-row">
                  <span className="flex items-center gap-1.5 text-ink-muted"><TrendingUp className="h-3.5 w-3.5" />Targets</span>
                  <span className="font-mono text-risk-bull">${setup.targets.join(" / $")}</span>
                </div>
                <div className="data-row">
                  <span className="text-ink-muted">Risk:Reward</span>
                  <span className="font-mono">{setup.rr}</span>
                </div>
                <div className="data-row">
                  <span className="text-ink-muted">Suggested Size</span>
                  <span className="font-mono">{setup.positionPct}% of book</span>
                </div>
              </div>
            )}

            <div className="mt-3 p-3 rounded-md bg-bg-elev/40 border border-bg-line text-[12px] text-ink-muted leading-relaxed">
              <div className="label-xs mb-1">RATIONALE</div>
              {setup.rationale}
            </div>
            <div className="mt-3 text-[10px] text-ink-dim italic">
              Education & research only. Not investment advice. AI outputs may contain errors.
            </div>
          </Panel>

          <Panel title="Catalysts">
            <ul className="space-y-1.5 text-sm">
              {stock.catalysts.map((c) => (
                <li key={c} className="flex items-start gap-2">
                  <span className="text-accent-cyan mt-0.5">▸</span>
                  <span className="text-ink-muted">{c}</span>
                </li>
              ))}
            </ul>
          </Panel>

          <Panel title="Recent News" action={<span className="text-[10px] font-mono text-ink-dim">{stockNews.length || 0} items</span>}>
            {stockNews.length === 0 ? (
              <div className="text-xs text-ink-dim">No tagged news in current window.</div>
            ) : (
              <div className="space-y-2">
                {stockNews.map((n) => (
                  <div key={n.id} className="p-2.5 rounded-md border border-bg-line bg-bg-elev/30">
                    <div className="flex justify-between text-[10px] font-mono text-ink-dim">
                      <span>{n.source}</span>
                      <span className={n.sentiment === "bullish" ? "text-risk-bull" : n.sentiment === "bearish" ? "text-risk-bear" : "text-ink-muted"}>{n.sentiment}</span>
                    </div>
                    <div className="text-xs mt-1">{n.title}</div>
                  </div>
                ))}
              </div>
            )}
          </Panel>
        </div>
      </div>
    </div>
  );
}
