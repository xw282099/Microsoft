"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { Panel } from "@/components/ui/Panel";
import { Sparkline } from "@/components/charts/Sparkline";
import { STOCKS, SECTORS, generateSeries, Sector } from "@/lib/data";
import { compositeScore } from "@/lib/scoring";
import { fmtPct, fmtUsd, cn } from "@/lib/utils";
import { Filter, ArrowUpDown } from "lucide-react";

type SortKey = "composite" | "trend" | "fund" | "val" | "risk" | "change" | "mcap";

export default function StocksPage() {
  const [sector, setSector] = useState<Sector | "All">("All");
  const [sortKey, setSortKey] = useState<SortKey>("composite");
  const [minScore, setMinScore] = useState(0);
  const [search, setSearch] = useState("");

  const enriched = useMemo(() =>
    STOCKS.map((s) => ({ ...s, scores: compositeScore(s) })),
  []);

  const rows = useMemo(() => {
    return enriched
      .filter((s) => (sector === "All" ? true : s.sector === sector))
      .filter((s) => (search ? (s.symbol + " " + s.name).toLowerCase().includes(search.toLowerCase()) : true))
      .filter((s) => s.scores.composite >= minScore)
      .sort((a, b) => {
        switch (sortKey) {
          case "trend": return b.scores.trend - a.scores.trend;
          case "fund": return b.scores.fund - a.scores.fund;
          case "val": return b.scores.val - a.scores.val;
          case "risk": return b.scores.risk - a.scores.risk;
          case "change": return b.change - a.change;
          case "mcap": return b.marketCap - a.marketCap;
          default: return b.scores.composite - a.scores.composite;
        }
      });
  }, [enriched, sector, sortKey, minScore, search]);

  return (
    <div className="p-4 lg:p-6 space-y-4">
      <div>
        <div className="label-xs">AI Stock Scoring Engine</div>
        <h1 className="text-2xl font-bold tracking-tight">Screen the Universe by AI Composite Score</h1>
        <p className="text-sm text-ink-muted mt-1">
          Trend (40%) · Fundamentals (28%) · Valuation (18%) · Risk (14%) — trend-first weighting.
        </p>
      </div>

      <Panel title="Filters" subtitle="Narrow the universe to your edge">
        <div className="flex flex-wrap items-center gap-3">
          <input
            placeholder="Search ticker or name…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="bg-bg-elev border border-bg-line rounded-md px-3 py-1.5 text-sm w-56 focus:border-accent-cyan/60 outline-none"
          />
          <div className="flex items-center gap-2 text-xs">
            <Filter className="h-3.5 w-3.5 text-ink-dim" />
            <select
              value={sector}
              onChange={(e) => setSector(e.target.value as any)}
              className="bg-bg-elev border border-bg-line rounded-md px-2 py-1.5"
            >
              <option value="All">All sectors</option>
              {SECTORS.map((s) => <option key={s}>{s}</option>)}
            </select>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <span className="text-ink-dim">Min AI Score</span>
            <input type="range" min={0} max={100} value={minScore}
              onChange={(e) => setMinScore(Number(e.target.value))} className="w-40 accent-accent-cyan" />
            <span className="font-mono w-6">{minScore}</span>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <ArrowUpDown className="h-3.5 w-3.5 text-ink-dim" />
            <select value={sortKey} onChange={(e) => setSortKey(e.target.value as SortKey)}
              className="bg-bg-elev border border-bg-line rounded-md px-2 py-1.5">
              <option value="composite">AI Composite</option>
              <option value="trend">Trend Score</option>
              <option value="fund">Fundamental Score</option>
              <option value="val">Valuation Score</option>
              <option value="risk">Risk Score</option>
              <option value="change">% Change</option>
              <option value="mcap">Market Cap</option>
            </select>
          </div>
          <div className="ml-auto text-xs text-ink-muted font-mono">{rows.length} matches</div>
        </div>
      </Panel>

      <Panel title="Universe Screener" bodyClass="p-0">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-[10px] uppercase font-mono text-ink-dim border-b border-bg-line bg-bg-panel/60">
              <tr className="text-left">
                <th className="py-2.5 px-3">Symbol</th>
                <th>Sector</th>
                <th className="text-right">Price</th>
                <th className="text-right">Chg</th>
                <th className="text-right">Mcap</th>
                <th className="text-right">Trend</th>
                <th className="text-right">Fund</th>
                <th className="text-right">Val</th>
                <th className="text-right">Risk</th>
                <th className="text-right pr-3">AI Score</th>
                <th className="pr-3">7D</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((s) => {
                const { trend, fund, val, risk, composite } = s.scores;
                return (
                  <tr key={s.symbol} className="border-b border-bg-line/40 hover:bg-bg-elev/30">
                    <td className="py-2.5 px-3">
                      <Link href={`/stocks/${s.symbol}`}
                        className="font-mono font-bold text-ink hover:text-accent-cyan">{s.symbol}</Link>
                      <div className="text-[11px] text-ink-dim truncate max-w-[180px]">{s.name}</div>
                    </td>
                    <td className="text-[11px] text-ink-muted whitespace-nowrap">{s.sector}</td>
                    <td className="text-right font-mono">${s.price.toFixed(2)}</td>
                    <td className={"text-right font-mono " + (s.change >= 0 ? "text-risk-bull" : "text-risk-bear")}>
                      {fmtPct(s.change)}
                    </td>
                    <td className="text-right font-mono text-ink-muted">{fmtUsd(s.marketCap)}</td>
                    <td className={cn("text-right font-mono", scoreColor(trend))}>{trend}</td>
                    <td className={cn("text-right font-mono", scoreColor(fund))}>{fund}</td>
                    <td className={cn("text-right font-mono", scoreColor(val))}>{val}</td>
                    <td className={cn("text-right font-mono", scoreColor(risk))}>{risk}</td>
                    <td className="text-right pr-3">
                      <span className={"chip " + (composite >= 75 ? "chip-bull" : composite >= 55 ? "chip-cyan" : composite >= 40 ? "chip-neutral" : "chip-bear")}>
                        {composite}
                      </span>
                    </td>
                    <td className="pr-3 w-24">
                      <Sparkline data={generateSeries(s.symbol, 30, s.price)} color={s.change >= 0 ? "#22d39a" : "#ff4d6d"} height={28} />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}

function scoreColor(n: number) {
  if (n >= 75) return "text-risk-bull";
  if (n >= 55) return "text-accent-cyan";
  if (n >= 40) return "text-risk-neutral";
  return "text-risk-bear";
}
