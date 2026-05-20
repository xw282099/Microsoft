import { Panel } from "@/components/ui/Panel";
import { Heatmap } from "@/components/charts/Heatmap";
import { AgentBlockCard } from "@/components/panels/AgentBlockCard";
import { SECTOR_ROTATION } from "@/lib/data";
import { fmtPct } from "@/lib/utils";
import { sectorAgent } from "@/lib/ai";

export default function SectorsPage() {
  const sorted = [...SECTOR_ROTATION].sort((a, b) => b.rsRank - a.rsRank);

  return (
    <div className="p-4 lg:p-6 space-y-4">
      <div>
        <div className="label-xs">Sector Rotation Matrix</div>
        <h1 className="text-2xl font-bold tracking-tight">Where is capital flowing?</h1>
        <p className="text-sm text-ink-muted mt-1">
          Relative strength, institutional flow, momentum, and valuation compression — all in one view.
        </p>
      </div>

      <Panel title="Rotation Matrix" subtitle="11 sectors · sorted by Relative Strength">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-[10px] uppercase font-mono text-ink-dim border-b border-bg-line">
              <tr className="text-left">
                <th className="py-2 pl-1">Sector</th>
                <th className="text-right">RS Rank</th>
                <th className="text-right">Inst. Flow</th>
                <th className="text-right">Momentum</th>
                <th className="text-right">1W</th>
                <th className="text-right">1M</th>
                <th className="text-right">YTD</th>
                <th>Narrative</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((s) => {
                const flowClass = s.flow > 30 ? "text-risk-bull" : s.flow > 0 ? "text-accent-cyan" : "text-risk-bear";
                return (
                  <tr key={s.sector} className="border-b border-bg-line/40 hover:bg-bg-elev/30">
                    <td className="py-3 pl-1 font-semibold">{s.sector}</td>
                    <td className="text-right">
                      <div className="inline-flex items-center gap-2">
                        <div className="w-20 h-1.5 rounded-full bg-bg-line overflow-hidden">
                          <div className="h-full bg-gradient-to-r from-accent-cyan to-risk-bull"
                            style={{ width: `${s.rsRank}%` }} />
                        </div>
                        <span className="font-mono text-xs">{s.rsRank}</span>
                      </div>
                    </td>
                    <td className={"text-right font-mono " + flowClass}>{s.flow > 0 ? "+" : ""}{s.flow}</td>
                    <td className="text-right font-mono">{s.momentum}</td>
                    <td className={"text-right font-mono " + (s.weekChange >= 0 ? "text-risk-bull" : "text-risk-bear")}>{fmtPct(s.weekChange)}</td>
                    <td className={"text-right font-mono " + (s.monthChange >= 0 ? "text-risk-bull" : "text-risk-bear")}>{fmtPct(s.monthChange)}</td>
                    <td className={"text-right font-mono " + (s.ytdChange >= 0 ? "text-risk-bull" : "text-risk-bear")}>{fmtPct(s.ytdChange)}</td>
                    <td className="text-xs text-ink-muted max-w-[280px] truncate" title={s.narrative}>{s.narrative}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Panel>

      <div className="grid lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2">
          <Panel title="Sector Heat Map" subtitle="Tile size = market cap · color = % change">
            <Heatmap />
          </Panel>
        </div>
        <div className="space-y-4">
          <AgentBlockCard block={sectorAgent()} />
          <Panel title="Trade Themes">
            <ul className="space-y-2 text-sm text-ink-muted">
              <li className="flex gap-2"><span className="text-accent-cyan">▸</span> AI Power Complex: CEG, VST, ETN, PWR</li>
              <li className="flex gap-2"><span className="text-accent-cyan">▸</span> Liquid Cooling: VRT, ANET (back-end)</li>
              <li className="flex gap-2"><span className="text-accent-cyan">▸</span> AI Inference Silicon: NVDA, AVGO, AMD</li>
              <li className="flex gap-2"><span className="text-accent-cyan">▸</span> Storage Cycle: WDC, STX (HBM-adjacent demand)</li>
              <li className="flex gap-2"><span className="text-accent-cyan">▸</span> Sovereign AI Buildout: NBIS, ORCL (OCI capacity)</li>
            </ul>
          </Panel>
        </div>
      </div>
    </div>
  );
}
