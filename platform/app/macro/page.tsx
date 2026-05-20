import { Panel } from "@/components/ui/Panel";
import { Stat } from "@/components/ui/Stat";
import { Gauge } from "@/components/charts/Gauge";
import { PriceChart } from "@/components/charts/PriceChart";
import { AgentBlockCard } from "@/components/panels/AgentBlockCard";
import { generateSeries, MACRO_SIGNALS } from "@/lib/data";
import { fmtPct } from "@/lib/utils";
import { macroAgent } from "@/lib/ai";

export default function MacroPage() {
  return (
    <div className="p-4 lg:p-6 space-y-4">
      <div>
        <div className="label-xs">Macro Engine</div>
        <h1 className="text-2xl font-bold tracking-tight">AI Macro Trend Analysis</h1>
        <p className="text-sm text-ink-muted mt-1">
          Liquidity · Rates · Volatility · Credit · Policy — fused into a single trend regime score.
        </p>
      </div>

      <div className="grid lg:grid-cols-4 gap-3">
        <div className="panel p-4 lg:col-span-1 flex flex-col items-center justify-center">
          <Gauge value={78} label="MARKET TREND SCORE" size={200} />
          <div className="mt-3 text-center">
            <div className="text-sm font-semibold text-risk-bull">BULL · Stage 2</div>
            <div className="text-xs text-ink-muted">Trend confirmed across major indices</div>
          </div>
        </div>

        <div className="lg:col-span-3 grid grid-cols-2 md:grid-cols-4 gap-2">
          {MACRO_SIGNALS.map((m) => (
            <div key={m.key} className="panel p-3">
              <div className="flex items-center justify-between">
                <span className="label-xs">{m.label}</span>
                <span className={"chip " + (m.state === "bull" ? "chip-bull" : m.state === "bear" ? "chip-bear" : "chip-neutral")}>
                  {m.state}
                </span>
              </div>
              <div className="mt-1 text-xl font-mono font-semibold">{m.value}{m.unit || ""}</div>
              <div className={"text-[11px] font-mono " + (m.change >= 0 ? "text-risk-bull" : "text-risk-bear")}>
                {fmtPct(m.change)}
              </div>
              <div className="text-[10px] text-ink-dim mt-2 leading-snug">{m.hint}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-4">
        <Panel title="SPY · S&P 500" subtitle="Trend Structure · EMA 20 / 50">
          <PriceChart data={generateSeries("SPY", 180, 586)} />
        </Panel>
        <Panel title="QQQ · Nasdaq 100" subtitle="Mega-cap growth leadership">
          <PriceChart data={generateSeries("QQQ", 180, 512)} />
        </Panel>
        <Panel title="VIX · Volatility" subtitle="Risk regime indicator">
          <PriceChart data={generateSeries("VIX", 180, 14.5)} />
        </Panel>
        <Panel title="US10Y · Treasury Yields" subtitle="Discount rate / liquidity proxy">
          <PriceChart data={generateSeries("US10Y", 180, 4.24)} />
        </Panel>
      </div>

      <div className="grid lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2">
          <AgentBlockCard block={macroAgent()} />
        </div>
        <Panel title="Regime Cheat-Sheet" subtitle="Playbook by macro state">
          <div className="space-y-2 text-sm">
            <div className="data-row"><span>Risk-On / Bull</span><span className="chip-bull chip">Buy growth, momentum</span></div>
            <div className="data-row"><span>Mixed / Sideways</span><span className="chip-neutral chip">Selective, factor neutral</span></div>
            <div className="data-row"><span>Risk-Off / Bear</span><span className="chip-bear chip">Cash, quality, hedges</span></div>
            <div className="data-row"><span>High Vol Spike</span><span className="chip-bear chip">Reduce leverage</span></div>
            <div className="data-row"><span>Liquidity Crunch</span><span className="chip-bear chip">Defensive sectors</span></div>
          </div>
          <div className="mt-4 p-3 rounded-md bg-accent-ai/10 border border-accent-ai/30 text-xs">
            <div className="label-xs text-accent-ai mb-1">AI INSIGHT</div>
            "Current regime resembles late-2017 risk-on phase. Recommend max participation
            in trend leaders with disciplined stops."
          </div>
        </Panel>
      </div>
    </div>
  );
}
