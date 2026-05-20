import { Panel } from "@/components/ui/Panel";
import { Stat } from "@/components/ui/Stat";
import { Gauge } from "@/components/charts/Gauge";
import { AgentBlockCard } from "@/components/panels/AgentBlockCard";
import { riskAgent, portfolioAgent } from "@/lib/ai";

export default function RiskPage() {
  return (
    <div className="p-4 lg:p-6 space-y-4">
      <div>
        <div className="label-xs">Risk Control Center</div>
        <h1 className="text-2xl font-bold tracking-tight">Protect Capital First.</h1>
        <p className="text-sm text-ink-muted mt-1">
          Portfolio risk · Sector exposure · Correlation · Volatility · Earnings tail · Liquidity.
        </p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <div className="panel p-4 flex flex-col items-center">
          <Gauge value={72} label="OVERALL STABILITY" />
        </div>
        <div className="panel p-4 flex flex-col items-center">
          <Gauge value={36} label="CONCENTRATION RISK" />
        </div>
        <div className="panel p-4 flex flex-col items-center">
          <Gauge value={82} label="LIQUIDITY BUFFER" />
        </div>
        <div className="panel p-4 flex flex-col items-center">
          <Gauge value={48} label="EARNINGS TAIL" />
        </div>
        <div className="panel p-4 flex flex-col items-center">
          <Gauge value={66} label="HEDGE COVERAGE" />
        </div>
      </div>

      <div className="grid lg:grid-cols-3 gap-4">
        <Panel title="Sector Exposure" className="lg:col-span-2">
          <div className="space-y-2">
            {[
              ["AI Infrastructure", 28, "high"],
              ["Semiconductors", 22, "high"],
              ["Data Centers", 14, "moderate"],
              ["Cloud Software", 10, "moderate"],
              ["Nuclear / Power", 9, "moderate"],
              ["Cybersecurity", 6, "low"],
              ["Cash", 11, "buffer"],
            ].map(([s, pct, t]: any) => (
              <div key={s} className="">
                <div className="flex justify-between text-xs">
                  <span className="text-ink">{s}</span>
                  <span className="font-mono">{pct}%</span>
                </div>
                <div className="mt-1 h-2 rounded-full bg-bg-line overflow-hidden">
                  <div
                    className={
                      "h-full " +
                      (t === "high" ? "bg-risk-bear/70"
                      : t === "moderate" ? "bg-risk-neutral/70"
                      : t === "buffer" ? "bg-accent-cyan/70"
                      : "bg-risk-bull/70")
                    }
                    style={{ width: `${pct * 3}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
          <div className="mt-4 p-3 rounded-md bg-risk-bear/5 border border-risk-bear/30 text-xs">
            <div className="label-xs text-risk-bear mb-1">CONCENTRATION ALERT</div>
            AI complex = 64% of book. Recommend trim or pair with QQQ put hedge.
          </div>
        </Panel>

        <div className="space-y-4">
          <AgentBlockCard block={riskAgent()} />
          <AgentBlockCard block={portfolioAgent()} />
        </div>
      </div>

      <Panel title="Position-Level Risk">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-[10px] uppercase font-mono text-ink-dim border-b border-bg-line">
              <tr className="text-left">
                <th className="py-2">Symbol</th>
                <th className="text-right">Weight</th>
                <th className="text-right">Cost Basis</th>
                <th className="text-right">Live P&L</th>
                <th className="text-right">Stop Distance</th>
                <th className="text-right">Earnings In</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {POSITIONS.map((p) => (
                <tr key={p.sym} className="border-b border-bg-line/40">
                  <td className="py-2.5 font-mono font-bold">{p.sym}</td>
                  <td className="text-right font-mono">{p.weight}%</td>
                  <td className="text-right font-mono text-ink-muted">${p.cost}</td>
                  <td className={"text-right font-mono " + (p.pnl >= 0 ? "text-risk-bull" : "text-risk-bear")}>
                    {p.pnl >= 0 ? "+" : ""}{p.pnl.toFixed(1)}%
                  </td>
                  <td className="text-right font-mono">{p.stop}%</td>
                  <td className="text-right font-mono">{p.earn}d</td>
                  <td><span className={"chip " + p.statusClass}>{p.status}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}

const POSITIONS = [
  { sym: "NVDA", weight: 12, cost: 118.40, pnl: 20.1, stop: 7.2, earn: 21, status: "TRAIL", statusClass: "chip-bull" },
  { sym: "AVGO", weight: 10, cost: 162.10, pnl: 10.4, stop: 6.0, earn: 14, status: "HOLD", statusClass: "chip-cyan" },
  { sym: "VRT",  weight: 8,  cost: 102.80, pnl: 15.2, stop: 8.5, earn: 31, status: "ADD ON PB", statusClass: "chip-cyan" },
  { sym: "CEG",  weight: 7,  cost: 240.00, pnl: 12.0, stop: 9.0, earn: 28, status: "HOLD", statusClass: "chip-cyan" },
  { sym: "PLTR", weight: 6,  cost: 58.20,  pnl: 13.1, stop: 12.0, earn: 9,  status: "TRIM 1/3", statusClass: "chip-neutral" },
  { sym: "ANET", weight: 5,  cost: 380.00, pnl: 7.4,  stop: 7.5, earn: 6,  status: "EARN RISK", statusClass: "chip-bear" },
  { sym: "ZS",   weight: 3,  cost: 210.00, pnl: -5.5, stop: 4.0, earn: 12, status: "CLOSE", statusClass: "chip-bear" },
];
