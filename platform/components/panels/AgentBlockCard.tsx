import { AgentBlock } from "@/lib/ai";
import { Brain, Compass, Layers, LineChart, Newspaper, Shield, Wallet } from "lucide-react";
import { cn } from "@/lib/utils";

const ICONS: Record<string, any> = {
  Macro: Compass,
  "Sector Rotation": Layers,
  Fundamentals: LineChart,
  Technical: LineChart,
  News: Newspaper,
  Risk: Shield,
  Portfolio: Wallet,
};

export function AgentBlockCard({ block }: { block: AgentBlock }) {
  const Icon = ICONS[block.agent] || Brain;
  const tone =
    block.tone === "bull" ? "border-risk-bull/40 bg-risk-bull/5"
    : block.tone === "bear" ? "border-risk-bear/40 bg-risk-bear/5"
    : "border-bg-line bg-bg-card";
  const toneText =
    block.tone === "bull" ? "text-risk-bull"
    : block.tone === "bear" ? "text-risk-bear"
    : "text-accent-cyan";
  return (
    <div className={cn("rounded-xl border p-4", tone)}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Icon className={cn("h-4 w-4", toneText)} />
          <div className="text-sm font-semibold">{block.agent} Agent</div>
        </div>
        <span className={cn("chip", toneText, "border " + tone)}>{block.tone.toUpperCase()}</span>
      </div>
      <div className="text-sm text-ink mb-2">{block.headline}</div>
      <ul className="space-y-1.5 text-[13px] text-ink-muted">
        {block.bullets.map((b, i) => (
          <li key={i} className="flex gap-2">
            <span className="text-accent-cyan mt-0.5">▸</span>
            <span>{b}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
