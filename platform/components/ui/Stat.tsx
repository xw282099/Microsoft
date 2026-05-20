import { cn } from "@/lib/utils";

interface Props {
  label: string;
  value: string | number;
  delta?: string;
  tone?: "bull" | "bear" | "neutral";
  hint?: string;
}

export function Stat({ label, value, delta, tone = "neutral", hint }: Props) {
  const toneClass =
    tone === "bull" ? "text-risk-bull" : tone === "bear" ? "text-risk-bear" : "text-ink-muted";
  return (
    <div className="p-3 rounded-lg bg-bg-elev/40 border border-bg-line">
      <div className="label-xs">{label}</div>
      <div className="mt-1 flex items-baseline gap-2">
        <span className="text-xl font-mono font-semibold">{value}</span>
        {delta && <span className={cn("text-xs font-mono", toneClass)}>{delta}</span>}
      </div>
      {hint && <div className="text-[11px] text-ink-dim mt-1 leading-snug">{hint}</div>}
    </div>
  );
}
