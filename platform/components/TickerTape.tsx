import { STOCKS } from "@/lib/data";
import { fmtPct } from "@/lib/utils";

export function TickerTape() {
  const items = [...STOCKS, ...STOCKS]; // duplicate for seamless loop
  return (
    <div className="relative overflow-hidden border-t border-bg-line bg-bg-panel">
      <div className="marquee py-1.5">
        {items.map((s, i) => (
          <span key={i} className="flex items-center gap-2 text-[11px] font-mono whitespace-nowrap">
            <span className="text-ink-muted">{s.symbol}</span>
            <span className="text-ink">${s.price.toFixed(2)}</span>
            <span className={s.change >= 0 ? "text-risk-bull" : "text-risk-bear"}>
              {fmtPct(s.change)}
            </span>
            <span className="text-ink-faint">·</span>
          </span>
        ))}
      </div>
    </div>
  );
}
