"use client";

import { STOCKS } from "@/lib/data";
import { cn, fmtPct } from "@/lib/utils";
import Link from "next/link";

/** Sector-grouped heat tiles, sized by market cap, colored by change. */
export function Heatmap() {
  const grouped = STOCKS.reduce<Record<string, typeof STOCKS>>((acc, s) => {
    (acc[s.sector] ||= []).push(s);
    return acc;
  }, {});

  const colorFor = (c: number) => {
    if (c >= 3) return "bg-risk-bull/70 text-black";
    if (c >= 1.5) return "bg-risk-bull/45 text-white";
    if (c >= 0.5) return "bg-risk-bull/25 text-white";
    if (c > -0.5) return "bg-bg-elev text-ink-muted";
    if (c > -1.5) return "bg-risk-bear/25 text-white";
    if (c > -3) return "bg-risk-bear/45 text-white";
    return "bg-risk-bear/70 text-white";
  };

  const sizeFor = (mcap: number) => {
    if (mcap > 1e12) return "col-span-3 row-span-2";
    if (mcap > 3e11) return "col-span-2 row-span-2";
    if (mcap > 1e11) return "col-span-2";
    return "";
  };

  return (
    <div className="space-y-4">
      {Object.entries(grouped).map(([sector, list]) => (
        <div key={sector}>
          <div className="label-xs mb-1.5">{sector}</div>
          <div className="grid grid-cols-6 md:grid-cols-8 auto-rows-[60px] gap-1.5">
            {list
              .sort((a, b) => b.marketCap - a.marketCap)
              .map((s) => (
                <Link
                  key={s.symbol}
                  href={`/stocks/${s.symbol}`}
                  className={cn(
                    "rounded-md p-2 flex flex-col justify-between transition-transform hover:scale-[1.03] border border-white/5",
                    colorFor(s.change),
                    sizeFor(s.marketCap)
                  )}
                >
                  <div className="text-xs font-mono font-bold leading-none">{s.symbol}</div>
                  <div className="text-[10px] font-mono opacity-90">{fmtPct(s.change)}</div>
                </Link>
              ))}
          </div>
        </div>
      ))}
    </div>
  );
}
