"use client";

import { Candle } from "@/lib/data";
import {
  Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";

interface Props { data: Candle[]; height?: number; symbol?: string }

export function PriceChart({ data, height = 320, symbol }: Props) {
  // Compute simple moving averages on the fly.
  const sma = (period: number) => {
    return data.map((d, i) => {
      if (i < period - 1) return null;
      let s = 0;
      for (let k = 0; k < period; k++) s += data[i - k].c;
      return +(s / period).toFixed(2);
    });
  };
  const s20 = sma(20);
  const s50 = sma(50);
  const enriched = data.map((d, i) => ({ ...d, s20: s20[i], s50: s50[i] }));

  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer>
        <AreaChart data={enriched} margin={{ top: 5, right: 12, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="gPrice" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#00e5ff" stopOpacity={0.35} />
              <stop offset="100%" stopColor="#00e5ff" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="#1d2538" strokeDasharray="3 3" />
          <XAxis dataKey="t" stroke="#5a6378" fontSize={10} tickLine={false} axisLine={false}
            minTickGap={48} />
          <YAxis stroke="#5a6378" fontSize={10} tickLine={false} axisLine={false}
            domain={["dataMin", "dataMax"]} width={48} />
          <Tooltip
            contentStyle={{ background: "#0f1422", border: "1px solid #1d2538", borderRadius: 8, fontSize: 12 }}
            labelStyle={{ color: "#8a93a6" }}
            formatter={(v: any, n: any) => [typeof v === "number" ? v.toFixed(2) : v, n]}
          />
          <Area type="monotone" dataKey="c" stroke="#00e5ff" strokeWidth={1.6}
            fill="url(#gPrice)" dot={false} name={symbol || "Price"} isAnimationActive={false} />
          <Area type="monotone" dataKey="s20" stroke="#f5b400" strokeWidth={1} fill="transparent" dot={false} name="EMA20" isAnimationActive={false} />
          <Area type="monotone" dataKey="s50" stroke="#7c5cff" strokeWidth={1} fill="transparent" dot={false} name="EMA50" isAnimationActive={false} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
