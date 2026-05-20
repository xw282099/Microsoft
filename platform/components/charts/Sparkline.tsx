"use client";

import { Candle } from "@/lib/data";
import { Area, AreaChart, ResponsiveContainer } from "recharts";

interface Props { data: Candle[]; color?: string; height?: number }

export function Sparkline({ data, color = "#00e5ff", height = 40 }: Props) {
  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer>
        <AreaChart data={data}>
          <defs>
            <linearGradient id={`g-${color.replace("#", "")}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity={0.55} />
              <stop offset="100%" stopColor={color} stopOpacity={0} />
            </linearGradient>
          </defs>
          <Area type="monotone" dataKey="c" stroke={color} strokeWidth={1.5}
            fill={`url(#g-${color.replace("#", "")})`} dot={false} isAnimationActive={false} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
