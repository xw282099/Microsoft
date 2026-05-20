"use client";

interface Props { value: number; label?: string; size?: number }

export function Gauge({ value, label, size = 160 }: Props) {
  const v = Math.max(0, Math.min(100, value));
  const angle = (v / 100) * 180 - 90; // -90..90
  const color = v >= 75 ? "#22d39a" : v >= 55 ? "#00e5ff" : v >= 40 ? "#f5b400" : "#ff4d6d";
  const r = size / 2 - 12;
  const cx = size / 2;
  const cy = size / 2;

  // Arc path
  const arc = (start: number, end: number) => {
    const s = (start - 90) * (Math.PI / 180);
    const e = (end - 90) * (Math.PI / 180);
    const x1 = cx + r * Math.cos(s);
    const y1 = cy + r * Math.sin(s);
    const x2 = cx + r * Math.cos(e);
    const y2 = cy + r * Math.sin(e);
    const large = end - start > 180 ? 1 : 0;
    return `M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2}`;
  };

  return (
    <div className="flex flex-col items-center">
      <svg width={size} height={size / 1.5} viewBox={`0 0 ${size} ${size / 1.5 + 10}`}>
        <path d={arc(-90, 90)} stroke="#1d2538" strokeWidth={10} fill="none" strokeLinecap="round" />
        <path d={arc(-90, -90 + (v / 100) * 180)} stroke={color} strokeWidth={10} fill="none" strokeLinecap="round" />
        <line
          x1={cx} y1={cy}
          x2={cx + (r - 10) * Math.cos((angle * Math.PI) / 180)}
          y2={cy + (r - 10) * Math.sin((angle * Math.PI) / 180)}
          stroke={color} strokeWidth={2}
        />
        <circle cx={cx} cy={cy} r={4} fill={color} />
        <text x={cx} y={cy + 22} textAnchor="middle" fontSize="22" fill="#e6ebf5"
          fontFamily="ui-monospace" fontWeight={700}>{v}</text>
      </svg>
      {label && <div className="label-xs mt-1">{label}</div>}
    </div>
  );
}
