import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function fmtUsd(n: number, digits = 2) {
  if (n >= 1e12) return `$${(n / 1e12).toFixed(2)}T`;
  if (n >= 1e9) return `$${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `$${(n / 1e6).toFixed(2)}M`;
  return `$${n.toFixed(digits)}`;
}

export function fmtPct(n: number, digits = 2) {
  const sign = n > 0 ? "+" : "";
  return `${sign}${n.toFixed(digits)}%`;
}

export function fmtNum(n: number, digits = 0) {
  return n.toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function colorForChange(n: number) {
  if (n > 0.01) return "text-risk-bull";
  if (n < -0.01) return "text-risk-bear";
  return "text-ink-muted";
}

export function colorForScore(n: number) {
  if (n >= 75) return "text-risk-bull";
  if (n >= 55) return "text-accent-cyan";
  if (n >= 40) return "text-risk-neutral";
  return "text-risk-bear";
}

/** Deterministic pseudo-random (seed-based) so server/client match. */
export function seeded(seed: number) {
  let s = seed;
  return () => {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  };
}
