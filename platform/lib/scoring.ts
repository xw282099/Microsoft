import { Stock, Candle } from "./data";

/** Clamp into [0, 100]. */
const clamp = (n: number) => Math.max(0, Math.min(100, n));

/* ---------------- Trend Score (technical) ---------------- */
export function trendScore(s: Stock, candles?: Candle[]): number {
  let score = 50;
  if (candles && candles.length > 50) {
    const closes = candles.map((c) => c.c);
    const last = closes[closes.length - 1];
    const ema = (period: number) => {
      const k = 2 / (period + 1);
      let e = closes[0];
      for (let i = 1; i < closes.length; i++) e = closes[i] * k + e * (1 - k);
      return e;
    };
    const e20 = ema(20);
    const e50 = ema(50);
    const e200 = closes.length >= 200 ? ema(200) : ema(closes.length - 1);
    if (last > e20) score += 10;
    if (last > e50) score += 10;
    if (last > e200) score += 10;
    if (e20 > e50) score += 5;
    if (e50 > e200) score += 5;

    // 20-day momentum
    const mom = (last - closes[closes.length - 21]) / closes[closes.length - 21];
    score += Math.max(-15, Math.min(15, mom * 100));
  }
  score += s.change * 1.2;
  return Math.round(clamp(score));
}

/* ---------------- Fundamental Score ---------------- */
export function fundamentalScore(s: Stock): number {
  let score = 40;
  score += Math.min(25, s.revGrowth * 0.4); // up to ~25
  score += Math.min(15, s.epsGrowth * 0.12);
  score += Math.min(10, (s.grossMargin - 40) * 0.4);
  if (s.fcf > 0) score += 5;
  if (s.fcf > 5e9) score += 5;
  return Math.round(clamp(score));
}

/* ---------------- Valuation Score (higher = cheaper relative to growth) ---------------- */
export function valuationScore(s: Stock): number {
  if (!s.peg || s.peg <= 0) {
    // Use rev growth vs market cap proxy
    return Math.round(clamp(40 + Math.min(40, s.revGrowth * 0.4)));
  }
  let score = 80 - s.peg * 18; // peg 1.0 -> 62, peg 2.0 -> 44, peg 3.0 -> 26
  if (s.pe && s.pe > 0) score -= Math.max(0, (s.pe - 40) * 0.4);
  return Math.round(clamp(score));
}

/* ---------------- Risk Score (higher = SAFER) ---------------- */
export function riskScore(s: Stock): number {
  let score = 70;
  score -= Math.max(0, (s.beta - 1) * 18);
  score -= Math.max(0, (s.shortInterest - 2) * 4);
  if (s.insiderTrend === "sell") score -= 12;
  if (s.insiderTrend === "buy") score += 8;
  if (s.fcf < 0) score -= 15;
  return Math.round(clamp(score));
}

/* ---------------- Composite AI Score ---------------- */
export function compositeScore(s: Stock, candles?: Candle[]) {
  const trend = trendScore(s, candles);
  const fund = fundamentalScore(s);
  const val = valuationScore(s);
  const risk = riskScore(s);
  // Trend-first weighting per platform philosophy
  const composite = Math.round(
    trend * 0.40 + fund * 0.28 + val * 0.18 + risk * 0.14
  );
  return { trend, fund, val, risk, composite };
}

/* ---------------- AI Trade Setup Generator ---------------- */
export interface TradeSetup {
  bias: "long" | "wait" | "avoid";
  watch: [number, number];
  entry: [number, number];
  stop: number;
  targets: number[];
  rr: number;
  positionPct: number;
  rationale: string;
}
export function buildTradeSetup(s: Stock, comp: number): TradeSetup {
  const p = s.price;
  const atrApprox = p * 0.025;
  if (comp < 45) {
    return {
      bias: "avoid",
      watch: [+(p * 0.88).toFixed(2), +(p * 0.92).toFixed(2)],
      entry: [0, 0],
      stop: 0,
      targets: [],
      rr: 0,
      positionPct: 0,
      rationale: "Composite score below quality threshold; no constructive setup.",
    };
  }
  if (comp < 60) {
    return {
      bias: "wait",
      watch: [+(p * 0.93).toFixed(2), +(p * 0.97).toFixed(2)],
      entry: [+(p * 0.95).toFixed(2), +(p * 0.98).toFixed(2)],
      stop: +(p * 0.91).toFixed(2),
      targets: [+(p * 1.06).toFixed(2), +(p * 1.12).toFixed(2)],
      rr: +(((p * 1.12 - p * 0.965) / (p * 0.965 - p * 0.91))).toFixed(2),
      positionPct: 2,
      rationale: "Constructive but not extended — wait for pullback into demand zone.",
    };
  }
  const entryLow = +(p - atrApprox * 0.4).toFixed(2);
  const entryHigh = +(p + atrApprox * 0.2).toFixed(2);
  const stop = +(p - atrApprox * 1.8).toFixed(2);
  const t1 = +(p + atrApprox * 3).toFixed(2);
  const t2 = +(p + atrApprox * 6).toFixed(2);
  const rr = +(((t2 - (entryLow + entryHigh) / 2) / ((entryLow + entryHigh) / 2 - stop))).toFixed(2);
  return {
    bias: "long",
    watch: [+(p * 0.97).toFixed(2), +(p * 1.00).toFixed(2)],
    entry: [entryLow, entryHigh],
    stop,
    targets: [t1, t2],
    rr,
    positionPct: comp >= 80 ? 5 : 3,
    rationale:
      "Trend, growth, and capital flow aligned. Use staged entry with hard stop below structure.",
  };
}
