import { seeded } from "./utils";

export type Sector =
  | "Semiconductors"
  | "AI Infrastructure"
  | "Data Centers"
  | "Cloud Software"
  | "Cybersecurity"
  | "Energy"
  | "Utilities"
  | "Networking"
  | "Optical"
  | "Nuclear"
  | "Storage";

export interface Stock {
  symbol: string;
  name: string;
  sector: Sector;
  price: number;
  change: number; // %
  marketCap: number; // USD
  pe: number;
  peg: number;
  revGrowth: number; // %
  epsGrowth: number; // %
  grossMargin: number; // %
  fcf: number; // USD
  beta: number;
  shortInterest: number; // %
  insiderTrend: "buy" | "sell" | "neutral";
  institutionalFlow: "accumulating" | "distributing" | "neutral";
  catalysts: string[];
}

export const SECTORS: Sector[] = [
  "Semiconductors",
  "AI Infrastructure",
  "Data Centers",
  "Cloud Software",
  "Cybersecurity",
  "Energy",
  "Utilities",
  "Networking",
  "Optical",
  "Nuclear",
  "Storage",
];

/** Curated universe — AI / growth / trend names. */
export const STOCKS: Stock[] = [
  { symbol: "NVDA", name: "NVIDIA Corp.", sector: "Semiconductors", price: 142.18, change: 2.84, marketCap: 3.49e12, pe: 58.2, peg: 1.1, revGrowth: 122.4, epsGrowth: 168.0, grossMargin: 75.7, fcf: 56.5e9, beta: 1.65, shortInterest: 1.1, insiderTrend: "neutral", institutionalFlow: "accumulating", catalysts: ["Blackwell ramp", "Hyperscaler capex", "Sovereign AI"] },
  { symbol: "AMD",  name: "Advanced Micro Devices", sector: "Semiconductors", price: 168.42, change: 1.92, marketCap: 2.72e11, pe: 47.8, peg: 1.4, revGrowth: 18.0, epsGrowth: 35.4, grossMargin: 51.4, fcf: 4.5e9, beta: 1.85, shortInterest: 2.4, insiderTrend: "neutral", institutionalFlow: "accumulating", catalysts: ["MI355X launch", "AI PC", "EPYC share gains"] },
  { symbol: "AVGO", name: "Broadcom Inc.", sector: "AI Infrastructure", price: 178.95, change: 1.54, marketCap: 8.35e11, pe: 41.6, peg: 1.6, revGrowth: 47.3, epsGrowth: 24.8, grossMargin: 61.8, fcf: 19.2e9, beta: 1.20, shortInterest: 0.9, insiderTrend: "neutral", institutionalFlow: "accumulating", catalysts: ["Custom ASIC TAM", "VMware integration"] },
  { symbol: "MSFT", name: "Microsoft Corp.", sector: "Cloud Software", price: 432.18, change: 0.42, marketCap: 3.21e12, pe: 35.4, peg: 2.1, revGrowth: 15.4, epsGrowth: 21.0, grossMargin: 70.1, fcf: 74.0e9, beta: 0.92, shortInterest: 0.7, insiderTrend: "neutral", institutionalFlow: "accumulating", catalysts: ["Azure AI", "Copilot ARR", "OpenAI partnership"] },
  { symbol: "META", name: "Meta Platforms", sector: "Cloud Software", price: 568.12, change: -0.34, marketCap: 1.45e12, pe: 27.6, peg: 1.3, revGrowth: 22.1, epsGrowth: 60.0, grossMargin: 81.4, fcf: 49.0e9, beta: 1.21, shortInterest: 1.0, insiderTrend: "neutral", institutionalFlow: "neutral", catalysts: ["Reels monetization", "AI ad targeting", "Llama 4"] },
  { symbol: "ORCL", name: "Oracle Corp.", sector: "Cloud Software", price: 174.62, change: 0.81, marketCap: 4.91e11, pe: 32.5, peg: 2.5, revGrowth: 6.8, epsGrowth: 10.2, grossMargin: 71.0, fcf: 11.8e9, beta: 1.05, shortInterest: 1.3, insiderTrend: "neutral", institutionalFlow: "accumulating", catalysts: ["OCI capacity", "Stargate buildout", "RPO growth"] },
  { symbol: "PLTR", name: "Palantir Tech.", sector: "Cloud Software", price: 65.84, change: 3.12, marketCap: 1.46e11, pe: 215.0, peg: 4.4, revGrowth: 30.0, epsGrowth: 65.0, grossMargin: 81.7, fcf: 1.2e9, beta: 2.10, shortInterest: 4.1, insiderTrend: "sell", institutionalFlow: "accumulating", catalysts: ["AIP adoption", "DoD wins", "Foundry commercial"] },
  { symbol: "CRWD", name: "CrowdStrike", sector: "Cybersecurity", price: 348.55, change: 0.62, marketCap: 8.45e10, pe: 90.0, peg: 2.9, revGrowth: 28.4, epsGrowth: 38.0, grossMargin: 77.4, fcf: 1.0e9, beta: 1.15, shortInterest: 2.0, insiderTrend: "neutral", institutionalFlow: "neutral", catalysts: ["Module attach rates", "Falcon Flex"] },
  { symbol: "SNOW", name: "Snowflake Inc.", sector: "Cloud Software", price: 158.40, change: 1.23, marketCap: 5.30e10, pe: 0, peg: 0, revGrowth: 28.0, epsGrowth: 0, grossMargin: 67.5, fcf: 0.95e9, beta: 1.05, shortInterest: 3.2, insiderTrend: "neutral", institutionalFlow: "accumulating", catalysts: ["Cortex AI", "Iceberg adoption"] },
  { symbol: "NET",  name: "Cloudflare", sector: "Cloud Software", price: 92.10, change: 0.34, marketCap: 3.20e10, pe: 0, peg: 0, revGrowth: 28.6, epsGrowth: 0, grossMargin: 77.0, fcf: 0.20e9, beta: 1.45, shortInterest: 3.8, insiderTrend: "sell", institutionalFlow: "neutral", catalysts: ["Workers AI", "Pool consumption"] },
  { symbol: "DDOG", name: "Datadog", sector: "Cloud Software", price: 132.18, change: -0.42, marketCap: 4.45e10, pe: 78.0, peg: 2.4, revGrowth: 26.0, epsGrowth: 32.0, grossMargin: 81.0, fcf: 0.78e9, beta: 1.18, shortInterest: 2.9, insiderTrend: "neutral", institutionalFlow: "neutral", catalysts: ["AI agent observability"] },
  { symbol: "VRT",  name: "Vertiv Holdings", sector: "Data Centers", price: 118.40, change: 2.61, marketCap: 4.55e10, pe: 38.0, peg: 1.2, revGrowth: 22.0, epsGrowth: 60.0, grossMargin: 36.0, fcf: 1.2e9, beta: 1.40, shortInterest: 1.8, insiderTrend: "neutral", institutionalFlow: "accumulating", catalysts: ["Liquid cooling", "AI factory power"] },
  { symbol: "ETN",  name: "Eaton Corp.", sector: "Data Centers", price: 322.50, change: 0.95, marketCap: 1.28e11, pe: 35.0, peg: 2.0, revGrowth: 8.0, epsGrowth: 18.0, grossMargin: 38.6, fcf: 3.3e9, beta: 1.10, shortInterest: 0.9, insiderTrend: "neutral", institutionalFlow: "accumulating", catalysts: ["Data center electrification"] },
  { symbol: "ANET", name: "Arista Networks", sector: "Networking", price: 408.20, change: 1.41, marketCap: 1.28e11, pe: 44.0, peg: 2.0, revGrowth: 18.0, epsGrowth: 22.0, grossMargin: 64.0, fcf: 2.4e9, beta: 1.05, shortInterest: 1.4, insiderTrend: "neutral", institutionalFlow: "accumulating", catalysts: ["800G ramp", "AI backend share"] },
  { symbol: "CIEN", name: "Ciena Corp.", sector: "Optical", price: 76.40, change: 2.10, marketCap: 1.10e10, pe: 28.0, peg: 1.5, revGrowth: 12.0, epsGrowth: 20.0, grossMargin: 43.0, fcf: 0.45e9, beta: 1.30, shortInterest: 3.0, insiderTrend: "neutral", institutionalFlow: "accumulating", catalysts: ["DCI demand", "ZR pluggables"] },
  { symbol: "COHR", name: "Coherent Corp.", sector: "Optical", price: 92.10, change: 1.85, marketCap: 1.42e10, pe: 0, peg: 0, revGrowth: 24.0, epsGrowth: 0, grossMargin: 35.0, fcf: 0.30e9, beta: 1.55, shortInterest: 4.2, insiderTrend: "neutral", institutionalFlow: "accumulating", catalysts: ["AI transceivers"] },
  { symbol: "NBIS", name: "Nebius Group", sector: "Data Centers", price: 48.20, change: 3.84, marketCap: 1.20e10, pe: 0, peg: 0, revGrowth: 600.0, epsGrowth: 0, grossMargin: 22.0, fcf: -0.6e9, beta: 2.20, shortInterest: 6.0, insiderTrend: "neutral", institutionalFlow: "accumulating", catalysts: ["NVIDIA partnership", "GPU capacity expansion"] },
  { symbol: "CEG",  name: "Constellation Energy", sector: "Nuclear", price: 268.90, change: 1.74, marketCap: 8.40e10, pe: 30.0, peg: 1.6, revGrowth: 12.0, epsGrowth: 30.0, grossMargin: 25.0, fcf: 2.6e9, beta: 0.95, shortInterest: 1.5, insiderTrend: "neutral", institutionalFlow: "accumulating", catalysts: ["Hyperscaler PPAs", "SMR optionality"] },
  { symbol: "VST",  name: "Vistra Energy", sector: "Energy", price: 168.20, change: 2.40, marketCap: 5.70e10, pe: 26.0, peg: 1.4, revGrowth: 15.0, epsGrowth: 28.0, grossMargin: 30.0, fcf: 2.0e9, beta: 1.20, shortInterest: 2.3, insiderTrend: "neutral", institutionalFlow: "accumulating", catalysts: ["Texas demand", "Nuclear fleet"] },
  { symbol: "PWR",  name: "Quanta Services", sector: "Utilities", price: 312.40, change: 0.88, marketCap: 4.55e10, pe: 42.0, peg: 1.8, revGrowth: 14.0, epsGrowth: 22.0, grossMargin: 15.0, fcf: 1.2e9, beta: 1.10, shortInterest: 1.3, insiderTrend: "neutral", institutionalFlow: "accumulating", catalysts: ["Grid buildout", "T&D backlog"] },
  { symbol: "WDC",  name: "Western Digital", sector: "Storage", price: 64.80, change: 1.20, marketCap: 2.23e10, pe: 12.0, peg: 0.5, revGrowth: 40.0, epsGrowth: 200.0, grossMargin: 30.0, fcf: 1.1e9, beta: 1.60, shortInterest: 3.5, insiderTrend: "neutral", institutionalFlow: "accumulating", catalysts: ["NAND cycle", "HDD pricing"] },
  { symbol: "STX",  name: "Seagate Tech.", sector: "Storage", price: 102.30, change: 0.95, marketCap: 2.15e10, pe: 22.0, peg: 0.9, revGrowth: 32.0, epsGrowth: 180.0, grossMargin: 32.0, fcf: 1.5e9, beta: 1.50, shortInterest: 2.8, insiderTrend: "neutral", institutionalFlow: "accumulating", catalysts: ["HAMR ramp", "Mass capacity demand"] },
  { symbol: "PANW", name: "Palo Alto Networks", sector: "Cybersecurity", price: 372.40, change: -0.21, marketCap: 1.21e11, pe: 50.0, peg: 2.6, revGrowth: 15.0, epsGrowth: 26.0, grossMargin: 74.0, fcf: 3.2e9, beta: 1.10, shortInterest: 1.6, insiderTrend: "neutral", institutionalFlow: "neutral", catalysts: ["Platformization"] },
  { symbol: "ZS",   name: "Zscaler", sector: "Cybersecurity", price: 198.50, change: 0.45, marketCap: 3.10e10, pe: 110.0, peg: 3.6, revGrowth: 26.0, epsGrowth: 40.0, grossMargin: 80.0, fcf: 0.6e9, beta: 1.30, shortInterest: 4.0, insiderTrend: "neutral", institutionalFlow: "neutral", catalysts: ["Zero trust adoption"] },
];

export interface SectorRotation {
  sector: Sector;
  rsRank: number; // 0..100
  flow: number;   // -100..100  positive = inflow
  momentum: number; // -100..100
  weekChange: number; // %
  monthChange: number; // %
  ytdChange: number; // %
  narrative: string;
}

export const SECTOR_ROTATION: SectorRotation[] = [
  { sector: "AI Infrastructure", rsRank: 96, flow: 78, momentum: 84, weekChange: 4.8, monthChange: 12.4, ytdChange: 48.6, narrative: "Hyperscaler capex acceleration; second-derivative beneficiaries leading." },
  { sector: "Semiconductors",    rsRank: 92, flow: 64, momentum: 76, weekChange: 3.6, monthChange: 9.2,  ytdChange: 42.1, narrative: "Custom silicon and HBM tailwinds; broad strength across compute layer." },
  { sector: "Data Centers",      rsRank: 91, flow: 72, momentum: 79, weekChange: 4.1, monthChange: 11.6, ytdChange: 51.0, narrative: "Liquid cooling, power, and modular DC build-out remain in accumulation." },
  { sector: "Optical",           rsRank: 84, flow: 55, momentum: 71, weekChange: 3.0, monthChange: 8.2,  ytdChange: 36.5, narrative: "800G/1.6T transceiver cycle; AI back-end fabrics drive shipments." },
  { sector: "Networking",        rsRank: 80, flow: 48, momentum: 66, weekChange: 2.4, monthChange: 6.8,  ytdChange: 28.4, narrative: "AI ethernet share gains; merchant silicon dynamics shifting." },
  { sector: "Nuclear",           rsRank: 76, flow: 42, momentum: 60, weekChange: 2.1, monthChange: 6.0,  ytdChange: 95.0, narrative: "Multi-year PPAs with hyperscalers; SMR optionality re-priced." },
  { sector: "Energy",            rsRank: 64, flow: 22, momentum: 38, weekChange: 1.1, monthChange: 3.0,  ytdChange: 18.5, narrative: "Power demand inflection benefits IPP / independent generators." },
  { sector: "Utilities",         rsRank: 62, flow: 30, momentum: 35, weekChange: 0.8, monthChange: 2.6,  ytdChange: 22.0, narrative: "Grid transmission and electrification backlog scaling." },
  { sector: "Storage",           rsRank: 58, flow: 18, momentum: 30, weekChange: 0.9, monthChange: 2.2,  ytdChange: 15.4, narrative: "HDD pricing cycle, HAMR ramp, NAND inflection." },
  { sector: "Cloud Software",    rsRank: 52, flow: -6, momentum: 18, weekChange: -0.4, monthChange: 1.0, ytdChange: 8.5,  narrative: "Mega-cap leadership but mid-cap SaaS still consolidating." },
  { sector: "Cybersecurity",     rsRank: 48, flow: -10, momentum: 12, weekChange: -0.6, monthChange: -1.2, ytdChange: 4.0, narrative: "Platformization noise; mixed earnings reactions." },
];

export interface MacroSignal {
  key: string;
  label: string;
  value: number;
  unit?: string;
  change: number; // bps or %
  state: "bull" | "bear" | "neutral";
  hint: string;
}

export const MACRO_SIGNALS: MacroSignal[] = [
  { key: "SPY",   label: "S&P 500",   value: 586.42, change: 0.34, state: "bull",    hint: "Above 21/50/200 EMA, HH/HL structure intact." },
  { key: "QQQ",   label: "Nasdaq 100", value: 512.18, change: 0.62, state: "bull",    hint: "Mega-cap tech leadership; breadth thrust active." },
  { key: "IWM",   label: "Russell 2000", value: 224.80, change: -0.42, state: "neutral", hint: "Below 200 EMA; awaiting rate-cut catalyst." },
  { key: "VIX",   label: "VIX",        value: 14.52, change: -2.10, state: "bull",    hint: "Compressed vol; risk-on regime." },
  { key: "DXY",   label: "US Dollar",  value: 102.10, change: -0.18, state: "bull",    hint: "Weak dollar supports global risk." },
  { key: "US10Y", label: "10Y Yield",  value: 4.24, unit: "%", change: -3.0, state: "bull", hint: "Yields rolling over; supportive for duration." },
  { key: "HYG",   label: "HY Credit",  value: 79.40, change: 0.12, state: "bull",    hint: "Tight spreads; risk appetite confirmed." },
  { key: "FED",   label: "Fed Stance", value: 0, change: 0, state: "bull", hint: "Cutting cycle; dot plot dovish." },
];

export interface NewsItem {
  id: string;
  ts: string; // ISO
  source: string;
  title: string;
  symbols: string[];
  sentiment: "bullish" | "bearish" | "neutral";
  impact: "high" | "medium" | "low";
  summary: string;
}

export const NEWS: NewsItem[] = [
  { id: "n1", ts: "2026-05-20T13:42:00Z", source: "Reuters",  title: "NVIDIA secures multi-year sovereign AI deal with European consortium", symbols: ["NVDA"], sentiment: "bullish", impact: "high", summary: "Multi-billion order strengthens FY27 visibility; sovereign AI thesis intact." },
  { id: "n2", ts: "2026-05-20T13:30:00Z", source: "Bloomberg", title: "Vertiv raises FY guidance citing liquid cooling backlog", symbols: ["VRT", "ETN"], sentiment: "bullish", impact: "high", summary: "Power & cooling capacity in tight supply through 2027." },
  { id: "n3", ts: "2026-05-20T13:18:00Z", source: "SEC 13F",   title: "Tiger Global increases PLTR stake by 22%", symbols: ["PLTR"], sentiment: "bullish", impact: "medium", summary: "Institutional accumulation continues despite stretched valuation." },
  { id: "n4", ts: "2026-05-20T13:05:00Z", source: "Fed",       title: "Powell signals openness to additional cuts if disinflation continues", symbols: ["SPY", "QQQ"], sentiment: "bullish", impact: "high", summary: "Dovish lean supports duration and growth multiples." },
  { id: "n5", ts: "2026-05-20T12:50:00Z", source: "Wedbush",   title: "Analyst lifts Arista price target to $475", symbols: ["ANET"], sentiment: "bullish", impact: "medium", summary: "AI back-end ethernet share gains accelerating." },
  { id: "n6", ts: "2026-05-20T12:32:00Z", source: "CNBC",      title: "Energy IPPs surge on hyperscaler PPA chatter", symbols: ["CEG", "VST"], sentiment: "bullish", impact: "high", summary: "Power as the new bottleneck for AI buildout." },
  { id: "n7", ts: "2026-05-20T12:18:00Z", source: "X / @unusual_whales", title: "Unusual call sweeps in AVGO ahead of analyst day", symbols: ["AVGO"], sentiment: "bullish", impact: "medium", summary: "Aggressive premium on weekly OTM strikes." },
  { id: "n8", ts: "2026-05-20T12:00:00Z", source: "WSJ",       title: "ZS guides below consensus on enterprise deal timing", symbols: ["ZS", "PANW"], sentiment: "bearish", impact: "medium", summary: "Cyber spending recalibration; platformization fatigue." },
];

/* -------------------------------------------------------------------------- */
/*  Synthetic OHLC time series — deterministic so SSR & CSR match.            */
/* -------------------------------------------------------------------------- */
export interface Candle { t: string; o: number; h: number; l: number; c: number; v: number }

export function generateSeries(symbol: string, days = 180, basePrice?: number): Candle[] {
  const base = basePrice ?? 100;
  const rand = seeded(
    symbol.split("").reduce((a, c) => a + c.charCodeAt(0), 0) * 13 + 7
  );
  const out: Candle[] = [];
  let price = base * 0.65;
  const trend = 0.0018 + rand() * 0.0012;
  for (let i = 0; i < days; i++) {
    const shock = (rand() - 0.5) * 0.035;
    const drift = trend + shock;
    const o = price;
    const c = Math.max(0.01, price * (1 + drift));
    const h = Math.max(o, c) * (1 + Math.abs(rand()) * 0.012);
    const l = Math.min(o, c) * (1 - Math.abs(rand()) * 0.012);
    const v = Math.floor(1e6 + rand() * 8e6);
    const date = new Date(Date.now() - (days - i) * 86400000);
    out.push({
      t: date.toISOString().slice(0, 10),
      o: +o.toFixed(2),
      h: +h.toFixed(2),
      l: +l.toFixed(2),
      c: +c.toFixed(2),
      v,
    });
    price = c;
  }
  return out;
}

export function getStock(symbol: string): Stock | undefined {
  return STOCKS.find((s) => s.symbol.toUpperCase() === symbol.toUpperCase());
}
