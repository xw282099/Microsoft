import { STOCKS, SECTOR_ROTATION, MACRO_SIGNALS, NEWS, getStock, Stock } from "./data";
import { compositeScore, buildTradeSetup } from "./scoring";

/* -------------------------------------------------------------------------- */
/*  AI Agent orchestration (deterministic, offline).                           */
/*  Each "agent" is a pure function that returns a structured analysis block. */
/* -------------------------------------------------------------------------- */

export type AgentName =
  | "Macro"
  | "Sector Rotation"
  | "Fundamentals"
  | "Technical"
  | "News"
  | "Risk"
  | "Portfolio";

export interface AgentBlock {
  agent: AgentName;
  headline: string;
  bullets: string[];
  tone: "bull" | "bear" | "neutral";
}

export function macroAgent(): AgentBlock {
  const bull = MACRO_SIGNALS.filter((m) => m.state === "bull").length;
  const bear = MACRO_SIGNALS.filter((m) => m.state === "bear").length;
  const tone = bull > bear ? "bull" : bear > bull ? "bear" : "neutral";
  return {
    agent: "Macro",
    headline:
      tone === "bull"
        ? "Risk-on regime. Liquidity supportive, vol compressed, yields cooperative."
        : tone === "bear"
        ? "Defensive regime. De-risk equities, raise cash."
        : "Mixed signals. Selectivity required.",
    bullets: [
      "QQQ above 21/50/200 EMA with positive breadth thrust.",
      "VIX < 16 — vol risk-on; favors trend continuation in growth.",
      "10Y yield rolling over; supportive for duration & long-multiple assets.",
      "USD weak — global liquidity tailwind for risk assets.",
    ],
    tone,
  };
}

export function sectorAgent(): AgentBlock {
  const top = [...SECTOR_ROTATION].sort((a, b) => b.rsRank - a.rsRank).slice(0, 3);
  const bottom = [...SECTOR_ROTATION].sort((a, b) => a.rsRank - b.rsRank).slice(0, 2);
  return {
    agent: "Sector Rotation",
    headline: `Leadership: ${top.map((t) => t.sector).join(", ")}.`,
    bullets: [
      ...top.map((t) => `${t.sector} — RS ${t.rsRank}, flow ${t.flow > 0 ? "+" : ""}${t.flow}, ${t.narrative}`),
      `Laggards: ${bottom.map((b) => b.sector).join(", ")} — avoid or use as funding source.`,
    ],
    tone: "bull",
  };
}

export function fundamentalsAgent(s: Stock): AgentBlock {
  return {
    agent: "Fundamentals",
    headline: `Revenue growth ${s.revGrowth.toFixed(1)}%, EPS growth ${s.epsGrowth.toFixed(1)}%.`,
    bullets: [
      `Gross margin ${s.grossMargin.toFixed(1)}% — ${s.grossMargin > 65 ? "premium software/platform economics." : "hardware-style structure."}`,
      `FCF: ${s.fcf >= 0 ? "+" : ""}$${(s.fcf / 1e9).toFixed(2)}B — ${s.fcf > 0 ? "self-funding capex." : "external capital dependency."}`,
      `Forward PE ${s.pe || "n/a"}, PEG ${s.peg || "n/a"} — ${s.peg && s.peg < 1.5 ? "reasonable vs growth." : "premium multiple, requires execution."}`,
      `Catalysts: ${s.catalysts.join("; ")}.`,
    ],
    tone: s.revGrowth > 18 && s.epsGrowth > 18 ? "bull" : s.revGrowth < 5 ? "bear" : "neutral",
  };
}

export function technicalAgent(s: Stock): AgentBlock {
  const tone = s.change > 1 ? "bull" : s.change < -1 ? "bear" : "neutral";
  return {
    agent: "Technical",
    headline: `${s.symbol} ${s.change >= 0 ? "+" : ""}${s.change.toFixed(2)}% — ${
      tone === "bull" ? "trend continuation" : tone === "bear" ? "distribution risk" : "consolidation"
    }.`,
    bullets: [
      "Multi-timeframe trend: weekly higher-highs / higher-lows intact.",
      "Above key EMAs (21/50/200) on daily.",
      "Volume profile supportive into upper range.",
      "Watch reaction at prior swing-high — breakout = trend extension.",
    ],
    tone,
  };
}

export function newsAgent(symbol?: string): AgentBlock {
  const relevant = symbol
    ? NEWS.filter((n) => n.symbols.includes(symbol))
    : NEWS.slice(0, 4);
  const bull = relevant.filter((n) => n.sentiment === "bullish").length;
  const bear = relevant.filter((n) => n.sentiment === "bearish").length;
  const tone = bull > bear ? "bull" : bear > bull ? "bear" : "neutral";
  return {
    agent: "News",
    headline: `${relevant.length} relevant items — net ${tone}.`,
    bullets: relevant.length
      ? relevant.map((n) => `[${n.source}] ${n.title}`)
      : ["No high-impact news in window."],
    tone,
  };
}

export function riskAgent(s?: Stock): AgentBlock {
  if (!s)
    return {
      agent: "Risk",
      headline: "Portfolio risk: moderate. VIX 14.5, credit tight.",
      bullets: [
        "Concentration risk: top 3 holdings = 38% of book — consider trimming.",
        "Sector exposure: 64% AI complex — single-factor risk.",
        "Earnings season tail risk in 11 days — reduce gamma exposure.",
      ],
      tone: "neutral",
    };
  return {
    agent: "Risk",
    headline: `Beta ${s.beta.toFixed(2)}, short interest ${s.shortInterest.toFixed(1)}%.`,
    bullets: [
      `Insider trend: ${s.insiderTrend}.`,
      `Institutional flow: ${s.institutionalFlow}.`,
      s.fcf < 0 ? "Negative FCF — sensitive to rate regime." : "FCF positive — drawdown resilience.",
      "Position sizing: cap initial risk at 0.5% of book.",
    ],
    tone: s.beta > 1.8 || s.fcf < 0 ? "bear" : "neutral",
  };
}

export function portfolioAgent(): AgentBlock {
  return {
    agent: "Portfolio",
    headline: "Suggested book: 70% trend longs / 20% laggard hedges / 10% cash.",
    bullets: [
      "Add: NVDA, AVGO, VRT, CEG on pullback.",
      "Trim: ZS, NET into rallies.",
      "Hedge: long-dated QQQ puts as cheap tail insurance.",
    ],
    tone: "neutral",
  };
}

/* -------------------------------------------------------------------------- */
/*  AI Chat Orchestrator                                                       */
/* -------------------------------------------------------------------------- */

export interface ChatResponse {
  blocks: AgentBlock[];
  summary: string;
  disclaimer: string;
}

const DISCLAIMER =
  "For educational and research purposes only. Not investment advice. AI outputs may contain errors.";

export function runAgents(query: string): ChatResponse {
  const q = query.toLowerCase();
  const blocks: AgentBlock[] = [];

  // Detect ticker mentions
  const symbol = STOCKS.map((s) => s.symbol).find(
    (sym) => new RegExp(`\\b${sym.toLowerCase()}\\b`).test(q)
  );
  const stock = symbol ? getStock(symbol) : undefined;

  const wantsMacro = /macro|fed|cpi|rate|liquidity|risk-?on|risk-?off|vix|yield/i.test(query);
  const wantsSector = /sector|rotation|industry|infra|infrastructure|semicond|cloud|cyber|energy|power|nuclear|optical|networking|storage|data ?center/i.test(query);
  const wantsNews = /news|catalyst|filing|earnings|sec|analyst|sentiment/i.test(query);
  const wantsRisk = /risk|hedge|exposure|stop|sizing|drawdown/i.test(query);
  const wantsPortfolio = /portfolio|book|allocation|position|cash/i.test(query);

  if (stock) {
    blocks.push(fundamentalsAgent(stock));
    blocks.push(technicalAgent(stock));
    blocks.push(newsAgent(stock.symbol));
    blocks.push(riskAgent(stock));
  } else {
    if (wantsMacro || (!wantsSector && !wantsRisk && !wantsPortfolio && !wantsNews)) blocks.push(macroAgent());
    if (wantsSector) blocks.push(sectorAgent());
    if (wantsNews) blocks.push(newsAgent());
    if (wantsRisk) blocks.push(riskAgent());
    if (wantsPortfolio) blocks.push(portfolioAgent());
    if (blocks.length === 0) {
      blocks.push(macroAgent(), sectorAgent(), newsAgent());
    }
  }

  let summary = "Macro: risk-on. Leadership: AI infrastructure, semis, data centers.";
  if (stock) {
    const sc = compositeScore(stock);
    const setup = buildTradeSetup(stock, sc.composite);
    summary = `${stock.symbol} composite score ${sc.composite}/100 — ${
      setup.bias === "long"
        ? `LONG bias. Entry ${setup.entry[0]}–${setup.entry[1]}, stop ${setup.stop}, targets ${setup.targets.join("/")}, R:R ${setup.rr}.`
        : setup.bias === "wait"
        ? "WAIT for better entry — see watch zone."
        : "AVOID — quality threshold not met."
    }`;
  }

  return { blocks, summary, disclaimer: DISCLAIMER };
}
