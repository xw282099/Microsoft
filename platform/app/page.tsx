import Link from "next/link";
import { ArrowRight, Brain, Compass, Layers, LineChart, Shield, Zap, Activity, BarChart3, Newspaper, MessageSquare } from "lucide-react";
import { STOCKS, MACRO_SIGNALS, SECTOR_ROTATION } from "@/lib/data";
import { fmtPct } from "@/lib/utils";

export default function Landing() {
  const topMovers = [...STOCKS].sort((a, b) => b.change - a.change).slice(0, 6);
  const topSectors = [...SECTOR_ROTATION].sort((a, b) => b.rsRank - a.rsRank).slice(0, 5);

  return (
    <div className="min-h-screen bg-bg-base text-ink overflow-hidden">
      {/* Top header */}
      <header className="relative z-20 px-6 lg:px-10 py-4 flex items-center justify-between border-b border-bg-line/50 backdrop-blur bg-bg-base/60">
        <Link href="/" className="flex items-center gap-2">
          <div className="h-8 w-8 rounded-md bg-accent-ai/20 border border-accent-ai/40 flex items-center justify-center">
            <Brain className="h-4 w-4 text-accent-ai" />
          </div>
          <div>
            <div className="text-sm font-semibold tracking-wide">SENTINEL</div>
            <div className="text-[10px] font-mono text-ink-dim uppercase">AI Terminal</div>
          </div>
        </Link>
        <nav className="hidden md:flex gap-7 text-sm text-ink-muted">
          <a href="#features" className="hover:text-ink">Features</a>
          <a href="#agents" className="hover:text-ink">AI Agents</a>
          <a href="#workflow" className="hover:text-ink">Workflow</a>
          <Link href="/pricing" className="hover:text-ink">Pricing</Link>
        </nav>
        <div className="flex items-center gap-2">
          <Link href="/dashboard" className="btn-ghost text-xs">Live Demo</Link>
          <Link href="/dashboard" className="btn-primary text-xs">
            Launch Terminal <ArrowRight className="h-3.5 w-3.5" />
          </Link>
        </div>
      </header>

      {/* Hero */}
      <section className="relative">
        <div className="absolute inset-0 grid-bg opacity-30 pointer-events-none" />
        <div className="absolute inset-0 bg-aurora pointer-events-none" />
        <div className="relative max-w-7xl mx-auto px-6 lg:px-10 pt-16 pb-20 grid lg:grid-cols-12 gap-10 items-center">
          <div className="lg:col-span-7">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-accent-ai/40 bg-accent-ai/10 text-accent-ai text-xs font-mono">
              <span className="h-1.5 w-1.5 rounded-full bg-accent-ai animate-pulseSoft" />
              AI AGENTS · LIVE · 7 MODELS COLLABORATING
            </div>
            <h1 className="mt-5 text-4xl md:text-6xl font-bold leading-[1.05] tracking-tight">
              AI-Powered <span className="text-accent-cyan">US Stock</span><br />
              Intelligence Platform
            </h1>
            <p className="mt-5 text-lg text-ink-muted max-w-2xl">
              Analyze market trends, sector rotation, risk signals, and high-probability trade setups — all driven by collaborating AI agents.
              Bloomberg-grade data. Hedge-fund workflow. Trend-first methodology.
            </p>
            <div className="mt-7 flex flex-wrap gap-3">
              <Link href="/dashboard" className="btn-primary">
                <Zap className="h-4 w-4" /> Start Analysis
              </Link>
              <Link href="/agent" className="btn-ai">
                <Brain className="h-4 w-4" /> Talk to AI Agent
              </Link>
              <Link href="/sectors" className="btn-ghost">
                <Layers className="h-4 w-4" /> View Sector Rotation
              </Link>
            </div>

            <div className="mt-8 grid grid-cols-2 sm:grid-cols-4 gap-3 max-w-2xl">
              {[
                ["7", "AI Agents"],
                ["120+", "Indicators"],
                ["1.2M", "Data Points/Day"],
                ["<50ms", "Stream Latency"],
              ].map(([v, l]) => (
                <div key={l} className="p-3 rounded-lg border border-bg-line bg-bg-card/60">
                  <div className="text-2xl font-mono font-bold text-accent-cyan">{v}</div>
                  <div className="text-[11px] text-ink-dim font-mono uppercase tracking-wider">{l}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Live preview card */}
          <div className="lg:col-span-5">
            <div className="relative">
              <div className="absolute -inset-1 bg-gradient-to-br from-accent-ai/30 to-accent-cyan/20 blur-2xl rounded-2xl" />
              <div className="relative rounded-2xl border border-bg-line bg-bg-panel/90 backdrop-blur p-4 shadow-glow">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2 text-xs font-mono text-ink-muted">
                    <span className="h-2 w-2 rounded-full bg-risk-bull animate-pulseSoft" />
                    LIVE FEED · MACRO
                  </div>
                  <div className="text-[10px] font-mono text-ink-dim">2026-05-20 · NYSE</div>
                </div>

                <div className="grid grid-cols-2 gap-2 mb-3">
                  {MACRO_SIGNALS.slice(0, 4).map((m) => (
                    <div key={m.key} className="p-2.5 rounded-md bg-bg-card border border-bg-line">
                      <div className="flex items-center justify-between">
                        <span className="text-[10px] font-mono text-ink-dim uppercase">{m.label}</span>
                        <span className={"text-[10px] font-mono " + (m.state === "bull" ? "text-risk-bull" : m.state === "bear" ? "text-risk-bear" : "text-risk-neutral")}>
                          {m.state.toUpperCase()}
                        </span>
                      </div>
                      <div className="mt-0.5 flex items-baseline gap-2">
                        <span className="text-base font-mono font-semibold">{m.value}{m.unit}</span>
                        <span className={"text-[10px] font-mono " + (m.change >= 0 ? "text-risk-bull" : "text-risk-bear")}>
                          {fmtPct(m.change)}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>

                <div className="text-[11px] font-mono uppercase tracking-widest text-ink-dim mb-1.5">Top movers</div>
                <div className="space-y-1">
                  {topMovers.map((s) => (
                    <Link key={s.symbol} href={`/stocks/${s.symbol}`}
                      className="flex items-center justify-between text-xs py-1.5 px-2 rounded-md hover:bg-bg-elev/60">
                      <div className="flex items-center gap-2">
                        <span className="font-mono font-bold w-12">{s.symbol}</span>
                        <span className="text-ink-muted truncate">{s.name}</span>
                      </div>
                      <div className="flex items-center gap-3 font-mono">
                        <span>${s.price.toFixed(2)}</span>
                        <span className={s.change >= 0 ? "text-risk-bull" : "text-risk-bear"}>{fmtPct(s.change)}</span>
                      </div>
                    </Link>
                  ))}
                </div>

                <div className="mt-3 p-3 rounded-md border border-accent-ai/30 bg-accent-ai/5">
                  <div className="text-[10px] font-mono uppercase tracking-widest text-accent-ai">AI Market Narrative</div>
                  <p className="text-xs text-ink mt-1 leading-relaxed">
                    "AI infrastructure spending accelerates; liquidity supportive. Capital rotating from extended mega-cap SaaS into 2nd-derivative power, cooling, and optical names."
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Core philosophy */}
      <section className="border-y border-bg-line bg-bg-panel/50">
        <div className="max-w-7xl mx-auto px-6 lg:px-10 py-12">
          <div className="text-center max-w-3xl mx-auto">
            <div className="label-xs">CORE PHILOSOPHY</div>
            <h2 className="mt-2 text-3xl font-bold">Trend First. Entry Second.</h2>
            <p className="mt-3 text-ink-muted">
              Every decision flows through a single, disciplined chain — from global macro
              regime down to position-level risk. If the trend isn't there, the stock isn't worth trading.
            </p>
          </div>
          <div className="mt-8 grid md:grid-cols-5 gap-2">
            {[
              { icon: Compass, label: "Macro", caption: "Liquidity · Rates · Vol" },
              { icon: Layers, label: "Sector", caption: "Rotation · Flow · RS" },
              { icon: BarChart3, label: "Stock", caption: "Score · Setup · Catalyst" },
              { icon: Activity, label: "Timing", caption: "Entry · Stop · Target" },
              { icon: Shield, label: "Risk", caption: "Size · Hedge · Cash" },
            ].map(({ icon: Icon, label, caption }, i) => (
              <div key={label} className="relative">
                <div className="p-4 rounded-lg border border-bg-line bg-bg-card text-center">
                  <Icon className="h-5 w-5 text-accent-cyan mx-auto" />
                  <div className="mt-2 text-sm font-semibold">{label}</div>
                  <div className="text-[11px] text-ink-dim font-mono mt-0.5">{caption}</div>
                </div>
                {i < 4 && (
                  <div className="hidden md:block absolute top-1/2 -right-2 -translate-y-1/2 text-accent-cyan">→</div>
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features */}
      <section id="features" className="max-w-7xl mx-auto px-6 lg:px-10 py-16">
        <div className="text-center mb-10">
          <div className="label-xs">INTELLIGENCE STACK</div>
          <h2 className="mt-2 text-3xl font-bold">Seven Modules. One Terminal.</h2>
        </div>
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          {FEATURES.map((f) => (
            <div key={f.title} className="p-5 rounded-xl border border-bg-line bg-bg-card hover:border-accent-cyan/40 transition">
              <f.icon className="h-5 w-5 text-accent-cyan" />
              <div className="mt-3 text-base font-semibold">{f.title}</div>
              <p className="mt-1.5 text-sm text-ink-muted leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* AI Agents */}
      <section id="agents" className="border-y border-bg-line bg-bg-panel/40">
        <div className="max-w-7xl mx-auto px-6 lg:px-10 py-16">
          <div className="grid lg:grid-cols-2 gap-10 items-center">
            <div>
              <div className="label-xs">AGENT ARCHITECTURE</div>
              <h2 className="mt-2 text-3xl font-bold">Seven AI Agents. One Investment Thesis.</h2>
              <p className="mt-4 text-ink-muted">
                Each agent owns a domain — macro, rotation, fundamentals, technicals, news, risk, portfolio.
                The orchestrator routes your query, runs the right specialists in parallel, and synthesizes a
                structured, auditable answer with citations and risk notes.
              </p>
              <Link href="/agent" className="btn-ai mt-6">
                <MessageSquare className="h-4 w-4" /> Open AI Terminal
              </Link>
            </div>
            <div className="grid grid-cols-2 gap-3">
              {AGENTS.map((a) => (
                <div key={a.name} className="p-4 rounded-lg border border-bg-line bg-bg-card">
                  <div className="flex items-center gap-2">
                    <a.icon className="h-4 w-4 text-accent-ai" />
                    <div className="text-sm font-semibold">{a.name}</div>
                  </div>
                  <div className="mt-1.5 text-xs text-ink-muted">{a.desc}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Sector rotation preview */}
      <section className="max-w-7xl mx-auto px-6 lg:px-10 py-16">
        <div className="text-center mb-8">
          <div className="label-xs">LIVE SECTOR ROTATION</div>
          <h2 className="mt-2 text-3xl font-bold">Where is institutional capital going?</h2>
        </div>
        <div className="grid md:grid-cols-5 gap-3">
          {topSectors.map((s) => (
            <div key={s.sector} className="p-4 rounded-xl border border-bg-line bg-bg-card relative overflow-hidden">
              <div className="absolute top-2 right-2 text-[10px] font-mono text-accent-cyan">RS {s.rsRank}</div>
              <div className="text-sm font-semibold">{s.sector}</div>
              <div className="mt-2 text-xs text-ink-muted leading-relaxed">{s.narrative}</div>
              <div className="mt-3 flex justify-between text-[11px] font-mono">
                <span className="text-ink-dim">FLOW</span>
                <span className={s.flow > 0 ? "text-risk-bull" : "text-risk-bear"}>{s.flow > 0 ? "+" : ""}{s.flow}</span>
              </div>
            </div>
          ))}
        </div>
        <div className="text-center mt-8">
          <Link href="/sectors" className="btn-primary">Full Rotation Matrix <ArrowRight className="h-4 w-4" /></Link>
        </div>
      </section>

      {/* CTA */}
      <section className="border-t border-bg-line">
        <div className="max-w-5xl mx-auto px-6 lg:px-10 py-16 text-center bg-aurora">
          <h2 className="text-3xl font-bold">Trade with conviction. Trade with AI.</h2>
          <p className="mt-3 text-ink-muted max-w-2xl mx-auto">
            Stop guessing. Stop reacting. Let seven specialized AI agents do the heavy lifting so you can focus on execution.
          </p>
          <div className="mt-6 flex justify-center gap-3">
            <Link href="/dashboard" className="btn-primary">Launch Terminal</Link>
            <Link href="/pricing" className="btn-ghost">View Pricing</Link>
          </div>
        </div>
      </section>

      <footer className="border-t border-bg-line px-6 py-6 text-[11px] text-ink-dim flex flex-wrap gap-x-4 gap-y-1 justify-between">
        <div>© 2026 Sentinel AI Terminal. Educational and research use only — not investment advice. Trading involves risk of loss.</div>
        <div className="font-mono">Engine v1.0 · Synthetic demo data</div>
      </footer>
    </div>
  );
}

const FEATURES = [
  { icon: Compass, title: "AI Macro Trend Engine", desc: "SPY, QQQ, IWM, VIX, DXY, US10Y, liquidity, Fed, CPI — synthesized into a single market regime score." },
  { icon: Layers, title: "Sector Rotation System", desc: "Detect institutional flow, relative strength, valuation compression across 11+ sectors in real time." },
  { icon: BarChart3, title: "AI Stock Scoring", desc: "Composite score: trend, fundamentals, valuation, risk. Filter the universe with a single number." },
  { icon: LineChart, title: "AI Trade Setups", desc: "Auto-generated watch / entry / stop / target zones with R:R and position sizing." },
  { icon: Newspaper, title: "News Intelligence", desc: "SEC filings, earnings calls, Fed speeches, X / Reddit sentiment — scored for impact and bias." },
  { icon: Shield, title: "Risk Control Center", desc: "Portfolio risk, sector exposure, correlation, earnings tail risk. Protect capital first." },
];

const AGENTS = [
  { icon: Compass, name: "Macro Agent", desc: "Liquidity, rates, vol regime." },
  { icon: Layers, name: "Sector Rotation Agent", desc: "Relative strength & flow." },
  { icon: LineChart, name: "Fundamentals Agent", desc: "Growth, margins, FCF." },
  { icon: Activity, name: "Technical Agent", desc: "Trend, structure, momentum." },
  { icon: Newspaper, name: "News Agent", desc: "Catalysts & sentiment." },
  { icon: Shield, name: "Risk Agent", desc: "Sizing, stops, hedges." },
  { icon: BarChart3, name: "Portfolio Agent", desc: "Allocation & rebalancing." },
  { icon: Brain, name: "Orchestrator", desc: "Routes & synthesizes." },
];
