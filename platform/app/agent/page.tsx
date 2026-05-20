"use client";

import { useState } from "react";
import { Brain, Send, Sparkles } from "lucide-react";
import { Panel } from "@/components/ui/Panel";
import { AgentBlockCard } from "@/components/panels/AgentBlockCard";
import { runAgents, ChatResponse } from "@/lib/ai";

const SUGGESTIONS = [
  "Analyze NVDA from macro to valuation.",
  "Which AI infrastructure stocks are under accumulation?",
  "Find high-quality stocks with strong EPS growth and low valuation compression.",
  "Risk-on or risk-off — what's the current regime?",
  "Top sector rotation themes right now.",
  "Build me a trade setup for AVGO.",
  "What are the biggest news catalysts this hour?",
  "How should I hedge concentrated AI exposure?",
];

interface Turn { role: "user" | "ai"; text: string; response?: ChatResponse }

export default function AgentPage() {
  const [input, setInput] = useState("");
  const [thinking, setThinking] = useState(false);
  const [history, setHistory] = useState<Turn[]>([
    {
      role: "ai",
      text: "Welcome to Sentinel AI Terminal. I orchestrate seven specialized agents — Macro, Sector Rotation, Fundamentals, Technical, News, Risk, and Portfolio. Ask me anything about the US market.",
    },
  ]);

  function send(text?: string) {
    const q = (text ?? input).trim();
    if (!q) return;
    setHistory((h) => [...h, { role: "user", text: q }]);
    setInput("");
    setThinking(true);
    // Simulate agent latency
    setTimeout(() => {
      const response = runAgents(q);
      setHistory((h) => [...h, { role: "ai", text: response.summary, response }]);
      setThinking(false);
    }, 700);
  }

  return (
    <div className="p-4 lg:p-6 grid lg:grid-cols-4 gap-4 min-h-[calc(100vh-120px)]">
      <div className="lg:col-span-3 flex flex-col">
        <Panel
          title={<span className="flex items-center gap-2"><Brain className="h-4 w-4 text-accent-ai" />Agent Terminal</span>}
          subtitle="Multi-agent orchestrator · streams synthesized analysis"
          glow
          className="flex-1 flex flex-col"
          bodyClass="p-0 flex-1 flex flex-col"
        >
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {history.map((t, i) => (
              <div key={i} className={t.role === "user" ? "flex justify-end" : ""}>
                {t.role === "user" ? (
                  <div className="max-w-[80%] px-4 py-2.5 rounded-2xl bg-accent-cyan/10 border border-accent-cyan/30 text-sm">
                    {t.text}
                  </div>
                ) : (
                  <div className="space-y-3">
                    <div className="flex items-start gap-3">
                      <div className="h-7 w-7 rounded-md bg-accent-ai/20 border border-accent-ai/40 flex items-center justify-center shrink-0">
                        <Brain className="h-3.5 w-3.5 text-accent-ai" />
                      </div>
                      <div className="rounded-2xl bg-bg-card border border-bg-line px-4 py-3 text-sm flex-1">
                        <div className="text-ink">{t.text}</div>
                        {t.response && (
                          <div className="mt-1 text-[10px] font-mono text-ink-dim uppercase tracking-widest">
                            ↳ Orchestrator · synthesized from {t.response.blocks.length} agents
                          </div>
                        )}
                      </div>
                    </div>
                    {t.response && (
                      <div className="ml-10 space-y-2">
                        {t.response.blocks.map((b, j) => (
                          <AgentBlockCard key={j} block={b} />
                        ))}
                        <div className="text-[10px] text-ink-dim italic px-1">{t.response.disclaimer}</div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
            {thinking && (
              <div className="flex items-center gap-3 text-sm text-ink-muted">
                <div className="h-7 w-7 rounded-md bg-accent-ai/20 border border-accent-ai/40 flex items-center justify-center">
                  <Sparkles className="h-3.5 w-3.5 text-accent-ai animate-pulseSoft" />
                </div>
                Orchestrator routing to specialist agents…
              </div>
            )}
          </div>

          <div className="border-t border-bg-line p-3 bg-bg-panel/50">
            <div className="flex gap-2">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && send()}
                placeholder="Ask AI: 'Analyze NVDA from macro to valuation'…"
                className="flex-1 bg-bg-elev border border-bg-line rounded-md px-3 py-2 text-sm focus:border-accent-cyan/60 outline-none"
              />
              <button onClick={() => send()} className="btn-ai">
                <Send className="h-4 w-4" /> Send
              </button>
            </div>
          </div>
        </Panel>
      </div>

      <div className="space-y-4">
        <Panel title="Suggested Prompts">
          <div className="space-y-1.5">
            {SUGGESTIONS.map((s) => (
              <button
                key={s}
                onClick={() => send(s)}
                className="w-full text-left text-xs p-2 rounded-md border border-bg-line bg-bg-elev/30 hover:bg-bg-elev hover:border-accent-cyan/40 transition"
              >
                {s}
              </button>
            ))}
          </div>
        </Panel>

        <Panel title="Active Agents">
          <ul className="space-y-1.5 text-xs">
            {["Macro", "Sector Rotation", "Fundamentals", "Technical", "News", "Risk", "Portfolio"].map((a) => (
              <li key={a} className="flex items-center justify-between">
                <span className="flex items-center gap-2"><span className="h-1.5 w-1.5 rounded-full bg-risk-bull animate-pulseSoft" /> {a}</span>
                <span className="text-ink-dim font-mono">ready</span>
              </li>
            ))}
          </ul>
        </Panel>
      </div>
    </div>
  );
}
