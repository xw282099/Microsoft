"use client";

import { useState } from "react";
import { Panel } from "@/components/ui/Panel";
import { Stat } from "@/components/ui/Stat";
import { BookOpen, PlusCircle } from "lucide-react";

interface Entry {
  id: string;
  date: string;
  symbol: string;
  side: "long" | "short";
  thesis: string;
  emotion: "calm" | "fomo" | "fear" | "greed" | "discipline";
  result?: "win" | "loss" | "open";
  notes: string;
}

const SEED: Entry[] = [
  { id: "j1", date: "2026-05-18", symbol: "NVDA", side: "long", thesis: "Hyperscaler capex tailwind + Blackwell ramp. Trend in stage 2 mark-up.", emotion: "discipline", result: "open", notes: "Sized 4%; stop -7% from entry; trail with 20EMA." },
  { id: "j2", date: "2026-05-17", symbol: "VRT",  side: "long", thesis: "Liquid cooling backlog + raised guide. Sector RS 91.", emotion: "discipline", result: "open", notes: "Added 2% on pullback to 21EMA." },
  { id: "j3", date: "2026-05-15", symbol: "ZS",   side: "short", thesis: "Cyber spending recalibration, weak guide.", emotion: "calm", result: "win", notes: "Closed 1/2 at +5%. Trailing balance." },
  { id: "j4", date: "2026-05-12", symbol: "META", side: "long", thesis: "Reels monetization inflection. FOMO entry near highs.", emotion: "fomo", result: "loss", notes: "Stopped out -3%. Lesson: wait for pullback, don't chase." },
];

export default function JournalPage() {
  const [entries, setEntries] = useState<Entry[]>(SEED);
  const [open, setOpen] = useState(false);
  const [form, setForm] = useState<Omit<Entry, "id" | "date">>({
    symbol: "", side: "long", thesis: "", emotion: "discipline", notes: "", result: "open",
  });

  function add() {
    if (!form.symbol || !form.thesis) return;
    setEntries((e) => [
      { id: "j" + (e.length + 1), date: new Date().toISOString().slice(0, 10), ...form, symbol: form.symbol.toUpperCase() },
      ...e,
    ]);
    setForm({ symbol: "", side: "long", thesis: "", emotion: "discipline", notes: "", result: "open" });
    setOpen(false);
  }

  const wins = entries.filter((e) => e.result === "win").length;
  const losses = entries.filter((e) => e.result === "loss").length;
  const winRate = wins + losses > 0 ? Math.round((wins / (wins + losses)) * 100) : 0;
  const fomoCount = entries.filter((e) => e.emotion === "fomo").length;

  return (
    <div className="p-4 lg:p-6 space-y-4">
      <div className="flex items-end justify-between flex-wrap gap-3">
        <div>
          <div className="label-xs">AI Trading Journal</div>
          <h1 className="text-2xl font-bold tracking-tight">Reflect. Refine. Repeat.</h1>
          <p className="text-sm text-ink-muted mt-1">AI analyzes your trades for emotional bias and pattern leaks.</p>
        </div>
        <button onClick={() => setOpen(true)} className="btn-primary text-xs">
          <PlusCircle className="h-4 w-4" /> New Entry
        </button>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Stat label="Total Entries" value={entries.length} />
        <Stat label="Win Rate" value={`${winRate}%`} tone={winRate >= 55 ? "bull" : "neutral"} />
        <Stat label="FOMO Entries" value={fomoCount} tone={fomoCount > 0 ? "bear" : "bull"} hint="Track discipline leaks" />
        <Stat label="Open Positions" value={entries.filter((e) => e.result === "open").length} />
      </div>

      <Panel title="AI Behavioral Analysis" glow>
        <div className="text-sm text-ink leading-relaxed">
          Your largest loss this month came from a <strong className="text-risk-bear">FOMO entry on META near the highs</strong>.
          You've shown <strong className="text-risk-bull">strong discipline on shorts</strong> (1W, 0L) but a tendency
          to chase momentum on long ideas. Recommendation: institute a <em>mandatory 4-hour cooling-off</em> before any long entry within 2% of the prior session high.
        </div>
      </Panel>

      <Panel title="Trade Log">
        <div className="space-y-2">
          {entries.map((e) => (
            <div key={e.id} className="p-3 rounded-md border border-bg-line bg-bg-elev/30">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span className="font-mono font-bold">{e.symbol}</span>
                  <span className={"chip " + (e.side === "long" ? "chip-bull" : "chip-bear")}>{e.side}</span>
                  <span className="chip chip-neutral">{e.emotion}</span>
                  {e.result && (
                    <span className={"chip " + (e.result === "win" ? "chip-bull" : e.result === "loss" ? "chip-bear" : "chip-cyan")}>
                      {e.result}
                    </span>
                  )}
                </div>
                <span className="text-[11px] font-mono text-ink-dim">{e.date}</span>
              </div>
              <div className="mt-1.5 text-sm">{e.thesis}</div>
              <div className="mt-1 text-[12px] text-ink-muted">{e.notes}</div>
            </div>
          ))}
        </div>
      </Panel>

      {open && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="w-full max-w-md panel p-5">
            <div className="flex items-center gap-2 mb-4">
              <BookOpen className="h-4 w-4 text-accent-cyan" />
              <h2 className="text-sm font-semibold">New Journal Entry</h2>
            </div>
            <div className="space-y-2.5 text-sm">
              <div>
                <div className="label-xs mb-1">Symbol</div>
                <input value={form.symbol} onChange={(e) => setForm({ ...form, symbol: e.target.value.toUpperCase() })}
                  className="w-full bg-bg-elev border border-bg-line rounded-md px-2 py-1.5" />
              </div>
              <div>
                <div className="label-xs mb-1">Side</div>
                <select value={form.side} onChange={(e) => setForm({ ...form, side: e.target.value as any })}
                  className="w-full bg-bg-elev border border-bg-line rounded-md px-2 py-1.5">
                  <option value="long">Long</option>
                  <option value="short">Short</option>
                </select>
              </div>
              <div>
                <div className="label-xs mb-1">Thesis</div>
                <textarea value={form.thesis} onChange={(e) => setForm({ ...form, thesis: e.target.value })}
                  rows={3} className="w-full bg-bg-elev border border-bg-line rounded-md px-2 py-1.5" />
              </div>
              <div>
                <div className="label-xs mb-1">Emotion</div>
                <select value={form.emotion} onChange={(e) => setForm({ ...form, emotion: e.target.value as any })}
                  className="w-full bg-bg-elev border border-bg-line rounded-md px-2 py-1.5">
                  <option value="discipline">Discipline</option>
                  <option value="calm">Calm</option>
                  <option value="fomo">FOMO</option>
                  <option value="fear">Fear</option>
                  <option value="greed">Greed</option>
                </select>
              </div>
              <div>
                <div className="label-xs mb-1">Notes</div>
                <textarea value={form.notes} onChange={(e) => setForm({ ...form, notes: e.target.value })}
                  rows={2} className="w-full bg-bg-elev border border-bg-line rounded-md px-2 py-1.5" />
              </div>
            </div>
            <div className="mt-4 flex justify-end gap-2">
              <button onClick={() => setOpen(false)} className="btn-ghost text-xs">Cancel</button>
              <button onClick={add} className="btn-primary text-xs">Save Entry</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
