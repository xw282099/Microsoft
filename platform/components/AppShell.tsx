"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Activity, BarChart3, Brain, Compass, Layers, LineChart, MessageSquare,
  Newspaper, Shield, BookOpen, CircleDollarSign, Search, Bell, Wifi
} from "lucide-react";
import { TickerTape } from "./TickerTape";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/dashboard", label: "Command Center", icon: Activity },
  { href: "/macro", label: "Macro Engine", icon: Compass },
  { href: "/sectors", label: "Sector Rotation", icon: Layers },
  { href: "/stocks", label: "AI Scoring", icon: BarChart3 },
  { href: "/agent", label: "AI Agent", icon: Brain },
  { href: "/risk", label: "Risk Center", icon: Shield },
  { href: "/journal", label: "Journal", icon: BookOpen },
  { href: "/pricing", label: "Pricing", icon: CircleDollarSign },
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const path = usePathname();
  const isLanding = path === "/";

  if (isLanding) {
    // Landing page renders without the dashboard chrome.
    return <>{children}</>;
  }

  return (
    <div className="flex min-h-screen">
      {/* Sidebar */}
      <aside className="hidden lg:flex w-60 flex-col bg-bg-panel border-r border-bg-line sticky top-0 h-screen">
        <div className="px-5 py-5 border-b border-bg-line flex items-center gap-2">
          <div className="h-8 w-8 rounded-md bg-accent-ai/20 border border-accent-ai/40 flex items-center justify-center">
            <Brain className="h-4 w-4 text-accent-ai" />
          </div>
          <div className="leading-tight">
            <div className="text-sm font-semibold tracking-wide">SENTINEL</div>
            <div className="text-[10px] font-mono text-ink-dim uppercase">AI Terminal v1.0</div>
          </div>
        </div>
        <nav className="flex-1 px-2 py-3 space-y-0.5">
          {NAV.map(({ href, label, icon: Icon }) => {
            const active = path === href || path.startsWith(href + "/");
            return (
              <Link
                key={href}
                href={href}
                className={cn(
                  "flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors",
                  active
                    ? "bg-accent-cyan/10 text-accent-cyan border border-accent-cyan/30"
                    : "text-ink-muted hover:bg-bg-elev hover:text-ink"
                )}
              >
                <Icon className="h-4 w-4" />
                {label}
              </Link>
            );
          })}
        </nav>
        <div className="px-3 py-3 border-t border-bg-line">
          <div className="rounded-md p-3 bg-gradient-to-br from-accent-ai/10 to-accent-cyan/5 border border-accent-ai/30">
            <div className="text-[10px] font-mono uppercase tracking-wider text-accent-ai">Pro Plan</div>
            <div className="text-xs text-ink-muted mt-1">AI Agents · Live Risk · Full Universe</div>
            <Link href="/pricing" className="mt-2 inline-block text-xs text-accent-cyan hover:underline">
              Upgrade →
            </Link>
          </div>
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 min-w-0 flex flex-col">
        {/* Top bar */}
        <div className="sticky top-0 z-40 bg-bg-panel/95 backdrop-blur border-b border-bg-line">
          <div className="flex items-center gap-3 px-4 py-2.5">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-ink-dim" />
              <input
                placeholder="Search ticker, sector, or ask AI…"
                className="w-full bg-bg-card border border-bg-line rounded-md pl-9 pr-3 py-1.5 text-sm placeholder:text-ink-dim focus:outline-none focus:border-accent-cyan/60"
              />
            </div>
            <div className="hidden md:flex items-center gap-2 text-[11px] font-mono">
              <span className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-bg-card border border-bg-line">
                <span className="h-1.5 w-1.5 rounded-full bg-risk-bull animate-pulseSoft" />
                <span className="text-ink-muted">MARKET</span>
                <span className="text-risk-bull">OPEN</span>
              </span>
              <span className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-bg-card border border-bg-line">
                <Wifi className="h-3 w-3 text-accent-cyan" />
                <span className="text-ink-muted">LATENCY</span>
                <span className="text-ink">38ms</span>
              </span>
            </div>
            <button className="p-2 rounded-md bg-bg-card border border-bg-line hover:bg-bg-elev">
              <Bell className="h-4 w-4" />
            </button>
            <div className="h-8 w-8 rounded-full bg-accent-cyan/20 border border-accent-cyan/40 flex items-center justify-center text-xs font-semibold">
              AI
            </div>
          </div>
          <TickerTape />
        </div>

        <div className="flex-1 min-w-0">{children}</div>

        <footer className="border-t border-bg-line px-4 py-3 text-[11px] text-ink-dim flex flex-wrap gap-x-4 gap-y-1 justify-between">
          <div>© 2026 Sentinel AI Terminal. Education & research only — not investment advice.</div>
          <div className="font-mono">Data: synthetic demo · Engine: Sentinel Agent v1.0</div>
        </footer>
      </main>
    </div>
  );
}
