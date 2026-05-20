import Link from "next/link";
import { Check, Crown, Building2, Sparkles } from "lucide-react";
import { Panel } from "@/components/ui/Panel";

const TIERS = [
  {
    name: "Free",
    icon: Sparkles,
    price: "$0",
    desc: "For curious traders exploring the AI workflow.",
    cta: "Start Free",
    features: [
      "Basic macro dashboard",
      "Daily AI market summary",
      "Limited universe (15 tickers)",
      "Delayed data (15-min)",
      "Community access",
    ],
    accent: "border-bg-line",
  },
  {
    name: "Pro",
    icon: Crown,
    price: "$49",
    badge: "Most Popular",
    desc: "For active traders who want institutional tooling.",
    cta: "Upgrade to Pro",
    features: [
      "Full universe (500+ stocks)",
      "AI stock scoring engine",
      "AI trade setups with R:R",
      "Real-time risk monitoring",
      "All 7 AI agents",
      "Trading journal + behavioral AI",
      "Live news intelligence",
      "Email & mobile alerts",
    ],
    accent: "border-accent-cyan/60",
  },
  {
    name: "Institutional",
    icon: Building2,
    price: "Custom",
    desc: "For funds, family offices, and prop desks.",
    cta: "Contact Sales",
    features: [
      "Everything in Pro",
      "Multi-user / multi-account",
      "API access (REST + WebSocket)",
      "Custom AI agents",
      "Whitelabel terminal",
      "Compliance / audit logs",
      "Dedicated support",
      "SLA + on-prem option",
    ],
    accent: "border-accent-ai/60",
  },
];

export default function PricingPage() {
  return (
    <div className="p-4 lg:p-6 space-y-8">
      <div className="text-center max-w-2xl mx-auto">
        <div className="label-xs">Pricing</div>
        <h1 className="text-3xl font-bold mt-1">Plans that scale with your edge.</h1>
        <p className="text-sm text-ink-muted mt-2">
          All plans include unlimited AI queries and full macro coverage. Upgrade for execution-grade tools.
        </p>
      </div>

      <div className="grid md:grid-cols-3 gap-4 max-w-6xl mx-auto">
        {TIERS.map((t) => (
          <div key={t.name} className={`relative p-6 rounded-2xl border-2 bg-bg-card ${t.accent}`}>
            {t.badge && (
              <div className="absolute -top-3 left-1/2 -translate-x-1/2 px-3 py-0.5 rounded-full bg-accent-cyan/20 border border-accent-cyan/40 text-[10px] font-mono uppercase tracking-widest text-accent-cyan">
                {t.badge}
              </div>
            )}
            <div className="flex items-center gap-2">
              <t.icon className="h-5 w-5 text-accent-cyan" />
              <div className="text-base font-semibold">{t.name}</div>
            </div>
            <div className="mt-3 flex items-baseline gap-1">
              <span className="text-4xl font-bold">{t.price}</span>
              {t.name === "Pro" && <span className="text-sm text-ink-muted">/mo</span>}
            </div>
            <p className="mt-2 text-sm text-ink-muted">{t.desc}</p>
            <ul className="mt-5 space-y-2 text-sm">
              {t.features.map((f) => (
                <li key={f} className="flex gap-2">
                  <Check className="h-4 w-4 text-risk-bull shrink-0 mt-0.5" />
                  <span>{f}</span>
                </li>
              ))}
            </ul>
            <Link href="/dashboard" className={`mt-6 inline-flex w-full justify-center btn ${t.name === "Pro" ? "btn-primary" : "btn-ghost"}`}>
              {t.cta}
            </Link>
          </div>
        ))}
      </div>

      <Panel title="Compare features" className="max-w-6xl mx-auto">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-[10px] font-mono uppercase text-ink-dim border-b border-bg-line">
              <tr className="text-left">
                <th className="py-2">Feature</th>
                <th className="text-center">Free</th>
                <th className="text-center">Pro</th>
                <th className="text-center">Institutional</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["Macro dashboard", "✓", "✓", "✓"],
                ["AI stock scoring", "–", "✓", "✓"],
                ["AI trade setups", "–", "✓", "✓"],
                ["Real-time risk center", "–", "✓", "✓"],
                ["All 7 AI agents", "–", "✓", "✓"],
                ["AI Trading Journal", "Limited", "✓", "✓"],
                ["API access", "–", "–", "✓"],
                ["Custom agents", "–", "–", "✓"],
                ["Whitelabel", "–", "–", "✓"],
                ["SLA + support", "Community", "Email", "Dedicated"],
              ].map((r, i) => (
                <tr key={i} className="border-b border-bg-line/40">
                  <td className="py-2.5">{r[0]}</td>
                  <td className="text-center text-ink-muted">{r[1]}</td>
                  <td className="text-center text-accent-cyan">{r[2]}</td>
                  <td className="text-center text-accent-ai">{r[3]}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}
