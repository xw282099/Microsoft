import Link from "next/link";

export default function NotFound() {
  return (
    <div className="min-h-[60vh] flex flex-col items-center justify-center text-center p-8">
      <div className="text-7xl font-mono font-bold text-accent-cyan">404</div>
      <div className="mt-2 text-ink-muted">Symbol or page not in universe.</div>
      <Link href="/dashboard" className="btn-primary mt-6 text-xs">Back to Command Center</Link>
    </div>
  );
}
