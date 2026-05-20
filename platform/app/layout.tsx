import "./globals.css";
import type { Metadata } from "next";
import { AppShell } from "@/components/AppShell";

export const metadata: Metadata = {
  title: "Sentinel AI — US Stock Intelligence Terminal",
  description:
    "Institutional-grade AI-powered US stock intelligence platform. Macro → Sector → Stock → Risk.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-bg-base text-ink antialiased">
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
