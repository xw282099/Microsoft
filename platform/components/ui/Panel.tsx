import { cn } from "@/lib/utils";
import { ReactNode } from "react";

interface Props {
  title?: ReactNode;
  subtitle?: ReactNode;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClass?: string;
  glow?: boolean;
}

export function Panel({ title, subtitle, action, children, className, bodyClass, glow }: Props) {
  return (
    <section
      className={cn(
        "bg-bg-card border border-bg-line rounded-xl overflow-hidden",
        glow && "shadow-glow",
        className
      )}
    >
      {(title || action) && (
        <header className="px-4 py-2.5 flex items-center justify-between border-b border-bg-line bg-bg-panel/60">
          <div className="min-w-0">
            {title && <div className="text-sm font-semibold tracking-wide">{title}</div>}
            {subtitle && <div className="text-[11px] text-ink-dim mt-0.5 truncate">{subtitle}</div>}
          </div>
          {action && <div className="shrink-0">{action}</div>}
        </header>
      )}
      <div className={cn("p-4", bodyClass)}>{children}</div>
    </section>
  );
}
