import { AlertCircle, CheckCircle2, Info, TriangleAlert } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

import type { Tone } from "./Badge";

const TONES: Record<Tone, { frame: string; icon: string; title: string; body: string }> = {
  neutral: {
    frame: "border-line bg-canvas",
    icon: "text-ink-muted",
    title: "text-ink",
    body: "text-ink-secondary",
  },
  info: {
    frame: "border-accent-100 bg-accent-50",
    icon: "text-accent-600",
    title: "text-accent-800",
    body: "text-accent-800/80",
  },
  success: {
    frame: "border-positive-100 bg-positive-50",
    icon: "text-positive-600",
    title: "text-positive-700",
    body: "text-positive-700/85",
  },
  warning: {
    frame: "border-caution-100 bg-caution-50",
    icon: "text-caution-600",
    title: "text-caution-700",
    body: "text-caution-700/85",
  },
  danger: {
    frame: "border-critical-100 bg-critical-50",
    icon: "text-critical-600",
    title: "text-critical-700",
    body: "text-critical-700/85",
  },
};

const ICONS: Record<Tone, LucideIcon> = {
  neutral: Info,
  info: Info,
  success: CheckCircle2,
  warning: TriangleAlert,
  danger: AlertCircle,
};

/**
 * Safety and status messaging. Every state that blocks or warns an operator uses
 * this so the treatment is identical on every screen.
 */
export function Alert({
  tone = "info",
  title,
  children,
  action,
  icon,
  role,
  className = "",
}: {
  tone?: Tone;
  title?: ReactNode;
  children?: ReactNode;
  action?: ReactNode;
  icon?: LucideIcon | null;
  role?: "alert" | "status";
  className?: string;
}) {
  const styles = TONES[tone];
  const Icon = icon === null ? null : (icon ?? ICONS[tone]);
  return (
    <div
      role={role ?? (tone === "danger" ? "alert" : undefined)}
      className={`flex flex-wrap items-start gap-3 rounded-lg border px-4 py-3 ${styles.frame} ${className}`}
    >
      {Icon && <Icon className={`mt-0.5 h-4 w-4 shrink-0 ${styles.icon}`} aria-hidden="true" />}
      <div className="min-w-0 flex-1">
        {title && <p className={`text-xs font-semibold ${styles.title}`}>{title}</p>}
        {children && (
          <div className={`${title ? "mt-1" : ""} text-xs leading-5 ${styles.body}`}>{children}</div>
        )}
      </div>
      {action && <div className="shrink-0">{action}</div>}
    </div>
  );
}
