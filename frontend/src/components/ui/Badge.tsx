import type { ReactNode } from "react";

import { sentenceCase } from "../../lib/format";

/**
 * Status vocabulary for the whole product. Green, amber, and red never appear
 * outside these tones, so a colored pill always means "state", never "decoration".
 */
export type Tone = "neutral" | "info" | "success" | "warning" | "danger";

const BADGE_TONES: Record<Tone, string> = {
  neutral: "bg-line-subtle text-ink-secondary",
  info: "bg-accent-50 text-accent-700",
  success: "bg-positive-50 text-positive-700",
  warning: "bg-caution-50 text-caution-700",
  danger: "bg-critical-50 text-critical-700",
};

const DOT_TONES: Record<Tone, string> = {
  neutral: "bg-ink-faint",
  info: "bg-accent-500",
  success: "bg-positive-500",
  warning: "bg-caution-500",
  danger: "bg-critical-500",
};

export function StatusDot({
  tone,
  pulse = false,
  className = "",
}: {
  tone: Tone;
  pulse?: boolean;
  className?: string;
}) {
  return (
    <span
      aria-hidden="true"
      className={`h-1.5 w-1.5 shrink-0 rounded-full ${DOT_TONES[tone]} ${pulse ? "animate-pulse" : ""} ${className}`}
    />
  );
}

export function Badge({
  tone = "neutral",
  children,
  dot = false,
  pulse = false,
  className = "",
  title,
}: {
  tone?: Tone;
  children: ReactNode;
  dot?: boolean;
  pulse?: boolean;
  className?: string;
  title?: string;
}) {
  return (
    <span
      title={title}
      className={`inline-flex items-center gap-1.5 whitespace-nowrap rounded-full px-2 py-0.5 text-2xs font-medium ${BADGE_TONES[tone]} ${className}`}
    >
      {dot && <StatusDot tone={tone} pulse={pulse} />}
      {typeof children === "string" ? sentenceCase(children) : children}
    </span>
  );
}

/** Monospace chip for identifiers, revisions, and namespaces. */
export function Mono({
  children,
  className = "",
  title,
}: {
  children: ReactNode;
  className?: string;
  title?: string;
}) {
  return (
    <span
      title={title}
      className={`rounded-md bg-line-subtle px-1.5 py-0.5 font-mono text-2xs text-ink-secondary ${className}`}
    >
      {children}
    </span>
  );
}
