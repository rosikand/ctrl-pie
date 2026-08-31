import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

/**
 * A row of key numbers. One frame with hairline dividers instead of one card per
 * number, so a metric row reads as a single object.
 */
export function StatGrid({
  children,
  columns = 4,
  className = "",
  bordered = true,
}: {
  children: ReactNode;
  columns?: 2 | 3 | 4;
  className?: string;
  bordered?: boolean;
}) {
  const columnClass =
    columns === 2
      ? "sm:grid-cols-2"
      : columns === 3
        ? "sm:grid-cols-3"
        : "sm:grid-cols-2 xl:grid-cols-4";
  return (
    <dl
      className={[
        "grid grid-cols-2 divide-line",
        columnClass,
        bordered ? "overflow-hidden rounded-xl border border-line bg-surface divide-x" : "divide-x",
        className,
      ].join(" ")}
    >
      {children}
    </dl>
  );
}

export function Stat({
  label,
  value,
  hint,
  icon: Icon,
  className = "",
}: {
  label: ReactNode;
  value: ReactNode;
  hint?: ReactNode;
  icon?: LucideIcon;
  className?: string;
}) {
  return (
    <div className={`px-5 py-4 ${className}`}>
      <dt className="flex items-center gap-1.5 text-2xs font-medium uppercase tracking-[0.08em] text-ink-faint">
        {Icon && <Icon className="h-3.5 w-3.5" />}
        {label}
      </dt>
      <dd className="mt-2 text-xl font-semibold leading-none tracking-tight text-ink">{value}</dd>
      {hint && (
        <p
          className="mt-1.5 truncate text-2xs text-ink-muted"
          title={typeof hint === "string" ? hint : undefined}
        >
          {hint}
        </p>
      )}
    </div>
  );
}

/** A single large number for a hero position. */
export function HeroStat({
  label,
  value,
  hint,
}: {
  label: ReactNode;
  value: ReactNode;
  hint?: ReactNode;
}) {
  return (
    <div>
      <p className="text-2xs font-medium uppercase tracking-[0.08em] text-ink-faint">{label}</p>
      <p className="mt-2 text-4xl font-semibold leading-none tracking-tight text-ink">{value}</p>
      {hint && <p className="mt-2 text-xs text-ink-muted">{hint}</p>}
    </div>
  );
}

/** Horizontal fill meter used for gripper travel and similar bounded values. */
export function Meter({
  value,
  label,
  leadingLabel,
  trailingLabel,
}: {
  value: number;
  label?: ReactNode;
  leadingLabel?: ReactNode;
  trailingLabel?: ReactNode;
}) {
  const percent = Math.max(0, Math.min(100, value));
  return (
    <div>
      <div className="mb-2 flex items-center justify-between text-2xs text-ink-muted">
        <span>{leadingLabel}</span>
        {label && <span className="font-mono font-medium text-ink">{label}</span>}
        <span>{trailingLabel}</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-line">
        <div
          className="h-full rounded-full bg-accent-500 transition-[width] duration-150"
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  );
}
