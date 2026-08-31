import type { ReactNode } from "react";
import { Link } from "react-router-dom";

type Align = "left" | "right";

/**
 * Table shell. Datasets, models, runs, deployments, robots, and sessions all use
 * this instead of card grids so scanning many rows stays cheap.
 */
export function Table({
  children,
  minWidth = "44rem",
  label,
  busy,
}: {
  children: ReactNode;
  minWidth?: string;
  label?: string;
  busy?: boolean;
}) {
  return (
    <div className="overflow-hidden rounded-xl border border-line bg-surface">
      <div className="overflow-x-auto">
        <table
          aria-label={label}
          aria-busy={busy}
          className="w-full border-collapse text-left"
          style={{ minWidth }}
        >
          {children}
        </table>
      </div>
    </div>
  );
}

export function TableHead({ children }: { children: ReactNode }) {
  return (
    <thead>
      <tr className="border-b border-line bg-canvas/70">{children}</tr>
    </thead>
  );
}

export function TableHeaderCell({
  children,
  align = "left",
  className = "",
}: {
  children?: ReactNode;
  align?: Align;
  className?: string;
}) {
  return (
    <th
      scope="col"
      className={`whitespace-nowrap px-4 py-2.5 text-2xs font-medium uppercase tracking-[0.08em] text-ink-faint first:pl-5 last:pr-5 ${
        align === "right" ? "text-right" : ""
      } ${className}`}
    >
      {children}
    </th>
  );
}

export function TableBody({ children }: { children: ReactNode }) {
  return <tbody className="divide-y divide-line-subtle">{children}</tbody>;
}

export function TableRow({
  children,
  interactive = false,
  selected = false,
  onClick,
  className = "",
}: {
  children: ReactNode;
  interactive?: boolean;
  selected?: boolean;
  onClick?: () => void;
  className?: string;
}) {
  return (
    <tr
      onClick={onClick}
      className={[
        "relative transition-colors",
        interactive ? "cursor-pointer hover:bg-canvas" : "",
        selected ? "bg-accent-50/60" : "",
        className,
      ]
        .filter(Boolean)
        .join(" ")}
    >
      {children}
    </tr>
  );
}

export function TableCell({
  children,
  align = "left",
  mono = false,
  muted = false,
  className = "",
  colSpan,
}: {
  children?: ReactNode;
  align?: Align;
  mono?: boolean;
  muted?: boolean;
  className?: string;
  colSpan?: number;
}) {
  return (
    <td
      colSpan={colSpan}
      className={[
        "px-4 py-3 text-[13px] first:pl-5 last:pr-5",
        align === "right" ? "text-right tabular-nums" : "",
        mono ? "font-mono text-xs" : "",
        muted ? "text-ink-muted" : "text-ink-secondary",
        className,
      ]
        .filter(Boolean)
        .join(" ")}
    >
      {children}
    </td>
  );
}

/**
 * Stretches over the whole row so the entire row is one click target while the
 * accessible name stays on a single link.
 */
export function RowLink({
  to,
  children,
  className = "",
}: {
  to: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <Link
      to={to}
      className={`font-medium text-ink after:absolute after:inset-0 after:content-[''] hover:text-accent-700 ${className}`}
    >
      {children}
    </Link>
  );
}

/** Row-wide activation for rows that open a drawer instead of a page. */
export function RowButton({
  onClick,
  children,
  disabled = false,
  className = "",
}: {
  onClick: () => void;
  children: ReactNode;
  disabled?: boolean;
  className?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`text-left font-medium text-ink after:absolute after:inset-0 after:content-[''] hover:text-accent-700 disabled:cursor-not-allowed disabled:text-ink-muted ${className}`}
    >
      {children}
    </button>
  );
}

export function TableCaptionRow({ children }: { children: ReactNode }) {
  return (
    <tr>
      <td colSpan={99} className="px-5 py-4 text-xs text-ink-muted">
        {children}
      </td>
    </tr>
  );
}
