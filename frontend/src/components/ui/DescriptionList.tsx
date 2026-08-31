import type { ReactNode } from "react";

export type DescriptionItem = {
  label: ReactNode;
  value: ReactNode;
  mono?: boolean;
  span?: boolean;
  hint?: ReactNode;
};

/**
 * Label/value metadata without a box per value. Two columns on wide screens,
 * one on narrow, with generous vertical rhythm.
 */
export function DescriptionList({
  items,
  columns = 2,
  className = "",
}: {
  items: DescriptionItem[];
  columns?: 1 | 2 | 3;
  className?: string;
}) {
  const columnClass =
    columns === 1 ? "" : columns === 2 ? "sm:grid-cols-2" : "sm:grid-cols-2 xl:grid-cols-3";
  return (
    <dl className={`grid gap-x-8 gap-y-5 ${columnClass} ${className}`}>
      {items.map((item, index) => (
        <div key={index} className={`min-w-0 ${item.span ? "sm:col-span-full" : ""}`}>
          <dt className="text-2xs font-medium uppercase tracking-[0.08em] text-ink-faint">
            {item.label}
          </dt>
          <dd
            className={`mt-1.5 break-words text-[13px] leading-5 text-ink ${item.mono ? "font-mono text-xs" : ""}`}
          >
            {item.value}
          </dd>
          {item.hint && <p className="mt-1 text-2xs leading-4 text-ink-muted">{item.hint}</p>}
        </div>
      ))}
    </dl>
  );
}
