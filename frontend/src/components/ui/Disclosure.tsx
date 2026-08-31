import { ChevronRight } from "lucide-react";
import { useId, useState } from "react";
import type { ReactNode } from "react";

/**
 * Collapsed-by-default detail. Secondary telemetry, diagnostics, and metadata
 * live in these instead of being shown all at once.
 */
export function Disclosure({
  title,
  meta,
  defaultOpen = false,
  children,
  className = "",
}: {
  title: ReactNode;
  meta?: ReactNode;
  defaultOpen?: boolean;
  children: ReactNode;
  className?: string;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const id = useId();
  return (
    <div className={`border-b border-line last:border-b-0 ${className}`}>
      <button
        type="button"
        aria-expanded={open}
        aria-controls={id}
        onClick={() => setOpen((current) => !current)}
        className="flex w-full items-center justify-between gap-3 py-3.5 text-left transition hover:text-ink"
      >
        <span className="flex min-w-0 items-center gap-2">
          <ChevronRight
            aria-hidden="true"
            className={`h-4 w-4 shrink-0 text-ink-faint transition-transform ${open ? "rotate-90" : ""}`}
          />
          <span className="truncate text-[13px] font-medium text-ink">{title}</span>
        </span>
        {meta && <span className="shrink-0 text-2xs text-ink-muted">{meta}</span>}
      </button>
      {open && (
        <div id={id} className="pb-5 pl-6 pr-1">
          {children}
        </div>
      )}
    </div>
  );
}

/** Wrapper that gives a stack of disclosures one shared frame. */
export function DisclosureGroup({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={`rounded-xl border border-line bg-surface px-5 ${className}`}>{children}</div>
  );
}
