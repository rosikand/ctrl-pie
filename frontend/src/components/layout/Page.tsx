import { ArrowLeft } from "lucide-react";
import type { ReactNode } from "react";
import { Link } from "react-router-dom";

/** Page container: one max width, one horizontal rhythm, generous vertical space. */
export function Page({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <div className={`mx-auto w-full max-w-page px-6 py-10 sm:px-8 lg:px-12 lg:py-12 ${className}`}>
      {children}
    </div>
  );
}

/**
 * Every screen opens the same way: an optional back link, one title, one
 * sentence, and at most one primary action.
 */
export function PageHeader({
  title,
  description,
  actions,
  meta,
  back,
  className = "",
}: {
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
  meta?: ReactNode;
  back?: { to: string; label: string };
  className?: string;
}) {
  return (
    <header className={className}>
      {back && (
        <Link
          to={back.to}
          className="inline-flex items-center gap-1.5 text-xs font-medium text-ink-muted transition hover:text-ink"
        >
          <ArrowLeft className="h-3.5 w-3.5" aria-hidden="true" />
          {back.label}
        </Link>
      )}
      <div
        className={`flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between ${back ? "mt-4" : ""}`}
      >
        <div className="min-w-0">
          <h1 className="text-2xl font-semibold tracking-tight text-ink sm:text-[28px]">{title}</h1>
          {description && (
            <p className="mt-2 max-w-prose text-[13px] leading-6 text-ink-muted">{description}</p>
          )}
          {meta && <div className="mt-3 flex flex-wrap items-center gap-2">{meta}</div>}
        </div>
        {actions && (
          <div className="flex shrink-0 flex-wrap items-center gap-2 sm:justify-end">{actions}</div>
        )}
      </div>
    </header>
  );
}

/** Vertical rhythm between page sections. */
export function PageSection({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return <section className={`mt-10 ${className}`}>{children}</section>;
}
