import type { ReactNode } from "react";

/**
 * A single bordered surface. Used sparingly: most content sits directly on the
 * canvas and is separated by whitespace and section headings instead.
 */
export function Panel({
  children,
  className = "",
  as: Tag = "section",
  ...rest
}: {
  children: ReactNode;
  className?: string;
  as?: "section" | "div" | "article" | "aside";
  [key: string]: unknown;
}) {
  return (
    <Tag
      className={`overflow-hidden rounded-xl border border-line bg-surface ${className}`}
      {...rest}
    >
      {children}
    </Tag>
  );
}

export function PanelHeader({
  title,
  description,
  actions,
  className = "",
}: {
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`flex flex-wrap items-start justify-between gap-3 border-b border-line px-5 py-4 ${className}`}
    >
      <div className="min-w-0">
        <h2 className="text-sm font-semibold tracking-tight text-ink">{title}</h2>
        {description && (
          <p className="mt-1 max-w-prose text-xs leading-5 text-ink-muted">{description}</p>
        )}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </div>
  );
}

export function PanelBody({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return <div className={`px-5 py-5 ${className}`}>{children}</div>;
}

/**
 * Section heading used on the canvas itself, so a group of content can have a
 * title without paying for another border.
 */
export function SectionHeading({
  title,
  description,
  actions,
  className = "",
  level = 2,
}: {
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
  className?: string;
  level?: 2 | 3;
}) {
  const Heading = level === 2 ? "h2" : "h3";
  return (
    <div className={`flex flex-wrap items-end justify-between gap-3 ${className}`}>
      <div className="min-w-0">
        <Heading className="text-sm font-semibold tracking-tight text-ink">{title}</Heading>
        {description && (
          <p className="mt-1 max-w-prose text-xs leading-5 text-ink-muted">{description}</p>
        )}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </div>
  );
}
