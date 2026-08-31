import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

/**
 * One empty/error/zero-result treatment for the whole product.
 * `bordered` is off when the state already sits inside a panel or table.
 */
export function EmptyState({
  icon: Icon,
  title,
  description,
  action,
  detail,
  bordered = true,
  role,
  className = "",
}: {
  icon?: LucideIcon;
  title: ReactNode;
  description?: ReactNode;
  action?: ReactNode;
  detail?: ReactNode;
  bordered?: boolean;
  role?: "alert" | "status";
  className?: string;
}) {
  return (
    <div
      role={role}
      className={[
        "grid place-items-center px-6 py-16 text-center",
        bordered ? "rounded-xl border border-line bg-surface" : "",
        className,
      ]
        .filter(Boolean)
        .join(" ")}
    >
      <div className="max-w-sm">
        {Icon && (
          <div className="mx-auto grid h-10 w-10 place-items-center rounded-lg bg-line-subtle text-ink-muted">
            <Icon className="h-5 w-5" strokeWidth={1.7} aria-hidden="true" />
          </div>
        )}
        <p className="mt-4 text-sm font-semibold text-ink">{title}</p>
        {description && (
          <p className="mt-1.5 text-[13px] leading-6 text-ink-muted">{description}</p>
        )}
        {detail && <p className="mt-2 text-2xs leading-5 text-ink-faint">{detail}</p>}
        {action && <div className="mt-5 flex flex-wrap justify-center gap-2">{action}</div>}
      </div>
    </div>
  );
}
