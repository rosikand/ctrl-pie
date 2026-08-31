import type { ReactNode } from "react";
import { NavLink } from "react-router-dom";

export type TabItem<T extends string = string> = {
  id: T;
  label: string;
  count?: number | null;
};

const TAB =
  "relative -mb-px whitespace-nowrap border-b-2 px-1 pb-2.5 pt-1 text-[13px] font-medium transition";
const TAB_ACTIVE = "border-ink text-ink";
const TAB_IDLE = "border-transparent text-ink-muted hover:border-line-strong hover:text-ink";

/** In-page tabs. Used to hide advanced detail behind one click. */
export function Tabs<T extends string>({
  items,
  value,
  onChange,
  label,
  className = "",
}: {
  items: TabItem<T>[];
  value: T;
  onChange: (id: T) => void;
  label: string;
  className?: string;
}) {
  return (
    <div className={`border-b border-line ${className}`}>
      <div role="tablist" aria-label={label} className="flex gap-6 overflow-x-auto">
        {items.map((item) => {
          const active = item.id === value;
          return (
            <button
              key={item.id}
              type="button"
              role="tab"
              id={`tab-${item.id}`}
              aria-selected={active}
              aria-controls={`panel-${item.id}`}
              onClick={() => onChange(item.id)}
              className={`${TAB} ${active ? TAB_ACTIVE : TAB_IDLE}`}
            >
              {item.label}
              {item.count !== undefined && item.count !== null && (
                <span className="ml-1.5 text-2xs tabular-nums text-ink-faint">{item.count}</span>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}

export function TabPanel({
  id,
  active,
  children,
  keepMounted = false,
  className = "",
}: {
  id: string;
  active: boolean;
  children: ReactNode;
  /** Keeps unsaved form state alive while another tab is shown. */
  keepMounted?: boolean;
  className?: string;
}) {
  if (!active && !keepMounted) return null;
  return (
    <div
      role="tabpanel"
      id={`panel-${id}`}
      aria-labelledby={`tab-${id}`}
      hidden={!active}
      className={className}
    >
      {children}
    </div>
  );
}

/** Route-driven tabs, for sub-pages that keep their own URL. */
export function TabLinks({
  items,
  label,
  className = "",
}: {
  items: { to: string; label: string; end?: boolean }[];
  label: string;
  className?: string;
}) {
  return (
    <nav aria-label={label} className={`border-b border-line ${className}`}>
      <div className="flex gap-6 overflow-x-auto">
        {items.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            className={({ isActive }) => `${TAB} ${isActive ? TAB_ACTIVE : TAB_IDLE}`}
          >
            {item.label}
          </NavLink>
        ))}
      </div>
    </nav>
  );
}
