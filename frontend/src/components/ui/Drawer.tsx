import { X } from "lucide-react";
import { useEffect, useRef } from "react";
import type { ReactNode } from "react";
import { createPortal } from "react-dom";

import { IconButton } from "./Button";

/**
 * Right-side sheet for advanced detail that should not cost a page.
 * Closes on Escape or backdrop click and restores focus to the opener.
 */
export function Drawer({
  open,
  onClose,
  title,
  description,
  children,
  footer,
  width = "max-w-lg",
}: {
  open: boolean;
  onClose: () => void;
  title: ReactNode;
  description?: ReactNode;
  children: ReactNode;
  footer?: ReactNode;
  width?: string;
}) {
  const panelRef = useRef<HTMLDivElement | null>(null);
  const openerRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!open) return;
    openerRef.current = document.activeElement as HTMLElement | null;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKeyDown);
    const frame = window.requestAnimationFrame(() => panelRef.current?.focus());
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      window.cancelAnimationFrame(frame);
      document.body.style.overflow = previousOverflow;
      openerRef.current?.focus?.();
    };
  }, [onClose, open]);

  if (!open) return null;

  return createPortal(
    <div className="fixed inset-0 z-50 flex justify-end">
      <div
        aria-hidden="true"
        onClick={onClose}
        className="absolute inset-0 animate-fade-in bg-ink/25 backdrop-blur-[1px]"
      />
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-label={typeof title === "string" ? title : undefined}
        tabIndex={-1}
        className={`relative flex h-full w-full ${width} animate-slide-in-right flex-col border-l border-line bg-surface shadow-overlay outline-none`}
      >
        <div className="flex items-start justify-between gap-4 border-b border-line px-6 py-4">
          <div className="min-w-0">
            <h2 className="truncate text-sm font-semibold tracking-tight text-ink">{title}</h2>
            {description && (
              <p className="mt-1 text-xs leading-5 text-ink-muted">{description}</p>
            )}
          </div>
          <IconButton icon={X} label="Close" variant="ghost" size="sm" onClick={onClose} />
        </div>
        <div className="scroll-quiet flex-1 overflow-y-auto px-6 py-5">{children}</div>
        {footer && <div className="border-t border-line px-6 py-4">{footer}</div>}
      </div>
    </div>,
    document.body,
  );
}
