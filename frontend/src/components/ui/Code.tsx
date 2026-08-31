import { Check, Copy } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";

function useCopy(value: string) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<number | undefined>(undefined);

  useEffect(() => () => window.clearTimeout(timer.current), []);

  const copy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      window.clearTimeout(timer.current);
      timer.current = window.setTimeout(() => setCopied(false), 1_600);
    } catch {
      // Clipboard access can be denied; the text stays selectable either way.
    }
  }, [value]);

  return { copied, copy };
}

export function CopyButton({
  value,
  label = "Copy",
  className = "",
}: {
  value: string;
  label?: string;
  className?: string;
}) {
  const { copied, copy } = useCopy(value);
  return (
    <button
      type="button"
      onClick={() => void copy()}
      aria-label={copied ? "Copied" : label}
      className={`inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-2xs font-medium text-ink-muted transition hover:bg-line-subtle hover:text-ink ${className}`}
    >
      {copied ? (
        <Check className="h-3.5 w-3.5 text-positive-600" aria-hidden="true" />
      ) : (
        <Copy className="h-3.5 w-3.5" aria-hidden="true" />
      )}
      {copied ? "Copied" : label}
    </button>
  );
}

/** Terminal-style block with copy, used by the setup guide and credential help. */
export function CodeBlock({
  code,
  caption,
  className = "",
}: {
  code: string;
  caption?: ReactNode;
  className?: string;
}) {
  return (
    <figure className={`overflow-hidden rounded-lg border border-line bg-canvas ${className}`}>
      <figcaption className="flex items-center justify-between gap-3 border-b border-line px-3 py-1.5">
        <span className="truncate text-2xs font-medium text-ink-muted">{caption ?? "shell"}</span>
        <CopyButton value={code} />
      </figcaption>
      <pre className="scroll-quiet overflow-x-auto px-3 py-3 font-mono text-xs leading-6 text-ink-secondary">
        <code>{code}</code>
      </pre>
    </figure>
  );
}

export function InlineCode({ children }: { children: ReactNode }) {
  return (
    <code className="rounded bg-line-subtle px-1 py-0.5 font-mono text-[0.9em] text-ink-secondary">
      {children}
    </code>
  );
}
