import { ChevronDown } from "lucide-react";
import type {
  InputHTMLAttributes,
  ReactNode,
  SelectHTMLAttributes,
  TextareaHTMLAttributes,
} from "react";
import { useId } from "react";

const CONTROL =
  "w-full rounded-lg border bg-surface px-3 text-[13px] text-ink outline-none transition placeholder:text-ink-faint disabled:cursor-not-allowed disabled:bg-canvas disabled:text-ink-muted";
const CONTROL_IDLE = "border-line focus:border-accent-500 focus:ring-4 focus:ring-accent-100";
const CONTROL_INVALID =
  "border-critical-500 focus:border-critical-500 focus:ring-4 focus:ring-critical-100";

export function Field({
  label,
  hint,
  error,
  optional = false,
  children,
  htmlFor,
  className = "",
}: {
  label: ReactNode;
  hint?: ReactNode;
  error?: ReactNode;
  optional?: boolean;
  children: ReactNode;
  htmlFor?: string;
  className?: string;
}) {
  return (
    <div className={`min-w-0 ${className}`}>
      <label htmlFor={htmlFor} className="block text-xs font-medium text-ink-secondary">
        {label}
        {optional && <span className="ml-1 font-normal text-ink-faint">optional</span>}
      </label>
      <div className="mt-1.5">{children}</div>
      {error ? (
        <p className="mt-1.5 text-2xs leading-4 text-critical-700">{error}</p>
      ) : (
        hint && <p className="mt-1.5 text-2xs leading-4 text-ink-muted">{hint}</p>
      )}
    </div>
  );
}

export function TextInput({
  invalid = false,
  className = "",
  ...rest
}: InputHTMLAttributes<HTMLInputElement> & { invalid?: boolean }) {
  return (
    <input
      aria-invalid={invalid || undefined}
      className={`${CONTROL} h-9 ${invalid ? CONTROL_INVALID : CONTROL_IDLE} ${className}`}
      {...rest}
    />
  );
}

export function TextArea({
  invalid = false,
  className = "",
  rows = 3,
  ...rest
}: TextareaHTMLAttributes<HTMLTextAreaElement> & { invalid?: boolean }) {
  return (
    <textarea
      rows={rows}
      aria-invalid={invalid || undefined}
      className={`${CONTROL} resize-none py-2 leading-5 ${invalid ? CONTROL_INVALID : CONTROL_IDLE} ${className}`}
      {...rest}
    />
  );
}

export function Select({
  className = "",
  children,
  ...rest
}: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <div className="relative">
      <select
        className={`${CONTROL} ${CONTROL_IDLE} h-9 appearance-none pr-9 ${className}`}
        {...rest}
      >
        {children}
      </select>
      <ChevronDown
        aria-hidden="true"
        className="pointer-events-none absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 text-ink-faint"
      />
    </div>
  );
}

export function Checkbox({
  label,
  description,
  tone = "neutral",
  className = "",
  ...rest
}: InputHTMLAttributes<HTMLInputElement> & {
  label: ReactNode;
  description?: ReactNode;
  tone?: "neutral" | "warning" | "danger";
}) {
  const id = useId();
  const frame =
    tone === "danger"
      ? "border-critical-100 bg-critical-50"
      : tone === "warning"
        ? "border-caution-100 bg-caution-50"
        : "border-line bg-surface";
  const text =
    tone === "danger" ? "text-critical-700" : tone === "warning" ? "text-caution-700" : "text-ink";
  return (
    <label
      htmlFor={id}
      className={`flex cursor-pointer items-start gap-2.5 rounded-lg border p-3 ${frame} ${className}`}
    >
      <input
        id={id}
        type="checkbox"
        className="mt-0.5 h-4 w-4 shrink-0 rounded border-line-strong text-accent-600 focus:ring-accent-500"
        {...rest}
      />
      <span className="min-w-0">
        <span className={`block text-xs font-medium leading-5 ${text}`}>{label}</span>
        {description && (
          <span className="mt-0.5 block text-2xs leading-4 text-ink-muted">{description}</span>
        )}
      </span>
    </label>
  );
}

/** Inline checkbox without the bordered frame, for dense control clusters. */
export function InlineCheckbox({
  label,
  className = "",
  ...rest
}: InputHTMLAttributes<HTMLInputElement> & { label: ReactNode }) {
  const id = useId();
  return (
    <label
      htmlFor={id}
      className={`inline-flex cursor-pointer items-center gap-2 text-xs font-medium text-ink-secondary ${className}`}
    >
      <input
        id={id}
        type="checkbox"
        className="h-4 w-4 rounded border-line-strong text-accent-600 focus:ring-accent-500"
        {...rest}
      />
      {label}
    </label>
  );
}

/** Large selectable option used by the inference deployment flow. */
export function ChoiceCard({
  selected,
  title,
  description,
  meta,
  disabled = false,
  onSelect,
  name,
}: {
  selected: boolean;
  title: ReactNode;
  description?: ReactNode;
  meta?: ReactNode;
  disabled?: boolean;
  onSelect: () => void;
  name: string;
}) {
  return (
    <label
      className={[
        "flex cursor-pointer items-start gap-3 rounded-lg border px-4 py-3 transition",
        selected ? "border-accent-500 bg-accent-50/50" : "border-line bg-surface hover:bg-canvas",
        disabled ? "cursor-not-allowed opacity-50" : "",
      ].join(" ")}
    >
      <input
        type="radio"
        name={name}
        checked={selected}
        disabled={disabled}
        onChange={onSelect}
        className="mt-0.5 h-4 w-4 shrink-0 border-line-strong text-accent-600 focus:ring-accent-500"
      />
      <span className="min-w-0 flex-1">
        <span className="flex flex-wrap items-center justify-between gap-2">
          <span className="text-[13px] font-medium text-ink">{title}</span>
          {meta && <span className="text-2xs text-ink-muted">{meta}</span>}
        </span>
        {description && (
          <span className="mt-1 line-clamp-2 block text-xs leading-5 text-ink-muted">{description}</span>
        )}
      </span>
    </label>
  );
}
