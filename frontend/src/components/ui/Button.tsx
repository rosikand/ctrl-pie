import { LoaderCircle } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import type { ButtonHTMLAttributes, ReactNode } from "react";

export type ButtonVariant = "primary" | "secondary" | "ghost" | "danger" | "dangerSubtle";
export type ButtonSize = "sm" | "md" | "lg";

const VARIANTS: Record<ButtonVariant, string> = {
  primary: "bg-accent-600 text-white hover:bg-accent-700",
  secondary: "border border-line bg-surface text-ink hover:bg-canvas",
  ghost: "text-ink-secondary hover:bg-line-subtle hover:text-ink",
  danger: "bg-critical-600 text-white hover:bg-critical-700",
  dangerSubtle: "border border-critical-100 bg-surface text-critical-700 hover:bg-critical-50",
};

const SIZES: Record<ButtonSize, string> = {
  sm: "h-8 gap-1.5 rounded-md px-2.5 text-xs",
  md: "h-9 gap-2 rounded-lg px-3.5 text-[13px]",
  lg: "h-11 gap-2 rounded-lg px-5 text-sm",
};

/**
 * Shared button surface. Exported separately so links can look like buttons
 * without nesting an anchor inside a button.
 */
export function buttonClass(
  variant: ButtonVariant = "secondary",
  size: ButtonSize = "md",
  fullWidth = false,
): string {
  return [
    "inline-flex items-center justify-center whitespace-nowrap font-medium transition",
    "disabled:cursor-not-allowed disabled:opacity-40",
    VARIANTS[variant],
    SIZES[size],
    fullWidth ? "w-full" : "",
  ]
    .filter(Boolean)
    .join(" ");
}

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: ButtonVariant;
  size?: ButtonSize;
  icon?: LucideIcon;
  iconRight?: LucideIcon;
  loading?: boolean;
  fullWidth?: boolean;
  children?: ReactNode;
};

export function Button({
  variant = "secondary",
  size = "md",
  icon: Icon,
  iconRight: IconRight,
  loading = false,
  fullWidth = false,
  disabled,
  className = "",
  children,
  type = "button",
  ...rest
}: ButtonProps) {
  const glyph = size === "sm" ? "h-3.5 w-3.5" : "h-4 w-4";
  return (
    <button
      type={type}
      disabled={disabled || loading}
      className={`${buttonClass(variant, size, fullWidth)} ${className}`}
      {...rest}
    >
      {loading ? (
        <LoaderCircle className={`${glyph} animate-spin`} aria-hidden="true" />
      ) : (
        Icon && <Icon className={glyph} strokeWidth={1.9} aria-hidden="true" />
      )}
      {children}
      {IconRight && !loading && (
        <IconRight className={glyph} strokeWidth={1.9} aria-hidden="true" />
      )}
    </button>
  );
}

type IconButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  icon: LucideIcon;
  label: string;
  size?: Exclude<ButtonSize, "lg">;
  variant?: Extract<ButtonVariant, "secondary" | "ghost">;
  spinning?: boolean;
};

export function IconButton({
  icon: Icon,
  label,
  size = "md",
  variant = "secondary",
  spinning = false,
  className = "",
  type = "button",
  ...rest
}: IconButtonProps) {
  return (
    <button
      type={type}
      aria-label={label}
      title={label}
      className={[
        "inline-grid place-items-center rounded-lg transition disabled:cursor-not-allowed disabled:opacity-40",
        size === "sm" ? "h-8 w-8" : "h-9 w-9",
        variant === "secondary"
          ? "border border-line bg-surface text-ink-secondary hover:bg-canvas hover:text-ink"
          : "text-ink-muted hover:bg-line-subtle hover:text-ink",
        className,
      ].join(" ")}
      {...rest}
    >
      <Icon
        className={`${size === "sm" ? "h-3.5 w-3.5" : "h-4 w-4"} ${spinning ? "animate-spin" : ""}`}
        strokeWidth={1.9}
        aria-hidden="true"
      />
    </button>
  );
}
