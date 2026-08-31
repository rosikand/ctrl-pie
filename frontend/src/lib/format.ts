const countFormatter = new Intl.NumberFormat();
const decimalFormatter = new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 });
const scalarFormatter = new Intl.NumberFormat(undefined, { maximumSignificantDigits: 6 });

/** Thousands-separated integers for tables, axes, and counters. */
export function formatCount(value: number | null | undefined): string {
  return value === null || value === undefined || !Number.isFinite(value)
    ? "—"
    : countFormatter.format(value);
}

/** One decimal place for rates and latencies. */
export function formatDecimal(value: number | null | undefined): string {
  return value === null || value === undefined || !Number.isFinite(value)
    ? "—"
    : decimalFormatter.format(value);
}

/** Significant-digit formatting for reported training scalars. */
export function formatScalar(value: number): string {
  return Number.isFinite(value) ? scalarFormatter.format(value) : "—";
}

export function formatDate(value: string | null | undefined): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) return "—";
  return date.toLocaleDateString([], { month: "short", day: "numeric", year: "numeric" });
}

export function formatDateTime(value: string | null | undefined): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) return "—";
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function formatTime(value: string | null | undefined): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) return "—";
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

/** Compact "3m ago" style age used in tables and the activity feed. */
export function formatRelative(value: string | null | undefined, now = Date.now()): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) return "—";
  const seconds = Math.round((now - date.valueOf()) / 1_000);
  if (seconds < 0) return "just now";
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  if (days < 30) return `${days}d ago`;
  return formatDate(value);
}

/** mm:ss for episode and session clocks. */
export function formatDuration(seconds: number): string {
  const whole = Number.isFinite(seconds) ? Math.max(0, Math.floor(seconds)) : 0;
  const minutes = Math.floor(whole / 60);
  const remainder = whole % 60;
  return `${minutes.toString().padStart(2, "0")}:${remainder.toString().padStart(2, "0")}`;
}

/** mm:ss.hh for frame-accurate dataset playback. */
export function formatPreciseDuration(seconds: number): string {
  const safe = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  const minutes = Math.floor(safe / 60);
  const remainder = (safe % 60).toFixed(2).padStart(5, "0");
  return `${minutes.toString().padStart(2, "0")}:${remainder}`;
}

export function degrees(radians: number, digits = 1): string {
  return `${((radians * 180) / Math.PI).toFixed(digits)}°`;
}

export function signed(value: number, digits = 2): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;
}

export function optionalSigned(value: number | null, digits: number, suffix: string): string {
  return value === null ? "—" : `${signed(value, digits)}${suffix}`;
}

export function optionalFixed(value: number | null, digits: number, suffix: string): string {
  return value === null ? "—" : `${value.toFixed(digits)}${suffix}`;
}

export function shortRevision(revision: string | null | undefined, length = 8): string {
  return revision ? revision.slice(0, length) : "—";
}

/** Uppercases only the first character, so "gravity comp" does not become "Gravity Comp". */
export function sentenceCase(value: string): string {
  return value ? value.charAt(0).toUpperCase() + value.slice(1) : value;
}

/** Turns `gravity_comp` into `gravity comp` for status text. */
export function humanize(value: string): string {
  return value.replaceAll("_", " ");
}

/** Appends validated path segments to a Hugging Face URL. */
export function appendHubPath(base: string, segments: string[]): string | null {
  const url = new URL(base);
  const pathSegments = segments.flatMap((segment) => segment.split("/"));
  if (pathSegments.some((segment) => !segment || segment === "." || segment === "..")) {
    return null;
  }
  const suffix = pathSegments.map(encodeURIComponent).join("/");
  url.pathname = `${url.pathname.replace(/\/$/, "")}/${suffix}`;
  return url.toString();
}

/** Only ever returns an https://huggingface.co URL. */
export function safeHubUrl(value: string | null | undefined): string | null {
  if (!value) return null;
  try {
    const url = new URL(value);
    return url.protocol === "https:" &&
      url.hostname === "huggingface.co" &&
      !url.port &&
      !url.username &&
      !url.password
      ? url.toString()
      : null;
  } catch {
    return null;
  }
}

/** Builds a Hub URL from a `namespace/name` repository id. */
export function hubRepoUrl(
  repoId: string | null,
  kind: "model" | "dataset" = "model",
): string | null {
  if (!repoId) return null;
  const parts = repoId.split("/");
  if (parts.length !== 2 || parts.some((part) => !/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(part))) {
    return null;
  }
  const path = parts.map(encodeURIComponent).join("/");
  return `https://huggingface.co/${kind === "dataset" ? "datasets/" : ""}${path}`;
}

/** Suggests a valid Hub repository name from free-form session text. */
export function suggestedRepoName(value: string, fallback = "robot-demonstrations"): string {
  const slug = value
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "-")
    .replace(/\.{2,}/g, ".")
    .replace(/-{2,}/g, "-")
    .replace(/^[-.]+|[-.]+$/g, "")
    .slice(0, 64);
  return slug || fallback;
}

/** Mirrors the backend's repository-name rules so uploads fail in the form, not the API. */
export function repoNameIssue(repoName: string): string | null {
  const value = repoName.trim();
  if (!value) return "Enter a dataset repository name.";
  if (value.length > 96) return "Repository names are limited to 96 characters.";
  if (value.includes("/")) return "Enter the repository name only, without a namespace.";
  if (
    !/^[A-Za-z0-9_][A-Za-z0-9._-]*[A-Za-z0-9_]$/.test(value) &&
    !/^[A-Za-z0-9_]$/.test(value)
  ) {
    return "Use letters, numbers, underscore, hyphen, or period; do not start or end with a hyphen or period.";
  }
  if (value.includes("--") || value.includes("..")) {
    return "Consecutive hyphens or periods are not allowed.";
  }
  return null;
}
