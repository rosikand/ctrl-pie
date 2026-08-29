import {
  AlertCircle,
  Database,
  LoaderCircle,
  Lock,
  RefreshCw,
  WifiOff,
} from "lucide-react";
import type { ReactNode } from "react";
import { Link } from "react-router-dom";

import type { TrainingLoadError } from "../types/training";

export const trainerCountFormatter = new Intl.NumberFormat();

export function formatTrainerTimestamp(value: string | null, includeTime = true): string {
  if (!value) return "Unknown";
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) return "Unknown";
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    year: "numeric",
    ...(includeTime ? { hour: "2-digit", minute: "2-digit" } : {}),
  });
}

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

export function TrainerErrorPanel({
  error,
  context,
  onRetry,
  loading,
}: {
  error: TrainingLoadError;
  context: "runs" | "models" | "run";
  onRetry: () => void;
  loading: boolean;
}) {
  let title = context === "models" ? "Models could not be loaded" : "Training runs could not be loaded";
  let description = "The backend could not be reached. Check the service and try again.";
  let Icon = WifiOff;
  if (error.status === 503) {
    title = context === "models" ? "Model discovery is not configured" : "Training storage is unavailable";
    description = context === "models"
      ? "Complete the server-side Hugging Face configuration to discover model repositories."
      : "Connect the configured database before browsing reported training runs.";
    Icon = Database;
  } else if (error.status === 403) {
    title = "Hugging Face access was denied";
    description = "The backend cannot access models in the configured namespace.";
    Icon = Lock;
  } else if (error.status === 502) {
    title = "Hugging Face is unavailable";
    description = "The backend could not enumerate model repositories from the Hub.";
  } else if (error.status === 404 && context === "run") {
    title = "Training run not found";
    description = "This run may have been removed since the list was loaded.";
  }
  const showSettings = context === "models" && (error.status === 403 || error.status === 503);

  return (
    <section className="grid min-h-[20rem] place-items-center rounded-xl border border-slate-200 bg-white px-6 text-center shadow-panel" role="alert">
      <div className="max-w-md">
        <div className="mx-auto grid h-11 w-11 place-items-center rounded-xl bg-slate-100 text-slate-500">
          <Icon className="h-5 w-5" aria-hidden="true" />
        </div>
        <h2 className="mt-4 text-sm font-semibold text-slate-900">{title}</h2>
        <p className="mt-2 text-sm leading-6 text-slate-500">{description}</p>
        <p className="mt-2 text-xs text-slate-400">{error.message}</p>
        <div className="mt-5 flex flex-wrap justify-center gap-2">
          <button
            type="button"
            onClick={onRetry}
            disabled={loading}
            className="inline-flex items-center gap-2 rounded-lg bg-slate-900 px-3.5 py-2 text-xs font-semibold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" aria-hidden="true" /> : <RefreshCw className="h-3.5 w-3.5" aria-hidden="true" />}
            Try again
          </button>
          {showSettings && (
            <Link to="/settings" className="rounded-lg border border-slate-200 bg-white px-3.5 py-2 text-xs font-semibold text-slate-700 transition hover:bg-slate-50">
              Open settings
            </Link>
          )}
        </div>
      </div>
    </section>
  );
}

export function TrainerInlineError({ error, onRetry, loading }: { error: TrainingLoadError; onRetry: () => void; loading: boolean }) {
  return (
    <div className="mb-5 flex flex-col gap-3 rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800 sm:flex-row sm:items-center sm:justify-between" role="alert">
      <div className="flex items-start gap-2">
        <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
        <p>{error.message}</p>
      </div>
      <button type="button" onClick={onRetry} disabled={loading} className="shrink-0 self-start rounded-md border border-rose-200 bg-white px-3 py-1.5 text-xs font-semibold transition hover:bg-rose-100 disabled:opacity-50 sm:self-auto">
        Retry
      </button>
    </div>
  );
}

export function TrainerViewHeader({
  title,
  description,
  count,
  refreshing,
  onRefresh,
}: {
  title: string;
  description: string;
  count: number | null;
  refreshing: boolean;
  onRefresh: () => void;
}) {
  return (
    <div className="mb-5 flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
      <div>
        <h2 className="text-sm font-semibold text-slate-900">{title}</h2>
        <p className="mt-1 text-xs leading-5 text-slate-400">{description}</p>
        {count !== null && <p className="mt-2 text-xs text-slate-500" aria-live="polite">{trainerCountFormatter.format(count)} total</p>}
      </div>
      <button
        type="button"
        onClick={onRefresh}
        disabled={refreshing}
        className="inline-flex w-fit items-center gap-2 rounded-lg border border-slate-200 bg-white px-3.5 py-2 text-xs font-semibold text-slate-700 shadow-sm transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
      >
        <RefreshCw className={`h-3.5 w-3.5 ${refreshing ? "animate-spin" : ""}`} aria-hidden="true" />
        {refreshing ? "Refreshing…" : "Refresh"}
      </button>
    </div>
  );
}

export function TrainerDetailValue({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="min-w-0 rounded-lg bg-slate-50 px-3 py-3">
      <dt className="text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-400">{label}</dt>
      <dd className="mt-1.5 truncate text-sm font-semibold text-slate-800" title={typeof children === "string" ? children : undefined}>{children}</dd>
    </div>
  );
}
