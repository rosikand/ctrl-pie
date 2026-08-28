import {
  AlertCircle,
  Bot,
  CalendarDays,
  Database,
  ExternalLink,
  FileQuestion,
  Layers3,
  LoaderCircle,
  Lock,
  RefreshCw,
  ShieldAlert,
  Tag,
  Unlock,
  WifiOff,
} from "lucide-react";
import type { ReactNode } from "react";
import { Link } from "react-router-dom";

import { useDatasets } from "../hooks/useDatasets";
import type { DatasetLoadError } from "../hooks/useDatasets";
import type { DatasetSummary } from "../types/datasets";

const numberFormatter = new Intl.NumberFormat();

function formatCount(value: number | null): string {
  return value === null ? "—" : numberFormatter.format(value);
}

function formatFps(value: number | null): string {
  if (value === null) return "—";
  return `${numberFormatter.format(value)} Hz`;
}

function formatTimestamp(value: string | null, includeTime = false): string {
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

function externalHubUrl(value: string): string | null {
  try {
    const url = new URL(value);
    return url.protocol === "https:" ? url.toString() : null;
  } catch {
    return null;
  }
}

function MetadataValue({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="min-w-0 rounded-lg bg-slate-50 px-3 py-2.5">
      <dt className="text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-400">{label}</dt>
      <dd className="mt-1 truncate text-sm font-semibold text-slate-800" title={typeof children === "string" ? children : undefined}>
        {children}
      </dd>
    </div>
  );
}

function DegradedMetadata({ children }: { children: ReactNode }) {
  return (
    <div className="flex items-start gap-2 rounded-lg border border-amber-100 bg-amber-50/70 px-3 py-2 text-xs leading-5 text-amber-800">
      <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden="true" />
      <span>{children}</span>
    </div>
  );
}

function DatasetCard({ dataset }: { dataset: DatasetSummary }) {
  const card = dataset.card;
  const lerobot = dataset.lerobot;
  const title = card?.title?.trim() || dataset.name;
  const hubUrl = externalHubUrl(dataset.hub_url);
  const taskCategories = card?.task_categories ?? [];
  const visibleTags = dataset.tags.slice(0, 4);
  const hiddenTagCount = Math.max(0, dataset.tags.length - visibleTags.length);

  return (
    <article className="flex min-h-full flex-col rounded-xl border border-slate-200 bg-white shadow-panel">
      <div className="border-b border-slate-100 px-5 py-4">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <p className="truncate font-mono text-[11px] text-slate-400" title={dataset.repo_id}>{dataset.repo_id}</p>
            <h2 className="mt-1.5 text-base font-semibold leading-6 text-slate-950">{title}</h2>
          </div>
          <div className="flex shrink-0 flex-wrap justify-end gap-1.5">
            <span
              className={`inline-flex items-center gap-1 rounded-full px-2 py-1 text-[10px] font-semibold ${
                dataset.private ? "bg-slate-100 text-slate-700" : "bg-emerald-50 text-emerald-700"
              }`}
            >
              {dataset.private ? <Lock className="h-3 w-3" aria-hidden="true" /> : <Unlock className="h-3 w-3" aria-hidden="true" />}
              {dataset.private ? "Private" : "Public"}
            </span>
            {dataset.gated && (
              <span className="inline-flex items-center gap-1 rounded-full bg-amber-50 px-2 py-1 text-[10px] font-semibold text-amber-700">
                <ShieldAlert className="h-3 w-3" aria-hidden="true" />
                Gated
              </span>
            )}
          </div>
        </div>

        {card ? (
          <>
            <p className="mt-3 min-h-10 line-clamp-3 text-sm leading-5 text-slate-500">
              {card.description?.trim() || "No description was provided in the dataset card."}
            </p>
            <div className="mt-3 flex flex-wrap items-center gap-2 text-[11px] text-slate-500">
              <span className="rounded-md bg-slate-100 px-2 py-1 font-medium">
                License: {card.license || "not specified"}
              </span>
              {taskCategories.slice(0, 3).map((task, index) => (
                <span key={`${task}-${index}`} className="max-w-full truncate rounded-md bg-blue-50 px-2 py-1 font-medium text-blue-700" title={task}>
                  {task}
                </span>
              ))}
              {taskCategories.length > 3 && <span>+{taskCategories.length - 3} tasks</span>}
            </div>
          </>
        ) : (
          <div className="mt-3">
            <DegradedMetadata>Dataset card metadata is unavailable for this repository.</DegradedMetadata>
          </div>
        )}
      </div>

      <div className="flex flex-1 flex-col px-5 py-4">
        {lerobot ? (
          <dl className="grid grid-cols-2 gap-2 sm:grid-cols-3">
            <MetadataValue label="Episodes">{formatCount(lerobot.total_episodes)}</MetadataValue>
            <MetadataValue label="Frames">{formatCount(lerobot.total_frames)}</MetadataValue>
            <MetadataValue label="Tasks">{formatCount(lerobot.total_tasks)}</MetadataValue>
            <MetadataValue label="FPS">{formatFps(lerobot.fps)}</MetadataValue>
            <MetadataValue label="Robot">{lerobot.robot_type || "—"}</MetadataValue>
            <MetadataValue label="LeRobot">{lerobot.codebase_version || "—"}</MetadataValue>
          </dl>
        ) : (
          <DegradedMetadata>LeRobot metadata is missing or could not be read.</DegradedMetadata>
        )}

        {lerobot && (
          <div className="mt-3 flex items-center gap-2 text-xs text-slate-400">
            <Layers3 className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
            <span>{lerobot.features.length ? `${numberFormatter.format(lerobot.features.length)} recorded features` : "No feature schema reported"}</span>
          </div>
        )}

        {visibleTags.length > 0 && (
          <div className="mt-3 flex items-start gap-2">
            <Tag className="mt-1 h-3.5 w-3.5 shrink-0 text-slate-300" aria-hidden="true" />
            <div className="flex min-w-0 flex-wrap gap-1.5">
              {visibleTags.map((tag, index) => (
                <span key={`${tag}-${index}`} className="max-w-[12rem] truncate rounded bg-slate-50 px-1.5 py-1 text-[10px] font-medium text-slate-500" title={tag}>
                  {tag}
                </span>
              ))}
              {hiddenTagCount > 0 && <span className="px-1 py-1 text-[10px] text-slate-400">+{hiddenTagCount}</span>}
            </div>
          </div>
        )}

        <div className="mt-auto pt-4">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-slate-100 pt-3 text-[11px] text-slate-400">
            <span className="inline-flex items-center gap-1.5" title={dataset.created_at ?? undefined}>
              <CalendarDays className="h-3.5 w-3.5" aria-hidden="true" />
              Created {formatTimestamp(dataset.created_at)}
            </span>
            <span title={dataset.last_modified ?? undefined}>Updated {formatTimestamp(dataset.last_modified)}</span>
            {dataset.revision && <span className="font-mono" title={dataset.revision}>@{dataset.revision.slice(0, 8)}</span>}
          </div>
          {hubUrl ? (
            <a
              href={hubUrl}
              target="_blank"
              rel="noreferrer"
              className="mt-3 inline-flex items-center gap-1.5 text-xs font-semibold text-brand-600 transition hover:text-brand-700 focus-visible:rounded focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500 focus-visible:ring-offset-2"
            >
              Open on Hugging Face
              <ExternalLink className="h-3.5 w-3.5" aria-hidden="true" />
            </a>
          ) : (
            <p className="mt-3 text-xs text-slate-400">Hub link unavailable</p>
          )}
        </div>
      </div>
    </article>
  );
}

function DatasetSkeleton() {
  return (
    <div className="min-h-[25rem] animate-pulse rounded-xl border border-slate-200 bg-white p-5 shadow-panel" aria-hidden="true">
      <div className="h-3 w-36 rounded bg-slate-100" />
      <div className="mt-3 h-5 w-2/3 rounded bg-slate-100" />
      <div className="mt-5 h-3 w-full rounded bg-slate-100" />
      <div className="mt-2 h-3 w-5/6 rounded bg-slate-100" />
      <div className="mt-6 grid grid-cols-3 gap-2">
        {Array.from({ length: 6 }).map((_, index) => (
          <div key={index} className="h-14 rounded-lg bg-slate-50" />
        ))}
      </div>
      <div className="mt-6 h-3 w-1/2 rounded bg-slate-100" />
      <div className="mt-12 h-4 w-32 rounded bg-slate-100" />
    </div>
  );
}

function errorDetails(error: DatasetLoadError): { title: string; description: string; Icon: typeof AlertCircle; settings: boolean } {
  if (error.status === 503) {
    return {
      title: "Hugging Face is not configured",
      description: "Complete the server-side Hugging Face configuration before browsing this namespace.",
      Icon: Database,
      settings: true,
    };
  }
  if (error.status === 403) {
    return {
      title: "Hugging Face access was denied",
      description: "The backend could not access the configured namespace. Verify its credentials and namespace access.",
      Icon: Lock,
      settings: true,
    };
  }
  if (error.status === 502) {
    return {
      title: "The Hugging Face Hub is unavailable",
      description: "The backend could not enumerate datasets from the Hub. Your existing catalog has not been changed.",
      Icon: WifiOff,
      settings: false,
    };
  }
  if (error.status === 422) {
    return {
      title: "The dataset page could not be continued",
      description: "The pagination cursor was rejected. Refresh the catalog to start again from the first page.",
      Icon: AlertCircle,
      settings: false,
    };
  }
  return {
    title: "Datasets could not be loaded",
    description: "The app could not reach the dataset service. Check the backend connection and try again.",
    Icon: WifiOff,
    settings: false,
  };
}

function InitialError({ error, onRetry, busy }: { error: DatasetLoadError; onRetry: () => void; busy: boolean }) {
  const details = errorDetails(error);
  const Icon = details.Icon;
  return (
    <section className="mt-8 grid min-h-[22rem] place-items-center rounded-xl border border-slate-200 bg-white px-6 text-center shadow-panel" role="alert">
      <div className="max-w-md">
        <div className="mx-auto grid h-11 w-11 place-items-center rounded-xl bg-slate-100 text-slate-500">
          <Icon className="h-5 w-5" strokeWidth={1.8} aria-hidden="true" />
        </div>
        <h2 className="mt-4 text-sm font-semibold text-slate-900">{details.title}</h2>
        <p className="mt-2 text-sm leading-6 text-slate-500">{details.description}</p>
        <p className="mt-2 text-xs text-slate-400">{error.message}</p>
        <div className="mt-5 flex flex-wrap justify-center gap-2">
          <button
            type="button"
            onClick={onRetry}
            disabled={busy}
            className="inline-flex items-center gap-2 rounded-lg bg-slate-900 px-3.5 py-2 text-xs font-semibold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {busy ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" aria-hidden="true" /> : <RefreshCw className="h-3.5 w-3.5" aria-hidden="true" />}
            Try again
          </button>
          {details.settings && (
            <Link to="/settings" className="rounded-lg border border-slate-200 bg-white px-3.5 py-2 text-xs font-semibold text-slate-700 transition hover:bg-slate-50">
              Open settings
            </Link>
          )}
        </div>
      </div>
    </section>
  );
}

function InlineError({ error, onRetry, busy }: { error: DatasetLoadError; onRetry: () => void; busy: boolean }) {
  const details = errorDetails(error);
  return (
    <div className="mt-5 flex flex-col gap-3 rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800 sm:flex-row sm:items-center sm:justify-between" role="alert">
      <div className="flex items-start gap-2">
        <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
        <p><span className="font-semibold">{details.title}.</span> {error.message}</p>
      </div>
      <button type="button" onClick={onRetry} disabled={busy} className="shrink-0 self-start rounded-md border border-rose-200 bg-white px-3 py-1.5 text-xs font-semibold transition hover:bg-rose-100 disabled:opacity-50 sm:self-auto">
        {error.status === 422 ? "Refresh" : "Retry"}
      </button>
    </div>
  );
}

export function DatasetsPage() {
  const {
    datasets,
    namespace,
    total,
    nextCursor,
    fetchedAt,
    initialLoading,
    refreshing,
    loadingMore,
    error,
    busy,
    refresh,
    loadMore,
    retry,
  } = useDatasets();

  const hasDatasets = datasets.length > 0;

  return (
    <div className="mx-auto max-w-7xl px-5 py-8 sm:px-8 lg:px-10 lg:py-10">
      <header className="flex flex-col gap-5 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-brand-600">Hugging Face</p>
          <h1 className="mt-2 text-2xl font-semibold tracking-tight text-slate-950 sm:text-3xl">Datasets</h1>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500">
            Browse LeRobot datasets in the server-configured Hub namespace.
          </p>
          {(namespace || hasDatasets) && (
            <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-slate-400">
              {namespace && <span className="rounded-md bg-white px-2 py-1 font-mono ring-1 ring-slate-200">{namespace}</span>}
              <span aria-live="polite">Showing {numberFormatter.format(datasets.length)} of {numberFormatter.format(total)}</span>
              {fetchedAt && <span title={fetchedAt}>Synced {formatTimestamp(fetchedAt, true)}</span>}
            </div>
          )}
        </div>
        <button
          type="button"
          onClick={() => void refresh()}
          disabled={busy}
          aria-label={refreshing ? "Refreshing datasets" : "Refresh datasets from Hugging Face"}
          className="inline-flex w-fit items-center gap-2 rounded-lg border border-slate-200 bg-white px-3.5 py-2 text-xs font-semibold text-slate-700 shadow-sm transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${refreshing ? "animate-spin" : ""}`} aria-hidden="true" />
          {refreshing ? "Refreshing…" : "Refresh"}
        </button>
      </header>

      {initialLoading && !hasDatasets && (
        <section className="mt-8" aria-label="Loading datasets" aria-busy="true">
          <p className="sr-only" role="status">Loading datasets</p>
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {Array.from({ length: 6 }).map((_, index) => <DatasetSkeleton key={index} />)}
          </div>
        </section>
      )}

      {!initialLoading && error && !hasDatasets && (
        <InitialError error={error} onRetry={() => void retry()} busy={busy} />
      )}

      {!initialLoading && !error && !hasDatasets && (
        <section className="mt-8 grid min-h-[22rem] place-items-center rounded-xl border border-slate-200 bg-white px-6 text-center shadow-panel">
          <div className="max-w-md">
            <div className="mx-auto grid h-11 w-11 place-items-center rounded-xl bg-slate-100 text-slate-500">
              <FileQuestion className="h-5 w-5" strokeWidth={1.8} aria-hidden="true" />
            </div>
            <h2 className="mt-4 text-sm font-semibold text-slate-900">No LeRobot datasets found</h2>
            <p className="mt-2 text-sm leading-6 text-slate-500">
              The configured namespace has no datasets to show yet. Record and upload an episode to create one.
            </p>
            <Link to="/record" className="mt-5 inline-flex items-center gap-2 rounded-lg bg-slate-900 px-3.5 py-2 text-xs font-semibold text-white transition hover:bg-slate-800">
              <Bot className="h-3.5 w-3.5" aria-hidden="true" />
              Record an episode
            </Link>
          </div>
        </section>
      )}

      {hasDatasets && (
        <>
          {refreshing && (
            <div className="mt-6 flex items-center gap-2 rounded-lg border border-blue-100 bg-blue-50 px-4 py-2.5 text-xs font-medium text-blue-700" role="status">
              <LoaderCircle className="h-3.5 w-3.5 animate-spin" aria-hidden="true" />
              Refreshing from Hugging Face. Current datasets remain available.
            </div>
          )}
          {error && <InlineError error={error} onRetry={() => void retry()} busy={busy} />}
          <section className="mt-6" aria-label="LeRobot datasets" aria-busy={refreshing || loadingMore}>
            <div className="grid items-stretch gap-4 md:grid-cols-2 xl:grid-cols-3">
              {datasets.map((dataset) => <DatasetCard key={dataset.repo_id} dataset={dataset} />)}
            </div>
          </section>

          <div className="mt-7 flex flex-col items-center gap-2 border-t border-slate-200 pt-6">
            {nextCursor ? (
              <button
                type="button"
                onClick={() => void loadMore()}
                disabled={busy}
                className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-4 py-2.5 text-xs font-semibold text-slate-700 shadow-sm transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {loadingMore ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" aria-hidden="true" /> : <Database className="h-3.5 w-3.5" aria-hidden="true" />}
                {loadingMore ? "Loading more…" : "Load more"}
              </button>
            ) : (
              <p className="text-xs text-slate-400">All {numberFormatter.format(datasets.length)} datasets loaded</p>
            )}
          </div>
        </>
      )}
    </div>
  );
}
