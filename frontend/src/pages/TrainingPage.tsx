import {
  Activity,
  AlertCircle,
  BrainCircuit,
  CheckCircle2,
  CircleDashed,
  Code2,
  Database,
  ExternalLink,
  FileBox,
  GitBranch,
  Library,
  LoaderCircle,
  Lock,
  Package,
  RefreshCw,
  ShieldAlert,
  Tag,
  Unlock,
  WifiOff,
  XCircle,
} from "lucide-react";
import { useMemo, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";

import { useTrainerModels } from "../hooks/useTrainerModels";
import {
  useTrainingRuns,
  type TrainingLoadError,
} from "../hooks/useTrainingRuns";
import type {
  MetricPoint,
  TrainerModelSummary,
  TrainingRun,
  TrainingRunStatus,
} from "../types/training";

type TrainingView = "runs" | "models";

const countFormatter = new Intl.NumberFormat();
const scalarFormatter = new Intl.NumberFormat(undefined, {
  maximumSignificantDigits: 6,
});

const statusStyles: Record<TrainingRunStatus, string> = {
  created: "bg-slate-100 text-slate-600",
  running: "bg-blue-50 text-blue-700",
  completed: "bg-emerald-50 text-emerald-700",
  failed: "bg-rose-50 text-rose-700",
  cancelled: "bg-amber-50 text-amber-700",
};

function formatTimestamp(value: string | null, includeTime = true): string {
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

function safeHubUrl(value: string): string | null {
  try {
    const url = new URL(value);
    return url.protocol === "https:" && url.hostname === "huggingface.co"
      ? url.toString()
      : null;
  } catch {
    return null;
  }
}

function repoUrl(repoId: string | null, kind: "model" | "dataset" = "model"): string | null {
  if (!repoId) return null;
  const parts = repoId.split("/");
  if (
    parts.length !== 2 ||
    parts.some((part) => !/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(part))
  ) {
    return null;
  }
  const path = parts.map(encodeURIComponent).join("/");
  return `https://huggingface.co/${kind === "dataset" ? "datasets/" : ""}${path}`;
}

function appendHubPath(base: string, segments: string[]): string | null {
  const safe = safeHubUrl(base);
  if (!safe) return null;
  const url = new URL(safe);
  const pathSegments = segments.flatMap((segment) => segment.split("/"));
  if (pathSegments.some((segment) => !segment || segment === "." || segment === "..")) {
    return null;
  }
  const suffix = pathSegments
    .map(encodeURIComponent)
    .join("/");
  url.pathname = `${url.pathname.replace(/\/$/, "")}/${suffix}`;
  return url.toString();
}

function StatusBadge({ status }: { status: TrainingRunStatus }) {
  const Icon = status === "running"
    ? Activity
    : status === "completed"
      ? CheckCircle2
      : status === "failed" || status === "cancelled"
        ? XCircle
        : CircleDashed;
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[10px] font-semibold capitalize ${statusStyles[status]}`}>
      <Icon className={`h-3 w-3 ${status === "running" ? "animate-pulse" : ""}`} aria-hidden="true" />
      {status}
    </span>
  );
}

function ErrorPanel({
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

function InlineError({ error, onRetry, loading }: { error: TrainingLoadError; onRetry: () => void; loading: boolean }) {
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

function ViewHeader({
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
        {count !== null && <p className="mt-2 text-xs text-slate-500" aria-live="polite">{countFormatter.format(count)} total</p>}
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

function RunsLoading() {
  return (
    <div className="grid gap-5 lg:grid-cols-[19rem_minmax(0,1fr)]" aria-busy="true">
      <p className="sr-only" role="status">Loading training runs</p>
      <div className="h-96 animate-pulse rounded-xl border border-slate-200 bg-white p-4 shadow-panel">
        <div className="h-4 w-24 rounded bg-slate-100" />
        <div className="mt-5 space-y-3">{Array.from({ length: 4 }).map((_, index) => <div key={index} className="h-16 rounded-lg bg-slate-50" />)}</div>
      </div>
      <div className="h-[34rem] animate-pulse rounded-xl border border-slate-200 bg-white p-5 shadow-panel">
        <div className="h-6 w-52 rounded bg-slate-100" />
        <div className="mt-6 grid grid-cols-4 gap-3">{Array.from({ length: 4 }).map((_, index) => <div key={index} className="h-16 rounded-lg bg-slate-50" />)}</div>
        <div className="mt-7 h-48 rounded-lg bg-slate-50" />
      </div>
    </div>
  );
}

function RunButton({ run, selected, onSelect }: { run: TrainingRun; selected: boolean; onSelect: () => void }) {
  return (
    <button
      type="button"
      onClick={onSelect}
      aria-current={selected ? "true" : undefined}
      className={`w-full rounded-lg px-3 py-3 text-left transition ${selected ? "bg-brand-50 ring-1 ring-brand-100" : "hover:bg-slate-50"}`}
    >
      <span className="flex items-start justify-between gap-2">
        <span className={`min-w-0 truncate text-xs font-semibold ${selected ? "text-brand-800" : "text-slate-800"}`} title={run.name}>{run.name}</span>
        <StatusBadge status={run.status} />
      </span>
      <span className="mt-2 flex items-center justify-between gap-2 text-[10px] text-slate-400">
        <span>Step {countFormatter.format(run.current_step)}</span>
        <span>{formatTimestamp(run.updated_at, false)}</span>
      </span>
    </button>
  );
}

function RunList({ runs, selectedRunId, onSelect }: { runs: TrainingRun[]; selectedRunId: string; onSelect: (runId: string) => void }) {
  const active = runs.filter((run) => run.status === "created" || run.status === "running");
  const past = runs.filter((run) => run.status !== "created" && run.status !== "running");
  const groups = [
    { label: "Current", runs: active },
    { label: "Past", runs: past },
  ].filter((group) => group.runs.length > 0);
  return (
    <aside className="rounded-xl border border-slate-200 bg-white p-2 shadow-panel">
      <div className="max-h-[46rem] overflow-y-auto">
        {groups.map((group) => (
          <section key={group.label} className="mb-3 last:mb-0">
            <h3 className="px-3 pb-1 pt-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-400">{group.label}</h3>
            <div className="space-y-1">
              {group.runs.map((run) => <RunButton key={run.id} run={run} selected={run.id === selectedRunId} onSelect={() => onSelect(run.id)} />)}
            </div>
          </section>
        ))}
      </div>
    </aside>
  );
}

function DetailValue({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="min-w-0 rounded-lg bg-slate-50 px-3 py-3">
      <dt className="text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-400">{label}</dt>
      <dd className="mt-1.5 truncate text-sm font-semibold text-slate-800" title={typeof children === "string" ? children : undefined}>{children}</dd>
    </div>
  );
}

function MetricChart({ name, points }: { name: string; points: MetricPoint[] }) {
  const plotted = useMemo(
    () => points.filter((point) => Number.isFinite(point.step) && Number.isFinite(point.value)).sort((a, b) => a.step - b.step),
    [points],
  );
  if (plotted.length === 0) {
    return (
      <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-panel">
        <h4 className="font-mono text-xs font-semibold text-slate-700">{name}</h4>
        <p className="mt-8 text-center text-xs text-slate-400">No valid points</p>
      </div>
    );
  }
  const width = 360;
  const height = 130;
  const padX = 18;
  const padY = 16;
  const minStep = plotted[0].step;
  const maxStep = plotted.at(-1)?.step ?? minStep;
  const values = plotted.map((point) => point.value);
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const stepRange = maxStep - minStep;
  const valueRange = maxValue - minValue;
  const coordinates = plotted.map((point) => ({
    x: stepRange === 0 ? width / 2 : padX + ((point.step - minStep) / stepRange) * (width - padX * 2),
    y: valueRange === 0 ? height / 2 : padY + ((maxValue - point.value) / valueRange) * (height - padY * 2),
  }));
  const latest = plotted.at(-1)!;
  const lastCoordinate = coordinates.at(-1)!;

  return (
    <figure className="rounded-xl border border-slate-200 bg-white p-4 shadow-panel">
      <figcaption className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h4 className="truncate font-mono text-xs font-semibold text-slate-700" title={name}>{name}</h4>
          <p className="mt-1 text-[10px] text-slate-400">{countFormatter.format(plotted.length)} points</p>
        </div>
        <div className="text-right">
          <p className="font-mono text-sm font-semibold tabular-nums text-slate-900">{scalarFormatter.format(latest.value)}</p>
          <p className="mt-0.5 text-[10px] text-slate-400">step {countFormatter.format(latest.step)}</p>
        </div>
      </figcaption>
      <svg viewBox={`0 0 ${width} ${height}`} className="mt-3 h-32 w-full" role="img" aria-label={`${name} metric from step ${minStep} to ${maxStep}`}>
        <title>{name} metric curve</title>
        {[0.25, 0.5, 0.75].map((fraction) => (
          <line key={fraction} x1={padX} x2={width - padX} y1={height * fraction} y2={height * fraction} stroke="#e2e8f0" strokeWidth="1" />
        ))}
        {coordinates.length > 1 && (
          <polyline points={coordinates.map((point) => `${point.x},${point.y}`).join(" ")} fill="none" stroke="#2563eb" strokeWidth="2.25" strokeLinejoin="round" strokeLinecap="round" />
        )}
        <circle cx={lastCoordinate.x} cy={lastCoordinate.y} r="3.5" fill="#2563eb" stroke="white" strokeWidth="2" />
      </svg>
      <div className="flex items-center justify-between font-mono text-[9px] text-slate-400">
        <span>{scalarFormatter.format(minValue)} min</span>
        <span>steps {countFormatter.format(minStep)}–{countFormatter.format(maxStep)}</span>
        <span>{scalarFormatter.format(maxValue)} max</span>
      </div>
    </figure>
  );
}

function ArtifactLink({ repoId, kind = "model" }: { repoId: string | null; kind?: "model" | "dataset" }) {
  const url = repoUrl(repoId, kind);
  if (!repoId) return <span className="text-slate-400">Not reported</span>;
  return url ? (
    <a href={url} target="_blank" rel="noreferrer" className="inline-flex min-w-0 items-center gap-1.5 text-brand-600 hover:text-brand-700">
      <span className="truncate font-mono" title={repoId}>{repoId}</span>
      <ExternalLink className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
    </a>
  ) : <span className="break-all font-mono text-slate-600">{repoId}</span>;
}

function RunDetail({ run }: { run: TrainingRun }) {
  const metrics = Object.entries(run.metrics).sort(([left], [right]) => left.localeCompare(right));
  const configText = JSON.stringify(run.config, null, 2);
  const outputUrl = repoUrl(run.output_model_repo);
  const checkpointUrl = outputUrl && run.checkpoint_revision
    ? appendHubPath(outputUrl, ["tree", run.checkpoint_revision])
    : null;

  return (
    <div className="min-w-0 space-y-5">
      <section className="rounded-xl border border-slate-200 bg-white shadow-panel">
        <div className="flex flex-col gap-3 border-b border-slate-100 px-5 py-4 sm:flex-row sm:items-start sm:justify-between">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="text-lg font-semibold text-slate-950">{run.name}</h2>
              <StatusBadge status={run.status} />
            </div>
            <p className="mt-1 font-mono text-[10px] text-slate-400">{run.id}</p>
          </div>
          <p className="text-[11px] text-slate-400">Updated {formatTimestamp(run.updated_at)}</p>
        </div>
        <dl className="grid gap-2 p-5 sm:grid-cols-2 xl:grid-cols-4">
          <DetailValue label="Current step">{countFormatter.format(run.current_step)}</DetailValue>
          <DetailValue label="Runtime">{run.runtime || "Not reported"}</DetailValue>
          <DetailValue label="Framework">{run.framework || "Not reported"}</DetailValue>
          <DetailValue label="Created">{formatTimestamp(run.created_at, false)}</DetailValue>
        </dl>
        <dl className="grid gap-4 border-t border-slate-100 px-5 py-4 sm:grid-cols-2">
          <div>
            <dt className="text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-400">Dataset</dt>
            <dd className="mt-1.5 text-xs"><ArtifactLink repoId={run.dataset_repo} kind="dataset" /></dd>
          </div>
          <div>
            <dt className="text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-400">Base model</dt>
            <dd className="mt-1.5 text-xs"><ArtifactLink repoId={run.base_model} /></dd>
          </div>
        </dl>
      </section>

      <section>
        <div className="mb-3 flex items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-slate-900">Scalar metrics</h3>
            <p className="mt-1 text-xs text-slate-400">Values reported by the external training script.</p>
          </div>
          <span className="rounded-full bg-white px-2.5 py-1 text-[10px] font-semibold text-slate-500 ring-1 ring-slate-200">{countFormatter.format(metrics.length)} series</span>
        </div>
        {metrics.length ? (
          <div className="grid gap-4 xl:grid-cols-2">{metrics.map(([name, points]) => <MetricChart key={name} name={name} points={points} />)}</div>
        ) : (
          <div className="grid min-h-40 place-items-center rounded-xl border border-slate-200 bg-white px-6 text-center shadow-panel">
            <div><Activity className="mx-auto h-5 w-5 text-slate-300" aria-hidden="true" /><p className="mt-2 text-sm font-semibold text-slate-800">No metrics reported</p><p className="mt-1 text-xs text-slate-400">Curves appear when the trainer client logs scalar values.</p></div>
          </div>
        )}
      </section>

      <div className="grid gap-5 xl:grid-cols-2">
        <section className="rounded-xl border border-slate-200 bg-white shadow-panel">
          <div className="flex items-center gap-2 border-b border-slate-100 px-5 py-4">
            <FileBox className="h-4 w-4 text-slate-400" aria-hidden="true" />
            <h3 className="text-sm font-semibold text-slate-900">Output and checkpoints</h3>
          </div>
          <dl className="space-y-4 p-5 text-xs">
            <div><dt className="text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-400">Output model</dt><dd className="mt-1.5"><ArtifactLink repoId={run.output_model_repo} /></dd></div>
            <div><dt className="text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-400">Current revision</dt><dd className="mt-1.5">{run.checkpoint_revision ? (checkpointUrl ? <a href={checkpointUrl} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1.5 font-mono text-brand-600 hover:text-brand-700">{run.checkpoint_revision.slice(0, 12)}<ExternalLink className="h-3.5 w-3.5" aria-hidden="true" /></a> : <span className="font-mono">{run.checkpoint_revision}</span>) : <span className="text-slate-400">Not registered</span>}</dd></div>
          </dl>
          <div className="border-t border-slate-100 px-5 py-4">
            <p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-400">Registered checkpoints</p>
            {run.checkpoints.length ? (
              <ul className="mt-2 space-y-2">
                {run.checkpoints.map((checkpoint, index) => {
                  const base = repoUrl(checkpoint.repo_id);
                  const url = base ? appendHubPath(base, ["tree", checkpoint.revision]) : null;
                  return <li key={`${checkpoint.repo_id}-${checkpoint.revision}-${index}`} className="flex items-center justify-between gap-3 rounded-lg bg-slate-50 px-3 py-2 text-[11px]"><span className="min-w-0 truncate font-mono text-slate-600" title={`${checkpoint.repo_id}@${checkpoint.revision}`}>{checkpoint.repo_id}@{checkpoint.revision.slice(0, 8)}</span><span className="flex shrink-0 items-center gap-2 text-slate-400">step {countFormatter.format(checkpoint.step)}{url && <a href={url} target="_blank" rel="noreferrer" aria-label={`Open checkpoint at step ${checkpoint.step}`} className="text-brand-600 hover:text-brand-700"><ExternalLink className="h-3.5 w-3.5" /></a>}</span></li>;
                })}
              </ul>
            ) : <p className="mt-3 text-xs text-slate-400">No checkpoint revisions registered.</p>}
          </div>
        </section>

        <section className="rounded-xl border border-slate-200 bg-white shadow-panel">
          <div className="flex items-center gap-2 border-b border-slate-100 px-5 py-4">
            <Code2 className="h-4 w-4 text-slate-400" aria-hidden="true" />
            <h3 className="text-sm font-semibold text-slate-900">Configuration</h3>
          </div>
          {Object.keys(run.config).length ? (
            <pre className="max-h-80 overflow-auto whitespace-pre-wrap break-words p-5 font-mono text-[11px] leading-5 text-slate-600">{configText}</pre>
          ) : <p className="px-5 py-10 text-center text-xs text-slate-400">No configuration was reported.</p>}
        </section>
      </div>
    </div>
  );
}

function RunDetailLoading() {
  return <div className="h-[34rem] animate-pulse rounded-xl border border-slate-200 bg-white p-5 shadow-panel" aria-busy="true"><p className="sr-only" role="status">Loading run detail</p><div className="h-6 w-52 rounded bg-slate-100" /><div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">{Array.from({ length: 4 }).map((_, index) => <div key={index} className="h-16 rounded-lg bg-slate-50" />)}</div><div className="mt-7 h-52 rounded-lg bg-slate-50" /></div>;
}

function RunsView() {
  const { runs, selectedRunId, detail, initialLoading, refreshing, detailLoading, listError, detailError, selectRun, refresh, retryList, retryDetail } = useTrainingRuns();
  const detailIsCurrent = Boolean(detail && detail.id === selectedRunId);
  const retrySelectedRun = detailError?.status === 404 || detailError?.status === 409
    ? refresh
    : retryDetail;
  return (
    <div>
      <ViewHeader title="Reported runs" description="Runs are created and updated by external trainer clients; ctrl-π does not launch training." count={initialLoading ? null : runs.length} refreshing={refreshing || initialLoading} onRefresh={() => void refresh()} />
      {initialLoading && <RunsLoading />}
      {!initialLoading && listError && runs.length === 0 && <ErrorPanel error={listError} context="runs" onRetry={() => void retryList()} loading={initialLoading} />}
      {!initialLoading && !listError && runs.length === 0 && (
        <section className="grid min-h-[20rem] place-items-center rounded-xl border border-slate-200 bg-white px-6 text-center shadow-panel"><div className="max-w-md"><BrainCircuit className="mx-auto h-7 w-7 text-slate-300" aria-hidden="true" /><h2 className="mt-3 text-sm font-semibold text-slate-900">No training runs reported</h2><p className="mt-2 text-sm leading-6 text-slate-500">External scripts will appear here after they create a run through the Trainer API.</p></div></section>
      )}
      {!initialLoading && runs.length > 0 && selectedRunId && (
        <>
          {listError && <InlineError error={listError} onRetry={() => void refresh()} loading={refreshing} />}
          <div className="grid items-start gap-5 lg:grid-cols-[19rem_minmax(0,1fr)]">
            <RunList runs={runs} selectedRunId={selectedRunId} onSelect={selectRun} />
            <div className="min-w-0">
              {(detailLoading || (!detailError && !detailIsCurrent)) && <RunDetailLoading />}
              {!detailLoading && detailError && <ErrorPanel error={detailError} context="run" onRetry={() => void retrySelectedRun()} loading={detailLoading || refreshing} />}
              {!detailLoading && !detailError && detail && detailIsCurrent && <RunDetail run={detail} />}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function ModelsLoading() {
  return <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3" aria-busy="true"><p className="sr-only" role="status">Loading models</p>{Array.from({ length: 6 }).map((_, index) => <div key={index} className="h-[29rem] animate-pulse rounded-xl border border-slate-200 bg-white p-5 shadow-panel"><div className="h-3 w-40 rounded bg-slate-100" /><div className="mt-3 h-5 w-2/3 rounded bg-slate-100" /><div className="mt-5 h-12 rounded bg-slate-50" /><div className="mt-6 h-20 rounded bg-slate-50" /><div className="mt-6 h-24 rounded bg-slate-50" /></div>)}</div>;
}

function ModelCard({ model }: { model: TrainerModelSummary }) {
  const hubUrl = safeHubUrl(model.hub_url);
  const revisionUrl = hubUrl && model.revision ? appendHubPath(hubUrl, ["tree", model.revision]) : null;
  const visibleTags = model.tags.slice(0, 4);
  return (
    <article className="flex min-h-full flex-col rounded-xl border border-slate-200 bg-white shadow-panel">
      <div className="border-b border-slate-100 px-5 py-4">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0"><p className="truncate font-mono text-[11px] text-slate-400" title={model.repo_id}>{model.repo_id}</p><h3 className="mt-1.5 text-base font-semibold text-slate-950">{model.name}</h3></div>
          <div className="flex shrink-0 flex-wrap justify-end gap-1.5">
            <span className={`inline-flex items-center gap-1 rounded-full px-2 py-1 text-[10px] font-semibold ${model.private ? "bg-slate-100 text-slate-700" : "bg-emerald-50 text-emerald-700"}`}>{model.private ? <Lock className="h-3 w-3" aria-hidden="true" /> : <Unlock className="h-3 w-3" aria-hidden="true" />}{model.private ? "Private" : "Public"}</span>
            {model.gated && <span className="inline-flex items-center gap-1 rounded-full bg-amber-50 px-2 py-1 text-[10px] font-semibold text-amber-700"><ShieldAlert className="h-3 w-3" aria-hidden="true" />Gated</span>}
          </div>
        </div>
        {model.card ? <p className="mt-3 min-h-10 line-clamp-3 text-sm leading-5 text-slate-500">{model.card.description?.trim() || "No description was provided in the model card."}</p> : <div className="mt-3 flex items-start gap-2 rounded-lg border border-amber-100 bg-amber-50 px-3 py-2 text-xs text-amber-800"><AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden="true" />Model card metadata is unavailable.</div>}
      </div>
      <div className="flex flex-1 flex-col px-5 py-4">
        <dl className="grid grid-cols-2 gap-2">
          <DetailValue label="Pipeline">{model.pipeline_tag || "—"}</DetailValue>
          <DetailValue label="Library">{model.library_name || "—"}</DetailValue>
        </dl>
        {model.card && (
          <div className="mt-4 space-y-3 text-xs">
            <div><p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-400">Base model</p><div className="mt-1.5 flex flex-wrap gap-1.5">{model.card.base_model.length ? model.card.base_model.slice(0, 3).map((repo, index) => <span key={`${repo}-${index}`} className="max-w-full truncate rounded bg-blue-50 px-2 py-1 font-mono text-[10px] text-blue-700" title={repo}>{repo}</span>) : <span className="text-slate-400">Not specified</span>}</div></div>
            <div><p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-400">Datasets</p><div className="mt-1.5 flex flex-wrap gap-1.5">{model.card.datasets.length ? model.card.datasets.slice(0, 3).map((repo, index) => <span key={`${repo}-${index}`} className="max-w-full truncate rounded bg-violet-50 px-2 py-1 font-mono text-[10px] text-violet-700" title={repo}>{repo}</span>) : <span className="text-slate-400">Not specified</span>}</div></div>
          </div>
        )}
        <div className="mt-4 rounded-lg bg-slate-50 px-3 py-3">
          <div className="flex items-center justify-between gap-2"><p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-400">Checkpoint files</p><span className="text-[10px] text-slate-400">{countFormatter.format(model.checkpoints.length)}</span></div>
          {model.checkpoints.length ? <ul className="mt-2 space-y-1.5">{model.checkpoints.slice(0, 5).map((checkpoint) => { const url = hubUrl && model.revision ? appendHubPath(hubUrl, ["blob", model.revision, checkpoint]) : null; return <li key={checkpoint} className="flex min-w-0 items-center gap-1.5 font-mono text-[10px] text-slate-600"><GitBranch className="h-3 w-3 shrink-0 text-slate-300" aria-hidden="true" />{url ? <a href={url} target="_blank" rel="noreferrer" className="truncate hover:text-brand-600" title={checkpoint}>{checkpoint}</a> : <span className="truncate" title={checkpoint}>{checkpoint}</span>}</li>; })}{model.checkpoints.length > 5 && <li className="text-[10px] text-slate-400">+{model.checkpoints.length - 5} more files</li>}</ul> : <p className="mt-2 text-[11px] text-slate-400">No checkpoint files discovered.</p>}
        </div>
        {visibleTags.length > 0 && <div className="mt-3 flex items-start gap-2"><Tag className="mt-1 h-3.5 w-3.5 shrink-0 text-slate-300" aria-hidden="true" /><div className="flex flex-wrap gap-1">{visibleTags.map((tag, index) => <span key={`${tag}-${index}`} className="max-w-[11rem] truncate rounded px-1.5 py-1 text-[10px] text-slate-500 ring-1 ring-slate-100" title={tag}>{tag}</span>)}{model.tags.length > visibleTags.length && <span className="px-1.5 py-1 text-[10px] text-slate-400">+{model.tags.length - visibleTags.length}</span>}</div></div>}
        <div className="mt-auto pt-4">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-slate-100 pt-3 text-[10px] text-slate-400"><span>Updated {formatTimestamp(model.last_modified)}</span>{model.revision ? (revisionUrl ? <a href={revisionUrl} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 font-mono text-brand-600 hover:text-brand-700" title={model.revision}>@{model.revision.slice(0, 8)}<ExternalLink className="h-3 w-3" aria-hidden="true" /></a> : <span className="font-mono">@{model.revision.slice(0, 8)}</span>) : <span className="text-amber-600">Revision unavailable</span>}</div>
          {hubUrl && <a href={hubUrl} target="_blank" rel="noreferrer" className="mt-3 inline-flex items-center gap-1.5 text-xs font-semibold text-brand-600 hover:text-brand-700">Open model on Hugging Face<ExternalLink className="h-3.5 w-3.5" aria-hidden="true" /></a>}
        </div>
      </div>
    </article>
  );
}

function ModelsView() {
  const { data, initialLoading, refreshing, error, refresh, retry } = useTrainerModels();
  const models = data?.models ?? [];
  return (
    <div>
      <ViewHeader title="Namespace models" description={data ? `Hugging Face models discovered under ${data.namespace}.` : "Model artifacts discovered through the backend's configured Hugging Face namespace."} count={data?.total ?? (initialLoading ? null : models.length)} refreshing={refreshing || initialLoading} onRefresh={() => void refresh()} />
      {data && <div className="mb-5 flex flex-wrap items-center gap-2 text-xs text-slate-400"><span className="rounded-md bg-white px-2 py-1 font-mono ring-1 ring-slate-200">{data.namespace}</span><span title={data.fetched_at}>Synced {formatTimestamp(data.fetched_at)}</span></div>}
      {initialLoading && <ModelsLoading />}
      {!initialLoading && error && !data && <ErrorPanel error={error} context="models" onRetry={() => void retry()} loading={refreshing || initialLoading} />}
      {!initialLoading && data && error && <InlineError error={error} onRetry={() => void retry()} loading={refreshing} />}
      {!initialLoading && data && !error && models.length === 0 && <section className="grid min-h-[20rem] place-items-center rounded-xl border border-slate-200 bg-white px-6 text-center shadow-panel"><div className="max-w-md"><Package className="mx-auto h-7 w-7 text-slate-300" aria-hidden="true" /><h2 className="mt-3 text-sm font-semibold text-slate-900">No model repositories found</h2><p className="mt-2 text-sm leading-6 text-slate-500">The configured namespace has no models to show yet.</p></div></section>}
      {!initialLoading && data && models.length > 0 && <div className="grid items-stretch gap-4 md:grid-cols-2 xl:grid-cols-3" aria-busy={refreshing}>{models.map((model) => <ModelCard key={model.repo_id} model={model} />)}</div>}
    </div>
  );
}

export function TrainingPage() {
  const [view, setView] = useState<TrainingView>("runs");
  return (
    <div className="mx-auto max-w-7xl px-5 py-8 sm:px-8 lg:px-10 lg:py-10">
      <header>
        <p className="text-xs font-semibold uppercase tracking-[0.16em] text-brand-600">External training</p>
        <h1 className="mt-2 text-2xl font-semibold tracking-tight text-slate-950 sm:text-3xl">Training</h1>
        <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500">Monitor runs reported through the Trainer API and inspect model artifacts already stored on Hugging Face.</p>
        <div className="mt-6 inline-flex rounded-lg border border-slate-200 bg-slate-100/70 p-1" role="tablist" aria-label="Training views">
          <button id="training-runs-tab" type="button" role="tab" aria-selected={view === "runs"} aria-controls="training-runs-panel" onClick={() => setView("runs")} className={`inline-flex items-center gap-2 rounded-md px-4 py-2 text-xs font-semibold transition ${view === "runs" ? "bg-white text-slate-900 shadow-sm" : "text-slate-500 hover:text-slate-800"}`}><Activity className="h-3.5 w-3.5" aria-hidden="true" />Runs</button>
          <button id="training-models-tab" type="button" role="tab" aria-selected={view === "models"} aria-controls="training-models-panel" onClick={() => setView("models")} className={`inline-flex items-center gap-2 rounded-md px-4 py-2 text-xs font-semibold transition ${view === "models" ? "bg-white text-slate-900 shadow-sm" : "text-slate-500 hover:text-slate-800"}`}><Library className="h-3.5 w-3.5" aria-hidden="true" />Models</button>
        </div>
      </header>
      <section id={`training-${view}-panel`} role="tabpanel" aria-labelledby={`training-${view}-tab`} className="mt-7">
        {view === "runs" ? <RunsView /> : <ModelsView />}
      </section>
    </div>
  );
}
