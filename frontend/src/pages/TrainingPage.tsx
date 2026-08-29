import {
  Activity,
  Boxes,
  BrainCircuit,
  CheckCircle2,
  CircleDashed,
  Code2,
  ExternalLink,
  FileBox,
  Terminal,
  XCircle,
} from "lucide-react";
import { useMemo } from "react";

import {
  appendHubPath,
  formatTrainerTimestamp as formatTimestamp,
  trainerCountFormatter as countFormatter,
  TrainerDetailValue as DetailValue,
  TrainerErrorPanel as ErrorPanel,
  TrainerInlineError as InlineError,
  TrainerViewHeader as ViewHeader,
} from "../components/TrainerView";
import { useTrainingRuns } from "../hooks/useTrainingRuns";
import type {
  ManagedTrainingJobSummary,
  MetricPoint,
  TrainingConsoleLog,
  TrainingLoadError,
  TrainingRun,
  TrainingRunStatus,
} from "../types/training";

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

const managedStatusStyles: Record<ManagedTrainingJobSummary["status"], string> = {
  created: "bg-slate-100 text-slate-600",
  launching: "bg-blue-50 text-blue-700",
  running: "bg-blue-50 text-blue-700",
  finalizing: "bg-violet-50 text-violet-700",
  cancelling: "bg-amber-50 text-amber-700",
  completed: "bg-emerald-50 text-emerald-700",
  failed: "bg-rose-50 text-rose-700",
  cancelled: "bg-amber-50 text-amber-700",
};

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
        <span>{run.managed_job ? "Managed · " : ""}Step {countFormatter.format(run.current_step)}</span>
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

function ConsoleOutput({
  logs,
  loading,
  truncated,
  error,
  onRetry,
}: {
  logs: TrainingConsoleLog[];
  loading: boolean;
  truncated: boolean;
  error: TrainingLoadError | null;
  onRetry: () => void;
}) {
  const visibleLogs = logs.slice().reverse();
  return (
    <section className="overflow-hidden rounded-xl border border-slate-800 bg-slate-950 shadow-panel">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 px-5 py-3.5">
        <div className="flex items-center gap-2">
          <Terminal className="h-4 w-4 text-slate-400" aria-hidden="true" />
          <div>
            <h3 className="text-sm font-semibold text-slate-100">Trainer console</h3>
            <p className="mt-0.5 text-[10px] text-slate-500">Newest reported line first</p>
          </div>
        </div>
        <span className="font-mono text-[10px] text-slate-500">
          {countFormatter.format(logs.length)} retained in this view
        </span>
      </div>
      {truncated && (
        <p className="border-b border-amber-900/50 bg-amber-950/40 px-5 py-2 text-[10px] text-amber-300">
          Older output is unavailable because the bounded server or browser tail was trimmed.
        </p>
      )}
      {error && (
        <div className="flex items-center justify-between gap-3 border-b border-rose-900/50 bg-rose-950/40 px-5 py-2 text-[11px] text-rose-300" role="alert">
          <span>{error.message}</span>
          <button type="button" onClick={onRetry} className="shrink-0 font-semibold text-rose-200 underline underline-offset-2">Retry</button>
        </div>
      )}
      {loading && logs.length === 0 ? (
        <p className="px-5 py-12 text-center text-xs text-slate-500" role="status">Loading trainer output…</p>
      ) : visibleLogs.length > 0 ? (
        <ol className="max-h-80 overflow-y-auto px-5 py-3 font-mono text-[11px] leading-5" aria-live="polite" aria-relevant="additions text">
          {visibleLogs.map((log) => (
            <li key={log.sequence} className="grid gap-x-3 border-b border-slate-900 py-1.5 last:border-0 sm:grid-cols-[5.5rem_4.5rem_minmax(0,1fr)]">
              <span className="tabular-nums text-slate-600" title={formatTimestamp(log.timestamp)}>#{log.sequence} · {new Date(log.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}</span>
              <span className={log.source === "stderr" ? "text-amber-400" : log.source === "system" ? "text-violet-400" : "text-emerald-400"}>{log.source}{log.step === null ? "" : ` · ${countFormatter.format(log.step)}`}</span>
              <span className="whitespace-pre-wrap break-words text-slate-300">{log.line}</span>
            </li>
          ))}
        </ol>
      ) : (
        <p className="px-5 py-12 text-center text-xs text-slate-500">No trainer output is available yet.</p>
      )}
    </section>
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

function ManagedJobCard({ job }: { job: ManagedTrainingJobSummary }) {
  const outputUrl = repoUrl(job.output_model_repo);
  const verifiedRevision = job.output_revision ?? job.output_marker_revision;
  const revisionUrl = outputUrl && verifiedRevision
    ? appendHubPath(outputUrl, ["tree", verifiedRevision])
    : null;
  const terminal = job.status === "completed" || job.status === "failed" || job.status === "cancelled";

  return (
    <section className="rounded-xl border border-violet-200 bg-violet-50/40 shadow-panel">
      <div className="flex flex-col gap-3 border-b border-violet-100 px-5 py-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex items-start gap-3">
          <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-violet-100 text-violet-700">
            <Boxes className="h-4 w-4" aria-hidden="true" />
          </span>
          <div>
            <h3 className="text-sm font-semibold text-slate-900">
              {job.target_kind === "modal" ? "Managed Modal training" : "Managed training simulation"}
            </h3>
            <p className="mt-1 font-mono text-[10px] text-slate-400">{job.id}</p>
          </div>
        </div>
        <span className={`inline-flex w-fit rounded-full px-2.5 py-1 text-[10px] font-semibold capitalize ${managedStatusStyles[job.status]}`}>
          {job.status}
        </span>
      </div>
      <dl className="grid gap-2 p-5 sm:grid-cols-2 xl:grid-cols-4">
        <DetailValue label="Compute">{job.target_kind === "modal" ? job.compute_size : "Stub · no GPU"}</DetailValue>
        <DetailValue label="Provider">{job.provider_state}</DetailValue>
        <DetailValue label="Outcome">{job.outcome}</DetailValue>
        <DetailValue label="Hard deadline">{formatTimestamp(job.deadline_at)}</DetailValue>
      </dl>
      <div className="grid gap-4 border-t border-violet-100 px-5 py-4 text-xs sm:grid-cols-2">
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-400">Output artifact</p>
          <p className="mt-1.5 min-w-0">
            {job.target_kind === "modal" && revisionUrl ? (
              <a href={revisionUrl} target="_blank" rel="noreferrer" className="inline-flex max-w-full items-center gap-1.5 text-brand-600 hover:text-brand-700">
                <span className="truncate font-mono" title={job.output_model_repo}>{job.output_model_repo}</span>
                <span className="shrink-0 font-mono text-[10px] text-slate-400">@{verifiedRevision?.slice(0, 12)}</span>
                <ExternalLink className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
              </a>
            ) : job.target_kind === "modal" ? (
              <span><span className="break-all font-mono text-slate-600">{job.output_model_repo}</span><span className="mt-1 block text-[10px] text-slate-400">Requested repo · existence not yet verified</span></span>
            ) : (
              <span className="text-slate-500">Simulated · no Hub artifact created</span>
            )}
          </p>
        </div>
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-400">Compute teardown</p>
          <p className={`mt-1.5 font-semibold ${job.teardown_verified ? "text-emerald-700" : terminal ? "text-rose-700" : "text-slate-600"}`}>
            {job.teardown_verified
              ? job.target_kind === "stub"
                ? "Simulation teardown complete · no provider tasks"
                : "Provider-verified stopped · zero tasks"
              : terminal
                ? "Cleanup is not verified"
                : "Cleanup verification pending"}
          </p>
        </div>
      </div>
      {job.event_gap && (
        <p className="border-t border-amber-200 bg-amber-50 px-5 py-3 text-xs text-amber-800" role="status">
          Part of the bounded provider event stream was unavailable after reconnect. Stored metrics, checkpoints, and output remain visible, but this history is incomplete.
        </p>
      )}
      {job.last_error && (
        <p className="border-t border-rose-200 bg-rose-50 px-5 py-3 text-xs text-rose-800" role="alert">
          {job.last_error}
        </p>
      )}
    </section>
  );
}

function RunDetail({
  run,
  consoleLogs,
  consoleLoading,
  consoleTruncated,
  consoleError,
  liveError,
  lastLiveUpdate,
  onRetryConsole,
}: {
  run: TrainingRun;
  consoleLogs: TrainingConsoleLog[];
  consoleLoading: boolean;
  consoleTruncated: boolean;
  consoleError: TrainingLoadError | null;
  liveError: TrainingLoadError | null;
  lastLiveUpdate: string | null;
  onRetryConsole: () => void;
}) {
  const metrics = Object.entries(run.metrics).sort(([left], [right]) => left.localeCompare(right));
  const configText = JSON.stringify(run.config, null, 2);
  const outputUrl = repoUrl(run.output_model_repo);
  const managedArtifactRevision = run.managed_job?.output_revision ?? run.managed_job?.output_marker_revision;
  const managedArtifactUrl = outputUrl && managedArtifactRevision
    ? appendHubPath(outputUrl, ["tree", managedArtifactRevision])
    : null;
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
          <div className="text-right text-[11px] text-slate-400">
            <p>Updated {formatTimestamp(run.updated_at)}</p>
            <p className="mt-1 inline-flex items-center gap-1.5 text-emerald-600">
              <span className={`h-1.5 w-1.5 rounded-full bg-emerald-500 ${run.status === "running" ? "animate-pulse" : ""}`} />
              Auto-refresh · 2s while visible{lastLiveUpdate ? ` · checked ${formatTimestamp(lastLiveUpdate)}` : ""}
            </p>
          </div>
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

      {liveError && (
        <p className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-xs text-amber-800" role="status">
          Live metric refresh is temporarily unavailable; the last successful values remain visible.
        </p>
      )}

      {run.managed_job && <ManagedJobCard job={run.managed_job} />}

      <section>
        <div className="mb-3 flex items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-slate-900">Scalar metrics</h3>
            <p className="mt-1 text-xs text-slate-400">Values reported by a managed worker or external training script.</p>
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

      <ConsoleOutput
        logs={consoleLogs}
        loading={consoleLoading}
        truncated={consoleTruncated}
        error={consoleError}
        onRetry={onRetryConsole}
      />

      <div className="grid gap-5 xl:grid-cols-2">
        <section className="rounded-xl border border-slate-200 bg-white shadow-panel">
          <div className="flex items-center gap-2 border-b border-slate-100 px-5 py-4">
            <FileBox className="h-4 w-4 text-slate-400" aria-hidden="true" />
            <h3 className="text-sm font-semibold text-slate-900">Output and checkpoints</h3>
          </div>
          <dl className="space-y-4 p-5 text-xs">
            <div>
              <dt className="text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-400">Output model</dt>
              <dd className="mt-1.5">
                {run.managed_job ? (
                  managedArtifactUrl ? (
                    <a href={managedArtifactUrl} target="_blank" rel="noreferrer" className="inline-flex max-w-full items-center gap-1.5 text-brand-600 hover:text-brand-700">
                      <span className="truncate font-mono" title={run.output_model_repo ?? undefined}>{run.output_model_repo}</span>
                      <span className="shrink-0 font-mono text-[10px] text-slate-400">@{managedArtifactRevision?.slice(0, 12)}</span>
                      <ExternalLink className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                    </a>
                  ) : (
                    <span><span className="break-all font-mono text-slate-600">{run.output_model_repo}</span><span className="mt-1 block text-[10px] text-slate-400">Requested repo · existence not yet verified</span></span>
                  )
                ) : <ArtifactLink repoId={run.output_model_repo} />}
              </dd>
            </div>
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
  const { runs, selectedRunId, detail, consoleLogs, consoleLoading, consoleTruncated, consoleError, liveError, lastLiveUpdate, initialLoading, refreshing, detailLoading, listError, detailError, selectRun, refresh, retryList, retryDetail, retryConsole } = useTrainingRuns();
  const detailIsCurrent = Boolean(detail && detail.id === selectedRunId);
  const retrySelectedRun = detailError?.status === 404 || detailError?.status === 409
    ? refresh
    : retryDetail;
  return (
    <div>
      <ViewHeader title="Training runs" description="Observe managed training jobs and runs reported by external trainer clients." count={initialLoading ? null : runs.length} refreshing={refreshing || initialLoading} onRefresh={() => void refresh()} />
      {initialLoading && <RunsLoading />}
      {!initialLoading && listError && runs.length === 0 && <ErrorPanel error={listError} context="runs" onRetry={() => void retryList()} loading={initialLoading} />}
      {!initialLoading && !listError && runs.length === 0 && (
        <section className="grid min-h-[20rem] place-items-center rounded-xl border border-slate-200 bg-white px-6 text-center shadow-panel"><div className="max-w-md"><BrainCircuit className="mx-auto h-7 w-7 text-slate-300" aria-hidden="true" /><h2 className="mt-3 text-sm font-semibold text-slate-900">No training runs yet</h2><p className="mt-2 text-sm leading-6 text-slate-500">Launch a managed job from the Python SDK or report an external run through the Trainer API. Mock mode simulates managed compute without Modal or Hugging Face.</p></div></section>
      )}
      {!initialLoading && runs.length > 0 && selectedRunId && (
        <>
          {listError && <InlineError error={listError} onRetry={() => void refresh()} loading={refreshing} />}
          <div className="grid items-start gap-5 lg:grid-cols-[19rem_minmax(0,1fr)]">
            <RunList runs={runs} selectedRunId={selectedRunId} onSelect={selectRun} />
            <div className="min-w-0">
              {(detailLoading || (!detailError && !detailIsCurrent)) && <RunDetailLoading />}
              {!detailLoading && detailError && <ErrorPanel error={detailError} context="run" onRetry={() => void retrySelectedRun()} loading={detailLoading || refreshing} />}
              {!detailLoading && !detailError && detail && detailIsCurrent && <RunDetail run={detail} consoleLogs={consoleLogs} consoleLoading={consoleLoading} consoleTruncated={consoleTruncated} consoleError={consoleError} liveError={liveError} lastLiveUpdate={lastLiveUpdate} onRetryConsole={() => void retryConsole()} />}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

export function TrainingPage() {
  return (
    <div className="mx-auto max-w-7xl px-5 py-8 sm:px-8 lg:px-10 lg:py-10">
      <header>
        <p className="text-xs font-semibold uppercase tracking-[0.16em] text-brand-600">Managed and external training</p>
        <h1 className="mt-2 text-2xl font-semibold tracking-tight text-slate-950 sm:text-3xl">Training</h1>
        <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500">Monitor managed training jobs and externally reported experiments with bounded metrics, checkpoints, and sanitized console output.</p>
      </header>
      <section className="mt-7">
        <RunsView />
      </section>
    </div>
  );
}
