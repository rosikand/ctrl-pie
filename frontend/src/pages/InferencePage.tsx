import {
  Activity,
  AlertCircle,
  Bot,
  CheckCircle2,
  CloudUpload,
  Cpu,
  ExternalLink,
  Gauge,
  LoaderCircle,
  Play,
  RadioTower,
  RefreshCw,
  ShieldCheck,
  Square,
  Timer,
  TriangleAlert,
  Wifi,
  WifiOff,
} from "lucide-react";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type ReactNode,
} from "react";
import { Link } from "react-router-dom";

import { useArms } from "../hooks/useArms";
import {
  useInference,
  type InferenceRequestError,
} from "../hooks/useInference";
import { useTrainerModels } from "../hooks/useTrainerModels";
import {
  fetchPublicSettings,
  fetchSettingsStatus,
  uploadRecording,
  type PublicSettings,
  type SettingsStatus,
} from "../lib/api";
import type { ArmTelemetry } from "../types/arms";
import type {
  DeploymentRead,
  DeploymentStatus,
  InferenceComputeSize,
  InferenceRuntime,
  InferenceSessionStatus,
  InferenceStateRead,
} from "../types/inference";
import type { UploadRecordingResponse } from "../types/recordings";
import type { TrainerModelSummary } from "../types/training";

const countFormatter = new Intl.NumberFormat();
const rateFormatter = new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 });
const GPU_OPTIONS: Exclude<InferenceComputeSize, "CPU">[] = [
  "Modal: A10G",
  "Modal: A100",
  "Modal: H100",
];
const MOCK_POLICY: TrainerModelSummary = {
  repo_id: "ctrl-pi/mock-policy",
  name: "mock-policy",
  revision: "0000000000000000000000000000000000000000",
  hub_url: "",
  private: true,
  gated: false,
  last_modified: null,
  pipeline_tag: "robotics",
  library_name: "ctrl-pi-stub",
  tags: ["mock", "offline"],
  card: {
    description: "Deterministic no-network policy for the complete mock inference loop.",
    base_model: [],
    datasets: [],
  },
  checkpoints: [],
};

const deploymentStatusStyles: Record<DeploymentStatus, string> = {
  created: "bg-slate-100 text-slate-600",
  deploying: "bg-amber-50 text-amber-700",
  running: "bg-emerald-50 text-emerald-700",
  stopping: "bg-amber-50 text-amber-700",
  stopped: "bg-slate-100 text-slate-600",
  failed: "bg-rose-50 text-rose-700",
};

const sessionStatusStyles: Record<InferenceSessionStatus, string> = {
  idle: "bg-slate-100 text-slate-600",
  starting: "bg-amber-50 text-amber-700",
  running: "bg-blue-50 text-blue-700",
  stopping: "bg-amber-50 text-amber-700",
  stopped: "bg-slate-100 text-slate-600",
  failed: "bg-rose-50 text-rose-700",
};

function formatTimestamp(value: string | null): string {
  if (!value) return "Not yet";
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) return "Unknown";
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatDuration(seconds: number): string {
  const whole = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(whole / 60);
  const remainder = whole % 60;
  return `${minutes.toString().padStart(2, "0")}:${remainder.toString().padStart(2, "0")}`;
}

function shortRevision(revision: string | null): string {
  return revision ? revision.slice(0, 10) : "default branch";
}

function suggestedRepoName(value: string): string {
  const slug = value
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "-")
    .replace(/\.{2,}/g, ".")
    .replace(/-{2,}/g, "-")
    .replace(/^[-.]+|[-.]+$/g, "")
    .slice(0, 64);
  return slug || "inference-recording";
}

function repoNameIssue(repoName: string): string | null {
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

function safeHubUrl(value: string): string | null {
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

function safeDatasetRepoUrl(repoId: string, namespace: string | null): string | null {
  if (!namespace || repoId !== `${namespace}/${repoId.slice(namespace.length + 1)}`) return null;
  const separator = repoId.indexOf("/");
  if (separator !== namespace.length || separator !== repoId.lastIndexOf("/")) return null;
  const repoName = repoId.slice(separator + 1);
  if (repoNameIssue(repoName)) return null;
  return `https://huggingface.co/datasets/${encodeURIComponent(namespace)}/${encodeURIComponent(repoName)}`;
}

function deploymentIsActive(deployment: DeploymentRead | null): boolean {
  return Boolean(
    deployment &&
      !["stopped", "failed"].includes(deployment.status),
  );
}

function sessionIsActive(status: InferenceSessionStatus | undefined): boolean {
  return status === "starting" || status === "running" || status === "stopping";
}

function StatusBadge({ status, kind }: { status: DeploymentStatus | InferenceSessionStatus; kind: "deployment" | "session" }) {
  const styles = kind === "deployment"
    ? deploymentStatusStyles[status as DeploymentStatus]
    : sessionStatusStyles[status as InferenceSessionStatus];
  const animated = status === "deploying" || status === "starting" || status === "running" || status === "stopping";
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[10px] font-semibold capitalize ${styles}`}>
      <span className={`h-1.5 w-1.5 rounded-full bg-current ${animated ? "animate-pulse" : ""}`} />
      {status}
    </span>
  );
}

function useInferenceSettings() {
  const [settings, setSettings] = useState<PublicSettings | null>(null);
  const [status, setStatus] = useState<SettingsStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const sequence = useRef(0);
  const controllerRef = useRef<AbortController | null>(null);

  const load = useCallback(async () => {
    const requestSequence = sequence.current + 1;
    sequence.current = requestSequence;
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setLoading(true);
    setError(null);
    const [settingsResult, statusResult] = await Promise.allSettled([
      fetchPublicSettings(controller.signal),
      fetchSettingsStatus(controller.signal),
    ]);
    if (controller.signal.aborted || sequence.current !== requestSequence) return;
    if (settingsResult.status === "fulfilled") setSettings(settingsResult.value);
    if (statusResult.status === "fulfilled") setStatus(statusResult.value);
    const failures = [settingsResult, statusResult].filter(
      (result): result is PromiseRejectedResult => result.status === "rejected",
    );
    if (failures.length > 0) {
      const reason = failures[0].reason as unknown;
      setError(reason instanceof Error ? reason.message : "Could not load inference settings.");
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    void load();
    return () => {
      sequence.current += 1;
      controllerRef.current?.abort();
    };
  }, [load]);

  return { settings, status, loading, error, refresh: load };
}

function ReadinessPanel({
  status,
  loading,
  error,
  onRetry,
}: {
  status: SettingsStatus | null;
  loading: boolean;
  error: string | null;
  onRetry: () => void;
}) {
  const readiness = status?.inference;
  if (loading && !readiness) {
    return <div className="h-20 animate-pulse rounded-xl border border-slate-200 bg-white shadow-panel" aria-label="Checking inference readiness" />;
  }
  if (!readiness) {
    return (
      <div className="flex flex-col gap-3 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-xs text-amber-800 sm:flex-row sm:items-center sm:justify-between" role="alert">
        <span>{error ?? "Inference readiness is unavailable."}</span>
        <button type="button" onClick={onRetry} className="w-fit rounded-md border border-amber-200 bg-white px-3 py-1.5 font-semibold hover:bg-amber-100">Retry</button>
      </div>
    );
  }

  const checks = [
    { label: "Hugging Face", ready: readiness.hf_configured },
    { label: "Modal API", ready: readiness.modal_configured },
    { label: "Proxy tokens", ready: readiness.modal_proxy_configured },
  ];
  return (
    <section className={`rounded-xl border px-4 py-3 ${readiness.mock_mode ? "border-blue-200 bg-blue-50" : "border-slate-200 bg-white shadow-panel"}`}>
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-start gap-2.5">
          {readiness.mock_mode ? <Cpu className="mt-0.5 h-4 w-4 text-blue-600" /> : <ShieldCheck className="mt-0.5 h-4 w-4 text-slate-500" />}
          <div>
            <p className="text-xs font-semibold text-slate-800">{readiness.mock_mode ? "Mock inference is ready" : "Provider readiness"}</p>
            <p className="mt-0.5 text-xs text-slate-500">
              {readiness.mock_mode
                ? "LeRobot and OpenPI selections run through the deterministic local stub without cloud credentials."
                : "Model access, Modal lifecycle credentials, and endpoint proxy tokens must all be ready."}
            </p>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {!readiness.mock_mode && checks.map((check) => (
            <span key={check.label} className={`rounded-full px-2 py-1 text-[10px] font-semibold ${check.ready ? "bg-emerald-50 text-emerald-700" : "bg-amber-50 text-amber-700"}`}>
              {check.label}: {check.ready ? "ready" : "missing"}
            </span>
          ))}
          <Link to="/settings" className="rounded-md border border-slate-200 bg-white px-2.5 py-1 text-[10px] font-semibold text-slate-600 hover:bg-slate-50">Settings</Link>
        </div>
      </div>
    </section>
  );
}

function RequestError({ error, onDismiss }: { error: InferenceRequestError; onDismiss?: () => void }) {
  let guidance = "Check the backend connection and try again.";
  if (error.status === 409) guidance = "Refresh the authoritative state before retrying this lifecycle action.";
  if (error.status === 422) guidance = "Review the selected model, runtime, compute, and robot fields.";
  if (error.status === 503) guidance = "Complete the required backend configuration in Settings.";
  if (error.status === 502) guidance = "The backend safely rejected a provider or runtime failure; retry after checking server configuration.";
  return (
    <div className="flex flex-col gap-3 rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-xs text-rose-800 sm:flex-row sm:items-start sm:justify-between" role="alert">
      <div className="flex items-start gap-2">
        <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
        <div><p className="font-semibold">{error.message}</p><p className="mt-1 text-rose-700/80">{guidance}</p></div>
      </div>
      {onDismiss && <button type="button" onClick={onDismiss} className="w-fit font-semibold underline underline-offset-2">Dismiss</button>}
    </div>
  );
}

function DeploymentList({
  deployments,
  selectedId,
  disabled,
  loading,
  refreshing,
  error,
  onSelect,
  onRefresh,
}: {
  deployments: DeploymentRead[];
  selectedId: string | null;
  disabled: boolean;
  loading: boolean;
  refreshing: boolean;
  error: InferenceRequestError | null;
  onSelect: (id: string) => void;
  onRefresh: () => void;
}) {
  return (
    <aside className="rounded-xl border border-slate-200 bg-white shadow-panel">
      <div className="flex items-center justify-between border-b border-slate-100 px-4 py-3">
        <div><h2 className="text-xs font-semibold text-slate-900">Deployments</h2><p className="mt-0.5 text-[10px] text-slate-400">Durable provider history</p></div>
        <button type="button" onClick={onRefresh} disabled={refreshing || disabled} aria-label="Refresh deployments" className="rounded-md p-2 text-slate-400 hover:bg-slate-50 hover:text-slate-700 disabled:opacity-50"><RefreshCw className={`h-3.5 w-3.5 ${refreshing ? "animate-spin" : ""}`} /></button>
      </div>
      {loading ? (
        <div className="space-y-2 p-3" aria-busy="true">{Array.from({ length: 3 }).map((_, index) => <div key={index} className="h-16 animate-pulse rounded-lg bg-slate-50" />)}</div>
      ) : deployments.length === 0 ? (
        <div className="px-5 py-8 text-center"><RadioTower className="mx-auto h-5 w-5 text-slate-300" /><p className="mt-3 text-xs font-medium text-slate-600">No deployments yet</p><p className="mt-1 text-[11px] leading-5 text-slate-400">Configure a policy and deploy it when ready.</p></div>
      ) : (
        <div className="max-h-[32rem] space-y-1 overflow-y-auto p-2">
          {deployments.map((deployment) => (
            <button key={deployment.id} type="button" disabled={disabled} onClick={() => onSelect(deployment.id)} aria-current={deployment.id === selectedId ? "true" : undefined} className={`w-full rounded-lg px-3 py-3 text-left transition disabled:cursor-not-allowed ${deployment.id === selectedId ? "bg-brand-50 ring-1 ring-brand-100" : "hover:bg-slate-50"}`}>
              <span className="flex items-start justify-between gap-2"><span className="min-w-0 truncate text-xs font-semibold text-slate-800">{deployment.name}</span><StatusBadge status={deployment.status} kind="deployment" /></span>
              <span className="mt-2 block truncate font-mono text-[10px] text-slate-400" title={deployment.model_repo}>{deployment.model_repo}</span>
              <span className="mt-1 flex items-center justify-between text-[10px] text-slate-400"><span>{deployment.runtime} · {deployment.compute_size.replace("Modal: ", "")}</span><span>{formatTimestamp(deployment.updated_at)}</span></span>
            </button>
          ))}
        </div>
      )}
      {error && <div className="border-t border-rose-100 bg-rose-50 px-4 py-3 text-[11px] text-rose-700">{error.message}</div>}
    </aside>
  );
}

function MetricCard({ label, value, detail, icon: Icon }: { label: string; value: string; detail: string; icon: typeof Activity }) {
  return (
    <div className="rounded-lg bg-slate-50 px-3 py-3">
      <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.1em] text-slate-400"><Icon className="h-3.5 w-3.5" />{label}</div>
      <p className="mt-2 font-mono text-lg font-semibold tabular-nums text-slate-900">{value}</p>
      <p className="mt-0.5 text-[10px] text-slate-400">{detail}</p>
    </div>
  );
}

function UploadPanel({ recordingId, name, namespace }: { recordingId: string; name: string; namespace: string | null }) {
  const [repoName, setRepoName] = useState(() => suggestedRepoName(name));
  const [isPrivate, setIsPrivate] = useState(true);
  const [publicConfirmed, setPublicConfirmed] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<UploadRecordingResponse | null>(null);
  const controllerRef = useRef<AbortController | null>(null);
  const issue = repoNameIssue(repoName);

  useEffect(() => () => controllerRef.current?.abort(), []);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (issue || !namespace || (!isPrivate && !publicConfirmed)) return;
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setPending(true);
    setError(null);
    setResult(null);
    try {
      const response = await uploadRecording(recordingId, {
        repo_name: repoName.trim(),
        private: isPrivate,
      }, controller.signal);
      if (!controller.signal.aborted) setResult(response);
    } catch (reason) {
      if (!controller.signal.aborted) {
        setError(reason instanceof Error ? reason.message : "Could not upload the recording.");
      }
    } finally {
      if (!controller.signal.aborted) setPending(false);
    }
  }

  const resultUrl = result ? safeHubUrl(result.repo_url) : null;
  return (
    <section className="rounded-lg border border-slate-200 bg-slate-50 p-4">
      <div className="flex items-center gap-2"><CloudUpload className="h-4 w-4 text-slate-500" /><h3 className="text-xs font-semibold text-slate-800">Upload finalized recording</h3></div>
      {result ? (
        <div className="mt-3 rounded-lg border border-emerald-200 bg-emerald-50 p-3 text-xs text-emerald-800" role="status">
          <p className="font-semibold">Uploaded to {result.repo_id}</p>
          {resultUrl && <a href={resultUrl} target="_blank" rel="noreferrer" className="mt-2 inline-flex items-center gap-1 font-semibold underline underline-offset-2">Open on Hugging Face <ExternalLink className="h-3 w-3" /></a>}
        </div>
      ) : (
        <form onSubmit={submit} className="mt-3 space-y-3">
          <label className="block text-[11px] font-medium text-slate-600">Dataset repository
            <div className="mt-1.5 flex rounded-lg border border-slate-200 bg-white focus-within:border-brand-500 focus-within:ring-4 focus-within:ring-brand-100">
              <span className="max-w-[45%] truncate border-r border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-400">{namespace ? `${namespace}/` : "namespace/"}</span>
              <input value={repoName} disabled={pending || !namespace} onChange={(event) => setRepoName(event.target.value)} aria-invalid={Boolean(issue)} className="min-w-0 flex-1 rounded-r-lg px-3 py-2 text-xs outline-none disabled:bg-slate-50" />
            </div>
          </label>
          {issue && <p className="text-[10px] leading-4 text-rose-600">{issue}</p>}
          {!namespace && <p className="text-[10px] text-amber-700">Configure the server-side Hugging Face namespace before upload.</p>}
          <label className="flex items-center gap-2 text-[11px] font-medium text-slate-600"><input type="checkbox" checked={isPrivate} disabled={pending} onChange={(event) => { setIsPrivate(event.target.checked); setPublicConfirmed(false); }} className="h-4 w-4 rounded border-slate-300 text-brand-600" />Private dataset <span className="font-normal text-slate-400">(recommended)</span></label>
          {!isPrivate && (
            <label className="flex items-start gap-2 rounded-md border border-amber-200 bg-amber-50 p-2.5 text-[10px] leading-4 text-amber-800"><input type="checkbox" checked={publicConfirmed} disabled={pending} onChange={(event) => setPublicConfirmed(event.target.checked)} className="mt-0.5 h-3.5 w-3.5 rounded border-amber-300" />I understand this makes recorded robot data public.</label>
          )}
          {error && <p className="text-[11px] text-rose-600" role="alert">{error}</p>}
          <button type="submit" disabled={pending || Boolean(issue) || !namespace || (!isPrivate && !publicConfirmed)} className="inline-flex items-center gap-2 rounded-lg bg-slate-900 px-3.5 py-2 text-xs font-semibold text-white hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50">{pending ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : <CloudUpload className="h-3.5 w-3.5" />}{pending ? "Uploading…" : error ? "Retry upload" : "Upload dataset"}</button>
        </form>
      )}
    </section>
  );
}

function LivePanel({ state, fallback, connection, now, hfNamespace }: { state: InferenceStateRead | null; fallback: DeploymentRead | null; connection: ReturnType<typeof useInference>["connection"]; now: number; hfNamespace: string | null }) {
  const deployment = state ?? fallback;
  if (!deployment) {
    return (
      <section className="grid min-h-[24rem] place-items-center rounded-xl border border-slate-200 bg-white px-6 text-center shadow-panel">
        <div><RadioTower className="mx-auto h-6 w-6 text-slate-300" /><p className="mt-3 text-sm font-semibold text-slate-700">No deployment selected</p><p className="mt-1 max-w-sm text-xs leading-5 text-slate-400">Deploy a policy to verify its endpoint before starting robot actions.</p></div>
      </section>
    );
  }
  const sessionSeconds = state?.session_started_at
    ? Math.max(0, (now - new Date(state.session_started_at).valueOf()) / 1_000)
    : 0;
  const inferenceFrequency = state && sessionSeconds > 0
    ? state.requests_completed / sessionSeconds
    : null;
  const streamLabel = connection === "live" ? "Live" : connection === "reconnecting" ? "Reconnecting" : connection === "connecting" ? "Connecting" : "Snapshot";
  const uploadedRecordingUrl = state?.recording.hf_repo_id
    ? safeDatasetRepoUrl(state.recording.hf_repo_id, hfNamespace)
    : null;
  return (
    <section className="rounded-xl border border-slate-200 bg-white shadow-panel">
      <div className="flex flex-col gap-4 border-b border-slate-100 px-5 py-4 sm:flex-row sm:items-start sm:justify-between sm:px-6">
        <div className="min-w-0"><div className="flex flex-wrap items-center gap-2"><h2 className="truncate text-sm font-semibold text-slate-900">{deployment.name}</h2><StatusBadge status={deployment.status} kind="deployment" />{state && <StatusBadge status={state.session_status} kind="session" />}</div><p className="mt-2 truncate font-mono text-xs text-slate-500" title={`${deployment.model_repo}@${deployment.checkpoint_revision ?? "default"}`}>{deployment.model_repo}@{shortRevision(deployment.checkpoint_revision)}</p></div>
        <span className={`inline-flex w-fit items-center gap-1.5 rounded-full px-2.5 py-1 text-[10px] font-semibold ${connection === "live" ? "bg-emerald-50 text-emerald-700" : connection === "reconnecting" ? "bg-amber-50 text-amber-700" : "bg-slate-100 text-slate-500"}`}>{connection === "live" ? <Wifi className="h-3 w-3" /> : <WifiOff className="h-3 w-3" />}{streamLabel}</span>
      </div>
      <div className="space-y-5 p-5 sm:p-6">
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <MetricCard icon={Timer} label="Latency" value={state?.last_latency_ms == null ? "—" : `${rateFormatter.format(state.last_latency_ms)} ms`} detail={state?.average_latency_ms == null ? "No completed request" : `${rateFormatter.format(state.average_latency_ms)} ms average`} />
          <MetricCard icon={Activity} label="Inference rate" value={inferenceFrequency == null ? "—" : `${rateFormatter.format(inferenceFrequency)} Hz`} detail={`${countFormatter.format(state?.requests_completed ?? 0)} chunks completed`} />
          <MetricCard icon={Gauge} label="Action rate" value={state?.frequency_hz == null ? "—" : `${rateFormatter.format(state.frequency_hz)} Hz`} detail={`${countFormatter.format(state?.steps_executed ?? 0)} actions executed`} />
          <MetricCard icon={RadioTower} label="Queue" value={countFormatter.format(state?.queue_depth ?? 0)} detail={`${countFormatter.format(state?.dropped_chunks ?? 0)} chunks dropped`} />
        </div>
        <dl className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <Detail label="Exact loaded policy"><span title={deployment.checkpoint_revision ?? "No pinned revision"}>{deployment.model_repo}<br /><span className="font-mono text-[11px] text-slate-500">{deployment.checkpoint_revision ?? "revision pending"}</span></span></Detail>
          <Detail label="Runtime / compute">{deployment.runtime === "lerobot" ? "LeRobot" : deployment.runtime === "openpi" ? "OpenPI" : "Stub"}<br /><span className="text-[11px] font-normal text-slate-500">{deployment.compute_size}</span></Detail>
          <Detail label="Endpoint health"><span className={state?.endpoint_healthy ? "text-emerald-700" : "text-slate-500"}>{state?.endpoint_healthy ? "Healthy · identity verified" : deployment.endpoint_url ? "Assigned · awaiting health" : "Not available"}</span><br /><span className="font-mono text-[10px] font-normal text-slate-400" title={deployment.endpoint_url ?? undefined}>{deployment.provider_app_id ?? deployment.endpoint_id}</span></Detail>
          <Detail label="Follower arm">{deployment.arm_id ?? "Not started"}<br /><span className="text-[11px] font-normal text-slate-500">Started {formatTimestamp(deployment.started_at)}</span></Detail>
        </dl>
        {state?.last_error && <div className="flex items-start gap-2 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2.5 text-xs text-rose-700" role="alert"><AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />{state.last_error}</div>}
        {state && (state.status === "stopped" || state.status === "failed") && (
          <div className={`flex items-start gap-2 rounded-lg border px-3 py-3 text-xs ${state.teardown_verified ? "border-emerald-200 bg-emerald-50 text-emerald-800" : "border-amber-200 bg-amber-50 text-amber-800"}`}>
            {state.teardown_verified ? <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0" /> : <TriangleAlert className="mt-0.5 h-4 w-4 shrink-0" />}
            <div><p className="font-semibold">{state.teardown_verified ? "Teardown verified" : "Teardown is not yet verified"}</p><p className="mt-0.5 opacity-80">{state.teardown_verified ? "The action loop is joined, the queue is empty, and the provider resource is stopped." : "Do not assume the provider resource is gone; retry Stop or inspect the backend."}</p></div>
          </div>
        )}
        {state?.recording.enabled && (
          <div className="space-y-3 rounded-lg border border-slate-200 p-4">
            <div className="flex flex-wrap items-center justify-between gap-2"><div><h3 className="text-xs font-semibold text-slate-800">Session recording</h3><p className="mt-1 text-[11px] text-slate-400">{state.recording.episode_count} episode · {formatDuration(state.recording.duration_seconds)} · {state.recording.status}</p></div>{state.recording.status === "recording" && <span className="inline-flex items-center gap-1.5 rounded-full bg-rose-50 px-2.5 py-1 text-[10px] font-semibold text-rose-700"><span className="h-1.5 w-1.5 animate-pulse rounded-full bg-rose-500" />REC</span>}</div>
            {state.recording.hf_repo_id && (
              <p className="text-xs font-medium text-emerald-700">
                Uploaded as {state.recording.hf_repo_id}
                {uploadedRecordingUrl && <a href={uploadedRecordingUrl} target="_blank" rel="noreferrer" className="ml-2 inline-flex items-center gap-1 underline underline-offset-2">Open on Hugging Face <ExternalLink className="h-3 w-3" /></a>}
              </p>
            )}
          </div>
        )}
      </div>
    </section>
  );
}

function Detail({ label, children }: { label: string; children: ReactNode }) {
  return <div className="min-w-0 rounded-lg bg-slate-50 px-3 py-3"><dt className="text-[10px] font-semibold uppercase tracking-[0.1em] text-slate-400">{label}</dt><dd className="mt-1.5 break-words text-xs font-semibold leading-5 text-slate-800">{children}</dd></div>;
}

export function InferencePage() {
  const arms = useArms();
  const models = useTrainerModels();
  const inference = useInference();
  const inferenceSettings = useInferenceSettings();
  const [deploymentName, setDeploymentName] = useState("");
  const [modelRepo, setModelRepo] = useState("");
  const [runtime, setRuntime] = useState<Exclude<InferenceRuntime, "stub">>("lerobot");
  const [computeSize, setComputeSize] = useState<Exclude<InferenceComputeSize, "CPU">>("Modal: A10G");
  const [armId, setArmId] = useState("");
  const [task, setTask] = useState("");
  const [recordSession, setRecordSession] = useState(false);
  const [recordingName, setRecordingName] = useState("");
  const [operator, setOperator] = useState("");
  const [recordingNotes, setRecordingNotes] = useState("");
  const [now, setNow] = useState(Date.now());
  const defaultsApplied = useRef(false);

  const readiness = inferenceSettings.status?.inference;
  const catalogModels = models.data?.models;
  const availableModels = useMemo(() => {
    const discovered = catalogModels ?? [];
    if (!readiness?.mock_mode) return discovered;
    return [MOCK_POLICY, ...discovered.filter((model) => model.repo_id !== MOCK_POLICY.repo_id)];
  }, [catalogModels, readiness?.mock_mode]);
  const followers = useMemo(
    () => arms.arms.filter((arm) => arm.role === "follower" && arm.connected),
    [arms.arms],
  );
  const selectedModel = availableModels.find((model) => model.repo_id === modelRepo) ?? null;
  const selectedState = inference.state?.id === inference.selectedDeploymentId ? inference.state : null;
  const selectedDeployment = inference.selectedDeployment;
  const activeProvider = deploymentIsActive(selectedDeployment);
  const activeSession = sessionIsActive(selectedState?.session_status);
  const teardownPending = selectedDeployment?.status === "failed" &&
    selectedState?.teardown_verified !== true;
  const lifecycleLocked = activeProvider || teardownPending;
  const configurationLocked = Boolean(inference.busy) || lifecycleLocked;
  const sessionLocked = Boolean(inference.busy) || activeSession;
  const environmentReady = Boolean(
    readiness && (
      readiness.mock_mode ||
      (readiness.hf_configured && readiness.modal_configured && readiness.modal_proxy_configured)
    ),
  );

  useEffect(() => {
    if (!inferenceSettings.settings || defaultsApplied.current) return;
    defaultsApplied.current = true;
    if (selectedDeployment) return;
    setRuntime(inferenceSettings.settings.default_runtime);
    setComputeSize(inferenceSettings.settings.default_compute);
  }, [inferenceSettings.settings, selectedDeployment]);

  useEffect(() => {
    if (availableModels.length === 0) return;
    const selectedHistoryModel = inference.selectedDeployment?.model_repo;
    if (
      !availableModels.some((model) => model.repo_id === modelRepo) &&
      modelRepo !== selectedHistoryModel
    ) {
      setModelRepo(availableModels[0].repo_id);
    }
  }, [availableModels, inference.selectedDeployment?.model_repo, modelRepo]);

  useEffect(() => {
    if (!followers.some((arm) => arm.id === armId)) setArmId(followers[0]?.id ?? "");
  }, [armId, followers]);

  useEffect(() => {
    if (!selectedDeployment) return;
    setDeploymentName(selectedDeployment.name);
    setModelRepo(selectedDeployment.model_repo);
    if (selectedDeployment.runtime !== "stub") setRuntime(selectedDeployment.runtime);
    if (selectedDeployment.compute_size !== "CPU") setComputeSize(selectedDeployment.compute_size);
    if (selectedDeployment.arm_id) setArmId(selectedDeployment.arm_id);
  }, [selectedDeployment?.id]);

  useEffect(() => {
    if (selectedState?.session_status !== "running") return;
    setNow(Date.now());
    const timer = window.setInterval(() => setNow(Date.now()), 1_000);
    return () => window.clearInterval(timer);
  }, [selectedState?.session_status]);

  async function deploy(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedModel || !deploymentName.trim() || !environmentReady) return;
    await inference.deploy({
      name: deploymentName.trim(),
      model_repo: selectedModel.repo_id,
      checkpoint_revision: selectedModel.revision,
      runtime,
      compute_size: computeSize,
    });
  }

  async function start() {
    if (!selectedState || !armId || !task.trim()) return;
    const metadata: { operator?: string; notes?: string } = {};
    if (operator.trim()) metadata.operator = operator.trim();
    if (recordingNotes.trim()) metadata.notes = recordingNotes.trim();
    await inference.start(selectedState.id, {
      arm_id: armId,
      task: task.trim(),
      record_session: recordSession,
      recording_name: recordSession && recordingName.trim() ? recordingName.trim() : null,
      ...(recordSession && Object.keys(metadata).length ? { recording_metadata: metadata } : {}),
    });
  }

  async function stop() {
    if (!selectedDeployment) return;
    await inference.stop(selectedDeployment.id, {
      recording_success: true,
      recording_notes: recordingNotes.trim() || null,
    });
  }

  const canDeploy = !configurationLocked &&
    !inference.initialLoading &&
    !inference.refreshing &&
    !inference.listError &&
    environmentReady &&
    Boolean(selectedModel && deploymentName.trim());
  const canStart = Boolean(
    selectedState &&
      selectedState.status === "running" &&
      selectedState.endpoint_healthy &&
      selectedState.session_status === "idle" &&
      armId &&
      task.trim() &&
      !sessionLocked,
  );
  const canStop = Boolean(
    selectedDeployment &&
      !inference.busy &&
      (["running", "stopping", "failed"] as DeploymentStatus[]).includes(selectedDeployment.status) &&
      !(selectedDeployment.status === "failed" && selectedState?.teardown_verified),
  );
  const recordingReady = selectedState?.recording.enabled &&
    selectedState.recording.status === "ready" &&
    !selectedState.recording.hf_repo_id &&
    selectedState.recording.recording_id;

  return (
    <div className="mx-auto max-w-7xl px-5 py-8 sm:px-8 lg:px-10 lg:py-10">
      <header className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div><p className="text-xs font-semibold uppercase tracking-[0.16em] text-brand-600">Policy execution</p><h1 className="mt-2 text-2xl font-semibold tracking-tight text-slate-950 sm:text-3xl">Inference</h1><p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500">Deploy one pinned policy, verify its endpoint, then explicitly start and tear down a follower-arm session.</p></div>
        <span className="inline-flex w-fit items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-500 shadow-sm"><span className={`h-1.5 w-1.5 rounded-full ${readiness?.mock_mode ? "bg-blue-500" : readiness ? "bg-emerald-500" : "bg-slate-300"}`} />{readiness?.mock_mode ? "Mock runtime" : readiness ? "Modal runtime" : "Checking runtime"}</span>
      </header>

      <div className="mt-6"><ReadinessPanel status={inferenceSettings.status} loading={inferenceSettings.loading} error={inferenceSettings.error} onRetry={inferenceSettings.refresh} /></div>
      {inference.operationError && <div className="mt-4"><RequestError error={inference.operationError} onDismiss={inference.clearOperationError} /></div>}

      <div className="mt-6 grid gap-6 xl:grid-cols-[18rem_minmax(0,1fr)]">
        <DeploymentList deployments={inference.deployments} selectedId={inference.selectedDeploymentId} disabled={Boolean(inference.busy) || lifecycleLocked || activeSession} loading={inference.initialLoading} refreshing={inference.refreshing} error={inference.listError} onSelect={inference.selectDeployment} onRefresh={inference.refresh} />
        <div className="min-w-0 space-y-6">
          <section className="rounded-xl border border-slate-200 bg-white shadow-panel">
            <div className="border-b border-slate-100 px-5 py-4 sm:px-6"><div className="flex items-center gap-2"><Cpu className="h-4 w-4 text-slate-400" /><h2 className="text-sm font-semibold text-slate-900">1. Deploy policy endpoint</h2></div><p className="mt-1 text-xs text-slate-400">Choose a configured-namespace model or the offline mock policy. Hub and Modal credentials remain server-side.</p></div>
            <form onSubmit={deploy} className="space-y-4 p-5 sm:p-6">
              <div className="grid gap-4 md:grid-cols-2">
                <label className="block text-xs font-medium text-slate-700">Deployment name<input required maxLength={160} disabled={configurationLocked} value={deploymentName} onChange={(event) => setDeploymentName(event.target.value)} placeholder="Pick policy" className="mt-1.5 w-full rounded-lg border border-slate-200 px-3 py-2.5 text-sm outline-none ring-brand-100 placeholder:text-slate-300 focus:border-brand-500 focus:ring-4 disabled:bg-slate-50" /></label>
                <label className="block text-xs font-medium text-slate-700">Model repository<select required disabled={configurationLocked || (models.initialLoading && !readiness?.mock_mode) || availableModels.length === 0} value={modelRepo} onChange={(event) => setModelRepo(event.target.value)} className="mt-1.5 w-full rounded-lg border border-slate-200 bg-white px-3 py-2.5 text-sm outline-none ring-brand-100 focus:border-brand-500 focus:ring-4 disabled:bg-slate-50">{availableModels.length === 0 && <option value="">No models available</option>}{selectedDeployment && !availableModels.some((model) => model.repo_id === selectedDeployment.model_repo) && <option value={selectedDeployment.model_repo}>{selectedDeployment.model_repo} (not in current catalog)</option>}{availableModels.map((model) => <option key={model.repo_id} value={model.repo_id}>{model.repo_id === MOCK_POLICY.repo_id ? "Offline mock policy" : model.repo_id}</option>)}</select></label>
                <label className="block text-xs font-medium text-slate-700">Runtime<select disabled={configurationLocked} value={runtime} onChange={(event) => setRuntime(event.target.value as Exclude<InferenceRuntime, "stub">)} className="mt-1.5 w-full rounded-lg border border-slate-200 bg-white px-3 py-2.5 text-sm outline-none ring-brand-100 focus:border-brand-500 focus:ring-4 disabled:bg-slate-50"><option value="lerobot">LeRobot</option><option value="openpi">OpenPI</option></select><span className="mt-1.5 block text-[10px] leading-4 text-slate-400">OpenPI is emulated in mock mode and intentionally unavailable on real Modal in V1.</span></label>
                <label className="block text-xs font-medium text-slate-700">Compute<select disabled={configurationLocked} value={computeSize} onChange={(event) => setComputeSize(event.target.value as Exclude<InferenceComputeSize, "CPU">)} className="mt-1.5 w-full rounded-lg border border-slate-200 bg-white px-3 py-2.5 text-sm outline-none ring-brand-100 focus:border-brand-500 focus:ring-4 disabled:bg-slate-50">{GPU_OPTIONS.map((option) => <option key={option}>{option}</option>)}</select></label>
              </div>
              {models.error && <div className="flex items-center justify-between gap-3 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2.5 text-xs text-amber-800"><span>{models.error.message}</span><button type="button" onClick={models.retry} className="shrink-0 font-semibold underline">Retry</button></div>}
              {selectedModel && <div className="grid gap-3 rounded-lg bg-slate-50 p-3 sm:grid-cols-2"><label className="block text-[10px] font-semibold uppercase tracking-[0.1em] text-slate-400">Policy revision<select value={selectedModel.revision ?? ""} disabled className="mt-1.5 w-full rounded-md border border-slate-200 bg-white px-2.5 py-2 font-mono text-[11px] normal-case tracking-normal text-slate-600"><option value={selectedModel.revision ?? ""}>{selectedModel.revision ?? "Repository default · resolved on deploy"}</option></select><span className="mt-1.5 block font-normal normal-case tracking-normal text-slate-400">The catalog exposes one deployable Git revision per model.</span></label><div><p className="text-[10px] font-semibold uppercase tracking-[0.1em] text-slate-400">Checkpoint artifacts</p><p className="mt-1 text-xs text-slate-700">{selectedModel.checkpoints.length ? `${selectedModel.checkpoints.length} files · read-only metadata` : selectedModel.repo_id === MOCK_POLICY.repo_id ? "Built-in deterministic policy" : "No checkpoint files reported"}</p>{selectedModel.checkpoints.length > 0 && <p className="mt-1 truncate font-mono text-[10px] text-slate-400" title={selectedModel.checkpoints.join(", ")}>{selectedModel.checkpoints.slice(0, 3).join(", ")}</p>}<p className="mt-1.5 text-[10px] leading-4 text-slate-400">Artifact paths are not offered as Hub revisions because the runtime cannot load them as Git refs.</p></div></div>}
              <div className="flex flex-col gap-3 border-t border-slate-100 pt-4 sm:flex-row sm:items-center sm:justify-between"><p className="text-[11px] leading-5 text-slate-400">Deploy verifies exact runtime/model/revision identity. It does not move the robot.</p><button type="submit" disabled={!canDeploy} className="inline-flex items-center justify-center gap-2 rounded-lg bg-slate-900 px-4 py-2.5 text-xs font-semibold text-white hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50">{inference.busy === "deploy" ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : <RadioTower className="h-3.5 w-3.5" />}{inference.busy === "deploy" ? "Deploying and verifying…" : "Deploy"}</button></div>
            </form>
          </section>

          <section className="rounded-xl border border-slate-200 bg-white shadow-panel">
            <div className="border-b border-slate-100 px-5 py-4 sm:px-6"><div className="flex items-center gap-2"><Bot className="h-4 w-4 text-slate-400" /><h2 className="text-sm font-semibold text-slate-900">2. Start robot execution</h2></div><p className="mt-1 text-xs text-slate-400">Start is enabled only after a healthy endpoint snapshot and a connected follower are authoritative.</p></div>
            <div className="space-y-4 p-5 sm:p-6">
              <div className="grid gap-4 md:grid-cols-2">
                <label className="block text-xs font-medium text-slate-700">Connected follower<select required disabled={sessionLocked || !selectedState || selectedState.session_status !== "idle"} value={armId} onChange={(event) => setArmId(event.target.value)} className="mt-1.5 w-full rounded-lg border border-slate-200 bg-white px-3 py-2.5 text-sm outline-none ring-brand-100 focus:border-brand-500 focus:ring-4 disabled:bg-slate-50">{followers.length === 0 && <option value="">No connected follower</option>}{followers.map((arm: ArmTelemetry) => <option key={arm.id} value={arm.id}>{arm.name} · {arm.id}</option>)}</select><span className="mt-1.5 block text-[10px] text-slate-400">Arm telemetry: {arms.connectionState}</span></label>
                <label className="block text-xs font-medium text-slate-700">Task<textarea required rows={2} maxLength={512} disabled={sessionLocked || !selectedState || selectedState.session_status !== "idle"} value={task} onChange={(event) => setTask(event.target.value)} placeholder="Pick up the blue block and place it on the target." className="mt-1.5 w-full resize-none rounded-lg border border-slate-200 px-3 py-2.5 text-sm leading-5 outline-none ring-brand-100 placeholder:text-slate-300 focus:border-brand-500 focus:ring-4 disabled:bg-slate-50" /></label>
              </div>
              {arms.error && <p className="text-xs text-rose-600">{arms.error}</p>}
              <div className="rounded-lg border border-slate-200 p-4"><label className="flex items-center gap-2 text-xs font-semibold text-slate-700"><input type="checkbox" checked={recordSession} disabled={sessionLocked || !selectedState || selectedState.session_status !== "idle"} onChange={(event) => setRecordSession(event.target.checked)} className="h-4 w-4 rounded border-slate-300 text-brand-600" />Record this inference session</label>{recordSession && <div className="mt-4 grid gap-4 md:grid-cols-2"><label className="block text-[11px] font-medium text-slate-600">Recording name <span className="font-normal text-slate-400">(optional)</span><input maxLength={160} disabled={sessionLocked} value={recordingName} onChange={(event) => setRecordingName(event.target.value)} placeholder="Generated if blank" className="mt-1.5 w-full rounded-lg border border-slate-200 px-3 py-2 text-xs outline-none focus:border-brand-500" /></label><label className="block text-[11px] font-medium text-slate-600">Operator <span className="font-normal text-slate-400">(optional)</span><input maxLength={160} disabled={sessionLocked} value={operator} onChange={(event) => setOperator(event.target.value)} className="mt-1.5 w-full rounded-lg border border-slate-200 px-3 py-2 text-xs outline-none focus:border-brand-500" /></label><label className="block text-[11px] font-medium text-slate-600 md:col-span-2">Notes <span className="font-normal text-slate-400">(optional)</span><textarea rows={2} maxLength={2_000} disabled={sessionLocked} value={recordingNotes} onChange={(event) => setRecordingNotes(event.target.value)} className="mt-1.5 w-full resize-none rounded-lg border border-slate-200 px-3 py-2 text-xs outline-none focus:border-brand-500" /></label></div>}</div>
              <div className="flex flex-col gap-3 border-t border-slate-100 pt-4 sm:flex-row sm:items-center sm:justify-end"><button type="button" onClick={start} disabled={!canStart} className="inline-flex items-center justify-center gap-2 rounded-lg bg-brand-600 px-4 py-2.5 text-xs font-semibold text-white hover:bg-brand-700 disabled:cursor-not-allowed disabled:opacity-50">{inference.busy === "start" ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5 fill-current" />}{inference.busy === "start" ? "Starting safely…" : "Start execution"}</button><button type="button" onClick={stop} disabled={!canStop} className="inline-flex items-center justify-center gap-2 rounded-lg border border-rose-200 bg-white px-4 py-2.5 text-xs font-semibold text-rose-700 hover:bg-rose-50 disabled:cursor-not-allowed disabled:opacity-50">{inference.busy === "stop" ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : <Square className="h-3.5 w-3.5 fill-current" />}{inference.busy === "stop" ? "Stopping and verifying…" : activeSession ? "Stop session + endpoint" : "Stop endpoint"}</button></div>
            </div>
          </section>

          {inference.stateError && <div><RequestError error={inference.stateError} /><button type="button" onClick={inference.retryState} disabled={inference.stateLoading} className="mt-2 inline-flex items-center gap-2 rounded-md border border-slate-200 bg-white px-3 py-1.5 text-xs font-semibold text-slate-600"><RefreshCw className={`h-3.5 w-3.5 ${inference.stateLoading ? "animate-spin" : ""}`} />Reload state</button></div>}
          {inference.stateLoading && selectedDeployment ? <div className="h-96 animate-pulse rounded-xl border border-slate-200 bg-white shadow-panel" aria-label="Loading inference state" /> : <LivePanel state={selectedState} fallback={selectedDeployment} connection={inference.connection} now={now} hfNamespace={inferenceSettings.settings?.hf_namespace ?? null} />}

          {recordingReady && (
            <UploadPanel key={recordingReady} recordingId={recordingReady} name={recordingName || selectedDeployment?.name || "inference-recording"} namespace={inferenceSettings.settings?.hf_namespace ?? null} />
          )}
        </div>
      </div>
    </div>
  );
}
