import {
  Activity,
  AlertCircle,
  Bot,
  Camera,
  ChevronRight,
  Clock3,
  FileText,
  Grip,
  LoaderCircle,
  Play,
  Plus,
  Radio,
  RefreshCw,
  Square,
  Video,
  Wifi,
  WifiOff,
} from "lucide-react";
import type { FormEvent } from "react";
import { useEffect, useMemo, useState } from "react";

import { useArms } from "../hooks/useArms";
import { useRecordings } from "../hooks/useRecordings";
import type { ArmTelemetry } from "../types/arms";
import type {
  CreateRecordingRequest,
  Recording,
  RecordingState,
  RecordingStatus,
} from "../types/recordings";

function formatDuration(seconds: number): string {
  const wholeSeconds = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(wholeSeconds / 60);
  const remainder = wholeSeconds % 60;
  return `${minutes.toString().padStart(2, "0")}:${remainder.toString().padStart(2, "0")}`;
}

function shortDate(timestamp: string): string {
  const date = new Date(timestamp);
  if (Number.isNaN(date.valueOf())) return "Unknown date";
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function degrees(radians: number): string {
  return `${((radians * 180) / Math.PI).toFixed(0)}°`;
}

const statusStyles: Record<RecordingStatus, string> = {
  draft: "bg-slate-100 text-slate-600",
  teleop: "bg-blue-50 text-blue-700",
  recording: "bg-rose-50 text-rose-700",
  ready: "bg-emerald-50 text-emerald-700",
  uploading: "bg-amber-50 text-amber-700",
  uploaded: "bg-emerald-50 text-emerald-700",
  failed: "bg-rose-50 text-rose-700",
};

function StatusBadge({ status }: { status: RecordingStatus }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[10px] font-semibold capitalize ${statusStyles[status]}`}
    >
      <span
        className={`h-1.5 w-1.5 rounded-full ${
          status === "recording"
            ? "animate-pulse bg-rose-500"
            : status === "failed"
              ? "bg-rose-500"
              : status === "teleop"
                ? "bg-blue-500"
                : status === "ready" || status === "uploaded"
                  ? "bg-emerald-500"
                  : "bg-slate-400"
        }`}
      />
      {status}
    </span>
  );
}

function SessionSetup({
  arms,
  disabled,
  onCreate,
}: {
  arms: ArmTelemetry[];
  disabled: boolean;
  onCreate: (payload: CreateRecordingRequest) => Promise<Recording | null>;
}) {
  const leaders = arms.filter((arm) => arm.role === "leader");
  const followers = arms.filter((arm) => arm.role === "follower");
  const [name, setName] = useState("");
  const [task, setTask] = useState("");
  const [leaderId, setLeaderId] = useState("");
  const [followerId, setFollowerId] = useState("");
  const [operator, setOperator] = useState("");
  const [notes, setNotes] = useState("");

  useEffect(() => {
    if (!leaders.some((arm) => arm.id === leaderId)) setLeaderId(leaders[0]?.id ?? "");
    if (!followers.some((arm) => arm.id === followerId)) {
      setFollowerId(followers[0]?.id ?? "");
    }
  }, [followerId, followers, leaderId, leaders]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const metadata: Record<string, unknown> = {};
    if (operator.trim()) metadata.operator = operator.trim();
    if (notes.trim()) metadata.notes = notes.trim();
    const created = await onCreate({
      name: name.trim(),
      task: task.trim(),
      leader_robot_id: leaderId,
      follower_robot_id: followerId,
      metadata,
    });
    if (created) {
      setName("");
      setTask("");
      setNotes("");
    }
  }

  const canSubmit =
    name.trim().length > 0 &&
    task.trim().length > 0 &&
    leaderId.length > 0 &&
    followerId.length > 0 &&
    leaderId !== followerId;

  return (
    <section className="rounded-xl border border-slate-200 bg-white shadow-panel">
      <div className="border-b border-slate-100 px-5 py-4 sm:px-6">
        <div className="flex items-center gap-2">
          <FileText className="h-4 w-4 text-slate-400" strokeWidth={1.8} />
          <h2 className="text-sm font-semibold text-slate-900">New recording session</h2>
        </div>
        <p className="mt-1 text-xs text-slate-400">Choose one YAM leader/follower pair and describe the task.</p>
      </div>
      <form onSubmit={submit} className="space-y-4 p-5 sm:p-6">
        <div className="grid gap-4 lg:grid-cols-2">
          <label className="block text-xs font-medium text-slate-700">
            Session name
            <input
              required
              disabled={disabled}
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="Cup stacking demos"
              className="mt-1.5 w-full rounded-lg border border-slate-200 px-3 py-2.5 text-sm outline-none ring-brand-100 transition placeholder:text-slate-300 focus:border-brand-500 focus:ring-4"
            />
          </label>
          <label className="block text-xs font-medium text-slate-700">
            Operator <span className="font-normal text-slate-400">(optional)</span>
            <input
              value={operator}
              disabled={disabled}
              onChange={(event) => setOperator(event.target.value)}
              placeholder="Operator name"
              className="mt-1.5 w-full rounded-lg border border-slate-200 px-3 py-2.5 text-sm outline-none ring-brand-100 transition placeholder:text-slate-300 focus:border-brand-500 focus:ring-4"
            />
          </label>
        </div>
        <label className="block text-xs font-medium text-slate-700">
          Task
          <textarea
            required
            disabled={disabled}
            rows={2}
            value={task}
            onChange={(event) => setTask(event.target.value)}
            placeholder="Pick up the blue cup and place it on the marked target."
            className="mt-1.5 w-full resize-none rounded-lg border border-slate-200 px-3 py-2.5 text-sm leading-5 outline-none ring-brand-100 transition placeholder:text-slate-300 focus:border-brand-500 focus:ring-4"
          />
        </label>
        <div className="grid gap-4 sm:grid-cols-2">
          <label className="block text-xs font-medium text-slate-700">
            Leader arm
            <select
              required
              disabled={disabled}
              value={leaderId}
              onChange={(event) => setLeaderId(event.target.value)}
              className="mt-1.5 w-full rounded-lg border border-slate-200 bg-white px-3 py-2.5 text-sm outline-none ring-brand-100 focus:border-brand-500 focus:ring-4"
            >
              {leaders.length === 0 && <option value="">No leader available</option>}
              {leaders.map((arm) => (
                <option key={arm.id} value={arm.id}>{arm.name}</option>
              ))}
            </select>
          </label>
          <label className="block text-xs font-medium text-slate-700">
            Follower arm
            <select
              required
              disabled={disabled}
              value={followerId}
              onChange={(event) => setFollowerId(event.target.value)}
              className="mt-1.5 w-full rounded-lg border border-slate-200 bg-white px-3 py-2.5 text-sm outline-none ring-brand-100 focus:border-brand-500 focus:ring-4"
            >
              {followers.length === 0 && <option value="">No follower available</option>}
              {followers.map((arm) => (
                <option key={arm.id} value={arm.id}>{arm.name}</option>
              ))}
            </select>
          </label>
        </div>
        <label className="block text-xs font-medium text-slate-700">
          Session notes <span className="font-normal text-slate-400">(optional)</span>
          <input
            value={notes}
            disabled={disabled}
            onChange={(event) => setNotes(event.target.value)}
            placeholder="Lighting, fixture, or reset details"
            className="mt-1.5 w-full rounded-lg border border-slate-200 px-3 py-2.5 text-sm outline-none ring-brand-100 transition placeholder:text-slate-300 focus:border-brand-500 focus:ring-4"
          />
        </label>
        <div className="flex items-center justify-between gap-4 border-t border-slate-100 pt-4">
          <p className="text-[11px] text-slate-400">Metadata is saved with this session.</p>
          <button
            type="submit"
            disabled={disabled || !canSubmit}
            className="inline-flex items-center gap-2 rounded-lg bg-ink px-3.5 py-2.5 text-xs font-semibold text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {disabled ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : <Plus className="h-3.5 w-3.5" />}
            Create session
          </button>
        </div>
      </form>
    </section>
  );
}

function CameraFeed() {
  const [imageKey, setImageKey] = useState(0);
  const [cameraState, setCameraState] = useState<"loading" | "live" | "error">("loading");

  function reconnect() {
    setCameraState("loading");
    setImageKey((current) => current + 1);
  }

  return (
    <section className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-panel">
      <div className="flex items-center justify-between border-b border-slate-100 px-4 py-3 sm:px-5">
        <div className="flex items-center gap-2">
          <Camera className="h-4 w-4 text-slate-400" strokeWidth={1.8} />
          <h2 className="text-sm font-semibold text-slate-900">Workspace camera</h2>
        </div>
        <span className={`inline-flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider ${cameraState === "live" ? "text-emerald-600" : cameraState === "error" ? "text-rose-600" : "text-slate-400"}`}>
          <span className={`h-1.5 w-1.5 rounded-full ${cameraState === "live" ? "animate-pulse bg-emerald-500" : cameraState === "error" ? "bg-rose-500" : "animate-pulse bg-slate-400"}`} />
          {cameraState === "live" ? "Live" : cameraState}
        </span>
      </div>
      <div className="relative aspect-[4/3] overflow-hidden bg-slate-950">
        <img
          key={imageKey}
          src={`/api/camera/mock/stream?connection=${imageKey}`}
          alt="Live workspace view from the mock camera"
          onLoad={() => setCameraState("live")}
          onError={() => setCameraState("error")}
          className={`h-full w-full object-contain transition-opacity ${cameraState === "live" ? "opacity-100" : "opacity-20"}`}
        />
        {cameraState !== "live" && (
          <div className="absolute inset-0 grid place-items-center text-center">
            <div>
              {cameraState === "error" ? (
                <WifiOff className="mx-auto h-6 w-6 text-slate-500" />
              ) : (
                <LoaderCircle className="mx-auto h-6 w-6 animate-spin text-slate-500" />
              )}
              <p className="mt-2 text-xs font-medium text-slate-300">
                {cameraState === "error" ? "Camera stream unavailable" : "Connecting to mock camera"}
              </p>
              {cameraState === "error" && (
                <button type="button" onClick={reconnect} className="mt-3 text-[11px] font-semibold text-blue-300 hover:text-blue-200">
                  Reconnect
                </button>
              )}
            </div>
          </div>
        )}
        <div className="absolute bottom-3 left-3 rounded-md bg-slate-950/70 px-2 py-1 font-mono text-[10px] text-white/70 backdrop-blur">
          MOCK CAMERA · MJPEG
        </div>
      </div>
    </section>
  );
}

function CompactArmState({ arm, label }: { arm: ArmTelemetry | undefined; label: string }) {
  if (!arm) {
    return (
      <article className="rounded-lg border border-dashed border-slate-200 p-4">
        <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">{label}</p>
        <p className="mt-2 text-xs text-slate-500">Arm telemetry unavailable</p>
      </article>
    );
  }

  return (
    <article className="rounded-lg border border-slate-100 bg-slate-50/60 p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">{label}</p>
          <p className="mt-1 text-sm font-semibold text-slate-800">{arm.name}</p>
        </div>
        <span className={`inline-flex items-center gap-1.5 text-[10px] font-semibold ${arm.connected ? "text-emerald-600" : "text-rose-600"}`}>
          <span className={`h-1.5 w-1.5 rounded-full ${arm.connected ? "bg-emerald-500" : "bg-rose-500"}`} />
          {arm.connected ? "Live" : "Offline"}
        </span>
      </div>
      <div className="mt-4 grid grid-cols-6 gap-1">
        {arm.joints.map((joint, index) => (
          <div key={joint.name} title={joint.name.replaceAll("_", " ")} className="rounded-md bg-white px-1 py-2 text-center ring-1 ring-slate-100">
            <p className="text-[8px] font-semibold text-slate-400">J{index + 1}</p>
            <p className="mt-0.5 font-mono text-[10px] font-semibold text-slate-700">{degrees(joint.position_radians)}</p>
          </div>
        ))}
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-[10px] text-slate-500">
        <span className="inline-flex items-center gap-1"><Activity className="h-3 w-3 text-slate-400" />{arm.control_loop.frequency_hz.toFixed(0)} Hz</span>
        <span className="inline-flex items-center gap-1"><Grip className="h-3 w-3 text-slate-400" />{(arm.gripper.position * 100).toFixed(0)}% open</span>
        <span className="font-mono text-slate-400">XYZ {(arm.pose.x_m * 1_000).toFixed(0)} / {(arm.pose.y_m * 1_000).toFixed(0)} / {(arm.pose.z_m * 1_000).toFixed(0)} mm</span>
      </div>
    </article>
  );
}

function LivePairState({ recording, arms }: { recording: Recording; arms: ArmTelemetry[] }) {
  const leader = arms.find((arm) => arm.id === recording.leader_robot_id);
  const follower = arms.find((arm) => arm.id === recording.follower_robot_id);
  return (
    <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-panel sm:p-6">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Bot className="h-4 w-4 text-slate-400" strokeWidth={1.8} />
          <h2 className="text-sm font-semibold text-slate-900">Live pair state</h2>
        </div>
        <span className="inline-flex items-center gap-1.5 text-[10px] font-medium text-slate-400">
          <Wifi className="h-3 w-3" /> WebSocket telemetry
        </span>
      </div>
      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        <CompactArmState arm={leader} label="Leader" />
        <CompactArmState arm={follower} label="Follower" />
      </div>
    </section>
  );
}

function SessionControls({
  recording,
  state,
  busy,
  onStartTeleop,
  onStopTeleop,
  onStartEpisode,
  onStopEpisode,
}: {
  recording: Recording | null;
  state: RecordingState | null;
  busy: boolean;
  onStartTeleop: () => Promise<RecordingState | null>;
  onStopTeleop: () => Promise<RecordingState | null>;
  onStartEpisode: (payload: { metadata?: { operator?: string; notes?: string } }) => Promise<RecordingState | null>;
  onStopEpisode: (payload: { success?: boolean; notes?: string }) => Promise<RecordingState | null>;
}) {
  const [operator, setOperator] = useState("");
  const [notes, setNotes] = useState("");
  const [success, setSuccess] = useState(true);
  const stateReady = recording !== null && state !== null;

  async function beginEpisode() {
    const metadata: { operator?: string; notes?: string } = {};
    if (operator.trim()) metadata.operator = operator.trim();
    if (notes.trim()) metadata.notes = notes.trim();
    await onStartEpisode({ metadata });
  }

  async function finishEpisode() {
    const updated = await onStopEpisode({ success, notes: notes.trim() || undefined });
    if (updated) {
      setNotes("");
      setSuccess(true);
    }
  }

  return (
    <section className="rounded-xl border border-slate-200 bg-white shadow-panel">
      <div className="border-b border-slate-100 px-5 py-4">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-400">Active session</p>
            <h2 className="mt-1 truncate text-sm font-semibold text-slate-900">{recording?.name ?? "No session selected"}</h2>
          </div>
          {state && <StatusBadge status={state.status} />}
        </div>
        {recording && <p className="mt-2 line-clamp-2 text-xs leading-5 text-slate-500">{recording.task}</p>}
      </div>

      <div className="grid grid-cols-3 divide-x divide-slate-100 border-b border-slate-100">
        <div className="px-3 py-4 text-center">
          <Clock3 className="mx-auto h-3.5 w-3.5 text-slate-400" />
          <p className="mt-2 font-mono text-base font-semibold text-slate-800">
            {formatDuration((recording?.duration_seconds ?? 0) + (state?.episode_active ? state.episode_duration_seconds : 0))}
          </p>
          <p className="mt-0.5 text-[9px] font-semibold uppercase tracking-wider text-slate-400">Session time</p>
        </div>
        <div className="px-3 py-4 text-center">
          <Video className="mx-auto h-3.5 w-3.5 text-slate-400" />
          <p className="mt-2 font-mono text-base font-semibold text-slate-800">{state?.episode_count ?? recording?.episode_count ?? 0}</p>
          <p className="mt-0.5 text-[9px] font-semibold uppercase tracking-wider text-slate-400">Episodes</p>
        </div>
        <div className="px-3 py-4 text-center">
          <Radio className={`mx-auto h-3.5 w-3.5 ${state?.teleop_active ? "text-blue-500" : "text-slate-400"}`} />
          <p className="mt-2 text-xs font-semibold text-slate-800">{state?.teleop_active ? "Active" : "Stopped"}</p>
          <p className="mt-1 text-[9px] font-semibold uppercase tracking-wider text-slate-400">Teleop</p>
        </div>
      </div>

      <div className="space-y-5 p-5">
        <div>
          <div className="flex items-center justify-between">
            <p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-500">1 · Teleoperation</p>
            {state?.teleop_active && <span className="text-[10px] font-medium text-blue-600">Leader mirroring</span>}
          </div>
          <button
            type="button"
            disabled={!stateReady || busy || state?.episode_active}
            onClick={() => void (state?.teleop_active ? onStopTeleop() : onStartTeleop())}
            className={`mt-2 inline-flex w-full items-center justify-center gap-2 rounded-lg px-3 py-2.5 text-xs font-semibold transition disabled:cursor-not-allowed disabled:opacity-40 ${state?.teleop_active ? "border border-slate-200 bg-white text-slate-700 hover:bg-slate-50" : "bg-ink text-white hover:bg-slate-700"}`}
          >
            {busy ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : state?.teleop_active ? <Square className="h-3 w-3" /> : <Play className="h-3.5 w-3.5" />}
            {state?.teleop_active ? "Stop teleop" : "Start teleop"}
          </button>
          {state?.episode_active && <p className="mt-1.5 text-[10px] text-amber-600">Stop the active episode before stopping teleop.</p>}
        </div>

        <div className="border-t border-slate-100 pt-5">
          <div className="flex items-center justify-between">
            <p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-500">2 · Episode</p>
            {state?.episode_active && <span className="inline-flex items-center gap-1.5 font-mono text-[10px] font-semibold text-rose-600"><span className="h-1.5 w-1.5 animate-pulse rounded-full bg-rose-500" />EP {(state.current_episode_index ?? 0) + 1} · REC {formatDuration(state.episode_duration_seconds)}</span>}
          </div>
          <div className="mt-3 grid gap-3">
            <label className="block text-[10px] font-medium uppercase tracking-wider text-slate-400">
              Operator
              <input
                value={operator}
                onChange={(event) => setOperator(event.target.value)}
                disabled={state?.episode_active || busy}
                placeholder="Optional"
                className="mt-1.5 w-full rounded-lg border border-slate-200 px-3 py-2 text-xs normal-case tracking-normal text-slate-700 outline-none focus:border-brand-500 disabled:bg-slate-50"
              />
            </label>
            <label className="block text-[10px] font-medium uppercase tracking-wider text-slate-400">
              Episode notes
              <textarea
                rows={2}
                value={notes}
                onChange={(event) => setNotes(event.target.value)}
                placeholder="Variation, reset, or outcome notes"
                className="mt-1.5 w-full resize-none rounded-lg border border-slate-200 px-3 py-2 text-xs normal-case leading-5 tracking-normal text-slate-700 outline-none focus:border-brand-500"
              />
            </label>
            {state?.episode_active && (
              <label className="flex cursor-pointer items-center gap-2 text-xs text-slate-600">
                <input type="checkbox" checked={success} onChange={(event) => setSuccess(event.target.checked)} className="h-3.5 w-3.5 rounded border-slate-300 text-brand-600 focus:ring-brand-500" />
                Mark episode successful
              </label>
            )}
          </div>
          <button
            type="button"
            disabled={!stateReady || busy || (!state?.teleop_active && !state?.episode_active)}
            onClick={() => void (state?.episode_active ? finishEpisode() : beginEpisode())}
            className={`mt-3 inline-flex w-full items-center justify-center gap-2 rounded-lg px-3 py-2.5 text-xs font-semibold text-white transition disabled:cursor-not-allowed disabled:opacity-40 ${state?.episode_active ? "bg-rose-600 hover:bg-rose-700" : "bg-brand-600 hover:bg-brand-700"}`}
          >
            {busy ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : state?.episode_active ? <Square className="h-3 w-3" /> : <Radio className="h-3.5 w-3.5" />}
            {state?.episode_active ? "Stop & save episode" : "Record episode"}
          </button>
          {!state?.teleop_active && !state?.episode_active && recording && <p className="mt-1.5 text-[10px] text-slate-400">Start teleop before recording an episode.</p>}
        </div>
      </div>
    </section>
  );
}

function RecentSessions({
  recordings,
  selectedId,
  loading,
  locked,
  onSelect,
  onRefresh,
}: {
  recordings: Recording[];
  selectedId: string;
  loading: boolean;
  locked: boolean;
  onSelect: (id: string) => void;
  onRefresh: () => void;
}) {
  return (
    <section className="rounded-xl border border-slate-200 bg-white shadow-panel">
      <div className="flex items-center justify-between border-b border-slate-100 px-5 py-4 sm:px-6">
        <div>
          <h2 className="text-sm font-semibold text-slate-900">Recent sessions</h2>
          <p className="mt-1 text-xs text-slate-400">Resume or inspect saved recording sessions.</p>
        </div>
        <button type="button" onClick={onRefresh} disabled={loading} aria-label="Refresh sessions" className="grid h-8 w-8 place-items-center rounded-lg border border-slate-200 text-slate-500 hover:bg-slate-50 disabled:opacity-40">
          <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
        </button>
      </div>
      {recordings.length === 0 ? (
        <div className="px-5 py-8 text-center text-sm text-slate-400 sm:px-6">No recording sessions yet.</div>
      ) : (
        <div className="divide-y divide-slate-100">
          {recordings.slice(0, 8).map((recording) => (
            <button
              type="button"
              key={recording.id}
              onClick={() => onSelect(recording.id)}
              disabled={locked && selectedId !== recording.id}
              className={`grid w-full grid-cols-[1fr_auto] items-center gap-4 px-5 py-4 text-left transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-45 sm:px-6 ${selectedId === recording.id ? "bg-blue-50/50" : ""}`}
            >
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <p className="truncate text-sm font-semibold text-slate-800">{recording.name}</p>
                  <StatusBadge status={recording.status} />
                </div>
                <p className="mt-1 truncate text-xs text-slate-500">{recording.task}</p>
                <p className="mt-1.5 text-[10px] text-slate-400">{shortDate(recording.created_at)} · {recording.episode_count} episode{recording.episode_count === 1 ? "" : "s"} · {formatDuration(recording.duration_seconds)}</p>
              </div>
              <ChevronRight className={`h-4 w-4 ${selectedId === recording.id ? "text-brand-500" : "text-slate-300"}`} />
            </button>
          ))}
        </div>
      )}
    </section>
  );
}

export function RecordPage() {
  const { arms, connectionState } = useArms();
  const {
    recordings,
    selectedRecording,
    selectedId,
    setSelectedId,
    state,
    loading,
    error,
    activeAction,
    refreshRecordings,
    createRecording,
    startTeleop,
    stopTeleop,
    startEpisode,
    stopEpisode,
  } = useRecordings();
  const busy = activeAction !== null;
  const sessionLocked = state?.teleop_active === true || state?.episode_active === true;

  const activePair = useMemo(() => {
    if (!selectedRecording) return [];
    return arms.filter(
      (arm) =>
        arm.id === selectedRecording.leader_robot_id ||
        arm.id === selectedRecording.follower_robot_id,
    );
  }, [arms, selectedRecording]);

  return (
    <div className="mx-auto max-w-7xl px-5 py-8 sm:px-8 lg:px-10 lg:py-10">
      <header className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-brand-600">Demonstrations</p>
          <h1 className="mt-2 text-2xl font-semibold tracking-tight text-slate-950 sm:text-3xl">Record / Teleop</h1>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500">Operate a YAM leader/follower pair and capture structured episodes.</p>
        </div>
        <span className={`inline-flex self-start items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-semibold ${connectionState === "live" ? "bg-emerald-50 text-emerald-700" : "bg-amber-50 text-amber-700"}`}>
          {connectionState === "live" ? <Wifi className="h-3.5 w-3.5" /> : <WifiOff className="h-3.5 w-3.5" />}
          {connectionState === "live" ? "Robot state live" : "Robot state reconnecting"}
        </span>
      </header>

      {error && (
        <div role="alert" className="mt-6 flex items-start gap-2 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      <div className="mt-7">
        <SessionSetup arms={arms} disabled={busy || sessionLocked} onCreate={createRecording} />
      </div>

      <div className="mt-5 grid items-start gap-5 xl:grid-cols-[minmax(0,1.55fr)_minmax(320px,0.75fr)]">
        <div className="space-y-5">
          <CameraFeed />
          {selectedRecording ? (
            <LivePairState recording={selectedRecording} arms={activePair} />
          ) : (
            <section className="grid min-h-36 place-items-center rounded-xl border border-dashed border-slate-200 bg-white px-5 text-center">
              <div>
                <Bot className="mx-auto h-5 w-5 text-slate-300" />
                <p className="mt-2 text-xs font-medium text-slate-500">Create or select a session to monitor its arm pair.</p>
              </div>
            </section>
          )}
        </div>
        <SessionControls
          key={selectedRecording?.id ?? "empty"}
          recording={selectedRecording}
          state={state}
          busy={busy}
          onStartTeleop={startTeleop}
          onStopTeleop={stopTeleop}
          onStartEpisode={startEpisode}
          onStopEpisode={stopEpisode}
        />
      </div>

      <div className="mt-5">
        <RecentSessions recordings={recordings} selectedId={selectedId} loading={loading} locked={busy || sessionLocked} onSelect={setSelectedId} onRefresh={() => void refreshRecordings()} />
      </div>
    </div>
  );
}
