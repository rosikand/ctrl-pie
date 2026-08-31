import {
  Camera,
  ChevronRight,
  LoaderCircle,
  Play,
  Plus,
  Radio,
  RefreshCw,
  Square,
  Video,
  WifiOff,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import type { FormEvent } from "react";

import { DatasetUploadForm } from "../components/DatasetUploadForm";
import { Page, PageHeader, PageSection } from "../components/layout/Page";
import { Alert } from "../components/ui/Alert";
import { Badge, StatusDot } from "../components/ui/Badge";
import { Button } from "../components/ui/Button";
import { Checkbox, Field, InlineCheckbox, Select, TextArea, TextInput } from "../components/ui/Form";
import { Disclosure, DisclosureGroup } from "../components/ui/Disclosure";
import { Drawer } from "../components/ui/Drawer";
import { EmptyState } from "../components/ui/EmptyState";
import { Panel, PanelHeader, SectionHeading } from "../components/ui/Panel";
import {
  RowButton,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
} from "../components/ui/Table";
import { useArms } from "../hooks/useArms";
import { usePublicSettings } from "../hooks/usePublicSettings";
import { useRecordings } from "../hooks/useRecordings";
import {
  degrees,
  formatDateTime,
  formatDuration,
  formatRelative,
  suggestedRepoName,
} from "../lib/format";
import { isBusyStatus, recordingTone } from "../lib/status";
import type { ArmTelemetry } from "../types/arms";
import type {
  CreateRecordingRequest,
  Recording,
  RecordingState,
  UploadRecordingRequest,
  UploadRecordingResponse,
} from "../types/recordings";
import { TelemetryBadge } from "./RobotsPage";

type StoredUploadTarget = {
  repoId: string;
  namespace: string;
  repoName: string;
};

/** A failed upload must retry its original repository target. */
function storedUploadTarget(recording: Recording | null): StoredUploadTarget | null {
  if (!recording || recording.status !== "failed") return null;
  const upload = recording.metadata.upload;
  if (!upload || typeof upload !== "object" || Array.isArray(upload)) return null;
  const repoId = (upload as Record<string, unknown>).repo_id;
  if (typeof repoId !== "string") return null;
  const separator = repoId.indexOf("/");
  if (separator < 1 || separator !== repoId.lastIndexOf("/")) return null;
  const storedNamespace = repoId.slice(0, separator);
  const storedRepoName = repoId.slice(separator + 1);
  if (!storedNamespace || !storedRepoName) return null;
  return { repoId, namespace: storedNamespace, repoName: storedRepoName };
}

function CameraFeed() {
  const [imageKey, setImageKey] = useState(0);
  const [cameraState, setCameraState] = useState<"loading" | "live" | "error">("loading");

  return (
    <figure className="overflow-hidden rounded-xl border border-line bg-surface">
      <div className="flex items-center justify-between border-b border-line px-5 py-3">
        <div className="flex items-center gap-2">
          <Camera className="h-4 w-4 text-ink-faint" aria-hidden="true" />
          <h2 className="text-sm font-semibold tracking-tight text-ink">Workspace camera</h2>
        </div>
        <Badge
          tone={cameraState === "live" ? "success" : cameraState === "error" ? "danger" : "neutral"}
          dot
          pulse={cameraState !== "error"}
        >
          {cameraState === "live" ? "Live" : cameraState}
        </Badge>
      </div>
      <div className="relative aspect-[4/3] bg-ink">
        <img
          key={imageKey}
          src={`/api/camera/stream?connection=${imageKey}`}
          alt="Live workspace camera view"
          onLoad={() => setCameraState("live")}
          onError={() => setCameraState("error")}
          className={`h-full w-full object-contain transition-opacity ${
            cameraState === "live" ? "opacity-100" : "opacity-20"
          }`}
        />
        {cameraState !== "live" && (
          <div className="absolute inset-0 grid place-items-center text-center">
            <div>
              {cameraState === "error" ? (
                <WifiOff className="mx-auto h-6 w-6 text-white/60" aria-hidden="true" />
              ) : (
                <LoaderCircle className="mx-auto h-6 w-6 animate-spin text-white/60" aria-hidden="true" />
              )}
              <p className="mt-2 text-xs font-medium text-white/80">
                {cameraState === "error" ? "Camera stream unavailable" : "Connecting to camera"}
              </p>
              {cameraState === "error" && (
                <button
                  type="button"
                  onClick={() => {
                    setCameraState("loading");
                    setImageKey((current) => current + 1);
                  }}
                  className="mt-3 text-xs font-medium text-accent-300 hover:text-accent-200"
                >
                  Reconnect
                </button>
              )}
            </div>
          </div>
        )}
      </div>
    </figure>
  );
}

function RobotReadiness({
  arm,
  role,
}: {
  arm: ArmTelemetry | undefined;
  role: "Leader" | "Follower";
}) {
  if (!arm) {
    return (
      <div className="px-5 py-4">
        <p className="text-2xs font-medium uppercase tracking-[0.08em] text-ink-faint">{role}</p>
        <p className="mt-1.5 text-xs text-ink-muted">Telemetry unavailable</p>
      </div>
    );
  }
  return (
    <div className="px-5 py-4">
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <p className="text-2xs font-medium uppercase tracking-[0.08em] text-ink-faint">{role}</p>
          <p className="mt-1 truncate text-[13px] font-medium text-ink">{arm.name}</p>
        </div>
        <span className="inline-flex items-center gap-1.5 text-2xs font-medium">
          <StatusDot tone={arm.connected ? "success" : "danger"} />
          <span className={arm.connected ? "text-positive-700" : "text-critical-700"}>
            {arm.connected ? "Connected" : "Offline"}
          </span>
        </span>
      </div>
      <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 font-mono text-2xs text-ink-muted">
        {arm.joints.slice(0, 6).map((joint, index) => (
          <span key={joint.name} title={joint.name}>
            J{index + 1} {degrees(joint.position_radians, 0)}
          </span>
        ))}
      </div>
      <p className="mt-2 text-2xs text-ink-faint">
        {arm.control_loop.frequency_hz.toFixed(0)} Hz · gripper{" "}
        {(arm.gripper.position * 100).toFixed(0)}% open
      </p>
    </div>
  );
}

function NewSessionDrawer({
  open,
  onClose,
  arms,
  disabled,
  creating,
  onCreate,
}: {
  open: boolean;
  onClose: () => void;
  arms: ArmTelemetry[];
  disabled: boolean;
  creating: boolean;
  onCreate: (payload: CreateRecordingRequest) => Promise<Recording | null>;
}) {
  const pairs = useMemo(() => {
    const byPair = new Map<string, ArmTelemetry[]>();
    arms.forEach((arm) => {
      if (arm.pair_id) byPair.set(arm.pair_id, [...(byPair.get(arm.pair_id) ?? []), arm]);
    });
    const declared = [...byPair.entries()].flatMap(([pairId, pairArms]) => {
      const leaders = pairArms.filter((arm) => arm.role === "leader");
      const followers = pairArms.filter((arm) => arm.role === "follower");
      if (leaders.length !== 1 || followers.length !== 1) return [];
      return [{ key: pairId, pairId, leader: leaders[0], follower: followers[0] }];
    });
    if (declared.length > 0) return declared;
    // V1.1 compatibility: its only leader and follower had no pair metadata.
    const leaders = arms.filter((arm) => arm.role === "leader");
    const followers = arms.filter((arm) => arm.role === "follower");
    return leaders.length === 1 && followers.length === 1
      ? [{ key: "legacy-pair", pairId: null, leader: leaders[0], follower: followers[0] }]
      : [];
  }, [arms]);

  const [name, setName] = useState("");
  const [task, setTask] = useState("");
  const [pairKey, setPairKey] = useState("");
  const [operator, setOperator] = useState("");
  const [notes, setNotes] = useState("");

  useEffect(() => {
    if (!pairs.some((pair) => pair.key === pairKey)) setPairKey(pairs[0]?.key ?? "");
  }, [pairKey, pairs]);

  const selectedPair = pairs.find((pair) => pair.key === pairKey) ?? null;
  const canSubmit =
    name.trim().length > 0 &&
    task.trim().length > 0 &&
    selectedPair !== null &&
    selectedPair.leader.connected &&
    selectedPair.follower.connected;

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const metadata: Record<string, unknown> = {};
    if (operator.trim()) metadata.operator = operator.trim();
    if (notes.trim()) metadata.notes = notes.trim();
    if (selectedPair?.pairId) metadata.pair_id = selectedPair.pairId;
    if (selectedPair?.leader.group_id) metadata.group_id = selectedPair.leader.group_id;
    if (selectedPair?.leader.side) metadata.side = selectedPair.leader.side;
    const created = await onCreate({
      name: name.trim(),
      task: task.trim(),
      leader_robot_id: selectedPair?.leader.id ?? "",
      follower_robot_id: selectedPair?.follower.id ?? "",
      metadata,
    });
    if (created) {
      setName("");
      setTask("");
      setNotes("");
      onClose();
    }
  }

  return (
    <Drawer
      open={open}
      onClose={onClose}
      title="New recording session"
      description="Choose one declared YAM leader/follower pair and describe the task."
    >
      <form id="new-session" onSubmit={submit} className="space-y-5">
        <Field label="Session name">
          <TextInput
            required
            value={name}
            disabled={disabled}
            placeholder="Cup stacking demos"
            onChange={(event) => setName(event.target.value)}
          />
        </Field>

        <Field label="Task">
          <TextArea
            required
            rows={3}
            value={task}
            disabled={disabled}
            placeholder="Pick up the blue cup and place it on the marked target."
            onChange={(event) => setTask(event.target.value)}
          />
        </Field>

        <Field
          label="Declared leader / follower pair"
          hint="Pair routing comes from saved cell metadata. Cross-pair leader/follower combinations cannot be selected."
          error={
            selectedPair && (!selectedPair.leader.connected || !selectedPair.follower.connected)
              ? "Connect both arms in this pair before creating a session."
              : undefined
          }
        >
          <Select
            required
            value={pairKey}
            disabled={disabled}
            onChange={(event) => setPairKey(event.target.value)}
          >
            {pairs.length === 0 && <option value="">No complete declared pair</option>}
            {pairs.map((pair) => (
              <option key={pair.key} value={pair.key}>
                {pair.pairId ?? "Legacy pair"} · {pair.leader.name} → {pair.follower.name}
                {pair.leader.side ? ` · ${pair.leader.side}` : ""}
                {pair.leader.group_id ? ` · ${pair.leader.group_id}` : ""}
              </option>
            ))}
          </Select>
        </Field>

        <Field label="Operator" optional>
          <TextInput
            value={operator}
            disabled={disabled}
            placeholder="Operator name"
            onChange={(event) => setOperator(event.target.value)}
          />
        </Field>

        <Field label="Session notes" optional>
          <TextInput
            value={notes}
            disabled={disabled}
            placeholder="Lighting, fixture, or reset details"
            onChange={(event) => setNotes(event.target.value)}
          />
        </Field>

        <Button
          type="submit"
          variant="primary"
          icon={Plus}
          fullWidth
          loading={creating}
          disabled={disabled || !canSubmit}
        >
          Create session
        </Button>
      </form>
    </Drawer>
  );
}

function SessionControls({
  recording,
  state,
  busy,
  uploadBusy,
  uploadError,
  namespace,
  namespaceError,
  onRetryNamespace,
  onStartTeleop,
  onStopTeleop,
  onEnableSync,
  onDisableSync,
  onStartEpisode,
  onStopEpisode,
  onUpload,
}: {
  recording: Recording;
  state: RecordingState | null;
  busy: boolean;
  uploadBusy: boolean;
  uploadError: string | null;
  namespace: string | null | undefined;
  namespaceError: string | null;
  onRetryNamespace: () => void;
  onStartTeleop: () => Promise<RecordingState | null>;
  onStopTeleop: () => Promise<RecordingState | null>;
  onEnableSync: () => Promise<RecordingState | null>;
  onDisableSync: () => Promise<RecordingState | null>;
  onStartEpisode: (payload: {
    metadata?: { operator?: string; notes?: string };
  }) => Promise<RecordingState | null>;
  onStopEpisode: (payload: { success?: boolean; notes?: string }) => Promise<RecordingState | null>;
  onUpload: (payload: UploadRecordingRequest) => Promise<UploadRecordingResponse | null>;
}) {
  const [operator, setOperator] = useState("");
  const [notes, setNotes] = useState("");
  const [success, setSuccess] = useState(true);
  const [syncConfirmed, setSyncConfirmed] = useState(false);
  const [uploadResult, setUploadResult] = useState<UploadRecordingResponse | null>(null);

  const lifecycleStatus = state?.status ?? recording.status;
  const lifecycleClosed =
    lifecycleStatus === "uploading" || lifecycleStatus === "uploaded" || lifecycleStatus === "failed";
  const episodeCount = state?.episode_count ?? recording.episode_count;
  const inactive = state !== null && !state.teleop_active && !state.episode_active;
  const retryTarget = storedUploadTarget(recording);
  const namespaceMismatch =
    retryTarget && namespace && retryTarget.namespace !== namespace
      ? `This retry target belongs to ${retryTarget.namespace}, but HF_NAMESPACE is ${namespace}. Restore the original namespace to retry.`
      : null;
  const uploadAllowed =
    inactive && episodeCount > 0 && (lifecycleStatus === "ready" || lifecycleStatus === "failed");
  // One primary action at a time: the next step in the capture chain, or the
  // upload once the chain has nothing left to offer.
  const nextStep: "teleop" | "sync" | "episode" | "upload" | null = lifecycleClosed
    ? uploadAllowed
      ? "upload"
      : null
    : !state || !state.teleop_active
      ? "teleop"
      : state.sync_in_progress
        ? null
        : !state.sync_enabled
          ? "sync"
          : !state.episode_active
            ? "episode"
            : null;

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

  const uploadedRepoId = uploadResult?.repo_id ?? recording.hf_repo_id;

  return (
    <div className="space-y-5">
      <Panel>
        <PanelHeader
          title={recording.name}
          description={recording.task}
          actions={
            <Badge
              tone={recordingTone[lifecycleStatus]}
              dot
              pulse={isBusyStatus(lifecycleStatus)}
            >
              {lifecycleStatus}
            </Badge>
          }
        />
        <dl className="grid grid-cols-3 divide-x divide-line border-b border-line">
          <div className="px-4 py-3 text-center">
            <dt className="text-2xs font-medium uppercase tracking-[0.08em] text-ink-faint">
              Session
            </dt>
            <dd className="mt-1.5 font-mono text-sm font-medium text-ink">
              {formatDuration(
                recording.duration_seconds + (state?.episode_active ? state.episode_duration_seconds : 0),
              )}
            </dd>
          </div>
          <div className="px-4 py-3 text-center">
            <dt className="text-2xs font-medium uppercase tracking-[0.08em] text-ink-faint">
              Episodes
            </dt>
            <dd className="mt-1.5 font-mono text-sm font-medium text-ink">{episodeCount}</dd>
          </div>
          <div className="px-4 py-3 text-center">
            <dt className="text-2xs font-medium uppercase tracking-[0.08em] text-ink-faint">
              Teleop
            </dt>
            <dd className="mt-1.5 text-sm font-medium text-ink">
              {state?.teleop_active ? "Active" : "Stopped"}
            </dd>
          </div>
        </dl>

        <div className="space-y-5 px-5 py-5">
          {/* 1 · Observation-only teleoperation. */}
          <div>
            <div className="flex items-center justify-between gap-2">
              <p className="text-2xs font-medium uppercase tracking-[0.08em] text-ink-faint">
                1 · Teleoperation
              </p>
              {state?.teleop_active && (
                <span
                  className={`text-2xs font-medium ${
                    state.sync_enabled ? "text-positive-700" : "text-caution-700"
                  }`}
                >
                  {state.sync_enabled ? "Sync enabled" : "Observing · sync disabled"}
                </span>
              )}
            </div>
            <Button
              className="mt-2"
              fullWidth
              variant={nextStep === "teleop" ? "primary" : "secondary"}
              icon={state?.teleop_active ? Square : Play}
              loading={busy}
              disabled={
                busy ||
                state === null ||
                state.episode_active ||
                state.sync_enabled ||
                state.sync_in_progress ||
                lifecycleClosed
              }
              onClick={() => void (state?.teleop_active ? onStopTeleop() : onStartTeleop())}
            >
              {state?.teleop_active ? "Stop teleop" : "Start teleop (sync off)"}
            </Button>
            {!state?.teleop_active && !lifecycleClosed && (
              <p className="mt-2 text-2xs leading-5 text-ink-muted">
                Starting teleop only observes the declared pair. It makes zero follower writes and
                synchronization begins disabled.
              </p>
            )}
            {state?.sync_enabled && (
              <p className="mt-2 text-2xs text-caution-700">
                Disable sync cleanly before stopping teleop.
              </p>
            )}
            {state?.episode_active && (
              <p className="mt-2 text-2xs text-caution-700">
                Stop the active episode before stopping teleop.
              </p>
            )}
            {lifecycleClosed && (
              <p className="mt-2 text-2xs leading-5 text-ink-muted">
                {lifecycleStatus === "uploaded"
                  ? "This session is finalized on Hugging Face."
                  : lifecycleStatus === "uploading"
                    ? "Robot controls are locked while the dataset uploads."
                    : "Recording controls are locked after a failed pipeline step; retry the upload below."}
              </p>
            )}
          </div>

          {/* 2 · Explicit follower-motion boundary. */}
          <div className="border-t border-line pt-5">
            <div className="flex items-center justify-between gap-2">
              <p className="text-2xs font-medium uppercase tracking-[0.08em] text-ink-faint">
                2 · Pair synchronization
              </p>
              {state?.sync_in_progress && (
                <span className="inline-flex items-center gap-1 text-2xs font-medium text-caution-700">
                  <LoaderCircle className="h-3 w-3 animate-spin" aria-hidden="true" />
                  Slow correction
                </span>
              )}
            </div>
            {state?.teleop_active ? (
              <>
                <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1.5 font-mono text-2xs text-ink-muted">
                  {Object.entries(state.joint_deltas_radians).map(([joint, delta], index) => (
                    <span key={joint} title={joint}>
                      J{index + 1} {degrees(delta, 0)}
                    </span>
                  ))}
                </div>
                {!state.sync_enabled && !state.sync_in_progress && (
                  <Checkbox
                    className="mt-3"
                    tone="warning"
                    checked={syncConfirmed}
                    onChange={(event) => setSyncConfirmed(event.target.checked)}
                    label="I inspected the live leader/follower deltas and cleared the workspace."
                    description="Enabling sync will move the follower slowly toward the leader over approximately 3 seconds."
                  />
                )}
                <Button
                  className="mt-3"
                  fullWidth
                  variant={
                    state.sync_enabled || state.sync_in_progress
                      ? "dangerSubtle"
                      : nextStep === "sync"
                        ? "primary"
                        : "secondary"
                  }
                  icon={state.sync_enabled || state.sync_in_progress ? Square : Play}
                  disabled={
                    busy || (!state.sync_enabled && !state.sync_in_progress && !syncConfirmed)
                  }
                  onClick={() =>
                    void (state.sync_enabled || state.sync_in_progress
                      ? onDisableSync().then((updated) => {
                          if (updated) setSyncConfirmed(false);
                        })
                      : onEnableSync())
                  }
                >
                  {state.sync_in_progress
                    ? "Stop correction now"
                    : state.sync_enabled
                      ? "Disable sync"
                      : "Enable slow sync"}
                </Button>
              </>
            ) : (
              <p className="mt-2 text-2xs leading-5 text-ink-muted">
                Start teleop to inspect live deltas. No follower command is sent until this separate
                boundary is acknowledged.
              </p>
            )}
          </div>

          {/* 3 · Episode capture. */}
          <div className="border-t border-line pt-5">
            <div className="flex items-center justify-between gap-2">
              <p className="text-2xs font-medium uppercase tracking-[0.08em] text-ink-faint">
                3 · Episode
              </p>
              {state?.episode_active && (
                <span className="inline-flex items-center gap-1.5 font-mono text-2xs font-medium text-critical-700">
                  <StatusDot tone="danger" pulse />
                  EP {(state.current_episode_index ?? 0) + 1} ·{" "}
                  {formatDuration(state.episode_duration_seconds)}
                </span>
              )}
            </div>

            {state?.episode_active && (
              <InlineCheckbox
                className="mt-3"
                label="Mark episode successful"
                checked={success}
                onChange={(event) => setSuccess(event.target.checked)}
              />
            )}

            <Button
              className="mt-3"
              fullWidth
              size="lg"
              variant={
                state?.episode_active ? "danger" : nextStep === "episode" ? "primary" : "secondary"
              }
              icon={state?.episode_active ? Square : Radio}
              loading={busy}
              disabled={
                busy ||
                state === null ||
                lifecycleClosed ||
                (!state.sync_enabled && !state.episode_active) ||
                state.sync_in_progress
              }
              onClick={() => void (state?.episode_active ? finishEpisode() : beginEpisode())}
            >
              {state?.episode_active ? "Stop & save episode" : "Record episode"}
            </Button>

            {!lifecycleClosed && !state?.sync_enabled && !state?.episode_active && (
              <p className="mt-2 text-2xs leading-5 text-ink-muted">
                Start teleop, inspect alignment, and explicitly enable sync before recording an
                episode.
              </p>
            )}

            <div className="mt-3">
              <DisclosureGroup className="px-4">
                <Disclosure title="Episode metadata" meta="optional">
                  <div className="space-y-3">
                    <Field label="Operator" optional>
                      <TextInput
                        value={operator}
                        disabled={state?.episode_active || busy || lifecycleClosed}
                        onChange={(event) => setOperator(event.target.value)}
                      />
                    </Field>
                    <Field label="Episode notes" optional>
                      <TextArea
                        rows={2}
                        value={notes}
                        disabled={busy || lifecycleClosed}
                        placeholder="Variation, reset, or outcome notes"
                        onChange={(event) => setNotes(event.target.value)}
                      />
                    </Field>
                  </div>
                </Disclosure>
              </DisclosureGroup>
            </div>
          </div>

          {/* 4 · Publish to Hugging Face. */}
          <div className="border-t border-line pt-5">
            <p className="text-2xs font-medium uppercase tracking-[0.08em] text-ink-faint">
              4 · Hugging Face
            </p>
            <div className="mt-3">
              {lifecycleStatus === "uploaded" && uploadedRepoId && !uploadResult ? (
                <Alert tone="success" title="Dataset uploaded" role="status">
                  <p className="font-mono text-2xs">{uploadedRepoId}</p>
                  <p className="mt-1">Create a new session to collect additional episodes.</p>
                </Alert>
              ) : uploadBusy || lifecycleStatus === "uploading" ? (
                <Alert tone="info" role="status" title="Packaging and uploading dataset…">
                  Keep this page open while episodes are converted to LeRobot format and sent to the
                  Hub.
                </Alert>
              ) : (
                <DatasetUploadForm
                  key={recording.id}
                  namespace={namespace}
                  namespaceError={namespaceError}
                  onRetryNamespace={onRetryNamespace}
                  defaultRepoName={retryTarget?.repoName ?? suggestedRepoName(recording.name)}
                  lockedRepoName={retryTarget?.repoName ?? null}
                  lockedNote={
                    retryTarget && !namespaceMismatch ? (
                      <Alert tone="warning" title={`Retrying ${retryTarget.repoId}`}>
                        The backend requires a failed upload to retry its original repository target.
                      </Alert>
                    ) : namespaceMismatch ? (
                      <Alert tone="danger">{namespaceMismatch}</Alert>
                    ) : null
                  }
                  disabled={busy || Boolean(namespaceMismatch)}
                  primary={nextStep === "upload"}
                  busy={uploadBusy}
                  error={uploadError}
                  result={uploadResult}
                  blockingHint={
                    episodeCount === 0
                      ? "Save at least one episode before uploading this session."
                      : !uploadAllowed
                        ? "Stop the episode and teleoperation before uploading."
                        : undefined
                  }
                  onSubmit={(payload) => {
                    void onUpload(payload).then((uploaded) => {
                      if (uploaded) setUploadResult(uploaded);
                    });
                  }}
                />
              )}
            </div>
          </div>
        </div>
      </Panel>
    </div>
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
    uploadError,
    activeAction,
    refreshRecordings,
    createRecording,
    startTeleop,
    stopTeleop,
    enableSync,
    disableSync,
    startEpisode,
    stopEpisode,
    uploadRecording,
  } = useRecordings();
  const { settings, error: settingsError, refresh: refreshSettings } = usePublicSettings();
  const [drawerOpen, setDrawerOpen] = useState(false);

  const busy = activeAction !== null;
  const sessionLocked = state?.teleop_active === true || state?.episode_active === true;

  const leader = arms.find((arm) => arm.id === selectedRecording?.leader_robot_id);
  const follower = arms.find((arm) => arm.id === selectedRecording?.follower_robot_id);

  const onRetryNamespace = useCallback(() => void refreshSettings(), [refreshSettings]);

  return (
    <Page>
      <PageHeader
        title="Record"
        description="Operate one YAM leader/follower pair and capture structured demonstration episodes."
        meta={<TelemetryBadge state={connectionState} />}
        actions={
          <Button
            icon={Plus}
            variant={selectedRecording ? "secondary" : "primary"}
            disabled={busy || sessionLocked}
            onClick={() => setDrawerOpen(true)}
          >
            New session
          </Button>
        }
      />

      {error && (
        <PageSection className="mt-8">
          <Alert tone="danger">{error}</Alert>
        </PageSection>
      )}

      <PageSection className="mt-8">
        <div className="grid items-start gap-6 xl:grid-cols-[minmax(0,1fr)_24rem]">
          <div className="space-y-6">
            <CameraFeed />

            {selectedRecording ? (
              <Panel>
                <PanelHeader
                  title="Robot readiness"
                  description="Live pair telemetry for the selected session."
                />
                <div className="grid divide-y divide-line sm:grid-cols-2 sm:divide-x sm:divide-y-0">
                  <RobotReadiness arm={leader} role="Leader" />
                  <RobotReadiness arm={follower} role="Follower" />
                </div>
              </Panel>
            ) : (
              <EmptyState
                icon={Video}
                title="No session selected"
                description="Create a recording session to monitor its pair and capture episodes."
                action={
                  <Button variant="primary" icon={Plus} onClick={() => setDrawerOpen(true)}>
                    New session
                  </Button>
                }
              />
            )}
          </div>

          {selectedRecording ? (
            <SessionControls
              key={selectedRecording.id}
              recording={selectedRecording}
              state={state}
              busy={busy}
              uploadBusy={activeAction === "upload"}
              uploadError={uploadError}
              namespace={settingsError ? null : settings ? settings.hf_namespace : undefined}
              namespaceError={settingsError}
              onRetryNamespace={onRetryNamespace}
              onStartTeleop={startTeleop}
              onStopTeleop={stopTeleop}
              onEnableSync={enableSync}
              onDisableSync={disableSync}
              onStartEpisode={startEpisode}
              onStopEpisode={stopEpisode}
              onUpload={uploadRecording}
            />
          ) : null}
        </div>
      </PageSection>

      <PageSection>
        <SectionHeading
          title="Sessions"
          description="Resume or inspect saved recording sessions."
          className="mb-4"
          actions={
            <Button
              size="sm"
              icon={RefreshCw}
              loading={loading}
              disabled={loading}
              onClick={() => void refreshRecordings()}
            >
              Refresh
            </Button>
          }
        />
        {recordings.length === 0 ? (
          <EmptyState
            icon={Video}
            title="No recording sessions yet"
            description="Sessions you create appear here with their episode counts and upload state."
          />
        ) : (
          <Table label="Recording sessions" minWidth="48rem">
            <TableHead>
              <TableHeaderCell>Session</TableHeaderCell>
              <TableHeaderCell>Status</TableHeaderCell>
              <TableHeaderCell align="right">Episodes</TableHeaderCell>
              <TableHeaderCell align="right">Duration</TableHeaderCell>
              <TableHeaderCell>Dataset</TableHeaderCell>
              <TableHeaderCell align="right">Created</TableHeaderCell>
              <TableHeaderCell align="right" />
            </TableHead>
            <TableBody>
              {recordings.map((recording) => (
                <TableRow
                  key={recording.id}
                  interactive={!(busy || sessionLocked) || recording.id === selectedId}
                  selected={recording.id === selectedId}
                >
                  <TableCell>
                    <RowButton
                      disabled={(busy || sessionLocked) && recording.id !== selectedId}
                      onClick={() => setSelectedId(recording.id)}
                    >
                      {recording.name}
                    </RowButton>
                    <p className="mt-0.5 truncate text-2xs text-ink-muted">{recording.task}</p>
                  </TableCell>
                  <TableCell>
                    <Badge
                      tone={recordingTone[recording.status]}
                      dot
                      pulse={isBusyStatus(recording.status)}
                    >
                      {recording.status}
                    </Badge>
                  </TableCell>
                  <TableCell align="right">{recording.episode_count}</TableCell>
                  <TableCell align="right">{formatDuration(recording.duration_seconds)}</TableCell>
                  <TableCell mono muted className="max-w-[16rem] truncate">
                    {recording.hf_repo_id ?? "—"}
                  </TableCell>
                  <TableCell align="right" muted>
                    <span title={formatDateTime(recording.created_at)}>
                      {formatRelative(recording.created_at)}
                    </span>
                  </TableCell>
                  <TableCell align="right">
                    <ChevronRight
                      className={`h-4 w-4 ${
                        recording.id === selectedId ? "text-accent-600" : "text-ink-faint"
                      }`}
                      aria-hidden="true"
                    />
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </PageSection>

      <NewSessionDrawer
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        arms={arms}
        disabled={busy || sessionLocked}
        creating={activeAction === "create"}
        onCreate={createRecording}
      />
    </Page>
  );
}
