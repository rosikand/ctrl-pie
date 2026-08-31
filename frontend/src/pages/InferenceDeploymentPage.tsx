import {
  Activity,
  CheckCircle2,
  ExternalLink,
  Gauge,
  Play,
  RadioTower,
  RefreshCw,
  Square,
  Timer,
  TriangleAlert,
  Wifi,
  WifiOff,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useLocation, useParams } from "react-router-dom";

import { DatasetUploadForm } from "../components/DatasetUploadForm";
import { Page, PageHeader, PageSection } from "../components/layout/Page";
import { LoadErrorState } from "../components/LoadError";
import { Alert } from "../components/ui/Alert";
import { Badge, Mono } from "../components/ui/Badge";
import { Button, buttonClass } from "../components/ui/Button";
import { DescriptionList } from "../components/ui/DescriptionList";
import { Disclosure, DisclosureGroup } from "../components/ui/Disclosure";
import { Checkbox, Field, Select, TextArea, TextInput } from "../components/ui/Form";
import { Panel, PanelHeader, SectionHeading } from "../components/ui/Panel";
import { Skeleton } from "../components/ui/Skeleton";
import { Stat, StatGrid } from "../components/ui/Stat";
import { useArms } from "../hooks/useArms";
import { useDeployment } from "../hooks/useInference";
import { usePublicSettings } from "../hooks/usePublicSettings";
import { uploadRecording } from "../lib/api";
import {
  formatCount,
  formatDateTime,
  formatDecimal,
  formatDuration,
  hubRepoUrl,
  shortRevision,
  suggestedRepoName,
} from "../lib/format";
import { deploymentTone, isBusyStatus, sessionTone } from "../lib/status";
import type { DeploymentStatus } from "../types/inference";
import type { UploadRecordingResponse } from "../types/recordings";

const RUNTIME_LABELS: Record<string, string> = {
  lerobot: "LeRobot",
  openpi: "OpenPI",
  stub: "Stub",
};

function SessionRecordingUpload({
  recordingId,
  defaultName,
  namespace,
}: {
  recordingId: string;
  defaultName: string;
  namespace: string | null;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<UploadRecordingResponse | null>(null);
  const controller = useRef<AbortController | null>(null);

  useEffect(() => () => controller.current?.abort(), []);

  const submit = useCallback(
    async (payload: { repo_name: string; private: boolean }) => {
      controller.current?.abort();
      const abort = new AbortController();
      controller.current = abort;
      setBusy(true);
      setError(null);
      try {
        const response = await uploadRecording(recordingId, payload, abort.signal);
        if (!abort.signal.aborted) setResult(response);
      } catch (reason) {
        if (!abort.signal.aborted) {
          setError(reason instanceof Error ? reason.message : "Could not upload the recording.");
        }
      } finally {
        if (!abort.signal.aborted) setBusy(false);
      }
    },
    [recordingId],
  );

  return (
    <Panel>
      <PanelHeader
        title="Upload session recording"
        description="Publish the finalized inference recording as a LeRobot dataset."
      />
      <div className="px-5 py-5">
        <DatasetUploadForm
          namespace={namespace}
          defaultRepoName={suggestedRepoName(defaultName, "inference-recording")}
          busy={busy}
          error={error}
          result={result}
          onSubmit={(payload) => void submit(payload)}
        />
      </div>
    </Panel>
  );
}

export function InferenceDeploymentPage() {
  const { deploymentId = "" } = useParams<{ deploymentId: string }>();
  const location = useLocation();
  const preselectedArmId = (location.state as { armId?: string } | null)?.armId ?? "";
  const arms = useArms();
  const { settings } = usePublicSettings();
  const {
    state,
    stateLoading,
    stateError,
    operationError,
    clearOperationError,
    busy,
    connection,
    retryState,
    start,
    stop,
  } = useDeployment(deploymentId);

  const [armId, setArmId] = useState(preselectedArmId);
  const [task, setTask] = useState("");
  const [recordSession, setRecordSession] = useState(false);
  const [recordingName, setRecordingName] = useState("");
  const [operator, setOperator] = useState("");
  const [recordingNotes, setRecordingNotes] = useState("");
  const [now, setNow] = useState(() => Date.now());

  const followers = useMemo(
    () => arms.arms.filter((arm) => arm.role === "follower" && arm.connected),
    [arms.arms],
  );
  const selectedFollower = followers.find((arm) => arm.id === armId) ?? null;

  useEffect(() => {
    if (state?.arm_id && followers.some((arm) => arm.id === state.arm_id)) {
      setArmId(state.arm_id);
      return;
    }
    if (!followers.some((arm) => arm.id === armId)) setArmId(followers[0]?.id ?? "");
  }, [armId, followers, state?.arm_id]);

  useEffect(() => {
    if (state?.session_status !== "running") return;
    setNow(Date.now());
    const timer = window.setInterval(() => setNow(Date.now()), 1_000);
    return () => window.clearInterval(timer);
  }, [state?.session_status]);

  if (stateLoading && !state) {
    return (
      <Page>
        <PageHeader back={{ to: "/inference", label: "Inference" }} title="Loading deployment…" />
        <PageSection>
          <Skeleton className="h-24 w-full" />
          <Skeleton className="mt-5 h-72 w-full" />
        </PageSection>
      </Page>
    );
  }

  if (!state) {
    return (
      <Page>
        <PageHeader back={{ to: "/inference", label: "Inference" }} title="Deployment" />
        <PageSection>
          <LoadErrorState
            error={stateError ?? { message: "This deployment is unavailable.", status: 404 }}
            resource="inference"
            onRetry={retryState}
            busy={stateLoading}
          />
        </PageSection>
      </Page>
    );
  }

  const sessionSeconds = state.session_started_at
    ? Math.max(0, (now - new Date(state.session_started_at).valueOf()) / 1_000)
    : 0;
  const inferenceFrequency =
    sessionSeconds > 0 ? state.requests_completed / sessionSeconds : null;
  const activeSession =
    state.session_status === "starting" ||
    state.session_status === "running" ||
    state.session_status === "stopping";
  const sessionLocked = Boolean(busy) || activeSession;
  const canStart = Boolean(
    state.status === "running" &&
      state.endpoint_healthy &&
      state.session_status === "idle" &&
      selectedFollower &&
      task.trim() &&
      !sessionLocked,
  );
  const canStop =
    !busy &&
    (["running", "stopping", "failed"] as DeploymentStatus[]).includes(state.status) &&
    !(state.status === "failed" && state.teardown_verified);
  const recordingReady =
    state.recording.enabled &&
    state.recording.status === "ready" &&
    !state.recording.hf_repo_id &&
    state.recording.recording_id;
  const uploadedRecordingUrl = state.recording.hf_repo_id
    ? hubRepoUrl(state.recording.hf_repo_id, "dataset")
    : null;
  const streamLabel =
    connection === "live"
      ? "Live"
      : connection === "reconnecting"
        ? "Reconnecting"
        : connection === "connecting"
          ? "Connecting"
          : "Snapshot";

  async function onStart() {
    if (!selectedFollower || !task.trim()) return;
    const metadata: { operator?: string; notes?: string } = {};
    if (operator.trim()) metadata.operator = operator.trim();
    if (recordingNotes.trim()) metadata.notes = recordingNotes.trim();
    await start({
      arm_id: selectedFollower.id,
      task: task.trim(),
      record_session: recordSession,
      recording_name: recordSession && recordingName.trim() ? recordingName.trim() : null,
      ...(recordSession && Object.keys(metadata).length ? { recording_metadata: metadata } : {}),
    });
  }

  async function onStop() {
    await stop({
      recording_success: true,
      recording_notes: recordingNotes.trim() || null,
    });
  }

  return (
    <Page>
      <PageHeader
        back={{ to: "/inference", label: "Inference" }}
        title={state.name}
        meta={
          <>
            <Badge tone={deploymentTone[state.status]} dot pulse={isBusyStatus(state.status)}>
              {state.status}
            </Badge>
            <Badge
              tone={sessionTone[state.session_status]}
              dot
              pulse={isBusyStatus(state.session_status)}
            >
              {`Session ${state.session_status}`}
            </Badge>
            <span
              className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-2xs font-medium ${
                connection === "live"
                  ? "bg-positive-50 text-positive-700"
                  : connection === "reconnecting"
                    ? "bg-caution-50 text-caution-700"
                    : "bg-line-subtle text-ink-muted"
              }`}
            >
              {connection === "live" ? (
                <Wifi className="h-3 w-3" aria-hidden="true" />
              ) : (
                <WifiOff className="h-3 w-3" aria-hidden="true" />
              )}
              {streamLabel}
            </span>
            <Mono title={`${state.model_repo}@${state.checkpoint_revision ?? "default"}`}>
              {state.model_repo}@{shortRevision(state.checkpoint_revision, 10) || "default"}
            </Mono>
          </>
        }
        actions={
          <Button
            variant="dangerSubtle"
            icon={Square}
            loading={busy === "stop"}
            disabled={!canStop}
            onClick={() => void onStop()}
          >
            {busy === "stop"
              ? "Stopping and verifying…"
              : activeSession
                ? "Stop session + endpoint"
                : "Stop endpoint"}
          </Button>
        }
      />

      <PageSection className="mt-8 space-y-5">
        {operationError && (
          <Alert
            tone="danger"
            title={operationError.message}
            action={
              <Button size="sm" variant="ghost" onClick={clearOperationError}>
                Dismiss
              </Button>
            }
          >
            {operationError.status === 409
              ? "Refresh the authoritative state before retrying this lifecycle action."
              : operationError.status === 422
                ? "Review the selected robot and task."
                : operationError.status === 503
                  ? "Complete the required backend configuration in Settings."
                  : "The backend safely rejected the action. Check the server configuration and retry."}
          </Alert>
        )}

        {stateError && (
          <Alert
            tone="danger"
            title={stateError.message}
            action={
              <Button size="sm" icon={RefreshCw} loading={stateLoading} onClick={retryState}>
                Reload state
              </Button>
            }
          />
        )}

        {state.last_error && <Alert tone="danger" title="Runtime error">{state.last_error}</Alert>}

        {(state.status === "stopped" || state.status === "failed") && (
          <Alert
            tone={state.teardown_verified ? "success" : "warning"}
            icon={state.teardown_verified ? CheckCircle2 : TriangleAlert}
            title={state.teardown_verified ? "Teardown verified" : "Teardown is not yet verified"}
          >
            {state.teardown_verified
              ? "The action loop is joined, the queue is empty, and the provider resource is stopped."
              : "Do not assume the provider resource is gone; retry Stop or inspect the backend."}
          </Alert>
        )}
      </PageSection>

      <PageSection className="mt-6">
        <StatGrid columns={4}>
          <Stat
            icon={Timer}
            label="Latency"
            value={state.last_latency_ms == null ? "—" : `${formatDecimal(state.last_latency_ms)} ms`}
            hint={
              state.average_latency_ms == null
                ? "No completed request"
                : `${formatDecimal(state.average_latency_ms)} ms average`
            }
          />
          <Stat
            icon={Activity}
            label="Inference rate"
            value={inferenceFrequency == null ? "—" : `${formatDecimal(inferenceFrequency)} Hz`}
            hint={`${formatCount(state.requests_completed)} chunks completed`}
          />
          <Stat
            icon={Gauge}
            label="Action rate"
            value={state.frequency_hz == null ? "—" : `${formatDecimal(state.frequency_hz)} Hz`}
            hint={`${formatCount(state.steps_executed)} actions executed`}
          />
          <Stat
            icon={RadioTower}
            label="Queue"
            value={formatCount(state.queue_depth)}
            hint={`${formatCount(state.dropped_chunks)} chunks dropped`}
          />
        </StatGrid>
      </PageSection>

      <PageSection>
        <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_22rem]">
          <div className="space-y-6">
            {state.recording.enabled && (
              <Panel>
                <PanelHeader
                  title="Session recording"
                  description={`${state.recording.episode_count} episode · ${formatDuration(
                    state.recording.duration_seconds,
                  )} · ${state.recording.status}`}
                  actions={
                    state.recording.status === "recording" ? (
                      <Badge tone="danger" dot pulse>
                        REC
                      </Badge>
                    ) : undefined
                  }
                />
                {state.recording.hf_repo_id && (
                  <p className="px-5 py-4 text-xs text-positive-700">
                    Uploaded as {state.recording.hf_repo_id}
                    {uploadedRecordingUrl && (
                      <a
                        href={uploadedRecordingUrl}
                        target="_blank"
                        rel="noreferrer"
                        className="ml-2 inline-flex items-center gap-1 underline underline-offset-2"
                      >
                        Open on Hugging Face
                        <ExternalLink className="h-3 w-3" aria-hidden="true" />
                      </a>
                    )}
                  </p>
                )}
              </Panel>
            )}

            {recordingReady && (
              <SessionRecordingUpload
                key={recordingReady}
                recordingId={recordingReady}
                defaultName={recordingName || state.name}
                namespace={settings?.hf_namespace ?? null}
              />
            )}

            <div>
              <SectionHeading title="Deployment details" className="mb-3" />
              <DisclosureGroup>
                <Disclosure title="Policy and runtime" defaultOpen>
                  <DescriptionList
                    items={[
                      { label: "Model repository", value: state.model_repo, mono: true },
                      {
                        label: "Pinned revision",
                        value: state.checkpoint_revision ?? "revision pending",
                        mono: true,
                      },
                      {
                        label: "Runtime",
                        value: RUNTIME_LABELS[state.runtime] ?? state.runtime,
                      },
                      { label: "Compute", value: state.compute_size },
                    ]}
                  />
                </Disclosure>
                <Disclosure title="Endpoint and provider">
                  <DescriptionList
                    items={[
                      {
                        label: "Endpoint health",
                        value: state.endpoint_healthy
                          ? state.session_status === "idle"
                            ? "Provider ready · identity verified at deploy"
                            : "Healthy · identity verified"
                          : state.endpoint_url
                            ? "Assigned · awaiting readiness"
                            : "Not available",
                      },
                      {
                        label: "Provider id",
                        value: state.provider_app_id ?? state.endpoint_id,
                        mono: true,
                      },
                      { label: "Timeout", value: `${state.timeout_seconds}s` },
                      { label: "Target", value: state.target_kind },
                    ]}
                  />
                </Disclosure>
                <Disclosure title="Timing">
                  <DescriptionList
                    items={[
                      { label: "Created", value: formatDateTime(state.created_at) },
                      { label: "Started", value: formatDateTime(state.started_at) },
                      { label: "Session started", value: formatDateTime(state.session_started_at) },
                      { label: "Session stopped", value: formatDateTime(state.session_stopped_at) },
                    ]}
                  />
                </Disclosure>
              </DisclosureGroup>
            </div>
          </div>

          <Panel className="h-fit xl:sticky xl:top-6">
            <PanelHeader
              title="Robot execution"
              description={
                activeSession
                  ? "A session owns this follower until it is stopped."
                  : "Start is enabled only after a healthy endpoint snapshot and a connected follower are authoritative."
              }
              actions={
                <Badge
                  tone={sessionTone[state.session_status]}
                  dot
                  pulse={isBusyStatus(state.session_status)}
                >
                  {state.session_status}
                </Badge>
              }
            />
            {activeSession ? (
              <div className="space-y-4 px-5 py-5">
                <DescriptionList
                  columns={1}
                  items={[
                    {
                      label: "Follower",
                      value: state.arm_id ?? "Not reported",
                      mono: true,
                    },
                    { label: "Started", value: formatDateTime(state.session_started_at) },
                    {
                      label: "Recording",
                      value: state.recording.enabled
                        ? `${state.recording.status} · ${state.recording.episode_count} episode`
                        : "Not recording",
                    },
                  ]}
                />
                <Alert tone="warning" title="The robot is executing this policy">
                  Stop the session to release the follower. Stop clears writes, safe-idles to
                  gravity compensation, and releases the motion lease; an explicit Settings
                  Disconnect is what returns the arm to a de-energized state.
                </Alert>
                <Button
                  variant="danger"
                  size="lg"
                  icon={Square}
                  fullWidth
                  loading={busy === "stop"}
                  disabled={!canStop}
                  onClick={() => void onStop()}
                >
                  {busy === "stop" ? "Stopping and verifying…" : "Stop session + endpoint"}
                </Button>
              </div>
            ) : (
            <div className="space-y-4 px-5 py-5">
              <Field
                label="Connected follower"
                hint={`Telemetry: ${arms.connectionState}`}
              >
                <Select
                  value={armId}
                  disabled={sessionLocked || state.session_status !== "idle"}
                  onChange={(event) => setArmId(event.target.value)}
                >
                  {followers.length === 0 && <option value="">No connected follower</option>}
                  {followers.map((arm) => (
                    <option key={arm.id} value={arm.id}>
                      {arm.name} · {arm.id}
                      {arm.side ? ` · ${arm.side}` : ""}
                      {arm.pair_id ? ` · pair ${arm.pair_id}` : ""}
                    </option>
                  ))}
                </Select>
              </Field>

              {selectedFollower?.warnings.some((warning) => warning.includes("NO SASH GUARD")) && (
                <Alert tone="danger" title="NO SASH GUARD">
                  This follower has no soft limits configured.
                </Alert>
              )}

              <Field label="Task">
                <TextArea
                  rows={3}
                  maxLength={512}
                  value={task}
                  disabled={sessionLocked || state.session_status !== "idle"}
                  placeholder="Pick up the blue block and place it on the target."
                  onChange={(event) => setTask(event.target.value)}
                />
              </Field>

              <Checkbox
                label="Record this inference session"
                checked={recordSession}
                disabled={sessionLocked || state.session_status !== "idle"}
                onChange={(event) => setRecordSession(event.target.checked)}
              />

              {recordSession && (
                <div className="space-y-3">
                  <Field label="Recording name" optional>
                    <TextInput
                      maxLength={160}
                      value={recordingName}
                      disabled={sessionLocked}
                      placeholder="Generated if blank"
                      onChange={(event) => setRecordingName(event.target.value)}
                    />
                  </Field>
                  <Field label="Operator" optional>
                    <TextInput
                      maxLength={160}
                      value={operator}
                      disabled={sessionLocked}
                      onChange={(event) => setOperator(event.target.value)}
                    />
                  </Field>
                  <Field label="Notes" optional>
                    <TextArea
                      rows={2}
                      maxLength={2_000}
                      value={recordingNotes}
                      disabled={sessionLocked}
                      onChange={(event) => setRecordingNotes(event.target.value)}
                    />
                  </Field>
                </div>
              )}

              {arms.error && <p className="text-xs text-critical-700">{arms.error}</p>}

              <Button
                variant="primary"
                size="lg"
                icon={Play}
                fullWidth
                loading={busy === "start"}
                disabled={!canStart}
                onClick={() => void onStart()}
              >
                {busy === "start" ? "Starting safely…" : "Start execution"}
              </Button>

              {followers.length === 0 && (
                <Link to="/settings#yam-setup" className={buttonClass("secondary", "sm", true)}>
                  Connect a follower arm
                </Link>
              )}
            </div>
            )}
          </Panel>
        </div>
      </PageSection>
    </Page>
  );
}
