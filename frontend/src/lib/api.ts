import type { ArmTelemetry, ArmsResponse, JogCommand } from "../types/arms";
import type {
  DatasetEpisodeDetail,
  DatasetEpisodesResponse,
} from "../types/datasetEpisodes";
import type { DatasetsResponse } from "../types/datasets";
import type {
  CreateInferenceDeploymentRequest,
  DeploymentRead,
  InferenceDeploymentsResponse,
  InferenceStateRead,
  StartInferenceSessionRequest,
  StopInferenceDeploymentRequest,
} from "../types/inference";
import type {
  CreateRecordingRequest,
  Recording,
  RecordingsResponse,
  RecordingState,
  StartEpisodeRequest,
  StopEpisodeRequest,
  UploadRecordingRequest,
  UploadRecordingResponse,
} from "../types/recordings";
import type {
  TrainerModelsResponse,
  TrainingConsoleLogsResponse,
  TrainingRun,
  TrainingRunsResponse,
} from "../types/training";

export type ServiceStatus = {
  id: "postgres" | "huggingface" | "modal" | "arms";
  label: string;
  status: "connected" | "configured" | "missing" | "error";
  detail: string;
  required: boolean;
};

export type SettingsStatus = {
  mode: "mock" | "hardware";
  setup_complete: boolean;
  services: ServiceStatus[];
  inference: {
    mock_mode: boolean;
    hf_configured: boolean;
    modal_configured: boolean;
    modal_proxy_configured: boolean;
  };
};

export type PublicSettings = {
  hf_namespace: string | null;
  recording_fps: number;
  default_runtime: "lerobot" | "openpi";
  default_compute: "Modal: A10G" | "Modal: A100" | "Modal: H100";
  modal_timeout_minutes: number;
};

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: unknown } | null;
    const message = typeof payload?.detail === "string"
      ? payload.detail
      : `Request failed (${response.status})`;
    throw new ApiError(response.status, message);
  }
  return response.json() as Promise<T>;
}

export function fetchDatasets({
  cursor,
  refresh = false,
  signal,
}: {
  cursor?: string | null;
  refresh?: boolean;
  signal?: AbortSignal;
} = {}): Promise<DatasetsResponse> {
  const query = new URLSearchParams({ limit: "24" });
  if (cursor) query.set("cursor", cursor);
  if (refresh) query.set("refresh", "true");
  return request<DatasetsResponse>(`/api/datasets?${query.toString()}`, {
    signal,
    cache: refresh ? "no-store" : "default",
  });
}

export function fetchDatasetEpisodes(
  repoName: string,
  signal?: AbortSignal,
): Promise<DatasetEpisodesResponse> {
  return request<DatasetEpisodesResponse>(
    `/api/datasets/${encodeURIComponent(repoName)}/episodes`,
    { signal, cache: "no-store" },
  );
}

export function fetchDatasetEpisode(
  repoName: string,
  episodeIndex: number,
  revision: string,
  signal?: AbortSignal,
): Promise<DatasetEpisodeDetail> {
  const query = new URLSearchParams({ revision });
  return request<DatasetEpisodeDetail>(
    `/api/datasets/${encodeURIComponent(repoName)}/episodes/${episodeIndex}?${query.toString()}`,
    { signal },
  );
}

export function fetchTrainingRuns(signal?: AbortSignal): Promise<TrainingRunsResponse> {
  return request<TrainingRunsResponse>("/api/trainer/runs", { signal });
}

export function fetchTrainingRun(
  runId: string,
  signal?: AbortSignal,
): Promise<TrainingRun> {
  return request<TrainingRun>(`/api/trainer/runs/${encodeURIComponent(runId)}`, {
    signal,
    cache: "no-store",
  });
}

export function fetchTrainingConsoleLogs({
  runId,
  afterSequence,
  limit = 200,
  signal,
}: {
  runId: string;
  afterSequence?: number;
  limit?: number;
  signal?: AbortSignal;
}): Promise<TrainingConsoleLogsResponse> {
  const query = new URLSearchParams({ limit: String(limit) });
  if (afterSequence !== undefined) {
    query.set("after_sequence", String(afterSequence));
  }
  return request<TrainingConsoleLogsResponse>(
    `/api/trainer/runs/${encodeURIComponent(runId)}/logs?${query.toString()}`,
    { signal, cache: "no-store" },
  );
}

export function fetchTrainerModels(
  refresh = false,
  signal?: AbortSignal,
): Promise<TrainerModelsResponse> {
  const query = new URLSearchParams();
  if (refresh) query.set("refresh", "true");
  const suffix = query.size ? `?${query.toString()}` : "";
  return request<TrainerModelsResponse>(`/api/trainer/models${suffix}`, {
    signal,
    cache: refresh ? "no-store" : "default",
  });
}

export function fetchArms(): Promise<ArmsResponse> {
  return request<ArmsResponse>("/api/arms");
}

export function fetchArm(armId: string): Promise<ArmTelemetry> {
  return request<ArmTelemetry>(`/api/arms/${encodeURIComponent(armId)}`);
}

export function jogArm(armId: string, command: JogCommand): Promise<ArmTelemetry> {
  return request<ArmTelemetry>(`/api/arms/${encodeURIComponent(armId)}/jog`, {
    method: "POST",
    body: JSON.stringify(command),
  });
}

export function fetchRecordings(): Promise<RecordingsResponse> {
  return request<RecordingsResponse>("/api/recordings");
}

export function createRecording(payload: CreateRecordingRequest): Promise<Recording> {
  return request<Recording>("/api/recordings", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function fetchRecordingState(recordingId: string): Promise<RecordingState> {
  return request<RecordingState>(
    `/api/recordings/${encodeURIComponent(recordingId)}/state`,
  );
}

export function startTeleop(recordingId: string): Promise<RecordingState> {
  return request<RecordingState>(
    `/api/recordings/${encodeURIComponent(recordingId)}/teleop/start`,
    { method: "POST" },
  );
}

export function stopTeleop(recordingId: string): Promise<RecordingState> {
  return request<RecordingState>(
    `/api/recordings/${encodeURIComponent(recordingId)}/teleop/stop`,
    { method: "POST" },
  );
}

export function startEpisode(
  recordingId: string,
  payload: StartEpisodeRequest,
): Promise<RecordingState> {
  return request<RecordingState>(
    `/api/recordings/${encodeURIComponent(recordingId)}/episodes/start`,
    { method: "POST", body: JSON.stringify(payload) },
  );
}

export function stopEpisode(
  recordingId: string,
  payload: StopEpisodeRequest,
): Promise<RecordingState> {
  return request<RecordingState>(
    `/api/recordings/${encodeURIComponent(recordingId)}/episodes/stop`,
    { method: "POST", body: JSON.stringify(payload) },
  );
}

export function uploadRecording(
  recordingId: string,
  payload: UploadRecordingRequest,
  signal?: AbortSignal,
): Promise<UploadRecordingResponse> {
  return request<UploadRecordingResponse>(
    `/api/recordings/${encodeURIComponent(recordingId)}/upload`,
    { method: "POST", body: JSON.stringify(payload), signal },
  );
}

export function fetchSettingsStatus(signal?: AbortSignal): Promise<SettingsStatus> {
  return request<SettingsStatus>("/api/settings/status", { signal });
}

export function fetchPublicSettings(signal?: AbortSignal): Promise<PublicSettings> {
  return request<PublicSettings>("/api/settings", { signal });
}

export function savePublicSettings(
  settings: Omit<PublicSettings, "hf_namespace">,
): Promise<PublicSettings> {
  return request<PublicSettings>("/api/settings", {
    method: "PATCH",
    body: JSON.stringify(settings),
  });
}

export function fetchInferenceDeployments(
  signal?: AbortSignal,
): Promise<InferenceDeploymentsResponse> {
  return request<InferenceDeploymentsResponse>("/api/inference/deployments", {
    signal,
    cache: "no-store",
  });
}

export function fetchInferenceDeployment(
  deploymentId: string,
  signal?: AbortSignal,
): Promise<DeploymentRead> {
  return request<DeploymentRead>(
    `/api/inference/deployments/${encodeURIComponent(deploymentId)}`,
    { signal, cache: "no-store" },
  );
}

export function createInferenceDeployment(
  payload: CreateInferenceDeploymentRequest,
  signal?: AbortSignal,
): Promise<DeploymentRead> {
  return request<DeploymentRead>("/api/inference/deployments", {
    method: "POST",
    body: JSON.stringify(payload),
    signal,
  });
}

export function fetchInferenceState(
  deploymentId: string,
  signal?: AbortSignal,
): Promise<InferenceStateRead> {
  return request<InferenceStateRead>(
    `/api/inference/deployments/${encodeURIComponent(deploymentId)}/state`,
    { signal, cache: "no-store" },
  );
}

export function startInferenceSession(
  deploymentId: string,
  payload: StartInferenceSessionRequest,
  signal?: AbortSignal,
): Promise<InferenceStateRead> {
  return request<InferenceStateRead>(
    `/api/inference/deployments/${encodeURIComponent(deploymentId)}/start`,
    { method: "POST", body: JSON.stringify(payload), signal },
  );
}

export function stopInferenceDeployment(
  deploymentId: string,
  payload: StopInferenceDeploymentRequest = {},
  signal?: AbortSignal,
): Promise<InferenceStateRead> {
  return request<InferenceStateRead>(
    `/api/inference/deployments/${encodeURIComponent(deploymentId)}/stop`,
    { method: "POST", body: JSON.stringify(payload), signal },
  );
}
