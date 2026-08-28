import type { ArmTelemetry, ArmsResponse, JogCommand } from "../types/arms";
import type {
  DatasetEpisodeDetail,
  DatasetEpisodesResponse,
} from "../types/datasetEpisodes";
import type { DatasetsResponse } from "../types/datasets";
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
): Promise<UploadRecordingResponse> {
  return request<UploadRecordingResponse>(
    `/api/recordings/${encodeURIComponent(recordingId)}/upload`,
    { method: "POST", body: JSON.stringify(payload) },
  );
}

export function fetchSettingsStatus(): Promise<SettingsStatus> {
  return request<SettingsStatus>("/api/settings/status");
}

export function fetchPublicSettings(): Promise<PublicSettings> {
  return request<PublicSettings>("/api/settings");
}

export function savePublicSettings(
  settings: Omit<PublicSettings, "hf_namespace">,
): Promise<PublicSettings> {
  return request<PublicSettings>("/api/settings", {
    method: "PATCH",
    body: JSON.stringify(settings),
  });
}
