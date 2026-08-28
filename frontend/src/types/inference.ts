export type InferenceRuntime = "stub" | "lerobot" | "openpi";

export type InferenceComputeSize =
  | "CPU"
  | "Modal: A10G"
  | "Modal: A100"
  | "Modal: H100";

export type DeploymentStatus =
  | "created"
  | "deploying"
  | "running"
  | "stopping"
  | "stopped"
  | "failed";

export type InferenceSessionStatus =
  | "idle"
  | "starting"
  | "running"
  | "stopping"
  | "stopped"
  | "failed";

export type InferenceRecordingStatus =
  | "disabled"
  | "starting"
  | "recording"
  | "finalizing"
  | "ready"
  | "failed";

export type DeploymentRead = {
  id: string;
  endpoint_id: string;
  name: string;
  target_kind: "stub" | "modal";
  status: DeploymentStatus;
  model_repo: string;
  checkpoint_revision: string | null;
  runtime: InferenceRuntime;
  compute_size: InferenceComputeSize;
  endpoint_url: string | null;
  provider_app_id: string | null;
  arm_id: string | null;
  record_session: boolean;
  recording_id: string | null;
  started_at: string | null;
  stopped_at: string | null;
  created_at: string;
  updated_at: string;
};

export type InferenceRecordingState = {
  enabled: boolean;
  status: InferenceRecordingStatus;
  recording_id: string | null;
  episode_count: number;
  duration_seconds: number;
  hf_repo_id: string | null;
};

export type InferenceStateRead = DeploymentRead & {
  session_status: InferenceSessionStatus;
  endpoint_healthy: boolean;
  teardown_verified: boolean;
  steps_executed: number;
  requests_completed: number;
  dropped_chunks: number;
  queue_depth: number;
  last_latency_ms: number | null;
  average_latency_ms: number | null;
  frequency_hz: number | null;
  last_error: string | null;
  session_started_at: string | null;
  session_stopped_at: string | null;
  recording: InferenceRecordingState;
};

export type InferenceDeploymentsResponse = {
  deployments: DeploymentRead[];
};

export type CreateInferenceDeploymentRequest = {
  name: string;
  model_repo: string;
  checkpoint_revision: string | null;
  runtime: InferenceRuntime;
  compute_size: InferenceComputeSize;
};

export type StartInferenceSessionRequest = {
  arm_id: string;
  task: string;
  record_session: boolean;
  recording_name: string | null;
  recording_metadata?: {
    operator?: string;
    notes?: string;
  };
};

export type StopInferenceDeploymentRequest = {
  recording_success?: boolean;
  recording_notes?: string | null;
};

export type InferenceStreamMessage = {
  type: "inference_state";
  timestamp: string;
  state: InferenceStateRead;
};

export type InferenceStreamConnection =
  | "idle"
  | "connecting"
  | "live"
  | "reconnecting"
  | "closed";
