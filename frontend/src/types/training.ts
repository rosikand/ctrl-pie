export type TrainingRunStatus =
  | "created"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

export type ManagedTrainingJobStatus =
  | "created"
  | "launching"
  | "running"
  | "finalizing"
  | "cancelling"
  | "completed"
  | "failed"
  | "cancelled";

export type ManagedTrainingOutcome =
  | "pending"
  | "succeeded"
  | "failed"
  | "cancelled";

export type ManagedTrainingProviderState =
  | "pending"
  | "running"
  | "succeeded"
  | "failed"
  | "cancelled"
  | "stopping"
  | "stopped"
  | "unknown";

export type ManagedTrainingJobSummary = {
  id: string;
  status: ManagedTrainingJobStatus;
  outcome: ManagedTrainingOutcome;
  target_kind: "stub" | "modal";
  compute_size: string;
  deadline_at: string;
  provider_state: ManagedTrainingProviderState;
  teardown_verified: boolean;
  output_model_repo: string;
  output_marker_revision: string | null;
  output_revision: string | null;
  last_error: string | null;
  event_gap: boolean;
};

export type MetricPoint = {
  step: number;
  value: number;
};

export type TrainingCheckpoint = {
  repo_id: string;
  revision: string;
  step: number;
};

export type TrainingConsoleLogSource = "stdout" | "stderr" | "system";

export type TrainingConsoleLog = {
  sequence: number;
  source: TrainingConsoleLogSource;
  line: string;
  step: number | null;
  timestamp: string;
};

export type TrainingConsoleLogsResponse = {
  logs: TrainingConsoleLog[];
  oldest_sequence: number | null;
  latest_sequence: number | null;
  next_sequence: number;
  truncated: boolean;
  has_more: boolean;
};

export type TrainingRun = {
  id: string;
  name: string;
  status: TrainingRunStatus;
  current_step: number;
  dataset_repo: string | null;
  base_model: string | null;
  runtime: string | null;
  framework: string | null;
  output_model_repo: string | null;
  checkpoint_revision: string | null;
  config: Record<string, unknown>;
  metrics: Record<string, MetricPoint[]>;
  checkpoints: TrainingCheckpoint[];
  managed_job: ManagedTrainingJobSummary | null;
  created_at: string;
  updated_at: string;
};

export type TrainingRunsResponse = {
  runs: TrainingRun[];
};

export type ModelCardMetadata = {
  description: string | null;
  base_model: string[];
  datasets: string[];
};

export type TrainerModelSummary = {
  repo_id: string;
  name: string;
  revision: string | null;
  hub_url: string;
  private: boolean;
  gated: boolean;
  last_modified: string | null;
  pipeline_tag: string | null;
  library_name: string | null;
  tags: string[];
  card: ModelCardMetadata | null;
  checkpoints: string[];
};

export type TrainerModelsResponse = {
  namespace: string;
  models: TrainerModelSummary[];
  total: number;
  fetched_at: string;
};

export type TrainingLoadError = {
  message: string;
  status: number | null;
};
