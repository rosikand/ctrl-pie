export type TrainingRunStatus =
  | "created"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

export type MetricPoint = {
  step: number;
  value: number;
};

export type TrainingCheckpoint = {
  repo_id: string;
  revision: string;
  step: number;
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
