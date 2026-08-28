export type DatasetCardMetadata = {
  title: string | null;
  description: string | null;
  license: string | null;
  task_categories: string[];
};

export type LeRobotMetadata = {
  codebase_version: string | null;
  robot_type: string | null;
  fps: number | null;
  total_episodes: number | null;
  total_frames: number | null;
  total_tasks: number | null;
  features: string[];
};

export type DatasetSummary = {
  repo_id: string;
  name: string;
  revision: string | null;
  hub_url: string;
  private: boolean;
  gated: boolean;
  created_at: string | null;
  last_modified: string | null;
  tags: string[];
  card: DatasetCardMetadata | null;
  lerobot: LeRobotMetadata | null;
};

export type DatasetsResponse = {
  namespace: string;
  datasets: DatasetSummary[];
  total: number;
  next_cursor: string | null;
  fetched_at: string;
};
