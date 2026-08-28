export type RecordingStatus =
  | "draft"
  | "teleop"
  | "recording"
  | "ready"
  | "uploading"
  | "uploaded"
  | "failed";

export type Recording = {
  id: string;
  name: string;
  task: string;
  status: RecordingStatus;
  leader_robot_id: string;
  follower_robot_id: string;
  episode_count: number;
  duration_seconds: number;
  hf_repo_id: string | null;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
};

export type RecordingsResponse = {
  recordings: Recording[];
};

export type RecordingState = {
  recording_id: string;
  teleop_active: boolean;
  episode_active: boolean;
  current_episode_index: number | null;
  episode_duration_seconds: number;
  episode_count: number;
  status: RecordingStatus;
};

export type CreateRecordingRequest = {
  name: string;
  task: string;
  leader_robot_id: string;
  follower_robot_id: string;
  metadata?: Record<string, unknown>;
};

export type StartEpisodeRequest = {
  metadata?: {
    operator?: string;
    notes?: string;
  };
};

export type StopEpisodeRequest = {
  success?: boolean;
  notes?: string;
};

export type UploadRecordingRequest = {
  repo_name: string;
  private: boolean;
};

export type UploadRecordingResponse = {
  recording: Recording;
  repo_id: string;
  repo_url: string;
  revision: string | null;
};
