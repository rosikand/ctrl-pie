export type EpisodeSummary = {
  episode_index: number;
  tasks: string[];
  frame_count: number;
  duration_seconds: number;
  dataset_from_index: number;
  dataset_to_index: number;
  video_from_timestamp: number | null;
  video_to_timestamp: number | null;
};

export type TimelineFrame = {
  timestamp: number;
  frame_index: number;
  state: number[];
  action: number[];
};

export type DatasetEpisodesResponse = {
  repo_id: string;
  revision: string;
  fps: number;
  state_names: string[];
  action_names: string[];
  video_key: string | null;
  total_episodes: number;
  episodes: EpisodeSummary[];
};

export type DatasetEpisodeDetail = {
  repo_id: string;
  revision: string;
  fps: number;
  state_names: string[];
  action_names: string[];
  video_key: string | null;
  episode: EpisodeSummary;
  frames: TimelineFrame[];
  sampled_frame_count: number;
  frames_truncated: boolean;
  video_url: string | null;
};
