import {
  AlertCircle,
  ArrowLeft,
  ChevronLeft,
  ChevronRight,
  Clock3,
  Film,
  Gauge,
  Layers3,
  LoaderCircle,
  RefreshCw,
  Video,
  VideoOff,
} from "lucide-react";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Link, Navigate, useParams } from "react-router-dom";

import {
  useDatasetEpisodes,
  type EpisodeLoadError,
} from "../hooks/useDatasetEpisodes";
import type {
  DatasetEpisodeDetail,
  EpisodeSummary,
  TimelineFrame,
} from "../types/datasetEpisodes";

const countFormatter = new Intl.NumberFormat();
const valueFormatter = new Intl.NumberFormat(undefined, {
  maximumFractionDigits: 4,
  minimumFractionDigits: 0,
});

function formatDuration(seconds: number): string {
  const safeSeconds = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  const minutes = Math.floor(safeSeconds / 60);
  const remainder = (safeSeconds % 60).toFixed(2).padStart(5, "0");
  return `${minutes.toString().padStart(2, "0")}:${remainder}`;
}

function formatValue(value: number | undefined): string {
  if (value === undefined || !Number.isFinite(value)) return "—";
  const magnitude = Math.abs(value);
  if (magnitude !== 0 && (magnitude >= 10_000 || magnitude < 0.0001)) {
    return value.toExponential(3);
  }
  return valueFormatter.format(value);
}

function safeVideoProxyUrl(value: string | null): string | null {
  if (!value) return null;
  try {
    const resolved = new URL(value, window.location.origin);
    if (
      resolved.origin !== window.location.origin ||
      !resolved.pathname.startsWith("/api/datasets/")
    ) {
      return null;
    }
    return `${resolved.pathname}${resolved.search}`;
  } catch {
    return null;
  }
}

function nearestFrame(frames: TimelineFrame[], timestamp: number): TimelineFrame | null {
  if (frames.length === 0) return null;
  let low = 0;
  let high = frames.length - 1;
  while (low <= high) {
    const middle = Math.floor((low + high) / 2);
    const value = frames[middle].timestamp;
    if (value === timestamp) return frames[middle];
    if (value < timestamp) low = middle + 1;
    else high = middle - 1;
  }
  const before = frames[Math.max(0, high)];
  const after = frames[Math.min(frames.length - 1, low)];
  return Math.abs(before.timestamp - timestamp) <= Math.abs(after.timestamp - timestamp)
    ? before
    : after;
}

function errorTitle(error: EpisodeLoadError, scope: "dataset" | "episode"): string {
  if (error.status === 503) return "Hugging Face is not configured";
  if (error.status === 403) return "Dataset access was denied";
  if (error.status === 404) return scope === "dataset" ? "Dataset not found" : "Episode not found";
  if (error.status === 409) return "Dataset revision changed";
  if (error.status === 422) return "Dataset metadata is not readable";
  if (error.status === 502) return "Hugging Face is unavailable";
  return scope === "dataset" ? "Dataset could not be loaded" : "Episode could not be loaded";
}

function ErrorPanel({
  error,
  scope,
  onRetry,
  loading,
}: {
  error: EpisodeLoadError;
  scope: "dataset" | "episode";
  onRetry: () => void;
  loading: boolean;
}) {
  const canOpenSettings = error.status === 403 || error.status === 503;
  return (
    <section
      className="grid min-h-[20rem] place-items-center rounded-xl border border-slate-200 bg-white px-6 text-center shadow-panel"
      role="alert"
    >
      <div className="max-w-md">
        <div className="mx-auto grid h-11 w-11 place-items-center rounded-xl bg-rose-50 text-rose-600">
          <AlertCircle className="h-5 w-5" aria-hidden="true" />
        </div>
        <h2 className="mt-4 text-sm font-semibold text-slate-900">{errorTitle(error, scope)}</h2>
        <p className="mt-2 text-sm leading-6 text-slate-500">{error.message}</p>
        <div className="mt-5 flex flex-wrap justify-center gap-2">
          <button
            type="button"
            onClick={onRetry}
            disabled={loading}
            className="inline-flex items-center gap-2 rounded-lg bg-slate-900 px-3.5 py-2 text-xs font-semibold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? (
              <LoaderCircle className="h-3.5 w-3.5 animate-spin" aria-hidden="true" />
            ) : (
              <RefreshCw className="h-3.5 w-3.5" aria-hidden="true" />
            )}
            Try again
          </button>
          {canOpenSettings && (
            <Link
              to="/settings"
              className="rounded-lg border border-slate-200 bg-white px-3.5 py-2 text-xs font-semibold text-slate-700 transition hover:bg-slate-50"
            >
              Open settings
            </Link>
          )}
        </div>
      </div>
    </section>
  );
}

function DatasetLoading() {
  return (
    <section className="grid gap-5 lg:grid-cols-[17rem_minmax(0,1fr)]" aria-busy="true" aria-label="Loading dataset episodes">
      <p className="sr-only" role="status">Loading dataset episodes</p>
      <div className="h-80 animate-pulse rounded-xl border border-slate-200 bg-white p-5 shadow-panel">
        <div className="h-4 w-24 rounded bg-slate-100" />
        <div className="mt-6 space-y-3">
          {Array.from({ length: 4 }).map((_, index) => <div key={index} className="h-14 rounded-lg bg-slate-50" />)}
        </div>
      </div>
      <div className="h-[34rem] animate-pulse rounded-xl border border-slate-200 bg-white p-5 shadow-panel">
        <div className="aspect-video rounded-lg bg-slate-100" />
        <div className="mt-5 h-4 w-2/3 rounded bg-slate-100" />
        <div className="mt-4 h-10 rounded bg-slate-50" />
      </div>
    </section>
  );
}

function EpisodePicker({
  episodes,
  selectedEpisodeIndex,
  onSelect,
}: {
  episodes: EpisodeSummary[];
  selectedEpisodeIndex: number;
  onSelect: (episodeIndex: number) => void;
}) {
  const selectedPosition = Math.max(
    0,
    episodes.findIndex((episode) => episode.episode_index === selectedEpisodeIndex),
  );
  const previous = episodes[selectedPosition - 1];
  const next = episodes[selectedPosition + 1];

  return (
    <aside className="rounded-xl border border-slate-200 bg-white shadow-panel">
      <div className="border-b border-slate-100 px-4 py-4">
        <div className="flex items-center justify-between gap-3">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-400">Episode browser</p>
            <p className="mt-1 text-xs text-slate-500">{countFormatter.format(episodes.length)} available</p>
          </div>
          <div className="flex items-center gap-1">
            <button
              type="button"
              aria-label="Previous episode"
              disabled={!previous}
              onClick={() => previous && onSelect(previous.episode_index)}
              className="rounded-md border border-slate-200 p-1.5 text-slate-500 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-30"
            >
              <ChevronLeft className="h-4 w-4" aria-hidden="true" />
            </button>
            <button
              type="button"
              aria-label="Next episode"
              disabled={!next}
              onClick={() => next && onSelect(next.episode_index)}
              className="rounded-md border border-slate-200 p-1.5 text-slate-500 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-30"
            >
              <ChevronRight className="h-4 w-4" aria-hidden="true" />
            </button>
          </div>
        </div>
        <label className="mt-4 block text-xs font-medium text-slate-600">
          Selected episode
          <select
            value={selectedEpisodeIndex}
            onChange={(event) => onSelect(Number(event.target.value))}
            className="mt-1.5 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm outline-none ring-brand-100 focus:border-brand-500 focus:ring-4"
          >
            {episodes.map((episode) => (
              <option key={episode.episode_index} value={episode.episode_index}>
                Episode {episode.episode_index + 1} · {formatDuration(episode.duration_seconds)}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="max-h-[31rem] space-y-1.5 overflow-y-auto p-2" aria-label="Episodes">
        {episodes.map((episode) => {
          const selected = episode.episode_index === selectedEpisodeIndex;
          return (
            <button
              key={episode.episode_index}
              type="button"
              onClick={() => onSelect(episode.episode_index)}
              aria-current={selected ? "true" : undefined}
              className={`w-full rounded-lg px-3 py-3 text-left transition ${
                selected
                  ? "bg-brand-50 text-brand-700 ring-1 ring-brand-100"
                  : "text-slate-600 hover:bg-slate-50"
              }`}
            >
              <span className="flex items-center justify-between gap-2">
                <span className="text-xs font-semibold">Episode {episode.episode_index + 1}</span>
                <span className="font-mono text-[10px] opacity-70">{formatDuration(episode.duration_seconds)}</span>
              </span>
              <span className="mt-1 block truncate text-[11px] opacity-70" title={episode.tasks.join(", ")}>
                {episode.tasks.length ? episode.tasks.join(" · ") : "No task label"}
              </span>
              <span className="mt-1.5 block text-[10px] opacity-60">
                {countFormatter.format(episode.frame_count)} frames · rows {countFormatter.format(episode.dataset_from_index)}–{countFormatter.format(Math.max(episode.dataset_from_index, episode.dataset_to_index - 1))}
              </span>
            </button>
          );
        })}
      </div>
    </aside>
  );
}

function VectorValues({
  title,
  names,
  values,
  accent,
}: {
  title: string;
  names: string[];
  values: number[];
  accent: "blue" | "violet";
}) {
  const rowCount = Math.max(names.length, values.length);
  const badge = accent === "blue" ? "bg-blue-50 text-blue-700" : "bg-violet-50 text-violet-700";
  return (
    <section className="min-w-0 rounded-xl border border-slate-200 bg-white shadow-panel">
      <div className="flex items-center justify-between gap-3 border-b border-slate-100 px-4 py-3.5">
        <h3 className="text-sm font-semibold text-slate-900">{title}</h3>
        <span className={`rounded-full px-2 py-1 text-[10px] font-semibold ${badge}`}>
          {countFormatter.format(rowCount)} values
        </span>
      </div>
      {rowCount > 0 ? (
        <dl className="max-h-[24rem] divide-y divide-slate-100 overflow-y-auto px-4">
          {Array.from({ length: rowCount }).map((_, index) => {
            const name = names[index] || `value.${index}`;
            return (
              <div key={`${name}-${index}`} className="flex items-center justify-between gap-4 py-2.5">
                <dt className="min-w-0 truncate font-mono text-[11px] text-slate-500" title={name}>{name}</dt>
                <dd className="shrink-0 font-mono text-xs font-semibold tabular-nums text-slate-800">{formatValue(values[index])}</dd>
              </div>
            );
          })}
        </dl>
      ) : (
        <p className="px-4 py-8 text-center text-xs leading-5 text-slate-400">No values were reported for this sample.</p>
      )}
    </section>
  );
}

function EpisodePlayer({ detail }: { detail: DatasetEpisodeDetail }) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [timelineTime, setTimelineTime] = useState(0);
  const [mediaLoading, setMediaLoading] = useState(Boolean(detail.video_url));
  const [mediaError, setMediaError] = useState<string | null>(null);
  const videoUrl = useMemo(() => safeVideoProxyUrl(detail.video_url), [detail.video_url]);
  const episode = detail.episode;
  const lastFrameTimestamp = detail.frames.at(-1)?.timestamp ?? 0;
  const timelineDuration = Math.max(0, episode.duration_seconds, lastFrameTimestamp);
  const videoFrom = episode.video_from_timestamp ?? 0;
  const videoTo = Math.max(
    videoFrom,
    episode.video_to_timestamp ?? videoFrom + timelineDuration,
  );
  const timelineStep = detail.fps > 0 ? 1 / detail.fps : 0.01;
  const currentFrame = useMemo(
    () => nearestFrame(detail.frames, timelineTime),
    [detail.frames, timelineTime],
  );

  useEffect(() => {
    setTimelineTime(0);
    setMediaError(null);
    setMediaLoading(Boolean(detail.video_url));
    const video = videoRef.current;
    if (video) {
      video.pause();
      video.load();
    }
  }, [detail.episode.episode_index, detail.revision, detail.video_url]);

  const updateFromMedia = useCallback((video: HTMLVideoElement) => {
    const epsilon = 0.002;
    if (video.currentTime < videoFrom - epsilon) {
      video.currentTime = videoFrom;
      setTimelineTime(0);
      return;
    }
    if (video.currentTime >= videoTo - epsilon) {
      video.pause();
      if (video.currentTime > videoTo + epsilon) video.currentTime = videoTo;
      setTimelineTime(Math.min(timelineDuration, Math.max(0, videoTo - videoFrom)));
      return;
    }
    setTimelineTime(Math.min(timelineDuration, Math.max(0, video.currentTime - videoFrom)));
  }, [timelineDuration, videoFrom, videoTo]);

  const seekTimeline = useCallback((value: number) => {
    const nextTime = Math.min(timelineDuration, Math.max(0, value));
    setTimelineTime(nextTime);
    const video = videoRef.current;
    if (video && videoUrl) {
      video.currentTime = Math.min(videoTo, Math.max(videoFrom, videoFrom + nextTime));
    }
  }, [timelineDuration, videoFrom, videoTo, videoUrl]);

  return (
    <>
      <section className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-panel">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 px-5 py-4">
          <div className="flex min-w-0 items-center gap-2">
            <Video className="h-4 w-4 shrink-0 text-slate-400" aria-hidden="true" />
            <h2 className="text-sm font-semibold text-slate-900">Episode {episode.episode_index + 1}</h2>
            {detail.video_key && <span className="truncate font-mono text-[10px] text-slate-400" title={detail.video_key}>{detail.video_key}</span>}
          </div>
          <div className="flex items-center gap-3 text-[11px] text-slate-400">
            <span>{countFormatter.format(episode.frame_count)} frames</span>
            <span>{formatDuration(episode.duration_seconds)}</span>
          </div>
        </div>

        <div className="bg-slate-950">
          {videoUrl ? (
            <div className="relative aspect-video w-full">
              <video
                key={`${detail.repo_id}-${detail.revision}-${episode.episode_index}`}
                ref={videoRef}
                src={videoUrl}
                controls
                preload="metadata"
                playsInline
                className="h-full w-full bg-black object-contain"
                onLoadStart={() => setMediaLoading(true)}
                onLoadedMetadata={(event) => {
                  event.currentTarget.currentTime = videoFrom;
                  setTimelineTime(0);
                }}
                onCanPlay={() => setMediaLoading(false)}
                onPlaying={() => setMediaLoading(false)}
                onPlay={(event) => {
                  if (
                    event.currentTarget.currentTime < videoFrom ||
                    event.currentTarget.currentTime >= videoTo
                  ) {
                    event.currentTarget.currentTime = videoFrom;
                    setTimelineTime(0);
                  }
                }}
                onSeeking={(event) => updateFromMedia(event.currentTarget)}
                onTimeUpdate={(event) => updateFromMedia(event.currentTarget)}
                onEnded={(event) => updateFromMedia(event.currentTarget)}
                onError={() => {
                  setMediaLoading(false);
                  setMediaError("The proxied episode video could not be played.");
                }}
              >
                This browser cannot play MP4 video.
              </video>
              {mediaLoading && !mediaError && (
                <div className="pointer-events-none absolute inset-0 grid place-items-center bg-slate-950/35 text-white" role="status">
                  <span className="inline-flex items-center gap-2 rounded-full bg-slate-950/70 px-3 py-1.5 text-xs">
                    <LoaderCircle className="h-4 w-4 animate-spin" aria-hidden="true" />
                    Loading video…
                  </span>
                </div>
              )}
            </div>
          ) : (
            <div className="grid aspect-video place-items-center px-6 text-center text-slate-300">
              <div>
                <VideoOff className="mx-auto h-7 w-7 text-slate-500" aria-hidden="true" />
                <p className="mt-3 text-sm font-medium">No episode video</p>
                <p className="mt-1 text-xs text-slate-500">
                  {detail.video_url
                    ? "The backend did not provide a valid same-origin media URL."
                    : "Synchronized state and action samples remain available below."}
                </p>
              </div>
            </div>
          )}
        </div>

        <div className="px-5 py-4">
          {mediaError && (
            <div className="mb-4 flex items-start gap-2 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-700" role="alert">
              <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden="true" />
              {mediaError}
            </div>
          )}
          <div className="flex items-center justify-between gap-4 text-xs">
            <span className="font-mono font-semibold tabular-nums text-slate-800">
              {formatDuration(timelineTime)} / {formatDuration(timelineDuration)}
            </span>
            <span className="text-slate-400">
              {currentFrame ? `Frame ${countFormatter.format(currentFrame.frame_index)}` : "No synchronized frame"}
            </span>
          </div>
          <label className="mt-3 block">
            <span className="sr-only">Episode timeline</span>
            <input
              type="range"
              min={0}
              max={timelineDuration}
              step={timelineStep}
              value={Math.min(timelineTime, timelineDuration)}
              disabled={timelineDuration <= 0}
              onChange={(event) => seekTimeline(Number(event.target.value))}
              aria-valuetext={`${formatDuration(timelineTime)}, ${currentFrame ? `frame ${currentFrame.frame_index}` : "no frame"}`}
              className="h-2 w-full cursor-pointer accent-brand-600 disabled:cursor-not-allowed disabled:opacity-40"
            />
          </label>
          <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-2 text-[11px] text-slate-400">
            <span className="inline-flex items-center gap-1.5">
              <Gauge className="h-3.5 w-3.5" aria-hidden="true" />
              {countFormatter.format(detail.fps)} FPS
            </span>
            <span className="inline-flex items-center gap-1.5">
              <Clock3 className="h-3.5 w-3.5" aria-hidden="true" />
              Episode-relative timeline
            </span>
            {episode.video_from_timestamp !== null && episode.video_to_timestamp !== null && (
              <span title="Playback is constrained to this episode's segment in the packed video file.">
                Media segment {formatDuration(episode.video_from_timestamp)}–{formatDuration(episode.video_to_timestamp)}
              </span>
            )}
          </div>
        </div>
      </section>

      <section className="mt-5">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div>
            <h2 className="text-sm font-semibold text-slate-900">Synchronized values</h2>
            <p className="mt-1 text-xs text-slate-400">Nearest sample to the selected episode time.</p>
          </div>
          {currentFrame && (
            <span className="rounded-full bg-white px-2.5 py-1 font-mono text-[10px] text-slate-500 ring-1 ring-slate-200">
              t={currentFrame.timestamp.toFixed(3)}s · frame {currentFrame.frame_index}
            </span>
          )}
        </div>

        {currentFrame ? (
          <div className="grid gap-4 md:grid-cols-2">
            <VectorValues title="Observation state" names={detail.state_names} values={currentFrame.state} accent="blue" />
            <VectorValues title="Action" names={detail.action_names} values={currentFrame.action} accent="violet" />
          </div>
        ) : (
          <div className="grid min-h-40 place-items-center rounded-xl border border-amber-200 bg-amber-50 px-6 text-center">
            <div>
              <Layers3 className="mx-auto h-5 w-5 text-amber-500" aria-hidden="true" />
              <p className="mt-2 text-sm font-semibold text-amber-900">No synchronized samples</p>
              <p className="mt-1 text-xs leading-5 text-amber-700">The episode can be browsed, but its frame timeline is unavailable.</p>
            </div>
          </div>
        )}
      </section>
    </>
  );
}

function EpisodeDetailLoading() {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-panel" aria-busy="true">
      <p className="sr-only" role="status">Loading episode detail</p>
      <div className="aspect-video animate-pulse rounded-lg bg-slate-100" />
      <div className="mt-5 h-4 w-52 animate-pulse rounded bg-slate-100" />
      <div className="mt-4 h-9 animate-pulse rounded bg-slate-50" />
    </div>
  );
}

function DatasetEpisodeWorkspace({ repoName }: { repoName: string }) {
  const {
    dataset,
    selectedEpisodeIndex,
    detail,
    datasetLoading,
    detailLoading,
    datasetError,
    detailError,
    loadDataset,
    selectEpisode,
    retryDetail,
  } = useDatasetEpisodes(repoName);

  const retryEpisode = detailError?.status === 404 || detailError?.status === 409
    ? loadDataset
    : retryDetail;
  const detailIsCurrent = Boolean(
    detail &&
    dataset &&
    detail.repo_id === dataset.repo_id &&
    detail.revision === dataset.revision &&
    detail.episode.episode_index === selectedEpisodeIndex,
  );

  return (
    <div className="mx-auto max-w-7xl px-5 py-8 sm:px-8 lg:px-10 lg:py-10">
      <header>
        <Link to="/datasets" className="inline-flex items-center gap-1.5 text-xs font-semibold text-slate-500 transition hover:text-slate-800">
          <ArrowLeft className="h-3.5 w-3.5" aria-hidden="true" />
          All datasets
        </Link>
        <div className="mt-4 flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
          <div className="min-w-0">
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-brand-600">Episode visualizer</p>
            <h1 className="mt-2 break-words text-2xl font-semibold tracking-tight text-slate-950 sm:text-3xl">
              {dataset?.repo_id ?? repoName}
            </h1>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500">
              Scrub camera playback with synchronized observation state and action values.
            </p>
            {dataset && (
              <div className="mt-3 flex flex-wrap items-center gap-2 text-[11px] text-slate-400">
                <span className="rounded-md bg-white px-2 py-1 font-mono ring-1 ring-slate-200" title={dataset.revision}>@{dataset.revision.slice(0, 8)}</span>
                <span>{countFormatter.format(dataset.total_episodes)} episodes</span>
                <span>{countFormatter.format(dataset.fps)} FPS</span>
                <span>{dataset.video_key ? "Camera + robot data" : "Robot data only"}</span>
              </div>
            )}
          </div>
          {dataset && (
            <button
              type="button"
              onClick={() => void loadDataset()}
              disabled={datasetLoading}
              className="inline-flex w-fit items-center gap-2 rounded-lg border border-slate-200 bg-white px-3.5 py-2 text-xs font-semibold text-slate-700 shadow-sm transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${datasetLoading ? "animate-spin" : ""}`} aria-hidden="true" />
              Reload
            </button>
          )}
        </div>
      </header>

      <div className="mt-8">
        {datasetLoading && <DatasetLoading />}
        {!datasetLoading && datasetError && (
          <ErrorPanel error={datasetError} scope="dataset" onRetry={() => void loadDataset()} loading={datasetLoading} />
        )}
        {!datasetLoading && dataset && dataset.episodes.length === 0 && (
          <section className="grid min-h-[22rem] place-items-center rounded-xl border border-slate-200 bg-white px-6 text-center shadow-panel">
            <div className="max-w-md">
              <Film className="mx-auto h-7 w-7 text-slate-300" aria-hidden="true" />
              <h2 className="mt-3 text-sm font-semibold text-slate-900">No episodes found</h2>
              <p className="mt-2 text-sm leading-6 text-slate-500">
                This LeRobot dataset does not contain any episodes that can be visualized.
              </p>
            </div>
          </section>
        )}
        {!datasetLoading && dataset && dataset.episodes.length > 0 && selectedEpisodeIndex !== null && (
          <div className="grid items-start gap-5 lg:grid-cols-[17rem_minmax(0,1fr)]">
            <EpisodePicker episodes={dataset.episodes} selectedEpisodeIndex={selectedEpisodeIndex} onSelect={selectEpisode} />
            <div className="min-w-0">
              {(detailLoading || (!detailError && !detailIsCurrent)) && <EpisodeDetailLoading />}
              {!detailLoading && detailError && (
                <ErrorPanel error={detailError} scope="episode" onRetry={() => void retryEpisode()} loading={detailLoading || datasetLoading} />
              )}
              {!detailLoading && !detailError && detail && detailIsCurrent && <EpisodePlayer detail={detail} />}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export function DatasetEpisodePage() {
  const { repoName } = useParams<{ repoName: string }>();
  if (!repoName) return <Navigate to="/datasets" replace />;
  return <DatasetEpisodeWorkspace key={repoName} repoName={repoName} />;
}
