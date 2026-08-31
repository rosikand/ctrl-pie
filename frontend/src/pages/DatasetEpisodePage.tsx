import { ChevronLeft, ChevronRight, Film, LoaderCircle, RefreshCw, VideoOff } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Navigate, useParams } from "react-router-dom";

import { Page, PageHeader, PageSection } from "../components/layout/Page";
import { LoadErrorState } from "../components/LoadError";
import { Alert } from "../components/ui/Alert";
import { Mono } from "../components/ui/Badge";
import { Button, IconButton } from "../components/ui/Button";
import { EmptyState } from "../components/ui/EmptyState";
import { Select } from "../components/ui/Form";
import { Panel, PanelHeader } from "../components/ui/Panel";
import { Skeleton } from "../components/ui/Skeleton";
import { useDatasetEpisodes } from "../hooks/useDatasetEpisodes";
import { formatCount, formatPreciseDuration, shortRevision } from "../lib/format";
import type {
  DatasetEpisodeDetail,
  EpisodeSummary,
  TimelineFrame,
} from "../types/datasetEpisodes";

const valueFormatter = new Intl.NumberFormat(undefined, {
  maximumFractionDigits: 4,
  minimumFractionDigits: 0,
});

function formatValue(value: number | undefined): string {
  if (value === undefined || !Number.isFinite(value)) return "—";
  const magnitude = Math.abs(value);
  if (magnitude !== 0 && (magnitude >= 10_000 || magnitude < 0.0001)) {
    return value.toExponential(3);
  }
  return valueFormatter.format(value);
}

/** Only ever plays media the backend proxied on this origin. */
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
    <Panel as="aside">
      <PanelHeader
        title="Episodes"
        description={`${formatCount(episodes.length)} available`}
        actions={
          <div className="flex items-center gap-1">
            <IconButton
              icon={ChevronLeft}
              label="Previous episode"
              size="sm"
              disabled={!previous}
              onClick={() => previous && onSelect(previous.episode_index)}
            />
            <IconButton
              icon={ChevronRight}
              label="Next episode"
              size="sm"
              disabled={!next}
              onClick={() => next && onSelect(next.episode_index)}
            />
          </div>
        }
      />
      <div className="border-b border-line px-5 py-4">
        <label className="block text-xs font-medium text-ink-secondary" htmlFor="episode-select">
          Selected episode
        </label>
        <Select
          id="episode-select"
          className="mt-1.5"
          value={selectedEpisodeIndex}
          onChange={(event) => onSelect(Number(event.target.value))}
        >
          {episodes.map((episode) => (
            <option key={episode.episode_index} value={episode.episode_index}>
              Episode {episode.episode_index + 1} · {formatPreciseDuration(episode.duration_seconds)}
            </option>
          ))}
        </Select>
      </div>
      <ul className="scroll-quiet max-h-[28rem] divide-y divide-line-subtle overflow-y-auto">
        {episodes.map((episode) => {
          const selected = episode.episode_index === selectedEpisodeIndex;
          return (
            <li key={episode.episode_index}>
              <button
                type="button"
                onClick={() => onSelect(episode.episode_index)}
                aria-current={selected ? "true" : undefined}
                className={`w-full px-5 py-3 text-left transition ${
                  selected ? "bg-accent-50/60" : "hover:bg-canvas"
                }`}
              >
                <span className="flex items-center justify-between gap-2">
                  <span className={`text-xs font-medium ${selected ? "text-accent-800" : "text-ink"}`}>
                    Episode {episode.episode_index + 1}
                  </span>
                  <span className="font-mono text-2xs text-ink-muted">
                    {formatPreciseDuration(episode.duration_seconds)}
                  </span>
                </span>
                <span className="mt-1 block truncate text-2xs text-ink-muted" title={episode.tasks.join(", ")}>
                  {episode.tasks.length ? episode.tasks.join(" · ") : "No task label"}
                </span>
                <span className="mt-1 block text-2xs text-ink-faint">
                  {formatCount(episode.frame_count)} frames · rows{" "}
                  {formatCount(episode.dataset_from_index)}–
                  {formatCount(Math.max(episode.dataset_from_index, episode.dataset_to_index - 1))}
                </span>
              </button>
            </li>
          );
        })}
      </ul>
    </Panel>
  );
}

function VectorValues({
  title,
  names,
  values,
}: {
  title: string;
  names: string[];
  values: number[];
}) {
  const rowCount = Math.max(names.length, values.length);
  return (
    <Panel>
      <PanelHeader title={title} actions={<span className="text-2xs text-ink-muted">{formatCount(rowCount)} values</span>} />
      {rowCount > 0 ? (
        <dl className="scroll-quiet max-h-[22rem] divide-y divide-line-subtle overflow-y-auto px-5">
          {Array.from({ length: rowCount }).map((_, index) => {
            const name = names[index] || `value.${index}`;
            return (
              <div key={`${name}-${index}`} className="flex items-center justify-between gap-4 py-2.5">
                <dt className="min-w-0 truncate font-mono text-2xs text-ink-muted" title={name}>
                  {name}
                </dt>
                <dd className="shrink-0 font-mono text-xs font-medium tabular-nums text-ink">
                  {formatValue(values[index])}
                </dd>
              </div>
            );
          })}
        </dl>
      ) : (
        <p className="px-5 py-10 text-center text-xs leading-5 text-ink-muted">
          No values were reported for this sample.
        </p>
      )}
    </Panel>
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
  const videoTo = Math.max(videoFrom, episode.video_to_timestamp ?? videoFrom + timelineDuration);
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

  const updateFromMedia = useCallback(
    (video: HTMLVideoElement) => {
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
    },
    [timelineDuration, videoFrom, videoTo],
  );

  const seekTimeline = useCallback(
    (value: number) => {
      const nextTime = Math.min(timelineDuration, Math.max(0, value));
      setTimelineTime(nextTime);
      const video = videoRef.current;
      if (video && videoUrl) {
        video.currentTime = Math.min(videoTo, Math.max(videoFrom, videoFrom + nextTime));
      }
    },
    [timelineDuration, videoFrom, videoTo, videoUrl],
  );

  return (
    <div className="space-y-6">
      <Panel>
        <PanelHeader
          title={`Episode ${episode.episode_index + 1}`}
          description={detail.video_key ?? undefined}
          actions={
            <span className="text-2xs text-ink-muted">
              {detail.frames_truncated
                ? `${formatCount(detail.sampled_frame_count)} sampled / ${formatCount(episode.frame_count)} frames`
                : `${formatCount(episode.frame_count)} frames`}
            </span>
          }
        />

        <div className="bg-ink">
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
                <div
                  className="pointer-events-none absolute inset-0 grid place-items-center bg-ink/35 text-white"
                  role="status"
                >
                  <span className="inline-flex items-center gap-2 rounded-full bg-ink/70 px-3 py-1.5 text-xs">
                    <LoaderCircle className="h-4 w-4 animate-spin" aria-hidden="true" />
                    Loading video…
                  </span>
                </div>
              )}
            </div>
          ) : (
            <div className="grid aspect-video place-items-center px-6 text-center">
              <div>
                <VideoOff className="mx-auto h-7 w-7 text-white/50" aria-hidden="true" />
                <p className="mt-3 text-sm font-medium text-white/90">No episode video</p>
                <p className="mt-1 text-xs text-white/60">
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
            <Alert tone="danger" className="mb-4">
              {mediaError}
            </Alert>
          )}
          <div className="flex items-center justify-between gap-4 text-xs">
            <span className="font-mono font-medium tabular-nums text-ink">
              {formatPreciseDuration(timelineTime)} / {formatPreciseDuration(timelineDuration)}
            </span>
            <span className="text-ink-muted">
              {currentFrame ? `Frame ${formatCount(currentFrame.frame_index)}` : "No synchronized frame"}
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
              aria-valuetext={`${formatPreciseDuration(timelineTime)}, ${
                currentFrame ? `frame ${currentFrame.frame_index}` : "no frame"
              }`}
              className="h-1.5 w-full cursor-pointer accent-accent-600 disabled:cursor-not-allowed disabled:opacity-40"
            />
          </label>
          <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-2 text-2xs text-ink-muted">
            <span>{formatCount(detail.fps)} FPS</span>
            <span>Episode-relative timeline</span>
            {detail.frames_truncated && (
              <span title="The backend includes the first and last frame and evenly samples the frames between them.">
                Values sampled: {formatCount(detail.sampled_frame_count)} of{" "}
                {formatCount(episode.frame_count)}
              </span>
            )}
            {episode.video_from_timestamp !== null && episode.video_to_timestamp !== null && (
              <span title="Playback is constrained to this episode's segment in the packed video file.">
                Media segment {formatPreciseDuration(episode.video_from_timestamp)}–
                {formatPreciseDuration(episode.video_to_timestamp)}
              </span>
            )}
          </div>
        </div>
      </Panel>

      {currentFrame ? (
        <div className="grid gap-5 md:grid-cols-2">
          <VectorValues
            title="Observation state"
            names={detail.state_names}
            values={currentFrame.state}
          />
          <VectorValues title="Action" names={detail.action_names} values={currentFrame.action} />
        </div>
      ) : (
        <Alert tone="warning" title="No synchronized samples">
          The episode can be browsed, but its frame timeline is unavailable.
        </Alert>
      )}
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

  const retryEpisode =
    detailError?.status === 404 || detailError?.status === 409 ? loadDataset : retryDetail;
  const detailIsCurrent = Boolean(
    detail &&
      dataset &&
      detail.repo_id === dataset.repo_id &&
      detail.revision === dataset.revision &&
      detail.episode.episode_index === selectedEpisodeIndex,
  );

  return (
    <Page>
      <PageHeader
        back={{ to: "/datasets", label: "Datasets" }}
        title={dataset?.repo_id ?? repoName}
        description="Scrub camera playback with synchronized observation state and action values."
        meta={
          dataset ? (
            <>
              <Mono title={dataset.revision}>@{shortRevision(dataset.revision)}</Mono>
              <span className="text-xs text-ink-muted">
                {formatCount(dataset.total_episodes)} episode{dataset.total_episodes === 1 ? "" : "s"}
              </span>
              <span className="text-xs text-ink-muted">{formatCount(dataset.fps)} FPS</span>
              <span className="text-xs text-ink-faint">
                {dataset.video_key ? "Camera + robot data" : "Robot data only"}
              </span>
            </>
          ) : undefined
        }
        actions={
          dataset ? (
            <Button
              icon={RefreshCw}
              loading={datasetLoading}
              disabled={datasetLoading}
              onClick={() => void loadDataset()}
            >
              Reload
            </Button>
          ) : undefined
        }
      />

      <PageSection className="mt-8">
        {datasetLoading && (
          <div className="grid gap-5 lg:grid-cols-[18rem_minmax(0,1fr)]">
            <Skeleton className="h-80 w-full" />
            <Skeleton className="h-[32rem] w-full" />
          </div>
        )}

        {!datasetLoading && datasetError && (
          <LoadErrorState
            error={datasetError}
            resource="dataset"
            onRetry={() => void loadDataset()}
            busy={datasetLoading}
          />
        )}

        {!datasetLoading && dataset && dataset.episodes.length === 0 && (
          <EmptyState
            icon={Film}
            title="No episodes found"
            description="This LeRobot dataset does not contain any episodes that can be visualized."
          />
        )}

        {!datasetLoading && dataset && dataset.episodes.length > 0 && selectedEpisodeIndex !== null && (
          <div className="grid items-start gap-5 lg:grid-cols-[18rem_minmax(0,1fr)]">
            <EpisodePicker
              episodes={dataset.episodes}
              selectedEpisodeIndex={selectedEpisodeIndex}
              onSelect={selectEpisode}
            />
            <div className="min-w-0">
              {(detailLoading || (!detailError && !detailIsCurrent)) && (
                <Skeleton className="h-[32rem] w-full" />
              )}
              {!detailLoading && detailError && (
                <LoadErrorState
                  error={detailError}
                  resource="episode"
                  onRetry={() => void retryEpisode()}
                  busy={detailLoading || datasetLoading}
                />
              )}
              {!detailLoading && !detailError && detail && detailIsCurrent && (
                <EpisodePlayer detail={detail} />
              )}
            </div>
          </div>
        )}
      </PageSection>
    </Page>
  );
}

export function DatasetEpisodePage() {
  const { repoName } = useParams<{ repoName: string }>();
  if (!repoName) return <Navigate to="/datasets" replace />;
  return <DatasetEpisodeWorkspace key={repoName} repoName={repoName} />;
}
