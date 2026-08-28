from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator


class EpisodeAuthenticationError(RuntimeError):
    pass


class EpisodeHubError(RuntimeError):
    pass


class EpisodeDatasetNotFoundError(RuntimeError):
    pass


class EpisodeNotFoundError(RuntimeError):
    pass


class EpisodeFormatError(RuntimeError):
    pass


class ByteRangeError(ValueError):
    pass


@dataclass(frozen=True)
class EpisodeSummary:
    episode_index: int
    tasks: list[str]
    frame_count: int
    duration_seconds: float
    dataset_from_index: int
    dataset_to_index: int
    video_from_timestamp: float | None
    video_to_timestamp: float | None


@dataclass(frozen=True)
class TimelineFrame:
    timestamp: float
    frame_index: int
    state: list[float]
    action: list[float]


@dataclass(frozen=True)
class EpisodeList:
    repo_id: str
    revision: str
    fps: float
    state_names: list[str]
    action_names: list[str]
    video_key: str | None
    total_episodes: int
    episodes: list[EpisodeSummary]


@dataclass(frozen=True)
class EpisodeDetail:
    repo_id: str
    revision: str
    fps: float
    state_names: list[str]
    action_names: list[str]
    video_key: str | None
    episode: EpisodeSummary
    frames: list[TimelineFrame]
    video_url: str | None


@dataclass(frozen=True)
class VideoAsset:
    path: Path
    size: int
    etag: str
    revision: str


@dataclass(frozen=True)
class _Layout:
    fps: float
    total_episodes: int
    total_frames: int
    state_size: int
    action_size: int
    state_names: list[str]
    action_names: list[str]
    video_key: str | None
    data_path: str
    video_path: str | None


@dataclass(frozen=True)
class _EpisodeRecord:
    summary: EpisodeSummary
    data_chunk_index: int
    data_file_index: int
    video_chunk_index: int | None
    video_file_index: int | None


@dataclass(frozen=True)
class _LoadedDataset:
    repo_id: str
    revision: str
    layout: _Layout
    episodes: list[_EpisodeRecord]
    repo_files: set[str]


class HFEpisodeBrowser:
    """Reads LeRobot v3 episode data from one configured Hub namespace."""

    _REPO_SLUG = re.compile(
        r"[A-Za-z0-9_](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9_])?"
    )
    _REVISION = re.compile(r"[0-9a-fA-F]{40}")
    _EPISODES_FILE = re.compile(
        r"meta/episodes/chunk-(\d{3})/file-(\d{3})\.parquet"
    )
    _FEATURE_KEY = re.compile(r"[A-Za-z0-9_.-]{1,160}")
    _DATA_TEMPLATE = (
        "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    )
    _VIDEO_TEMPLATE = (
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    )

    def __init__(
        self,
        *,
        hub_api_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self._hub_api_factory_override = hub_api_factory

    def list_episodes(
        self,
        *,
        namespace: str,
        repo_name: str,
        token: str,
    ) -> EpisodeList:
        dataset = self._load_dataset(
            namespace=namespace,
            repo_name=repo_name,
            token=token,
            revision=None,
        )
        return EpisodeList(
            repo_id=dataset.repo_id,
            revision=dataset.revision,
            fps=dataset.layout.fps,
            state_names=dataset.layout.state_names,
            action_names=dataset.layout.action_names,
            video_key=dataset.layout.video_key,
            total_episodes=len(dataset.episodes),
            episodes=[record.summary for record in dataset.episodes],
        )

    def episode_detail(
        self,
        *,
        namespace: str,
        repo_name: str,
        episode_index: int,
        revision: str,
        token: str,
    ) -> EpisodeDetail:
        if not self._REVISION.fullmatch(revision):
            raise EpisodeFormatError("The dataset revision is invalid.")
        dataset = self._load_dataset(
            namespace=namespace,
            repo_name=repo_name,
            token=token,
            revision=revision,
        )
        episode = self._episode(dataset, episode_index)
        api = self._hub_api(token)
        frames = self._load_frames(api, dataset, episode, token)
        video_url = (
            f"/api/datasets/{repo_name}/episodes/{episode_index}/video"
            f"?revision={dataset.revision}"
            if episode.video_chunk_index is not None
            else None
        )
        return EpisodeDetail(
            repo_id=dataset.repo_id,
            revision=dataset.revision,
            fps=dataset.layout.fps,
            state_names=dataset.layout.state_names,
            action_names=dataset.layout.action_names,
            video_key=dataset.layout.video_key,
            episode=episode.summary,
            frames=frames,
            video_url=video_url,
        )

    def video_asset(
        self,
        *,
        namespace: str,
        repo_name: str,
        episode_index: int,
        revision: str,
        token: str,
    ) -> VideoAsset:
        if not self._REVISION.fullmatch(revision):
            raise EpisodeFormatError("The dataset revision is invalid.")
        dataset = self._load_dataset(
            namespace=namespace,
            repo_name=repo_name,
            token=token,
            revision=revision,
        )
        episode = self._episode(dataset, episode_index)
        if (
            dataset.layout.video_key is None
            or episode.video_chunk_index is None
            or episode.video_file_index is None
        ):
            raise EpisodeNotFoundError("This episode has no video stream.")
        video_path = self._format_video_path(
            dataset.layout,
            chunk_index=episode.video_chunk_index,
            file_index=episode.video_file_index,
        )
        if video_path not in dataset.repo_files:
            raise EpisodeFormatError("The episode video artifact is missing.")
        api = self._hub_api(token)
        path = self._download_required(
            api,
            repo_id=dataset.repo_id,
            filename=video_path,
            revision=dataset.revision,
            token=token,
        )
        try:
            size = path.stat().st_size
        except OSError as error:
            raise EpisodeHubError("The episode video could not be opened.") from error
        if size <= 0:
            raise EpisodeFormatError("The episode video artifact is empty.")
        digest = hashlib.sha256(
            f"{dataset.revision}:{video_path}:{size}".encode("utf-8")
        ).hexdigest()
        return VideoAsset(
            path=path,
            size=size,
            etag=f'"{digest}"',
            revision=dataset.revision,
        )

    def _load_dataset(
        self,
        *,
        namespace: str,
        repo_name: str,
        token: str,
        revision: str | None,
    ) -> _LoadedDataset:
        if (
            not self._REPO_SLUG.fullmatch(repo_name)
            or ".." in repo_name
            or "--" in repo_name
        ):
            raise EpisodeFormatError("The dataset repository name is invalid.")
        repo_id = f"{namespace}/{repo_name}"
        api = self._hub_api(token)
        self._validate_namespace(api, namespace, token)
        resolved_revision = self._resolve_revision(
            api,
            repo_id=repo_id,
            revision=revision,
            token=token,
        )
        info_path = self._download_required(
            api,
            repo_id=repo_id,
            filename="meta/info.json",
            revision=resolved_revision,
            token=token,
        )
        layout = self._parse_layout(info_path)
        repo_files = self._repo_files(
            api,
            repo_id=repo_id,
            revision=resolved_revision,
            token=token,
        )
        episode_paths = sorted(
            path for path in repo_files if self._EPISODES_FILE.fullmatch(path)
        )
        if len(episode_paths) > 1024:
            raise EpisodeFormatError("The LeRobot episode index is missing or too large.")
        if layout.total_episodes > 0 and not episode_paths:
            raise EpisodeFormatError("The LeRobot episode index is missing or too large.")
        episodes: list[_EpisodeRecord] = []
        for episode_path in episode_paths:
            local_path = self._download_required(
                api,
                repo_id=repo_id,
                filename=episode_path,
                revision=resolved_revision,
                token=token,
            )
            episodes.extend(self._read_episode_records(local_path, layout))
        episodes.sort(key=lambda record: record.summary.episode_index)
        self._validate_episode_index(episodes, layout)
        return _LoadedDataset(
            repo_id=repo_id,
            revision=resolved_revision,
            layout=layout,
            episodes=episodes,
            repo_files=repo_files,
        )

    def _resolve_revision(
        self,
        api: Any,
        *,
        repo_id: str,
        revision: str | None,
        token: str,
    ) -> str:
        try:
            info = api.dataset_info(
                repo_id=repo_id,
                revision=revision,
                expand=["sha", "tags"],
                token=token,
            )
        except Exception as error:
            self._raise_hub_error(error, not_found_message="Dataset was not found.")
        returned_id = getattr(info, "id", None)
        sha = getattr(info, "sha", None)
        tags = getattr(info, "tags", None)
        if returned_id != repo_id or not isinstance(sha, str) or not self._REVISION.fullmatch(sha):
            raise EpisodeFormatError("The Hub returned an invalid dataset revision.")
        if revision is not None and sha.casefold() != revision.casefold():
            raise EpisodeFormatError("The requested dataset revision did not resolve exactly.")
        if not isinstance(tags, (list, tuple)) or not any(
            isinstance(tag, str) and tag.casefold() == "lerobot" for tag in tags
        ):
            raise EpisodeFormatError("The repository is not a LeRobot dataset.")
        return sha

    def _repo_files(
        self,
        api: Any,
        *,
        repo_id: str,
        revision: str,
        token: str,
    ) -> set[str]:
        try:
            paths = api.list_repo_files(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                token=token,
            )
        except Exception as error:
            self._raise_hub_error(error)
        if not isinstance(paths, list) or len(paths) > 100_000:
            raise EpisodeFormatError("The dataset file index is invalid or too large.")
        return {path for path in paths if isinstance(path, str)}

    def _download_required(
        self,
        api: Any,
        *,
        repo_id: str,
        filename: str,
        revision: str,
        token: str,
    ) -> Path:
        if not self._safe_hub_path(filename):
            raise EpisodeFormatError("The dataset contains an unsafe artifact path.")
        try:
            path = Path(
                api.hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type="dataset",
                    revision=revision,
                    token=token,
                )
            )
        except Exception as error:
            self._raise_hub_error(error)
        if not path.is_file():
            raise EpisodeHubError("A required dataset artifact is unavailable.")
        return path

    @classmethod
    def _parse_layout(cls, path: Path) -> _Layout:
        try:
            if path.stat().st_size > 2 * 1024 * 1024:
                raise EpisodeFormatError("The LeRobot metadata file is too large.")
            raw = json.loads(path.read_text(encoding="utf-8"))
        except EpisodeFormatError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise EpisodeFormatError("The LeRobot metadata file is malformed.") from error
        version = raw.get("codebase_version") if isinstance(raw, dict) else None
        if not isinstance(version, str) or not version.startswith("v3."):
            raise EpisodeFormatError("Only LeRobot v3 datasets are supported.")
        fps = cls._positive_float(raw.get("fps"), "dataset FPS")
        total_episodes = cls._nonnegative_int(raw.get("total_episodes"), "episode count")
        total_frames = cls._nonnegative_int(raw.get("total_frames"), "frame count")
        if total_episodes > 100_000 or total_frames > 100_000_000:
            raise EpisodeFormatError("The dataset dimensions exceed supported bounds.")
        features = raw.get("features")
        if not isinstance(features, dict):
            raise EpisodeFormatError("The dataset feature metadata is missing.")
        state_size, state_names = cls._vector_feature(
            features.get("observation.state"), "state"
        )
        action_size, action_names = cls._vector_feature(
            features.get("action"), "action"
        )
        video_keys = sorted(
            key
            for key, feature in features.items()
            if isinstance(key, str)
            and cls._FEATURE_KEY.fullmatch(key)
            and isinstance(feature, dict)
            and feature.get("dtype") == "video"
        )
        video_key = video_keys[0] if video_keys else None
        data_path = raw.get("data_path")
        if data_path != cls._DATA_TEMPLATE:
            raise EpisodeFormatError("The dataset data path is not a safe LeRobot v3 template.")
        video_path = raw.get("video_path")
        if video_key is not None and video_path != cls._VIDEO_TEMPLATE:
            raise EpisodeFormatError("The dataset video path is not a safe LeRobot v3 template.")
        return _Layout(
            fps=fps,
            total_episodes=total_episodes,
            total_frames=total_frames,
            state_size=state_size,
            action_size=action_size,
            state_names=state_names,
            action_names=action_names,
            video_key=video_key,
            data_path=data_path,
            video_path=video_path if video_key is not None else None,
        )

    @classmethod
    def _read_episode_records(
        cls, path: Path, layout: _Layout
    ) -> list[_EpisodeRecord]:
        import pyarrow.parquet as pq

        columns = [
            "episode_index",
            "tasks",
            "length",
            "data/chunk_index",
            "data/file_index",
            "dataset_from_index",
            "dataset_to_index",
        ]
        if layout.video_key is not None:
            prefix = f"videos/{layout.video_key}"
            columns.extend(
                [
                    f"{prefix}/chunk_index",
                    f"{prefix}/file_index",
                    f"{prefix}/from_timestamp",
                    f"{prefix}/to_timestamp",
                ]
            )
        try:
            rows = pq.read_table(path, columns=columns).to_pylist()
        except Exception as error:
            raise EpisodeFormatError("The LeRobot episode index is malformed.") from error
        records: list[_EpisodeRecord] = []
        for row in rows:
            episode_index = cls._nonnegative_int(row.get("episode_index"), "episode index")
            frame_count = cls._positive_int(row.get("length"), "episode length")
            data_chunk = cls._bounded_index(row.get("data/chunk_index"), "data chunk")
            data_file = cls._bounded_index(row.get("data/file_index"), "data file")
            data_from = cls._nonnegative_int(row.get("dataset_from_index"), "data start")
            data_to = cls._positive_int(row.get("dataset_to_index"), "data end")
            if data_to - data_from != frame_count:
                raise EpisodeFormatError("Episode row bounds do not match its frame count.")
            tasks = cls._tasks(row.get("tasks"))
            video_chunk: int | None = None
            video_file: int | None = None
            video_from: float | None = None
            video_to: float | None = None
            if layout.video_key is not None:
                prefix = f"videos/{layout.video_key}"
                video_values = [
                    row.get(f"{prefix}/chunk_index"),
                    row.get(f"{prefix}/file_index"),
                    row.get(f"{prefix}/from_timestamp"),
                    row.get(f"{prefix}/to_timestamp"),
                ]
                if any(value is not None for value in video_values):
                    if any(value is None for value in video_values):
                        raise EpisodeFormatError("Episode video bounds are incomplete.")
                    video_chunk = cls._bounded_index(video_values[0], "video chunk")
                    video_file = cls._bounded_index(video_values[1], "video file")
                    video_from = cls._nonnegative_float(video_values[2], "video start")
                    video_to = cls._positive_float(video_values[3], "video end")
                    if video_to <= video_from:
                        raise EpisodeFormatError("Episode video timestamps are invalid.")
            duration = (
                video_to - video_from
                if video_from is not None and video_to is not None
                else frame_count / layout.fps
            )
            records.append(
                _EpisodeRecord(
                    summary=EpisodeSummary(
                        episode_index=episode_index,
                        tasks=tasks,
                        frame_count=frame_count,
                        duration_seconds=duration,
                        dataset_from_index=data_from,
                        dataset_to_index=data_to,
                        video_from_timestamp=video_from,
                        video_to_timestamp=video_to,
                    ),
                    data_chunk_index=data_chunk,
                    data_file_index=data_file,
                    video_chunk_index=video_chunk,
                    video_file_index=video_file,
                )
            )
        return records

    @classmethod
    def _validate_episode_index(
        cls, episodes: list[_EpisodeRecord], layout: _Layout
    ) -> None:
        if len(episodes) != layout.total_episodes or [
            record.summary.episode_index for record in episodes
        ] != list(range(layout.total_episodes)):
            raise EpisodeFormatError("Episode indices do not match dataset metadata.")
        previous_end = 0
        for record in episodes:
            summary = record.summary
            if summary.dataset_from_index != previous_end:
                raise EpisodeFormatError("Episode data bounds are not contiguous.")
            previous_end = summary.dataset_to_index
        if previous_end != layout.total_frames:
            raise EpisodeFormatError("Episode frame bounds do not match dataset metadata.")

    def _load_frames(
        self,
        api: Any,
        dataset: _LoadedDataset,
        episode: _EpisodeRecord,
        token: str,
    ) -> list[TimelineFrame]:
        import pyarrow.parquet as pq

        data_path = self._format_data_path(
            episode.data_chunk_index, episode.data_file_index
        )
        if data_path not in dataset.repo_files:
            raise EpisodeFormatError("The episode data artifact is missing.")
        path = self._download_required(
            api,
            repo_id=dataset.repo_id,
            filename=data_path,
            revision=dataset.revision,
            token=token,
        )
        same_file = [
            record
            for record in dataset.episodes
            if record.data_chunk_index == episode.data_chunk_index
            and record.data_file_index == episode.data_file_index
        ]
        file_from = min(record.summary.dataset_from_index for record in same_file)
        local_from = episode.summary.dataset_from_index - file_from
        columns = [
            "observation.state",
            "action",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
        ]
        try:
            table = pq.read_table(path, columns=columns)
            if local_from < 0 or local_from + episode.summary.frame_count > table.num_rows:
                raise EpisodeFormatError("Episode bounds exceed its data parquet file.")
            rows = table.slice(local_from, episode.summary.frame_count).to_pylist()
        except EpisodeFormatError:
            raise
        except Exception as error:
            raise EpisodeFormatError("The episode data parquet is malformed.") from error
        frames: list[TimelineFrame] = []
        previous_timestamp = -math.inf
        for offset, row in enumerate(rows):
            expected_index = episode.summary.dataset_from_index + offset
            if (
                self._exact_int(row.get("episode_index"))
                != episode.summary.episode_index
                or self._exact_int(row.get("frame_index")) != offset
                or self._exact_int(row.get("index")) != expected_index
            ):
                raise EpisodeFormatError("Episode frame indices are inconsistent.")
            timestamp = self._nonnegative_float(row.get("timestamp"), "frame timestamp")
            if timestamp <= previous_timestamp:
                raise EpisodeFormatError("Episode frame timestamps are not monotonic.")
            previous_timestamp = timestamp
            if timestamp > episode.summary.duration_seconds + (1.0 / dataset.layout.fps):
                raise EpisodeFormatError("Episode frame timestamp exceeds its duration.")
            frames.append(
                TimelineFrame(
                    timestamp=timestamp,
                    frame_index=offset,
                    state=self._vector(
                        row.get("observation.state"),
                        dataset.layout.state_size,
                        "state",
                    ),
                    action=self._vector(
                        row.get("action"),
                        dataset.layout.action_size,
                        "action",
                    ),
                )
            )
        if len(frames) != episode.summary.frame_count:
            raise EpisodeFormatError("Episode data row count does not match metadata.")
        return frames

    @staticmethod
    def parse_range(value: str | None, size: int) -> tuple[int, int] | None:
        if size <= 0:
            raise ByteRangeError("The video byte range is unsatisfiable.")
        if value is None:
            return None
        if len(value) > 128:
            raise ByteRangeError("The video byte range is invalid.")
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", value.strip())
        if match is None or not (match.group(1) or match.group(2)):
            raise ByteRangeError("The video byte range is invalid.")
        first, last = match.groups()
        try:
            if first:
                start = int(first)
                end = size - 1 if not last else int(last)
                if start >= size or end < start:
                    raise ByteRangeError("The video byte range is unsatisfiable.")
                end = min(end, size - 1)
            else:
                suffix = int(last)
                if suffix <= 0:
                    raise ByteRangeError("The video byte range is unsatisfiable.")
                start = max(size - suffix, 0)
                end = size - 1
        except ValueError as error:
            raise ByteRangeError("The video byte range is invalid.") from error
        return start, end

    @staticmethod
    def stream_file(
        path: Path, *, start: int, length: int, chunk_size: int = 64 * 1024
    ) -> Iterator[bytes]:
        with path.open("rb") as stream:
            stream.seek(start)
            remaining = length
            while remaining:
                chunk = stream.read(min(chunk_size, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    @staticmethod
    def _episode(dataset: _LoadedDataset, episode_index: int) -> _EpisodeRecord:
        if episode_index < 0 or episode_index >= len(dataset.episodes):
            raise EpisodeNotFoundError("Episode was not found.")
        record = dataset.episodes[episode_index]
        if record.summary.episode_index != episode_index:
            raise EpisodeNotFoundError("Episode was not found.")
        return record

    @classmethod
    def _format_data_path(cls, chunk_index: int, file_index: int) -> str:
        return cls._DATA_TEMPLATE.format(
            chunk_index=chunk_index, file_index=file_index
        )

    @classmethod
    def _format_video_path(
        cls, layout: _Layout, *, chunk_index: int, file_index: int
    ) -> str:
        if layout.video_key is None or layout.video_path != cls._VIDEO_TEMPLATE:
            raise EpisodeFormatError("The dataset has no safe video path.")
        return cls._VIDEO_TEMPLATE.format(
            video_key=layout.video_key,
            chunk_index=chunk_index,
            file_index=file_index,
        )

    @classmethod
    def _vector_feature(
        cls, value: Any, label: str
    ) -> tuple[int, list[str]]:
        if not isinstance(value, dict):
            raise EpisodeFormatError(f"The {label} feature metadata is missing.")
        shape = value.get("shape")
        if (
            not isinstance(shape, (list, tuple))
            or len(shape) != 1
            or isinstance(shape[0], bool)
            or not isinstance(shape[0], int)
            or not 1 <= shape[0] <= 1024
        ):
            raise EpisodeFormatError(f"The {label} feature dimension is invalid.")
        size = shape[0]
        names = value.get("names")
        if names is None:
            return size, [f"{label}_{index}" for index in range(size)]
        if (
            not isinstance(names, (list, tuple))
            or len(names) != size
            or not all(isinstance(name, str) and name.strip() for name in names)
        ):
            raise EpisodeFormatError(f"The {label} feature names are invalid.")
        return size, [name.strip() for name in names]

    @staticmethod
    def _vector(value: Any, expected_size: int, label: str) -> list[float]:
        if not isinstance(value, (list, tuple)) or len(value) != expected_size:
            raise EpisodeFormatError(f"An episode {label} vector has invalid dimensions.")
        result: list[float] = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise EpisodeFormatError(f"An episode {label} vector is not numeric.")
            number = float(item)
            if not math.isfinite(number):
                raise EpisodeFormatError(f"An episode {label} vector is not finite.")
            result.append(number)
        return result

    @staticmethod
    def _tasks(value: Any) -> list[str]:
        if not isinstance(value, (list, tuple)):
            return []
        return [task.strip() for task in value if isinstance(task, str) and task.strip()]

    @staticmethod
    def _safe_hub_path(value: str) -> bool:
        path = Path(value)
        return (
            bool(value)
            and not path.is_absolute()
            and "\\" not in value
            and all(part not in {"", ".", ".."} for part in path.parts)
        )

    @staticmethod
    def _positive_float(value: Any, label: str) -> float:
        number = HFEpisodeBrowser._nonnegative_float(value, label)
        if number <= 0:
            raise EpisodeFormatError(f"The {label} must be positive.")
        return number

    @staticmethod
    def _nonnegative_float(value: Any, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EpisodeFormatError(f"The {label} is invalid.")
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise EpisodeFormatError(f"The {label} is invalid.")
        return number

    @staticmethod
    def _positive_int(value: Any, label: str) -> int:
        number = HFEpisodeBrowser._nonnegative_int(value, label)
        if number <= 0:
            raise EpisodeFormatError(f"The {label} must be positive.")
        return number

    @staticmethod
    def _nonnegative_int(value: Any, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise EpisodeFormatError(f"The {label} is invalid.")
        return value

    @staticmethod
    def _bounded_index(value: Any, label: str) -> int:
        number = HFEpisodeBrowser._nonnegative_int(value, label)
        if number > 999_999:
            raise EpisodeFormatError(f"The {label} exceeds supported bounds.")
        return number

    @staticmethod
    def _exact_int(value: Any) -> int | None:
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return value

    @staticmethod
    def _validate_namespace(api: Any, namespace: str, token: str) -> None:
        try:
            identity = api.whoami(token=token)
        except Exception as error:
            HFEpisodeBrowser._raise_hub_error(error)
        if not isinstance(identity, dict):
            raise EpisodeHubError("Hugging Face returned an invalid identity response.")
        organizations = identity.get("orgs", [])
        if not isinstance(organizations, (list, tuple)):
            raise EpisodeHubError("Hugging Face returned an invalid identity response.")
        available = {str(identity.get("name", "")).casefold()}
        for organization in organizations:
            name = organization.get("name") if isinstance(organization, dict) else organization
            if name:
                available.add(str(name).casefold())
        if namespace.casefold() not in available:
            raise EpisodeAuthenticationError(
                "HF_NAMESPACE is not the authenticated user or one of its organizations."
            )

    @staticmethod
    def _raise_hub_error(
        error: Exception,
        *,
        not_found_message: str | None = None,
    ) -> None:
        status_code = getattr(getattr(error, "response", None), "status_code", None)
        if status_code in {401, 403}:
            raise EpisodeAuthenticationError("HF_TOKEN could not be authenticated.") from error
        if status_code == 404 and not_found_message is not None:
            raise EpisodeDatasetNotFoundError(not_found_message) from error
        raise EpisodeHubError("Hugging Face dataset access failed.") from error

    def _hub_api(self, token: str) -> Any:
        if self._hub_api_factory_override is not None:
            return self._hub_api_factory_override(token)
        from huggingface_hub import HfApi

        return HfApi(token=token)
