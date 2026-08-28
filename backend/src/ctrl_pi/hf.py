from __future__ import annotations

import contextlib
import json
import math
import shutil
import threading
import uuid
from dataclasses import dataclass
from itertools import chain, zip_longest
from pathlib import Path
from typing import Any, Callable, Iterator

from ctrl_pi.drivers.yam import JOINT_NAMES

DATASET_MARKER_FILENAME = ".ctrl-pi.json"
DATASET_MARKER_SCHEMA = "ctrl-pi.recording-dataset"
DATASET_MARKER_VERSION = 1


class DatasetConversionError(RuntimeError):
    """A safe, user-displayable staged-artifact error."""


class HubAuthenticationError(RuntimeError):
    """A safe, user-displayable Hub identity or namespace error."""


class HubUploadError(RuntimeError):
    """A safe, user-displayable Hub transfer error."""

    def __init__(self, message: str, *, remote_repo_created: bool = False) -> None:
        super().__init__(message)
        self.remote_repo_created = remote_repo_created


class UploadConflictError(RuntimeError):
    pass


@dataclass(frozen=True)
class RecordingUploadSource:
    recording_id: str
    task: str
    episode_count: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ConvertedDataset:
    dataset: Any
    root: Path
    artifact_key: str
    fps: int
    total_frames: int


@dataclass(frozen=True)
class DatasetUploadResult:
    repo_id: str
    repo_url: str
    revision: str | None
    total_frames: int
    fps: int


@dataclass(frozen=True)
class _EpisodeSource:
    index: int
    directory: Path
    video_path: Path
    samples_path: Path
    artifact_key: str
    sample_count: int
    fps: int


class HFDatasetUploader:
    """Converts ctrl-π staging into LeRobot v3 and uploads with explicit auth."""

    def __init__(
        self,
        recording_staging_dir: Path,
        dataset_staging_dir: Path | None = None,
        *,
        dataset_class: Any | None = None,
        hub_api_factory: Callable[[str], Any] | None = None,
        card_factory: Callable[..., Any] | None = None,
        frame_decoder: Callable[[Path], Iterator[Any]] | None = None,
        dataset_verifier: Callable[[str, Path, int, int], None] | None = None,
    ) -> None:
        self.recording_staging_dir = recording_staging_dir
        self.dataset_staging_dir = (
            dataset_staging_dir
            if dataset_staging_dir is not None
            else recording_staging_dir.parent / "lerobot"
        )
        self._dataset_class_override = dataset_class
        self._hub_api_factory_override = hub_api_factory
        self._card_factory_override = card_factory
        self._frame_decoder_override = frame_decoder
        self._dataset_verifier_override = dataset_verifier
        self._active: set[str] = set()
        self._active_repositories: set[str] = set()
        self._active_lock = threading.Lock()

    def reserve(self, recording_id: str, repo_id: str) -> None:
        with self._active_lock:
            if recording_id in self._active:
                raise UploadConflictError("this recording is already uploading")
            if repo_id in self._active_repositories:
                raise UploadConflictError("this dataset repository is already uploading")
            self._active.add(recording_id)
            self._active_repositories.add(repo_id)

    def release(self, recording_id: str, repo_id: str) -> None:
        with self._active_lock:
            self._active.discard(recording_id)
            self._active_repositories.discard(repo_id)

    def is_active(self, recording_id: str) -> bool:
        with self._active_lock:
            return recording_id in self._active

    def repository_owned_by(
        self,
        *,
        repo_id: str,
        recording_id: str,
        token: str,
    ) -> bool:
        """Read and validate ctrl-pi ownership without mutating the Hub repo."""

        try:
            api = self._hub_api(token)
            self._verify_remote_marker(
                api,
                repo_id=repo_id,
                recording_id=recording_id,
                token=token,
                ownership_check=True,
            )
        except (UploadConflictError, HubUploadError):
            return False
        return True

    @staticmethod
    def repo_id(namespace: str, repo_name: str) -> str:
        if "/" in repo_name:
            raise ValueError("repo_name must be a slug without a namespace")
        try:
            from huggingface_hub.utils import HFValidationError, validate_repo_id

            candidate = f"{namespace}/{repo_name}"
            validate_repo_id(candidate)
        except HFValidationError as error:
            raise ValueError("repo_name is not a valid Hugging Face repository slug") from error
        return candidate

    def upload(
        self,
        source: RecordingUploadSource,
        namespace: str,
        repo_name: str,
        private: bool,
        token: str,
    ) -> DatasetUploadResult:
        repo_id = self.repo_id(namespace, repo_name)
        converted = self.convert(source, repo_id=repo_id, repo_name=repo_name)
        remote_repo_created = False
        try:
            api = self._hub_api(token)
            self._validate_namespace(api, namespace)
            previous_upload = source.metadata.get("upload", {})
            trusted_intent = (
                isinstance(previous_upload, dict)
                and previous_upload.get("repo_id") == repo_id
                and previous_upload.get("owner_recording_id") == source.recording_id
            )
            remote_repo_created = bool(
                trusted_intent and previous_upload.get("remote_repo_created") is True
            )
            repo_exists = api.repo_exists(repo_id=repo_id, repo_type="dataset")
            if repo_exists and not trusted_intent:
                raise UploadConflictError(
                    "The target dataset repository already exists and is not associated with this recording."
                )
            card = self._card_factory()(
                tags=["ctrl-pi", "yam"],
                dataset_info=converted.dataset.meta.info,
                repo_id=repo_id,
                dataset_description=source.task,
            )
            card.save(converted.root / "README.md")
            self._write_dataset_marker(converted.root, source.recording_id)
            marker_path = converted.root / DATASET_MARKER_FILENAME
            if repo_exists:
                self._verify_remote_marker(
                    api,
                    repo_id=repo_id,
                    recording_id=source.recording_id,
                    token=token,
                    ownership_check=True,
                    bootstrap_marker_path=marker_path,
                )
                remote_repo_created = True
                repo_url = f"https://huggingface.co/datasets/{repo_id}"
            else:
                try:
                    repo_url = api.create_repo(
                        repo_id=repo_id,
                        repo_type="dataset",
                        private=private,
                        exist_ok=False,
                    )
                except Exception as error:
                    if getattr(
                        getattr(error, "response", None), "status_code", None
                    ) == 409:
                        raise UploadConflictError(
                            "The target dataset repository was created by another process."
                        ) from error
                    raise
                remote_repo_created = True
                self._upload_remote_marker(
                    api,
                    repo_id=repo_id,
                    marker_path=marker_path,
                    token=token,
                )
                self._verify_remote_marker(
                    api,
                    repo_id=repo_id,
                    recording_id=source.recording_id,
                    token=token,
                )
            api.update_repo_settings(
                repo_id=repo_id,
                repo_type="dataset",
                private=private,
            )
            commit = api.upload_folder(
                repo_id=repo_id,
                repo_type="dataset",
                folder_path=converted.root,
                ignore_patterns=["images/"],
                commit_message="Upload ctrl-pi LeRobot dataset",
            )
            revision = getattr(commit, "oid", None)
            if not revision:
                raise HubUploadError("Hugging Face did not return an upload revision.")
            remote_info = api.repo_info(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
            )
            if getattr(remote_info, "sha", None) != revision:
                raise HubUploadError("The uploaded dataset revision could not be verified.")
            remote_files = set(
                api.list_repo_files(
                    repo_id=repo_id,
                    repo_type="dataset",
                    revision=revision,
                )
            )
            if (
                DATASET_MARKER_FILENAME not in remote_files
                or "README.md" not in remote_files
                or "meta/info.json" not in remote_files
                or not any(path.startswith("data/") and path.endswith(".parquet") for path in remote_files)
                or not any(path.startswith("videos/") and path.endswith(".mp4") for path in remote_files)
            ):
                raise HubUploadError("The uploaded dataset is missing required LeRobot artifacts.")
            self._verify_remote_marker(
                api,
                repo_id=repo_id,
                recording_id=source.recording_id,
                token=token,
                revision=revision,
            )
        except (HubAuthenticationError, UploadConflictError):
            raise
        except HubUploadError as error:
            if error.remote_repo_created == remote_repo_created:
                raise
            raise HubUploadError(
                str(error), remote_repo_created=remote_repo_created
            ) from error
        except Exception as error:
            raise HubUploadError(
                "Hugging Face upload failed; verify the token, namespace permissions, and network.",
                remote_repo_created=remote_repo_created,
            ) from error
        finally:
            self._remove_generated_dataset(converted.root)

        return DatasetUploadResult(
            repo_id=repo_id,
            repo_url=str(repo_url),
            revision=revision,
            total_frames=converted.total_frames,
            fps=converted.fps,
        )

    def convert(
        self,
        source: RecordingUploadSource,
        repo_id: str,
        repo_name: str,
    ) -> ConvertedDataset:
        episodes = self._episode_sources(source)
        fps_values = {episode.fps for episode in episodes}
        if len(fps_values) != 1:
            raise DatasetConversionError(
                "All episodes in one LeRobot dataset must use the same recording FPS."
            )
        fps = fps_values.pop()

        first_frames = self._decode_frames(episodes[0].video_path)
        try:
            first_frame = next(first_frames)
        except StopIteration as error:
            raise DatasetConversionError("The first staged episode contains no video frames.") from error
        height, width, channels = self._frame_shape(first_frame)
        if channels != 3:
            raise DatasetConversionError("Staged camera video must decode to three RGB channels.")

        output_parent = self.dataset_staging_dir / source.recording_id
        output_parent.mkdir(parents=True, exist_ok=True)
        directory_name = f"{repo_name}-{uuid.uuid4().hex[:12]}"
        dataset_root = output_parent / directory_name
        artifact_key = f"{source.recording_id}/{directory_name}"
        dataset = None
        try:
            dataset = self._dataset_class().create(
                repo_id=repo_id,
                fps=fps,
                features={
                    "observation.state": {
                        "dtype": "float32",
                        "shape": (len(JOINT_NAMES) + 1,),
                        "names": [*JOINT_NAMES, "gripper"],
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": (len(JOINT_NAMES) + 1,),
                        "names": [*JOINT_NAMES, "gripper"],
                    },
                    "observation.images.workspace": {
                        "dtype": "video",
                        "shape": (height, width, channels),
                        "names": ["height", "width", "channels"],
                    },
                },
                root=dataset_root,
                robot_type="yam",
                use_videos=True,
                vcodec="h264",
                image_writer_threads=0,
            )
            total_frames = 0
            for episode_number, episode in enumerate(episodes):
                frames = (
                    first_frames
                    if episode_number == 0
                    else self._decode_frames(episode.video_path)
                )
                first = first_frame if episode_number == 0 else None
                total_frames += self._add_episode(
                    dataset,
                    episode,
                    source.task,
                    frames,
                    expected_shape=(height, width, channels),
                    first_frame=first,
                )
                dataset.save_episode(parallel_encoding=False)
            dataset.finalize()
            self._verify_dataset(
                repo_id,
                dataset_root,
                expected_episodes=len(episodes),
                expected_frames=total_frames,
            )
        except DatasetConversionError:
            if dataset is not None:
                with contextlib.suppress(Exception):
                    dataset.finalize()
            self._remove_generated_dataset(dataset_root)
            raise
        except Exception as error:
            if dataset is not None:
                with contextlib.suppress(Exception):
                    dataset.finalize()
            self._remove_generated_dataset(dataset_root)
            raise DatasetConversionError(
                "Could not convert staged episodes with LeRobot 0.4.4."
            ) from error

        return ConvertedDataset(
            dataset=dataset,
            root=dataset_root,
            artifact_key=artifact_key,
            fps=fps,
            total_frames=total_frames,
        )

    def _episode_sources(self, source: RecordingUploadSource) -> list[_EpisodeSource]:
        raw_episodes = source.metadata.get("episodes")
        if not isinstance(raw_episodes, list) or not raw_episodes:
            raise DatasetConversionError("The recording has no finalized episode artifacts.")
        if len(raw_episodes) != source.episode_count:
            raise DatasetConversionError("Episode metadata does not match the recording episode count.")

        staging_root = self.recording_staging_dir.resolve()
        episodes: list[_EpisodeSource] = []
        seen_indices: set[int] = set()
        for raw in raw_episodes:
            if not isinstance(raw, dict):
                raise DatasetConversionError("Episode metadata is invalid.")
            try:
                index = int(raw["index"])
                sample_count = int(raw["sample_count"])
                artifact_key = str(raw["artifact_key"])
                fps = self._episode_fps(raw)
            except (KeyError, TypeError, ValueError) as error:
                raise DatasetConversionError("Episode metadata is incomplete.") from error
            if index < 0 or index in seen_indices or sample_count <= 0:
                raise DatasetConversionError("Episode indices and sample counts must be valid.")
            seen_indices.add(index)

            relative = Path(artifact_key)
            directory = (staging_root / relative).resolve()
            if (
                relative.is_absolute()
                or not directory.is_relative_to(staging_root)
                or not relative.parts
                or relative.parts[0] != source.recording_id
            ):
                raise DatasetConversionError("Episode artifact key is outside recording staging.")
            video_path = (directory / "video.mp4").resolve()
            samples_path = (directory / "samples.jsonl").resolve()
            if (
                not video_path.is_relative_to(directory)
                or not samples_path.is_relative_to(directory)
                or not video_path.is_file()
                or not samples_path.is_file()
            ):
                raise DatasetConversionError("A finalized episode artifact is missing.")
            episodes.append(
                _EpisodeSource(
                    index=index,
                    directory=directory,
                    video_path=video_path,
                    samples_path=samples_path,
                    artifact_key=artifact_key,
                    sample_count=sample_count,
                    fps=fps,
                )
            )
        return sorted(episodes, key=lambda item: item.index)

    @staticmethod
    def _episode_fps(raw: dict[str, Any]) -> int:
        if "fps" in raw:
            fps = int(raw["fps"])
        else:
            duration = float(raw["duration_seconds"])
            if duration <= 0:
                raise ValueError("duration must be positive")
            fps = round(int(raw["sample_count"]) / duration)
        if not 1 <= fps <= 60:
            raise ValueError("fps is outside supported bounds")
        return fps

    def _add_episode(
        self,
        dataset: Any,
        episode: _EpisodeSource,
        task: str,
        frames: Iterator[Any],
        expected_shape: tuple[int, int, int],
        first_frame: Any | None,
    ) -> int:
        if first_frame is not None:
            frames = chain((first_frame,), frames)
        sentinel = object()
        count = 0
        previous_timestamp = -math.inf
        for sample, frame in zip_longest(
            self._samples(episode.samples_path),
            frames,
            fillvalue=sentinel,
        ):
            if sample is sentinel or frame is sentinel:
                raise DatasetConversionError(
                    f"Episode {episode.index} video and sample counts do not match."
                )
            if self._frame_shape(frame) != expected_shape:
                raise DatasetConversionError("Camera dimensions changed between episodes.")
            try:
                frame_index = int(sample["frame_index"])
                timestamp = float(sample["timestamp_seconds"])
            except (KeyError, TypeError, ValueError) as error:
                raise DatasetConversionError(
                    f"Episode {episode.index} has invalid frame timing metadata."
                ) from error
            if (
                frame_index != count
                or not math.isfinite(timestamp)
                or timestamp < 0
                or timestamp <= previous_timestamp
            ):
                raise DatasetConversionError(
                    f"Episode {episode.index} frame indices/timestamps are not monotonic."
                )
            previous_timestamp = timestamp
            observation, action = self._vectors(sample)
            dataset.add_frame(
                {
                    "observation.state": observation,
                    "action": action,
                    "observation.images.workspace": frame,
                    "task": task,
                }
            )
            count += 1
        if count != episode.sample_count:
            raise DatasetConversionError(
                f"Episode {episode.index} sample count does not match its metadata."
            )
        return count

    @staticmethod
    def _samples(path: Path) -> Iterator[dict[str, Any]]:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as error:
                    raise DatasetConversionError("An episode sample file contains invalid JSON.") from error
                if not isinstance(sample, dict):
                    raise DatasetConversionError("An episode sample must be a JSON object.")
                yield sample

    @staticmethod
    def _vectors(sample: dict[str, Any]) -> tuple[Any, Any]:
        import numpy as np

        try:
            observation = sample["observation"]
            observed_joints = {
                joint["name"]: joint["position_radians"]
                for joint in observation["joints"]
            }
            observation_values = [
                *[observed_joints[name] for name in JOINT_NAMES],
                observation["gripper"]["position"],
            ]
            staged_action = sample["action"]
            action_values = [
                *[
                    staged_action["joint_positions_radians"][name]
                    for name in JOINT_NAMES
                ],
                staged_action["gripper_position"],
            ]
        except (KeyError, TypeError) as error:
            raise DatasetConversionError("An episode sample has an invalid YAM state/action.") from error
        observation_array = np.asarray(observation_values, dtype=np.float32)
        action_array = np.asarray(action_values, dtype=np.float32)
        if not np.isfinite(observation_array).all() or not np.isfinite(action_array).all():
            raise DatasetConversionError("Episode state and action values must be finite.")
        return observation_array, action_array

    def _decode_frames(self, path: Path) -> Iterator[Any]:
        if self._frame_decoder_override is not None:
            yield from self._frame_decoder_override(path)
            return
        try:
            import av

            with av.open(str(path)) as container:
                for frame in container.decode(video=0):
                    yield frame.to_ndarray(format="rgb24")
        except Exception as error:
            raise DatasetConversionError("A staged episode video could not be decoded.") from error

    @staticmethod
    def _frame_shape(frame: Any) -> tuple[int, int, int]:
        shape = getattr(frame, "shape", None)
        if shape is None or len(shape) != 3:
            raise DatasetConversionError("A staged video frame has an invalid shape.")
        return int(shape[0]), int(shape[1]), int(shape[2])

    def _dataset_class(self) -> Any:
        if self._dataset_class_override is not None:
            return self._dataset_class_override
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        return LeRobotDataset

    def _hub_api(self, token: str) -> Any:
        if self._hub_api_factory_override is not None:
            return self._hub_api_factory_override(token)
        from huggingface_hub import HfApi

        return HfApi(token=token)

    def _card_factory(self) -> Callable[..., Any]:
        if self._card_factory_override is not None:
            return self._card_factory_override
        from lerobot.datasets.lerobot_dataset import create_lerobot_dataset_card

        return create_lerobot_dataset_card

    def _verify_dataset(
        self,
        repo_id: str,
        root: Path,
        expected_episodes: int,
        expected_frames: int,
    ) -> None:
        if self._dataset_verifier_override is not None:
            self._dataset_verifier_override(
                repo_id, root, expected_episodes, expected_frames
            )
            return
        dataset = self._dataset_class()(repo_id=repo_id, root=root)
        if dataset.num_episodes != expected_episodes or dataset.num_frames != expected_frames:
            raise DatasetConversionError("The finalized LeRobot dataset failed count verification.")
        if not (root / "meta" / "info.json").is_file():
            raise DatasetConversionError("The finalized LeRobot dataset has no metadata file.")
        if not any((root / "data").rglob("*.parquet")) or not any(
            (root / "videos").rglob("*.mp4")
        ):
            raise DatasetConversionError("The finalized LeRobot dataset is incomplete.")

    @staticmethod
    def _validate_namespace(api: Any, namespace: str) -> None:
        try:
            identity = api.whoami()
        except Exception as error:
            raise HubAuthenticationError("HF_TOKEN could not be authenticated.") from error
        available = {str(identity.get("name", "")).casefold()}
        for organization in identity.get("orgs", []):
            name = organization.get("name") if isinstance(organization, dict) else organization
            if name:
                available.add(str(name).casefold())
        if namespace.casefold() not in available:
            raise HubAuthenticationError(
                "HF_NAMESPACE is not the authenticated user or one of its organizations."
            )

    @staticmethod
    def _write_dataset_marker(root: Path, recording_id: str) -> None:
        marker = {
            "schema": DATASET_MARKER_SCHEMA,
            "version": DATASET_MARKER_VERSION,
            "recording_id": recording_id,
        }
        (root / DATASET_MARKER_FILENAME).write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _verify_remote_marker(
        api: Any,
        *,
        repo_id: str,
        recording_id: str,
        token: str,
        revision: str | None = None,
        ownership_check: bool = False,
        bootstrap_marker_path: Path | None = None,
    ) -> None:
        try:
            marker_path = api.hf_hub_download(
                repo_id=repo_id,
                filename=DATASET_MARKER_FILENAME,
                repo_type="dataset",
                revision=revision,
                token=token,
            )
        except Exception as error:
            status_code = getattr(
                getattr(error, "response", None), "status_code", None
            )
            marker_missing = isinstance(error, FileNotFoundError) or status_code == 404
            if ownership_check and marker_missing and bootstrap_marker_path is not None:
                try:
                    remote_files = set(
                        api.list_repo_files(
                            repo_id=repo_id,
                            repo_type="dataset",
                            token=token,
                        )
                    )
                except Exception as list_error:
                    raise HubUploadError(
                        "The existing dataset repository could not be inspected safely."
                    ) from list_error
                if remote_files.issubset({".gitattributes"}):
                    HFDatasetUploader._upload_remote_marker(
                        api,
                        repo_id=repo_id,
                        marker_path=bootstrap_marker_path,
                        token=token,
                    )
                    HFDatasetUploader._verify_remote_marker(
                        api,
                        repo_id=repo_id,
                        recording_id=recording_id,
                        token=token,
                    )
                    return
                raise UploadConflictError(
                    "The existing nonempty dataset repository has no ctrl-pi ownership marker."
                ) from error
            if ownership_check and marker_missing:
                raise UploadConflictError(
                    "The existing dataset repository has no ctrl-pi ownership marker."
                ) from error
            raise HubUploadError(
                "The dataset ownership marker could not be downloaded for verification."
            ) from error
        try:
            marker = json.loads(Path(marker_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            if ownership_check:
                raise UploadConflictError(
                    "The existing dataset repository has an invalid ctrl-pi ownership marker."
                ) from error
            raise HubUploadError(
                "The uploaded dataset ownership marker is invalid."
            ) from error
        marker_matches = (
            isinstance(marker, dict)
            and set(marker) == {"schema", "version", "recording_id"}
            and marker.get("schema") == DATASET_MARKER_SCHEMA
            and type(marker.get("version")) is int
            and marker.get("version") == DATASET_MARKER_VERSION
            and marker.get("recording_id") == recording_id
        )
        if not marker_matches:
            if ownership_check:
                raise UploadConflictError(
                    "The existing dataset repository belongs to a different recording."
                )
            raise HubUploadError(
                "The uploaded dataset ownership marker does not match this recording."
            )

    @staticmethod
    def _upload_remote_marker(
        api: Any,
        *,
        repo_id: str,
        marker_path: Path,
        token: str,
    ) -> None:
        api.upload_file(
            path_or_fileobj=marker_path,
            path_in_repo=DATASET_MARKER_FILENAME,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message="Initialize ctrl-pi dataset ownership",
        )

    def _remove_generated_dataset(self, root: Path) -> None:
        dataset_staging = self.dataset_staging_dir.resolve()
        resolved_root = root.resolve()
        if (
            not resolved_root.exists()
            or resolved_root == dataset_staging
            or not resolved_root.is_relative_to(dataset_staging)
        ):
            return
        with contextlib.suppress(OSError):
            shutil.rmtree(resolved_root)
        if resolved_root.parent != dataset_staging:
            with contextlib.suppress(OSError):
                resolved_root.parent.rmdir()
