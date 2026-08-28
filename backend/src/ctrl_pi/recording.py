from __future__ import annotations

import asyncio
import json
import math
import os
import re
import shutil
import subprocess
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from ctrl_pi.camera import CameraFrame, MockCamera
from ctrl_pi.drivers.yam import ArmAction, ArmTelemetry, YAMDriver
from ctrl_pi.rig import RigLease, RigLeaseConflictError, RigLeaseToken


class RecordingConflictError(RuntimeError):
    pass


class RecordingRuntimeError(RuntimeError):
    pass


EPISODE_MANIFEST_FILENAME = "episode.json"
EPISODE_MANIFEST_SCHEMA = "ctrl-pi.recording-episode"
EPISODE_MANIFEST_VERSION = 1
MAX_EPISODE_MANIFEST_BYTES = 64 * 1024
_EPISODE_DIRECTORY = re.compile(r"episode_(\d{6,})\Z")
_RECORDING_COMPONENT = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")


@dataclass(frozen=True)
class RecordingStateSnapshot:
    recording_id: str
    teleop_active: bool
    episode_active: bool
    current_episode_index: int | None
    episode_duration_seconds: float
    episode_count: int
    status: str


@dataclass(frozen=True)
class EpisodeResult:
    index: int
    duration_seconds: float
    sample_count: int
    success: bool
    notes: str | None
    metadata: dict[str, Any]
    artifact_key: str
    started_at: datetime
    ended_at: datetime
    fps: int


def episode_result_metadata(result: EpisodeResult) -> dict[str, Any]:
    """Return the canonical bounded database representation of an episode."""

    return {
        "index": result.index,
        "duration_seconds": result.duration_seconds,
        "sample_count": result.sample_count,
        "success": result.success,
        "notes": result.notes,
        "metadata": result.metadata,
        "artifact_key": result.artifact_key,
        "started_at": result.started_at.isoformat(),
        "ended_at": result.ended_at.isoformat(),
        "fps": result.fps,
    }


def merge_episode_results(
    current: dict[str, Any], results: list[EpisodeResult]
) -> tuple[dict[str, Any], int, float, bool]:
    """Idempotently merge durable manifests into recording aggregates."""

    stored = current.get("episodes", [])
    raw_episodes = list(stored) if isinstance(stored, list) else []
    episodes: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    seen_artifacts: set[str] = set()
    for raw in raw_episodes:
        if not isinstance(raw, dict):
            continue
        try:
            index = raw["index"]
            artifact_key = raw["artifact_key"]
            duration = float(raw["duration_seconds"])
        except (KeyError, TypeError, ValueError):
            continue
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or not isinstance(artifact_key, str)
            or not artifact_key
            or not math.isfinite(duration)
            or not 0 <= duration <= 1_000_000_000
            or index in seen_indices
            or artifact_key in seen_artifacts
        ):
            continue
        episodes.append(dict(raw))
        seen_indices.add(index)
        seen_artifacts.add(artifact_key)

    for result in sorted(results, key=lambda item: item.index):
        canonical = episode_result_metadata(result)
        episodes = [
            item
            for item in episodes
            if item.get("index") != result.index
            and item.get("artifact_key") != result.artifact_key
        ]
        episodes.append(canonical)

    episodes.sort(key=lambda item: int(item["index"]))
    duration_seconds = sum(float(item["duration_seconds"]) for item in episodes)
    metadata = {**current, "episodes": episodes}
    return metadata, len(episodes), duration_seconds, metadata != current


def load_episode_manifest(
    directory: Path, *, expected_recording_id: str | None = None
) -> EpisodeResult:
    """Load one finalized manifest without including local paths in errors."""

    manifest_path = directory / EPISODE_MANIFEST_FILENAME
    try:
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError
        with manifest_path.open("rb") as stream:
            encoded = stream.read(MAX_EPISODE_MANIFEST_BYTES + 1)
        if not encoded or len(encoded) > MAX_EPISODE_MANIFEST_BYTES:
            raise ValueError
        payload = json.loads(encoded)
        if not isinstance(payload, dict):
            raise ValueError
        if (
            payload.get("schema") != EPISODE_MANIFEST_SCHEMA
            or payload.get("version") != EPISODE_MANIFEST_VERSION
            or set(payload) != {"schema", "version", "recording_id", "episode"}
        ):
            raise ValueError
        recording_id = payload["recording_id"]
        if (
            not isinstance(recording_id, str)
            or _RECORDING_COMPONENT.fullmatch(recording_id) is None
            or recording_id != directory.parent.name
            or (
                expected_recording_id is not None
                and recording_id != expected_recording_id
            )
        ):
            raise ValueError
        result = _episode_result_from_metadata(
            payload["episode"], recording_id=recording_id, directory=directory
        )
        _validate_finalized_files(directory)
        return result
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError):
        raise RecordingRuntimeError("episode manifest is invalid") from None


def _episode_result_from_metadata(
    raw: Any, *, recording_id: str, directory: Path
) -> EpisodeResult:
    if not isinstance(raw, dict):
        raise ValueError
    required = {
        "index",
        "duration_seconds",
        "sample_count",
        "success",
        "notes",
        "metadata",
        "artifact_key",
        "started_at",
        "ended_at",
        "fps",
    }
    if set(raw) != required:
        raise ValueError
    index = raw["index"]
    sample_count = raw["sample_count"]
    fps = raw["fps"]
    success = raw["success"]
    notes = raw["notes"]
    metadata = raw["metadata"]
    artifact_key = raw["artifact_key"]
    duration = float(raw["duration_seconds"])
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index <= 999_999_999
        or isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or not 1 <= sample_count <= 1_000_000_000
        or isinstance(fps, bool)
        or not isinstance(fps, int)
        or not 1 <= fps <= 60
        or not isinstance(success, bool)
        or (notes is not None and (not isinstance(notes, str) or len(notes) > 2000))
        or not isinstance(metadata, dict)
        or not isinstance(artifact_key, str)
        or not math.isfinite(duration)
        or duration <= 0
        or not math.isclose(
            duration,
            sample_count / fps,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    ):
        raise ValueError
    match = _EPISODE_DIRECTORY.fullmatch(directory.name)
    expected_artifact = f"{recording_id}/{directory.name}"
    if match is None or int(match.group(1)) != index or artifact_key != expected_artifact:
        raise ValueError
    started_at = datetime.fromisoformat(raw["started_at"])
    ended_at = datetime.fromisoformat(raw["ended_at"])
    if (
        started_at.tzinfo is None
        or ended_at.tzinfo is None
        or ended_at < started_at
    ):
        raise ValueError
    # This also rejects NaN/Infinity and bounds nested legacy metadata before
    # it can be promoted into a durable manifest.
    encoded_metadata = json.dumps(
        metadata, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if len(encoded_metadata) > MAX_EPISODE_MANIFEST_BYTES // 2:
        raise ValueError
    return EpisodeResult(
        index=index,
        duration_seconds=duration,
        sample_count=sample_count,
        success=success,
        notes=notes,
        metadata=metadata,
        artifact_key=artifact_key,
        started_at=started_at,
        ended_at=ended_at,
        fps=fps,
    )


def _validate_finalized_files(directory: Path) -> None:
    for name in ("video.mp4", "samples.jsonl"):
        path = directory / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise ValueError


class FFmpegVideoWriter:
    """Raw RGB writer that finalizes an MP4 only after ffmpeg exits cleanly."""

    def __init__(self, output_path: Path, width: int, height: int, fps: int) -> None:
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise RecordingRuntimeError("ffmpeg is required to record mock camera video")

        self.output_path = output_path
        self.width = width
        self.height = height
        self._closed = False
        try:
            self._process = subprocess.Popen(
                [
                    executable,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    f"{width}x{height}",
                    "-framerate",
                    str(fps),
                    "-i",
                    "pipe:0",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(output_path),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except OSError:
            raise RecordingRuntimeError("ffmpeg could not start") from None

    def write(self, frame: CameraFrame) -> None:
        if self._closed or self._process.stdin is None:
            raise RecordingRuntimeError("video writer is closed")
        if (frame.width, frame.height) != (self.width, self.height):
            raise RecordingRuntimeError("camera dimensions changed during an episode")
        try:
            self._process.stdin.write(frame.rgb)
        except (BrokenPipeError, OSError):
            raise RecordingRuntimeError("ffmpeg stopped while recording video") from None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close_error: OSError | None = None
        if self._process.stdin is not None:
            try:
                self._process.stdin.close()
            except OSError as error:
                close_error = error
            finally:
                self._process.stdin = None
        try:
            return_code = self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            with suppress(OSError):
                self._process.kill()
            with suppress(OSError, subprocess.TimeoutExpired):
                self._process.wait(timeout=5)
            raise RecordingRuntimeError("ffmpeg did not stop cleanly") from None
        if self._process.stderr is not None:
            # Drain diagnostics so the child can be reaped, but never surface
            # vendor-controlled ffmpeg text or local staging paths.
            self._process.stderr.read()
        if self._process.stderr is not None:
            self._process.stderr.close()
        if return_code != 0:
            raise RecordingRuntimeError("ffmpeg failed to finalize video")
        if close_error is not None:
            raise RecordingRuntimeError("ffmpeg pipe closed while finalizing video") from None
        try:
            produced_video = (
                self.output_path.is_file() and self.output_path.stat().st_size > 0
            )
        except OSError:
            produced_video = False
        if not produced_video:
            raise RecordingRuntimeError("ffmpeg produced no video")

    def terminate(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process.stdin is not None:
            try:
                self._process.stdin.close()
            except OSError:
                pass
            finally:
                self._process.stdin = None
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5)
        if self._process.stderr is not None:
            self._process.stderr.close()


@dataclass
class _ActiveEpisode:
    index: int
    directory: Path
    fps: int
    metadata: dict[str, Any]
    started_at: datetime
    started_monotonic: float
    writer: FFmpegVideoWriter
    samples_file: TextIO
    partial_video_path: Path
    partial_samples_path: Path
    sample_count: int = 0
    next_sample_monotonic: float = 0.0


@dataclass
class _RecordingRuntime:
    recording_id: str
    leader_robot_id: str
    follower_robot_id: str
    episode_count: int
    status: str
    teleop_task: asyncio.Task[None] | None = None
    episode: _ActiveEpisode | None = None
    finalizing_episode: _ActiveEpisode | None = None
    finalize_task: asyncio.Task[EpisodeResult] | None = None
    latest_leader: ArmTelemetry | None = None
    latest_follower: ArmTelemetry | None = None
    latest_action: ArmAction | None = None
    last_error: str | None = None
    rig_token: RigLeaseToken | None = None
    inference_active: bool = False
    inference_owner_id: str | None = None


class RecordingManager:
    """Owns the single process-local teleop rig and active episode resources."""

    def __init__(
        self,
        driver: YAMDriver,
        camera: MockCamera,
        staging_dir: Path,
        teleop_frequency_hz: float = 50.0,
        rig_lease: RigLease | None = None,
    ) -> None:
        self.driver = driver
        self.camera = camera
        self.staging_dir = staging_dir
        self.teleop_frequency_hz = teleop_frequency_hz
        self.rig_lease = rig_lease or RigLease()
        self._runtimes: dict[str, _RecordingRuntime] = {}
        self._filesystem_busy: set[str] = set()
        self._lock = asyncio.Lock()
        self._shutting_down = False

    async def ensure_session(
        self,
        recording_id: str,
        leader_robot_id: str,
        follower_robot_id: str,
        episode_count: int,
        status: str,
    ) -> _RecordingRuntime:
        self._recording_directory(recording_id)
        stable_status = (
            ("ready" if episode_count else "draft")
            if status in {"teleop", "recording"}
            else status
        )
        async with self._lock:
            runtime = self._runtimes.get(recording_id)
            if runtime is None:
                runtime = _RecordingRuntime(
                    recording_id=recording_id,
                    leader_robot_id=leader_robot_id,
                    follower_robot_id=follower_robot_id,
                    episode_count=episode_count,
                    status=stable_status,
                )
                self._runtimes[recording_id] = runtime
            elif (
                runtime.episode is None
                and runtime.finalizing_episode is None
                and not self._task_active(runtime.teleop_task)
                and not runtime.inference_active
            ):
                runtime.leader_robot_id = leader_robot_id
                runtime.follower_robot_id = follower_robot_id
                runtime.episode_count = episode_count
                if runtime.last_error is None:
                    runtime.status = stable_status
            return runtime

    async def startup(self) -> None:
        async with self._lock:
            self._shutting_down = False
        await self._run_blocking(self._cleanup_staging_after_restart)

    async def reconcile_recording_artifacts(
        self, recording_id: str, persisted_episodes: list[Any]
    ) -> list[EpisodeResult]:
        """Recover valid manifests and backfill complete pre-manifest episodes."""

        recording_directory = self._recording_directory(recording_id)
        async with self._lock:
            if recording_id in self._filesystem_busy:
                raise RecordingConflictError("recording artifacts are being reconciled")
            runtime = self._runtimes.get(recording_id)
            excluded = {
                episode.directory
                for episode in (
                    None if runtime is None else runtime.episode,
                    None if runtime is None else runtime.finalizing_episode,
                )
                if episode is not None
            }
            self._filesystem_busy.add(recording_id)
        try:
            return await self._run_blocking(
                self._reconcile_recording_directory,
                recording_id,
                recording_directory,
                persisted_episodes,
                excluded,
            )
        finally:
            async with self._lock:
                self._filesystem_busy.discard(recording_id)

    async def reconcile_persisted_episodes(
        self, recording_id: str, episode_count: int, status: str
    ) -> RecordingStateSnapshot | None:
        """Refresh a process-local sentinel after manifest recovery is durable."""

        async with self._lock:
            runtime = self._runtimes.get(recording_id)
            if runtime is None:
                return None
            completed_sentinel = (
                runtime.finalizing_episode is not None
                and runtime.finalize_task is None
                and episode_count > runtime.episode_count
            )
            runtime.episode_count = episode_count
            if completed_sentinel:
                runtime.finalizing_episode = None
            if runtime.last_error == "episode metadata could not be persisted":
                runtime.last_error = None
            active = (
                self._task_active(runtime.teleop_task)
                or runtime.episode is not None
                or runtime.inference_active
            )
            if not active and runtime.last_error is None:
                runtime.status = status
            return self._snapshot(runtime)

    async def recording_activity_active(self, recording_id: str) -> bool:
        async with self._lock:
            runtime = self._runtimes.get(recording_id)
            return runtime is not None and (
                self._task_active(runtime.teleop_task)
                or runtime.episode is not None
                or runtime.finalizing_episode is not None
                or runtime.inference_active
            )

    async def remove_uploaded_staging(self, recording_id: str) -> bool:
        """Remove raw artifacts only after the caller durably commits upload."""

        directory = self._recording_directory(recording_id)
        async with self._lock:
            if recording_id in self._filesystem_busy:
                return False
            runtime = self._runtimes.get(recording_id)
            if runtime is not None and (
                self._task_active(runtime.teleop_task)
                or runtime.episode is not None
                or runtime.finalizing_episode is not None
                or runtime.inference_active
            ):
                return False
            self._filesystem_busy.add(recording_id)
        try:
            return await self._run_blocking(self._remove_tree, directory)
        finally:
            async with self._lock:
                self._filesystem_busy.discard(recording_id)

    async def prune_unowned_staging(self, recording_ids: set[str]) -> None:
        """Delete staging roots that cannot belong to a durable DB row."""

        async with self._lock:
            active_ids = {
                recording_id
                for recording_id, runtime in self._runtimes.items()
                if self._task_active(runtime.teleop_task)
                or runtime.episode is not None
                or runtime.finalizing_episode is not None
                or runtime.inference_active
            }
            protected = recording_ids | active_ids | self._filesystem_busy
        await self._run_blocking(self._prune_unowned_staging, protected)

    async def state(
        self,
        recording_id: str,
        leader_robot_id: str,
        follower_robot_id: str,
        episode_count: int,
        status: str,
    ) -> RecordingStateSnapshot:
        runtime = await self.ensure_session(
            recording_id,
            leader_robot_id,
            follower_robot_id,
            episode_count,
            status,
        )
        async with self._lock:
            return self._snapshot(runtime)

    async def start_teleop(
        self,
        recording_id: str,
        leader_robot_id: str,
        follower_robot_id: str,
        episode_count: int,
        status: str,
    ) -> RecordingStateSnapshot:
        runtime = await self.ensure_session(
            recording_id,
            leader_robot_id,
            follower_robot_id,
            episode_count,
            status,
        )
        async with self._lock:
            if self._shutting_down:
                raise RecordingConflictError("recording service is shutting down")
            if runtime.finalizing_episode is not None:
                raise RecordingConflictError(
                    "the previous episode is still finalizing"
                )
            if self._task_active(runtime.teleop_task):
                raise RecordingConflictError("teleoperation is already active")
            if runtime.inference_active:
                raise RecordingConflictError("inference recording is already active")
            if any(
                other.recording_id != recording_id and self._task_active(other.teleop_task)
                for other in self._runtimes.values()
            ):
                raise RecordingConflictError("another recording session owns the mock rig")

            leader = self.driver.get_arm(leader_robot_id)
            follower = self.driver.get_arm(follower_robot_id)
            if leader.role != "leader" or follower.role != "follower":
                raise RecordingConflictError("teleop requires a leader arm and a follower arm")
            if leader.id == follower.id:
                raise RecordingConflictError("leader and follower must be different arms")

            try:
                rig_token = self.rig_lease.acquire("teleop", recording_id)
            except RigLeaseConflictError as error:
                raise RecordingConflictError(str(error)) from None
            runtime.rig_token = rig_token
            try:
                runtime.status = "teleop"
                runtime.last_error = None
                self._teleop_step(runtime)
                runtime.teleop_task = asyncio.create_task(
                    self._teleop_loop(runtime),
                    name=f"ctrl-pi-teleop-{recording_id}",
                )
            except Exception:
                runtime.rig_token = None
                runtime.status = "ready" if runtime.episode_count else "draft"
                self.rig_lease.release(rig_token)
                raise
            return self._snapshot(runtime)

    async def stop_teleop(self, recording_id: str) -> RecordingStateSnapshot:
        async with self._lock:
            runtime = self._require_runtime(recording_id)
            if runtime.episode is not None or runtime.finalizing_episode is not None:
                raise RecordingConflictError("stop the active episode before stopping teleop")
            if not self._task_active(runtime.teleop_task):
                raise RecordingConflictError("teleoperation is not active")
            task = runtime.teleop_task
            runtime.teleop_task = None
            rig_token = runtime.rig_token
            runtime.rig_token = None
            runtime.status = "ready" if runtime.episode_count else "draft"

        try:
            await self._cancel_task(task)
        finally:
            if rig_token is not None:
                self.rig_lease.release(rig_token)
        async with self._lock:
            return self._snapshot(runtime)

    async def start_episode(
        self,
        recording_id: str,
        fps: int,
        metadata: dict[str, Any],
    ) -> RecordingStateSnapshot:
        if not 1 <= fps <= 60:
            raise RecordingConflictError("recording fps must be between 1 and 60")

        async with self._lock:
            runtime = self._require_runtime(recording_id)
            if not self._task_active(runtime.teleop_task):
                raise RecordingConflictError("start teleop before recording an episode")
            if runtime.episode is not None:
                raise RecordingConflictError("an episode is already recording")
            if runtime.finalizing_episode is not None:
                raise RecordingConflictError("the previous episode is still finalizing")
            if recording_id in self._filesystem_busy:
                raise RecordingConflictError("recording artifacts are being reconciled")

            episode = self._open_episode(runtime, fps, metadata)
            try:
                self._teleop_step(runtime)
                self._capture_sample(runtime, episode)
            except Exception:
                self._abort_episode(episode)
                self._remove_failed_episode_files(episode)
                raise
            runtime.episode = episode
            runtime.status = "recording"
            return self._snapshot(runtime)

    async def start_inference_episode(
        self,
        recording_id: str,
        follower_robot_id: str,
        episode_count: int,
        status: str,
        fps: int,
        metadata: dict[str, Any],
        *,
        inference_owner_id: str,
    ) -> RecordingStateSnapshot:
        """Start passive capture while a RobotInferenceLoop owns the rig.

        This path never acquires or drives the rig. It verifies the existing
        inference lease and records only actions already applied by the loop.
        """

        if not 1 <= fps <= 60:
            raise RecordingConflictError("recording fps must be between 1 and 60")
        runtime = await self.ensure_session(
            recording_id,
            follower_robot_id,
            follower_robot_id,
            episode_count,
            status,
        )
        async with self._lock:
            if self._shutting_down:
                raise RecordingConflictError("recording service is shutting down")
            if self._task_active(runtime.teleop_task):
                raise RecordingConflictError("teleoperation is already active")
            if runtime.inference_active or runtime.episode is not None:
                raise RecordingConflictError("an episode is already recording")
            if runtime.finalizing_episode is not None:
                raise RecordingConflictError("the previous episode is still finalizing")
            if recording_id in self._filesystem_busy:
                raise RecordingConflictError("recording artifacts are being reconciled")
            lease = self.rig_lease.current()
            if (
                lease is None
                or lease.owner != "inference"
                or lease.owner_id != inference_owner_id
            ):
                raise RecordingConflictError(
                    "the inference session does not own the arm rig"
                )
            follower = self.driver.get_arm(follower_robot_id)
            if not follower.connected or follower.role != "follower":
                raise RecordingConflictError(
                    "inference recording requires a connected follower arm"
                )

            episode = self._open_episode(runtime, fps, metadata)
            try:
                no_op = ArmAction.from_telemetry(follower)
                runtime.latest_leader = follower
                runtime.latest_follower = follower
                runtime.latest_action = no_op
            except Exception:
                self._abort_episode(episode)
                self._remove_failed_episode_files(episode)
                raise
            runtime.episode = episode
            runtime.inference_active = True
            runtime.inference_owner_id = inference_owner_id
            runtime.status = "recording"
            runtime.last_error = None
            return self._snapshot(runtime)

    async def capture_inference_action(
        self,
        recording_id: str,
        observation: ArmTelemetry,
        action: ArmAction,
        applied: ArmTelemetry,
    ) -> None:
        """Record one applied policy action when the configured FPS is due."""

        async with self._lock:
            runtime = self._require_runtime(recording_id)
            episode = runtime.episode
            if not runtime.inference_active or episode is None:
                raise RecordingRuntimeError("inference recording is not active")
            lease = self.rig_lease.current()
            if (
                lease is None
                or lease.owner != "inference"
                or lease.owner_id != runtime.inference_owner_id
            ):
                raise RecordingRuntimeError("inference no longer owns the arm rig")
            if observation.id != runtime.follower_robot_id or applied.id != observation.id:
                raise RecordingRuntimeError("inference arm identity changed while recording")
            runtime.latest_leader = observation
            runtime.latest_action = action
            runtime.latest_follower = applied
            if time.monotonic() >= episode.next_sample_monotonic:
                self._capture_sample(runtime, episode)

    async def stop_inference_episode(
        self,
        recording_id: str,
        success: bool,
        notes: str | None,
    ) -> tuple[RecordingStateSnapshot, EpisodeResult]:
        async with self._lock:
            runtime = self._require_runtime(recording_id)
            episode = runtime.episode
            if episode is None or not runtime.inference_active:
                if runtime.finalizing_episode is not None:
                    raise RecordingConflictError("the episode is already finalizing")
                raise RecordingConflictError("no inference episode is recording")
            runtime.episode = None
            runtime.inference_active = False
            runtime.inference_owner_id = None
            runtime.finalizing_episode = episode
            # The public recording lifecycle has no transient `finalizing`
            # literal; the inference-session snapshot reports that finer
            # process-local phase while Recording remains `recording`.
            runtime.status = "recording"
            finalize_task = asyncio.create_task(
                asyncio.to_thread(self._finalize_episode, episode, success, notes),
                name=f"ctrl-pi-inference-finalize-{recording_id}-{episode.index}",
            )
            runtime.finalize_task = finalize_task

        try:
            result = await self._settle_task(finalize_task)
        except Exception:
            await self._run_blocking(self._abort_episode, episode)
            self._remove_failed_episode_files(episode)
            async with self._lock:
                runtime.status = "failed"
                runtime.last_error = "the episode could not be finalized safely"
                if runtime.finalize_task is finalize_task:
                    runtime.finalize_task = None
                    runtime.finalizing_episode = None
            raise

        async with self._lock:
            if runtime.finalize_task is finalize_task:
                runtime.finalize_task = None
            runtime.status = "ready"
            runtime.last_error = None
            snapshot = self._snapshot(runtime)
            return (
                RecordingStateSnapshot(
                    **{
                        **snapshot.__dict__,
                        "episode_active": False,
                        "current_episode_index": None,
                        "episode_duration_seconds": 0.0,
                        "episode_count": runtime.episode_count + 1,
                    }
                ),
                result,
            )

    async def abort_inference_episode(self, recording_id: str) -> None:
        async with self._lock:
            runtime = self._require_runtime(recording_id)
            episode = runtime.episode
            runtime.episode = None
            runtime.inference_active = False
            runtime.inference_owner_id = None
            runtime.status = "failed"
            runtime.last_error = "the inference recording stopped before finalization"
        if episode is not None:
            await self._run_blocking(self._abort_episode, episode)
            self._remove_failed_episode_files(episode)

    async def stop_episode(
        self,
        recording_id: str,
        success: bool,
        notes: str | None,
    ) -> tuple[RecordingStateSnapshot, EpisodeResult]:
        async with self._lock:
            runtime = self._require_runtime(recording_id)
            episode = runtime.episode
            if episode is None:
                if runtime.finalizing_episode is not None:
                    raise RecordingConflictError("the episode is already finalizing")
                raise RecordingConflictError("no episode is recording")
            runtime.episode = None
            runtime.finalizing_episode = episode
            finalize_task = asyncio.create_task(
                asyncio.to_thread(self._finalize_episode, episode, success, notes),
                name=f"ctrl-pi-finalize-{recording_id}-{episode.index}",
            )
            runtime.finalize_task = finalize_task

        try:
            # Once ffmpeg finalization starts it cannot safely be cancelled: the
            # worker thread would continue closing and renaming the same files.
            # Settle it before any caller or shutdown path changes ownership.
            result = await self._settle_task(finalize_task)
        except Exception:
            await self._run_blocking(self._abort_episode, episode)
            self._remove_failed_episode_files(episode)
            async with self._lock:
                runtime.status = "failed"
                runtime.last_error = "the episode could not be finalized safely"
                if runtime.finalize_task is finalize_task:
                    runtime.finalize_task = None
                    runtime.finalizing_episode = None
            raise

        async with self._lock:
            if runtime.finalize_task is finalize_task:
                runtime.finalize_task = None
            teleop_still_owns_rig = (
                self._task_active(runtime.teleop_task)
                and runtime.rig_token is not None
                and self.rig_lease.current() == runtime.rig_token
                and runtime.last_error is None
                and not self._shutting_down
            )
            if teleop_still_owns_rig:
                runtime.status = "teleop"
            else:
                runtime.status = "failed"
                if runtime.last_error is None:
                    runtime.last_error = "teleoperation stopped while finalizing the episode"
            snapshot = self._snapshot(runtime)
            snapshot = RecordingStateSnapshot(
                **{
                    **snapshot.__dict__,
                    "episode_active": False,
                    "current_episode_index": None,
                    "episode_duration_seconds": 0.0,
                    "episode_count": runtime.episode_count + 1,
                }
            )
            return snapshot, result

    async def confirm_episode_persisted(
        self, recording_id: str, episode_count: int, status: str
    ) -> RecordingStateSnapshot:
        async with self._lock:
            runtime = self._require_runtime(recording_id)
            runtime.episode_count = episode_count
            teleop_still_owns_rig = (
                self._task_active(runtime.teleop_task)
                and runtime.rig_token is not None
                and self.rig_lease.current() == runtime.rig_token
                and runtime.last_error is None
                and not self._shutting_down
            )
            if status == "teleop" and not teleop_still_owns_rig:
                runtime.status = "failed"
                if runtime.last_error is None:
                    runtime.last_error = (
                        "teleoperation stopped while persisting episode metadata"
                    )
            elif runtime.status != "failed":
                runtime.status = status
            runtime.finalizing_episode = None
            return self._snapshot(runtime)

    async def episode_persistence_failed(self, recording_id: str) -> None:
        """Stop process-local side effects while retaining a valid manifest."""

        await self.compensate_persistence_failure(
            recording_id, preserve_finalized=True
        )

    async def compensate_persistence_failure(
        self, recording_id: str, *, preserve_finalized: bool = False
    ) -> None:
        async with self._lock:
            runtime = self._require_runtime(recording_id)
            task = runtime.teleop_task
            runtime.teleop_task = None
            episode = runtime.episode
            runtime.episode = None
            finalizing_episode = runtime.finalizing_episode
            runtime.finalizing_episode = None
            finalize_task = runtime.finalize_task
            runtime.finalize_task = None
            rig_token = runtime.rig_token
            runtime.rig_token = None
            runtime.inference_active = False
            runtime.inference_owner_id = None
            runtime.status = "failed"
            runtime.last_error = "episode metadata could not be persisted"

        try:
            await self._cancel_task(task)
        finally:
            if rig_token is not None:
                self.rig_lease.release(rig_token)
        if episode is not None:
            await self._run_blocking(self._abort_episode, episode)
            self._remove_failed_episode_files(episode)
        if finalize_task is not None and not finalize_task.done():
            try:
                await self._settle_task(finalize_task)
            except Exception:
                if finalizing_episode is not None:
                    await self._run_blocking(self._abort_episode, finalizing_episode)
        if finalizing_episode is not None and not preserve_finalized:
            self._remove_failed_episode_files(finalizing_episode)

    async def shutdown(self) -> None:
        async with self._lock:
            self._shutting_down = True
            tasks = [
                runtime.teleop_task
                for runtime in self._runtimes.values()
                if self._task_active(runtime.teleop_task)
            ]
            episodes = [
                runtime.episode
                for runtime in self._runtimes.values()
                if runtime.episode is not None
            ]
            finalizers = [
                (runtime, runtime.finalizing_episode, runtime.finalize_task)
                for runtime in self._runtimes.values()
                if runtime.finalizing_episode is not None
                and runtime.finalize_task is not None
            ]
            for runtime in self._runtimes.values():
                runtime.teleop_task = None
                runtime.episode = None
                runtime.inference_active = False
                runtime.inference_owner_id = None
            rig_tokens = [
                runtime.rig_token
                for runtime in self._runtimes.values()
                if runtime.rig_token is not None
            ]
            for runtime in self._runtimes.values():
                runtime.rig_token = None

        try:
            for task in tasks:
                await self._cancel_task(task)
        finally:
            for rig_token in rig_tokens:
                self.rig_lease.release(rig_token)
        for episode in episodes:
            await self._run_blocking(self._abort_episode, episode)
            self._remove_failed_episode_files(episode)
        for runtime, episode, finalize_task in finalizers:
            try:
                await self._settle_task(finalize_task)
            except Exception:
                await self._run_blocking(self._abort_episode, episode)
                self._remove_failed_episode_files(episode)
            finally:
                async with self._lock:
                    if runtime.finalize_task is finalize_task:
                        runtime.finalize_task = None
                        runtime.finalizing_episode = None
        async with self._lock:
            # A completed episode can remain as a short persistence sentinel
            # after its finalizer task has settled. It owns no open resources.
            for runtime in self._runtimes.values():
                if runtime.finalize_task is None:
                    runtime.finalizing_episode = None

    async def active_resource_counts(self) -> tuple[int, int]:
        async with self._lock:
            return (
                sum(self._task_active(runtime.teleop_task) for runtime in self._runtimes.values()),
                sum(
                    runtime.episode is not None or runtime.finalizing_episode is not None
                    for runtime in self._runtimes.values()
                ),
            )

    async def _teleop_loop(self, runtime: _RecordingRuntime) -> None:
        interval = 1.0 / self.teleop_frequency_hz
        try:
            while True:
                cycle_started = time.monotonic()
                async with self._lock:
                    self._teleop_step(runtime)
                    episode = runtime.episode
                    if episode is not None and (
                        cycle_started >= episode.next_sample_monotonic
                    ):
                        self._capture_sample(runtime, episode)
                elapsed = time.monotonic() - cycle_started
                await asyncio.sleep(max(0.0, interval - elapsed))
        except asyncio.CancelledError:
            raise
        except Exception:
            rig_token: RigLeaseToken | None = None
            async with self._lock:
                runtime.last_error = "teleoperation stopped safely after a runtime failure"
                runtime.status = "failed"
                episode = runtime.episode
                runtime.episode = None
                runtime.teleop_task = None
                rig_token = runtime.rig_token
                runtime.rig_token = None
            if rig_token is not None:
                self.rig_lease.release(rig_token)
            if episode is not None:
                await self._run_blocking(self._abort_episode, episode)
                await self._run_blocking(self._remove_failed_episode_files, episode)

    def _teleop_step(self, runtime: _RecordingRuntime) -> None:
        leader = self.driver.get_arm(runtime.leader_robot_id)
        action = ArmAction.from_telemetry(leader)
        follower = self.driver.apply_action(runtime.follower_robot_id, action)
        runtime.latest_leader = leader
        runtime.latest_action = action
        runtime.latest_follower = follower

    def _capture_sample(
        self, runtime: _RecordingRuntime, episode: _ActiveEpisode
    ) -> None:
        if (
            runtime.latest_leader is None
            or runtime.latest_follower is None
            or runtime.latest_action is None
        ):
            raise RecordingRuntimeError("teleop state is unavailable")
        frame = self.camera.capture()
        episode.writer.write(frame)
        elapsed = time.monotonic() - episode.started_monotonic
        sample = {
            "frame_index": episode.sample_count,
            "timestamp_seconds": elapsed,
            "camera_timestamp": frame.timestamp.isoformat(),
            "observation": runtime.latest_follower.model_dump(mode="json"),
            "action": runtime.latest_action.model_dump(mode="json"),
            "leader_state": runtime.latest_leader.model_dump(mode="json"),
        }
        episode.samples_file.write(json.dumps(sample, separators=(",", ":")) + "\n")
        episode.samples_file.flush()
        episode.sample_count += 1
        period = 1.0 / episode.fps
        now = time.monotonic()
        episode.next_sample_monotonic += period
        while episode.next_sample_monotonic <= now:
            episode.next_sample_monotonic += period

    def _finalize_episode(
        self, episode: _ActiveEpisode, success: bool, notes: str | None
    ) -> EpisodeResult:
        try:
            episode.samples_file.flush()
            os.fsync(episode.samples_file.fileno())
            episode.samples_file.close()
            episode.writer.close()
            final_video = episode.directory / "video.mp4"
            final_samples = episode.directory / "samples.jsonl"
            self._fsync_file(episode.partial_video_path)
            episode.partial_video_path.replace(final_video)
            episode.partial_samples_path.replace(final_samples)
            self._fsync_directory(episode.directory)
        except (OSError, RecordingRuntimeError):
            raise RecordingRuntimeError("episode artifacts could not be finalized") from None

        ended_at = datetime.now(UTC)
        duration = episode.sample_count / episode.fps
        result = EpisodeResult(
            index=episode.index,
            duration_seconds=duration,
            sample_count=episode.sample_count,
            success=success,
            notes=notes,
            metadata=episode.metadata,
            artifact_key=f"{episode.directory.parent.name}/{episode.directory.name}",
            started_at=episode.started_at,
            ended_at=ended_at,
            fps=episode.fps,
        )
        self._write_episode_manifest(episode.directory, result)
        return result

    @staticmethod
    def _abort_episode(episode: _ActiveEpisode) -> None:
        if not episode.samples_file.closed:
            with suppress(OSError):
                episode.samples_file.flush()
            with suppress(OSError):
                episode.samples_file.close()
        with suppress(Exception):
            episode.writer.terminate()

    def _remove_failed_episode_files(self, episode: _ActiveEpisode) -> None:
        try:
            load_episode_manifest(
                episode.directory,
                expected_recording_id=episode.directory.parent.name,
            )
        except RecordingRuntimeError:
            self._remove_tree(episode.directory)
            return
        self._remove_partial_files(episode.directory)

    def _available_episode_index(self, runtime: _RecordingRuntime) -> int:
        index = runtime.episode_count
        while (self.staging_dir / runtime.recording_id / f"episode_{index:06d}").exists():
            index += 1
        return index

    def _open_episode(
        self,
        runtime: _RecordingRuntime,
        fps: int,
        metadata: dict[str, Any],
    ) -> _ActiveEpisode:
        episode_index = self._available_episode_index(runtime)
        episode_dir = (
            self.staging_dir / runtime.recording_id / f"episode_{episode_index:06d}"
        )
        partial_video = episode_dir / "video.partial.mp4"
        partial_samples = episode_dir / "samples.partial.jsonl"
        samples_file: TextIO | None = None
        try:
            episode_dir.mkdir(parents=True, exist_ok=False)
            # Persist both newly linked directory entries before capture can
            # report success. `parents=True` may have created the recording
            # root as well as the episode directory.
            self._fsync_directory(self.staging_dir)
            self._fsync_directory(episode_dir.parent)
            samples_file = partial_samples.open("x", encoding="utf-8")
            writer = FFmpegVideoWriter(
                partial_video,
                width=self.camera.width,
                height=self.camera.height,
                fps=fps,
            )
        except Exception:
            if samples_file is not None:
                with suppress(OSError):
                    samples_file.close()
            self._remove_tree(episode_dir)
            raise
        assert samples_file is not None
        now = time.monotonic()
        return _ActiveEpisode(
            index=episode_index,
            directory=episode_dir,
            fps=fps,
            metadata=metadata,
            started_at=datetime.now(UTC),
            started_monotonic=now,
            writer=writer,
            samples_file=samples_file,
            partial_video_path=partial_video,
            partial_samples_path=partial_samples,
            next_sample_monotonic=now,
        )

    def _write_episode_manifest(
        self, directory: Path, result: EpisodeResult
    ) -> None:
        payload = {
            "schema": EPISODE_MANIFEST_SCHEMA,
            "version": EPISODE_MANIFEST_VERSION,
            "recording_id": directory.parent.name,
            "episode": episode_result_metadata(result),
        }
        try:
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError):
            raise RecordingRuntimeError("episode metadata is invalid") from None
        if not encoded or len(encoded) > MAX_EPISODE_MANIFEST_BYTES:
            raise RecordingRuntimeError("episode metadata exceeds the safe manifest limit")
        temporary = directory / f".{EPISODE_MANIFEST_FILENAME}.partial"
        manifest = directory / EPISODE_MANIFEST_FILENAME
        try:
            temporary.unlink(missing_ok=True)
            with temporary.open("xb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(manifest)
            self._fsync_directory(directory)
            self._fsync_directory(directory.parent)
            self._fsync_directory(self.staging_dir)
            # Validate exactly what was made durable, not merely the in-memory
            # object used to create it.
            load_episode_manifest(
                directory, expected_recording_id=directory.parent.name
            )
        except RecordingRuntimeError:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
            raise
        except OSError:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
            raise RecordingRuntimeError("episode manifest could not be persisted") from None

    def _cleanup_staging_after_restart(self) -> None:
        try:
            self.staging_dir.mkdir(parents=True, exist_ok=True)
            entries = list(self.staging_dir.iterdir())
        except OSError:
            raise RecordingRuntimeError("recording staging is unavailable") from None
        for recording_directory in entries:
            if (
                recording_directory.is_symlink()
                or not recording_directory.is_dir()
                or _RECORDING_COMPONENT.fullmatch(recording_directory.name) is None
            ):
                continue
            self._cleanup_recognizable_episode_entries(recording_directory)

    def _cleanup_recognizable_episode_entries(
        self, recording_directory: Path
    ) -> None:
        touched = False
        for episode_directory in self._directory_entries(recording_directory):
            if _EPISODE_DIRECTORY.fullmatch(episode_directory.name) is None:
                continue
            if episode_directory.is_symlink() or not episode_directory.is_dir():
                self._remove_tree(episode_directory)
                touched = True
                continue
            if not self._has_episode_artifact_marker(episode_directory):
                # A shared directory can coincidentally contain an episode-like
                # name. Never delete it unless it contains ctrl-pi artifacts.
                continue
            touched = True
            manifest = episode_directory / EPISODE_MANIFEST_FILENAME
            if manifest.exists() or manifest.is_symlink():
                try:
                    load_episode_manifest(
                        episode_directory,
                        expected_recording_id=recording_directory.name,
                    )
                except RecordingRuntimeError:
                    self._remove_tree(episode_directory)
                else:
                    self._remove_partial_files(episode_directory)
                continue
            if self._complete_legacy_episode(episode_directory):
                self._remove_partial_files(episode_directory)
            else:
                self._remove_tree(episode_directory)
        if touched:
            self._remove_empty_directory(recording_directory)

    def _reconcile_recording_directory(
        self,
        recording_id: str,
        recording_directory: Path,
        persisted_episodes: list[Any],
        excluded: set[Path],
    ) -> list[EpisodeResult]:
        if not recording_directory.exists():
            return []
        if recording_directory.is_symlink() or not recording_directory.is_dir():
            self._remove_tree(recording_directory)
            return []
        persisted_by_artifact = {
            raw.get("artifact_key"): raw
            for raw in persisted_episodes
            if isinstance(raw, dict) and isinstance(raw.get("artifact_key"), str)
        }
        results: list[EpisodeResult] = []
        for episode_directory in self._directory_entries(recording_directory):
            if episode_directory in excluded:
                continue
            if (
                episode_directory.is_symlink()
                or not episode_directory.is_dir()
                or _EPISODE_DIRECTORY.fullmatch(episode_directory.name) is None
            ):
                self._remove_tree(episode_directory)
                continue
            try:
                result = load_episode_manifest(
                    episode_directory, expected_recording_id=recording_id
                )
            except RecordingRuntimeError:
                manifest = episode_directory / EPISODE_MANIFEST_FILENAME
                artifact_key = f"{recording_id}/{episode_directory.name}"
                raw = persisted_by_artifact.get(artifact_key)
                if (
                    manifest.exists()
                    or manifest.is_symlink()
                    or raw is None
                    or not self._complete_legacy_episode(episode_directory)
                ):
                    self._remove_tree(episode_directory)
                    continue
                try:
                    result = _episode_result_from_metadata(
                        raw,
                        recording_id=recording_id,
                        directory=episode_directory,
                    )
                    _validate_finalized_files(episode_directory)
                    self._write_episode_manifest(episode_directory, result)
                except (RecordingRuntimeError, OSError, TypeError, ValueError):
                    self._remove_tree(episode_directory)
                    continue
            self._remove_partial_files(episode_directory)
            results.append(result)
        self._remove_empty_directory(recording_directory)
        return sorted(results, key=lambda item: item.index)

    def _prune_unowned_staging(self, recording_ids: set[str]) -> None:
        try:
            entries = list(self.staging_dir.iterdir())
        except (FileNotFoundError, OSError):
            return
        for entry in entries:
            if (
                entry.name in recording_ids
                or entry.is_symlink()
                or not entry.is_dir()
                or _RECORDING_COMPONENT.fullmatch(entry.name) is None
            ):
                continue
            # A missing database row can mean a fresh/wrong database or a
            # restore mismatch, not that finalized raw data is disposable.
            # Remove only recognizable incomplete/invalid episode remnants;
            # valid manifests and unrelated shared-root content survive.
            self._cleanup_recognizable_episode_entries(entry)

    def _recording_directory(self, recording_id: str) -> Path:
        if (
            not isinstance(recording_id, str)
            or _RECORDING_COMPONENT.fullmatch(recording_id) is None
        ):
            raise RecordingConflictError("recording ID is invalid")
        return self.staging_dir / recording_id

    @staticmethod
    def _directory_entries(directory: Path) -> list[Path]:
        try:
            return list(directory.iterdir())
        except OSError:
            return []

    @staticmethod
    def _complete_legacy_episode(directory: Path) -> bool:
        try:
            _validate_finalized_files(directory)
        except (OSError, ValueError):
            return False
        return not any(
            (directory / name).exists()
            for name in (
                "video.partial.mp4",
                "samples.partial.jsonl",
                f".{EPISODE_MANIFEST_FILENAME}.partial",
            )
        )

    @staticmethod
    def _has_episode_artifact_marker(directory: Path) -> bool:
        return any(
            (directory / name).exists() or (directory / name).is_symlink()
            for name in (
                "video.mp4",
                "samples.jsonl",
                EPISODE_MANIFEST_FILENAME,
                "video.partial.mp4",
                "samples.partial.jsonl",
                f".{EPISODE_MANIFEST_FILENAME}.partial",
            )
        )

    @staticmethod
    def _remove_partial_files(directory: Path) -> None:
        for name in (
            "video.partial.mp4",
            "samples.partial.jsonl",
            f".{EPISODE_MANIFEST_FILENAME}.partial",
        ):
            with suppress(OSError):
                (directory / name).unlink(missing_ok=True)

    @staticmethod
    def _remove_empty_directory(directory: Path) -> None:
        with suppress(OSError):
            directory.rmdir()

    @staticmethod
    def _remove_tree(path: Path) -> bool:
        if not path.exists() and not path.is_symlink():
            return True
        try:
            if path.is_symlink() or not path.is_dir():
                path.unlink(missing_ok=True)
            else:
                shutil.rmtree(path)
        except OSError:
            return False
        return not path.exists() and not path.is_symlink()

    @staticmethod
    def _fsync_file(path: Path) -> None:
        with path.open("rb") as stream:
            os.fsync(stream.fileno())

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _task_active(task: asyncio.Task[None] | None) -> bool:
        return task is not None and not task.done()

    @staticmethod
    async def _cancel_task(task: asyncio.Task[None] | None) -> None:
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @staticmethod
    async def _run_blocking(operation: Any, *args: Any) -> Any:
        task = asyncio.create_task(asyncio.to_thread(operation, *args))
        return await RecordingManager._settle_task(task)

    @staticmethod
    async def _settle_task(task: asyncio.Task[Any]) -> Any:
        """Wait for an owned resource task, even if its caller is cancelled."""

        while True:
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                if task.cancelled():
                    raise
                if task.done():
                    return task.result()
                # The blocking worker still owns ffmpeg/files. Deferring caller
                # cancellation is the only way to avoid a concurrent abort.
                continue

    def _require_runtime(self, recording_id: str) -> _RecordingRuntime:
        try:
            return self._runtimes[recording_id]
        except KeyError as error:
            raise RecordingConflictError("recording session is not initialized") from error

    @staticmethod
    def _snapshot(runtime: _RecordingRuntime) -> RecordingStateSnapshot:
        episode = runtime.episode or runtime.finalizing_episode
        return RecordingStateSnapshot(
            recording_id=runtime.recording_id,
            teleop_active=RecordingManager._task_active(runtime.teleop_task),
            episode_active=episode is not None,
            current_episode_index=None if episode is None else episode.index,
            episode_duration_seconds=(
                0.0
                if episode is None
                else max(0.0, time.monotonic() - episode.started_monotonic)
            ),
            episode_count=runtime.episode_count,
            status=runtime.status,
        )
