from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import time
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

    def write(self, frame: CameraFrame) -> None:
        if self._closed or self._process.stdin is None:
            raise RecordingRuntimeError("video writer is closed")
        if (frame.width, frame.height) != (self.width, self.height):
            raise RecordingRuntimeError("camera dimensions changed during an episode")
        try:
            self._process.stdin.write(frame.rgb)
        except (BrokenPipeError, OSError) as error:
            raise RecordingRuntimeError("ffmpeg stopped while recording video") from error

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
        except subprocess.TimeoutExpired as error:
            self._process.kill()
            self._process.wait(timeout=5)
            raise RecordingRuntimeError("ffmpeg did not stop cleanly") from error
        stderr = b"" if self._process.stderr is None else self._process.stderr.read()
        if self._process.stderr is not None:
            self._process.stderr.close()
        if return_code != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise RecordingRuntimeError(f"ffmpeg failed to finalize video: {detail}")
        if close_error is not None:
            raise RecordingRuntimeError("ffmpeg pipe closed while finalizing video") from close_error
        if not self.output_path.is_file() or self.output_path.stat().st_size == 0:
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
            elif runtime.episode is None and not self._task_active(runtime.teleop_task):
                runtime.leader_robot_id = leader_robot_id
                runtime.follower_robot_id = follower_robot_id
                runtime.episode_count = episode_count
                if runtime.last_error is None:
                    runtime.status = stable_status
            return runtime

    async def startup(self) -> None:
        async with self._lock:
            self._shutting_down = False

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

            episode_index = self._available_episode_index(runtime)
            episode_dir = self.staging_dir / recording_id / f"episode_{episode_index:06d}"
            episode_dir.mkdir(parents=True, exist_ok=False)
            partial_video = episode_dir / "video.partial.mp4"
            partial_samples = episode_dir / "samples.partial.jsonl"
            samples_file = partial_samples.open("x", encoding="utf-8")
            try:
                writer = FFmpegVideoWriter(
                    partial_video,
                    width=self.camera.width,
                    height=self.camera.height,
                    fps=fps,
                )
            except Exception:
                samples_file.close()
                partial_samples.unlink(missing_ok=True)
                try:
                    episode_dir.rmdir()
                except OSError:
                    pass
                raise

            now = time.monotonic()
            episode = _ActiveEpisode(
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
        except Exception as error:
            await asyncio.to_thread(self._abort_episode, episode)
            self._remove_failed_episode_files(episode)
            async with self._lock:
                runtime.status = "failed"
                runtime.last_error = str(error)
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
        async with self._lock:
            runtime = self._require_runtime(recording_id)
            runtime.finalizing_episode = None
            runtime.status = "failed"
            runtime.last_error = "episode metadata could not be persisted"

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
            await asyncio.to_thread(self._abort_episode, episode)
        for runtime, episode, finalize_task in finalizers:
            try:
                await self._settle_task(finalize_task)
            except Exception:
                await asyncio.to_thread(self._abort_episode, episode)
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
        except Exception as error:
            rig_token: RigLeaseToken | None = None
            async with self._lock:
                runtime.last_error = str(error)
                runtime.status = "failed"
                episode = runtime.episode
                runtime.episode = None
                runtime.teleop_task = None
                rig_token = runtime.rig_token
                runtime.rig_token = None
            if rig_token is not None:
                self.rig_lease.release(rig_token)
            if episode is not None:
                self._abort_episode(episode)
                self._remove_failed_episode_files(episode)

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

    @staticmethod
    def _finalize_episode(
        episode: _ActiveEpisode, success: bool, notes: str | None
    ) -> EpisodeResult:
        episode.samples_file.flush()
        episode.samples_file.close()
        episode.writer.close()
        final_video = episode.directory / "video.mp4"
        final_samples = episode.directory / "samples.jsonl"
        episode.partial_video_path.replace(final_video)
        episode.partial_samples_path.replace(final_samples)
        ended_at = datetime.now(UTC)
        duration = episode.sample_count / episode.fps
        return EpisodeResult(
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

    @staticmethod
    def _abort_episode(episode: _ActiveEpisode) -> None:
        if not episode.samples_file.closed:
            episode.samples_file.flush()
            episode.samples_file.close()
        episode.writer.terminate()

    @staticmethod
    def _remove_failed_episode_files(episode: _ActiveEpisode) -> None:
        episode.partial_video_path.unlink(missing_ok=True)
        episode.partial_samples_path.unlink(missing_ok=True)
        try:
            episode.directory.rmdir()
        except OSError:
            pass

    def _available_episode_index(self, runtime: _RecordingRuntime) -> int:
        index = runtime.episode_count
        while (self.staging_dir / runtime.recording_id / f"episode_{index:06d}").exists():
            index += 1
        return index

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
    async def _settle_task(task: asyncio.Task[EpisodeResult]) -> EpisodeResult:
        """Wait for a non-cancellable file finalizer, even if its caller is cancelled."""

        while True:
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
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
