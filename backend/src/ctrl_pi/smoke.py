from __future__ import annotations

import argparse
import asyncio
import contextlib
import functools
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ctrl_pi.camera import MockCamera
from ctrl_pi.compute_stub import StubComputeTarget
from ctrl_pi.config import AppConfig
from ctrl_pi.db import Base
from ctrl_pi.deployments import DeploymentService
from ctrl_pi.drivers.mock_yam import MockYAMDriver
from ctrl_pi.hf import (
    DATASET_MARKER_FILENAME,
    DatasetUploadResult,
    HFDatasetUploader,
    RecordingUploadSource,
)
from ctrl_pi.hf_datasets import HFDatasetBrowser
from ctrl_pi.inference_sessions import (
    InferenceSessionManager,
    InferenceStartOptions,
    InferenceStopOptions,
)
from ctrl_pi.inference_runtime import RuntimeLoadSpec, StubInferenceRuntime
from ctrl_pi.inference_transport import InProcessInferenceTransport
from ctrl_pi.recording import EpisodeResult, RecordingManager
from ctrl_pi.rig import RigLease


_REVISION = re.compile(r"[0-9a-f]{40}")
_FAKE_NAMESPACE = "ctrl-pi-smoke"
_FAKE_TOKEN = "offline-smoke-token"
_POLICY_REPO = "ctrl-pi/stub-policy"
_POLICY_REVISION = "1" * 40
_RECORD_SECONDS = 5.0
_RECORD_FPS = 10
_ACTION_STEPS = 100


class SmokeFailure(RuntimeError):
    """A safe, actionable failure from the deterministic smoke gate."""


@dataclass(frozen=True)
class SmokeEvidence:
    hub_mode: str
    repo_id: str
    revision: str
    private: bool
    recording_wall_seconds: float
    episode_duration_seconds: float
    episode_samples: int
    video_bytes: int
    listed_in_datasets: bool
    listed_episodes: int
    listed_frames: int
    deployment_status: str
    action_steps: int
    recording_tasks: int
    recording_episodes: int
    inference_tasks: int
    inference_queue_depth: int
    active_compute_resources: int
    rig_released: bool
    upload_released: bool
    hub_repo_deleted: bool
    workspace_removed: bool


@dataclass
class _LocalRepo:
    repo_id: str
    path: Path
    private: bool
    created_at: datetime
    updated_at: datetime
    revision: str | None = None
    commit_count: int = 0


class _LocalHubApi:
    """Filesystem-only HfApi subset used only by ``--fake-hub`` and pytest."""

    def __init__(self, root: Path, *, namespace: str, token: str) -> None:
        self.root = root.resolve()
        self.namespace = namespace
        self._expected_token = token
        self._repos: dict[str, _LocalRepo] = {}
        self.active_transfers = 0

    def factory(self, token: str) -> _LocalHubApi:
        self._check_token(token)
        return self

    def whoami(self, *, token: str | None = None) -> dict[str, Any]:
        self._check_token(token)
        return {"name": self.namespace, "orgs": []}

    def repo_exists(
        self,
        *,
        repo_id: str,
        repo_type: str,
        token: str | None = None,
    ) -> bool:
        self._check_token(token)
        self._check_repo_type(repo_type)
        self._validate_repo_id(repo_id)
        return repo_id in self._repos

    def create_repo(
        self,
        *,
        repo_id: str,
        repo_type: str,
        private: bool,
        exist_ok: bool,
    ) -> str:
        self._check_repo_type(repo_type)
        self._validate_repo_id(repo_id)
        if repo_id in self._repos:
            if exist_ok:
                return f"mock://datasets/{repo_id}"
            raise FileExistsError(repo_id)
        path = (self.root / "repos" / repo_id).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("mock repository path escaped its root")
        (path / "worktree").mkdir(parents=True, exist_ok=False)
        (path / "snapshots").mkdir()
        now = datetime(2026, 1, 1, tzinfo=UTC)
        self._repos[repo_id] = _LocalRepo(
            repo_id=repo_id,
            path=path,
            private=private,
            created_at=now,
            updated_at=now,
        )
        return f"mock://datasets/{repo_id}"

    def upload_file(
        self,
        *,
        path_or_fileobj: str | Path,
        path_in_repo: str,
        repo_id: str,
        repo_type: str,
        token: str | None = None,
        commit_message: str,
    ) -> SimpleNamespace:
        del commit_message
        self._check_token(token)
        repo = self._repo(repo_id, repo_type)
        destination = self._destination(repo, path_in_repo)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(path_or_fileobj), destination)
        return SimpleNamespace(oid=self._commit(repo))

    def update_repo_settings(
        self,
        *,
        repo_id: str,
        repo_type: str,
        private: bool,
    ) -> None:
        repo = self._repo(repo_id, repo_type)
        repo.private = private

    def upload_folder(
        self,
        *,
        repo_id: str,
        repo_type: str,
        folder_path: str | Path,
        ignore_patterns: list[str],
        commit_message: str,
    ) -> SimpleNamespace:
        del commit_message
        repo = self._repo(repo_id, repo_type)
        source = Path(folder_path).resolve()
        self.active_transfers += 1
        try:
            for path in sorted(source.rglob("*")):
                if not path.is_file() or path.is_symlink():
                    continue
                relative = path.relative_to(source)
                relative_posix = relative.as_posix()
                if any(
                    pattern.endswith("/")
                    and relative_posix.startswith(pattern)
                    for pattern in ignore_patterns
                ):
                    continue
                destination = self._destination(repo, relative_posix)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
            revision = self._commit(repo)
        finally:
            self.active_transfers -= 1
        return SimpleNamespace(oid=revision)

    def repo_info(
        self,
        *,
        repo_id: str,
        repo_type: str,
        revision: str | None = None,
        token: str | None = None,
    ) -> SimpleNamespace:
        self._check_token(token)
        repo = self._repo(repo_id, repo_type)
        if repo.revision is None:
            raise FileNotFoundError("mock repository has no revision")
        if revision is not None and revision != repo.revision:
            snapshot = repo.path / "snapshots" / revision
            if not snapshot.is_dir():
                raise FileNotFoundError(revision)
        return SimpleNamespace(
            id=repo_id,
            repo_id=repo_id,
            sha=revision or repo.revision,
            private=repo.private,
        )

    def list_repo_files(
        self,
        *,
        repo_id: str,
        repo_type: str,
        revision: str | None = None,
        token: str | None = None,
    ) -> list[str]:
        self._check_token(token)
        repo = self._repo(repo_id, repo_type)
        root = self._revision_root(repo, revision)
        return sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        )

    def hf_hub_download(
        self,
        *,
        repo_id: str,
        filename: str,
        repo_type: str,
        revision: str | None = None,
        token: str | None = None,
    ) -> str:
        self._check_token(token)
        repo = self._repo(repo_id, repo_type)
        root = self._revision_root(repo, revision)
        relative = self._safe_relative(filename)
        result = (root / relative).resolve()
        if not result.is_relative_to(root.resolve()) or not result.is_file():
            raise FileNotFoundError(filename)
        return str(result)

    def list_datasets(
        self,
        *,
        author: str,
        filter: str,
        sort: str,
        direction: int,
        expand: list[str],
        token: str,
    ) -> list[SimpleNamespace]:
        del sort, direction, expand
        self._check_token(token)
        if author != self.namespace or filter != "LeRobot":
            return []
        items: list[SimpleNamespace] = []
        for repo in self._repos.values():
            if repo.revision is None:
                continue
            items.append(
                SimpleNamespace(
                    id=repo.repo_id,
                    repo_id=repo.repo_id,
                    sha=repo.revision,
                    private=repo.private,
                    gated=False,
                    created_at=repo.created_at,
                    last_modified=repo.updated_at,
                    tags=["LeRobot", "ctrl-pi", "yam"],
                )
            )
        return items

    def delete_repo(
        self,
        *,
        repo_id: str,
        repo_type: str,
        token: str | None = None,
        missing_ok: bool = False,
    ) -> None:
        self._check_token(token)
        self._check_repo_type(repo_type)
        self._validate_repo_id(repo_id)
        repo = self._repos.get(repo_id)
        if repo is None:
            if missing_ok:
                return
            raise FileNotFoundError(repo_id)
        shutil.rmtree(repo.path)
        del self._repos[repo_id]

    def _repo(self, repo_id: str, repo_type: str) -> _LocalRepo:
        self._check_repo_type(repo_type)
        self._validate_repo_id(repo_id)
        try:
            return self._repos[repo_id]
        except KeyError as error:
            raise FileNotFoundError(repo_id) from error

    def _commit(self, repo: _LocalRepo) -> str:
        repo.commit_count += 1
        revision = f"{repo.commit_count:040x}"
        snapshot = repo.path / "snapshots" / revision
        shutil.copytree(repo.path / "worktree", snapshot)
        repo.revision = revision
        repo.updated_at = repo.created_at.replace(second=repo.commit_count)
        return revision

    def _revision_root(self, repo: _LocalRepo, revision: str | None) -> Path:
        if revision is None:
            return repo.path / "worktree"
        root = repo.path / "snapshots" / revision
        if not root.is_dir():
            raise FileNotFoundError(revision)
        return root

    def _destination(self, repo: _LocalRepo, value: str) -> Path:
        root = (repo.path / "worktree").resolve()
        destination = (root / self._safe_relative(value)).resolve()
        if not destination.is_relative_to(root):
            raise ValueError("mock Hub path escaped its repository")
        return destination

    @staticmethod
    def _safe_relative(value: str) -> Path:
        posix = PurePosixPath(value)
        if (
            posix.is_absolute()
            or not posix.parts
            or any(part in {"", ".", ".."} for part in posix.parts)
        ):
            raise ValueError("mock Hub path is invalid")
        return Path(*posix.parts)

    def _validate_repo_id(self, repo_id: str) -> None:
        if repo_id.count("/") != 1 or not repo_id.startswith(f"{self.namespace}/"):
            raise ValueError("mock Hub repository is outside its namespace")
        owner, name = repo_id.split("/", 1)
        if owner != self.namespace or not name or name in {".", ".."} or "/" in name:
            raise ValueError("mock Hub repository id is invalid")

    @staticmethod
    def _check_repo_type(repo_type: str) -> None:
        if repo_type != "dataset":
            raise ValueError("mock Hub supports dataset repositories only")

    def _check_token(self, token: str | None) -> None:
        if token is not None and token != self._expected_token:
            raise PermissionError("mock Hub token mismatch")


def _episode_metadata(result: EpisodeResult) -> dict[str, Any]:
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


@contextlib.contextmanager
def _offline_environment(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    names = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {name: os.environ.get(name) for name in names}
    os.environ.update({name: "1" for name in names})
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


async def _to_thread_terminal(callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Settle a thread-backed operation before cancellation can release ownership."""

    task = asyncio.create_task(
        asyncio.to_thread(functools.partial(callback, *args, **kwargs))
    )
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        raise


async def _record_episode(
    manager: RecordingManager,
    *,
    recording_id: str,
) -> tuple[EpisodeResult, float, int]:
    await manager.startup()
    await manager.start_teleop(
        recording_id,
        "yam-leader",
        "yam-follower",
        episode_count=0,
        status="draft",
    )
    await manager.start_episode(
        recording_id,
        fps=_RECORD_FPS,
        metadata={"operator": "make-smoke", "notes": "deterministic mock episode"},
    )
    started = time.monotonic()
    await asyncio.sleep(_RECORD_SECONDS)
    wall_seconds = time.monotonic() - started
    _, result = await manager.stop_episode(
        recording_id,
        success=True,
        notes="ctrl-pi recurring smoke gate",
    )
    await manager.confirm_episode_persisted(recording_id, 1, "teleop")
    await manager.stop_teleop(recording_id)
    episode_root = manager.staging_dir / result.artifact_key
    video_path = episode_root / "video.mp4"
    samples_path = episode_root / "samples.jsonl"
    _require(wall_seconds >= _RECORD_SECONDS, "Recording ended before five real seconds elapsed.")
    _require(
        result.duration_seconds >= _RECORD_SECONDS,
        "The finalized episode is shorter than five seconds.",
    )
    _require(
        result.fps == _RECORD_FPS and result.sample_count >= 50,
        "The finalized episode has incomplete samples.",
    )
    _require(
        video_path.is_file() and video_path.stat().st_size > 0,
        "The finalized episode has no MP4 artifact.",
    )
    _require(
        samples_path.is_file() and samples_path.stat().st_size > 0,
        "The finalized episode has no sample artifact.",
    )
    return result, wall_seconds, video_path.stat().st_size


def _safe_delete_dataset(
    *,
    api: Any,
    uploader: HFDatasetUploader,
    repo_id: str,
    recording_id: str,
    token: str,
    expected_revision: str | None,
) -> bool:
    if not api.repo_exists(repo_id=repo_id, repo_type="dataset", token=token):
        return True
    info = api.repo_info(repo_id=repo_id, repo_type="dataset", token=token)
    observed_revision = getattr(info, "sha", None)
    _require(
        isinstance(observed_revision, str) and _REVISION.fullmatch(observed_revision) is not None,
        f"Dataset cleanup was refused for {repo_id}: current revision is invalid.",
    )
    if expected_revision is not None:
        _require(
            observed_revision == expected_revision,
            f"Dataset cleanup was refused for {repo_id}: its revision changed after verification.",
        )
    _require(
        uploader.repository_owned_by(repo_id=repo_id, recording_id=recording_id, token=token),
        f"Dataset cleanup was refused for {repo_id}: its ctrl-pi ownership marker does not match.",
    )
    HFDatasetUploader._verify_remote_marker(
        api,
        repo_id=repo_id,
        recording_id=recording_id,
        token=token,
        revision=observed_revision,
        ownership_check=True,
    )
    second = api.repo_info(repo_id=repo_id, repo_type="dataset", token=token)
    _require(
        getattr(second, "sha", None) == observed_revision,
        f"Dataset cleanup was refused for {repo_id}: "
        "its revision changed during ownership verification.",
    )
    api.delete_repo(
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        missing_ok=False,
    )
    _require(
        not api.repo_exists(repo_id=repo_id, repo_type="dataset", token=token),
        f"Dataset cleanup could not verify deletion of {repo_id}.",
    )
    return True


async def _wait_for_dataset(
    browser: HFDatasetBrowser,
    *,
    namespace: str,
    token: str,
    repo_id: str,
    revision: str,
    timeout_seconds: float = 30.0,
) -> Any:
    deadline = time.monotonic() + timeout_seconds
    while True:
        page = await asyncio.to_thread(
            browser.list_namespace,
            namespace=namespace,
            token=token,
            limit=100,
            cursor=None,
            refresh=True,
        )
        listed = next(
            (
                item
                for item in page.datasets
                if item.repo_id == repo_id
                and item.revision == revision
                and item.lerobot is not None
            ),
            None,
        )
        if listed is not None:
            return listed
        if time.monotonic() >= deadline:
            raise SmokeFailure(
                "The uploaded dataset did not become readable through Datasets."
            )
        await asyncio.sleep(1)


async def _exercise_inference(
    *,
    workspace: Path,
    driver: MockYAMDriver,
    camera: MockCamera,
    rig_lease: RigLease,
    recording_manager: RecordingManager,
) -> tuple[str, int, int, int, int]:
    engine = create_engine(f"sqlite+pysqlite:///{workspace / 'smoke.sqlite3'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    target = StubComputeTarget()
    service = DeploymentService(
        target,
        session_factory=factory,
        nonce_factory=lambda: "ctrl-pi-smoke-deployment",
        stop_verify_timeout_seconds=0,
        poll_interval_seconds=0,
    )

    def transport_factory(record: Any) -> InProcessInferenceTransport:
        _require(
            record.runtime == "stub",
            "Smoke attempted to construct a non-stub runtime.",
        )
        runtime = StubInferenceRuntime(runtime="stub")
        runtime.load(
            RuntimeLoadSpec(
                model_repo=record.model_repo,
                revision=record.checkpoint_revision,
                local_model_path=None,
                device="cpu",
                actions_per_chunk=32,
            )
        )
        return InProcessInferenceTransport(runtime)

    session_manager = InferenceSessionManager(
        deployment_service=service,
        driver=driver,
        camera=camera,
        rig_lease=rig_lease,
        recording_manager=recording_manager,
        transport_factory=transport_factory,
        session_factory=factory,
        recording_fps=_RECORD_FPS,
    )
    deployment_id: uuid.UUID | None = None
    final_status = "failed"
    steps = 0
    session_counts = (0, 0)
    try:
        await session_manager.startup()
        with factory() as db:
            deployment = await service.deploy(
                db,
                name="M11 smoke stub policy",
                model_repo=_POLICY_REPO,
                checkpoint_revision=_POLICY_REVISION,
                runtime="stub",
                compute_size="CPU",
                timeout_seconds=60,
            )
            deployment_id = deployment.id
            _require(
                deployment.status == "running",
                "The stub policy did not reach running state.",
            )
            started = await session_manager.start(
                db,
                deployment_id,
                InferenceStartOptions(
                    arm_id="yam-follower",
                    task="Execute the ctrl-pi smoke policy",
                    record_session=False,
                    max_steps=_ACTION_STEPS,
                ),
            )
            _require(
                started.session_status == "running" and started.endpoint_healthy,
                "The mock-arm inference session did not start healthily.",
            )

        deadline = time.monotonic() + 15
        terminal = await session_manager.read(deployment_id)
        while terminal.session_status not in {"stopped", "failed"}:
            _require(
                time.monotonic() < deadline,
                "The mock-arm inference session did not reach its action limit.",
            )
            await asyncio.sleep(0.01)
            terminal = await session_manager.read(deployment_id)
        with factory() as stop_db:
            terminal = await session_manager.stop(
                stop_db,
                deployment_id,
                InferenceStopOptions(),
            )
        _require(
            terminal.last_error is None,
            "The mock-arm policy loop stopped with an error.",
        )
        _require(
            terminal.steps_executed == _ACTION_STEPS,
            "The policy loop did not execute exactly 100 actions.",
        )
        _require(
            terminal.session_status == "stopped"
            and terminal.deployment.status == "stopped"
            and terminal.teardown_verified,
            "The inference session did not verify complete teardown.",
        )
        _require(
            terminal.queue_depth == 0,
            "The inference session retained queued actions.",
        )
        final_status = terminal.deployment.status
        steps = terminal.steps_executed
        session_counts = await session_manager.active_resource_counts()
        _require(
            session_counts == (0, 0),
            "The inference manager retained a task or queued action.",
        )
    finally:
        with contextlib.suppress(Exception):
            await session_manager.shutdown()
        with contextlib.suppress(Exception):
            session_counts = await session_manager.active_resource_counts()
        if deployment_id is not None:
            with factory() as cleanup_db:
                with contextlib.suppress(Exception):
                    stopped = await service.stop(cleanup_db, deployment_id)
                    final_status = stopped.status
        active_compute = sum(not state.stopped_verified for state in target.list_owned())
        engine.dispose()
    _require(active_compute == 0, "The stub compute target retained an active resource.")
    return final_status, steps, session_counts[0], session_counts[1], active_compute


async def run_smoke(
    *,
    fake_hub: bool = False,
    temp_parent: Path | None = None,
    output: Callable[[str], None] = print,
) -> SmokeEvidence:
    config = AppConfig()
    token = _FAKE_TOKEN if fake_hub else (
        config.hf_token.get_secret_value() if config.hf_token is not None else ""
    )
    namespace = _FAKE_NAMESPACE if fake_hub else (config.hf_namespace or "")
    _require(bool(token.strip()), "HF_TOKEN is required for make smoke.")
    _require(bool(namespace.strip()), "HF_NAMESPACE is required for make smoke.")
    _require(
        namespace == namespace.strip(),
        "HF_NAMESPACE must not contain surrounding whitespace.",
    )

    recording_id = str(uuid.uuid4())
    suffix = f"{datetime.now(UTC):%Y%m%d%H%M%S}-{uuid.uuid4().hex[:10]}"
    repo_name = f"ctrl-pi-smoke-{suffix}"
    repo_id = HFDatasetUploader.repo_id(namespace, repo_name)
    workspace_path: Path | None = None
    evidence: SmokeEvidence | None = None
    cleanup_errors: list[str] = []
    primary_error: BaseException | None = None

    with tempfile.TemporaryDirectory(prefix="ctrl-pi-smoke-", dir=temp_parent) as raw_workspace:
        workspace = Path(raw_workspace)
        workspace_path = workspace
        recording_root = workspace / "recordings"
        driver = MockYAMDriver()
        camera = MockCamera(width=96, height=64)
        rig_lease = RigLease()
        manager = RecordingManager(
            driver=driver,
            camera=camera,
            staging_dir=recording_root,
            rig_lease=rig_lease,
        )
        local_api = (
            _LocalHubApi(workspace / "hub", namespace=namespace, token=token)
            if fake_hub
            else None
        )
        if local_api is None:
            from huggingface_hub import HfApi

            api: Any = HfApi(token=token)
            hub_factory = None
        else:
            api = local_api
            hub_factory = local_api.factory
        uploader = HFDatasetUploader(
            recording_root,
            dataset_staging_dir=workspace / "lerobot",
            hub_api_factory=hub_factory,
        )
        browser = HFDatasetBrowser(hub_api_factory=hub_factory)
        upload_result: DatasetUploadResult | None = None
        upload_attempted = False
        reserved = False
        recording_counts = (0, 0)
        inference_counts = (0, 0)
        active_compute = 0
        deployment_status = "failed"
        action_steps = 0
        hub_deleted = False
        try:
            with _offline_environment(fake_hub):
                episode, wall_seconds, video_bytes = await _record_episode(
                    manager,
                    recording_id=recording_id,
                )
                output(
                    "[smoke] record: PASS "
                    f"wall={wall_seconds:.3f}s duration={episode.duration_seconds:.3f}s "
                    f"samples={episode.sample_count} mp4_bytes={video_bytes}"
                )

                source = RecordingUploadSource(
                    recording_id=recording_id,
                    task="ctrl-pi full mock smoke episode",
                    episode_count=1,
                    metadata={
                        "smoke": {"schema": "ctrl-pi.make-smoke", "version": 1},
                        "episodes": [_episode_metadata(episode)],
                    },
                )
                uploader.reserve(recording_id, repo_id)
                reserved = True
                upload_attempted = True
                try:
                    upload_result = await _to_thread_terminal(
                        uploader.upload,
                        source,
                        namespace,
                        repo_name,
                        True,
                        token,
                    )
                finally:
                    uploader.release(recording_id, repo_id)
                    reserved = False
                _require(
                    upload_result.repo_id == repo_id,
                    "The Hub upload returned the wrong repository identity.",
                )
                _require(
                    isinstance(upload_result.revision, str)
                    and _REVISION.fullmatch(upload_result.revision) is not None,
                    "The Hub upload did not return an immutable revision.",
                )
                uploaded_info = await asyncio.to_thread(
                    api.repo_info,
                    repo_id=repo_id,
                    repo_type="dataset",
                    revision=upload_result.revision,
                    token=token,
                )
                _require(
                    getattr(uploaded_info, "sha", None) == upload_result.revision,
                    "The Hub repository no longer points at the uploaded revision.",
                )
                _require(
                    getattr(uploaded_info, "private", None) is True,
                    "The Hub repository did not verify as private.",
                )
                output(
                    "[smoke] upload: PASS "
                    f"mode={'fake' if fake_hub else 'real'} private=true "
                    f"repo={repo_id} revision={upload_result.revision} "
                    f"frames={upload_result.total_frames}"
                )

                listed = await _wait_for_dataset(
                    browser,
                    namespace=namespace,
                    token=token,
                    repo_id=repo_id,
                    revision=upload_result.revision,
                )
                _require(listed.private, "The smoke dataset was not private.")
                _require(
                    listed.revision == upload_result.revision,
                    "Datasets returned a different Hub revision.",
                )
                _require(
                    listed.lerobot is not None,
                    "Datasets could not read the LeRobot metadata.",
                )
                assert listed.lerobot is not None
                _require(listed.lerobot.total_episodes == 1, "Datasets did not report one episode.")
                _require(
                    listed.lerobot.total_frames == upload_result.total_frames,
                    "Datasets reported a different frame count.",
                )
                output(
                    "[smoke] datasets: PASS "
                    f"repo={repo_id} revision={listed.revision} "
                    f"episodes={listed.lerobot.total_episodes} frames={listed.lerobot.total_frames}"
                )

                (
                    deployment_status,
                    action_steps,
                    inference_counts_0,
                    inference_counts_1,
                    active_compute,
                ) = await _exercise_inference(
                    workspace=workspace,
                    driver=driver,
                    camera=camera,
                    rig_lease=rig_lease,
                    recording_manager=manager,
                )
                inference_counts = (inference_counts_0, inference_counts_1)
                output("[smoke] deploy: PASS target=stub runtime=stub status=running")
                output(f"[smoke] inference: PASS steps={action_steps} arm=yam-follower")

                recording_counts = await manager.active_resource_counts()
                evidence = SmokeEvidence(
                    hub_mode="fake" if fake_hub else "real",
                    repo_id=repo_id,
                    revision=upload_result.revision,
                    private=listed.private,
                    recording_wall_seconds=wall_seconds,
                    episode_duration_seconds=episode.duration_seconds,
                    episode_samples=episode.sample_count,
                    video_bytes=video_bytes,
                    listed_in_datasets=True,
                    listed_episodes=listed.lerobot.total_episodes,
                    listed_frames=listed.lerobot.total_frames,
                    deployment_status=deployment_status,
                    action_steps=action_steps,
                    recording_tasks=recording_counts[0],
                    recording_episodes=recording_counts[1],
                    inference_tasks=inference_counts[0],
                    inference_queue_depth=inference_counts[1],
                    active_compute_resources=active_compute,
                    rig_released=rig_lease.current() is None,
                    upload_released=not uploader.is_active(recording_id),
                    hub_repo_deleted=False,
                    workspace_removed=False,
                )
        except BaseException as error:
            primary_error = error
        finally:
            if reserved:
                uploader.release(recording_id, repo_id)
            try:
                await manager.shutdown()
            except Exception:
                cleanup_errors.append("recording shutdown failed")
            try:
                recording_counts = await manager.active_resource_counts()
                if recording_counts != (0, 0):
                    cleanup_errors.append("recording resources remain active")
            except Exception:
                cleanup_errors.append("recording teardown could not be verified")
            if rig_lease.current() is not None:
                cleanup_errors.append("the robot rig lease remains held")
            if uploader.is_active(recording_id):
                cleanup_errors.append("the dataset uploader remains active")
            if local_api is not None and local_api.active_transfers != 0:
                cleanup_errors.append("the fake Hub retained an active transfer")
            if upload_attempted:
                try:
                    hub_deleted = _safe_delete_dataset(
                        api=api,
                        uploader=uploader,
                        repo_id=repo_id,
                        recording_id=recording_id,
                        token=token,
                        expected_revision=(
                            upload_result.revision if upload_result is not None else None
                        ),
                    )
                except Exception:
                    cleanup_errors.append(
                        f"remote dataset cleanup was refused; inspect {repo_id} manually"
                    )

        if evidence is not None:
            evidence = replace(
                evidence,
                recording_tasks=recording_counts[0],
                recording_episodes=recording_counts[1],
                rig_released=rig_lease.current() is None,
                upload_released=not uploader.is_active(recording_id),
                hub_repo_deleted=hub_deleted,
            )

    workspace_removed = workspace_path is not None and not workspace_path.exists()
    if evidence is not None:
        evidence = replace(evidence, workspace_removed=workspace_removed)
    if not workspace_removed:
        cleanup_errors.append("the temporary smoke workspace was not removed")
    if primary_error is not None:
        if cleanup_errors:
            raise SmokeFailure(
                "Smoke failed and cleanup was incomplete: " + "; ".join(cleanup_errors)
            ) from primary_error
        raise primary_error
    if cleanup_errors:
        raise SmokeFailure("Smoke cleanup failed: " + "; ".join(cleanup_errors))
    _require(evidence is not None, "Smoke produced no acceptance evidence.")
    assert evidence is not None
    _require(
        evidence.recording_tasks == 0 and evidence.recording_episodes == 0,
        "Recording teardown was not complete.",
    )
    _require(
        evidence.inference_tasks == 0 and evidence.inference_queue_depth == 0,
        "Inference teardown was not complete.",
    )
    _require(evidence.active_compute_resources == 0, "Compute teardown was not complete.")
    _require(
        evidence.rig_released and evidence.upload_released,
        "A process-local resource remained active.",
    )
    _require(
        evidence.hub_repo_deleted and evidence.workspace_removed,
        "Smoke artifacts were not cleaned up.",
    )
    output(
        "[smoke] teardown: PASS recording=0 inference=0 compute=0 "
        "rig=free upload=idle hub_repo=deleted workspace=removed"
    )
    output("[smoke] PASS")
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the ctrl-pi M11 full mock loop; the default uses the "
            "configured real HF account."
        )
    )
    parser.add_argument(
        "--fake-hub",
        action="store_true",
        help="use a deterministic filesystem Hub for offline pytest/development only",
    )
    arguments = parser.parse_args(argv)
    try:
        asyncio.run(run_smoke(fake_hub=arguments.fake_hub))
    except KeyboardInterrupt:
        print("[smoke] FAIL interrupted after cleanup")
        return 130
    except SmokeFailure as error:
        print(f"[smoke] FAIL {error}")
        return 1
    except Exception:
        print("[smoke] FAIL an unexpected error occurred after cleanup")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
