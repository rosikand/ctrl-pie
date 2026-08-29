from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from ctrl_pi.managed_training_artifacts import (
    HFManagedTrainingArtifactService,
    ManagedTrainingArtifactConflictError,
    ManagedTrainingArtifactProviderError,
)
from ctrl_pi.training_compute import (
    MANAGED_SMOLVLA_DEPENDENCY_DIR,
    MANAGED_SMOLVLA_DEPENDENCY_FILES,
    MANAGED_TRAINING_MARKER_PATH,
    managed_training_marker,
)


class FakeHub:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.marker: bytes | None = None
        self.marker_sha = "c" * 40
        self.output_private = True
        self.create_error: Exception | None = None
        self.create_calls: list[dict[str, object]] = []
        self.output_files = {
            MANAGED_TRAINING_MARKER_PATH,
            "config.json",
            "model.safetensors",
            "policy_preprocessor.json",
            "policy_postprocessor.json",
            *(
                f"{MANAGED_SMOLVLA_DEPENDENCY_DIR}/{name}"
                for name in MANAGED_SMOLVLA_DEPENDENCY_FILES
            ),
        }

    def whoami(self, **_kwargs):
        return {"name": "acme", "orgs": []}

    def dataset_info(self, *, repo_id: str, **_kwargs):
        return SimpleNamespace(id=repo_id, sha="a" * 40)

    def model_info(self, *, repo_id: str, revision=None, **_kwargs):
        if repo_id == "acme/base":
            return SimpleNamespace(id=repo_id, sha="b" * 40)
        return SimpleNamespace(
            id=repo_id,
            sha=revision,
            private=self.output_private,
            siblings=[SimpleNamespace(rfilename=name) for name in self.output_files],
        )

    def create_repo(self, **kwargs):
        self.create_calls.append(kwargs)
        if self.create_error is not None:
            raise self.create_error
        return SimpleNamespace(repo_id=kwargs["repo_id"])

    def upload_file(self, *, path_or_fileobj, **_kwargs):
        self.marker = bytes(path_or_fileobj)
        return SimpleNamespace(oid=self.marker_sha)

    def hf_hub_download(self, **_kwargs):
        path = self.root / "marker.json"
        path.write_bytes(self.marker or b"")
        return str(path)


def test_hf_artifacts_create_new_marked_repo_and_verify_final_root(
    tmp_path: Path,
) -> None:
    api = FakeHub(tmp_path)
    service = HFManagedTrainingArtifactService(
        "hf_test_token", "acme", hub_api_factory=lambda _token: api
    )
    job_id = uuid.uuid4()
    request_hash = "d" * 64

    prepared = service.prepare(
        job_id=job_id,
        request_hash=request_hash,
        dataset_repo="other/data",
        dataset_revision="main",
        base_model="acme/base",
        base_model_revision=None,
        output_model_repo="acme/output",
        output_private=True,
    )

    assert prepared.dataset_revision == "a" * 40
    assert prepared.base_model_revision == "b" * 40
    assert prepared.output_marker_revision == api.marker_sha
    assert api.marker == managed_training_marker(job_id, request_hash)
    assert api.create_calls[0]["exist_ok"] is False
    assert api.create_calls[0]["private"] is True
    assert (
        service.verify_output_revision(
            job_id=job_id,
            request_hash=request_hash,
            output_model_repo="acme/output",
            output_private=True,
            revision=api.marker_sha,
            require_deployable_root=True,
        )
        == api.marker_sha
    )


@pytest.mark.parametrize(
    ("status", "error_type"),
    [
        (409, ManagedTrainingArtifactConflictError),
        (503, ManagedTrainingArtifactProviderError),
        (None, ManagedTrainingArtifactProviderError),
    ],
)
def test_create_repo_distinguishes_confirmed_collision_from_ambiguous_failure(
    tmp_path: Path, status: int | None, error_type: type[Exception]
) -> None:
    api = FakeHub(tmp_path)
    error = RuntimeError("secret provider detail")
    if status is not None:
        error.response = SimpleNamespace(status_code=status)  # type: ignore[attr-defined]
    api.create_error = error
    service = HFManagedTrainingArtifactService(
        "hf_test_token", "acme", hub_api_factory=lambda _token: api
    )

    with pytest.raises(error_type) as caught:
        service.prepare(
            job_id=uuid.uuid4(),
            request_hash="d" * 64,
            dataset_repo="other/data",
            dataset_revision=None,
            base_model="acme/base",
            base_model_revision=None,
            output_model_repo="acme/output",
            output_private=True,
        )

    assert "secret provider detail" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert api.marker is None


def test_output_visibility_and_marker_are_reverified(tmp_path: Path) -> None:
    api = FakeHub(tmp_path)
    service = HFManagedTrainingArtifactService(
        "hf_test_token", "acme", hub_api_factory=lambda _token: api
    )
    job_id = uuid.uuid4()
    request_hash = "d" * 64
    api.marker = managed_training_marker(job_id, request_hash)
    api.output_private = False

    with pytest.raises(ManagedTrainingArtifactProviderError, match="identity"):
        service.verify_output_revision(
            job_id=job_id,
            request_hash=request_hash,
            output_model_repo="acme/output",
            output_private=True,
            revision=api.marker_sha,
            require_deployable_root=False,
        )


@pytest.mark.parametrize(
    "missing",
    [
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        *[
            f"{MANAGED_SMOLVLA_DEPENDENCY_DIR}/{name}"
            for name in MANAGED_SMOLVLA_DEPENDENCY_FILES
        ],
    ],
)
def test_final_output_requires_every_lerobot_runtime_root_file(
    tmp_path: Path, missing: str
) -> None:
    api = FakeHub(tmp_path)
    service = HFManagedTrainingArtifactService(
        "hf_test_token", "acme", hub_api_factory=lambda _token: api
    )
    job_id = uuid.uuid4()
    request_hash = "d" * 64
    api.marker = managed_training_marker(job_id, request_hash)
    api.output_files.remove(missing)

    with pytest.raises(ManagedTrainingArtifactProviderError, match="deployable"):
        service.verify_output_revision(
            job_id=job_id,
            request_hash=request_hash,
            output_model_repo="acme/output",
            output_private=True,
            revision=api.marker_sha,
            require_deployable_root=True,
        )
