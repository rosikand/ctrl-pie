from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ctrl_pi.training_compute import (
    MANAGED_SMOLVLA_DEPENDENCY_DIR,
    MANAGED_SMOLVLA_DEPENDENCY_FILES,
    MANAGED_TRAINING_MARKER_PATH,
    managed_training_marker,
)

_SHA = re.compile(r"[0-9a-f]{40}")
_NAMESPACE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?")


class ManagedTrainingArtifactError(RuntimeError):
    """A sanitized Hugging Face managed-training artifact failure."""


class ManagedTrainingArtifactConfigurationError(ManagedTrainingArtifactError):
    pass


class ManagedTrainingArtifactConflictError(ManagedTrainingArtifactError):
    pass


class ManagedTrainingArtifactProviderError(ManagedTrainingArtifactError):
    pass


class ManagedTrainingArtifactVerificationError(ManagedTrainingArtifactError):
    """An immutable output exists but violates the managed artifact contract."""

    pass


@dataclass(frozen=True)
class PreparedTrainingArtifacts:
    dataset_repo: str
    dataset_revision: str
    base_model: str
    base_model_revision: str
    output_model_repo: str
    output_marker_revision: str
    output_private: bool


class ManagedTrainingArtifactService(Protocol):
    def prepare(
        self,
        *,
        job_id: uuid.UUID,
        request_hash: str,
        dataset_repo: str,
        dataset_revision: str | None,
        base_model: str,
        base_model_revision: str | None,
        output_model_repo: str,
        output_private: bool,
    ) -> PreparedTrainingArtifacts: ...

    def verify_output_revision(
        self,
        *,
        job_id: uuid.UUID,
        request_hash: str,
        output_model_repo: str,
        output_private: bool,
        revision: str,
        require_deployable_root: bool,
    ) -> str: ...


class StubManagedTrainingArtifactService:
    """Offline artifact identity used only with the mock training target."""

    def prepare(
        self,
        *,
        job_id: uuid.UUID,
        request_hash: str,
        dataset_repo: str,
        dataset_revision: str | None,
        base_model: str,
        base_model_revision: str | None,
        output_model_repo: str,
        output_private: bool,
    ) -> PreparedTrainingArtifacts:
        marker = managed_training_marker(job_id, request_hash)
        return PreparedTrainingArtifacts(
            dataset_repo=dataset_repo,
            dataset_revision=_stub_sha("dataset", dataset_repo, dataset_revision),
            base_model=base_model,
            base_model_revision=_stub_sha("model", base_model, base_model_revision),
            output_model_repo=output_model_repo,
            output_marker_revision=hashlib.sha1(marker).hexdigest(),
            output_private=output_private,
        )

    def verify_output_revision(
        self,
        *,
        job_id: uuid.UUID,
        request_hash: str,
        output_model_repo: str,
        output_private: bool,
        revision: str,
        require_deployable_root: bool,
    ) -> str:
        managed_training_marker(job_id, request_hash)
        if _SHA.fullmatch(revision) is None:
            raise ManagedTrainingArtifactProviderError(
                "The mock output revision is invalid."
            )
        return revision


class HFManagedTrainingArtifactService:
    """Pins input commits and owns a new, marked output model repository."""

    def __init__(
        self,
        token: str | None,
        namespace: str | None,
        *,
        hub_api_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self._token = token
        self._namespace = namespace
        self._hub_api_factory = hub_api_factory

    def prepare(
        self,
        *,
        job_id: uuid.UUID,
        request_hash: str,
        dataset_repo: str,
        dataset_revision: str | None,
        base_model: str,
        base_model_revision: str | None,
        output_model_repo: str,
        output_private: bool,
    ) -> PreparedTrainingArtifacts:
        token, namespace, api = self._configured_api()
        self._validate_output_repo(output_model_repo, namespace)
        self._validate_identity(api, token, namespace)
        resolved_dataset = self._resolve(
            api,
            token=token,
            repo_id=dataset_repo,
            revision=dataset_revision,
            repo_type="dataset",
        )
        resolved_model = self._resolve(
            api,
            token=token,
            repo_id=base_model,
            revision=base_model_revision,
            repo_type="model",
        )
        create_failure: Exception | None = None
        try:
            api.create_repo(
                repo_id=output_model_repo,
                repo_type="model",
                private=output_private,
                exist_ok=False,
                token=token,
            )
        except Exception as error:
            create_failure = error
        if create_failure is not None:
            status_code = getattr(
                getattr(create_failure, "response", None), "status_code", None
            )
            if status_code == 409:
                raise ManagedTrainingArtifactConflictError(
                    "The output model repository already exists."
                ) from None
            raise ManagedTrainingArtifactProviderError(
                "The output model repository creation outcome could not be verified."
            ) from None

        marker_bytes = managed_training_marker(job_id, request_hash)
        upload_failed = False
        commit: Any = None
        try:
            commit = api.upload_file(
                path_or_fileobj=marker_bytes,
                path_in_repo=MANAGED_TRAINING_MARKER_PATH,
                repo_id=output_model_repo,
                repo_type="model",
                token=token,
                commit_message=f"Initialize ctrl-pi managed training job {job_id}",
            )
        except Exception:
            upload_failed = True
        if upload_failed:
            raise ManagedTrainingArtifactProviderError(
                "The output model ownership marker could not be uploaded."
            ) from None
        marker_revision = getattr(commit, "oid", None)
        if not isinstance(marker_revision, str) or re.fullmatch(
            r"[0-9a-fA-F]{40}", marker_revision
        ) is None:
            raise ManagedTrainingArtifactProviderError(
                "Hugging Face returned an invalid marker commit identity."
            )
        marker_revision = marker_revision.casefold()
        self._verify_output(
            api,
            token=token,
            job_id=job_id,
            request_hash=request_hash,
            output_model_repo=output_model_repo,
            output_private=output_private,
            revision=marker_revision,
            require_deployable_root=False,
        )
        return PreparedTrainingArtifacts(
            dataset_repo=dataset_repo,
            dataset_revision=resolved_dataset,
            base_model=base_model,
            base_model_revision=resolved_model,
            output_model_repo=output_model_repo,
            output_marker_revision=marker_revision,
            output_private=output_private,
        )

    def verify_output_revision(
        self,
        *,
        job_id: uuid.UUID,
        request_hash: str,
        output_model_repo: str,
        output_private: bool,
        revision: str,
        require_deployable_root: bool,
    ) -> str:
        if _SHA.fullmatch(revision) is None:
            raise ManagedTrainingArtifactProviderError(
                "The output model revision is not an immutable Hub commit."
            )
        token, namespace, api = self._configured_api()
        self._validate_output_repo(output_model_repo, namespace)
        self._verify_output(
            api,
            token=token,
            job_id=job_id,
            request_hash=request_hash,
            output_model_repo=output_model_repo,
            output_private=output_private,
            revision=revision,
            require_deployable_root=require_deployable_root,
        )
        return revision

    def _configured_api(self) -> tuple[str, str, Any]:
        token = self._token
        namespace = self._namespace
        if token is None or not token.strip():
            raise ManagedTrainingArtifactConfigurationError(
                "HF_TOKEN is required for managed training."
            )
        if (
            namespace is None
            or _NAMESPACE.fullmatch(namespace) is None
            or ".." in namespace
            or "--" in namespace
        ):
            raise ManagedTrainingArtifactConfigurationError(
                "HF_NAMESPACE is required for managed training."
            )
        construction_failed = False
        api: Any = None
        try:
            if self._hub_api_factory is None:
                from huggingface_hub import HfApi

                api = HfApi(token=token)
            else:
                api = self._hub_api_factory(token)
        except Exception:
            construction_failed = True
        if construction_failed:
            raise ManagedTrainingArtifactConfigurationError(
                "The Hugging Face client could not be initialized."
            ) from None
        return token, namespace, api

    @staticmethod
    def _validate_output_repo(repo_id: str, namespace: str) -> None:
        if repo_id.count("/") != 1 or repo_id.split("/", 1)[0] != namespace:
            raise ManagedTrainingArtifactConfigurationError(
                "The output model repository must use HF_NAMESPACE."
            )

    @staticmethod
    def _validate_identity(api: Any, token: str, namespace: str) -> None:
        failed = False
        identity: Any = None
        try:
            identity = api.whoami(token=token)
        except Exception:
            failed = True
        if failed or not isinstance(identity, dict):
            raise ManagedTrainingArtifactProviderError(
                "Hugging Face authentication could not be verified."
            ) from None
        organizations = identity.get("orgs", [])
        if not isinstance(organizations, (list, tuple)):
            raise ManagedTrainingArtifactProviderError(
                "Hugging Face returned an invalid identity."
            )
        names = {str(identity.get("name", "")).casefold()}
        for organization in organizations[:100]:
            name = (
                organization.get("name")
                if isinstance(organization, dict)
                else organization
            )
            if name:
                names.add(str(name).casefold())
        if namespace.casefold() not in names:
            raise ManagedTrainingArtifactConfigurationError(
                "HF_NAMESPACE is not controlled by HF_TOKEN."
            )

    @staticmethod
    def _resolve(
        api: Any,
        *,
        token: str,
        repo_id: str,
        revision: str | None,
        repo_type: str,
    ) -> str:
        failed = False
        info: Any = None
        try:
            if repo_type == "dataset":
                info = api.dataset_info(repo_id=repo_id, revision=revision, token=token)
            else:
                info = api.model_info(repo_id=repo_id, revision=revision, token=token)
        except Exception:
            failed = True
        if failed:
            raise ManagedTrainingArtifactProviderError(
                f"The Hugging Face {repo_type} revision could not be resolved."
            ) from None
        returned_repo = getattr(info, "id", None) or getattr(info, "repo_id", None)
        returned_sha = getattr(info, "sha", None)
        if (
            returned_repo != repo_id
            or not isinstance(returned_sha, str)
            or re.fullmatch(r"[0-9a-fA-F]{40}", returned_sha) is None
            or (
                revision is not None
                and re.fullmatch(r"[0-9a-fA-F]{40}", revision) is not None
                and returned_sha.casefold() != revision.casefold()
            )
        ):
            raise ManagedTrainingArtifactProviderError(
                f"Hugging Face returned an invalid {repo_type} revision identity."
            )
        return returned_sha.casefold()

    @staticmethod
    def _verify_output(
        api: Any,
        *,
        token: str,
        job_id: uuid.UUID,
        request_hash: str,
        output_model_repo: str,
        output_private: bool,
        revision: str,
        require_deployable_root: bool,
    ) -> None:
        info_failed = False
        info: Any = None
        try:
            info = api.model_info(
                repo_id=output_model_repo,
                revision=revision,
                token=token,
                files_metadata=False,
            )
        except TypeError:
            try:
                info = api.model_info(
                    repo_id=output_model_repo,
                    revision=revision,
                    token=token,
                )
            except Exception:
                info_failed = True
        except Exception:
            info_failed = True
        returned_repo = getattr(info, "id", None) or getattr(info, "repo_id", None)
        returned_sha = getattr(info, "sha", None)
        returned_private = getattr(info, "private", None)
        if info_failed:
            raise ManagedTrainingArtifactProviderError(
                "The output model repository could not be inspected."
            ) from None
        if (
            returned_repo != output_model_repo
            or not isinstance(returned_sha, str)
            or returned_sha.casefold() != revision
            or not isinstance(returned_private, bool)
            or returned_private is not output_private
        ):
            raise ManagedTrainingArtifactVerificationError(
                "The immutable output model repository identity is invalid."
            ) from None
        marker_failed = False
        marker_path: Any = None
        try:
            marker_path = api.hf_hub_download(
                repo_id=output_model_repo,
                filename=MANAGED_TRAINING_MARKER_PATH,
                repo_type="model",
                revision=revision,
                token=token,
            )
        except Exception:
            marker_failed = True
        marker_bytes: bytes | None = None
        if not marker_failed:
            try:
                path = Path(marker_path)
                if path.is_file() and path.stat().st_size <= 4096:
                    marker_bytes = path.read_bytes()
            except (OSError, TypeError, ValueError):
                marker_bytes = None
        if marker_failed:
            raise ManagedTrainingArtifactProviderError(
                "The output model ownership marker could not be fetched."
            ) from None
        if marker_bytes != managed_training_marker(job_id, request_hash):
            raise ManagedTrainingArtifactVerificationError(
                "The immutable output model ownership marker is invalid."
            ) from None
        if require_deployable_root:
            siblings = getattr(info, "siblings", None)
            if not isinstance(siblings, (list, tuple)) or len(siblings) > 10_000:
                raise ManagedTrainingArtifactVerificationError(
                    "The immutable output model file manifest is invalid."
                )
            files = {
                value
                for item in siblings
                if isinstance(
                    (value := (
                        item.get("rfilename")
                        if isinstance(item, dict)
                        else getattr(item, "rfilename", None)
                    )),
                    str,
                )
            }
            required_policy_files = {
                "config.json",
                "model.safetensors",
                "policy_preprocessor.json",
                "policy_postprocessor.json",
                *(
                    f"{MANAGED_SMOLVLA_DEPENDENCY_DIR}/{name}"
                    for name in MANAGED_SMOLVLA_DEPENDENCY_FILES
                ),
            }
            if not required_policy_files.issubset(files):
                raise ManagedTrainingArtifactVerificationError(
                    "The immutable output model lacks a deployable root checkpoint."
                )


def _stub_sha(kind: str, repo_id: str, revision: str | None) -> str:
    if revision is not None and _SHA.fullmatch(revision.casefold()) is not None:
        return revision.casefold()
    return hashlib.sha1(f"ctrl-pi-stub:{kind}:{repo_id}:{revision or 'main'}".encode()).hexdigest()


__all__ = [
    "HFManagedTrainingArtifactService",
    "ManagedTrainingArtifactConfigurationError",
    "ManagedTrainingArtifactConflictError",
    "ManagedTrainingArtifactError",
    "ManagedTrainingArtifactProviderError",
    "ManagedTrainingArtifactVerificationError",
    "ManagedTrainingArtifactService",
    "PreparedTrainingArtifacts",
    "StubManagedTrainingArtifactService",
]
