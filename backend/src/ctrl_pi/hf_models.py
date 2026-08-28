from __future__ import annotations

import hashlib
import re
import threading
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable


class ModelAuthenticationError(RuntimeError):
    """The configured Hub token or namespace cannot be used."""


class ModelHubError(RuntimeError):
    """A sanitized Hugging Face model discovery failure."""


@dataclass(frozen=True)
class ModelCardSummary:
    description: str | None
    base_model: list[str]
    datasets: list[str]


@dataclass(frozen=True)
class ModelSummary:
    repo_id: str
    name: str
    revision: str | None
    hub_url: str
    private: bool
    gated: bool
    last_modified: datetime | None
    pipeline_tag: str | None
    library_name: str | None
    tags: list[str]
    card: ModelCardSummary | None
    checkpoints: list[str]


@dataclass(frozen=True)
class ModelPage:
    namespace: str
    models: list[ModelSummary]
    total: int
    fetched_at: datetime


@dataclass(frozen=True)
class _ModelDescriptor:
    repo_id: str
    revision: str | None
    private: bool
    gated: bool
    last_modified: datetime | None
    pipeline_tag: str | None
    library_name: str | None
    tags: list[str]
    checkpoints: list[str]


@dataclass(frozen=True)
class _EnumerationCacheEntry:
    expires_at: float
    fetched_at: datetime
    models: list[_ModelDescriptor]


@dataclass(frozen=True)
class _CardCacheEntry:
    expires_at: float
    card: ModelCardSummary | None


class HFModelBrowser:
    """Discovers model repositories in exactly one configured namespace."""

    _EXPAND = [
        "author",
        "gated",
        "lastModified",
        "library_name",
        "pipeline_tag",
        "private",
        "sha",
        "siblings",
        "tags",
    ]
    _CHECKPOINT = re.compile(
        r"(?:^|/)(?:checkpoints?)(?:[-_/]|$)|"
        r"(?:^|/)(?:model|pytorch_model)(?:[-_.].*)?\.(?:safetensors|bin|pt)$",
        re.IGNORECASE,
    )

    def __init__(
        self,
        *,
        hub_api_factory: Callable[[str], Any] | None = None,
        enumeration_ttl_seconds: float = 30.0,
        card_ttl_seconds: float = 60.0,
        monotonic: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._hub_api_factory_override = hub_api_factory
        self._enumeration_ttl_seconds = enumeration_ttl_seconds
        self._card_ttl_seconds = card_ttl_seconds
        self._monotonic = monotonic
        self._now = now or (lambda: datetime.now(UTC))
        self._enumerations: dict[
            tuple[str, bytes], _EnumerationCacheEntry
        ] = {}
        self._cards: dict[tuple[bytes, str, str], _CardCacheEntry] = {}
        self._cache_lock = threading.Lock()

    def list_namespace(
        self,
        *,
        namespace: str,
        token: str,
        refresh: bool = False,
    ) -> ModelPage:
        token_key = hashlib.sha256(token.encode("utf-8")).digest()
        api = self._hub_api(token)
        descriptors, fetched_at = self._enumerate(
            api,
            namespace=namespace,
            token=token,
            token_key=token_key,
            refresh=refresh,
        )
        models = [
            self._hydrate(
                api,
                descriptor=descriptor,
                token=token,
                token_key=token_key,
                refresh=refresh,
            )
            for descriptor in descriptors
        ]
        return ModelPage(
            namespace=namespace,
            models=models,
            total=len(models),
            fetched_at=fetched_at,
        )

    def _enumerate(
        self,
        api: Any,
        *,
        namespace: str,
        token: str,
        token_key: bytes,
        refresh: bool,
    ) -> tuple[list[_ModelDescriptor], datetime]:
        cache_key = (namespace, token_key)
        now_monotonic = self._monotonic()
        if not refresh:
            with self._cache_lock:
                cached = self._enumerations.get(cache_key)
            if cached is not None and cached.expires_at > now_monotonic:
                return cached.models, cached.fetched_at

        self._validate_namespace(api, namespace, token)
        try:
            items = list(
                api.list_models(
                    author=namespace,
                    sort="last_modified",
                    direction=-1,
                    limit=1000,
                    expand=self._EXPAND,
                    token=token,
                )
            )
        except Exception as error:
            self._raise_hub_error(error, "Hugging Face model discovery failed.")

        by_repo: dict[str, _ModelDescriptor] = {}
        for item in items:
            descriptor = self._descriptor(item, namespace)
            if descriptor is not None and descriptor.repo_id not in by_repo:
                by_repo[descriptor.repo_id] = self._pin_descriptor(
                    api, descriptor=descriptor, token=token
                )
        descriptors = sorted(by_repo.values(), key=self._sort_key)
        fetched_at = self._as_utc(self._now())
        with self._cache_lock:
            self._prune(now_monotonic)
            self._enumerations[cache_key] = _EnumerationCacheEntry(
                expires_at=now_monotonic + self._enumeration_ttl_seconds,
                fetched_at=fetched_at,
                models=descriptors,
            )
        return descriptors, fetched_at

    def _hydrate(
        self,
        api: Any,
        *,
        descriptor: _ModelDescriptor,
        token: str,
        token_key: bytes,
        refresh: bool,
    ) -> ModelSummary:
        card: ModelCardSummary | None = None
        if descriptor.revision is not None:
            cache_key = (token_key, descriptor.repo_id, descriptor.revision)
            now_monotonic = self._monotonic()
            cached: _CardCacheEntry | None = None
            if not refresh:
                with self._cache_lock:
                    cached = self._cards.get(cache_key)
            if cached is not None and cached.expires_at > now_monotonic:
                card = cached.card
            else:
                card = self._fetch_card(
                    api,
                    repo_id=descriptor.repo_id,
                    revision=descriptor.revision,
                    token=token,
                )
                with self._cache_lock:
                    self._prune(now_monotonic)
                    if len(self._cards) >= 512:
                        oldest = min(
                            self._cards,
                            key=lambda key: self._cards[key].expires_at,
                        )
                        self._cards.pop(oldest, None)
                    self._cards[cache_key] = _CardCacheEntry(
                        expires_at=now_monotonic + self._card_ttl_seconds,
                        card=card,
                    )
        return ModelSummary(
            repo_id=descriptor.repo_id,
            name=descriptor.repo_id.split("/", 1)[1],
            revision=descriptor.revision,
            hub_url=f"https://huggingface.co/{descriptor.repo_id}",
            private=descriptor.private,
            gated=descriptor.gated,
            last_modified=descriptor.last_modified,
            pipeline_tag=descriptor.pipeline_tag,
            library_name=descriptor.library_name,
            tags=descriptor.tags,
            card=card,
            checkpoints=descriptor.checkpoints,
        )

    @staticmethod
    def _descriptor(item: Any, namespace: str) -> _ModelDescriptor | None:
        repo_id = HFModelBrowser._optional_text(
            getattr(item, "id", None) or getattr(item, "repo_id", None)
        )
        if repo_id is None or repo_id.count("/") != 1:
            return None
        owner, name = repo_id.split("/", 1)
        if owner != namespace or not name:
            return None
        tags = sorted(
            set(HFModelBrowser._string_list(getattr(item, "tags", None))),
            key=str.casefold,
        )
        return _ModelDescriptor(
            repo_id=repo_id,
            revision=HFModelBrowser._optional_text(getattr(item, "sha", None)),
            private=getattr(item, "private", None) is not False,
            gated=getattr(item, "gated", False) not in (None, False),
            last_modified=HFModelBrowser._optional_datetime(
                getattr(item, "last_modified", None)
            ),
            pipeline_tag=HFModelBrowser._optional_text(
                getattr(item, "pipeline_tag", None)
            ),
            library_name=HFModelBrowser._optional_text(
                getattr(item, "library_name", None)
            ),
            tags=tags,
            checkpoints=HFModelBrowser._checkpoint_paths(
                getattr(item, "siblings", None)
            ),
        )

    @classmethod
    def _pin_descriptor(
        cls,
        api: Any,
        *,
        descriptor: _ModelDescriptor,
        token: str,
    ) -> _ModelDescriptor:
        listed_revision = (
            descriptor.revision
            if descriptor.revision is not None
            and re.fullmatch(r"[0-9a-fA-F]{40}", descriptor.revision)
            else None
        )
        try:
            arguments: dict[str, Any] = {
                "repo_id": descriptor.repo_id,
                "expand": ["sha", "siblings"],
                "token": token,
            }
            if listed_revision is not None:
                arguments["revision"] = listed_revision
            info = api.model_info(
                **arguments,
            )
        except Exception:
            return replace(descriptor, revision=None, checkpoints=[])
        returned_id = cls._optional_text(
            getattr(info, "id", None) or getattr(info, "repo_id", None)
        )
        revision = cls._optional_text(getattr(info, "sha", None))
        if (
            returned_id != descriptor.repo_id
            or revision is None
            or not re.fullmatch(r"[0-9a-fA-F]{40}", revision)
            or (
                listed_revision is not None
                and revision.casefold() != listed_revision.casefold()
            )
        ):
            return replace(descriptor, revision=None, checkpoints=[])
        pinned_checkpoints = cls._checkpoint_paths(getattr(info, "siblings", None))
        checkpoint_candidates = (
            set(descriptor.checkpoints).union(pinned_checkpoints)
            if listed_revision is not None
            else set(pinned_checkpoints)
        )
        checkpoints = sorted(checkpoint_candidates, key=str.casefold)[:200]
        return replace(
            descriptor,
            revision=revision.casefold(),
            checkpoints=checkpoints,
        )

    @staticmethod
    def _checkpoint_paths(value: Any) -> list[str]:
        if not isinstance(value, (list, tuple)):
            return []
        paths: set[str] = set()
        for sibling in value:
            path = (
                sibling.get("rfilename")
                if isinstance(sibling, dict)
                else getattr(sibling, "rfilename", None)
            )
            if (
                isinstance(path, str)
                and len(path) <= 512
                and HFModelBrowser._safe_repo_path(path)
                and HFModelBrowser._CHECKPOINT.search(path)
            ):
                paths.add(path)
            if len(paths) >= 200:
                break
        return sorted(paths, key=str.casefold)

    @staticmethod
    def _safe_repo_path(value: str) -> bool:
        path = Path(value)
        return (
            bool(value)
            and not path.is_absolute()
            and "\\" not in value
            and all(part not in {"", ".", ".."} for part in path.parts)
        )

    def _fetch_card(
        self,
        api: Any,
        *,
        repo_id: str,
        revision: str,
        token: str,
    ) -> ModelCardSummary | None:
        try:
            downloaded = api.hf_hub_download(
                repo_id=repo_id,
                filename="README.md",
                repo_type="model",
                revision=revision,
                token=token,
            )
            path = Path(downloaded)
            if not path.is_file():
                return None
            from huggingface_hub import ModelCard

            card = ModelCard.load(path, ignore_metadata_errors=True)
            data = card.data.to_dict()
            return ModelCardSummary(
                description=self._first_paragraph(card.text),
                base_model=self._string_list(data.get("base_model")),
                datasets=self._string_list(data.get("datasets")),
            )
        except Exception:
            return None

    @staticmethod
    def _validate_namespace(api: Any, namespace: str, token: str) -> None:
        try:
            identity = api.whoami(token=token)
        except Exception as error:
            HFModelBrowser._raise_hub_error(
                error, "Hugging Face authentication failed."
            )
        if not isinstance(identity, dict):
            raise ModelHubError("Hugging Face returned an invalid identity response.")
        organizations = identity.get("orgs", [])
        if not isinstance(organizations, (list, tuple)):
            raise ModelHubError("Hugging Face returned an invalid identity response.")
        available = {str(identity.get("name", "")).casefold()}
        for organization in organizations:
            name = (
                organization.get("name")
                if isinstance(organization, dict)
                else organization
            )
            if name:
                available.add(str(name).casefold())
        if namespace.casefold() not in available:
            raise ModelAuthenticationError(
                "HF_NAMESPACE is not the authenticated user or one of its organizations."
            )

    @staticmethod
    def _raise_hub_error(error: Exception, message: str) -> None:
        status_code = getattr(getattr(error, "response", None), "status_code", None)
        if status_code in {401, 403}:
            raise ModelAuthenticationError(
                "HF_TOKEN could not be authenticated."
            ) from error
        raise ModelHubError(message) from error

    def _hub_api(self, token: str) -> Any:
        if self._hub_api_factory_override is not None:
            return self._hub_api_factory_override(token)
        from huggingface_hub import HfApi

        return HfApi(token=token)

    def _prune(self, now_monotonic: float) -> None:
        self._enumerations = {
            key: value
            for key, value in self._enumerations.items()
            if value.expires_at > now_monotonic
        }
        self._cards = {
            key: value
            for key, value in self._cards.items()
            if value.expires_at > now_monotonic
        }

    @staticmethod
    def _sort_key(descriptor: _ModelDescriptor) -> tuple[int, str]:
        timestamp = (
            int(descriptor.last_modified.timestamp() * 1_000_000)
            if descriptor.last_modified is not None
            else -(2**63)
        )
        return -timestamp, descriptor.repo_id

    @staticmethod
    def _optional_datetime(value: Any) -> datetime | None:
        if isinstance(value, datetime):
            return HFModelBrowser._as_utc(value)
        if isinstance(value, str):
            try:
                return HFModelBrowser._as_utc(
                    datetime.fromisoformat(value.replace("Z", "+00:00"))
                )
            except ValueError:
                return None
        return None

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @staticmethod
    def _optional_text(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        value = value.strip()
        return value or None

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple, set)):
            return []
        result: list[str] = []
        for item in value:
            text = HFModelBrowser._optional_text(item)
            if text is not None and text not in result:
                result.append(text)
        return result[:100]

    @staticmethod
    def _first_paragraph(text: str) -> str | None:
        for paragraph in text.split("\n\n"):
            lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
            if not lines or all(
                line.startswith(("#", "[!", "![", "<", "- ", "* "))
                for line in lines
            ):
                continue
            return " ".join(lines)[:500]
        return None
