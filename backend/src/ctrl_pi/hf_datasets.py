from __future__ import annotations

import base64
import hashlib
import json
import math
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable


class DatasetAuthenticationError(RuntimeError):
    """The configured Hub token or namespace is not usable."""


class DatasetHubError(RuntimeError):
    """A safe dataset-discovery failure without credential details."""


class DatasetCursorError(ValueError):
    """The opaque pagination cursor cannot be used for this namespace."""


@dataclass(frozen=True)
class DatasetCardSummary:
    title: str | None
    description: str | None
    license: str | None
    task_categories: list[str]


@dataclass(frozen=True)
class LeRobotDatasetSummary:
    codebase_version: str | None
    robot_type: str | None
    fps: float | None
    total_episodes: int | None
    total_frames: int | None
    total_tasks: int | None
    features: list[str]


@dataclass(frozen=True)
class DatasetSummary:
    repo_id: str
    name: str
    revision: str | None
    hub_url: str
    private: bool
    gated: bool
    created_at: datetime | None
    last_modified: datetime | None
    tags: list[str]
    card: DatasetCardSummary | None
    lerobot: LeRobotDatasetSummary | None


@dataclass(frozen=True)
class DatasetPage:
    namespace: str
    datasets: list[DatasetSummary]
    total: int
    next_cursor: str | None
    fetched_at: datetime


@dataclass(frozen=True)
class _DatasetDescriptor:
    repo_id: str
    revision: str | None
    private: bool
    gated: bool
    created_at: datetime | None
    last_modified: datetime | None
    tags: list[str]


@dataclass(frozen=True)
class _RevisionMetadata:
    card: DatasetCardSummary | None
    lerobot: LeRobotDatasetSummary | None


@dataclass(frozen=True)
class _EnumerationCacheEntry:
    expires_at: float
    fetched_at: datetime
    datasets: list[_DatasetDescriptor]


@dataclass(frozen=True)
class _RevisionCacheEntry:
    expires_at: float
    metadata: _RevisionMetadata


class HFDatasetBrowser:
    """Discovers one configured namespace without exposing Hub credentials."""

    _EXPAND = [
        "author",
        "cardData",
        "createdAt",
        "description",
        "gated",
        "lastModified",
        "private",
        "sha",
        "tags",
    ]

    def __init__(
        self,
        *,
        hub_api_factory: Callable[[str], Any] | None = None,
        enumeration_ttl_seconds: float = 30.0,
        revision_ttl_seconds: float = 60.0,
        monotonic: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._hub_api_factory_override = hub_api_factory
        self._enumeration_ttl_seconds = enumeration_ttl_seconds
        self._revision_ttl_seconds = revision_ttl_seconds
        self._monotonic = monotonic
        self._now = now or (lambda: datetime.now(UTC))
        self._enumerations: dict[
            tuple[str, bytes], _EnumerationCacheEntry
        ] = {}
        self._revisions: dict[
            tuple[bytes, str, str], _RevisionCacheEntry
        ] = {}
        self._cache_lock = threading.Lock()

    def list_namespace(
        self,
        *,
        namespace: str,
        token: str,
        limit: int,
        cursor: str | None,
        refresh: bool,
    ) -> DatasetPage:
        cursor_key = self._decode_cursor(cursor, namespace) if cursor else None
        token_key = hashlib.sha256(token.encode("utf-8")).digest()
        api = self._hub_api(token)
        descriptors, fetched_at = self._enumerate(
            api,
            namespace=namespace,
            token=token,
            token_key=token_key,
            refresh=refresh,
        )
        remaining = [
            descriptor
            for descriptor in descriptors
            if cursor_key is None or self._sort_key(descriptor) > cursor_key
        ]
        selected = remaining[:limit]
        next_cursor = (
            self._encode_cursor(namespace, selected[-1])
            if selected and len(remaining) > limit
            else None
        )
        datasets = [
            self._hydrate(
                api,
                descriptor=descriptor,
                token=token,
                token_key=token_key,
                refresh=refresh,
            )
            for descriptor in selected
        ]
        return DatasetPage(
            namespace=namespace,
            datasets=datasets,
            total=len(descriptors),
            next_cursor=next_cursor,
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
    ) -> tuple[list[_DatasetDescriptor], datetime]:
        cache_key = (namespace, token_key)
        now_monotonic = self._monotonic()
        if not refresh:
            with self._cache_lock:
                cached = self._enumerations.get(cache_key)
                if cached is not None and cached.expires_at > now_monotonic:
                    return cached.datasets, cached.fetched_at

        self._validate_namespace(api, namespace, token)
        try:
            hub_items = list(
                api.list_datasets(
                    author=namespace,
                    filter="LeRobot",
                    sort="last_modified",
                    direction=-1,
                    expand=self._EXPAND,
                    token=token,
                )
            )
        except Exception as error:
            self._raise_hub_error(error, "Hugging Face dataset discovery failed.")

        by_repo: dict[str, _DatasetDescriptor] = {}
        for item in hub_items:
            descriptor = self._descriptor(item, namespace)
            if descriptor is not None and descriptor.repo_id not in by_repo:
                by_repo[descriptor.repo_id] = descriptor
        descriptors = sorted(by_repo.values(), key=self._sort_key)
        fetched_at = self._as_utc(self._now())
        entry = _EnumerationCacheEntry(
            expires_at=now_monotonic + self._enumeration_ttl_seconds,
            fetched_at=fetched_at,
            datasets=descriptors,
        )
        with self._cache_lock:
            self._prune_caches(now_monotonic)
            self._enumerations[cache_key] = entry
        return descriptors, fetched_at

    def _hydrate(
        self,
        api: Any,
        *,
        descriptor: _DatasetDescriptor,
        token: str,
        token_key: bytes,
        refresh: bool,
    ) -> DatasetSummary:
        metadata = _RevisionMetadata(card=None, lerobot=None)
        if descriptor.revision:
            cache_key = (token_key, descriptor.repo_id, descriptor.revision)
            now_monotonic = self._monotonic()
            cached: _RevisionCacheEntry | None = None
            if not refresh:
                with self._cache_lock:
                    cached = self._revisions.get(cache_key)
            if cached is not None and cached.expires_at > now_monotonic:
                metadata = cached.metadata
            else:
                metadata = self._fetch_revision_metadata(
                    api,
                    repo_id=descriptor.repo_id,
                    revision=descriptor.revision,
                    token=token,
                )
                with self._cache_lock:
                    self._prune_caches(now_monotonic)
                    if len(self._revisions) >= 512:
                        oldest_key = min(
                            self._revisions,
                            key=lambda key: self._revisions[key].expires_at,
                        )
                        self._revisions.pop(oldest_key, None)
                    self._revisions[cache_key] = _RevisionCacheEntry(
                        expires_at=now_monotonic + self._revision_ttl_seconds,
                        metadata=metadata,
                    )
        return DatasetSummary(
            repo_id=descriptor.repo_id,
            name=descriptor.repo_id.split("/", 1)[1],
            revision=descriptor.revision,
            hub_url=f"https://huggingface.co/datasets/{descriptor.repo_id}",
            private=descriptor.private,
            gated=descriptor.gated,
            created_at=descriptor.created_at,
            last_modified=descriptor.last_modified,
            tags=descriptor.tags,
            card=metadata.card,
            lerobot=metadata.lerobot,
        )

    def _fetch_revision_metadata(
        self,
        api: Any,
        *,
        repo_id: str,
        revision: str,
        token: str,
    ) -> _RevisionMetadata:
        readme = self._download_optional(
            api,
            repo_id=repo_id,
            filename="README.md",
            revision=revision,
            token=token,
        )
        info = self._download_optional(
            api,
            repo_id=repo_id,
            filename="meta/info.json",
            revision=revision,
            token=token,
        )
        card = self._parse_card(readme) if readme is not None else None
        lerobot = self._parse_lerobot_info(info) if info is not None else None
        return _RevisionMetadata(card=card, lerobot=lerobot)

    @staticmethod
    def _download_optional(
        api: Any,
        *,
        repo_id: str,
        filename: str,
        revision: str,
        token: str,
    ) -> Path | None:
        try:
            path = api.hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                revision=revision,
                token=token,
            )
            resolved = Path(path)
            return resolved if resolved.is_file() else None
        except Exception:
            return None

    @staticmethod
    def _parse_card(path: Path) -> DatasetCardSummary | None:
        try:
            from huggingface_hub import DatasetCard

            card = DatasetCard.load(path, ignore_metadata_errors=True)
            data = card.data.to_dict()
            text = card.text
        except Exception:
            return None
        title = HFDatasetBrowser._optional_text(data.get("pretty_name"))
        if title is None:
            title = HFDatasetBrowser._first_heading(text)
        license_name = HFDatasetBrowser._text_or_joined(data.get("license"))
        tasks = HFDatasetBrowser._string_list(data.get("task_categories"))
        return DatasetCardSummary(
            title=title,
            description=(
                HFDatasetBrowser._section_paragraph(text, "Dataset Description")
                or HFDatasetBrowser._first_paragraph(text)
            ),
            license=license_name,
            task_categories=tasks,
        )

    @staticmethod
    def _parse_lerobot_info(path: Path) -> LeRobotDatasetSummary | None:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(raw, dict):
            return None
        features = raw.get("features")
        return LeRobotDatasetSummary(
            codebase_version=HFDatasetBrowser._optional_text(
                raw.get("codebase_version")
            ),
            robot_type=HFDatasetBrowser._optional_text(raw.get("robot_type")),
            fps=HFDatasetBrowser._positive_number(raw.get("fps")),
            total_episodes=HFDatasetBrowser._nonnegative_int(
                raw.get("total_episodes")
            ),
            total_frames=HFDatasetBrowser._nonnegative_int(raw.get("total_frames")),
            total_tasks=HFDatasetBrowser._nonnegative_int(raw.get("total_tasks")),
            features=(
                sorted(key for key in features if isinstance(key, str))
                if isinstance(features, dict)
                else []
            ),
        )

    @staticmethod
    def _descriptor(item: Any, namespace: str) -> _DatasetDescriptor | None:
        repo_id = HFDatasetBrowser._optional_text(
            getattr(item, "id", None) or getattr(item, "repo_id", None)
        )
        if repo_id is None or "/" not in repo_id:
            return None
        owner, _ = repo_id.split("/", 1)
        if owner != namespace:
            return None
        tags = HFDatasetBrowser._string_list(getattr(item, "tags", None))
        if not any(tag.casefold() == "lerobot" for tag in tags):
            return None
        return _DatasetDescriptor(
            repo_id=repo_id,
            revision=HFDatasetBrowser._optional_text(getattr(item, "sha", None)),
            private=getattr(item, "private", None) is not False,
            gated=getattr(item, "gated", False) not in (None, False),
            created_at=HFDatasetBrowser._optional_datetime(
                getattr(item, "created_at", None)
            ),
            last_modified=HFDatasetBrowser._optional_datetime(
                getattr(item, "last_modified", None)
            ),
            tags=sorted(set(tags), key=str.casefold),
        )

    @staticmethod
    def _sort_key(descriptor: _DatasetDescriptor) -> tuple[int, str]:
        timestamp = HFDatasetBrowser._timestamp_microseconds(
            descriptor.last_modified
        )
        return -timestamp, descriptor.repo_id

    @staticmethod
    def _encode_cursor(namespace: str, descriptor: _DatasetDescriptor) -> str:
        payload = {
            "v": 1,
            "n": namespace,
            "t": HFDatasetBrowser._timestamp_microseconds(
                descriptor.last_modified
            ),
            "r": descriptor.repo_id,
        }
        raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    @staticmethod
    def _decode_cursor(cursor: str, namespace: str) -> tuple[int, str]:
        try:
            if not cursor or len(cursor) > 1024:
                raise ValueError
            padding = "=" * (-len(cursor) % 4)
            payload = json.loads(
                base64.b64decode(
                    cursor + padding,
                    altchars=b"-_",
                    validate=True,
                ).decode("utf-8")
            )
            if (
                not isinstance(payload, dict)
                or set(payload) != {"v", "n", "t", "r"}
                or payload.get("v") != 1
                or payload.get("n") != namespace
                or type(payload.get("t")) is not int
                or not isinstance(payload.get("r"), str)
                or not payload["r"].startswith(f"{namespace}/")
            ):
                raise ValueError
            return -payload["t"], payload["r"]
        except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise DatasetCursorError("Dataset cursor is invalid or expired.") from error

    def _validate_namespace(self, api: Any, namespace: str, token: str) -> None:
        try:
            identity = api.whoami(token=token)
        except Exception as error:
            self._raise_hub_error(error, "Hugging Face authentication failed.")
        available = {str(identity.get("name", "")).casefold()}
        for organization in identity.get("orgs", []):
            name = (
                organization.get("name")
                if isinstance(organization, dict)
                else organization
            )
            if name:
                available.add(str(name).casefold())
        if namespace.casefold() not in available:
            raise DatasetAuthenticationError(
                "HF_NAMESPACE is not the authenticated user or one of its organizations."
            )

    @staticmethod
    def _raise_hub_error(error: Exception, message: str) -> None:
        status_code = getattr(getattr(error, "response", None), "status_code", None)
        if status_code in {401, 403}:
            raise DatasetAuthenticationError(
                "HF_TOKEN could not be authenticated."
            ) from error
        raise DatasetHubError(message) from error

    def _hub_api(self, token: str) -> Any:
        if self._hub_api_factory_override is not None:
            return self._hub_api_factory_override(token)
        from huggingface_hub import HfApi

        return HfApi(token=token)

    def _prune_caches(self, now_monotonic: float) -> None:
        self._enumerations = {
            key: value
            for key, value in self._enumerations.items()
            if value.expires_at > now_monotonic
        }
        self._revisions = {
            key: value
            for key, value in self._revisions.items()
            if value.expires_at > now_monotonic
        }

    @staticmethod
    def _timestamp_microseconds(value: datetime | None) -> int:
        if value is None:
            return -(2**63)
        return int(value.timestamp() * 1_000_000)

    @staticmethod
    def _optional_datetime(value: Any) -> datetime | None:
        if isinstance(value, datetime):
            return HFDatasetBrowser._as_utc(value)
        if isinstance(value, str):
            try:
                return HFDatasetBrowser._as_utc(
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
    def _text_or_joined(value: Any) -> str | None:
        text = HFDatasetBrowser._optional_text(value)
        if text is not None:
            return text
        values = HFDatasetBrowser._string_list(value)
        return ", ".join(values) if values else None

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple, set)):
            return []
        result: list[str] = []
        for item in value:
            text = HFDatasetBrowser._optional_text(item)
            if text is not None and text not in result:
                result.append(text)
        return result

    @staticmethod
    def _first_heading(text: str) -> str | None:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                return HFDatasetBrowser._optional_text(stripped[2:])
        return None

    @staticmethod
    def _first_paragraph(text: str) -> str | None:
        paragraphs = text.split("\n\n")
        for paragraph in paragraphs:
            lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
            if not lines or all(
                line.startswith(("#", "[!", "![", "<", "- ", "* "))
                for line in lines
            ):
                continue
            value = " ".join(lines)
            return value[:500]
        return None

    @staticmethod
    def _section_paragraph(text: str, heading: str) -> str | None:
        section_lines: list[str] = []
        in_section = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                title = stripped.lstrip("#").strip()
                if in_section:
                    break
                in_section = title.casefold() == heading.casefold()
                continue
            if in_section:
                section_lines.append(line)
        return HFDatasetBrowser._first_paragraph("\n".join(section_lines))

    @staticmethod
    def _positive_number(value: Any) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        number = float(value)
        return number if math.isfinite(number) and number > 0 else None

    @staticmethod
    def _nonnegative_int(value: Any) -> int | None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        return value
