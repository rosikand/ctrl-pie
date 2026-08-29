from __future__ import annotations

import copy
import json
import math
import re
import unicodedata
from datetime import UTC, datetime
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from ctrl_pi.models import ManagedTrainingJob, TrainingRun

MAX_STEP = 2_147_483_647
MAX_METRIC_NAMES = 128
MAX_METRIC_POINTS = 10_000
MAX_CHECKPOINTS = 512
MAX_CONSOLE_LOGS = 1_000
MAX_CONSOLE_LOG_BYTES = 512 * 1024
MAX_CONSOLE_LINE_BYTES = 4 * 1024
MAX_CONSOLE_SEQUENCE = 9_007_199_254_740_991

ConsoleSource = Literal["stdout", "stderr", "system"]
_METRIC_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_.\-/]{0,63})")
_REVISION = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._/-]{0,126}[A-Za-z0-9])?")
_SECRET_TOKEN = re.compile(
    r"(?i)(?:\bhf_[a-z0-9]{8,}\b|\b(?:ak|as|wk|ws)-[a-z0-9_-]{8,}\b|"
    r"://[^/\s:@]+:[^/\s@]+@|\bbearer\s+\S+)"
)
_ASSIGNMENT = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?P<key>[A-Za-z][A-Za-z0-9_.-]{0,127})"
    r"\s*[:=](?=\s*\S+)"
)
_SENSITIVE_PARTS = {
    "authorization",
    "cookie",
    "credential",
    "dsn",
    "passwd",
    "password",
    "secret",
}
_SENSITIVE_SUFFIXES = {
    "accesstoken",
    "apikey",
    "databaseurl",
    "dburl",
    "hftoken",
    "modaltoken",
    "privatekey",
}


class TrainingStoreError(RuntimeError):
    pass


class TrainingStoreLimitError(TrainingStoreError):
    pass


class TrainingStoreCorruptError(TrainingStoreError):
    pass


class ManagedTrainingRunMutationError(TrainingStoreError):
    pass


def assert_external_run_mutable(db: Session, run_id: object) -> None:
    managed = db.scalar(
        select(ManagedTrainingJob.id).where(ManagedTrainingJob.training_run_id == run_id)
    )
    if managed is not None:
        raise ManagedTrainingRunMutationError(
            "Managed training runs are updated only by their managed job."
        )


def append_metrics(run: TrainingRun, *, step: int, metrics: dict[str, float]) -> None:
    if isinstance(step, bool) or not 0 <= step <= MAX_STEP:
        raise ValueError("training metric step is invalid")
    if not 1 <= len(metrics) <= 64:
        raise ValueError("training metrics are invalid")
    stored = copy.deepcopy(run.metrics or {})
    for name, raw_value in metrics.items():
        if (
            not isinstance(name, str)
            or _METRIC_NAME.fullmatch(name) is None
            or isinstance(raw_value, bool)
            or not isinstance(raw_value, (int, float))
            or not math.isfinite(raw_value)
        ):
            raise ValueError("training metrics are invalid")
        value = float(raw_value)
        existing = stored.get(name, [])
        if not isinstance(existing, list):
            raise TrainingStoreCorruptError("Stored training metrics are invalid.")
        series = [point for point in existing if point.get("step") != step]
        series.append({"step": step, "value": value})
        try:
            series.sort(key=lambda point: point["step"])
        except (KeyError, TypeError, ValueError):
            raise TrainingStoreCorruptError(
                "Stored training metrics are invalid."
            ) from None
        stored[name] = series
    if len(stored) > MAX_METRIC_NAMES or sum(map(len, stored.values())) > MAX_METRIC_POINTS:
        raise TrainingStoreLimitError(
            "Training run metric storage limit was reached."
        )
    run.metrics = stored
    run.current_step = max(run.current_step, step)


def append_checkpoint(
    run: TrainingRun,
    *,
    repo_id: str,
    revision: str,
    step: int,
) -> None:
    if (
        not isinstance(repo_id, str)
        or not 3 <= len(repo_id) <= 255
        or repo_id.count("/") != 1
        or not isinstance(revision, str)
        or _REVISION.fullmatch(revision) is None
        or ".." in revision
        or "//" in revision
        or "--" in revision
        or isinstance(step, bool)
        or not 0 <= step <= MAX_STEP
    ):
        raise ValueError("training checkpoint is invalid")
    checkpoints = copy.deepcopy(run.checkpoints or [])
    if not isinstance(checkpoints, list):
        raise TrainingStoreCorruptError("Stored training checkpoints are invalid.")
    checkpoint = {"repo_id": repo_id, "revision": revision, "step": step}
    if checkpoint not in checkpoints:
        if len(checkpoints) >= MAX_CHECKPOINTS:
            raise TrainingStoreLimitError(
                "Training run checkpoint storage limit was reached."
            )
        checkpoints.append(checkpoint)
        try:
            checkpoints.sort(
                key=lambda item: (item["step"], item["repo_id"], item["revision"])
            )
        except (KeyError, TypeError, ValueError):
            raise TrainingStoreCorruptError(
                "Stored training checkpoints are invalid."
            ) from None
    run.checkpoints = checkpoints
    run.output_model_repo = repo_id
    run.checkpoint_revision = revision
    run.current_step = max(run.current_step, step)


def append_console_log(
    run: TrainingRun,
    *,
    source: ConsoleSource,
    line: str,
    step: int | None,
    timestamp: datetime | None = None,
    max_logs: int = MAX_CONSOLE_LOGS,
    max_bytes: int = MAX_CONSOLE_LOG_BYTES,
) -> dict[str, Any]:
    validate_console_line(source=source, line=line, step=step)
    logs = copy.deepcopy(run.console_logs or [])
    if not isinstance(logs, list):
        raise TrainingStoreCorruptError("Stored training console metadata is invalid.")
    last_sequence = 0
    for entry in logs:
        sequence = entry.get("sequence") if isinstance(entry, dict) else None
        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence <= last_sequence
            or sequence > MAX_CONSOLE_SEQUENCE
        ):
            raise TrainingStoreCorruptError(
                "Stored training console metadata is invalid."
            )
        last_sequence = sequence
    if last_sequence >= MAX_CONSOLE_SEQUENCE:
        raise TrainingStoreLimitError(
            "Training run console sequence limit was reached."
        )
    at = timestamp or datetime.now(UTC)
    if at.tzinfo is None or at.utcoffset() is None:
        raise ValueError("training console timestamp must include a timezone")
    entry = {
        "sequence": last_sequence + 1,
        "source": source,
        "line": line,
        "step": step,
        "timestamp": at.astimezone(UTC).isoformat(),
    }
    logs.append(entry)
    if not 1 <= max_logs <= MAX_CONSOLE_LOGS or not 1 <= max_bytes <= MAX_CONSOLE_LOG_BYTES:
        raise ValueError("training console retention bounds are invalid")
    while len(logs) > max_logs:
        logs.pop(0)
    try:
        while (
            len(
                json.dumps(
                    logs,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            > max_bytes
        ):
            logs.pop(0)
    except (AttributeError, TypeError, ValueError):
        raise TrainingStoreCorruptError(
            "Stored training console metadata is invalid."
        ) from None
    run.console_logs = logs
    return entry


def validate_console_line(
    *, source: ConsoleSource, line: str, step: int | None
) -> None:
    if source not in {"stdout", "stderr", "system"}:
        raise ValueError("training console source is invalid")
    if (
        not isinstance(line, str)
        or not line.strip()
        or len(line.encode("utf-8")) > MAX_CONSOLE_LINE_BYTES
        or any(
            character != "\t" and unicodedata.category(character).startswith("C")
            for character in line
        )
        or _contains_secret_assignment(line)
        or _SECRET_TOKEN.search(line)
    ):
        raise ValueError("training console line is invalid")
    if step is not None and (
        isinstance(step, bool) or not isinstance(step, int) or not 0 <= step <= MAX_STEP
    ):
        raise ValueError("training console step is invalid")


def _contains_secret_assignment(value: str) -> bool:
    for match in _ASSIGNMENT.finditer(value):
        key = match.group("key")
        separated = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", key)
        separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", separated)
        parts = {
            part for part in re.split(r"[^a-z0-9]+", separated.casefold()) if part
        }
        normalized = re.sub(r"[^a-z0-9]", "", key.casefold())
        if parts.intersection(_SENSITIVE_PARTS) or any(
            normalized.endswith(suffix) for suffix in _SENSITIVE_SUFFIXES
        ):
            return True
    return False


__all__ = [
    "MAX_CHECKPOINTS",
    "MAX_CONSOLE_LINE_BYTES",
    "MAX_CONSOLE_LOG_BYTES",
    "MAX_CONSOLE_LOGS",
    "MAX_CONSOLE_SEQUENCE",
    "MAX_METRIC_NAMES",
    "MAX_METRIC_POINTS",
    "MAX_STEP",
    "ManagedTrainingRunMutationError",
    "TrainingStoreCorruptError",
    "TrainingStoreError",
    "TrainingStoreLimitError",
    "append_checkpoint",
    "append_console_log",
    "append_metrics",
    "assert_external_run_mutable",
    "validate_console_line",
]
