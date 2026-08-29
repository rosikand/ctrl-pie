from __future__ import annotations

import json
import math
import os
import queue
import re
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Protocol, TextIO, cast

from ctrl_pi.training_compute import (
    MANAGED_SMOLVLA_DEPENDENCY_DIR,
    MANAGED_SMOLVLA_DEPENDENCY_FILES,
    MANAGED_TRAINING_MARKER_PATH,
    ManagedTrainingConfigurationError,
    ManagedTrainingSpec,
    managed_training_marker,
    training_gpu_spec,
)


MODAL_TRAINING_FUNCTION = "train"
MODAL_TRAINING_OWNERSHIP_TAG_KEY = "ctrl-pi-training-job"
MODAL_TRAINING_EVENT_PREFIX = "CTRL_PI_TRAIN_EVENT "
MODAL_TRAINING_EVENT_SCHEMA = "ctrl-pi.modal-training-event"
MODAL_TRAINING_RESULT_SCHEMA = "ctrl-pi.modal-training-result"
MODAL_TRAINING_PROTOCOL_VERSION = 1
MODAL_TRAINING_PACKAGES = (
    "huggingface-hub==0.35.3",
    "lerobot[smolvla]==0.4.4",
)
SMOLVLA_VLM_REPO = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
SMOLVLA_VLM_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
_SMOLVLA_VLM_FILES = MANAGED_SMOLVLA_DEPENDENCY_FILES

MAX_WORKER_EVENT_LINE_BYTES = 8 * 1024
MAX_WORKER_EVENT_BYTES = 8 * 1024 * 1024
MAX_WORKER_LOG_EVENTS = 4_096
MAX_WORKER_EVENTS = 12_000
_EVENT_RESERVE_BYTES = 512 * 1024
_OUTPUT_DIR = Path("/tmp/ctrl-pi-training-output")
_INPUT_DIR = Path("/tmp/ctrl-pi-training-input")
_REQUIRED_POLICY_FILES = {
    "config.json",
    "model.safetensors",
    "policy_postprocessor.json",
    "policy_preprocessor.json",
}
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_METRIC_VALUE = re.compile(
    r"\b(?P<name>[A-Za-z][A-Za-z0-9_.\-/]{0,63})\s*:\s*"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    r"(?P<suffix>[KMBTQ]?)(?![A-Za-z0-9_.])"
)
_TRACKER_SIGNATURE = re.compile(
    r"\bstep:(?P<step>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?[KMBTQ]?)\s+"
    r"smpl:(?P<samples>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?[KMBTQ]?)\s+"
    r"ep:(?P<episodes>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?[KMBTQ]?)\s+"
    r"epch:(?P<epochs>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?[KMBTQ]?)\b"
)
_METRIC_SUFFIX_MULTIPLIERS = {
    "": 1.0,
    "K": 1_000.0,
    "M": 1_000_000.0,
    "B": 1_000_000_000.0,
    "T": 1_000_000_000_000.0,
    "Q": 1_000_000_000_000_000.0,
}
_TOKEN_LIKE = re.compile(
    r"(?i)(?:hf_[A-Za-z0-9]{8,}|(?:bearer\s+)[A-Za-z0-9._~+/=-]{8,}|"
    r"(?:postgres(?:ql)?://)[^\s]+|(?:wk|ws)-[A-Za-z0-9_-]{8,})"
)
_STEP_DIR = re.compile(r"[0-9]{6,10}")
_MAX_REPO_FILES = 10_000
_MAX_CHECKPOINT_DIRECTORIES = 1_024
_MAX_TRAINING_STEP_BYTES = 256
_MAX_DATASET_INFO_BYTES = 512 * 1024
_MAX_DATASET_FEATURES = 128
_MAX_DATASET_CAMERAS = 16
_MAX_MODEL_CONFIG_BYTES = 256 * 1024
_MAX_PROCESSOR_CONFIG_BYTES = 512 * 1024
_MAX_PROCESSOR_STATE_BYTES = 64 * 1024 * 1024
_MAX_BASE_MODEL_BYTES = 16 * 1024 * 1024 * 1024
_MAX_BASE_MODEL_FILES = 2_048
_MAX_CHILD_LINE_CHARACTERS = 4_097
_SENSITIVE_ENV_SUFFIXES = (
    "_API_KEY",
    "_CREDENTIAL",
    "_CREDENTIALS",
    "_PASSWORD",
    "_SECRET",
    "_TOKEN",
)
_SMOLVLA_CONFIG_FIELDS = {
    "type",
    "n_obs_steps",
    "input_features",
    "output_features",
    "device",
    "use_amp",
    "use_peft",
    "push_to_hub",
    "repo_id",
    "private",
    "tags",
    "license",
    "pretrained_path",
    "chunk_size",
    "n_action_steps",
    "normalization_mapping",
    "max_state_dim",
    "max_action_dim",
    "resize_imgs_with_padding",
    "empty_cameras",
    "adapt_to_pi_aloha",
    "use_delta_joint_actions_aloha",
    "tokenizer_max_length",
    "num_steps",
    "use_cache",
    "freeze_vision_encoder",
    "train_expert_only",
    "train_state_proj",
    "optimizer_lr",
    "optimizer_betas",
    "optimizer_eps",
    "optimizer_weight_decay",
    "optimizer_grad_clip_norm",
    "scheduler_warmup_steps",
    "scheduler_decay_steps",
    "scheduler_decay_lr",
    "vlm_model_name",
    "load_vlm_weights",
    "add_image_special_tokens",
    "attention_mode",
    "prefix_length",
    "pad_language_to",
    "num_expert_layers",
    "num_vlm_layers",
    "self_attn_every_n_layers",
    "expert_width_multiplier",
    "min_period",
    "max_period",
    "rtc_config",
    "compile_model",
    "compile_mode",
}
_SMOLVLA_PREPROCESSOR_STEPS = (
    "rename_observations_processor",
    "to_batch_processor",
    "smolvla_new_line_processor",
    "tokenizer_processor",
    "device_processor",
    "normalizer_processor",
)
_SMOLVLA_POSTPROCESSOR_STEPS = (
    "unnormalizer_processor",
    "device_processor",
)
_UNTRUSTED_CODE_SUFFIXES = {
    ".bat",
    ".bin",
    ".cmd",
    ".com",
    ".dll",
    ".dylib",
    ".exe",
    ".joblib",
    ".pkl",
    ".pickle",
    ".ps1",
    ".pt",
    ".pth",
    ".py",
    ".pyc",
    ".pyo",
    ".sh",
    ".so",
}
_FORBIDDEN_CONFIG_KEYS = {
    "_target_",
    "auto_map",
    "class",
    "custom_pipelines",
    "trust_remote_code",
}


class _Image(Protocol):
    def apt_install(self, *packages: str) -> _Image: ...

    def uv_pip_install(self, *packages: str) -> _Image: ...

    def add_local_python_source(self, *modules: str, **kwargs: Any) -> _Image: ...


class _ModalModule(Protocol):
    App: Any
    Image: Any
    Secret: Any


@dataclass(frozen=True)
class ModalTrainingWorkload:
    app: Any
    training_function: Any


class _DuplicateJSONKey(ValueError):
    pass


class _NonFiniteJSONValue(ValueError):
    pass


def managed_training_spec_payload(spec: ManagedTrainingSpec) -> dict[str, object]:
    """Create the credential-free, strictly shaped Modal Function input."""

    payload = asdict(spec)
    payload["job_id"] = str(spec.job_id)
    payload["deadline_at"] = spec.deadline_at.isoformat()
    return cast(dict[str, object], payload)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey
        result[key] = value
    return result


def _reject_nonfinite(_value: str) -> None:
    raise _NonFiniteJSONValue


def _strict_json(value: str) -> object:
    try:
        return json.loads(
            value,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (
        json.JSONDecodeError,
        _DuplicateJSONKey,
        _NonFiniteJSONValue,
        RecursionError,
        TypeError,
    ):
        raise ValueError("Managed training protocol JSON is invalid.") from None


def _read_bounded_json(path: Path, *, max_bytes: int) -> object:
    try:
        if path.is_symlink():
            raise ValueError
        metadata = path.stat()
        if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= max_bytes:
            raise ValueError
        with path.open("rb") as stream:
            raw = stream.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise ValueError
        document = _strict_json(raw.decode("utf-8", errors="strict"))
    except (OSError, UnicodeDecodeError, ValueError):
        raise RuntimeError("The managed base model configuration is invalid.") from None
    _validate_bounded_json_tree(document)
    return document


def _validate_bounded_json_tree(document: object) -> None:
    pending: list[tuple[object, int]] = [(document, 0)]
    nodes = 0
    while pending:
        value, depth = pending.pop()
        nodes += 1
        if nodes > 20_000 or depth > 20:
            raise RuntimeError("The managed base model configuration is too complex.")
        if value is None or isinstance(value, (bool, int)):
            continue
        if isinstance(value, float):
            if not math.isfinite(value):
                raise RuntimeError("The managed base model configuration is invalid.")
            continue
        if isinstance(value, str):
            if len(value) > 4_096:
                raise RuntimeError("The managed base model configuration is too large.")
            continue
        if isinstance(value, list):
            if len(value) > 512:
                raise RuntimeError("The managed base model configuration is too large.")
            pending.extend((item, depth + 1) for item in value)
            continue
        if isinstance(value, dict):
            if len(value) > 512:
                raise RuntimeError("The managed base model configuration is too large.")
            for key, item in value.items():
                if not isinstance(key, str) or len(key) > 128:
                    raise RuntimeError("The managed base model configuration is invalid.")
                if key.casefold() in _FORBIDDEN_CONFIG_KEYS:
                    raise RuntimeError(
                        "The managed base model requests executable configuration."
                    )
                pending.append((item, depth + 1))
            continue
        raise RuntimeError("The managed base model configuration is invalid.")


def _regular_file_under(
    root: Path,
    path: Path,
    *,
    max_bytes: int,
    allow_empty: bool = False,
) -> int:
    try:
        if path.is_symlink():
            raise ValueError
        metadata = path.stat()
        resolved_root = root.resolve(strict=True)
        resolved_path = path.resolve(strict=True)
    except (OSError, ValueError):
        raise RuntimeError("The managed model artifact path is invalid.") from None
    if (
        not stat.S_ISREG(metadata.st_mode)
        or not resolved_path.is_relative_to(resolved_root)
        or metadata.st_size > max_bytes
        or (metadata.st_size == 0 and not allow_empty)
    ):
        raise RuntimeError("The managed model artifact path is invalid.")
    return metadata.st_size


def _validate_features(value: object) -> None:
    if value is None:
        return
    if not isinstance(value, dict) or len(value) > 64:
        raise RuntimeError("The managed SmolVLA feature configuration is invalid.")
    for name, feature in value.items():
        if (
            not isinstance(name, str)
            or not 1 <= len(name) <= 128
            or not isinstance(feature, dict)
            or set(feature) != {"type", "shape"}
            or feature.get("type") not in {"ACTION", "ENV", "STATE", "VISUAL"}
        ):
            raise RuntimeError("The managed SmolVLA feature configuration is invalid.")
        shape = feature.get("shape")
        if (
            not isinstance(shape, list)
            or not 1 <= len(shape) <= 4
            or any(
                isinstance(item, bool)
                or not isinstance(item, int)
                or not 1 <= item <= 4_096
                for item in shape
            )
            or math.prod(shape) > 64 * 1024 * 1024
        ):
            raise RuntimeError("The managed SmolVLA feature configuration is invalid.")


def _validate_managed_dataset_snapshot(
    dataset_path: Path,
    *,
    policy_config: dict[str, object],
) -> None:
    """Fail closed before LeRobot derives a policy shape from dataset metadata."""

    try:
        document = _read_bounded_json(
            dataset_path / "meta" / "info.json",
            max_bytes=_MAX_DATASET_INFO_BYTES,
        )
    except RuntimeError:
        raise RuntimeError("The managed training dataset metadata is invalid.") from None
    if not isinstance(document, dict):
        raise RuntimeError("The managed training dataset metadata is invalid.")
    features = document.get("features")
    if not isinstance(features, dict) or not 1 <= len(features) <= _MAX_DATASET_FEATURES:
        raise RuntimeError("The managed training dataset features are invalid.")

    max_state_dim = policy_config.get("max_state_dim", 32)
    max_action_dim = policy_config.get("max_action_dim", 32)
    if (
        isinstance(max_state_dim, bool)
        or not isinstance(max_state_dim, int)
        or isinstance(max_action_dim, bool)
        or not isinstance(max_action_dim, int)
    ):
        raise RuntimeError("The managed SmolVLA feature bounds are invalid.")

    visual_count = 0
    state_dimension: int | None = None
    action_dimension: int | None = None
    for name, feature in features.items():
        if (
            not isinstance(name, str)
            or not 1 <= len(name) <= 128
            or not isinstance(feature, dict)
            or not {"dtype", "shape", "names"}.issubset(feature)
        ):
            raise RuntimeError("The managed training dataset features are invalid.")
        dtype = feature.get("dtype")
        shape = feature.get("shape")
        names = feature.get("names")
        if (
            not isinstance(dtype, str)
            or not isinstance(shape, list)
            or not 1 <= len(shape) <= 4
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= 4_096
                for value in shape
            )
            or math.prod(shape) > 64 * 1024 * 1024
        ):
            raise RuntimeError("The managed training dataset features are invalid.")

        if name == "observation.state":
            if (
                dtype != "float32"
                or len(shape) != 1
                or shape[0] > max_state_dim
                or not isinstance(names, list)
                or len(names) != shape[0]
                or any(
                    not isinstance(value, str) or not 1 <= len(value) <= 128
                    for value in names
                )
            ):
                raise RuntimeError("The managed training dataset state is incompatible with SmolVLA.")
            state_dimension = shape[0]
        elif name == "action":
            if (
                dtype != "float32"
                or len(shape) != 1
                or shape[0] > max_action_dim
                or not isinstance(names, list)
                or len(names) != shape[0]
                or any(
                    not isinstance(value, str) or not 1 <= len(value) <= 128
                    for value in names
                )
            ):
                raise RuntimeError("The managed training dataset action is incompatible with SmolVLA.")
            action_dimension = shape[0]
        elif name.startswith("observation.images."):
            if (
                dtype not in {"image", "video"}
                or len(shape) != 3
                or shape[2] != 3
                or not isinstance(names, list)
                or len(names) != 3
                or names[2] not in {"channel", "channels"}
            ):
                raise RuntimeError("The managed training dataset camera is incompatible with SmolVLA.")
            visual_count += 1
            if visual_count > _MAX_DATASET_CAMERAS:
                raise RuntimeError("The managed training dataset has too many cameras.")
        elif name.startswith("observation.") or name.startswith("action"):
            # dataset_to_policy_features would otherwise introduce an unreviewed
            # SmolVLA input/output contract.
            raise RuntimeError("The managed training dataset has unsupported policy features.")

    if state_dimension is None or action_dimension is None or visual_count == 0:
        raise RuntimeError("The managed training dataset is missing required SmolVLA features.")


def _validate_smolvla_policy_config(
    base_model_path: Path, *, expected_vlm_name: str = SMOLVLA_VLM_REPO
) -> dict[str, object]:
    document = _read_bounded_json(
        base_model_path / "config.json", max_bytes=_MAX_MODEL_CONFIG_BYTES
    )
    if (
        not isinstance(document, dict)
        or not set(document).issubset(_SMOLVLA_CONFIG_FIELDS)
        or document.get("type") != "smolvla"
        or document.get("vlm_model_name", SMOLVLA_VLM_REPO) != expected_vlm_name
        or document.get("use_peft", False) is not False
        or document.get("pretrained_path") is not None
        or document.get("rtc_config") is not None
        or document.get("compile_model", False) is not False
    ):
        raise RuntimeError("Only fixed, built-in SmolVLA base models are supported.")
    boolean_fields = {
        "use_amp",
        "push_to_hub",
        "adapt_to_pi_aloha",
        "use_delta_joint_actions_aloha",
        "use_cache",
        "freeze_vision_encoder",
        "train_expert_only",
        "train_state_proj",
        "load_vlm_weights",
        "add_image_special_tokens",
    }
    if any(name in document and not isinstance(document[name], bool) for name in boolean_fields):
        raise RuntimeError("The managed SmolVLA configuration is invalid.")
    integer_bounds = {
        "n_obs_steps": (1, 16),
        "chunk_size": (1, 1_024),
        "n_action_steps": (1, 1_024),
        "max_state_dim": (1, 4_096),
        "max_action_dim": (1, 4_096),
        "empty_cameras": (0, 16),
        "tokenizer_max_length": (1, 4_096),
        "num_steps": (1, 10_000),
        "scheduler_warmup_steps": (0, 10_000_000),
        "scheduler_decay_steps": (1, 100_000_000),
        "prefix_length": (-1, 1_000_000),
        "num_expert_layers": (-1, 256),
        "num_vlm_layers": (-1, 256),
        "self_attn_every_n_layers": (-1, 256),
    }
    for name, (minimum, maximum) in integer_bounds.items():
        value = document.get(name)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not minimum <= value <= maximum
        ):
            raise RuntimeError("The managed SmolVLA configuration is out of bounds.")
    for name in (
        "optimizer_lr",
        "optimizer_eps",
        "optimizer_weight_decay",
        "optimizer_grad_clip_norm",
        "scheduler_decay_lr",
        "expert_width_multiplier",
        "min_period",
        "max_period",
    ):
        value = document.get(name)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0 <= float(value) <= 1_000_000
        ):
            raise RuntimeError("The managed SmolVLA configuration is out of bounds.")
    betas = document.get("optimizer_betas")
    if betas is not None and (
        not isinstance(betas, list)
        or len(betas) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0 <= float(value) < 1
            for value in betas
        )
    ):
        raise RuntimeError("The managed SmolVLA optimizer configuration is invalid.")
    resize = document.get("resize_imgs_with_padding")
    if resize is not None and (
        not isinstance(resize, list)
        or len(resize) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or not 16 <= value <= 4_096 for value in resize)
    ):
        raise RuntimeError("The managed SmolVLA image configuration is invalid.")
    if document.get("device") not in {None, "cpu", "cuda", "cuda:0"}:
        raise RuntimeError("The managed SmolVLA device configuration is invalid.")
    private = document.get("private")
    if private is not None and not isinstance(private, bool):
        raise RuntimeError("The managed SmolVLA Hub configuration is invalid.")
    repo_id = document.get("repo_id")
    if repo_id is not None and (
        not isinstance(repo_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}", repo_id)
        is None
    ):
        raise RuntimeError("The managed SmolVLA Hub configuration is invalid.")
    tags = document.get("tags")
    if tags is not None and (
        not isinstance(tags, list)
        or len(tags) > 32
        or any(not isinstance(tag, str) or not 1 <= len(tag) <= 64 for tag in tags)
    ):
        raise RuntimeError("The managed SmolVLA Hub configuration is invalid.")
    license_name = document.get("license")
    if license_name is not None and (
        not isinstance(license_name, str) or not 1 <= len(license_name) <= 128
    ):
        raise RuntimeError("The managed SmolVLA Hub configuration is invalid.")
    if document.get("attention_mode", "cross_attn") not in {"cross_attn", "self_attn"}:
        raise RuntimeError("The managed SmolVLA attention configuration is invalid.")
    if document.get("pad_language_to", "longest") not in {"longest", "max_length"}:
        raise RuntimeError("The managed SmolVLA tokenizer configuration is invalid.")
    normalization = document.get("normalization_mapping")
    if normalization is not None and (
        not isinstance(normalization, dict)
        or not set(normalization).issubset({"ACTION", "ENV", "STATE", "VISUAL"})
        or any(
            value not in {"IDENTITY", "MEAN_STD", "MIN_MAX"}
            for value in normalization.values()
        )
    ):
        raise RuntimeError("The managed SmolVLA normalization configuration is invalid.")
    chunk_size = document.get("chunk_size", 50)
    action_steps = document.get("n_action_steps", 50)
    if isinstance(chunk_size, int) and isinstance(action_steps, int) and action_steps > chunk_size:
        raise RuntimeError("The managed SmolVLA action configuration is invalid.")
    _validate_features(document.get("input_features"))
    _validate_features(document.get("output_features"))
    return cast(dict[str, object], document)


def _validate_processor_config(
    base_model_path: Path,
    *,
    filename: str,
    name: str,
    expected_steps: tuple[str, ...],
    expected_tokenizer_name: str = SMOLVLA_VLM_REPO,
) -> None:
    document = _read_bounded_json(
        base_model_path / filename, max_bytes=_MAX_PROCESSOR_CONFIG_BYTES
    )
    if (
        not isinstance(document, dict)
        or set(document) != {"name", "steps"}
        or document.get("name") != name
        or not isinstance(document.get("steps"), list)
        or len(document["steps"]) != len(expected_steps)
    ):
        raise RuntimeError("The managed SmolVLA processor configuration is invalid.")
    for index, (entry, expected_name) in enumerate(zip(document["steps"], expected_steps, strict=True)):
        if (
            not isinstance(entry, dict)
            or not {"registry_name", "config"}.issubset(entry)
            or not set(entry).issubset({"registry_name", "config", "state_file"})
            or entry.get("registry_name") != expected_name
            or not isinstance(entry.get("config"), dict)
        ):
            raise RuntimeError("The managed SmolVLA processor step is invalid.")
        config = entry["config"]
        if expected_name == "rename_observations_processor":
            rename_map = config.get("rename_map", {})
            if (
                set(config) != {"rename_map"}
                or not isinstance(rename_map, dict)
                or len(rename_map) > 64
                or any(
                    not isinstance(source, str)
                    or not isinstance(target, str)
                    or not 1 <= len(source) <= 128
                    or not 1 <= len(target) <= 128
                    for source, target in rename_map.items()
                )
            ):
                raise RuntimeError("The managed rename processor is invalid.")
        elif expected_name in {"to_batch_processor", "smolvla_new_line_processor"}:
            if config:
                raise RuntimeError("The managed fixed processor accepts no configuration.")
        elif expected_name == "tokenizer_processor":
            if (
                not set(config).issubset(
                    {"max_length", "task_key", "padding_side", "padding", "truncation", "tokenizer_name"}
                )
                or config.get("tokenizer_name") != expected_tokenizer_name
                or config.get("task_key", "task") != "task"
                or config.get("padding_side", "right") != "right"
                or config.get("padding", "max_length") not in {"longest", "max_length"}
                or not isinstance(config.get("truncation", True), bool)
                or isinstance(config.get("max_length", 512), bool)
                or not isinstance(config.get("max_length", 512), int)
                or not 1 <= config.get("max_length", 512) <= 4_096
            ):
                raise RuntimeError("The managed SmolVLA tokenizer is not allowed.")
        elif expected_name == "device_processor":
            device = config.get("device", "cpu")
            if (
                not set(config).issubset({"device", "float_dtype"})
                or device not in {"cpu", "cuda", "cuda:0"}
                or config.get("float_dtype") not in {None, "float16", "float32", "bfloat16"}
                or (name == "policy_postprocessor" and device != "cpu")
            ):
                raise RuntimeError("The managed device processor is invalid.")
        elif expected_name in {"normalizer_processor", "unnormalizer_processor"}:
            if not set(config).issubset(
                {"eps", "features", "norm_map", "normalize_observation_keys"}
            ):
                raise RuntimeError("The managed normalization processor is invalid.")
            epsilon = config.get("eps", 1e-8)
            if (
                isinstance(epsilon, bool)
                or not isinstance(epsilon, (int, float))
                or not math.isfinite(float(epsilon))
                or not 0 < float(epsilon) <= 1
            ):
                raise RuntimeError("The managed normalization processor is invalid.")
            _validate_features(config.get("features"))
            norm_map = config.get("norm_map", {})
            if (
                not isinstance(norm_map, dict)
                or not set(norm_map).issubset({"ACTION", "ENV", "STATE", "VISUAL"})
                or any(value not in {"IDENTITY", "MEAN_STD", "MIN_MAX"} for value in norm_map.values())
            ):
                raise RuntimeError("The managed normalization processor is invalid.")
            observation_keys = config.get("normalize_observation_keys")
            if observation_keys is not None and (
                not isinstance(observation_keys, list)
                or len(observation_keys) > 64
                or any(
                    not isinstance(value, str) or not 1 <= len(value) <= 128
                    for value in observation_keys
                )
            ):
                raise RuntimeError("The managed normalization processor is invalid.")
        state_file = entry.get("state_file")
        if state_file is not None:
            if (
                not isinstance(state_file, str)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\.safetensors", state_file) is None
            ):
                raise RuntimeError("The managed processor state path is invalid.")
            _regular_file_under(
                base_model_path,
                base_model_path / state_file,
                max_bytes=_MAX_PROCESSOR_STATE_BYTES,
            )
        if expected_name not in {"normalizer_processor", "unnormalizer_processor"} and state_file is not None:
            raise RuntimeError("The managed processor state is not allowed.")


def _validate_base_model_snapshot(
    base_model_path: Path, *, expected_vlm_name: str = SMOLVLA_VLM_REPO
) -> None:
    try:
        root_metadata = base_model_path.lstat()
        resolved_root = base_model_path.resolve(strict=True)
    except OSError:
        raise RuntimeError("The managed base model snapshot is unavailable.") from None
    if base_model_path.is_symlink() or not stat.S_ISDIR(root_metadata.st_mode):
        raise RuntimeError("The managed base model snapshot path is invalid.")
    count = 0
    total_bytes = 0
    for path in base_model_path.rglob("*"):
        count += 1
        if count > _MAX_BASE_MODEL_FILES:
            raise RuntimeError("The managed base model contains too many files.")
        try:
            metadata = path.lstat()
            resolved = path.resolve(strict=True)
        except OSError:
            raise RuntimeError("The managed base model contains an invalid path.") from None
        if path.is_symlink() or not resolved.is_relative_to(resolved_root):
            raise RuntimeError("The managed base model contains an unsafe path.")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("The managed base model contains an unsupported file.")
        if path.suffix.casefold() in _UNTRUSTED_CODE_SUFFIXES or path.name in {
            "adapter_config.json",
            "adapter_model.safetensors",
        }:
            raise RuntimeError("The managed base model contains executable or adapter content.")
        total_bytes += metadata.st_size
        if total_bytes > _MAX_BASE_MODEL_BYTES:
            raise RuntimeError("The managed base model is too large.")
    _regular_file_under(
        base_model_path,
        base_model_path / "model.safetensors",
        max_bytes=_MAX_BASE_MODEL_BYTES,
    )
    _validate_smolvla_policy_config(
        base_model_path, expected_vlm_name=expected_vlm_name
    )
    _validate_processor_config(
        base_model_path,
        filename="policy_preprocessor.json",
        name="policy_preprocessor",
        expected_steps=_SMOLVLA_PREPROCESSOR_STEPS,
        expected_tokenizer_name=expected_vlm_name,
    )


def _write_json_atomically(path: Path, document: dict[str, object]) -> None:
    encoded = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    if len(encoded) > _MAX_PROCESSOR_CONFIG_BYTES:
        raise RuntimeError("The sealed managed model configuration is too large.")
    temporary = path.parent / f".ctrl-pi-{path.name}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("The managed model configuration could not be sealed.") from None


def _seal_base_model_for_child(
    base_model_path: Path,
    vlm_model_path: Path,
    *,
    dataset_path: Path,
) -> None:
    local_vlm = str(vlm_model_path.resolve(strict=True))
    policy = _validate_smolvla_policy_config(base_model_path)
    _validate_managed_dataset_snapshot(dataset_path, policy_config=policy)
    policy.update(
        {
            "device": "cuda",
            "use_peft": False,
            "push_to_hub": False,
            "pretrained_path": None,
            "vlm_model_name": local_vlm,
            "load_vlm_weights": False,
            # LeRobot 0.4.4 derives these exact shapes from the selected
            # immutable dataset only when the pretrained config leaves them
            # empty. Retaining the generic SmolVLA base's state-6/camera1-3
            # declaration would make ctrl-pi's state-7/workspace data either
            # fail validation or persist an inference-incompatible config.
            "input_features": {},
            "output_features": {},
            "empty_cameras": 0,
            "adapt_to_pi_aloha": False,
            "use_delta_joint_actions_aloha": False,
        }
    )
    processor_path = base_model_path / "policy_preprocessor.json"
    processor = _read_bounded_json(
        processor_path, max_bytes=_MAX_PROCESSOR_CONFIG_BYTES
    )
    if not isinstance(processor, dict) or not isinstance(processor.get("steps"), list):
        raise RuntimeError("The managed SmolVLA processor configuration is invalid.")
    tokenizer_steps = [
        entry
        for entry in processor["steps"]
        if isinstance(entry, dict) and entry.get("registry_name") == "tokenizer_processor"
    ]
    if len(tokenizer_steps) != 1 or not isinstance(tokenizer_steps[0].get("config"), dict):
        raise RuntimeError("The managed SmolVLA tokenizer configuration is invalid.")
    tokenizer_steps[0]["config"]["tokenizer_name"] = local_vlm
    device_steps = [
        entry
        for entry in processor["steps"]
        if isinstance(entry, dict) and entry.get("registry_name") == "device_processor"
    ]
    if len(device_steps) != 1 or not isinstance(device_steps[0].get("config"), dict):
        raise RuntimeError("The managed SmolVLA device configuration is invalid.")
    device_steps[0]["config"]["device"] = "cuda"
    device_steps[0]["config"]["float_dtype"] = None
    _write_json_atomically(base_model_path / "config.json", policy)
    _write_json_atomically(processor_path, cast(dict[str, object], processor))
    _validate_base_model_snapshot(base_model_path, expected_vlm_name=local_vlm)
    _validate_processor_config(
        base_model_path,
        filename="policy_postprocessor.json",
        name="policy_postprocessor",
        expected_steps=_SMOLVLA_POSTPROCESSOR_STEPS,
    )


def _validate_vlm_snapshot(vlm_model_path: Path) -> None:
    for filename in _SMOLVLA_VLM_FILES:
        _regular_file_under(
            vlm_model_path,
            vlm_model_path / filename,
            max_bytes=8 * 1024 * 1024,
        )
    for filename in (
        "config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
    ):
        _read_bounded_json(
            vlm_model_path / filename,
            max_bytes=_MAX_PROCESSOR_CONFIG_BYTES,
        )


def _validated_worker_spec(payload: object) -> ManagedTrainingSpec:
    fields = {
        "job_id",
        "request_hash",
        "app_name",
        "ownership_tag",
        "dataset_repo",
        "dataset_revision",
        "base_model",
        "base_model_revision",
        "output_model_repo",
        "output_marker_revision",
        "output_private",
        "runtime",
        "max_steps",
        "batch_size",
        "log_every",
        "save_every",
        "seed",
        "num_workers",
        "compute_size",
        "timeout_seconds",
        "deadline_at",
    }
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError("Managed training payload is invalid.")
    strings = fields - {
        "output_private",
        "max_steps",
        "batch_size",
        "log_every",
        "save_every",
        "seed",
        "num_workers",
        "timeout_seconds",
    }
    if any(not isinstance(payload[name], str) for name in strings):
        raise ValueError("Managed training payload is invalid.")
    integers = {
        "max_steps",
        "batch_size",
        "log_every",
        "save_every",
        "seed",
        "num_workers",
        "timeout_seconds",
    }
    if any(
        isinstance(payload[name], bool) or not isinstance(payload[name], int)
        for name in integers
    ):
        raise ValueError("Managed training payload is invalid.")
    if not isinstance(payload["output_private"], bool):
        raise ValueError("Managed training payload is invalid.")
    try:
        job_id = uuid.UUID(cast(str, payload["job_id"]))
        deadline_at = datetime.fromisoformat(cast(str, payload["deadline_at"]))
        spec = ManagedTrainingSpec(
            **{
                **payload,
                "job_id": job_id,
                "deadline_at": deadline_at,
            }
        )
    except (TypeError, ValueError):
        raise ValueError("Managed training payload is invalid.") from None
    if payload != managed_training_spec_payload(spec):
        raise ValueError("Managed training payload is not canonical.")
    return spec


def _safe_line(value: str, *, secret: str) -> str | None:
    line = _ANSI_ESCAPE.sub("", value).rstrip("\r\n")
    line = "".join(
        character
        if character == "\t" or (character.isprintable() and character not in "\r\n")
        else " "
        for character in line
    )
    if not line.strip():
        return None
    if secret and secret in line:
        line = "[redacted trainer output]"
    else:
        line = _TOKEN_LIKE.sub("[redacted]", line)
    encoded = line.encode("utf-8")
    if len(encoded) > 4 * 1024:
        line = encoded[: 4 * 1024].decode("utf-8", errors="ignore")
    return line or None


class _EventEmitter:
    def __init__(
        self,
        *,
        job_id: uuid.UUID,
        request_hash: str,
        output: TextIO,
    ) -> None:
        self._job_id = job_id
        self._request_hash = request_hash
        self._output = output
        self.sequence = 0
        self.bytes_emitted = 0
        self.log_events = 0
        self.truncated = False

    def emit(self, kind: str, values: dict[str, object], *, essential: bool = False) -> bool:
        if self.sequence >= MAX_WORKER_EVENTS:
            self.truncated = True
            return False
        event = {
            "schema": MODAL_TRAINING_EVENT_SCHEMA,
            "version": MODAL_TRAINING_PROTOCOL_VERSION,
            "job_id": str(self._job_id),
            "request_hash": self._request_hash,
            "sequence": self.sequence + 1,
            "type": kind,
            **values,
        }
        try:
            encoded = json.dumps(
                event,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        except (TypeError, ValueError):
            self.truncated = True
            return False
        if len(encoded) > MAX_WORKER_EVENT_LINE_BYTES:
            self.truncated = True
            return False
        budget = MAX_WORKER_EVENT_BYTES
        if not essential:
            budget -= _EVENT_RESERVE_BYTES
        if self.bytes_emitted + len(encoded) + 1 > budget:
            self.truncated = True
            return False
        print(MODAL_TRAINING_EVENT_PREFIX + encoded.decode("utf-8"), file=self._output, flush=True)
        self.sequence += 1
        self.bytes_emitted += len(encoded) + 1
        if kind == "log":
            self.log_events += 1
        return True

    def log(self, *, source: str, line: str, step: int | None = None) -> None:
        if self.log_events >= MAX_WORKER_LOG_EVENTS:
            self.truncated = True
            return
        values: dict[str, object] = {"source": source, "line": line}
        if step is not None:
            values["step"] = step
        self.emit("log", values)


def _parse_metrics(line: str) -> tuple[int | None, dict[str, float]]:
    parsed: dict[str, float] = {}
    step: int | None = None
    for match in _METRIC_VALUE.finditer(line):
        name = match.group("name")
        try:
            value = float(match.group("value")) * _METRIC_SUFFIX_MULTIPLIERS[
                match.group("suffix")
            ]
        except (KeyError, ValueError, OverflowError):
            continue
        if not math.isfinite(value) or abs(value) > 1e300:
            continue
        if name == "step" and value.is_integer() and 0 <= value <= 2_147_483_647:
            step = int(value)
        elif len(parsed) < 64:
            parsed[name] = value
    return step, parsed


def _format_tracker_number(value: int) -> str:
    number = float(value)
    for suffix in ("", "K", "M", "B", "T", "Q"):
        if abs(number) < 1_000:
            return f"{number:.0f}{suffix}"
        number /= 1_000
    return str(number)


class _TrainingMetricParser:
    """Recover exact scheduled steps from LeRobot's rounded tracker display."""

    def __init__(self, *, log_every: int, max_steps: int) -> None:
        self._log_every = log_every
        self._max_steps = max_steps
        self._last_tracker_step = 0

    def parse(
        self,
        line: str,
        *,
        transport_truncated: bool,
    ) -> tuple[int | None, dict[str, float]]:
        displayed_step, metrics = _parse_metrics(line)
        tracker = _TRACKER_SIGNATURE.search(line)
        if tracker is None:
            return displayed_step, metrics

        candidate = self._last_tracker_step + self._log_every
        displayed_token = tracker.group("step")
        if transport_truncated:
            # One or more whole log lines may have been dropped. Advance only
            # to a schedule point compatible with LeRobot's rounded token; the
            # event tail is separately marked truncated so it is never claimed
            # complete.
            while (
                candidate <= self._max_steps
                and _format_tracker_number(candidate) != displayed_token
            ):
                candidate += self._log_every
        elif (
            candidate > self._max_steps
            or _format_tracker_number(candidate) != displayed_token
        ):
            return None, {}
        if candidate > self._max_steps:
            return None, {}
        self._last_tracker_step = candidate
        return candidate, metrics


def build_lerobot_training_command(
    spec: ManagedTrainingSpec,
    *,
    dataset_path: Path,
    base_model_path: Path,
    vlm_model_path: Path,
    output_dir: Path,
) -> list[str]:
    """Return the only executable recipe accepted by managed training."""

    _gpu_type, gpu_count = training_gpu_spec(spec.compute_size)
    module = ["-m", "lerobot.scripts.lerobot_train"]
    if gpu_count == 1:
        command = ["python", *module]
    else:
        command = [
            "accelerate",
            "launch",
            "--multi_gpu",
            "--num_processes",
            str(gpu_count),
            *module,
        ]
    return [
        *command,
        "--dataset.repo_id",
        spec.dataset_repo,
        "--dataset.revision",
        spec.dataset_revision,
        "--dataset.root",
        str(dataset_path),
        "--policy.path",
        str(base_model_path),
        "--policy.device",
        "cuda",
        "--policy.use_peft",
        "false",
        "--policy.vlm_model_name",
        str(vlm_model_path),
        "--policy.load_vlm_weights",
        "false",
        "--policy.push_to_hub",
        "false",
        "--output_dir",
        str(output_dir),
        "--steps",
        str(spec.max_steps),
        "--batch_size",
        str(spec.batch_size),
        "--log_freq",
        str(spec.log_every),
        "--save_checkpoint",
        "true",
        "--save_freq",
        str(spec.save_every),
        "--seed",
        str(spec.seed),
        "--num_workers",
        str(spec.num_workers),
        "--wandb.enable",
        "false",
    ]


def _repo_files(info: object) -> set[str]:
    files: set[str] = set()
    siblings = getattr(info, "siblings", None)
    if not isinstance(siblings, list):
        return files
    if len(siblings) > _MAX_REPO_FILES:
        raise RuntimeError("The managed model repository contains too many files.")
    for sibling in siblings:
        name = getattr(sibling, "rfilename", None)
        if isinstance(name, str):
            files.add(name)
    return files


def _verified_repo_revision(
    api: Any,
    *,
    spec: ManagedTrainingSpec,
    revision: str,
    required_prefix: str,
    marker_download: Any,
    token: str,
) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise RuntimeError("The Hub upload did not return an immutable revision.")
    info = api.model_info(
        repo_id=spec.output_model_repo,
        revision=revision,
        token=token,
    )
    if getattr(info, "sha", None) != revision:
        raise RuntimeError("The Hub upload revision could not be verified.")
    repo_identity = getattr(info, "id", getattr(info, "repo_id", None))
    if repo_identity != spec.output_model_repo:
        raise RuntimeError("The Hub upload repository identity is invalid.")
    if getattr(info, "private", None) is not spec.output_private:
        raise RuntimeError("The managed model repository visibility changed.")
    prefix = required_prefix.strip("/")
    required = {
        f"{prefix}/{filename}" if prefix else filename
        for filename in _REQUIRED_POLICY_FILES
    }
    required.update(
        f"{prefix}/{MANAGED_SMOLVLA_DEPENDENCY_DIR}/{filename}"
        if prefix
        else f"{MANAGED_SMOLVLA_DEPENDENCY_DIR}/{filename}"
        for filename in _SMOLVLA_VLM_FILES
    )
    if not required.issubset(_repo_files(info)):
        raise RuntimeError("The uploaded policy artifact is incomplete.")
    marker_path = Path(
        marker_download(
            repo_id=spec.output_model_repo,
            repo_type="model",
            filename=MANAGED_TRAINING_MARKER_PATH,
            revision=revision,
            token=token,
        )
    )
    if marker_path.stat().st_size > 4 * 1024 or marker_path.read_bytes() != managed_training_marker(
        spec.job_id, spec.request_hash
    ):
        raise RuntimeError("The managed model ownership marker changed.")
    return revision


def _upload_policy(
    api: Any,
    *,
    spec: ManagedTrainingSpec,
    policy_dir: Path,
    step: int,
    final: bool,
    vlm_model_path: Path,
    marker_download: Any,
    token: str,
) -> str:
    _prepare_policy_for_upload(
        policy_dir, spec=spec, vlm_model_path=vlm_model_path
    )
    path_in_repo = "" if final else f"checkpoints/{step:010d}"
    readme = policy_dir / "README.md"
    if final:
        # Always replace any generated/source card for this upload. ctrl-pi
        # cannot infer or grant a model license on the user's behalf.
        readme.write_text(
            "---\nlibrary_name: lerobot\ntags:\n  - ctrl-pi\n  - robotics\n---\n\n"
            "# ctrl-pi managed training output\n\n"
            "This policy artifact was produced by a user-owned ctrl-pi deployment.\n",
            encoding="utf-8",
        )
    try:
        commit = api.upload_folder(
            repo_id=spec.output_model_repo,
            repo_type="model",
            folder_path=str(policy_dir),
            path_in_repo=path_in_repo,
            commit_message=(
                f"ctrl-pi final checkpoint {step}"
                if final
                else f"ctrl-pi checkpoint {step}"
            ),
            token=token,
        )
    finally:
        if final:
            readme.unlink(missing_ok=True)
    revision = getattr(commit, "oid", None)
    if not isinstance(revision, str):
        raise RuntimeError("The Hub upload did not return a revision.")
    return _verified_repo_revision(
        api,
        spec=spec,
        revision=revision,
        required_prefix=path_in_repo,
        marker_download=marker_download,
        token=token,
    )


def _validate_upload_tree(policy_dir: Path) -> None:
    try:
        root_metadata = policy_dir.lstat()
        resolved_root = policy_dir.resolve(strict=True)
    except OSError:
        raise RuntimeError("The LeRobot checkpoint path is invalid.") from None
    if policy_dir.is_symlink() or not stat.S_ISDIR(root_metadata.st_mode):
        raise RuntimeError("The LeRobot checkpoint path is invalid.")
    files: set[str] = set()
    count = 0
    total_bytes = 0
    for path in policy_dir.rglob("*"):
        count += 1
        if count > _MAX_BASE_MODEL_FILES:
            raise RuntimeError("The LeRobot checkpoint contains too many files.")
        try:
            metadata = path.lstat()
            resolved = path.resolve(strict=True)
        except OSError:
            raise RuntimeError("The LeRobot checkpoint contains an invalid path.") from None
        if path.is_symlink() or not resolved.is_relative_to(resolved_root):
            raise RuntimeError("The LeRobot checkpoint contains an unsafe path.")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("The LeRobot checkpoint contains an unsupported file.")
        total_bytes += metadata.st_size
        if total_bytes > _MAX_BASE_MODEL_BYTES:
            raise RuntimeError("The LeRobot checkpoint is too large.")
        if path.parent == policy_dir:
            files.add(path.name)
    if not _REQUIRED_POLICY_FILES.issubset(files):
        raise RuntimeError("The LeRobot checkpoint is incomplete.")


def _prepare_policy_for_upload(
    policy_dir: Path, *, spec: ManagedTrainingSpec, vlm_model_path: Path
) -> None:
    _validate_upload_tree(policy_dir)
    config_path = policy_dir / "config.json"
    document = _read_bounded_json(config_path, max_bytes=_MAX_MODEL_CONFIG_BYTES)
    if not isinstance(document, dict) or document.get("type") != "smolvla":
        raise RuntimeError("The trained SmolVLA policy configuration is invalid.")
    child_vlm_name = document.get("vlm_model_name")
    if not isinstance(child_vlm_name, str) or not Path(child_vlm_name).is_absolute():
        raise RuntimeError("The trained SmolVLA dependency configuration is invalid.")
    _validate_processor_config(
        policy_dir,
        filename="policy_preprocessor.json",
        name="policy_preprocessor",
        expected_steps=_SMOLVLA_PREPROCESSOR_STEPS,
        expected_tokenizer_name=child_vlm_name,
    )
    _validate_processor_config(
        policy_dir,
        filename="policy_postprocessor.json",
        name="policy_postprocessor",
        expected_steps=_SMOLVLA_POSTPROCESSOR_STEPS,
    )
    document.update(
        {
            "device": "cuda",
            "use_peft": False,
            "push_to_hub": False,
            "repo_id": spec.output_model_repo,
            "private": spec.output_private,
            "license": None,
            "pretrained_path": None,
            "vlm_model_name": SMOLVLA_VLM_REPO,
            "load_vlm_weights": False,
        }
    )
    processor_path = policy_dir / "policy_preprocessor.json"
    processor = _read_bounded_json(
        processor_path, max_bytes=_MAX_PROCESSOR_CONFIG_BYTES
    )
    if not isinstance(processor, dict) or not isinstance(processor.get("steps"), list):
        raise RuntimeError("The trained SmolVLA processor configuration is invalid.")
    tokenizer_steps = [
        entry
        for entry in processor["steps"]
        if isinstance(entry, dict) and entry.get("registry_name") == "tokenizer_processor"
    ]
    if len(tokenizer_steps) != 1 or not isinstance(tokenizer_steps[0].get("config"), dict):
        raise RuntimeError("The trained SmolVLA tokenizer configuration is invalid.")
    tokenizer_steps[0]["config"]["tokenizer_name"] = SMOLVLA_VLM_REPO
    _write_json_atomically(config_path, cast(dict[str, object], document))
    _write_json_atomically(processor_path, cast(dict[str, object], processor))
    _validate_processor_config(
        policy_dir,
        filename="policy_preprocessor.json",
        name="policy_preprocessor",
        expected_steps=_SMOLVLA_PREPROCESSOR_STEPS,
    )
    dependency_path = policy_dir / MANAGED_SMOLVLA_DEPENDENCY_DIR
    if dependency_path.exists() or dependency_path.is_symlink():
        raise RuntimeError("The trained policy dependency path already exists.")
    try:
        dependency_path.mkdir(mode=0o755)
        for filename in _SMOLVLA_VLM_FILES:
            source = vlm_model_path / filename
            destination = dependency_path / filename
            _regular_file_under(
                vlm_model_path,
                source,
                max_bytes=8 * 1024 * 1024,
            )
            with source.open("rb") as source_stream, destination.open("xb") as destination_stream:
                shutil.copyfileobj(source_stream, destination_stream, length=64 * 1024)
    except OSError:
        raise RuntimeError("The trained policy dependency could not be sealed.") from None
    _validate_vlm_snapshot(dependency_path)
    _validate_upload_tree(policy_dir)


def _completed_checkpoint_step(checkpoint_dir: Path, expected_step: int) -> bool:
    """Return true only after LeRobot has written its bounded step sentinel.

    LeRobot writes ``training_step.json`` after ``pretrained_model``. Requiring
    the exact canonical shape prevents observing policy filenames midway
    through ``save_checkpoint`` and publishing a partially written file.
    """

    sentinel = checkpoint_dir / "training_state" / "training_step.json"
    try:
        if sentinel.is_symlink():
            return False
        metadata = sentinel.stat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size > _MAX_TRAINING_STEP_BYTES
            or not sentinel.resolve(strict=True).is_relative_to(
                checkpoint_dir.resolve(strict=True)
            )
        ):
            return False
        with sentinel.open("rb") as stream:
            raw = stream.read(_MAX_TRAINING_STEP_BYTES + 1)
    except OSError:
        return False
    if not raw or len(raw) > _MAX_TRAINING_STEP_BYTES:
        return False
    try:
        document = _strict_json(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, ValueError):
        return False
    return (
        isinstance(document, dict)
        and set(document) == {"step"}
        and isinstance(document["step"], int)
        and not isinstance(document["step"], bool)
        and document["step"] == expected_step
    )


def _checkpoint_dirs(output_dir: Path) -> list[tuple[int, Path]]:
    checkpoints = output_dir / "checkpoints"
    if checkpoints.is_symlink() or not checkpoints.is_dir():
        return []
    found: list[tuple[int, Path]] = []
    try:
        children = checkpoints.iterdir()
    except OSError:
        return []
    for index, child in enumerate(children):
        if index >= _MAX_CHECKPOINT_DIRECTORIES:
            raise RuntimeError("The LeRobot checkpoint directory limit was exceeded.")
        if child.is_symlink() or not child.is_dir() or _STEP_DIR.fullmatch(child.name) is None:
            continue
        step = int(child.name)
        if not _completed_checkpoint_step(child, step):
            continue
        policy_dir = child / "pretrained_model"
        if policy_dir.is_symlink() or not policy_dir.is_dir():
            continue
        try:
            files = {
                path.name
                for path in policy_dir.iterdir()
                if not path.is_symlink() and path.is_file()
            }
        except OSError:
            continue
        if _REQUIRED_POLICY_FILES.issubset(files):
            found.append((step, policy_dir))
    return sorted(found)


def _read_pipe(
    pipe: TextIO,
    source: str,
    messages: queue.Queue[tuple[str, str | None]],
    overflowed: threading.Event,
) -> None:
    try:
        while True:
            line = pipe.readline(_MAX_CHILD_LINE_CHARACTERS)
            if not line:
                break
            if len(line) == _MAX_CHILD_LINE_CHARACTERS and not line.endswith("\n"):
                overflowed.set()
            try:
                messages.put_nowait((source, line))
            except queue.Full:
                # Continue draining so a verbose child cannot deadlock on its pipe.
                overflowed.set()
                continue
    finally:
        try:
            messages.put_nowait((source, None))
        except queue.Full:
            pass


def _verify_output_marker(
    api: Any,
    *,
    spec: ManagedTrainingSpec,
    marker_download: Any,
    token: str,
) -> None:
    info = api.model_info(
        repo_id=spec.output_model_repo,
        revision=spec.output_marker_revision,
        token=token,
    )
    if getattr(info, "sha", None) != spec.output_marker_revision:
        raise RuntimeError("The managed model marker revision could not be verified.")
    repo_identity = getattr(info, "id", getattr(info, "repo_id", None))
    if repo_identity != spec.output_model_repo:
        raise RuntimeError("The managed model repository identity is invalid.")
    if getattr(info, "private", None) is not spec.output_private:
        raise RuntimeError("The managed model repository visibility is invalid.")
    marker_path = Path(
        marker_download(
            repo_id=spec.output_model_repo,
            repo_type="model",
            filename=MANAGED_TRAINING_MARKER_PATH,
            revision=spec.output_marker_revision,
            token=token,
        )
    )
    if marker_path.stat().st_size > 4 * 1024 or marker_path.read_bytes() != managed_training_marker(
        spec.job_id, spec.request_hash
    ):
        raise RuntimeError("The managed model ownership marker is invalid.")


def _require_before_deadline(
    spec: ManagedTrainingSpec,
    *,
    now: Callable[[], datetime],
) -> None:
    current = now()
    if current.tzinfo is None or current.utcoffset() is None:
        raise RuntimeError("The managed training clock is invalid.")
    if current >= spec.deadline_at:
        raise RuntimeError("The managed training deadline was reached.")


def _safe_child_environment(environ: dict[str, str] | os._Environ[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in environ.items():
        upper = key.upper()
        if upper in {"DATABASE_URL", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}:
            continue
        if upper.endswith(_SENSITIVE_ENV_SUFFIXES):
            continue
        result[key] = value
    return result


def _terminate_process_tree(process: Any) -> None:
    """Terminate the isolated LeRobot/Accelerate process group, then kill."""

    if process.poll() is not None:
        return
    process_id = getattr(process, "pid", None)
    try:
        if isinstance(process_id, int) and process_id > 0:
            os.killpg(process_id, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=5.0)
        return
    except BaseException:
        pass
    try:
        if isinstance(process_id, int) and process_id > 0:
            os.killpg(process_id, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=5.0)
    except BaseException:
        pass


def run_managed_training(
    payload: object,
    *,
    environ: dict[str, str] | os._Environ[str] | None = None,
    output: TextIO | None = None,
    api_factory: Any | None = None,
    snapshot: Any | None = None,
    marker_download: Any | None = None,
    popen_factory: Any = subprocess.Popen,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> dict[str, object]:
    """Modal worker entry point with no caller-controlled executable surface."""

    if environ is None:
        environ = os.environ
    if output is None:
        import sys

        output = sys.stdout
    token = environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("The Hugging Face runtime secret is unavailable.")
    spec = _validated_worker_spec(payload)
    _require_before_deadline(spec, now=now)
    if api_factory is None or snapshot is None or marker_download is None:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download

        api_factory = HfApi if api_factory is None else api_factory
        snapshot = snapshot_download if snapshot is None else snapshot
        marker_download = hf_hub_download if marker_download is None else marker_download
    api = api_factory(token=token)
    _verify_output_marker(
        api,
        spec=spec,
        marker_download=marker_download,
        token=token,
    )
    _require_before_deadline(spec, now=now)

    emitter = _EventEmitter(
        job_id=spec.job_id,
        request_hash=spec.request_hash,
        output=output,
    )
    emitter.log(source="system", line="Managed training inputs verified.")
    with tempfile.TemporaryDirectory(prefix="ctrl-pi-training-") as temp:
        root = Path(temp)
        input_root = root / _INPUT_DIR.name
        output_dir = root / _OUTPUT_DIR.name
        dataset_path = input_root / "dataset"
        base_model_path = input_root / "base-model"
        vlm_model_path = input_root / "smolvlm"
        snapshot(
            repo_id=spec.dataset_repo,
            repo_type="dataset",
            revision=spec.dataset_revision,
            local_dir=str(dataset_path),
            token=token,
        )
        _require_before_deadline(spec, now=now)
        snapshot(
            repo_id=spec.base_model,
            repo_type="model",
            revision=spec.base_model_revision,
            local_dir=str(base_model_path),
            token=token,
        )
        _require_before_deadline(spec, now=now)
        _validate_base_model_snapshot(base_model_path)
        _require_before_deadline(spec, now=now)
        snapshot(
            repo_id=SMOLVLA_VLM_REPO,
            repo_type="model",
            revision=SMOLVLA_VLM_REVISION,
            local_dir=str(vlm_model_path),
            allow_patterns=list(_SMOLVLA_VLM_FILES),
            token=token,
        )
        _require_before_deadline(spec, now=now)
        _validate_vlm_snapshot(vlm_model_path)
        _require_before_deadline(spec, now=now)
        _seal_base_model_for_child(
            base_model_path,
            vlm_model_path,
            dataset_path=dataset_path,
        )
        _require_before_deadline(spec, now=now)
        command = build_lerobot_training_command(
            spec,
            dataset_path=dataset_path,
            base_model_path=base_model_path,
            vlm_model_path=vlm_model_path,
            output_dir=output_dir,
        )
        child_environment = _safe_child_environment(environ)
        child_environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "PYTHONUNBUFFERED": "1",
            }
        )
        _require_before_deadline(spec, now=now)
        process: Any | None = None
        readers: list[threading.Thread] = []
        messages: queue.Queue[tuple[str, str | None]] = queue.Queue(maxsize=1024)
        pipe_overflowed = threading.Event()
        try:
            process = popen_factory(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=child_environment,
                shell=False,
                start_new_session=True,
            )
            if process.stdout is None or process.stderr is None:
                raise RuntimeError("The LeRobot process streams are unavailable.")
            readers = [
                threading.Thread(
                    target=_read_pipe,
                    args=(process.stdout, "stdout", messages, pipe_overflowed),
                    daemon=True,
                ),
                threading.Thread(
                    target=_read_pipe,
                    args=(process.stderr, "stderr", messages, pipe_overflowed),
                    daemon=True,
                ),
            ]
            for reader in readers:
                reader.start()
            uploaded: set[int] = set()
            metric_parser = _TrainingMetricParser(
                log_every=spec.log_every,
                max_steps=spec.max_steps,
            )
            while (
                process.poll() is None
                or any(reader.is_alive() for reader in readers)
                or not messages.empty()
            ):
                _require_before_deadline(spec, now=now)
                try:
                    source, raw_line = messages.get(timeout=0.5)
                except queue.Empty:
                    source, raw_line = "", ""
                if raw_line:
                    safe = _safe_line(raw_line, secret=token)
                    if safe is not None:
                        step, metrics = metric_parser.parse(
                            safe,
                            transport_truncated=pipe_overflowed.is_set(),
                        )
                        emitter.log(source=source, line=safe, step=step)
                        if step is not None and metrics:
                            emitter.emit(
                                "metric",
                                {"step": step, "metrics": metrics},
                            )
                for step, policy_dir in _checkpoint_dirs(output_dir):
                    if step in uploaded or step == spec.max_steps:
                        continue
                    _require_before_deadline(spec, now=now)
                    revision = _upload_policy(
                        api,
                        spec=spec,
                        policy_dir=policy_dir,
                        step=step,
                        final=False,
                        vlm_model_path=vlm_model_path,
                        marker_download=marker_download,
                        token=token,
                    )
                    uploaded.add(step)
                    emitter.emit(
                        "checkpoint",
                        {
                            "repo_id": spec.output_model_repo,
                            "revision": revision,
                            "step": step,
                            "final": False,
                        },
                        essential=True,
                    )
            return_code = process.wait(timeout=5.0)
            if pipe_overflowed.is_set():
                emitter.truncated = True
        finally:
            if process is not None and process.poll() is None:
                _terminate_process_tree(process)
            if process is not None:
                for pipe in (getattr(process, "stdout", None), getattr(process, "stderr", None)):
                    if pipe is not None:
                        try:
                            pipe.close()
                        except BaseException:
                            pass
            for reader in readers:
                reader.join(timeout=2.0)
        if return_code != 0:
            raise RuntimeError("The LeRobot training process failed.")
        _require_before_deadline(spec, now=now)
        final_candidates = [
            (step, path)
            for step, path in _checkpoint_dirs(output_dir)
            if step == spec.max_steps
        ]
        if len(final_candidates) != 1:
            raise RuntimeError("The final LeRobot checkpoint is unavailable.")
        final_step, final_dir = final_candidates[0]
        _require_before_deadline(spec, now=now)
        final_revision = _upload_policy(
            api,
            spec=spec,
            policy_dir=final_dir,
            step=final_step,
            final=True,
            vlm_model_path=vlm_model_path,
            marker_download=marker_download,
            token=token,
        )
        _require_before_deadline(spec, now=now)
        emitter.emit(
            "checkpoint",
            {
                "repo_id": spec.output_model_repo,
                "revision": final_revision,
                "step": final_step,
                "final": True,
            },
            essential=True,
        )
        if emitter.truncated:
            emitter.emit(
                "log",
                {
                    "source": "system",
                    "line": "Managed training event output was bounded and may contain gaps.",
                    "step": final_step,
                },
                essential=True,
            )
        return {
            "schema": MODAL_TRAINING_RESULT_SCHEMA,
            "version": MODAL_TRAINING_PROTOCOL_VERSION,
            "job_id": str(spec.job_id),
            "request_hash": spec.request_hash,
            "output_model_repo": spec.output_model_repo,
            "revision": final_revision,
            "step": final_step,
            "last_sequence": emitter.sequence,
            "events_truncated": emitter.truncated,
        }


def run_managed_training_modal(payload: object) -> dict[str, object]:
    """Credential-safe Modal boundary; raw dependency failures never escape."""

    try:
        return run_managed_training(payload)
    except Exception:
        raise RuntimeError("Managed training failed safely.") from None


def build_modal_training_workload(
    *,
    spec: ManagedTrainingSpec,
    hf_token: str,
    modal_module: _ModalModule | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> ModalTrainingWorkload:
    """Build, but never deploy, one bounded revision-pinned training App."""

    # The frozen dataclass performs all range and exact-identity validation.
    _validated_worker_spec(managed_training_spec_payload(spec))
    if not hf_token.strip():
        raise ManagedTrainingConfigurationError(
            "Hugging Face credentials are required for managed training."
        )
    gpu_type, gpu_count = training_gpu_spec(spec.compute_size)
    gpu = gpu_type if gpu_count == 1 else f"{gpu_type}:{gpu_count}"
    current = now()
    if current.tzinfo is None or current.utcoffset() is None:
        raise ManagedTrainingConfigurationError(
            "The managed training clock is invalid."
        )
    remaining_seconds = math.ceil((spec.deadline_at - current).total_seconds())
    if remaining_seconds <= 0:
        raise ManagedTrainingConfigurationError(
            "The managed training deadline has already passed."
        )
    function_timeout = min(spec.timeout_seconds, remaining_seconds, 86_400)
    if modal_module is None:
        import modal

        modal_module = modal
    runtime_secret = modal_module.Secret.from_dict({"HF_TOKEN": hf_token})
    image: _Image = modal_module.Image.debian_slim(
        python_version="3.11"
    ).apt_install(
        "ffmpeg",
        "git",
        "build-essential",
        "linux-libc-dev",
    ).uv_pip_install(*MODAL_TRAINING_PACKAGES)
    image = image.add_local_python_source("ctrl_pi", copy=True)
    app = modal_module.App(
        spec.app_name,
        image=image,
        tags={MODAL_TRAINING_OWNERSHIP_TAG_KEY: str(spec.job_id)},
    )
    training_function = app.function(
        name=MODAL_TRAINING_FUNCTION,
        image=image,
        gpu=gpu,
        secrets=[runtime_secret],
        min_containers=0,
        buffer_containers=0,
        max_containers=1,
        scaledown_window=60,
        timeout=function_timeout,
        retries=0,
        single_use_containers=True,
        restrict_modal_access=True,
    )(run_managed_training_modal)
    return ModalTrainingWorkload(app=app, training_function=training_function)


__all__ = [
    "MAX_WORKER_EVENT_BYTES",
    "MAX_WORKER_EVENT_LINE_BYTES",
    "MAX_WORKER_EVENTS",
    "MAX_WORKER_LOG_EVENTS",
    "MANAGED_SMOLVLA_DEPENDENCY_DIR",
    "MANAGED_SMOLVLA_DEPENDENCY_FILES",
    "MODAL_TRAINING_EVENT_PREFIX",
    "MODAL_TRAINING_EVENT_SCHEMA",
    "MODAL_TRAINING_FUNCTION",
    "MODAL_TRAINING_OWNERSHIP_TAG_KEY",
    "MODAL_TRAINING_PACKAGES",
    "MODAL_TRAINING_PROTOCOL_VERSION",
    "MODAL_TRAINING_RESULT_SCHEMA",
    "SMOLVLA_VLM_REPO",
    "SMOLVLA_VLM_REVISION",
    "ModalTrainingWorkload",
    "build_lerobot_training_command",
    "build_modal_training_workload",
    "managed_training_spec_payload",
    "run_managed_training",
    "run_managed_training_modal",
]
