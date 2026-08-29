from __future__ import annotations

import io
import json
import math
import os
import threading
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from ctrl_pi.inference_runtime import (
    MAX_ABSOLUTE_SCALAR,
    MAX_IMAGE_PIXELS,
    MAX_IMAGE_SIDE,
    ActionChunk,
    InferenceRuntimeError,
    ObservationRequest,
    RuntimeConfigurationError,
    RuntimeDescriptor,
    RuntimeFeature,
    RuntimeKind,
    RuntimeLoadSpec,
    RuntimeNotLoadedError,
    RuntimeProtocolError,
    validate_action_chunk,
    validate_observation,
)
from ctrl_pi.training_compute import (
    MANAGED_SMOLVLA_DEPENDENCY_DIR,
    MANAGED_SMOLVLA_DEPENDENCY_FILES,
)

_MARKER_NAME = ".ctrl-pi-runtime.json"
_MARKER_MAX_BYTES = 4096
_OFFLINE_TRUE = {"1", "ON", "TRUE", "YES"}
_REQUIRED_POLICY_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
)
_MAX_SMOLVLA_DEPENDENCY_FILE_BYTES = 8 * 1024 * 1024


class LeRobotRuntime:
    """Local-only LeRobot 0.4.4 policy adapter for a pinned model snapshot."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._spec: RuntimeLoadSpec | None = None
        self._descriptor: RuntimeDescriptor | None = None
        self._policy: Any = None
        self._preprocessor: Any = None
        self._postprocessor: Any = None

    @property
    def kind(self) -> RuntimeKind:
        return "lerobot"

    def load(self, spec: RuntimeLoadSpec) -> RuntimeDescriptor:
        with self._lock:
            if self._spec is not None:
                if self._spec != spec:
                    raise RuntimeConfigurationError(
                        "The LeRobot runtime already loaded another policy."
                    )
                return self.describe()

            if not self._offline_mode_enabled():
                raise RuntimeConfigurationError(
                    "The LeRobot serving runtime requires offline Hub mode."
                )

            loaded: tuple[Any, Any, Any, RuntimeDescriptor] | None = None
            try:
                model_path = self._verified_model_path(spec)
                loaded = self._load_local_policy(spec, model_path)
            except Exception:
                loaded = None
            if loaded is None:
                self._clear()
                raise RuntimeConfigurationError(
                    "The local LeRobot policy could not be loaded safely."
                )

            policy, preprocessor, postprocessor, descriptor = loaded
            self._policy = policy
            self._preprocessor = preprocessor
            self._postprocessor = postprocessor
            self._descriptor = descriptor
            self._spec = spec
            return descriptor.model_copy(deep=True)

    @staticmethod
    def _offline_mode_enabled() -> bool:
        return all(
            os.environ.get(name, "").strip().upper() in _OFFLINE_TRUE
            for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
        )

    def describe(self) -> RuntimeDescriptor:
        with self._lock:
            if self._descriptor is None:
                raise RuntimeNotLoadedError("The inference runtime is not loaded.")
            return self._descriptor.model_copy(deep=True)

    def predict(self, request: ObservationRequest) -> ActionChunk:
        received_at = datetime.now(UTC)
        with self._lock:
            descriptor = self.describe()
            validate_observation(request, descriptor)
            result: list[list[float]] | None = None
            try:
                observation = self._prepare_observation(request, descriptor)
                with torch.inference_mode():
                    processed = self._preprocessor(observation)
                    raw_chunk = self._policy.predict_action_chunk(processed)
                    result = self._postprocess_chunk(raw_chunk, descriptor)
            except RuntimeProtocolError:
                raise
            except Exception:
                result = None
            if result is None:
                raise InferenceRuntimeError(
                    "LeRobot policy inference failed safely."
                )

            chunk = ActionChunk(
                request_id=request.request_id,
                observation_sequence=request.sequence,
                revision=descriptor.revision,
                actions=result,
                server_received_at=received_at,
                server_completed_at=datetime.now(UTC),
            )
            validate_action_chunk(request, chunk, descriptor)
            return chunk

    def reset_session(self) -> None:
        with self._lock:
            if self._policy is None:
                raise RuntimeNotLoadedError("The inference runtime is not loaded.")
            failed = False
            try:
                self._policy.reset()
            except Exception:
                failed = True
            if failed:
                raise InferenceRuntimeError(
                    "The LeRobot policy session could not be reset safely."
                )

    def close(self) -> None:
        with self._lock:
            self._clear()

    def _clear(self) -> None:
        policy = self._policy
        self._policy = None
        self._preprocessor = None
        self._postprocessor = None
        self._descriptor = None
        self._spec = None
        if policy is not None:
            try:
                policy.to("cpu")
            except Exception:
                pass

    @staticmethod
    def _verified_model_path(spec: RuntimeLoadSpec) -> Path:
        if spec.local_model_path is None:
            raise RuntimeConfigurationError(
                "LeRobot requires a local pinned model snapshot."
            )
        try:
            root = spec.local_model_path.resolve(strict=True)
        except (OSError, RuntimeError):
            raise RuntimeConfigurationError(
                "The local LeRobot model snapshot is unavailable."
            ) from None
        if not root.is_dir():
            raise RuntimeConfigurationError(
                "The local LeRobot model snapshot is unavailable."
            )

        marker = root / _MARKER_NAME
        try:
            if marker.is_symlink() or not marker.is_file():
                raise OSError
            if marker.stat().st_size > _MARKER_MAX_BYTES:
                raise OSError
            marker_payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise RuntimeConfigurationError(
                "The local LeRobot model identity marker is invalid."
            ) from None
        if marker_payload != {
            "schema": 1,
            "model_repo": spec.model_repo,
            "revision": spec.revision,
        }:
            raise RuntimeConfigurationError(
                "The local LeRobot model identity does not match the deployment."
            )

        for filename in _REQUIRED_POLICY_FILES:
            candidate = root / filename
            try:
                resolved = candidate.resolve(strict=True)
            except (OSError, RuntimeError):
                raise RuntimeConfigurationError(
                    "The local LeRobot model snapshot is incomplete."
                ) from None
            if not resolved.is_file() or not resolved.is_relative_to(root):
                raise RuntimeConfigurationError(
                    "The local LeRobot model snapshot is invalid."
                )
        return root

    @staticmethod
    def _load_local_policy(
        spec: RuntimeLoadSpec,
        model_path: Path,
    ) -> tuple[Any, Any, Any, RuntimeDescriptor]:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.configs.types import FeatureType
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        config = PreTrainedConfig.from_pretrained(
            model_path,
            local_files_only=True,
        )
        policy_type = str(config.type)
        smolvlm_path: Path | None = None
        if policy_type == "smolvla":
            smolvlm_path = LeRobotRuntime._verified_smolvlm_dependency(model_path)
            config.vlm_model_name = str(smolvlm_path)
            config.load_vlm_weights = False
            config.use_peft = False
        policy_class = get_policy_class(policy_type)
        policy = policy_class.from_pretrained(
            model_path,
            config=config,
            local_files_only=True,
            strict=True,
        )
        policy.to(spec.device)
        policy.eval()
        policy.reset()
        device_override = {"device": spec.device}
        preprocessor_overrides: dict[str, dict[str, str]] = {
            "device_processor": device_override,
        }
        if smolvlm_path is not None:
            preprocessor_overrides["tokenizer_processor"] = {
                "tokenizer_name": str(smolvlm_path),
            }
        preprocessor, postprocessor = make_pre_post_processors(
            config,
            pretrained_path=str(model_path),
            preprocessor_overrides=preprocessor_overrides,
            postprocessor_overrides={
                "device_processor": device_override,
            },
        )

        feature_kinds = {
            FeatureType.STATE: "state",
            FeatureType.ENV: "environment",
            FeatureType.VISUAL: "visual",
            FeatureType.LANGUAGE: "language",
        }
        inputs: list[RuntimeFeature] = []
        for name, feature in (config.input_features or {}).items():
            kind = feature_kinds.get(feature.type)
            if kind is None:
                raise RuntimeConfigurationError(
                    "The LeRobot policy has an unsupported input feature."
                )
            inputs.append(
                RuntimeFeature(
                    name=name,
                    kind=kind,
                    shape=tuple(feature.shape),
                )
            )

        action_features = [
            (name, feature)
            for name, feature in (config.output_features or {}).items()
            if feature.type is FeatureType.ACTION
        ]
        if len(action_features) != 1:
            raise RuntimeConfigurationError(
                "The LeRobot policy must expose exactly one action feature."
            )
        action_name, action_feature = action_features[0]
        configured_chunk = getattr(config, "chunk_size", spec.actions_per_chunk)
        if isinstance(configured_chunk, bool) or not isinstance(configured_chunk, int):
            configured_chunk = spec.actions_per_chunk
        actions_per_chunk = min(spec.actions_per_chunk, max(configured_chunk, 1))
        descriptor = RuntimeDescriptor(
            runtime="lerobot",
            policy_type=policy_type,
            model_repo=spec.model_repo,
            revision=spec.revision,
            inputs=inputs,
            action=RuntimeFeature(
                name=action_name,
                kind="action",
                shape=tuple(action_feature.shape),
            ),
            actions_per_chunk=actions_per_chunk,
        )
        return policy, preprocessor, postprocessor, descriptor

    @staticmethod
    def _verified_smolvlm_dependency(model_path: Path) -> Path:
        dependency = model_path / MANAGED_SMOLVLA_DEPENDENCY_DIR
        try:
            if dependency.is_symlink():
                raise OSError
            resolved_dependency = dependency.resolve(strict=True)
            if not resolved_dependency.is_dir() or not resolved_dependency.is_relative_to(
                model_path
            ):
                raise OSError
            children = list(resolved_dependency.iterdir())
        except (OSError, RuntimeError):
            raise RuntimeConfigurationError(
                "The local SmolVLA dependency bundle is unavailable."
            ) from None
        if {child.name for child in children} != set(MANAGED_SMOLVLA_DEPENDENCY_FILES):
            raise RuntimeConfigurationError(
                "The local SmolVLA dependency bundle is invalid."
            )
        for filename in MANAGED_SMOLVLA_DEPENDENCY_FILES:
            candidate = resolved_dependency / filename
            try:
                if candidate.is_symlink():
                    raise OSError
                resolved = candidate.resolve(strict=True)
                metadata = resolved.stat()
            except (OSError, RuntimeError):
                raise RuntimeConfigurationError(
                    "The local SmolVLA dependency bundle is invalid."
                ) from None
            if (
                not resolved.is_file()
                or not resolved.is_relative_to(resolved_dependency)
                or not 0 < metadata.st_size <= _MAX_SMOLVLA_DEPENDENCY_FILE_BYTES
            ):
                raise RuntimeConfigurationError(
                    "The local SmolVLA dependency bundle is invalid."
                )
        return resolved_dependency

    def _prepare_observation(
        self,
        request: ObservationRequest,
        descriptor: RuntimeDescriptor,
    ) -> dict[str, Any]:
        observation: dict[str, Any] = {
            name: torch.tensor(values, dtype=torch.float32).unsqueeze(0)
            for name, values in request.vectors.items()
        }
        for feature in descriptor.inputs:
            if feature.kind != "visual":
                continue
            observation[feature.name] = self._decode_image(
                request.images[feature.name],
                feature,
            )
        if request.task or any(
            feature.kind == "language" for feature in descriptor.inputs
        ):
            observation["task"] = request.task
        return observation

    @staticmethod
    def _decode_image(encoded: Any, feature: RuntimeFeature) -> torch.Tensor:
        data = encoded.decoded()
        loaded: tuple[str | None, tuple[int, int], np.ndarray] | None = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(data)) as source:
                    image_format = source.format
                    width, height = source.size
                    if (
                        width <= 0
                        or height <= 0
                        or width > MAX_IMAGE_SIDE
                        or height > MAX_IMAGE_SIDE
                        or width * height > MAX_IMAGE_PIXELS
                    ):
                        raise RuntimeProtocolError("The image dimensions exceed the safe limit.")
                    channels, target_height, target_width = feature.shape
                    mode = {1: "L", 3: "RGB", 4: "RGBA"}[channels]
                    image = source.convert(mode)
                    if image.size != (target_width, target_height):
                        image = image.resize(
                            (target_width, target_height),
                            Image.Resampling.BILINEAR,
                        )
                    loaded = image_format, image.size, np.asarray(image).copy()
        except RuntimeProtocolError:
            raise
        except (
            OSError,
            UnidentifiedImageError,
            ValueError,
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
        ):
            loaded = None
        if loaded is None or loaded[0] != "JPEG":
            raise RuntimeProtocolError("The observation image is not a valid JPEG.")

        _, size, array = loaded
        channels, target_height, target_width = feature.shape
        if size != (target_width, target_height):
            raise RuntimeProtocolError("The observation image shape is invalid.")
        tensor = torch.from_numpy(array)
        if channels == 1:
            tensor = tensor.unsqueeze(-1)
        if tensor.shape != (target_height, target_width, channels):
            raise RuntimeProtocolError("The observation image channels are invalid.")
        return (
            tensor.permute(2, 0, 1)
            .to(dtype=torch.float32)
            .div_(255.0)
            .contiguous()
            .unsqueeze(0)
        )

    def _postprocess_chunk(
        self,
        raw_chunk: Any,
        descriptor: RuntimeDescriptor,
    ) -> list[list[float]]:
        if not isinstance(raw_chunk, torch.Tensor):
            raise RuntimeProtocolError("The policy returned an invalid action tensor.")
        if raw_chunk.ndim == 2:
            raw_chunk = raw_chunk.unsqueeze(0)
        if raw_chunk.ndim != 3 or raw_chunk.shape[0] != 1:
            raise RuntimeProtocolError("The policy returned an invalid action chunk shape.")
        if raw_chunk.shape[1] < 1 or raw_chunk.shape[2] != descriptor.action.shape[0]:
            raise RuntimeProtocolError("The policy returned an invalid action chunk shape.")

        raw_chunk = raw_chunk[:, : descriptor.actions_per_chunk, :]
        actions: list[list[float]] = []
        for index in range(raw_chunk.shape[1]):
            processed = self._postprocessor(raw_chunk[:, index, :])
            if not isinstance(processed, torch.Tensor):
                raise RuntimeProtocolError("The policy postprocessor returned invalid actions.")
            if processed.ndim == 2 and processed.shape[0] == 1:
                processed = processed.squeeze(0)
            if processed.ndim != 1 or processed.shape[0] != descriptor.action.shape[0]:
                raise RuntimeProtocolError("The policy postprocessor returned an invalid shape.")
            action = processed.detach().to("cpu").tolist()
            if any(
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or abs(float(value)) > MAX_ABSOLUTE_SCALAR
                for value in action
            ):
                raise RuntimeProtocolError(
                    "The policy postprocessor returned invalid actions."
                )
            actions.append([float(value) for value in action])
        return actions
