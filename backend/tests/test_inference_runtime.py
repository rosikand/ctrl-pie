from __future__ import annotations

import base64
import io
import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import torch
from PIL import Image
from pydantic import ValidationError

from ctrl_pi.inference_runtime import (
    ActionChunk,
    EncodedImage,
    MAX_VECTOR_SCALARS,
    ObservationRequest,
    OpenPIInferenceRuntime,
    RuntimeConfigurationError,
    RuntimeDescriptor,
    RuntimeFeature,
    RuntimeLoadSpec,
    RuntimeNotLoadedError,
    RuntimeProtocolError,
    RuntimeUnavailableError,
    StubInferenceRuntime,
    validate_action_chunk,
)
from ctrl_pi.runtime_lerobot import LeRobotRuntime

REPO_ID = "acme/tiny-act"
REVISION = "a" * 40


@pytest.fixture
def offline_lerobot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


def _spec(path: Path | None = None, *, actions_per_chunk: int = 4) -> RuntimeLoadSpec:
    return RuntimeLoadSpec(
        model_repo=REPO_ID,
        revision=REVISION,
        local_model_path=path,
        device="cpu",
        actions_per_chunk=actions_per_chunk,
    )


def _request(
    *,
    vectors: dict[str, list[float]] | None = None,
    images: dict[str, EncodedImage] | None = None,
    sequence: int = 3,
) -> ObservationRequest:
    return ObservationRequest(
        request_id=uuid.uuid4(),
        sequence=sequence,
        captured_at=datetime.now(UTC),
        vectors=vectors
        or {
            "observation.state": [0.0, -0.2, 0.4, 0.1, 0.0, 0.2, 0.5],
        },
        images=images or {},
        task="move the mock arm",
    )


@pytest.fixture
def tiny_act_snapshot(tmp_path: Path) -> Path:
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE

    config = ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(2,)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        },
        device="cpu",
        chunk_size=4,
        n_action_steps=4,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        use_vae=False,
        pretrained_backbone_weights=None,
        normalization_mapping={
            "STATE": NormalizationMode.IDENTITY,
            "ENV": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
            "VISUAL": NormalizationMode.IDENTITY,
        },
    )
    policy = ACTPolicy(config).eval()
    preprocessor, postprocessor = make_pre_post_processors(config)
    policy.save_pretrained(tmp_path)
    preprocessor.save_pretrained(
        tmp_path,
        push_to_hub=False,
        config_filename="policy_preprocessor.json",
    )
    postprocessor.save_pretrained(
        tmp_path,
        push_to_hub=False,
        config_filename="policy_postprocessor.json",
    )
    (tmp_path / ".ctrl-pi-runtime.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "model_repo": REPO_ID,
                "revision": REVISION,
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


@pytest.mark.parametrize("bad_value", [True, "1.25", None])
def test_observation_rejects_coerced_or_non_numeric_values(bad_value: object) -> None:
    payload = {
        "request_id": str(uuid.uuid4()),
        "sequence": 1,
        "captured_at": datetime.now(UTC).isoformat(),
        "vectors": {"observation.state": [bad_value]},
    }
    with pytest.raises(ValidationError):
        ObservationRequest.model_validate(payload)


def test_observations_and_actions_reject_huge_finite_scalars() -> None:
    observation = _request().model_dump(mode="json")
    observation["vectors"] = {"observation.state": [1e308]}
    with pytest.raises(ValidationError, match="finite"):
        ObservationRequest.model_validate(observation)

    now = datetime.now(UTC)
    with pytest.raises(ValidationError, match="finite"):
        ActionChunk(
            request_id=uuid.uuid4(),
            observation_sequence=1,
            revision=REVISION,
            actions=[[1e308]],
            server_received_at=now,
            server_completed_at=now,
        )


@pytest.mark.parametrize("bad_value", [True, "1.25", None])
def test_action_chunk_rejects_coerced_or_non_numeric_values(bad_value: object) -> None:
    now = datetime.now(UTC)
    payload = {
        "request_id": str(uuid.uuid4()),
        "observation_sequence": 1,
        "revision": REVISION,
        "actions": [[bad_value]],
        "server_received_at": now.isoformat(),
        "server_completed_at": now.isoformat(),
    }
    with pytest.raises(ValidationError):
        ActionChunk.model_validate(payload)


def test_sequence_shape_and_runtime_spec_are_strict() -> None:
    payload = _request().model_dump(mode="json")
    payload["sequence"] = True
    with pytest.raises(ValidationError):
        ObservationRequest.model_validate(payload)
    with pytest.raises(ValidationError):
        RuntimeFeature(name="state", kind="state", shape=("7",))
    with pytest.raises(RuntimeConfigurationError):
        RuntimeLoadSpec(
            model_repo=REPO_ID,
            revision=REVISION,
            local_model_path=None,
            device="cpu",
            actions_per_chunk=True,
        )


def test_descriptor_rejects_an_impossible_aggregate_vector_shape() -> None:
    features = [
        RuntimeFeature(name=f"observation.state_{index}", kind="state", shape=(4096,))
        for index in range(MAX_VECTOR_SCALARS // 4096 + 1)
    ]
    with pytest.raises(ValidationError, match="scalar limit"):
        RuntimeDescriptor(
            runtime="stub",
            policy_type="stub",
            model_repo=REPO_ID,
            revision=REVISION,
            inputs=features,
            action=RuntimeFeature(name="action", kind="action", shape=(7,)),
            actions_per_chunk=4,
        )


def test_encoded_image_rejects_invalid_or_oversized_decoded_content() -> None:
    with pytest.raises(ValidationError):
        EncodedImage(data_base64="not-base64")
    oversized = base64.b64encode(b"x" * (3 * 1024 * 1024 + 1)).decode("ascii")
    with pytest.raises(ValidationError):
        EncodedImage(data_base64=oversized)


def test_stub_runtime_is_deterministic_and_can_emulate_openpi() -> None:
    runtime = StubInferenceRuntime(runtime="openpi")
    descriptor = runtime.load(_spec())
    request = _request(sequence=9)

    first = runtime.predict(request)
    second = runtime.predict(request)

    assert descriptor.runtime == runtime.kind == "openpi"
    assert descriptor.policy_type == "openpi-stub"
    assert first.actions == second.actions
    assert first.request_id == request.request_id
    assert first.observation_sequence == request.sequence
    assert first.revision == REVISION
    assert len(first.actions) == descriptor.actions_per_chunk == 4
    assert all(len(action) == 7 for action in first.actions)

    runtime.close()
    with pytest.raises(RuntimeNotLoadedError):
        runtime.describe()


def test_stub_runtime_validates_exact_features_and_one_loaded_identity() -> None:
    runtime = StubInferenceRuntime()
    runtime.load(_spec())
    with pytest.raises(RuntimeProtocolError, match="features"):
        runtime.predict(_request(vectors={"wrong": [0.0] * 7}))
    with pytest.raises(RuntimeConfigurationError, match="another policy"):
        runtime.load(
            RuntimeLoadSpec(
                model_repo="acme/other",
                revision="b" * 40,
                local_model_path=None,
                device="cpu",
                actions_per_chunk=4,
            )
        )


def test_stub_runtime_session_reset_requires_a_loaded_policy() -> None:
    runtime = StubInferenceRuntime()
    with pytest.raises(RuntimeNotLoadedError):
        runtime.reset_session()
    runtime.load(_spec())
    runtime.reset_session()


def test_real_openpi_runtime_is_explicitly_unavailable() -> None:
    runtime = OpenPIInferenceRuntime()
    with pytest.raises(RuntimeUnavailableError, match="not available"):
        runtime.load(_spec())
    with pytest.raises(RuntimeUnavailableError, match="not available"):
        runtime.describe()
    with pytest.raises(RuntimeUnavailableError, match="not available"):
        runtime.reset_session()
    runtime.close()


def test_action_chunk_requires_exact_request_revision_and_shape() -> None:
    descriptor = StubInferenceRuntime().load(_spec())
    request = _request()
    now = datetime.now(UTC)
    wrong = ActionChunk(
        request_id=uuid.uuid4(),
        observation_sequence=request.sequence,
        revision=REVISION,
        actions=[[0.0] * 7],
        server_received_at=now,
        server_completed_at=now + timedelta(milliseconds=1),
    )
    with pytest.raises(RuntimeProtocolError, match="does not match"):
        validate_action_chunk(request, wrong, descriptor)


def test_lerobot_runtime_loads_and_runs_real_tiny_act_snapshot(
    tiny_act_snapshot: Path,
    offline_lerobot: None,
) -> None:
    del offline_lerobot
    runtime = LeRobotRuntime()
    descriptor = runtime.load(_spec(tiny_act_snapshot))
    request = _request(
        vectors={
            "observation.state": [0.0] * 7,
            "observation.environment_state": [0.25, -0.25],
        }
    )

    chunk = runtime.predict(request)

    assert descriptor.runtime == "lerobot"
    assert descriptor.policy_type == "act"
    assert [(feature.name, feature.shape) for feature in descriptor.inputs] == [
        ("observation.state", (7,)),
        ("observation.environment_state", (2,)),
    ]
    assert descriptor.action.shape == (7,)
    assert len(chunk.actions) == 4
    assert all(len(action) == 7 for action in chunk.actions)
    assert all(torch.isfinite(torch.tensor(action)).all() for action in chunk.actions)
    assert runtime.describe() == descriptor
    runtime.reset_session()
    runtime.close()


def test_lerobot_runtime_rejects_mismatched_marker_without_raw_context(
    tiny_act_snapshot: Path,
    offline_lerobot: None,
) -> None:
    del offline_lerobot
    (tiny_act_snapshot / ".ctrl-pi-runtime.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "model_repo": REPO_ID,
                "revision": "b" * 40,
            }
        ),
        encoding="utf-8",
    )
    runtime = LeRobotRuntime()

    with pytest.raises(RuntimeConfigurationError) as captured:
        runtime.load(_spec(tiny_act_snapshot))

    assert "tiny_act_snapshot" not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    with pytest.raises(RuntimeNotLoadedError):
        runtime.describe()


def test_lerobot_runtime_fails_closed_without_both_offline_flags(
    tiny_act_snapshot: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    runtime = LeRobotRuntime()

    with pytest.raises(RuntimeConfigurationError, match="offline Hub mode"):
        runtime.load(_spec(tiny_act_snapshot))

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(RuntimeConfigurationError, match="offline Hub mode"):
        runtime.load(_spec(tiny_act_snapshot))


def _encoded_image(
    *,
    image_format: str = "JPEG",
    size: tuple[int, int] = (16, 12),
) -> EncodedImage:
    image = Image.new("RGB", size, (30, 80, 120))
    output = io.BytesIO()
    image.save(output, format=image_format)
    return EncodedImage(data_base64=base64.b64encode(output.getvalue()).decode("ascii"))


def test_lerobot_image_decode_is_bounded_batched_and_deterministically_resized() -> None:
    feature = RuntimeFeature(
        name="observation.images.camera",
        kind="visual",
        shape=(3, 8, 10),
    )
    tensor = LeRobotRuntime._decode_image(_encoded_image(), feature)

    assert tensor.shape == (1, 3, 8, 10)
    assert tensor.dtype is torch.float32
    assert 0.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0

    with pytest.raises(RuntimeProtocolError, match="valid JPEG"):
        LeRobotRuntime._decode_image(_encoded_image(image_format="PNG"), feature)


def test_lerobot_image_decode_rejects_pixel_bomb_before_array_allocation() -> None:
    feature = RuntimeFeature(
        name="observation.images.camera",
        kind="visual",
        shape=(3, 8, 8),
    )
    oversized = _encoded_image(size=(2049, 1))
    with pytest.raises(RuntimeProtocolError, match="dimensions"):
        LeRobotRuntime._decode_image(oversized, feature)
