from __future__ import annotations

import io
import json
import queue
import threading
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import ANY

import modal
import pytest

from ctrl_pi.modal_training_workload import (
    MODAL_TRAINING_EVENT_PREFIX,
    MODAL_TRAINING_OWNERSHIP_TAG_KEY,
    MODAL_TRAINING_PACKAGES,
    MANAGED_SMOLVLA_DEPENDENCY_DIR,
    SMOLVLA_VLM_REPO,
    SMOLVLA_VLM_REVISION,
    _TrainingMetricParser,
    _SMOLVLA_VLM_FILES,
    _checkpoint_dirs,
    _parse_metrics,
    _prepare_policy_for_upload,
    _read_pipe,
    _seal_base_model_for_child,
    build_lerobot_training_command,
    build_modal_training_workload,
    managed_training_spec_payload,
    run_managed_training,
    run_managed_training_modal,
)
from ctrl_pi.training_compute import (
    MANAGED_TRAINING_MARKER_PATH,
    ManagedTrainingConfigurationError,
    ManagedTrainingSpec,
    managed_training_marker,
    training_app_name,
    training_ownership_tag,
)


JOB_ID = uuid.UUID("33333333-3333-4333-8333-333333333333")
REQUEST_HASH = "a" * 64
DATASET_REVISION = "b" * 40
BASE_REVISION = "c" * 40
MARKER_REVISION = "d" * 40
INTERMEDIATE_REVISION = "e" * 40
FINAL_REVISION = "f" * 40
HF_SECRET = "hf_training_secret_123456789"


def _safe_policy_config() -> dict[str, object]:
    return {
        "type": "smolvla",
        "device": "cpu",
        "use_peft": False,
        "vlm_model_name": SMOLVLA_VLM_REPO,
        "load_vlm_weights": True,
        "max_state_dim": 32,
        "max_action_dim": 32,
        "empty_cameras": 0,
        # Match the immutable official smolvla_base shape that is deliberately
        # incompatible with ctrl-pi recordings until the worker seals it.
        "input_features": {
            "observation.state": {"type": "STATE", "shape": [6]},
            "observation.images.camera1": {"type": "VISUAL", "shape": [3, 256, 256]},
            "observation.images.camera2": {"type": "VISUAL", "shape": [3, 256, 256]},
            "observation.images.camera3": {"type": "VISUAL", "shape": [3, 256, 256]},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [6]}},
    }


def _ctrl_pi_dataset_features() -> dict[str, dict[str, object]]:
    joint_names = [
        "shoulder_yaw",
        "shoulder_pitch",
        "elbow_pitch",
        "wrist_roll",
        "wrist_pitch",
        "wrist_yaw",
        "gripper",
    ]
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": [7],
            "names": joint_names,
        },
        "action": {
            "dtype": "float32",
            "shape": [7],
            "names": joint_names,
        },
        "observation.images.workspace": {
            "dtype": "video",
            "shape": [64, 96, 3],
            "names": ["height", "width", "channels"],
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }


def _write_safe_dataset(path: Path) -> None:
    (path / "meta").mkdir(parents=True, exist_ok=True)
    (path / "meta" / "info.json").write_text(
        json.dumps({"features": _ctrl_pi_dataset_features()}),
        encoding="utf-8",
    )


def _write_safe_base_model(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(
        json.dumps(_safe_policy_config()), encoding="utf-8"
    )
    (path / "model.safetensors").write_bytes(b"safe tensor fixture")
    (path / "policy_preprocessor.json").write_text(
        json.dumps(
            {
                "name": "policy_preprocessor",
                "steps": [
                    {
                        "registry_name": "rename_observations_processor",
                        "config": {"rename_map": {}},
                    },
                    {"registry_name": "to_batch_processor", "config": {}},
                    {"registry_name": "smolvla_new_line_processor", "config": {}},
                    {
                        "registry_name": "tokenizer_processor",
                        "config": {"tokenizer_name": SMOLVLA_VLM_REPO},
                    },
                    {"registry_name": "device_processor", "config": {"device": "cpu"}},
                    {
                        "registry_name": "normalizer_processor",
                        "config": {},
                        "state_file": "preprocessor_state.safetensors",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (path / "policy_postprocessor.json").write_text(
        json.dumps(
            {
                "name": "policy_postprocessor",
                "steps": [
                    {
                        "registry_name": "unnormalizer_processor",
                        "config": {},
                        "state_file": "postprocessor_state.safetensors",
                    },
                    {"registry_name": "device_processor", "config": {"device": "cpu"}},
                ],
            }
        ),
        encoding="utf-8",
    )
    (path / "preprocessor_state.safetensors").write_bytes(b"safe state fixture")
    (path / "postprocessor_state.safetensors").write_bytes(b"safe state fixture")


def _write_safe_vlm_snapshot(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    json_files = {
        "added_tokens.json",
        "chat_template.json",
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    for filename in _SMOLVLA_VLM_FILES:
        (path / filename).write_text(
            "{}" if filename in json_files else "fixed fixture", encoding="utf-8"
        )


def _populate_snapshot(spec: ManagedTrainingSpec, kwargs: dict[str, Any]) -> str:
    path = Path(kwargs["local_dir"])
    path.mkdir(parents=True, exist_ok=True)
    if kwargs["repo_id"] == spec.dataset_repo:
        _write_safe_dataset(path)
    elif kwargs["repo_id"] == spec.base_model:
        _write_safe_base_model(path)
    elif kwargs["repo_id"] == SMOLVLA_VLM_REPO:
        assert kwargs["revision"] == SMOLVLA_VLM_REVISION
        assert set(kwargs["allow_patterns"]) == set(_SMOLVLA_VLM_FILES)
        _write_safe_vlm_snapshot(path)
    return str(path)


def _spec(
    *,
    compute_size: str = "Modal: A10G",
    deadline_at: datetime | None = None,
    timeout_seconds: int = 3_600,
) -> ManagedTrainingSpec:
    return ManagedTrainingSpec(
        job_id=JOB_ID,
        request_hash=REQUEST_HASH,
        app_name=training_app_name(JOB_ID),
        ownership_tag=training_ownership_tag(JOB_ID),
        dataset_repo="owner/dataset",
        dataset_revision=DATASET_REVISION,
        base_model="owner/base-model",
        base_model_revision=BASE_REVISION,
        output_model_repo="owner/output-model",
        output_marker_revision=MARKER_REVISION,
        output_private=True,
        runtime="lerobot",
        max_steps=2,
        batch_size=8,
        log_every=1,
        save_every=1,
        seed=7,
        num_workers=2,
        compute_size=compute_size,  # type: ignore[arg-type]
        timeout_seconds=timeout_seconds,
        deadline_at=deadline_at or datetime.now(UTC) + timedelta(hours=1),
    )


class FakeImage:
    def __init__(self, owner: FakeModal) -> None:
        self.owner = owner

    def apt_install(self, *packages: str) -> FakeImage:
        self.owner.apt_packages = packages
        return self

    def uv_pip_install(self, *packages: str) -> FakeImage:
        self.owner.python_packages = packages
        return self

    def add_local_python_source(self, *modules: str, **kwargs: Any) -> FakeImage:
        self.owner.source = (modules, kwargs)
        return self


class FakeImageAPI:
    def __init__(self, owner: FakeModal) -> None:
        self.owner = owner

    def debian_slim(self, *, python_version: str) -> FakeImage:
        self.owner.python_version = python_version
        return FakeImage(self.owner)


class FakeSecretAPI:
    def __init__(self, owner: FakeModal) -> None:
        self.owner = owner

    def from_dict(self, values: dict[str, str]) -> object:
        self.owner.secret_values = values
        return self.owner.secret


class FakeApp:
    def __init__(self, owner: FakeModal) -> None:
        self.owner = owner

    def function(self, **kwargs: Any):
        self.owner.function_options = kwargs
        return lambda function: function


class FakeModal:
    def __init__(self) -> None:
        self.Image = FakeImageAPI(self)
        self.Secret = FakeSecretAPI(self)
        self.secret = object()
        self.python_version = ""
        self.apt_packages: tuple[str, ...] = ()
        self.python_packages: tuple[str, ...] = ()
        self.source: tuple[tuple[str, ...], dict[str, Any]] | None = None
        self.secret_values: dict[str, str] = {}
        self.app_name = ""
        self.tags: dict[str, str] = {}
        self.function_options: dict[str, Any] = {}

    def App(self, app_name: str, *, image: FakeImage, tags: dict[str, str]) -> FakeApp:
        assert isinstance(image, FakeImage)
        self.app_name = app_name
        self.tags = tags
        return FakeApp(self)


def test_workload_applies_exact_gpu_cost_timeout_and_secret_guardrails() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    spec = _spec(
        compute_size="Modal: 8xH100",
        deadline_at=now + timedelta(seconds=91),
        timeout_seconds=3_600,
    )
    fake = FakeModal()

    workload = build_modal_training_workload(
        spec=spec,
        hf_token=HF_SECRET,
        modal_module=fake,  # type: ignore[arg-type]
        now=lambda: now,
    )

    assert workload.app.owner is fake
    assert workload.training_function is run_managed_training_modal
    assert fake.app_name == training_app_name(JOB_ID)
    assert fake.tags == {MODAL_TRAINING_OWNERSHIP_TAG_KEY: str(JOB_ID)}
    assert fake.python_version == "3.11"
    assert fake.apt_packages == (
        "ffmpeg",
        "git",
        "build-essential",
        "linux-libc-dev",
    )
    assert fake.python_packages == MODAL_TRAINING_PACKAGES
    assert fake.source == (("ctrl_pi",), {"copy": True})
    assert fake.secret_values == {"HF_TOKEN": HF_SECRET}
    assert fake.function_options == {
        "name": "train",
        "image": ANY,
        "gpu": "H100:8",
        "secrets": [fake.secret],
        "min_containers": 0,
        "buffer_containers": 0,
        "max_containers": 1,
        "scaledown_window": 60,
        "timeout": 91,
        "retries": 0,
        "single_use_containers": True,
        "restrict_modal_access": True,
    }
    assert HF_SECRET not in repr(managed_training_spec_payload(spec))


@pytest.mark.parametrize(
    ("compute_size", "expected_gpu", "expected_processes"),
    [
        ("Modal: A10G", "A10G", None),
        ("Modal: A100", "A100", None),
        ("Modal: 2xA100", "A100:2", "2"),
        ("Modal: 4xA100", "A100:4", "4"),
        ("Modal: 8xA100", "A100:8", "8"),
        ("Modal: H100", "H100", None),
        ("Modal: 2xH100", "H100:2", "2"),
        ("Modal: 4xH100", "H100:4", "4"),
        ("Modal: 8xH100", "H100:8", "8"),
    ],
)
def test_every_gpu_label_maps_to_matching_modal_and_accelerate_count(
    compute_size: str,
    expected_gpu: str,
    expected_processes: str | None,
) -> None:
    fake = FakeModal()
    spec = _spec(compute_size=compute_size)

    build_modal_training_workload(
        spec=spec,
        hf_token=HF_SECRET,
        modal_module=fake,  # type: ignore[arg-type]
    )
    command = build_lerobot_training_command(
        spec,
        dataset_path=Path("/dataset"),
        base_model_path=Path("/base"),
        vlm_model_path=Path("/vlm"),
        output_dir=Path("/output"),
    )

    assert fake.function_options["gpu"] == expected_gpu
    if expected_processes is None:
        assert command[:3] == ["python", "-m", "lerobot.scripts.lerobot_train"]
        assert "accelerate" not in command
    else:
        assert command[:7] == [
            "accelerate",
            "launch",
            "--multi_gpu",
            "--num_processes",
            expected_processes,
            "-m",
            "lerobot.scripts.lerobot_train",
        ]
    assert "--policy.push_to_hub" in command
    assert command[command.index("--policy.push_to_hub") + 1] == "false"
    assert command[command.index("--policy.device") + 1] == "cuda"
    assert command[command.index("--policy.use_peft") + 1] == "false"
    assert command[command.index("--policy.vlm_model_name") + 1] == "/vlm"
    assert command[command.index("--policy.load_vlm_weights") + 1] == "false"
    assert "--dataset.revision" in command
    assert command[command.index("--dataset.revision") + 1] == DATASET_REVISION
    assert "--rename_map" not in command


def test_managed_smolvla_derives_and_persists_exact_ctrl_pi_dataset_features(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies import factory as policy_factory
    from lerobot.policies.factory import make_policy
    from lerobot.policies.smolvla.processor_smolvla import (
        make_smolvla_pre_post_processors,
    )
    from lerobot.processor import tokenizer_processor

    dataset_path = tmp_path / "dataset"
    dataset_features = _ctrl_pi_dataset_features()
    policy_dataset_features = {
        name: {
            **feature,
            "shape": tuple(feature["shape"]),
        }
        for name, feature in dataset_features.items()
        if name in {
            "observation.state",
            "action",
            "observation.images.workspace",
        }
    }
    dataset_meta = LeRobotDatasetMetadata.create(
        repo_id="owner/ctrl-pi-yam",
        fps=10,
        features=policy_dataset_features,
        robot_type="yam",
        root=dataset_path,
        use_videos=True,
    )
    base_model_path = tmp_path / "base-model"
    vlm_model_path = tmp_path / "smolvlm"
    _write_safe_base_model(base_model_path)
    _write_safe_vlm_snapshot(vlm_model_path)

    _seal_base_model_for_child(
        base_model_path,
        vlm_model_path,
        dataset_path=dataset_path,
    )
    sealed = json.loads((base_model_path / "config.json").read_text(encoding="utf-8"))
    assert sealed["input_features"] == {}
    assert sealed["output_features"] == {}
    assert sealed["empty_cameras"] == 0

    config = PreTrainedConfig.from_pretrained(base_model_path)
    config.pretrained_path = base_model_path

    class LightweightPolicy(torch.nn.Module):
        def __init__(self, policy_config: Any) -> None:
            super().__init__()
            self.config = policy_config

        @classmethod
        def from_pretrained(cls, *, config: Any, **kwargs: Any) -> LightweightPolicy:
            assert kwargs["pretrained_name_or_path"] == base_model_path
            return cls(config)

    monkeypatch.setattr(
        policy_factory,
        "get_policy_class",
        lambda policy_type: LightweightPolicy,
    )
    policy = make_policy(config, ds_meta=dataset_meta, rename_map={})

    assert {
        name: (feature.type.value, feature.shape)
        for name, feature in policy.config.input_features.items()
    } == {
        "observation.state": ("STATE", (7,)),
        "observation.images.workspace": ("VISUAL", (3, 64, 96)),
    }
    assert {
        name: (feature.type.value, feature.shape)
        for name, feature in policy.config.output_features.items()
    } == {"action": ("ACTION", (7,))}

    saved_policy = tmp_path / "saved-policy"
    policy.config.save_pretrained(saved_policy)
    saved_config = json.loads(
        (saved_policy / "config.json").read_text(encoding="utf-8")
    )
    assert saved_config["input_features"] == {
        "observation.state": {"type": "STATE", "shape": [7]},
        "observation.images.workspace": {
            "type": "VISUAL",
            "shape": [3, 64, 96],
        },
    }
    assert saved_config["output_features"] == {
        "action": {"type": "ACTION", "shape": [7]}
    }

    monkeypatch.setattr(tokenizer_processor, "_transformers_available", True)
    monkeypatch.setattr(
        tokenizer_processor,
        "AutoTokenizer",
        SimpleNamespace(
            from_pretrained=lambda path: SimpleNamespace(name_or_path=path)
        ),
    )
    stats = {
        "observation.state": {
            "mean": torch.zeros(7),
            "std": torch.ones(7),
        },
        "action": {
            "mean": torch.zeros(7),
            "std": torch.ones(7),
        },
    }
    preprocessor, postprocessor = make_smolvla_pre_post_processors(
        policy.config,
        dataset_stats=stats,
    )
    saved_processors = tmp_path / "saved-processors"
    preprocessor.save_pretrained(saved_processors)
    postprocessor.save_pretrained(saved_processors)
    processor_config = json.loads(
        (saved_processors / "policy_preprocessor.json").read_text(encoding="utf-8")
    )
    normalizer = next(
        step
        for step in processor_config["steps"]
        if step["registry_name"] == "normalizer_processor"
    )
    assert normalizer["config"]["features"] == {
        **saved_config["input_features"],
        **saved_config["output_features"],
    }


@pytest.mark.parametrize(
    ("line", "expected_step", "expected_metrics"),
    [
        ("step:999 loss:0.25", 999, {"loss": 0.25}),
        ("step:1K smpl:8K loss:0.25", 1_000, {"smpl": 8_000.0, "loss": 0.25}),
        ("step:1.5K loss:-2.5M", 1_500, {"loss": -2_500_000.0}),
        ("step:2.147483647B loss:1", 2_147_483_647, {"loss": 1.0}),
        ("step:9e300Q loss:inf", None, {}),
    ],
)
def test_metric_parser_expands_only_bounded_finite_suffix_values(
    line: str,
    expected_step: int | None,
    expected_metrics: dict[str, float],
) -> None:
    assert _parse_metrics(line) == (expected_step, expected_metrics)


def test_tracker_metric_parser_keeps_exact_steps_after_lerobot_rounds_to_1k() -> None:
    parser = _TrainingMetricParser(log_every=200, max_steps=1_400)
    lines = [
        "step:200 smpl:2K ep:1 epch:0.10 loss:0.9",
        "step:400 smpl:3K ep:2 epch:0.20 loss:0.8",
        "step:600 smpl:5K ep:3 epch:0.30 loss:0.7",
        "step:800 smpl:6K ep:4 epch:0.40 loss:0.6",
        "step:1K smpl:8K ep:5 epch:0.50 loss:0.5",
        "step:1K smpl:10K ep:6 epch:0.60 loss:0.4",
        "step:1K smpl:11K ep:7 epch:0.70 loss:0.3",
    ]

    parsed = [
        parser.parse(line, transport_truncated=False)[0]
        for line in lines
    ]

    assert parsed == [200, 400, 600, 800, 1_000, 1_200, 1_400]


def test_tracker_metric_parser_can_recover_exact_max_step_from_rounded_token() -> None:
    parser = _TrainingMetricParser(
        log_every=2_147_483_647,
        max_steps=2_147_483_647,
    )

    step, metrics = parser.parse(
        "step:2B smpl:2B ep:1K epch:1.00 loss:0.25",
        transport_truncated=False,
    )

    assert step == 2_147_483_647
    assert metrics["loss"] == 0.25


class FakeHub:
    def __init__(self, spec: ManagedTrainingSpec) -> None:
        self.spec = spec
        self.uploads: list[tuple[str, str, str]] = []
        self.revision_files: dict[str, set[str]] = {
            MARKER_REVISION: {MANAGED_TRAINING_MARKER_PATH}
        }

    def model_info(self, *, repo_id: str, revision: str, token: str) -> object:
        assert token == HF_SECRET
        files = self.revision_files.get(revision, set()) | {MANAGED_TRAINING_MARKER_PATH}
        return SimpleNamespace(
            id=repo_id,
            sha=revision,
            private=True,
            siblings=[SimpleNamespace(rfilename=name) for name in sorted(files)],
        )

    def upload_folder(self, **kwargs: Any) -> object:
        assert kwargs["repo_id"] == self.spec.output_model_repo
        assert kwargs["token"] == HF_SECRET
        path_in_repo = kwargs["path_in_repo"]
        folder = Path(kwargs["folder_path"])
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        processor = json.loads(
            (folder / "policy_preprocessor.json").read_text(encoding="utf-8")
        )
        tokenizer = next(
            step
            for step in processor["steps"]
            if step["registry_name"] == "tokenizer_processor"
        )
        assert config["vlm_model_name"] == SMOLVLA_VLM_REPO
        assert config["load_vlm_weights"] is False
        assert config["use_peft"] is False
        assert config["license"] is None
        assert tokenizer["config"]["tokenizer_name"] == SMOLVLA_VLM_REPO
        assert "/tmp/" not in json.dumps(processor)
        if path_in_repo == "":
            card = (folder / "README.md").read_text(encoding="utf-8")
            assert "license:" not in card.casefold()
            revision = FINAL_REVISION
        else:
            revision = INTERMEDIATE_REVISION
        names = {
            f"{path_in_repo}/{path.relative_to(folder).as_posix()}"
            if path_in_repo
            else path.relative_to(folder).as_posix()
            for path in folder.rglob("*")
            if path.is_file()
        }
        assert {
            f"{path_in_repo}/{MANAGED_SMOLVLA_DEPENDENCY_DIR}/{filename}"
            if path_in_repo
            else f"{MANAGED_SMOLVLA_DEPENDENCY_DIR}/{filename}"
            for filename in _SMOLVLA_VLM_FILES
        }.issubset(names)
        self.revision_files[revision] = names
        self.uploads.append((path_in_repo, revision, kwargs["commit_message"]))
        return SimpleNamespace(oid=revision)


class FinishedProcess:
    def __init__(self, command: list[str], **kwargs: Any) -> None:
        self.command = command
        self.kwargs = kwargs
        self.stdout = io.StringIO(
            f"step:1 loss:0.5 leaked={HF_SECRET}\nstep:2 loss:0.25\n"
        )
        self.stderr = io.StringIO("trainer warning\n")
        self.returncode = 0
        base_model = Path(command[command.index("--policy.path") + 1])
        local_vlm = command[command.index("--policy.vlm_model_name") + 1]
        sealed_config = json.loads(
            (base_model / "config.json").read_text(encoding="utf-8")
        )
        sealed_processor = json.loads(
            (base_model / "policy_preprocessor.json").read_text(encoding="utf-8")
        )
        sealed_tokenizer = next(
            step
            for step in sealed_processor["steps"]
            if step["registry_name"] == "tokenizer_processor"
        )
        assert sealed_config["vlm_model_name"] == local_vlm
        assert sealed_config["load_vlm_weights"] is False
        assert sealed_config["use_peft"] is False
        assert sealed_config["input_features"] == {}
        assert sealed_config["output_features"] == {}
        assert sealed_config["empty_cameras"] == 0
        assert sealed_tokenizer["config"]["tokenizer_name"] == local_vlm
        assert SMOLVLA_VLM_REPO not in json.dumps(sealed_processor)
        output_dir = Path(command[command.index("--output_dir") + 1])
        for step in (1, 2):
            checkpoint = output_dir / "checkpoints" / f"{step:06d}"
            policy = checkpoint / "pretrained_model"
            _write_safe_base_model(policy)
            config = json.loads((policy / "config.json").read_text(encoding="utf-8"))
            config["vlm_model_name"] = local_vlm
            config["load_vlm_weights"] = False
            config["input_features"] = {
                "observation.state": {"type": "STATE", "shape": [7]},
                "observation.images.workspace": {
                    "type": "VISUAL",
                    "shape": [3, 64, 96],
                },
            }
            config["output_features"] = {
                "action": {"type": "ACTION", "shape": [7]}
            }
            (policy / "config.json").write_text(json.dumps(config), encoding="utf-8")
            processor = json.loads(
                (policy / "policy_preprocessor.json").read_text(encoding="utf-8")
            )
            for entry in processor["steps"]:
                if entry["registry_name"] == "tokenizer_processor":
                    entry["config"]["tokenizer_name"] = local_vlm
                elif entry["registry_name"] == "device_processor":
                    entry["config"] = {"device": "cuda", "float_dtype": None}
            (policy / "policy_preprocessor.json").write_text(
                json.dumps(processor), encoding="utf-8"
            )
            training_state = checkpoint / "training_state"
            training_state.mkdir()
            (training_state / "training_step.json").write_text(
                f'{{"step": {step}}}\n', encoding="utf-8"
            )

    def poll(self) -> int:
        return self.returncode

    def wait(self, timeout: float) -> int:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


@pytest.mark.parametrize(
    "sentinel",
    [
        None,
        '{"step": 2}\n',
        '{"step": true}\n',
        '{"step": 1, "unexpected": 1}\n',
        '{"step": 1, "step": 1}\n',
        "{\"step\": 1, \"padding\": \"" + "x" * 256 + "\"}",
    ],
)
def test_checkpoint_discovery_requires_bounded_exact_step_sentinel(
    tmp_path: Path, sentinel: str | None
) -> None:
    checkpoint = tmp_path / "checkpoints" / "000001"
    policy = checkpoint / "pretrained_model"
    policy.mkdir(parents=True)
    for filename in (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    ):
        (policy / filename).write_text("artifact", encoding="utf-8")
    if sentinel is not None:
        training_state = checkpoint / "training_state"
        training_state.mkdir()
        (training_state / "training_step.json").write_text(
            sentinel, encoding="utf-8"
        )

    assert _checkpoint_dirs(tmp_path) == []


def test_checkpoint_discovery_accepts_completed_lerobot_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoints" / "000001"
    policy = checkpoint / "pretrained_model"
    policy.mkdir(parents=True)
    for filename in (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    ):
        (policy / filename).write_text("artifact", encoding="utf-8")
    training_state = checkpoint / "training_state"
    training_state.mkdir()
    (training_state / "training_step.json").write_text(
        '{"step": 1}\n', encoding="utf-8"
    )

    assert _checkpoint_dirs(tmp_path) == [(1, policy)]

def test_worker_pins_inputs_redacts_logs_and_explicitly_uploads_checkpoints(
    tmp_path: Path,
) -> None:
    spec = _spec()
    marker = tmp_path / "marker.json"
    marker.write_bytes(managed_training_marker(JOB_ID, REQUEST_HASH))
    hub = FakeHub(spec)
    snapshots: list[dict[str, Any]] = []
    process_holder: list[FinishedProcess] = []

    def snapshot(**kwargs: Any) -> str:
        snapshots.append(kwargs)
        return _populate_snapshot(spec, kwargs)

    def popen(command: list[str], **kwargs: Any) -> FinishedProcess:
        process = FinishedProcess(command, **kwargs)
        process_holder.append(process)
        return process

    output = io.StringIO()
    result = run_managed_training(
        managed_training_spec_payload(spec),
        environ={
            "HF_TOKEN": HF_SECRET,
            "MODAL_TOKEN_SECRET": "modal-secret",
            "DATABASE_URL": "postgresql://secret",
            "SAFE_VALUE": "yes",
        },
        output=output,
        api_factory=lambda **kwargs: hub,
        snapshot=snapshot,
        marker_download=lambda **kwargs: str(marker),
        popen_factory=popen,
    )

    assert [(item["repo_id"], item["revision"], item["repo_type"]) for item in snapshots] == [
        (spec.dataset_repo, DATASET_REVISION, "dataset"),
        (spec.base_model, BASE_REVISION, "model"),
        (SMOLVLA_VLM_REPO, SMOLVLA_VLM_REVISION, "model"),
    ]
    process = process_holder[0]
    assert process.kwargs["shell"] is False
    assert process.kwargs["start_new_session"] is True
    assert "HF_TOKEN" not in process.kwargs["env"]
    assert "MODAL_TOKEN_SECRET" not in process.kwargs["env"]
    assert "DATABASE_URL" not in process.kwargs["env"]
    assert process.kwargs["env"]["HF_HUB_OFFLINE"] == "1"
    assert hub.uploads == [
        ("checkpoints/0000000001", INTERMEDIATE_REVISION, "ctrl-pi checkpoint 1"),
        ("", FINAL_REVISION, "ctrl-pi final checkpoint 2"),
    ]
    assert result == {
        "schema": "ctrl-pi.modal-training-result",
        "version": 1,
        "job_id": str(JOB_ID),
        "request_hash": REQUEST_HASH,
        "output_model_repo": spec.output_model_repo,
        "revision": FINAL_REVISION,
        "step": 2,
        "last_sequence": result["last_sequence"],
        "events_truncated": False,
    }
    lines = output.getvalue().splitlines()
    assert all(line.startswith(MODAL_TRAINING_EVENT_PREFIX) for line in lines)
    assert HF_SECRET not in output.getvalue()
    assert "[redacted trainer output]" in output.getvalue()
    assert '"type":"metric"' in output.getvalue()
    assert '"final":true' in output.getvalue()


def test_dynamic_processor_class_is_rejected_before_popen(tmp_path: Path) -> None:
    spec = _spec()
    marker = tmp_path / "marker.json"
    marker.write_bytes(managed_training_marker(JOB_ID, REQUEST_HASH))
    hub = FakeHub(spec)
    popen_called = False

    def snapshot(**kwargs: Any) -> str:
        path = Path(_populate_snapshot(spec, kwargs))
        if kwargs["repo_id"] == spec.base_model:
            (path / "policy_preprocessor.json").write_text(
                json.dumps(
                    {
                        "name": "policy_preprocessor",
                        "steps": [
                            {
                                "class": "subprocess.Popen",
                                "config": {"args": ["touch", "/tmp/owned"]},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
        return str(path)

    def popen(*args: Any, **kwargs: Any) -> None:
        nonlocal popen_called
        popen_called = True

    with pytest.raises(RuntimeError, match="executable configuration"):
        run_managed_training(
            managed_training_spec_payload(spec),
            environ={"HF_TOKEN": HF_SECRET},
            output=io.StringIO(),
            api_factory=lambda **kwargs: hub,
            snapshot=snapshot,
            marker_download=lambda **kwargs: str(marker),
            popen_factory=popen,
        )

    assert popen_called is False


@pytest.mark.parametrize("mutation", ["missing_camera", "oversized_state", "extra_policy_feature"])
def test_incompatible_dataset_features_are_rejected_before_popen(
    tmp_path: Path,
    mutation: str,
) -> None:
    spec = _spec()
    marker = tmp_path / "marker.json"
    marker.write_bytes(managed_training_marker(JOB_ID, REQUEST_HASH))
    hub = FakeHub(spec)
    popen_called = False

    def snapshot(**kwargs: Any) -> str:
        path = Path(_populate_snapshot(spec, kwargs))
        if kwargs["repo_id"] != spec.dataset_repo:
            return str(path)
        info_path = path / "meta" / "info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        features = info["features"]
        if mutation == "missing_camera":
            del features["observation.images.workspace"]
        elif mutation == "oversized_state":
            features["observation.state"] = {
                "dtype": "float32",
                "shape": [33],
                "names": [f"joint_{index}" for index in range(33)],
            }
        else:
            features["observation.environment"] = {
                "dtype": "float32",
                "shape": [1],
                "names": ["unsafe_extra"],
            }
        info_path.write_text(json.dumps(info), encoding="utf-8")
        return str(path)

    def popen(*args: Any, **kwargs: Any) -> None:
        nonlocal popen_called
        popen_called = True

    with pytest.raises(RuntimeError, match="dataset"):
        run_managed_training(
            managed_training_spec_payload(spec),
            environ={"HF_TOKEN": HF_SECRET},
            output=io.StringIO(),
            api_factory=lambda **kwargs: hub,
            snapshot=snapshot,
            marker_download=lambda **kwargs: str(marker),
            popen_factory=popen,
        )

    assert popen_called is False


@pytest.mark.parametrize("mutation", ["peft", "unexpected_step_config"])
def test_unsafe_base_model_indirections_are_rejected_before_popen(
    tmp_path: Path, mutation: str
) -> None:
    spec = _spec()
    marker = tmp_path / "marker.json"
    marker.write_bytes(managed_training_marker(JOB_ID, REQUEST_HASH))
    hub = FakeHub(spec)
    popen_called = False

    def snapshot(**kwargs: Any) -> str:
        path = Path(_populate_snapshot(spec, kwargs))
        if kwargs["repo_id"] != spec.base_model:
            return str(path)
        if mutation == "peft":
            config = json.loads((path / "config.json").read_text(encoding="utf-8"))
            config["use_peft"] = True
            (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
        else:
            processor = json.loads(
                (path / "policy_preprocessor.json").read_text(encoding="utf-8")
            )
            processor["steps"][4]["config"]["unexpected"] = "attacker-controlled"
            (path / "policy_preprocessor.json").write_text(
                json.dumps(processor), encoding="utf-8"
            )
        return str(path)

    def popen(*args: Any, **kwargs: Any) -> None:
        nonlocal popen_called
        popen_called = True

    with pytest.raises(RuntimeError, match="SmolVLA|device processor"):
        run_managed_training(
            managed_training_spec_payload(spec),
            environ={"HF_TOKEN": HF_SECRET},
            output=io.StringIO(),
            api_factory=lambda **kwargs: hub,
            snapshot=snapshot,
            marker_download=lambda **kwargs: str(marker),
            popen_factory=popen,
        )

    assert popen_called is False


def test_checkpoint_upload_rejects_symlink_before_writing(tmp_path: Path) -> None:
    spec = _spec()
    policy = tmp_path / "policy"
    vlm = tmp_path / "vlm"
    dataset = tmp_path / "dataset"
    _write_safe_base_model(policy)
    _write_safe_vlm_snapshot(vlm)
    _write_safe_dataset(dataset)
    _seal_base_model_for_child(policy, vlm, dataset_path=dataset)
    outside = tmp_path / "outside"
    outside.write_text("do not overwrite", encoding="utf-8")
    (policy / "README.md").symlink_to(outside)

    with pytest.raises(RuntimeError, match="unsafe path"):
        _prepare_policy_for_upload(policy, spec=spec, vlm_model_path=vlm)

    assert outside.read_text(encoding="utf-8") == "do not overwrite"


class RunningProcess(FinishedProcess):
    def __init__(self, command: list[str], **kwargs: Any) -> None:
        super().__init__(command, **kwargs)
        self.returncode = None
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float) -> int:
        if self.returncode is None:
            raise TimeoutError
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


def test_worker_enforces_absolute_deadline_and_terminates_child(tmp_path: Path) -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    spec = _spec(deadline_at=start + timedelta(seconds=30))
    marker = tmp_path / "marker.json"
    marker.write_bytes(managed_training_marker(JOB_ID, REQUEST_HASH))
    hub = FakeHub(spec)
    calls = 0
    process_holder: list[RunningProcess] = []

    def now() -> datetime:
        nonlocal calls
        calls += 1
        return start if calls <= 9 else spec.deadline_at

    def snapshot(**kwargs: Any) -> str:
        return _populate_snapshot(spec, kwargs)

    def popen(command: list[str], **kwargs: Any) -> RunningProcess:
        process = RunningProcess(command, **kwargs)
        process_holder.append(process)
        return process

    with pytest.raises(RuntimeError, match="deadline"):
        run_managed_training(
            managed_training_spec_payload(spec),
            environ={"HF_TOKEN": HF_SECRET},
            output=io.StringIO(),
            api_factory=lambda **kwargs: hub,
            snapshot=snapshot,
            marker_download=lambda **kwargs: str(marker),
            popen_factory=popen,
            now=now,
        )

    assert process_holder[0].terminated is True
    assert process_holder[0].poll() == -15


def test_expired_deadline_rejects_before_modal_objects_are_created() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    fake = FakeModal()

    with pytest.raises(ManagedTrainingConfigurationError, match="deadline"):
        build_modal_training_workload(
            spec=_spec(deadline_at=now - timedelta(seconds=1)),
            hf_token=HF_SECRET,
            modal_module=fake,  # type: ignore[arg-type]
            now=lambda: now,
        )

    assert fake.app_name == ""


def test_workload_constructs_with_pinned_modal_sdk_without_network() -> None:
    workload = build_modal_training_workload(
        spec=_spec(compute_size="Modal: 2xA100"),
        hf_token=HF_SECRET,
        modal_module=modal,
    )

    assert isinstance(workload.app, modal.App)
    assert isinstance(workload.training_function, modal.Function)


def test_modal_worker_boundary_never_exposes_dependency_error_or_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ctrl_pi.modal_training_workload.run_managed_training",
        lambda payload: (_ for _ in ()).throw(RuntimeError(f"failed {HF_SECRET}")),
    )

    with pytest.raises(RuntimeError, match="failed safely") as caught:
        run_managed_training_modal({})

    assert HF_SECRET not in str(caught.value)
    assert caught.value.__cause__ is None


def test_pipe_reader_bounds_unterminated_lines_and_marks_gap() -> None:
    messages: queue.Queue[tuple[str, str | None]] = queue.Queue(maxsize=8)
    overflowed = threading.Event()

    _read_pipe(io.StringIO("x" * 20_000), "stdout", messages, overflowed)

    queued = [messages.get_nowait()[1] for _ in range(messages.qsize())]
    assert all(value is None or len(value) <= 4_097 for value in queued)
    assert overflowed.is_set()
