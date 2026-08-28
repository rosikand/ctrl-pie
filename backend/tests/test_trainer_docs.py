from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_trainer_api_python_examples_are_syntactically_valid() -> None:
    documentation = (
        Path(__file__).parents[2] / "docs" / "trainer-api.md"
    ).read_text(encoding="utf-8")
    blocks = documentation.split("```python\n")[1:]

    assert len(blocks) >= 6
    for index, block in enumerate(blocks):
        source = block.split("```", 1)[0]
        ast.parse(source, filename=f"trainer-api.md:python-block-{index}")


def test_trainer_api_documents_every_client_method_and_rest_endpoint() -> None:
    documentation = (
        Path(__file__).parents[2] / "docs" / "trainer-api.md"
    ).read_text(encoding="utf-8")

    for method in (
        "create_run",
        "list_runs",
        "get_run",
        "update_run",
        "log_metrics",
        "register_checkpoint",
    ):
        assert method in documentation
    for path in (
        "/api/trainer/runs",
        "/api/trainer/runs/{id}",
        "/api/trainer/runs/{id}/metrics",
        "/api/trainer/runs/{id}/checkpoints",
        "/api/trainer/models",
    ):
        assert path in documentation


def test_end_to_end_lerobot_example_executes_offline_with_fakes(
    tmp_path: Path,
) -> None:
    documentation = (
        Path(__file__).parents[2] / "docs" / "trainer-api.md"
    ).read_text(encoding="utf-8")
    after_marker = documentation.split(
        "<!-- executable-lerobot-example -->", 1
    )[1]
    source = after_marker.split("```python\n", 1)[1].split("```", 1)[0]
    module: dict[str, object] = {"__name__": "trainer_docs_example"}
    exec(compile(source, "trainer-api.md:lerobot-example", "exec"), module)

    events: list[tuple[str, object]] = []

    class FakeTrainer:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def create_run(self, name, **fields):
            events.append(("create", fields))
            return {"id": "11111111-1111-4111-8111-111111111111"}

        def update_run(self, run_id, **fields):
            events.append(("update", fields))

        def log_metrics(self, run_id, **fields):
            events.append(("metrics", fields))

        def register_checkpoint(self, run_id, **fields):
            events.append(("checkpoint", fields))

    class FakeHub:
        def whoami(self, **kwargs):
            assert kwargs == {"token": "hf_docs_token"}
            return {"name": "acme", "orgs": []}

        def create_repo(self, **kwargs):
            assert kwargs["private"] is True
            assert kwargs["exist_ok"] is False
            assert kwargs["token"] == "hf_docs_token"

        def model_info(self, **kwargs):
            assert kwargs["token"] == "hf_docs_token"
            assert "revision" not in kwargs
            return SimpleNamespace(private=True)

        def upload_folder(self, **kwargs):
            assert kwargs["token"] == "hf_docs_token"
            assert Path(kwargs["folder_path"]).is_dir()
            return SimpleNamespace(oid="a" * 40)

    output_dir = tmp_path / "training"
    base_path = tmp_path / "base"
    base_path.mkdir()

    class FakeProcess:
        stdout = ["step:100 loss:0.42 lr:0.0001\n"]

        @staticmethod
        def wait(timeout=None) -> int:
            return 0

        @staticmethod
        def poll() -> int:
            return 0

    def fake_popen(command, **kwargs):
        assert command[0] == "lerobot-train"
        assert kwargs["env"]["HF_TOKEN"] == "hf_docs_token"
        checkpoint = output_dir / "checkpoints" / "000100" / "pretrained_model"
        checkpoint.mkdir(parents=True)
        return FakeProcess()

    module["main"](
        env={
            "CTRL_PI_URL": "http://ctrl-pi.local:8000",
            "HF_TOKEN": "hf_docs_token",
            "DATASET_REPO": "acme/yam-data",
            "MODEL_REPO": "acme/new-act-model",
            "BASE_MODEL": "lerobot/act-base",
            "OUTPUT_DIR": str(output_dir),
            "STEPS": "100",
        },
        trainer_factory=lambda url, timeout: FakeTrainer(),
        hub_factory=lambda token: FakeHub(),
        snapshot=lambda **kwargs: str(base_path),
        popen_factory=fake_popen,
    )

    assert ("metrics", {"step": 100, "metrics": {"lerobot/loss": 0.42, "lerobot/lr": 0.0001}}) in events
    assert ("checkpoint", {"repo_id": "acme/new-act-model", "revision": "a" * 40, "step": 100}) in events
    assert events[-1] == ("update", {"status": "completed", "current_step": 100})


def test_end_to_end_lerobot_example_terminates_child_on_reporting_failure(
    tmp_path: Path,
) -> None:
    documentation = (
        Path(__file__).parents[2] / "docs" / "trainer-api.md"
    ).read_text(encoding="utf-8")
    source = documentation.split("<!-- executable-lerobot-example -->", 1)[1]
    source = source.split("```python\n", 1)[1].split("```", 1)[0]
    module: dict[str, object] = {"__name__": "trainer_docs_example"}
    exec(compile(source, "trainer-api.md:lerobot-example", "exec"), module)

    updates: list[str] = []

    class FailingTrainer:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        @staticmethod
        def create_run(name, **fields):
            return {"id": "11111111-1111-4111-8111-111111111111"}

        @staticmethod
        def log_metrics(run_id, **fields):
            raise RuntimeError("reporting unavailable")

        @staticmethod
        def register_checkpoint(run_id, **fields):
            raise AssertionError("checkpoint must not be reached")

        @staticmethod
        def update_run(run_id, **fields):
            updates.append(fields["status"])

    class FakeHub:
        @staticmethod
        def whoami(**kwargs):
            return {"name": "acme", "orgs": []}

        @staticmethod
        def create_repo(**kwargs):
            return None

        @staticmethod
        def model_info(**kwargs):
            assert "revision" not in kwargs
            return SimpleNamespace(private=True)

    class RunningProcess:
        stdout = ["step:100 loss:0.42\n"]
        terminated = False
        killed = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

        @staticmethod
        def wait(timeout=None):
            return 0

    process = RunningProcess()
    base_path = tmp_path / "base"
    base_path.mkdir()
    with pytest.raises(RuntimeError, match="reporting unavailable"):
        module["main"](
            env={
                "CTRL_PI_URL": "http://ctrl-pi.local:8000",
                "HF_TOKEN": "hf_docs_token",
                "DATASET_REPO": "acme/yam-data",
                "MODEL_REPO": "acme/new-act-model",
                "BASE_MODEL": "lerobot/act-base",
                "OUTPUT_DIR": str(tmp_path / "training"),
                "STEPS": "100",
            },
            trainer_factory=lambda url, timeout: FailingTrainer(),
            hub_factory=lambda token: FakeHub(),
            snapshot=lambda **kwargs: str(base_path),
            popen_factory=lambda command, **kwargs: process,
        )

    assert process.terminated is True
    assert process.killed is False
    assert updates == ["running", "failed"]
