# Trainer API

ctrl-π does not run training. The Trainer API is a small, local control-plane
API that lets an external LeRobot, OpenPI, or custom training script report
run status, scalar metrics, and Hugging Face checkpoint revisions. Metrics and
run metadata live in PostgreSQL; model weights stay on Hugging Face.

There is no ctrl-π authentication in V1. Run the backend on a trusted network
and use its LAN URL from the training machine. Never put Hugging Face tokens,
database URLs, passwords, private keys, or other credentials in a run config.

## Install the Python client

From a ctrl-pi checkout:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ./backend
```

The client supports Python 3.11 and uses synchronous `httpx`. It accepts only
the ctrl-pi base URL and a timeout; ctrl-pi has no client API key.

## Quickstart

Start the backend, then save this as `trainer_quickstart.py` and run it with
`python trainer_quickstart.py`:

```python
from ctrl_pi.trainer import Client


with Client("http://127.0.0.1:8000", timeout=10.0) as trainer:
    run = trainer.create_run(
        "ACT block pickup",
        dataset_repo="my-hf-user/yam-block-pickup",
        base_model="lerobot/act_aloha_sim_transfer_cube_human",
        runtime="lerobot",
        framework="pytorch",
        config={"batch_size": 32, "learning_rate": 1e-4},
    )
    run_id = run["id"]
    trainer.update_run(run_id, status="running")

    for step, loss in ((100, 0.82), (200, 0.51), (300, 0.34)):
        trainer.log_metrics(run_id, step=step, metrics={"train/loss": loss})

    trainer.update_run(run_id, status="completed", current_step=300)
    print(trainer.get_run(run_id))
```

Run statuses are `created`, `running`, `completed`, `failed`, and `cancelled`.
External scripts may resume or correct a status and may report late metrics or
checkpoint metadata. `current_step` never decreases: explicit updates below
the stored step return HTTP 409, while metric/checkpoint calls retain the
maximum observed step.

## Python client reference

Import `Client` and the sanitized `TrainerClientError` from
`ctrl_pi.trainer`. Every method returns a typed dictionary representing the
updated run, except `list_runs`, which returns a list of those dictionaries.

### `Client(base_url, *, timeout=10.0)`

Creates a synchronous client. Use it as a context manager or call `close()`.
`is_closed` reports whether the underlying HTTP client has closed.

### `create_run(name, **metadata)`

Creates a run. Optional keyword arguments are `status`, `current_step`,
`dataset_repo`, `base_model`, `runtime`, `framework`, `output_model_repo`,
`checkpoint_revision`, and `config`.

```python
from ctrl_pi.trainer import Client


trainer = Client("http://ctrl-pi-box:8000")
run = trainer.create_run(
    "SmolVLA placement",
    status="created",
    current_step=0,
    dataset_repo="my-org/yam-placement",
    base_model="lerobot/smolvla_base",
    runtime="lerobot",
    framework="pytorch",
    config={"batch_size": 16},
)
trainer.close()
```

### `list_runs(*, status=None)`

Lists runs newest first. Pass one status literal to filter the result.

### `get_run(run_id)`

Fetches one run by UUID.

### `update_run(run_id, **fields)`

Updates any supplied field among `status`, `current_step`, `dataset_repo`,
`base_model`, `runtime`, `framework`, `output_model_repo`,
`checkpoint_revision`, and `config`. Omitted fields remain unchanged. Passing
`None` clears a nullable repository/runtime/framework/revision field.

### `log_metrics(run_id, *, step, metrics)`

Logs one or more finite scalar metrics. Names may contain letters, numbers,
`.`, `_`, `-`, and `/`. Sending the same metric name and step again replaces
that point deterministically instead of appending a duplicate.

```python
trainer.log_metrics(
    run_id,
    step=500,
    metrics={"train/loss": 0.27, "eval/success_rate": 0.74},
)
```

### `register_checkpoint(run_id, *, repo_id, revision, step)`

Registers a checkpoint already pushed to a Hugging Face model repository.
The call also updates `output_model_repo`, `checkpoint_revision`, and
`current_step`. Repeating the exact call is idempotent.

```python
trainer.register_checkpoint(
    run_id,
    repo_id="my-org/yam-act-policy",
    revision="0123456789abcdef0123456789abcdef01234567",
    step=1000,
)
```

### Errors

`TrainerClientError.status_code` contains the HTTP status when a server
responded. Transport errors have `status_code=None`. Error strings are
deliberately generic and never include the response body, URL credentials, or
the underlying exception text.

```python
from ctrl_pi.trainer import Client, TrainerClientError


try:
    with Client("http://ctrl-pi-box:8000") as trainer:
        trainer.get_run("11111111-1111-4111-8111-111111111111")
except TrainerClientError as error:
    print(error.status_code, str(error))
```

## REST representation

A training run has this shape:

```json
{
  "id": "11111111-1111-4111-8111-111111111111",
  "name": "ACT block pickup",
  "status": "running",
  "current_step": 200,
  "dataset_repo": "my-org/yam-block-pickup",
  "base_model": "lerobot/act_aloha_sim_transfer_cube_human",
  "runtime": "lerobot",
  "framework": "pytorch",
  "output_model_repo": null,
  "checkpoint_revision": null,
  "config": {"batch_size": 32},
  "metrics": {
    "train/loss": [{"step": 100, "value": 0.82}]
  },
  "checkpoints": [],
  "created_at": "2026-08-28T12:00:00Z",
  "updated_at": "2026-08-28T12:05:00Z"
}
```

Repository IDs use `namespace/name` form. Revisions may be a Hub commit SHA,
branch, or tag using safe Git revision characters. Config is limited to 64
KiB of finite JSON values and recursively rejects secret-like keys. One run
may store at most 128 metric names, 10,000 total metric points, and 512
checkpoint records.

## REST endpoints

All endpoints are under the backend URL; there is no `/v1` prefix.

| Method | Path | Request | Response |
| --- | --- | --- | --- |
| `POST` | `/api/trainer/runs` | run creation object | run, HTTP 201 |
| `GET` | `/api/trainer/runs?status=running` | optional status query | `{"runs": [...]}` |
| `GET` | `/api/trainer/runs/{id}` | — | run |
| `PATCH` | `/api/trainer/runs/{id}` | partial run fields | updated run |
| `POST` | `/api/trainer/runs/{id}/metrics` | `{"step", "metrics"}` | updated run |
| `POST` | `/api/trainer/runs/{id}/checkpoints` | `{"repo_id", "revision", "step"}` | updated run |
| `GET` | `/api/trainer/models?refresh=false` | optional refresh query | namespace model listing |

Create and capture a run ID:

```bash
RUN_ID="$({
  curl --fail --silent --show-error \
    -H 'Content-Type: application/json' \
    -d '{
      "name":"ACT block pickup",
      "dataset_repo":"my-org/yam-data",
      "runtime":"lerobot",
      "framework":"pytorch",
      "config":{"batch_size":32}
    }' \
    http://127.0.0.1:8000/api/trainer/runs
} | python -c 'import json,sys; print(json.load(sys.stdin)["id"])')"
```

Read, list, and update runs:

```bash
curl --fail "http://127.0.0.1:8000/api/trainer/runs/$RUN_ID"
curl --fail 'http://127.0.0.1:8000/api/trainer/runs?status=running'
curl --fail -X PATCH -H 'Content-Type: application/json' \
  -d '{"status":"running","current_step":100}' \
  "http://127.0.0.1:8000/api/trainer/runs/$RUN_ID"
```

Log metrics and register a checkpoint:

```bash
curl --fail -X POST -H 'Content-Type: application/json' \
  -d '{"step":100,"metrics":{"train/loss":0.82,"lr":0.0001}}' \
  "http://127.0.0.1:8000/api/trainer/runs/$RUN_ID/metrics"

curl --fail -X POST -H 'Content-Type: application/json' \
  -d '{
    "repo_id":"my-org/yam-act-policy",
    "revision":"0123456789abcdef0123456789abcdef01234567",
    "step":100
  }' \
  "http://127.0.0.1:8000/api/trainer/runs/$RUN_ID/checkpoints"
```

Model discovery always uses the backend's configured `HF_NAMESPACE` and an
explicit backend-only `HF_TOKEN`; neither can be supplied by the browser:

```bash
curl --fail 'http://127.0.0.1:8000/api/trainer/models'
curl --fail 'http://127.0.0.1:8000/api/trainer/models?refresh=true'
```

The model response is:

```json
{
  "namespace": "my-org",
  "models": [{
    "repo_id": "my-org/yam-act-policy",
    "name": "yam-act-policy",
    "revision": "0123456789abcdef0123456789abcdef01234567",
    "hub_url": "https://huggingface.co/my-org/yam-act-policy",
    "private": true,
    "gated": false,
    "last_modified": "2026-08-28T12:10:00Z",
    "pipeline_tag": "robotics",
    "library_name": "lerobot",
    "tags": ["LeRobot"],
    "card": {
      "description": "ACT policy for YAM block pickup.",
      "base_model": ["lerobot/act_aloha_sim_transfer_cube_human"],
      "datasets": ["my-org/yam-block-pickup"]
    },
    "checkpoints": ["checkpoints/001000/model.safetensors"]
  }],
  "total": 1,
  "fetched_at": "2026-08-28T12:11:00Z"
}
```

Normal model responses use a 30-second private cache. `refresh=true` bypasses
both Hub enumeration and revision-card caches and returns `Cache-Control:
private, no-store`. A missing or malformed card degrades only that model.

HTTP errors are stable and sanitized:

- `403`: `HF_TOKEN` cannot access the configured namespace.
- `404`: a training run UUID does not exist.
- `409`: an explicit `current_step` would regress or a per-run metadata limit
  is reached.
- `422`: request, status, repository, revision, metric, or config validation.
- `502`: Hugging Face model enumeration failed.
- `503`: PostgreSQL or required Hugging Face configuration is unavailable.

## End-to-end LeRobot fine-tune and checkpoint flow

LeRobot 0.4.4 owns the training and checkpoint layout. ctrl-π only observes
small scalar/control-plane events. The following complete script launches the
real `lerobot-train` command, streams its scalar log lines into ctrl-π, pushes
the final `pretrained_model` directory, and registers the immutable Hub commit.

Set `CTRL_PI_URL`, `HF_TOKEN`, `DATASET_REPO`, and a **new, dedicated**
`MODEL_REPO`. The example intentionally uses `exist_ok=False`: it refuses to
write into any existing repository, so a typo cannot replace unrelated model
files. Pick a new model slug for another run. Optional variables are
`BASE_MODEL`, `OUTPUT_DIR`, and `STEPS`.

<!-- executable-lerobot-example -->
```python
from __future__ import annotations

import math
import os
import re
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import validate_repo_id

from ctrl_pi.trainer import Client


LOG_VALUE = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)\s*:\s*([^\s]+)")


def required(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Set {name} before starting training.")
    return value


def formatted_step(value: str) -> int:
    suffixes = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    suffix = value[-1:].upper()
    multiplier = suffixes.get(suffix, 1)
    number = value[:-1] if suffix in suffixes else value
    return int(float(number) * multiplier)


def main(
    *,
    env: Mapping[str, str] = os.environ,
    trainer_factory: Callable[..., Any] = Client,
    hub_factory: Callable[..., Any] = HfApi,
    snapshot: Callable[..., str] = snapshot_download,
    popen_factory: Callable[..., Any] = subprocess.Popen,
) -> None:
    ctrl_pi_url = required(env, "CTRL_PI_URL")
    hf_token = required(env, "HF_TOKEN")
    dataset_repo = required(env, "DATASET_REPO")
    model_repo = required(env, "MODEL_REPO")
    base_model = env.get(
        "BASE_MODEL", "lerobot/act_aloha_sim_transfer_cube_human"
    )
    output_dir = Path(env.get("OUTPUT_DIR", "outputs/ctrl-pi-act"))
    steps = int(env.get("STEPS", "1000"))
    if steps <= 0:
        raise RuntimeError("STEPS must be positive.")
    validate_repo_id(dataset_repo)
    validate_repo_id(model_repo)
    validate_repo_id(base_model)

    hub = hub_factory(token=hf_token)
    namespace = model_repo.split("/", 1)[0]
    identity = hub.whoami(token=hf_token)
    allowed_namespaces = {str(identity.get("name", "")).casefold()}
    allowed_namespaces.update(
        str(org.get("name", "")).casefold()
        for org in identity.get("orgs", [])
        if isinstance(org, dict)
    )
    if namespace.casefold() not in allowed_namespaces:
        raise RuntimeError("MODEL_REPO is outside the token's user or organizations.")

    with trainer_factory(ctrl_pi_url, timeout=30.0) as trainer:
        run = trainer.create_run(
            "ACT YAM fine-tune",
            dataset_repo=dataset_repo,
            base_model=base_model,
            runtime="lerobot",
            framework="pytorch",
            config={"steps": steps, "policy": "act"},
        )
        run_id = run["id"]
        trainer.update_run(run_id, status="running")
        process = None
        try:
            # A collision aborts here before training or uploading any weights.
            hub.create_repo(
                repo_id=model_repo,
                repo_type="model",
                private=True,
                exist_ok=False,
                token=hf_token,
            )
            visibility = hub.model_info(repo_id=model_repo, token=hf_token)
            if getattr(visibility, "private", None) is not True:
                raise RuntimeError("The newly created model repository is not private.")

            base_path = snapshot(
                repo_id=base_model,
                repo_type="model",
                token=hf_token,
            )
            command = [
                "lerobot-train",
                "--dataset.repo_id",
                dataset_repo,
                "--policy.type",
                "act",
                "--policy.pretrained_path",
                base_path,
                "--output_dir",
                str(output_dir),
                "--steps",
                str(steps),
                "--log_freq",
                "100",
                "--save_checkpoint",
                "true",
                "--save_freq",
                str(steps),
            ]
            process = popen_factory(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=dict(env),
            )
            if process.stdout is None:
                raise RuntimeError("LeRobot did not provide a readable log stream.")
            for line in process.stdout:
                print(line, end="")
                values = dict(LOG_VALUE.findall(line))
                if "step" not in values:
                    continue
                step = formatted_step(values.pop("step"))
                metrics: dict[str, float] = {}
                for name, raw_value in values.items():
                    try:
                        value = float(raw_value)
                    except ValueError:
                        continue
                    if math.isfinite(value):
                        metrics[f"lerobot/{name}"] = value
                if metrics:
                    trainer.log_metrics(run_id, step=step, metrics=metrics)
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError("LeRobot training failed; inspect its local logs.")

            width = max(6, len(str(steps)))
            checkpoint_dir = (
                output_dir
                / "checkpoints"
                / f"{steps:0{width}d}"
                / "pretrained_model"
            )
            if not checkpoint_dir.is_dir():
                raise RuntimeError("LeRobot did not produce the expected checkpoint.")
            commit = hub.upload_folder(
                repo_id=model_repo,
                repo_type="model",
                folder_path=checkpoint_dir,
                path_in_repo=f"checkpoints/{steps:0{width}d}",
                commit_message=f"LeRobot checkpoint at step {steps}",
                token=hf_token,
            )
            trainer.register_checkpoint(
                run_id,
                repo_id=model_repo,
                revision=commit.oid,
                step=steps,
            )
            trainer.update_run(run_id, status="completed", current_step=steps)
        except BaseException:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            try:
                trainer.update_run(run_id, status="failed")
            except Exception:
                # Preserve the training/Hub failure as the primary error.
                pass
            raise


if __name__ == "__main__":
    main()
```

Keep `HF_TOKEN` in the training process environment or its secret manager;
never include it in `config` or send it to ctrl-π. The token goes explicitly
to every Hub operation. Only the model repository ID, immutable commit
revision, and step are registered with ctrl-π.
