---
title: "Training"
description: "Track external runs, metrics, bounded console output, configuration, and checkpoints."
icon: "brain-circuit"
---

**Training** is a first-class primary page. ctrl-π does not execute or fund
training in this release. LeRobot, OpenPI, or custom scripts run
on compute you operate and report small control-plane events to ctrl-π through
the [Trainer API](/trainer-api).

```text
Training script ──► ctrl-π Trainer API ──► PostgreSQL
       │
       └──────────► Hugging Face model repository
```

Model weights never pass through ctrl-π. The external script uploads them with
its normal Hugging Face tooling, then registers only the repository ID,
immutable revision, and step.

## Training page

The page groups active `created`/`running` runs separately from
past `completed`/`failed`/`cancelled` runs. Select one to inspect:

- current status and monotonic step;
- source dataset and base model;
- runtime and framework labels;
- bounded JSON configuration;
- one chart per reported scalar metric;
- the bounded, incrementally polled console tail; and
- output model plus registered checkpoint revisions.

Metric names may use letters, numbers, `.`, `_`, `-`, and `/`. Reporting the
same metric name and step replaces that point deterministically. ctrl-π stores
at most 128 metric series, 10,000 total points, and 512 checkpoint records per
run.

Console records have a server-assigned sequence and timestamp plus a
`stdout`, `stderr`, or `system` source. The server keeps only the newest 1,000
records and trims to a compact 512 KiB ceiling; the UI shows a retention-gap
notice when older sequences are unavailable. Logs are control-plane progress
records, not a live attachment to an external process.

## Report a run

Install the client from the same checkout as the backend:

```bash
python -m pip install -e ./backend
```

Then report progress from the training process with the universal typed
client:

```python
from ctrl_pi import CtrlPiClient


with CtrlPiClient("http://ctrl-pi-box:8000") as trainer:
    run = trainer.create_run(
        "ACT block pickup",
        dataset_repo="my-org/yam-block-pickup",
        base_model="lerobot/act_aloha_sim_transfer_cube_human",
        runtime="lerobot",
        framework="pytorch",
        config={"batch_size": 32, "learning_rate": 1e-4},
    )
    run_id = run.id
    trainer.update_run(run_id, status="running")
    trainer.log_metrics(run_id, step=100, metrics={"train/loss": 0.82})
    trainer.log_console(
        run_id,
        source="stdout",
        line="step 100: checkpoint upload started",
        step=100,
    )
    trainer.update_run(run_id, status="completed", current_step=100)
```

`current_step` never decreases. An explicit regression returns HTTP 409,
while metric and checkpoint calls retain the maximum observed step. Status can
be corrected or resumed; this release intentionally does not impose a separate run
state machine.

Existing integrations can continue importing `Client` and
`TrainerClientError` from `ctrl_pi.trainer`; that compatibility client shares
the hardened HTTP transport and retains its established dictionary-returning
methods. New integrations should use `CtrlPiClient` and its typed Pydantic
responses. No managed-training job launch or cancellation API exists in this
release.

## Security boundary

The Trainer API has no authentication because ctrl-π itself has none. Use it
only over a trusted LAN. Never include tokens, database URLs, passwords,
private keys, or other credentials in `config`; secret-like keys are rejected,
but the training process remains responsible for its own secret manager.

For the typed product-wide surface, use [REST and Python SDK](/python-sdk).
For every focused Trainer compatibility method, REST endpoint, validation
limit, and an executable LeRobot fine-tune example, use
[Trainer API](/trainer-api). For artifact layout and deployment compatibility,
continue to [Models](/models).
