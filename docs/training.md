---
title: "Training"
description: "Track external training runs, scalar metrics, configuration, and checkpoint reports."
icon: "brain-circuit"
---

ctrl-π does not execute training in V1. LeRobot, OpenPI, or custom scripts run
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

## Runs view

The **Runs** subview groups active `created`/`running` runs separately from
past `completed`/`failed`/`cancelled` runs. Select one to inspect:

- current status and monotonic step;
- source dataset and base model;
- runtime and framework labels;
- bounded JSON configuration;
- one chart per reported scalar metric; and
- output model plus registered checkpoint revisions.

Metric names may use letters, numbers, `.`, `_`, `-`, and `/`. Reporting the
same metric name and step replaces that point deterministically. ctrl-π stores
at most 128 metric series, 10,000 total points, and 512 checkpoint records per
run.

## Report a run

Install the client from the same checkout as the backend:

```bash
python -m pip install -e ./backend
```

Then report progress from the training process:

```python
from ctrl_pi.trainer import Client


with Client("http://ctrl-pi-box:8000") as trainer:
    run = trainer.create_run(
        "ACT block pickup",
        dataset_repo="my-org/yam-block-pickup",
        base_model="lerobot/act_aloha_sim_transfer_cube_human",
        runtime="lerobot",
        framework="pytorch",
        config={"batch_size": 32, "learning_rate": 1e-4},
    )
    run_id = run["id"]
    trainer.update_run(run_id, status="running")
    trainer.log_metrics(run_id, step=100, metrics={"train/loss": 0.82})
    trainer.update_run(run_id, status="completed", current_step=100)
```

`current_step` never decreases. An explicit regression returns HTTP 409,
while metric and checkpoint calls retain the maximum observed step. Status can
be corrected or resumed; V1 intentionally does not impose a separate run
state machine.

## Security boundary

The Trainer API has no authentication because ctrl-π itself has none. Use it
only over a trusted LAN. Never include tokens, database URLs, passwords,
private keys, or other credentials in `config`; secret-like keys are rejected,
but the training process remains responsible for its own secret manager.

For every client method, REST endpoint, validation limit, and an executable
LeRobot fine-tune example, use [Trainer API](/trainer-api). For artifact layout
and deployment compatibility, continue to [Models](/models).
