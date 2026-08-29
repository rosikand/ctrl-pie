---
title: "Training"
description: "Observe managed Modal jobs and external runs with bounded metrics, console output, and checkpoints."
icon: "brain-circuit"
---

**Training** is a first-class primary page for two workflows:

- managed SmolVLA jobs launched through LeRobot 0.4.4 from the Python SDK on
  your Modal account; and
- LeRobot, OpenPI, or custom scripts on compute you operate, reporting small
  control-plane events through the [Trainer API](/trainer-api).

```text
Python SDK / external trainer ──► ctrl-π Trainer API ──► PostgreSQL
                                      │
                                      ├──► Modal (managed SmolVLA only)
                                      └──► Hugging Face model repository
```

Model weights never pass through the browser or PostgreSQL. External scripts
upload with their normal Hugging Face tooling. A managed worker uploads
directly from Modal to an exactly marked repo in `HF_NAMESPACE`; ctrl-π stores
only repository IDs, immutable revisions, steps, and lifecycle metadata.

## Training page

The page groups active `created`/`running` runs separately from
past `completed`/`failed`/`cancelled` runs. Select one to inspect:

- current status and monotonic step;
- source dataset and base model;
- runtime and framework labels;
- bounded JSON configuration;
- one chart per reported scalar metric;
- the bounded, incrementally polled console tail;
- output model plus registered checkpoint revisions; and
- for managed jobs, compute/deadline, execution and provider state, event-gap
  truth, final artifact revision, and verified teardown.

Metric names may use letters, numbers, `.`, `_`, `-`, and `/`. Reporting the
same metric name and step replaces that point deterministically. ctrl-π stores
at most 128 metric series, 10,000 total points, and 512 checkpoint records per
run.

Console records have a server-assigned sequence and timestamp plus a
`stdout`, `stderr`, or `system` source. The server keeps only the newest 1,000
records and trims to a compact 512 KiB ceiling; the UI shows a retention-gap
notice when older sequences are unavailable. Logs are control-plane progress
records. External logs are reported by the trainer client; managed logs are
parsed only from ctrl-π's versioned worker events rather than attaching to
arbitrary Modal consoles.

## Managed Modal training

Launch managed work with `CtrlPiClient`, not from the browser. The browser has
no training form or one-click research control; it observes the same durable
job and run that the SDK creates. See [Managed training](/managed-training)
for the exact request, all nine GPU choices, lifecycle and cancellation,
artifact ownership, mock behavior, and recovery procedure.

Only one nonterminal managed job is allowed at a time in V1.1. This is a hard
cost guard, not a queue. A second launch returns `409` until the first job has
both an execution outcome and provider-verified teardown. Lambda Labs, Auto
sizing, OpenPI managed training, and user-supplied code are not implemented.
The managed worker accepts only its validated built-in SmolVLA checkpoint
shape; report ACT and other policy-family training as external runs.

## Report an external run

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
be corrected or resumed for an external run; ctrl-π intentionally does not
impose the managed-job state machine on external reporters. Managed runs reject
external status, metric, log, and checkpoint mutation with `409` because only
their verified worker/supervisor may advance them.

Existing integrations can continue importing `Client` and
`TrainerClientError` from `ctrl_pi.trainer`; that compatibility client shares
the hardened HTTP transport and retains its established dictionary-returning
methods. New integrations should use `CtrlPiClient` and its typed Pydantic
responses. Both clients also expose the managed-job launch, inspect, bounded
logs/metrics/checkpoints, and cancellation methods added in V1.1.

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
