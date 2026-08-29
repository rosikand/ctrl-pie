---
title: "Managed training"
description: "Launch bounded SmolVLA jobs on Modal, follow artifacts, cancel safely, and verify teardown."
icon: "server-cog"
---

Managed training runs one validated built-in SmolVLA policy shape through
LeRobot 0.4.4 on GPU compute in **your** Modal account. It is launched through the REST/Python API; the
Training page observes it but deliberately has no launch form.

```text
CtrlPiClient ──► durable job + Training run in PostgreSQL
                         │
                         ├──► exact owned Modal App / FunctionCall
                         │          │
                         │          └──► bounded events
                         └──► exact marked Hugging Face model repo
```

## Prerequisites

<Warning>
  Run exactly one ctrl-π backend instance per `DATABASE_URL`. Every backend
  boot reconciles persisted managed jobs and owned provider resources; a
  second instance (including a stray development server, container, or
  in-process boot) that shares the database supervises the same durable jobs
  and can stop compute the other instance is still launching or watching. A
  backend built from older code is especially destructive: it reconciles
  newer rows and provider resources with rules that predate them.
</Warning>

Before a real launch, configure and verify:

- `DATABASE_URL` and an up-to-date Alembic schema;
- `HF_TOKEN` with access to the selected inputs and write access to
  `HF_NAMESPACE`;
- a **new** output model repo ID whose namespace is exactly `HF_NAMESPACE`;
- Modal API credentials (`MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`, or an active
  pinned-SDK profile); and
- the trusted-LAN boundary described in [Configuration](/configuration).

The output repo is private by default. Making it public requires an explicit
risk acknowledgement. ctrl-π writes a job/request ownership marker before it
creates compute and refuses an existing or mismatched repo. It does not choose
a license for your model.

Repository creation is deliberately not rolled back. Once preparation may
have created the output repo or marker, ctrl-π retains it after launch failure,
cancellation, or an ambiguous provider response so that ownership evidence is
not destroyed. The original idempotency key continues to identify the same
durable job, and the output slug remains reserved in PostgreSQL. Inspect the
retained private repo and marker, then retry with a new output slug and a new
idempotency key. If you deliberately delete the Hub repo yourself, that does
not make the old slug reusable in the existing ctrl-π database.
If a provider launch response is lost before its App/FunctionCall identity is
persisted, the one-job slot stays occupied while ctrl-π watches the exact
deterministic App identity through the job deadline. A late App is stopped;
absence is not declared safe immediately after the interrupted request.

## Launch from Python

Use a caller-owned idempotency UUID and retain it until the server response is
known. Retrying the exact request with the same key returns the same job;
reusing the key for a different request returns `409`.

```python
import time
from uuid import uuid4

from ctrl_pi import CtrlPiClient


idempotency_key = uuid4()
with CtrlPiClient("http://ctrl-pi-box:8000", timeout=10.0) as ctrl:
    job = ctrl.launch_managed_training(
        idempotency_key=idempotency_key,
        name="SmolVLA block pickup",
        dataset_repo="my-org/yam-block-pickup",
        dataset_revision="0123456789abcdef0123456789abcdef01234567",
        base_model="lerobot/smolvla_base",
        base_model_revision="c83c3163b8ca9b7e67c509fffd9121e66cb96205",
        output_model_repo="my-org/yam-block-pickup-smolvla-v1",
        output_private=True,
        acknowledge_public_model_risk=False,
        acknowledge_compute_cost=True,
        compute_size="Modal: A10G",
        max_steps=10_000,
        batch_size=8,
        log_every=20,
        save_every=1_000,
        seed=42,
        num_workers=4,
        timeout_minutes=120,
    )

    try:
        while job.status not in {"completed", "failed", "cancelled"}:
            time.sleep(2)
            job = ctrl.get_managed_training_job(job.id)
    except BaseException:
        # Cancellation is idempotent. It does not claim success until the
        # exact owned App is stopped with zero running tasks.
        ctrl.cancel_managed_training_job(job.id)
        raise

    if not job.teardown_verified:
        raise RuntimeError("Managed training compute cleanup is not verified")
    if job.status != "completed" or job.output_revision is None:
        raise RuntimeError(job.last_error or f"Training ended as {job.status}")
```

Mutable refs such as `main` may be supplied, but ctrl-π resolves and persists
one lowercase 40-character commit before any Modal mutation. The public launch
payload contains only those immutable identities and a bounded declarative
training specification—never user code, shell text, environment variables,
tokens, or a callback URL. Modal injects `HF_TOKEN`
only into ctrl-π's trusted wrapper so it can fetch and publish exact artifacts;
the wrapper strips that token from the fixed LeRobot child process and never
places it in events or returned state.

V1.1 does not accept arbitrary LeRobot policies in managed compute. The pinned
snapshot must be a bounded built-in `type=smolvla` artifact with the expected
registry-backed processor sequence. ctrl-π rejects dynamic processor classes,
remote-code metadata, PEFT/adapters, executable files, unsafe links, and
unexpected configuration before starting the child. It forces CUDA and
offline Hub behavior, pins the non-weight SmolVLM dependency at
`7b375e1b73b11138ff12fe22c8f2822d8fe03467`, and bundles that dependency
into every uploaded checkpoint so the final root can be loaded by ctrl-π
inference without a nested Hub fetch. Use the external Trainer reporting API
for ACT or other policy families.

## Compute choices and limits

| SDK value | Modal allocation |
| --- | --- |
| `Modal: A10G` | 1× A10G |
| `Modal: A100` / `Modal: H100` | 1× selected GPU |
| `Modal: 2xA100` / `Modal: 2xH100` | 2× selected GPU through Accelerate |
| `Modal: 4xA100` / `Modal: 4xH100` | 4× selected GPU through Accelerate |
| `Modal: 8xA100` / `Modal: 8xH100` | 8× selected GPU through Accelerate |

Every request must acknowledge compute cost. One ctrl-π backend allows exactly
one nonterminal managed job: there is no queue and a concurrent launch returns
`409`. Each App has zero warm/buffer containers, at most one training
container, no provider retries, a single-use container, and the persisted
1–1,440 minute hard deadline. Multi-GPU choices invoke Accelerate with the
matching process count so paid GPUs are not intentionally left unused.

## Lifecycle and observability

| Job status | Meaning |
| --- | --- |
| `created` / `launching` | Durable state exists; inputs/output ownership and provider launch are being established. |
| `running` | The exact persisted FunctionCall is supervised. |
| `finalizing` | Execution has an outcome; App teardown or artifact verification is still pending. |
| `cancelling` | User/deadline cancellation is converging through verified teardown. |
| `completed` / `failed` / `cancelled` | Terminal only after App stop/absence and zero tasks were verified. |

The worker emits only versioned, job-bound JSON events. ctrl-π validates their
identity, monotonic sequence, repository/revision, step and finite metrics,
then reuses the existing bounded Training-run stores: at most 10,000 scalar
points, 512 checkpoints, and the newest 1,000 console records within 512 KiB.
The provider log tail is also bounded. If restart or retention loses part of
that tail, `event_gap` stays true and the Training page says the history is
incomplete; a separately verified final model can still complete.

Read the same state through:

```python
jobs = ctrl.list_managed_training_jobs(limit=50)
detail = ctrl.get_managed_training_job(job.id)
logs = ctrl.list_managed_training_logs(job.id, limit=200)
metrics = ctrl.get_managed_training_metrics(job.id)
checkpoints = ctrl.list_managed_training_checkpoints(job.id)
```

The corresponding routes live under `/api/trainer/jobs`. External reporting
under `/api/trainer/runs` remains compatible, but managed runs reject external
mutation so an unverified caller cannot forge progress or completion.

## Cancellation, restart, and panic cleanup

Cancellation first verifies FunctionCall-to-App identity, requests call
cancellation when that identity is available, then stops the exact App ID and
polls provider lifecycle plus task counts. A lost call ID never prevents exact
owned App cleanup. A provider or database outage leaves the job nonterminal
and retryable; it does not fabricate a safe terminal result.

A normal backend shutdown preserves an intentionally running job. On restart,
ctrl-π reattaches only the persisted call whose App ID, exact deterministic
name, and ownership tag all agree. It never relaunches an ambiguous/crashed
attempt. Completion while the backend is down is ingested on restart before
the App is stopped and terminal state is published.

If normal reconciliation cannot finish, run:

```bash
make modal-panic
```

Panic independently enumerates inference Apps named `ctrl-pi-<UUID>` and
training Apps named `ctrl-pi-training-<UUID>`, requires each corresponding
exact ownership tag, continues cleanup when an unrelated candidate fails, and
exits successfully only after the final listing proves zero active tasks. See
[Modal operations](/modal-operations).

## Mock mode

Mock mode uses a deterministic credential-free training target and makes no
Hugging Face or Modal calls. It exercises the same job/run/event/artifact-
identity/teardown state machine, but the Training page labels it **simulation ·
no GPU** and does not link the synthetic output revision as a real Hub
artifact.

No real managed Modal training job was launched while implementing V1.1.
Before relying on it for valuable training, perform a short private A10G job,
verify the final root loads through the inference runtime, cancel a second job
mid-run, and require `make modal-panic` to report zero active inference and
training tasks.
