---
title: "Troubleshooting"
description: "Diagnose setup, recording, Hugging Face, training, Modal, and YAM failures safely."
icon: "wrench"
---

Start with the browser's sanitized error and the backend log. Do not print
`.env`, request headers, provider response bodies, presigned URLs, serial
paths, or local Hugging Face cache paths while diagnosing a problem.

## Setup and startup

**A managed training job failed with "provider launch is still being
reconciled" or stopped soon after launch.** Check that exactly one ctrl-π
backend instance is running against your `DATABASE_URL`. Concurrent instances
(a second dev server, a leftover container, a parallel `uvicorn`) reconcile
the same durable jobs and interfere with each other's provider supervision.
Stop the extras and inspect the original durable job. While no App is visible,
ctrl-π keeps checking for a late provider mutation until the hard deadline.
An exact App with zero running tasks can still be deploying or building its
image before spawn, so ctrl-π watches that identity through the deadline. It
does not claim the job is supervised or recover its missing FunctionCall from
that listing. If the App reports an active task without a durably persisted
FunctionCall identity, ctrl-π stops it promptly and verifies zero tasks. Retry
only after the original job is terminal with teardown verified, using a new
output repository and idempotency key.

### The setup banner says the backend is unavailable

Confirm the backend is listening on port 8000:

```bash
curl --fail http://127.0.0.1:8000/api/health
```

In source development, run Vite separately on port 5173 and keep its `/api`
and `/ws` proxy pointed at port 8000. In Docker, inspect:

```bash
docker compose ps
docker compose logs --tail=200 app
```

### Database-backed pages return HTTP 503

`DATABASE_URL` is blank, unreachable, or not migrated. Alembic does not load
`.env` during a source command, so export the value explicitly:

```bash
export DATABASE_URL='postgresql://ctrl_pi:password@localhost:5432/ctrl_pi'
.venv/bin/alembic -c alembic.ini upgrade head
```

Inside Docker, `localhost` names the container, not the database host.

### Settings remains incomplete in mock mode

Mock mode requires only a healthy PostgreSQL connection and connected mock
arms. Confirm `CTRL_PI_MOCK_MODE=true`, restart the backend, and choose
**Recheck**. Hugging Face and Modal remain optional until their workflows are
used.

## Arms and teleoperation

### Telemetry is offline or reconnecting

Check `/api/arms` and the backend log. Hardware mode intentionally stays
disconnected after a configuration, plugin, device, sample, or worker failure;
it never falls back to mock arms. Open **Settings → YAM onboarding** and read
the sanitized diagnostic.

- For a missing device, choose **Discover devices**, correct host visibility,
  and rerun **Run read-only preflight**. A saved setup with explicitly enabled
  automatic connection can make one connection attempt when a hot-plug changes
  readiness to ready.
- For a changed selection, rediscover, preflight, and save before connecting.
- For a latched connection, sample, worker, or command error, inspect the
  physical rig and use acknowledged manual **Connect** after correction, or
  restart the backend. Automatic restoration does not retry vendor failures in
  a loop.

Run the non-opening preflight before touching a real bus:

```bash
make yam-probe
```

The command-line probe uses the process configuration. The Settings flow is
the authoritative path for the persisted physical setup and its connection
consent; see [YAM setup](/yam-setup).

### A jog, teleop start, or inference start returns HTTP 409

Another mode owns the process-local `RigLease`, or the selected lifecycle is
already changing. Stop the active episode before teleop, stop teleop before a
different owner, and stop inference with verified teardown. Commands are not
queued.

### Cartesian jog is rejected on real YAM

This is expected. The pinned real follower accepts joint-space and gripper
commands only. Cartesian jog remains available in the complete mock driver.

## Recording and upload

### An episode fails to start

Verify all of the following:

- FFmpeg exists and reports the `libx264` encoder;
- leader and follower roles are distinct and connected;
- the staging directory is writable with enough free space; and
- no manual command or inference session owns the rig.

```bash
ffmpeg -hide_banner -encoders 2>/dev/null | grep libx264
```

### Teleop cannot stop

Stop and finalize the active episode first. ctrl-π refuses to end teleop while
capture or FFmpeg finalization is still active so a sample is not silently
lost.

### A Hub upload failed

Raw MP4, JSONL, and manifests are retained. A retry after remote mutation is
locked to the originally persisted repository target and succeeds only when
the remote `.ctrl-pi.json` ownership marker matches the same recording. Do not
delete staging manually or choose an unrelated existing repository.

### A public upload is disabled

Clear **Private dataset** only when publication is intentional, then confirm
the warning. ctrl-π does not choose a data license; add an appropriate license
to the dataset card separately.

## Datasets and models

### Hub returns 403, 502, or 503

| Status | Action |
| --- | --- |
| 403 | Verify the token identifies the exact `HF_NAMESPACE` user or organization and has access to the private repository. |
| 502 | Retry after checking Hub availability; ctrl-π has not changed the catalog. |
| 503 | Set both `HF_TOKEN` and `HF_NAMESPACE`, restart, and recheck Settings. |

### A dataset does not appear

The catalog includes only repositories in the exact namespace with the
LeRobot tag. Refresh to bypass short caches and verify the dataset card/meta
files were finalized by LeRobot.

### Episode playback says the revision changed

The list and detail must use one immutable commit. Reload the dataset, which
refreshes the episode list and selects its current SHA.

### A model appears but cannot deploy

Checkpoint artifact paths are not Git revisions. The selected model commit
must contain a deployable LeRobot policy at the repository root. Review
[Models](/models#deployable-lerobot-layout).

## Inference and Modal

### A managed training launch returns 409

V1.1 permits exactly one nonterminal managed job and has no queue. Inspect
`list_managed_training_jobs()` or the Training page. Reusing the same
idempotency key with an identical request returns the original job; reusing it
with different fields also returns `409`. Do not change keys and retry until
the original response/job state is authoritative.

### A managed job is finalizing or cancelling for a long time

Execution outcome and provider cleanup are separate. The job remains
nonterminal until the exact owned Modal App is stopped or absent with zero
tasks. Keep the original Modal account/profile configured and let
reconciliation retry. If cleanup cannot converge, run `make modal-panic`,
require zero active inference and training Apps, then restart ctrl-π and
inspect the job. Do not treat a final Hugging Face revision alone as proof that
GPU billing stopped.

### Managed history reports an event gap

Modal event tails and PostgreSQL observability are bounded. A gap means some
console/metric history was unavailable after restart or retention; it does not
by itself invalidate an exact verified final model revision. Review the final
status, step, output repo/revision, and teardown result together.

### Deploy is disabled in hardware mode

All three provider checks must be ready: Hugging Face namespace access, Modal
API credentials/profile, and the separate Modal Proxy Token pair. API tokens
begin `ak-`/`as-`; proxy tokens begin `wk-`/`ws-` and are not interchangeable.

### Real OpenPI deployment fails before creation

This is expected in V1. OpenPI has a deterministic mock adapter only. Real
Modal policy execution supports LeRobot; the backend rejects real OpenPI
before creating a provider resource.

### A stopped or failed deployment is not teardown-verified

Do not assume billing stopped. Retry **Stop** while the backend has the same
Modal account configuration. If ordinary cleanup was interrupted, run the
exact-ownership panic command:

```bash
make modal-panic
```

Require its zero-active confirmation. The command does not finalize local
recordings or rotate/delete Proxy Tokens. See [Modal operations](/modal-operations).

### Inference actions are dropped or stop unexpectedly

Inspect live latency, queue depth, dropped chunks, and the sanitized last
error. ctrl-π rejects oversized, stale, non-finite, out-of-order, or
identity-mismatched action chunks. A real arm also requires exactly seven
native values and must pass per-step safety limits.

## REST and Python SDK

### A `CtrlPiClient` call fails with a generic message

Inspect `CtrlPiError.status_code` and the matching sanitized backend log. The
client intentionally omits response bodies, credential-bearing URLs, and raw
transport exceptions. It also rejects redirects and base URLs containing
userinfo, a query, fragment, or path prefix. Connect it to the ctrl-π root URL,
for example `http://127.0.0.1:8000`, on a trusted network.

### Closing the client did not stop an operation

This is intentional. `close()` releases only local HTTP resources; it does
not stop teleoperation, finalize an active episode, stop inference, or tear
down Modal. Read the current state after an uncertain response and call the
matching stop method explicitly. Never blindly retry motion or lifecycle
mutations. See the cleanup examples in [REST and Python SDK](/python-sdk).

## Safe recovery order

1. Stop robot writes and use the physical emergency stop when hardware state
   is uncertain.
2. Stop/finalize local recording work.
3. Stop the exact provider resource and require teardown proof.
4. Preserve unuploaded recording staging.
5. Restart the single backend process only after the rig is safe.

For persistent failures, collect sanitized ctrl-π version, mode, endpoint
status codes, lifecycle IDs, and timestamps—never credentials or raw provider
payloads.
