---
title: "Architecture"
description: "Understand ctrl-π boundaries, ownership, persistence, control flows, and recovery."
icon: "network"
---

ctrl-π is a single-user, self-hosted control plane for the YAM robot-learning
workflow. Application traffic and private assets never bypass the backend to
reach hardware, PostgreSQL, Hugging Face, or Modal. One FastAPI process on the
arm-connected machine owns those boundaries; user-clicked Hub links may still
open the corresponding Hugging Face page in a new browser tab.

```text
Browser / ctrl_pi.CtrlPiClient
  │  HTTP; the browser also uses MJPEG, byte ranges, and WebSockets
  ▼
FastAPI control plane ───────────────► PostgreSQL
  │       │       │                    small durable metadata
  │       │       └──────────────────► Hugging Face Hub
  │       │                            datasets and model artifacts
  │       └──────────────────────────► Modal
  │                                    inference + managed SmolVLA training
  ▼
YAMDriver (four-arm mock or cell/legacy hardware adapter) + V1 MockCamera
  │
CAN / USB / local mock rig
```

The production container serves the built Vite application and API from one
origin. Its Uvicorn port is configurable; bridge mode defaults to 8000 and the
host-network YAM-cell path normally uses 8010. Source development runs Vite on
5173 and proxies `/api` and `/ws` to Uvicorn on 8000.

## Boundaries and components

| Layer | Responsibility | Representative code |
| --- | --- | --- |
| React frontend | Six workflow tabs plus Settings and YAM onboarding; renders typed API state only | `frontend/src/` |
| Python SDK | Typed, bounded access to the same public REST workflows; no direct hardware, Hub-token, or provider access | `client.py`, `sdk_models.py` |
| FastAPI routers | Validate browser input, return sanitized errors, and serialize public state | `backend/src/ctrl_pi/api/` |
| Hardware boundary | Typed multi-arm snapshots, stable CAN identity, supervised i2rt workers, bounded actions, and fail-closed lifecycle | `drivers/yam.py`, `drivers/mock_yam.py`, `drivers/yam_cell*.py`, `drivers/yam_i2rt_worker.py`, `drivers/real_yam.py` |
| Recording service | Teleop, camera capture, FFmpeg lifecycle, and synchronized samples | `recording.py`, `camera.py` |
| Hub services | LeRobot conversion/upload and SHA-pinned dataset/model browsing | `hf.py`, `hf_datasets.py`, `hf_episodes.py`, `hf_models.py` |
| Training control plane | External run reporting plus durable managed-job supervision and bounded event ingestion | `api/trainer.py`, `api/managed_training.py`, `managed_training.py`, `training_store.py` |
| Training compute boundary | Exact owned Stub/Modal job identity, fixed SmolVLA/LeRobot worker, provider inspection, cancellation, and verified stop | `training_compute.py`, `training_compute_stub.py`, `training_compute_modal.py`, `modal_training_workload.py` |
| Compute boundary | Provider-neutral deployment identity, health, inspection, and verified stop | `compute.py`, `compute_stub.py`, `compute_modal.py` |
| Runtime boundary | Strict observation/action envelopes and LeRobot policy execution | `inference_runtime.py`, `runtime_lerobot.py`, `inference_transport.py` |
| Inference orchestration | Arm loop, live metrics, passive recording, and teardown ordering | `robot_inference.py`, `inference_sessions.py` |

Framework- and provider-specific objects stop at these boundaries. React does
not import YAM, LeRobot, Hugging Face, or Modal concepts beyond the public API
fields it renders.

## State and ownership

ctrl-π deliberately separates four kinds of state.

### PostgreSQL: durable control-plane metadata

PostgreSQL stores robots, one non-secret `yam_cells` row plus normalized
`yam_cell_arms`, the readable legacy `yam_setups` row, recording summaries,
training runs and their bounded metric/checkpoint/log
metadata, durable managed-training lifecycle/identity/deadline state, inference
endpoints/deployments, and non-secret settings. Alembic
owns the schema. Robot rows use a database UUID internally
and a stable hardware-facing `driver_id` such as `yam-follower` at API and
driver boundaries. SocketCAN arms store USB-adapter serial; runtime `canN`
never becomes durable identity.

PostgreSQL never stores live joint telemetry, camera frames, raw observation
images, action queues, full policy artifacts, MP4 bytes, or LeRobot parquet
files.

### Hugging Face Hub: durable artifact storage

Uploaded demonstrations are canonical LeRobot v3 dataset repositories. Model
weights and cards remain model repositories produced by external tools or the
fixed managed SmolVLA/LeRobot worker. Managed output repositories are new, marked,
namespace-confined, and private by default; ctrl-π does not choose a model
license. Discovery is restricted to the configured `HF_NAMESPACE`; private
media is fetched and byte-range proxied by the backend at an immutable Hub
revision, so `HF_TOKEN` never enters browser code.

### Local staging: durable only on the host or Docker volume

Before upload, each finalized episode contains an MP4, synchronized JSONL
samples, and a bounded durable manifest below `RECORDING_STAGING_DIR`.
PostgreSQL holds only a relative artifact key and aggregate metadata. Preserve
this directory until the Hub revision is verified and the `uploaded` database
transition commits. Restoring PostgreSQL without the staging directory cannot
recreate an unuploaded episode. The production Compose volume persists this
data across normal container replacement, but `docker compose down --volumes`
deletes it.

After that durable upload transition, Hugging Face is the artifact source of
truth and ctrl-π removes the original raw staging. If immediate cleanup is
skipped or interrupted, startup observes the durable `uploaded` row and
retries it. Failed, cancelled, or uncertain uploads retain raw staging.

### Process memory: live state

Arm telemetry, camera frames, active teleop/inference tasks, rig ownership,
action queues, endpoint latency, frequency, and current episode duration live
only in one backend process. A restart resets these values and never resumes
motion. This is why ctrl-π must run with exactly one Uvicorn worker and must
not be horizontally scaled.

Modal Apps are external ephemeral resources. Inference deployments and managed
training jobs use disjoint deterministic names and ownership tags. Their
identity and lifecycle are represented in PostgreSQL, but a terminal state is
not safe until the exact App is stopped or absent with zero running tasks.

## Control flows

### Arms and teleoperation

`YAMDriver` returns typed point-in-time snapshots. `/ws/arms` publishes copies
without persistence. Mock mode selects a four-arm, two-pair `MockYAMDriver`.
Hardware mode selects a cell-aware adapter for configurable CAN teaching-handle
leaders and followers, with the narrow serial-GELLO adapter retained for
compatibility. It never substitutes a mock after hardware failure.

Passive discovery maps stable USB-CAN serials to current kernel interfaces
through bounded read-only sysfs inspection. For each explicitly connected CAN
arm, ctrl-π spawns one supervised Python child. That child verifies and imports
the exact operator-mounted i2rt checkout only when its filesystem reports the
kernel `ST_RDONLY` flag (Docker `/opt/i2rt:ro`); chmod and effective permissions
are not accepted as read-only proof. The child owns the YAM object and its
nested control threads, and exchanges bounded telemetry/actions over local IPC.
No Lux shell process or unauthenticated pair server is part of the product
boundary. Runtime faults revoke writes, latch the arm/pair, and never blindly
respawn.

The parent refreshes an active all-CAN target on a strict cadence. It
safe-idles after 125 ms without a refreshed action, before the child watchdog;
the child independently refuses commands older than 250 ms. After that
boundary, policy motion requires a fresh identity-checked command lifecycle
and teleop requires a fresh explicit sync boundary. No stale target is replayed
and one-shot all-CAN jog is disabled. These deadlines are command-age guards,
not field-measured loop-rate thresholds.

`YAMSetupManager` keeps the driver object stable while Settings discovers,
preflights, applies, and connects/disconnects selected logical arms. Mock setup
does not mutate the physical rows. General connection, unattended restoration,
and calibrated `linear_4310`/`crank_4310` jaw motion have backend-enforced
acknowledgements.
Resource-scoped `RigLease` ownership permits disjoint arms/pairs and rejects
overlap instead of treating every multi-arm workflow as one global lock.

The camera remains `MockCamera` in V1 even when real arms are selected, so
recordings from hardware mode still contain the documented synthetic video.

Teleoperation and inference write through the same driver and own explicit
logical-arm resource sets. The lease fails immediately on an overlap; it is not
a distributed lock or queue. The supervised all-CAN adapter rejects one-shot
jog in V1.2. Mock and retained legacy adapters keep bounded follower jog for
compatibility; teaching-handle leaders never accept it.

Teleop first validates/acquires one declared leader/follower pair and observes
it with synchronization disabled. It performs no follower write. A separate
freshly acknowledged action interpolates toward the latest mapped/clamped
leader pose over approximately three seconds before tracking. Disable stops
writes but retains pair telemetry. Recording captures post-map follower
commands plus camera/observation samples without changing the driver boundary.

### Dataset and model browsing

The backend authenticates to Hub with an explicit token, verifies that the
configured namespace belongs to the authenticated user or organization, and
post-filters results to that exact owner. Metadata and cards are resolved at a
40-character commit SHA and held in short in-memory caches. An unavailable or
malformed card degrades one item instead of failing the whole listing.

Episode detail reads LeRobot metadata/parquet at the same SHA. Video URLs carry
that revision through the backend proxy, preventing a mutable Hub `HEAD` from
mixing timeline data with a newer MP4. A detail response contains at most
2,000 deterministic, evenly spaced timeline frames and no more than one
million state/action scalar values; a lower scalar-derived limit applies to
wide vectors. The first and last frame are always included. The response
reports both the returned sample count and whether sampling truncated the
episode, and the UI labels sampled values accordingly.

The reader validates the episode index against Parquet file and row-group
bounds before returning data. It uses `ParquetFile` row groups and 1,024-row
batches to select only the requested offsets instead of converting the full
episode table to Python objects. Before decoding any batch, it rejects selected
row groups whose cumulative metadata exceeds one million rows or 16 million
state/action scalar slots. This keeps even a malicious 100-million-row single
row group from turning a bounded response into an unbounded scan; normal
LeRobot chunk files remain within the documented budget.

### Training

External scripts use the REST API, the synchronous universal
`ctrl_pi.CtrlPiClient`, or the compatible focused `ctrl_pi.trainer.Client` to
create runs, report bounded scalar curves and sanitized console lines, and
register model revisions. Separately, the SDK may create one durable managed
job for a fixed validated SmolVLA workload through LeRobot 0.4.4 on exactly owned Modal compute. It has
no arbitrary code, command, callback, Lambda, Auto-sizing, or OpenPI surface.
Both paths feed the same bounded PostgreSQL stores, and the Training tab
observes rather than launches or attaches to a raw provider console.

Managed launch resolves dataset/base refs to immutable commits, creates and
verifies a new marked output repository, commits the job before provider
mutation, and supervises structured bounded events. A caller UUID makes an
uncertain POST idempotent. Only one job may be nonterminal. Natural outcome,
cancellation, and hard deadline all converge through exact App stop/absence
and zero-task proof before the job becomes terminal. See [Managed training]
(/managed-training) and the [Trainer API](/trainer-api).

### Models

The first-class Models route reads repository, immutable revision, checkpoint,
and card metadata from the configured Hugging Face namespace through the
backend. The browser receives no Hub token and never uploads or mutates model
artifacts; external tooling or the fixed managed worker publishes weights.

### Inference

Provider deployment and robot motion are separate gates. Deploy first creates
one exactly owned Stub or Modal resource and verifies its runtime, model
repository, and immutable revision. A later Start selects one connected
follower, acquires the rig, verifies the same runtime identity again, and
runs one-in-flight observation/action streaming.

Idle state polling uses only bounded, cached provider lifecycle inspection;
it does not call the runtime web endpoint or wake a scaled-to-zero GPU. The
nonce/runtime identity request remains mandatory during Deploy and again at
explicit Start before the first arm write.

Optional inference recording is passive: it reuses the inference lease and
records only actions the loop actually applied. Stop joins local arm writes,
finalizes the episode, and then verifies provider teardown in a
cancellation-deferring path. Recording upload remains an explicit later
operation. Details are in [Compute and inference](inference.md).

## Startup, recovery, and shutdown

FastAPI lifespan startup performs these ordered operations:

1. restore the normalized physical cell (or readable legacy setup) into the
   stable driver and, only when explicitly enabled with all required consent,
   attempt its acknowledged automatic connection. A new cell remains closed
   until explicit save/connect;
2. enable and reconcile the recording manager;
3. tear down unattended resources through the provider adapter named by each
   row's persisted target kind, including rows left by a mock/hardware mode
   switch, without calling a runtime web endpoint;
4. reattach an exact persisted managed-training FunctionCall or fail closed
   and clean its exact App; never relaunch an ambiguous attempt or orphan;
5. start the deployment-lifetime watchdog while the setup manager passively
   watches a missing saved rig for prerequisites to appear.

Real-device construction and i2rt import are lazy and child-process isolated.
Discovery/preflight do not open devices. An acknowledged selected-arm connect
re-resolves stable identity, starts one owner per bus, accepts only an exact
ready handshake, and then publishes cached telemetry. Calibrated `linear_4310`
and `crank_4310` followers need a distinct calibration-motion acknowledgement. A configuration, source,
file, permission, worker, or connection failure leaves stable logical arms
visible with a sanitized diagnostic and never falls back to mocks. Missing
prerequisites may be passively rechecked; latched runtime/vendor faults are not
automatically retried.

No teleop, recording, inference, or policy-action loop is auto-resumed after a
crash or restart. Interrupted upload statuses
are reconciled to a safe retryable failure when no in-process uploader owns
them. Recording statuses such as `teleop` or `recording` are normalized when
their process-local task no longer exists. Complete episode manifests are
reconciled idempotently into PostgreSQL; incomplete partial or mixed episode
directories are removed. A valid manifest under a recording root unknown to
the connected database is preserved, as is unrelated content if the staging
path was accidentally pointed at a shared directory.

Shutdown stops local managed-training supervisors but preserves a healthy
running provider job, bounded by its immutable provider/deployment deadline,
for exact restart reattachment. A job already cancelling or finalizing still
receives an exact cleanup attempt. Shutdown then asks the inference session
manager to stop arm writes and verify inference compute teardown, joins or
aborts recording work, closes FFmpeg, and safe-idles before releasing rig
ownership. Driver shutdown revokes each worker, requests cooperative
safe-idle/close, then terminates, kills, and reaps on bounded deadlines. An
uncertain survivor keeps its bus blocked/error. No software call is an
electrical torque-free guarantee; physical emergency-stop and firmware safety
remain independent requirements. Compose
supplies a 90-second grace period. If Modal cleanup remains uncertain, use the
procedure in [Modal operator cleanup](modal-operations.md).

Cell preflight checks stable topology/resolution, link state, the exact local
read-only i2rt checkout/dependency marker, and optional map/limit artifacts
without importing or opening i2rt hardware. The separate handle check opens
only the acknowledged read-only encoder path. Connect, sync, and inference are
explicit all-CAN active boundaries; one-shot jog is disabled there. V1.2 has
fake-worker/mock validation only;
joint mapping, calibration, loop behavior, physical limits, failure response,
and least-privileged container access require the ordered Ubuntu-box procedure
in [YAM cell field acceptance](/yam-field-test).

## Trust and security boundary

V1.2 has no ctrl-π login, API key, or multi-user authorization. Run it only on
a trusted, firewalled host or LAN; do not publish Uvicorn ports 8000/8010 or
the Vite development server directly to the Internet.

Secrets come from the backend environment or Modal's local profile. They are
never accepted through Settings, stored in PostgreSQL, serialized in API
responses, placed in endpoint URLs, or sent to the browser. Modal API
credentials and Modal Proxy Tokens are separate credential pairs. Hub and
Modal calls use explicit credentials at their narrow service boundary, and
public errors discard provider response bodies and raw exception text.

Settings readiness is mode-aware. Mock setup requires PostgreSQL and the mock
arms; Hugging Face and Modal are reported truthfully but remain optional.
Hardware setup additionally requires an exact token-authenticated Hugging Face
user or organization namespace, a complete validated Modal API environment or
selected profile pair, a complete Modal proxy pair, and connected real arms.
The Modal readiness check validates local credential configuration rather than
claiming a live provider connection.

Private Hub videos use a same-origin backend proxy. Modal inference endpoints
require provider-native proxy authentication and accept only canonical HTTPS
`.modal.run` roots. Strict bounded JSON envelopes replace native pickle/gRPC
transport at the network boundary. Managed training uses Modal lifecycle API
credentials and a token-bearing trusted workload wrapper; tokens never enter
the public launch payload, job row, provider name/tag, LeRobot child process,
worker event, browser value, or log line.

This boundary assumes the machine and configured external accounts belong to
one operator. Authentication, multi-user collaboration, message queues,
Kubernetes, and additional compute providers are intentionally outside V1.1.

## Further reading

- [Setup and configuration](setup.md)
- [Development](development.md)
- [YAM driver interface](yam-driver.md)
- [Recording and teleoperation](recording.md)
- [Compute and inference](inference.md)
- [Managed training](managed-training.md)
- [Trainer API](trainer-api.md)
- [Docker deployment](docker-deployment.md)
- [Full mock smoke gate](smoke-test.md)
