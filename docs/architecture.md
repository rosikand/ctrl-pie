# Architecture

ctrl-π is a single-user, self-hosted control plane for the YAM robot-learning
workflow. Application traffic and private assets never bypass the backend to
reach hardware, PostgreSQL, Hugging Face, or Modal. One FastAPI process on the
arm-connected machine owns those boundaries; user-clicked Hub links may still
open the corresponding Hugging Face page in a new browser tab.

```text
Browser
  │  same-origin HTTP, MJPEG, byte ranges, and WebSockets
  ▼
FastAPI control plane ───────────────► PostgreSQL
  │       │       │                    small durable metadata
  │       │       └──────────────────► Hugging Face Hub
  │       │                            datasets and model artifacts
  │       └──────────────────────────► Modal
  │                                    inference workloads only
  ▼
YAMDriver (mock or real standard-YAM adapter) + V1 MockCamera
  │
CAN / USB / local mock rig
```

The production container serves the built Vite application and API from one
origin on port 8000. Source development runs Vite on port 5173 and proxies
`/api` and `/ws` to Uvicorn on port 8000. See [Development](development.md)
and [Docker deployment](docker-deployment.md) for the two launch paths.

## Boundaries and components

| Layer | Responsibility | Representative code |
| --- | --- | --- |
| React frontend | Five workflow tabs and Settings; renders typed API state only | `frontend/src/` |
| FastAPI routers | Validate browser input, return sanitized errors, and serialize public state | `backend/src/ctrl_pi/api/` |
| Hardware boundary | Typed cached arm snapshots, bounded jogs/actions, and fail-closed lifecycle | `drivers/yam.py`, `drivers/mock_yam.py`, `drivers/real_yam.py` |
| Recording service | Teleop, camera capture, FFmpeg lifecycle, and synchronized samples | `recording.py`, `camera.py` |
| Hub services | LeRobot conversion/upload and SHA-pinned dataset/model browsing | `hf.py`, `hf_datasets.py`, `hf_episodes.py`, `hf_models.py` |
| Training tracker | PostgreSQL run metadata plus the synchronous reporting client | `api/trainer.py`, `trainer.py` |
| Compute boundary | Provider-neutral deployment identity, health, inspection, and verified stop | `compute.py`, `compute_stub.py`, `compute_modal.py` |
| Runtime boundary | Strict observation/action envelopes and LeRobot policy execution | `inference_runtime.py`, `runtime_lerobot.py`, `inference_transport.py` |
| Inference orchestration | Arm loop, live metrics, passive recording, and teardown ordering | `robot_inference.py`, `inference_sessions.py` |

Framework- and provider-specific objects stop at these boundaries. React does
not import YAM, LeRobot, Hugging Face, or Modal concepts beyond the public API
fields it renders.

## State and ownership

ctrl-π deliberately separates four kinds of state.

### PostgreSQL: durable control-plane metadata

PostgreSQL stores robots, recording summaries, training runs and their bounded
metric/checkpoint metadata, inference endpoints/deployments, and non-secret
settings. Alembic owns the schema. Robot rows use a database UUID internally
and a stable hardware-facing `driver_id` such as `yam-follower` at API and
driver boundaries.

PostgreSQL never stores live joint telemetry, camera frames, raw observation
images, action queues, full policy artifacts, MP4 bytes, or LeRobot parquet
files.

### Hugging Face Hub: durable artifact storage

Uploaded demonstrations are canonical LeRobot v3 dataset repositories. Model
weights and cards remain model repositories produced by external training
tools. Discovery is restricted to the configured `HF_NAMESPACE`; private
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

Modal Apps are external ephemeral resources. Their identity and lifecycle are
represented in PostgreSQL, but a stopped deployment must also be proven
stopped at the provider with zero running tasks.

## Control flows

### Arms and teleoperation

`YAMDriver` returns typed point-in-time snapshots. `/ws/arms` publishes those
snapshots without persistence. Mock mode selects the complete in-memory
`MockYAMDriver`. Hardware mode selects the lazy real adapter for one standard
YAM SocketCAN follower, one GELLO/Dynamixel leader, and the `crank_4310`
gripper. It never substitutes the mock after a hardware failure.

The real adapter owns vendor devices in one backend process and refreshes a
cache on a 50 Hz target loop. Public telemetry reads copy that cache; they do
not discover devices or send commands. Pose is model-derived MuJoCo forward
kinematics. Leader effort is absent and temperatures may be absent, while
gripper force and bus error counters are unavailable in the pinned plugin and
cross the API as `null`. Driver-reported timing measures ctrl-π's sampling loop
rather than motor firmware. A sample, vendor-worker, or command failure marks
both arms disconnected and latches motion unavailable.

The camera remains `MockCamera` in V1 even when real arms are selected, so
recordings from hardware mode still contain the documented synthetic video.

Manual jog, teleoperation, and inference all write through the same driver and
compete for one process-local `RigLease`. The lease fails immediately on a
conflict; it is not a distributed lock or a queue. The real adapter accepts
only follower joint/gripper commands; Cartesian jog is unsupported.

Teleop samples the leader, converts it to an absolute `ArmAction`, and applies
that action to the follower. Recording adds camera capture and synchronized
observation/action samples without changing the driver boundary. The complete
lifecycle is in [Recording and teleoperation](recording.md).

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

Training never executes inside ctrl-π. External scripts use the REST API or
the synchronous `ctrl_pi.trainer.Client` to create runs, report bounded scalar
curves and sanitized console lines, and register model repository revisions.
PostgreSQL stores bounded JSONB metric and console tails. The selected run is
refreshed with non-overlapping, visibility-aware polling; the browser never
attaches to a trainer or provider console. The external training process still
owns execution and uploads weights to Hub. See the [Trainer API](trainer-api.md).

### Models

The first-class Models route reads repository, immutable revision, checkpoint,
and card metadata from the configured Hugging Face namespace through the
backend. The browser receives no Hub token and never uploads or mutates model
artifacts; external training tooling remains responsible for weights.

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

1. start the selected driver off the asynchronous event loop;
2. enable and reconcile the recording manager;
3. tear down unattended resources through the provider adapter named by each
   row's persisted target kind, including rows left by a mock/hardware mode
   switch, without calling a runtime web endpoint;
4. start the deployment-lifetime watchdog.

Real-driver construction and imports are lazy. On hardware startup it validates
the narrow standard-YAM/GELLO/crank configuration, connects with interactive
calibration disabled, takes a first sample, and then starts cached sampling. A
configuration, plugin, file, permission, or connection failure leaves the API
running with stable disconnected arm identities and a sanitized Settings
diagnostic. It never falls back to mock hardware and does not reconnect until
the backend restarts.

No loop is auto-resumed after a crash or restart. Interrupted upload statuses
are reconciled to a safe retryable failure when no in-process uploader owns
them. Recording statuses such as `teleop` or `recording` are normalized when
their process-local task no longer exists. Complete episode manifests are
reconciled idempotently into PostgreSQL; incomplete partial or mixed episode
directories are removed. A valid manifest under a recording root unknown to
the connected database is preserved, as is unrelated content if the staging
path was accidentally pointed at a shared directory.

Shutdown first asks the inference session manager to stop arm writes and
verify compute teardown. It then joins or aborts recording work, closes
FFmpeg, and releases rig ownership before driver shutdown rejects commands,
stops sampling, asks the follower plugin for its safe mode, and closes both
devices. That plugin call is not an electrical torque-free guarantee; physical
emergency-stop and firmware safety remain independent requirements. Compose
supplies a 90-second grace period. If Modal cleanup remains uncertain, use the
procedure in [Modal operator cleanup](modal-operations.md).

The default `make yam-probe` checks hardware-mode configuration, exact pinned
package metadata, readable model/calibration files, leader serial permissions,
and whether the named network interface exists. It does so without importing
vendor modules, constructing their devices, or opening a bus. The explicit
`python -m ctrl_pi.yam_probe --connect` path then opens both devices, obtains
connected telemetry, emits a sanitized summary, and always attempts bounded
driver shutdown. It does not issue an application jog/action but does start
vendor bus/control behavior. The real adapter has been tested with fake vendor
objects only in this cloud workspace.
Joint mapping, calibration contents, CAN type/bitrate, physical limits, failure
response, and container device access still require the Ubuntu/YAM procedure
in [YAM driver interface](yam-driver.md).

## Trust and security boundary

V1 has no ctrl-π login, API key, or multi-user authorization. Run it only on a
trusted, firewalled host or LAN; do not publish port 8000 or the Vite
development server directly to the Internet.

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
transport at the network boundary.

This boundary assumes the machine and configured external accounts belong to
one operator. Authentication, multi-user collaboration, message queues,
Kubernetes, and additional compute providers are intentionally outside V1.

## Further reading

- [Setup and configuration](setup.md)
- [Development](development.md)
- [YAM driver interface](yam-driver.md)
- [Recording and teleoperation](recording.md)
- [Compute and inference](inference.md)
- [Trainer API](trainer-api.md)
- [Docker deployment](docker-deployment.md)
- [Full mock smoke gate](smoke-test.md)
