# ctrl-π — SPEC

`ctrl-π` is an open-source, self-hosted web console for the basic robot-learning
workflow on YAM arms: collect demonstrations, browse datasets, track training,
and deploy policies for real-time inference.

This file is the canonical source of truth. When in doubt, do what this file
says. Record any ambiguity resolutions under "Decisions" at the bottom.

## Product shape

- User `git clone`s the repo, configures their own accounts (Postgres,
  Hugging Face, Modal), connects YAM hardware, and runs everything themselves.
- We host nothing and store no user data. There is no ctrl-π auth system in V1.
- The browser and the typed `ctrl_pi.CtrlPiClient` are peer clients of the
  same FastAPI control plane. Every meaningful product workflow is available
  through the public REST API; the Python SDK must not bypass services or add
  a privileged hardware/provider path.
- Durable control-plane state and uploaded artifacts live in services the user
  owns: their Postgres, their HF namespace, and their Modal account. Booting
  ctrl-π on a fresh machine with the same credentials restores that durable
  state. Finalized recordings that have not reached a verified Hub upload are
  still local staging data, however, so the operator must preserve or restore
  `RECORDING_STAGING_DIR` (the Docker `ctrl-pi-data` volume) as well; process-
  local live motion and capture sessions are deliberately never resumed.
- Target platform is Linux (Ubuntu box physically connected to the YAM arms
  over CAN/USB). Development on macOS against mocks is fine.
- Visual style: light Tailwind, sparse, simple, engineering-oriented. The
  README image is inspiration, not a pixel spec.

## Build order note

All hardware functionality is built first against a `MockYAMDriver` with clean
interfaces (`YAMDriver`) so the real driver can be swapped in later. Same for
anything requiring live Modal or HF credentials: real integrations are the
goal, but every feature must be runnable and demo-able in mock mode.

## Interface

Exactly six primary tabs, plus a small Settings page (gear icon, not a
primary tab).

### 1. Arms

Monitor and control connected YAM arms.

- connection/CAN status, leader/follower role
- real-time joint state, end-effector pose, gripper state
- control-loop statistics, basic diagnostics
- simple manual jog/control

Live state streams with low latency (WebSocket) and is never persisted.

### 2. Record / Teleop

Operate leader/follower pairs and collect demonstrations.

- select leader/follower arm pair
- live robot/camera state
- start/stop teleop; start/stop episode recording
- task/episode metadata; recording status, duration, episode count
- on save, upload the session as a LeRobot-format dataset repo
  (parquet + mp4 + meta) to the user's HF namespace

Optimize for the standard YAM + LeRobot workflow. Do not build a generic
robotics recording system.

### 3. Datasets

A browser over the LeRobot datasets in the user's configured HF namespace.

- dataset list with metadata/task, episode counts, revisions
  (rendered from dataset cards where available)
- episode browser: camera/video playback, synchronized state/action values,
  timeline scrubbing

Build incrementally: discovery/metadata first, richer episode visualizer
second. Private HF assets are fetched/proxied through the backend; `HF_TOKEN`
never reaches browser code.

### 4. Training

ctrl-π supports two complementary training paths:

**Trainer API** — a small reporting surface in the universal Python SDK and
REST API (with the legacy `ctrl_pi.trainer.Client` kept compatible) that
external training scripts call to:

- create/update a training run (status, current step, config)
- log scalar metrics over steps (loss curves rendered in the UI)
- report bounded, sanitized stdout/stderr/system console lines
- register the output HF model repo / checkpoint revisions

Run state goes to Postgres; weights go to HF Hub via the user's normal tooling.

**Managed Modal training** — the universal SDK and REST API can launch a
strictly validated SmolVLA job through LeRobot 0.4.4 on the user's Modal
account. A managed job is
durable control-plane state linked one-to-one with its Training run; it owns a
deterministically named and tagged Modal App and a newly marked Hugging Face
model repository. The public request selects only pinned dataset/base-model
revisions, bounded training parameters, an exact output repo, a hard deadline,
and one of these cost-acknowledged compute sizes:

- `Modal: A10G`
- `Modal: A100`, `Modal: 2xA100`, `Modal: 4xA100`, `Modal: 8xA100`
- `Modal: H100`, `Modal: 2xH100`, `Modal: 4xH100`, `Modal: 8xH100`

Launch is idempotent and asynchronous. The worker snapshots immutable inputs,
accepts only the built-in SmolVLA policy/configuration and registry-backed
processor pipeline, runs the pinned LeRobot training entry point (including
Accelerate for multi-GPU jobs), reports bounded structured events into the existing run
metrics/console/checkpoint stores, and explicitly uploads checkpoint folders
and one inference-compatible final root to HF. It never accepts user code,
shell commands, environment variables, secrets, or a callback URL. LeRobot's
implicit Hub uploader is disabled so ctrl-π can verify the exact output repo,
ownership marker, visibility, and immutable commit without choosing a license
for the user's model.

V1.1 permits exactly one nonterminal managed job per ctrl-π instance. This is
a hard cost guard, not a queue; a concurrent launch returns 409.

Completion, failure, cancellation, and deadline expiry all converge through
the same ownership-verified Modal teardown. A managed job cannot be reported as
terminal until App stop and zero running tasks are provider-verified. Cleanup
uncertainty remains visible and retryable. Backend restart reattaches an exact
persisted FunctionCall when it can prove the identity, never relaunches an
ambiguous job, and cleans up exact owned orphans. Graceful backend shutdown does
not cancel an intentionally running long-lived job; the provider deadline is
the independent backstop and startup reconciliation resumes supervision.

The Training tab is an experiment-tracking surface for both kinds of run:
current/past status and step, dataset, base model, runtime/framework, config,
live-polled metric curves and bounded console output, output model repo, and
registered checkpoints. Managed runs additionally show compute, provider and
execution state, deadline, event-gap truth, final artifact revision, and
provider-verified teardown. There is no browser launch form or one-click
training button.

Future work (do not build): Lambda Labs, Auto compute sizing, user-supplied
training code, and buttonized/agentic research workflows.

### 5. Models

A browser over model artifacts already stored in the user's configured HF
namespace.

- model repositories and card metadata
- immutable revisions and checkpoint files
- direct links to the corresponding Hub artifacts

Discovery is read-only. External training scripts or the fixed managed worker
remain responsible for producing and uploading weights. REST and the universal
SDK can select one of those immutable artifacts for inference, but do not
provide a raw model-file upload backdoor.

### 6. Inference

Deploy a policy to Modal and run it against the arms.

- select robot/arms, model/checkpoint (from HF), runtime (LeRobot or OpenPI),
  and compute size (dropdown; V1 options: "Modal: A10G", "Modal: A100",
  "Modal: H100"; "Auto" is future work)
- Deploy: ctrl-π spins up a serverless Modal inference server headless,
  loading the selected checkpoint from HF
- the ctrl-π backend (on the arm-connected box) runs the robot-side client:
  it streams observations to the Modal endpoint, receives action chunks back,
  and executes them on the arms
- Stop: tears down the Modal deployment **entirely**; nothing persists on
  Modal and no cost accrues after stop. This guarantee is a hard requirement.
- show endpoint health, latency / inference frequency, currently loaded policy
- inference/deployment sessions can optionally be recorded and uploaded as
  LeRobot datasets to HF, same pipeline as Record (learning from deployment)

Use each framework's native serving mechanism on Modal where practical
(LeRobot policy server, OpenPI policy server) behind a common
`InferenceBackend`/runtime-adapter interface. Keep framework-specific logic
out of frontend components.

Modal is the single first-class compute backend in V1. Design the adapter
interface so BYOC (user-managed GPU boxes, other clouds) can be added later,
but do not build it now.

### Real Modal inference is in V1 scope

The stub workload exists for unit tests and offline dev only. V1 must
**demonstrate a real policy running on real Modal GPU compute**, end to end:
mock arms produce observations → real Modal endpoint runs the policy →
action chunks stream back → executed on the mock arms, with latency and
frequency shown live in the Inference tab.

Demo policy: use a small pretrained LeRobot policy from the HF Hub (e.g. an
ACT or SmolVLA checkpoint for a standard task like PushT or ALOHA sim) as the
proof-of-life model. It is fine that its observation/action spaces don't match
YAM; the demo feeds it correctly-shaped synthetic observations and treats the
returned actions as opaque chunks to execute on the mock arms. The dropdown
still offers H100/A100 for real VLA checkpoints (π0.5-class); the demo policy
should default to the cheapest adequate GPU (A10G or L40S) to conserve credit.

Modal implementation notes:

- One Modal App per deployment, created programmatically by
  `ModalComputeTarget` (deploy from code; do not require the user to run
  `modal` CLI commands).
- Serve over a Modal web endpoint (WebSocket preferred; HTTP with chunked
  responses acceptable) that the robot-side client connects to.
- Bake model weights into the image or a Modal Volume at deploy time so
  cold start is paid once per deploy, not per request.
- Serverless discipline: no warm pools (`keep_warm`/`min_containers` = 0),
  idle scaledown ≤ 60s, and a hard per-deployment `timeout` (default 30 min)
  as a runaway backstop.
- Stop must call the Modal API to stop the app and verify via app list that
  nothing remains running. `make modal-panic` force-stops every ctrl-π Modal
  app; document it.

Cost guardrails (budget for all development and demos: ~$30 of Modal credit):

- Develop and test the deploy/teardown lifecycle against the stub first;
  touch real Modal only when the flow works end to end in mock mode.
- Real-GPU test sessions: cheapest adequate GPU, ≤ 10 minutes each, always
  torn down (verified) at the end, even on test failure. Never leave a
  deployment running unattended.

## Architecture

Frontend: React, TypeScript, Vite, Tailwind CSS.
Backend: Python, FastAPI, SQLAlchemy 2, Alembic, PostgreSQL, `huggingface_hub`,
`modal`.

```text
Browser / Python SDK ── LAN ──► ctrl-π backend (Ubuntu box) ── CAN/USB ──► YAM arms
                                      │
                                      ├── PostgreSQL (user's; Supabase recommended)
                                      ├── Hugging Face Hub (user's namespace)
                                      └── Modal (user's account; inference and managed training)
```

Key interfaces/components:

- `YAMDriver` / `MockYAMDriver`, a multi-arm `YAMCellDriver`, and the
  cell-aware `YAMSetupManager`
- `LeRobotRuntime`, `OpenPIRuntime` (runtime adapters)
- `ModalComputeTarget` (behind a generic `ComputeTarget` abstraction)
- `ManagedTrainingManager` / `ManagedTrainingTarget` (Modal plus a complete
  credential-free stub)
- universal typed REST/Python SDK (`ctrl_pi.CtrlPiClient`), with the Trainer
  reporting client retained as a compatibility facade

Boundaries: UI or SDK → ctrl-π API → services/interfaces → (Postgres | HF |
YAM | Modal). No YAM/CAN, Modal, or policy-framework details in frontend or
SDK code.

## Persistence

Three categories:

1. **PostgreSQL** — source of truth for small mutable control-plane state:
   robots/arms config, the one persisted physical YAM cell, recording
   metadata, training runs and managed-job lifecycle, inference
   endpoints/deployments, settings.
   Suggested tables: `robots`, `yam_cells`, `yam_cell_arms`, `recordings`,
   `training_runs`, `managed_training_jobs`, `inference_endpoints`,
   `deployments`, `settings`. The V1.1 `yam_setups` table remains readable as
   a legacy single-pair compatibility source, but new cell setup never stores
   an ephemeral `canN` as physical identity.
   SQLAlchemy + Alembic; depend only on standard Postgres via `DATABASE_URL`
   (Supabase is just the recommended host).
2. **Hugging Face Hub** — source of truth for artifacts: LeRobot datasets,
   model repos/weights/checkpoints, configs, dataset/model cards. Never
   duplicate large artifacts into Postgres.
3. **Ephemeral** — never persisted: live joints/pose, CAN health, loop
   frequency, camera streams, jog/teleop commands.

## Setup, configuration, credentials

- Secrets via a gitignored `.env`: `DATABASE_URL`, `HF_TOKEN`, and for Modal
  either `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` (plus
  `MODAL_PROXY_TOKEN_ID`/`MODAL_PROXY_TOKEN_SECRET` for the inference
  endpoint) or Modal's normal local credential file.
- A Settings page in the UI shows connection status for Postgres / HF / Modal
  / arms and lets the user enter or update non-secret config. Never store
  secrets in Postgres or HF.
- First-run experience: if required config is missing, the UI shows a clear
  setup checklist instead of broken pages.
- V1.2 YAM onboarding supports one configurable, YAM-specific cell containing
  zero or more leader/follower arms. Each arm has a stable logical ID, role,
  transport, durable physical identity, end-effector kind, and optional
  pair/group/side metadata. SocketCAN arms persist the USB-CAN adapter serial;
  passive discovery resolves that serial to the current runtime `canN` and
  never persists the interface name as identity. A retained legacy adapter
  supports the V1.1 serial-GELLO + CAN-follower topology.
- Passive discovery/preflight may inspect bounded sysfs, link state, files, and
  the explicitly mounted i2rt checkout, but never opens a device, imports a
  motion-capable vendor object, pings/enables a motor, calibrates a gripper, or
  changes CAN link state. Teaching-handle range health is a separate explicit,
  read-only CAN test; it reports stuck/unhealthy input but exposes no encoder-
  or joint-zeroing action.
- A passing physical cell can be persisted in PostgreSQL and applied to the
  existing `YAMDriver` object without a process restart. Explicit Connect may
  open only the selected arms and requires backend-enforced motion consent;
  connecting a calibrated `linear_4310` or `crank_4310` follower requires a
  distinct acknowledgement that its calibration moves the jaws. Connected followers are visibly holding/live
  until explicit disconnect joins local loops, returns torque to the safest
  supported state, and closes devices. Automatic boot/hot-plug connection is
  off by default and requires separate unattended-motion consent. Missing
  hardware remains fail-closed and is passively rechecked; latched runtime or
  vendor errors are never blindly retried.
- Mock onboarding implements the same discovery/preflight/apply/connect surface
  deterministically but cannot overwrite or delete the persisted physical
  setup and never implies physical discovery, calibration, or validation.

## Deployment & isolation

- The backend runs on the Ubuntu box attached to the arms; the browser
  connects over LAN.
- Ship a Docker-based option (`Dockerfile` + `docker-compose.yml`) so the
  whole app runs isolated from the rest of the user's machine, with USB/CAN
  device passthrough documented for the arms. Keep image size and overhead
  minimal. Plain `pip install` / `npm run dev` must also keep working for
  development.

## Non-goals for V1

Kubernetes, Redis, message queues, auth, multi-user collaboration, fleet
management, hosted ctrl-π, browser-buttonized or arbitrary-code training,
Auto sizing, BYOC/Lambda compute, automated VM provisioning, elaborate
dashboards, reimplementing LeRobot/OpenPI protocols, unnecessary abstractions.
Prefer boring, understandable, low-latency implementations.

## Naming

repo `ctrl-pi` · Python package `ctrl_pi` · universal client
`ctrl_pi.CtrlPiClient` · compatible trainer client `ctrl_pi.trainer.Client`.

## Addenda

- Repo layout: monorepo with `frontend/` (Vite app) and `backend/` (Python
  package `ctrl_pi`). Python 3.11+. Backend on :8000, frontend dev server on
  :5173 with a proxy to the backend.
- Mock camera: `MockYAMDriver` is paired with a mock camera source that
  generates synthetic frames (e.g. moving shapes + timestamp overlay), so
  Record/Teleop produces real mp4s and the episode visualizer has real video
  to play, all without hardware.
- LeRobot format: write datasets using the `lerobot` package's dataset
  tooling. Do not hand-roll the parquet/mp4/meta layout.
- Seed data: provide a `make seed` (or script) that populates mock robots, a
  couple of fake training runs with metric curves, and records + uploads one
  small sample dataset, so the UI is fully browsable immediately after setup.
- HF namespace: the target namespace is configured via `HF_NAMESPACE` in
  `.env`; never guess it.

## Milestones

1. App shell: six tabs + Settings, sparse Tailwind layout
2. Postgres schema + Alembic migrations + settings/first-run checklist
3. Mock Arms: `MockYAMDriver`, live telemetry over WebSocket, jog controls
4. Mock Record/Teleop: recording lifecycle, episode metadata
5. HF integration: namespace auth, dataset upload pipeline (LeRobot format)
6. Datasets tab: discovery + metadata/cards
7. Datasets tab: episode visualizer (video + synced state/actions + scrubbing)
8. Trainer API: REST endpoints, Python client, Training run analysis and a
   first-class Models tab,
   plus reference docs for the API (`docs/trainer-api.md`: install, quickstart
   script, every client method and REST endpoint, an end-to-end example of a
   LeRobot fine-tune logging to ctrl-π and pushing checkpoints to HF)
9. `ComputeTarget` abstraction + `ModalComputeTarget`: deploy/teardown
   lifecycle, first against the stub workload, then verified against real
   Modal with a trivial CPU workload (deploy, health check, teardown,
   confirm nothing left running)
10. Runtime adapters: LeRobot serving on Modal (OpenPI adapter stubbed to the
    same interface); robot-side client loop against `MockYAMDriver`
11. Inference tab: full deploy → stream → execute → stop flow, first in mock
    mode, then **live demo**: deploy the demo policy to real Modal GPU
    compute, run ≥ 60 seconds of real inference against the mock arms with
    live latency/frequency in the UI, record the session, upload it to HF as
    a LeRobot dataset, tear down, and verify teardown. Capture the evidence
    (latency stats, dataset repo link, teardown confirmation) in the
    milestone commit message.
12. Docker packaging + device-passthrough docs
13. Documentation: `docs/` with architecture overview, development guide,
    setup/configuration (env vars, Supabase, HF, Modal), YAM driver interface,
    recording/teleop workflow, compute targets and runtime adapters; a
    polished top-level `README.md` (what ctrl-π is, screenshot, quickstart);
    verify `docs/trainer-api.md` is complete and its examples run
14. Real `YAMDriver` integration points, polish, tests

After each milestone: run build/tests, commit as `milestone N: <name>`, note
open issues. Do not expand scope.

### V1.1 accepted evolution

V1.1 builds on the completed V1 in this order:

1. Training and Models become separate first-class tabs.
2. One YAM rig gains persisted, consent-gated first-time setup and automatic
   restoration.
3. The meaningful current control plane becomes available through one typed
   REST/Python SDK, while preserving the focused Trainer client.
4. The repository-hosted `docs/` tree becomes the production Mintlify
   documentation site, with a concise README, complete current-product
   navigation, validated examples and links, a screenshot gallery, and a
   changelog.
5. Managed, artifact-backed LeRobot training runs on the user's Modal account
   through the Trainer REST/Python API with bounded observability and verified
   teardown; Lambda, Auto sizing, arbitrary code, and a browser launch flow are
   deferred.
6. User-owned AI agents gain a canonical safety and integration guide over the
   universal SDK/REST surface. This adds no agent-specific API, privileged
   identity, or alternate control path.

## Decisions

(Agent: append dated entries here when resolving ambiguities.)

- 2026-08-28 — Milestone 1 includes a minimal FastAPI package, health test,
  and no-op Alembic bootstrap because the mandatory gate requires backend
  tests, a migration upgrade, and both dev servers after every milestone.
  Domain persistence remains milestone 2.
- 2026-08-28 — The milestone 3 mock rig exposes one leader and one follower,
  each with six named rotary joints plus a gripper. Jog offsets are bounded in
  memory and the mock pose is derived from those joints; the `YAMDriver`
  boundary keeps real hardware joint mappings out of the web/API layer.
- 2026-08-28 — Before milestone 5 uploads an episode, milestone 4 stages its
  MP4 and synchronized samples under the gitignored `.ctrl-pi/recordings`
  directory. PostgreSQL stores only lifecycle state, aggregate metadata, and
  staging pointers; Hugging Face becomes the artifact source of truth after
  upload.
- 2026-08-28 — Live arm IDs such as `yam-leader` are stable driver identities,
  not database UUIDs. The `robots.driver_id` column maps them to relational
  robot rows so recordings retain UUID foreign-key integrity while the API
  continues to use hardware-facing IDs.
- 2026-08-28 — Dataset writing is pinned to LeRobot 0.4.4, the current
  LeRobot v3 release compatible with the project's Python 3.11 floor. Hub
  uploads default to private, require an explicit confirmation to publish,
  and use an explicitly token-bound `HfApi` after LeRobot finalizes the
  package-generated dataset so credentials loaded from `.env` never depend on
  global login state.
- 2026-08-28 — Dataset discovery is limited to LeRobot-tagged repositories in
  the configured HF namespace and uses keyset pagination ordered by
  last-modified time then repository ID. Enumeration and SHA-pinned card
  metadata use short in-memory caches; an unavailable README or
  `meta/info.json` degrades
  only that dataset summary instead of failing the namespace listing.
- 2026-08-28 — Episode visualization reads the canonical LeRobot v3 metadata
  and parquet indexes, and pins the episode list, detail, and backend-proxied
  private MP4 to one immutable Hub SHA. Timeline values are episode-relative;
  packed-video timestamps provide the media seek offset, and the proxy serves
  byte ranges so native browser playback can scrub without exposing HF_TOKEN.
- 2026-08-28 — Training remains an external-script responsibility. (Superseded
  in part on 2026-08-29: V1.1 adds the SDK/REST-launched managed Modal
  training path described in its own decision below; external scripts remain
  fully supported.) The
  control plane stores only bounded JSONB scalar curves and checkpoint
  descriptors; row-locked, copy-on-write updates replace duplicate
  metric/step points and keep `current_step` monotonic without imposing an
  additional run-status state machine. Model discovery stays inside the
  configured HF namespace and pins cards/checkpoint paths to an immutable
  revision using an explicitly token-bound Hub client.
- 2026-08-28 — Each inference deployment owns exactly one deterministic
  compute app (`ctrl-pi-<deployment UUID>`) and an exact deployment ownership
  tag. The deployment row persists its `stub` or `modal` target kind so a
  restart or mode change can never stop a resource through the wrong
  provider. Stub and Modal targets share the same nonce-health and
  stop-with-verification lifecycle; only provider-proof transitions the
  atomic deployment/endpoint rows to `running` or `stopped`.
- 2026-08-28 — The Modal boundary is pinned to the Python 3.11-compatible
  `modal==1.5.4`. Deployment, App tags, and web-function lookup use public SDK
  APIs; task-count inspection, lifecycle proof, and stop-by-provider-ID use
  the same `AppList`, `AppGetLifecycle`, and `AppStop` protobuf calls as the
  pinned official CLI, isolated inside `ModalComputeTarget` because the stable
  App API does not yet expose those teardown primitives. Modal forbids `/` in
  App tag keys and values, so the provider losslessly represents the domain
  ownership tag as `ctrl-pi-deployment=<deployment UUID>`.
- 2026-08-28 — Runtime serving uses strict, bounded JSON observation/action
  envelopes over a provider-authenticated Modal POST/WebSocket surface rather
  than LeRobot's native pickle/gRPC transport. A real LeRobot adapter loads
  only an immutable local SHA-marked checkpoint through LeRobot 0.4.4's
  config, policy, preprocessing, and postprocessing APIs; OpenPI uses the same
  deterministic stub in mock mode while its real V1 adapter fails explicitly.
  After the revision-pinned image build, the serving image enforces Hub and
  Transformers offline mode so nested processor/model loads cannot resolve a
  mutable remote dependency at container start or inference time.
  Modal Proxy Tokens (`wk-`/`ws-`) are distinct backend-only configuration
  from Modal API credentials (`ak-`/`as-`). The robot loop holds the shared rig
  lease, verifies both runtime responses against the persisted deployment
  identity, permits one in-flight request, and ages every queued action before
  writing to the arm.
- 2026-08-28 — Provider deployment and robot motion are separate M11
  lifecycle gates: deployment first verifies the exact runtime/model/SHA,
  then an explicit start selects one connected follower and acquires the rig.
  Optional inference recording is a passive `RecordingManager` path opened
  under that same verified inference lease before the first action; it never
  starts teleoperation or acquires a competing lease. Stop joins local arm
  writes and finalizes the episode before verified provider teardown, while a
  restart never resumes motion and instead tears down unattended compute.
  Mock mode uses the immutable built-in identity
  `ctrl-pi/mock-policy@0000000000000000000000000000000000000000`, so all
  runtime selections remain deterministic and make no Hub or Modal calls.
- 2026-08-28 — The production Docker option is one non-root container where
  one Uvicorn process serves both the built Vite assets and the API and owns
  the process-local robot rig. PostgreSQL, Hugging Face, and Modal remain
  external user-owned services. The Ubuntu host configures SocketCAN; Docker
  receives only explicitly selected USB devices or an explicit host-network
  opt-in, never privileged access.
- 2026-08-28 — Hardware mode selects a fail-closed real driver for the
  standard SocketCAN YAM follower, GELLO serial leader, and `crank_4310`
  gripper using the exact compatible `0.1.1` YAM plugin releases. The pinned
  linear-gripper path is incompatible with LeRobot 0.4.4, so V1 rejects it
  before device access. ctrl-π keeps its normalized gripper contract
  (`0=closed`, `1=open`) by inverting the crank plugin's vendor boundary,
  maps every rotary joint explicitly by name, and derives pose from the
  configured MuJoCo model. Cloud tests cover the adapter with fake vendor
  devices; physical safety, directions, offsets, units, model fidelity, bus
  behavior, and emergency-stop operation remain mandatory validations on the
  target Ubuntu/YAM box.
- 2026-08-28 — A deployment snapshots the configured 1–30 minute lifetime in
  PostgreSQL when it is created. An in-process watchdog uses that immutable
  value to stop the robot loop before verified provider teardown, including
  when a deployment is idle; retryable teardown failures remain under
  reconciliation. Idle readiness is a short-lived cache of bounded,
  identity-checked provider lifecycle inspection and never calls the GPU web
  endpoint; deploy and explicit session start retain nonce/runtime identity
  health checks, while restart teardown never wakes the runtime endpoint.
  Cleanup is selected from each row's persisted target kind,
  including after a mock/hardware mode switch, and an unavailable cleanup
  target remains a loud retryable failure rather than false teardown proof.
- 2026-08-28 — Settings readiness is mode-aware: mock mode requires only a
  healthy PostgreSQL connection and mock arms, while hardware mode also
  requires exact token-authenticated Hugging Face namespace ownership,
  complete Modal API/profile and proxy credential pairs, and connected real
  arms. Modal readiness validates local credential configuration without
  claiming a provider connection, and public status exposes only sanitized
  text and booleans.
- 2026-08-28 — Episode detail responses are bounded to 2,000 deterministic
  timeline samples and one million state/action scalar values, with the first
  and last frame included. Additive sample-count/truncation metadata keeps the
  UI truthful. Parquet file/row-group bounds are validated, and selected row
  groups are rejected before decoding if they cumulatively exceed one million
  rows or 16 million state/action scalar slots; accepted groups are scanned in
  1,024-row batches rather than materializing an unbounded frame table.
- 2026-08-28 — A finalized ordinary or inference episode publishes a bounded,
  fsynced per-episode manifest atomically after both raw artifacts are durable.
  Startup reconciles those manifests idempotently into PostgreSQL and removes
  incomplete staging. Raw recording staging is deleted only after the Hub
  revision is verified and the `uploaded` database transition commits; it is
  retained on transfer failure, cancellation, or uncertain persistence. A
  later startup removes retained staging only when it observes an already
  durable `uploaded` row. If a staging root is unknown to the connected
  database, startup removes only recognizable incomplete or invalid ctrl-π
  episode remnants; valid manifests and unrelated shared-root entries are
  preserved so a fresh or wrong database cannot destroy finalized raw data.
- 2026-08-28 — ctrl-π does not infer a license grant for user-recorded data.
  Generated Hugging Face dataset cards leave the license field unset; making a
  repository public remains an explicit exposure choice, and selecting a data
  license remains the operator's responsibility.
- 2026-08-29 — Training and Models are separate first-class primary tabs.
  Training remains a read-only experiment-analysis surface for runs reported
  by external trainer clients, while Models discovers existing Hugging Face
  artifacts. External trainers may append strictly validated stdout, stderr,
  or system lines to a row-locked, server-sequenced, server-timestamped JSONB
  tail; the selected Training run polls bounded cursor pages and metrics while
  visible. Retention discards the oldest console entries at 1,000 records or
  512 KiB and reports gaps explicitly. This is an observer contract for
  external trainers and future workers, not managed training or direct Modal
  console attachment/streaming. (Superseded in part on 2026-08-29: the managed
  training worker later became a first-class producer on this same bounded
  metrics/console/checkpoint contract; raw provider console attachment remains
  excluded.)
- 2026-08-29 — V1.1 persists one bounded, non-secret physical YAM rig setup in
  PostgreSQL and applies it in place to the same `YAMDriver` object shared by
  Arms, recording, and inference. Discovery and preflight inspect bounded
  SocketCAN, stable serial-device, package, model, and calibration-file
  readiness without constructing vendor devices or opening hardware. Explicit
  connect and opt-in automatic restoration require separate hardware-motion
  acknowledgements; automatic restoration passively re-detects only a saved
  rig that was missing at boot, acquires the shared rig lease, and makes one
  serialized connection attempt when readiness changes. A failed active or
  runtime connection stays latched for manual review. A schema-valid readable
  calibration file is readiness evidence only, never proof that directions,
  offsets, limits, zero-torque behavior, or emergency-stop behavior are safe on
  a physical rig. Mock setup remains deterministic and never overwrites a
  saved physical setup when modes change.
- 2026-08-29 — V1.1 exposes the same FastAPI control-plane services used by
  the browser through the synchronous typed `ctrl_pi.CtrlPiClient`: system and
  non-secret settings status, YAM setup and bounded control, recording and Hub
  dataset workflows, model discovery, externally reported training, and
  inference lifecycle. The client uses a bounded, no-redirect, no-environment-
  proxy transport and validates typed responses; the existing
  `ctrl_pi.trainer.Client` remains source-compatible on that transport.
  `/api/models` is the canonical first-class model catalog route and
  `/api/trainer/models` remains a compatibility alias. Model artifacts still
  originate in the operator's Hugging Face tooling, while selecting a pinned
  model and starting inference is the supported model-to-YAM path. This
  milestone adds neither arbitrary action submission nor managed training.
- 2026-08-29 — V1.1 publishes the current product documentation from the
  repository-owned Mintlify tree under `docs/`. `docs/docs.json` is the
  navigation source of truth; every content page is navigated, carries
  Mintlify frontmatter, and is checked locally for broken routes, anchors,
  assets, shell/JSON examples, and credential leakage. The site covers the
  six product workflows, installation, YAM onboarding, the REST/Python SDK,
  architecture, operations, screenshots, troubleshooting, and release notes.
  The top-level README remains a concise entry point rather than a duplicate
  manual. Documentation describes only behavior present in the current tree,
  including SDK/REST-launched managed Modal training and its verified teardown
  boundary.
- 2026-08-29 — V1.1 managed training is an SDK/REST-launched, artifact-backed
  SmolVLA job through LeRobot 0.4.4 on the operator's Modal account, linked one-to-one with the
  existing Training run rather than replacing external reporting. The request
  is a bounded declarative specification with a caller-owned idempotency UUID,
  immutable Hub inputs, an exact configured-namespace output repo, explicit
  visibility and compute-cost acknowledgements, one fixed GPU choice, and a
  persisted hard deadline; arbitrary code, shell arguments, environment,
  Lambda, Auto sizing, and a browser launch form are excluded. ctrl-π creates
  the output repo private by default, writes and verifies an exact job marker,
  disables LeRobot's implicit Hub uploader, and explicitly publishes
  checkpoint folders plus a deployable final root without inferring a model
  license. Each job owns `ctrl-pi-training-<job UUID>` with the exact
  `ctrl-pi-training-job=<job UUID>` Modal tag and retains App/FunctionCall
  identity. Execution outcome remains separate from cleanup state: no job is
  terminal until exact App stop and zero tasks are verified, while gaps in the
  bounded provider event tail remain visible without invalidating an otherwise
  verified final artifact. Graceful backend shutdown preserves intentionally
  running training; deadline, explicit cancel, failure, startup reconciliation,
  and `make modal-panic` all use the same fail-closed ownership boundary.
  After an interrupted launch, absence or an exact zero-task App may represent
  a provider mutation or image build still settling, so reconciliation watches
  that deterministic identity through the immutable deadline. An App with an
  active task but no durably persisted FunctionCall cannot be supervised and
  is stopped promptly with verified teardown instead of consuming unobservable
  compute through the deadline.
  A partial unique database index enforces one nonterminal job, so this cost
  guard remains race-safe without introducing a queue. A possibly created
  output repo and its ownership marker are retained after every failed,
  cancelled, or ambiguous attempt; the idempotency row and output slug remain
  reserved so recovery never destroys evidence or silently reuses an artifact
  boundary.
  V1.1 managed training deliberately supports only a bounded, built-in
  SmolVLA checkpoint shape. Before the child starts, ctrl-π rejects dynamic
  processor classes, remote-code and adapter/PEFT indirection, seals device and
  Hub behavior, and pins the fixed SmolVLM configuration/tokenizer dependency
  at `7b375e1b73b11138ff12fe22c8f2822d8fe03467`. That non-weight
  dependency is bundled into each published checkpoint so the existing
  inference runtime can load the final artifact locally with Hub access
  disabled. Other LeRobot policy families remain available through the
  external reporting path, not this managed worker.
- 2026-08-29 — V1.1 user-facing AI-agent integration is a documentation and
  operating-policy layer over the same public `CtrlPiClient` and REST services,
  not a new identity, route, credential, or backdoor. Agents default to
  read-only access on a fixed trusted-LAN origin, treat all retrieved content
  as untrusted data, and never receive backend or provider credentials. Fresh
  human confirmation is required for motion, hardware connection or automatic
  restoration, paid compute, public artifacts, setup reset, and panic cleanup;
  uncertain mutations are reconciled through current state and caller-owned
  idempotency before any retry, and lifecycle cleanup is explicitly verified.
  Mock success and calibration readiness do not establish physical safety.
  Because V1.1 has no authentication or RBAC, capability restriction and human
  approval enforcement remain responsibilities of the operator and agent tool
  runner.

### V1.2 accepted evolution

V1.2 replaces only the narrow V1.1 physical-YAM topology and setup surface. It
preserves the Postgres/Hugging Face/Modal, training, deployment, recording,
inference, and public-control-plane architecture and proceeds in these coherent
milestones:

15. General YAM cell domain, persistence, stable SocketCAN identity resolution,
    four-arm mock parity, legacy setup compatibility, and typed REST/SDK models.
16. Supervised, pinned-checkout i2rt cell adapter; per-arm explicit
    connect/disconnect and teaching-handle health; pair-safe recording/teleop
    synchronization and follower-only inference routing.
17. Cell-centric Settings/Arms/Record/Inference UI, configurable Docker listen
    port and least-privileged hardware override, production documentation,
    physical field-test handoff, and final integration review.

- 2026-08-29 — V1.2 models one primary YAM-specific cell with typed arm rows.
  An empty arm list is a valid saved topology draft, but is not connect-ready
  and cannot enable automatic connection. Upgrading to V1.2 clears historical
  `yam_setups.auto_restore` consent while preserving the legacy topology; an
  operator must review the new boundary and explicitly opt in again before any
  unattended motion.
  A SocketCAN arm persists its USB adapter serial and resolves the current
  `canN` through bounded, read-only sysfs inspection. Logical ID, role,
  pair/group/side, transport, and end-effector metadata remain independent of
  enumeration order. Duplicate or missing serials, duplicate runtime buses,
  cross-pair routing, and pair-port collisions fail closed. A missing frame map
  means intentional 1:1 mapping; a missing soft-limit artifact is allowed but
  produces a prominent `NO SASH GUARD` warning and never invents a limit.
- 2026-08-29 — V1.2 conservatively treats both calibrated 4310 follower kinds,
  `linear_4310` and `crank_4310`, as jaw-motion operations. Explicit connect
  and unattended restoration require the same distinct gripper-calibration
  acknowledgement for either kind; no UI, REST, or SDK client may infer that
  the crank variant is motion-free.
- 2026-08-29 — The all-CAN hardware path integrates the field-tested i2rt fork
  through one supervised Python worker process per connected CAN arm, rather
  than importing motion-capable objects into FastAPI or supervising the Lux
  shell/Portal launchers. The operator supplies a read-only i2rt checkout and
  expected Git commit; preflight verifies both and never fetches public upstream
  or silently selects latest. The claimed checkout must have a clean tracked
  worktree/index and no untracked shadow files, and container hardware mode
  additionally proves that its baked dependency commit matches the mounted
  source commit. Read-only source is proven only by the filesystem's kernel
  `ST_RDONLY` mount flag (as provided by Docker's `/opt/i2rt:ro` bind); chmod,
  ownership, and effective permissions are not accepted as substitutes. Each worker calls the pinned i2rt factory and owns
  its CAN object and nested control threads. Bounded local IPC carries an exact
  configuration/identity handshake, fresh telemetry/heartbeats, named loop-rate
  diagnostics, latest-wins timestamped actions, safe-idle requests, and shutdown;
  no raw CAN/motor command is public and no unauthenticated pair server is bound.
  ctrl-π owns worker supervision, stable arm identity, frame-map/soft-limit
  application, rig leases, pair routing, and fail-closed teardown. It revokes
  writes first, requests cooperative safe-idle/close, then terminates, kills,
  and reaps on bounded deadlines; an uncertain survivor keeps the bus blocked
  and latches an error. A child-side command-stream watchdog enters safe idle
  when fresh follower commands stop, and Linux parent-death signaling covers an
  abrupt control-plane exit; runtime faults never auto-respawn. Arms surfaces the
  observed i2rt CAN/control-loop frequency without treating the field
  measurements (followers roughly 263–275 Hz and leaders roughly 419–426 Hz at
  `bilateral_kp=0.0`) as universal health thresholds.
- 2026-08-29 — Teleop start acquires and observes one declared pair with
  synchronization disabled and performs no follower write. A separate freshly
  acknowledged sync action interpolates the follower toward the latest mapped
  leader pose over approximately three seconds before latest-state tracking;
  disabling sync stops writes while leaving pair telemetry available. This
  preserves the proven first-sync boundary and makes live joint deltas visible
  before motion. Bimanual group identity is plumbed through configuration and
  selection, while true multi-follower policy execution and bimanual recording
  remain deferred.
- 2026-08-29 — The primary all-CAN Docker path uses host networking, read-only
  `/sys`, a read-only operator i2rt checkout, and only the raw-network capability
  expected for SocketCAN; it does not use `--privileged`, mutate host links, or
  require serial-device passthrough. These access assumptions remain H1 field
  hypotheses. A separate legacy override may pass exactly one GELLO serial
  device. The internal Uvicorn port is configurable so host-network hardware
  mode can use 8010 beside an existing service on 8000.
