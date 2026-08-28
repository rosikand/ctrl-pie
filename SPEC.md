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
- All state lives in services the user owns: their Postgres, their HF
  namespace, their Modal account. Booting ctrl-π on a fresh machine with the
  same credentials restores the same state.
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

Exactly five primary tabs, plus a small Settings page (gear icon, not a
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

ctrl-π does **not** run training in V1. Users train on their own compute with
their own LeRobot/OpenPI/custom scripts. ctrl-π provides:

**Trainer API** — a small Python SDK + REST API (wandb-like) that external
training scripts call to:

- create/update a training run (status, current step, config)
- log scalar metrics over steps (loss curves rendered in the UI)
- register the output HF model repo / checkpoint revisions

Run state goes to Postgres; weights go to HF Hub via the user's normal tooling.

The tab has two subviews:

- **Runs**: current/past runs, status/step, dataset, base model,
  runtime/framework, config, metric curves, output model repo
- **Models**: HF model repos in the namespace, revisions, checkpoints,
  card metadata

V2 (do not build): buttonized plug-and-play training on managed compute.

### 5. Inference

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
Browser ── LAN ──► ctrl-π backend (Ubuntu box) ── CAN/USB ──► YAM arms
                        │
                        ├── PostgreSQL (user's; Supabase recommended)
                        ├── Hugging Face Hub (user's namespace)
                        └── Modal (user's account; inference only)
```

Key interfaces/components:

- `YAMDriver` / `MockYAMDriver`
- `LeRobotRuntime`, `OpenPIRuntime` (runtime adapters)
- `ModalComputeTarget` (behind a generic `ComputeTarget` abstraction)
- Trainer API (REST + thin Python client package)

Boundaries: UI → ctrl-π API → services/interfaces → (Postgres | HF | YAM |
Modal). No YAM/CAN, Modal, or policy-framework details in frontend components.

## Persistence

Three categories:

1. **PostgreSQL** — source of truth for small mutable control-plane state:
   robots/arms config, recording metadata, training runs, inference
   endpoints/deployments, settings. Suggested tables: `robots`, `recordings`,
   `training_runs`, `inference_endpoints`, `deployments`, `settings`.
   SQLAlchemy + Alembic; depend only on standard Postgres via `DATABASE_URL`
   (Supabase is just the recommended host).
2. **Hugging Face Hub** — source of truth for artifacts: LeRobot datasets,
   model repos/weights/checkpoints, configs, dataset/model cards. Never
   duplicate large artifacts into Postgres.
3. **Ephemeral** — never persisted: live joints/pose, CAN health, loop
   frequency, camera streams, jog/teleop commands.

## Setup, configuration, credentials

- Secrets via a gitignored `.env`: `DATABASE_URL`, `HF_TOKEN`. Modal uses its
  normal local credential mechanism.
- A Settings page in the UI shows connection status for Postgres / HF / Modal
  / arms and lets the user enter or update non-secret config. Never store
  secrets in Postgres or HF.
- First-run experience: if required config is missing, the UI shows a clear
  setup checklist instead of broken pages.

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
management, hosted ctrl-π, buttonized training, BYOC/Lambda compute, automated
VM provisioning, elaborate dashboards, reimplementing LeRobot/OpenPI
protocols, unnecessary abstractions. Prefer boring, understandable,
low-latency implementations.

## Naming

repo `ctrl-pi` · Python package `ctrl_pi` · trainer client `ctrl_pi.trainer`.

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

1. App shell: five tabs + Settings, sparse Tailwind layout
2. Postgres schema + Alembic migrations + settings/first-run checklist
3. Mock Arms: `MockYAMDriver`, live telemetry over WebSocket, jog controls
4. Mock Record/Teleop: recording lifecycle, episode metadata
5. HF integration: namespace auth, dataset upload pipeline (LeRobot format)
6. Datasets tab: discovery + metadata/cards
7. Datasets tab: episode visualizer (video + synced state/actions + scrubbing)
8. Trainer API: REST endpoints, Python client, Runs/Models subviews,
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
- 2026-08-28 — Training remains an external-script responsibility. The
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
