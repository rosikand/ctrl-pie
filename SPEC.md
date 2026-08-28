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
  and compute size (dropdown; V1 options: "Modal: H100", "Modal: A100";
  "Auto" is future work)
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
   lifecycle against a stub workload
10. Runtime adapters: LeRobot + OpenPI serving on Modal; robot-side client
    loop against `MockYAMDriver`
11. Inference tab: full deploy → stream → execute → stop flow in mock mode;
    deployment-session recording to HF
12. Docker packaging + device-passthrough docs
13. Documentation: `docs/` with architecture overview, development guide,
    setup/configuration (env vars, Supabase, HF, Modal), YAM driver interface,
    recording/teleop workflow, compute targets and runtime adapters; a
    polished top-level `README.md` (what ctrl-π is, screenshot, quickstart, step by step how to use);
    verify `docs/trainer-api.md` is complete and its examples run
14. Real `YAMDriver` integration points, polish, tests

After each milestone: run build/tests, commit as `milestone N: <name>`, note
open issues. Do not expand scope.

## Decisions

(Agent: append dated entries here when resolving ambiguities.)
