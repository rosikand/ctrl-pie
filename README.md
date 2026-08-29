# ctrl-π

Self-hosted robot-learning operations for YAM arms: collect demonstrations,
inspect LeRobot datasets, track external training, and run policies against a
robot from one engineering-focused web console.

![ctrl-π Inference tab in deterministic mock mode](docs/assets/ctrl-pi-inference.png)

> [!WARNING]
> ctrl-π has no authentication in V1. Bind it to localhost or a trusted,
> firewalled LAN; never expose the service directly to the public Internet.

ctrl-π runs on the Linux machine attached to the robot. You operate the
service and retain its state: PostgreSQL holds small control-plane records,
your Hugging Face namespace holds datasets and model artifacts, and your
Modal account runs inference. Live telemetry and action-loop state remain
ephemeral. There is no hosted ctrl-π service.

## Six workflows

- **Arms** — watch connection and CAN health, joints, end-effector pose,
  gripper state, and loop diagnostics; issue bounded manual jog commands.
- **Record / Teleop** — pair leader and follower arms, watch the V1 synthetic
  camera feed, teleoperate, record episodes, and publish LeRobot v3
  datasets to your Hugging Face namespace.
- **Datasets** — discover namespace-scoped LeRobot repositories and inspect
  cards, revisions, episodes, synchronized state/action values, and proxied
  private video without sending `HF_TOKEN` to the browser.
- **Training** — track runs created by external scripts and render their
  configuration, status, checkpoints, live scalar metric curves, and bounded
  trainer-reported console output. ctrl-π does **not** launch or run training
  in V1.
- **Models** — browse Hugging Face model repositories, revisions, checkpoint
  files, and card metadata in the configured namespace.
- **Inference** — deploy one immutable policy revision, verify endpoint
  identity, explicitly start a follower-arm action loop, optionally record
  the run, and stop with provider teardown verification.

The complete UI and orchestration flow works in mock mode with
`MockYAMDriver`, the synthetic camera, and Stub compute. Mock mode is for
development and safe workflow checks; it does not pretend that cloud or
hardware work occurred.

## Quickstart with Docker

Requirements are Docker Engine, a current Compose/Buildx plugin, and a
PostgreSQL 14+ database reachable from the container.

```bash
git clone https://github.com/rosikand/ctrl-pie.git
cd ctrl-pie
cp .env.example .env
$EDITOR .env
docker compose up --build -d --wait
```

Leave `CTRL_PI_MOCK_MODE=true` for the deterministic stack. Set at least
`DATABASE_URL` for persisted workflows; Hugging Face and Modal credentials may
remain blank while exercising mock arms and Stub inference. Open
<http://127.0.0.1:8000>.

The production image builds Vite once, then runs the built SPA and FastAPI
from a single non-root Uvicorn worker. Migrations run before the service
starts. See [Docker deployment](docs/docker-deployment.md) for volume,
shutdown, USB serial, and SocketCAN guidance.

## Quickstart from source

Use Python 3.11, Node.js 22, PostgreSQL 14+, and FFmpeg with libx264. The
[development guide](docs/development.md) includes the CPU-only PyTorch install
that avoids pulling CUDA wheels onto the control-plane machine.

```bash
git clone https://github.com/rosikand/ctrl-pie.git
cd ctrl-pie

python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  'torch==2.10.0+cpu' 'torchvision==0.25.0+cpu'
python -m pip install -e './backend[dev]'
npm --prefix frontend ci

cp .env.example .env
$EDITOR .env
export DATABASE_URL='postgresql://ctrl_pi:password@localhost:5432/ctrl_pi'
alembic -c alembic.ini upgrade head
make seed
```

Start the two development processes in separate terminals:

```bash
PYTHONPATH=backend/src uvicorn ctrl_pi.main:app --reload --port 8000
npm --prefix frontend run dev
```

Open <http://127.0.0.1:5173>. With Hub credentials left blank, `make seed`
still adds the two mock robots and example training curves while safely
skipping the sample dataset upload.

## Runtime support and evidence

Real LeRobot 0.4.4 policy serving on Modal GPU is implemented, not represented
only by the offline stub. The Milestone 11 proof ran a revision-pinned ACT
policy on an A10G for 115.6 seconds: 290 inference chunks drove 5,778 mock-arm
actions at 49.45 Hz with 119.70 ms average inference latency and no dropped
chunks. It finalized a one-episode LeRobot dataset, stopped the Modal App, and
then verified zero active ctrl-π resources. The evidence is recorded in
milestone commit `b34d9a63e16e9642f821e61d4e9a40a28d769e9b`.

The boundaries are deliberately explicit:

- Real Modal inference supports LeRobot on `Modal: A10G`, `Modal: A100`, and
  `Modal: H100`.
- OpenPI can be selected for deterministic mock emulation, but a real OpenPI
  runtime is unavailable in V1 and fails clearly before provider deployment.
- `CTRL_PI_MOCK_MODE=false` selects the fail-closed real YAM adapter for a
  standard SocketCAN follower with a crank 4310 gripper and a calibrated
  GELLO serial leader. Missing configuration or hardware remains visibly
  disconnected and never falls back to `MockYAMDriver`.
- The real adapter is covered with fake-vendor mapping, lifecycle, fault, and
  concurrency tests, but this cloud workspace has no physical YAM rig. Motor
  directions/offsets, wrist mapping, gravity compensation, bus permissions,
  limits, emergency stop, and disconnect recovery must still be validated on
  the target Ubuntu/YAM box before motion.
- Training stays on compute you operate through LeRobot, OpenPI, or custom
  scripts. The Trainer API records progress and artifacts only.

Read [compute targets and inference](docs/inference.md) for the runtime wire
contract, cost guardrails, model packaging, robot loop, and teardown model.

## Documentation

| Guide | Contents |
| --- | --- |
| [Setup and configuration](docs/setup.md) | Environment variables, PostgreSQL/Supabase, Hugging Face, Modal credentials, and first-run checks |
| [Architecture](docs/architecture.md) | Process boundaries, persistence, live state, leases, and safety invariants |
| [Development](docs/development.md) | Source installation, dev servers, tests, migrations, seeding, and contribution workflow |
| [Docker deployment](docs/docker-deployment.md) | Production container, storage, graceful shutdown, USB, and SocketCAN |
| [Record and teleoperate](docs/recording.md) | Pairing, episode lifecycle, metadata, staging, upload, and recovery |
| [Datasets and inference](docs/inference.md) | Compute targets, Modal serving, runtime adapters, robot execution, and teardown |
| [Trainer API](docs/trainer-api.md) | Python client, every REST method, metric and console reporting, and a LeRobot fine-tune example |
| [YAM driver interface](docs/yam-driver.md) | Driver contract, telemetry, command bounds, leases, and the real-hardware boundary |
| [Modal operations](docs/modal-operations.md) | Owned-App naming and verified emergency cleanup with `make modal-panic` |
| [Mock smoke gate](docs/smoke-test.md) | The complete record → upload → discover → infer → teardown acceptance flow |
| [V1 final review](FINAL_REVIEW.md) | Exact release-gate evidence, launch commands, known limitations, and the remaining physical Ubuntu/YAM checklist |

Start with [setup](docs/setup.md), keep credentials in the backend-only
gitignored `.env`, and use the Settings checklist to confirm readiness without
ever returning secret values to the browser.
