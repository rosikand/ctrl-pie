# ctrl-π

Self-hosted robot-learning operations for YAM arms: collect demonstrations,
inspect LeRobot datasets and models, track training, and execute policies from
one web console or typed Python client.

![ctrl-π Inference deployment in deterministic mock mode](docs/assets/ctrl-pi-inference.png)

> [!WARNING]
> ctrl-π has no authentication in V1. Bind it to localhost or a trusted,
> firewalled LAN. Never expose it directly to the public Internet.

## Product

ctrl-π runs on the Linux machine attached to the robot. PostgreSQL stores
small control-plane records, Hugging Face remains the source of truth for
datasets and model artifacts, and Modal runs real inference and managed
LeRobot training workloads. There is no hosted ctrl-π service.

The console is grouped into Operate, Build, Deploy, and Admin:

- **Overview** — cell-wide status: robots, recording, training, deployment,
  recent activity, and service readiness.
- **Robots** — a list of every logical arm, then one arm's connection and
  joint state, with pose, gripper/handle, diagnostics, and identity collapsed
  behind detail sections. Bounded jog is available in mock and retained legacy
  mode; the supervised all-CAN adapter rejects one-shot jog.
- **Record** — operate one leader/follower pair with the camera as the primary
  view, capture episodes, and publish LeRobot v3 datasets.
- **Datasets** — browse namespace-scoped repositories and inspect immutable
  episodes, synchronized state/actions, and proxied video.
- **Training** — observe externally reported runs and SDK-launched managed
  Modal jobs through the same bounded metrics, console, configuration, and
  checkpoint surfaces.
- **Models** — browse Hugging Face model cards, immutable revisions, and
  checkpoint metadata in a separate read-only catalog.
- **Inference** — deploy one pinned policy revision through a Model → Robot →
  Compute → Deploy flow, explicitly start robot execution, and stop with
  provider teardown verification.
- **Set up** — the step-by-step path from `git clone` to connected YAM arms on
  an Ubuntu box.

Settings provides passive YAM-cell discovery, stable USB-CAN identity
assignment, typed pair/group/side configuration, per-arm preflight and
explicit connect/disconnect. A deterministic four-arm mock cell exercises two
independent pairs. Calibrating a real `linear_4310` or `crank_4310` follower
requires a jaw-motion acknowledgement in addition to the general connection
or unattended-restoration consent.

## Run locally with Docker

Requirements: Docker Engine with current Compose/Buildx, plus PostgreSQL 14+
reachable from the container.

```bash
git clone https://github.com/rosikand/ctrl-pie.git
cd ctrl-pie
cp .env.example .env
${EDITOR:-vi} .env
docker compose up --build -d --wait
curl --fail http://127.0.0.1:8000/api/health
```

Set `DATABASE_URL` and leave `CTRL_PI_MOCK_MODE=true` for the deterministic
stack. Hugging Face and Modal credentials can remain blank until their cloud
workflows are needed. Open <http://127.0.0.1:8000>. Migrations run before the
service starts. `CTRL_PI_LISTEN_PORT` changes Uvicorn's container port;
`CTRL_PI_PORT` independently changes the published bridge-mode host port.

The all-CAN hardware deployment is an explicit override using host
networking, `/sys:ro`, one operator-owned i2rt checkout mounted read-only, and
`NET_RAW` only. It never pulls i2rt from public upstream or selects a latest
revision. Build it through the exact-source `make docker-yam-cell` workflow.
A checked-in constraint carries i2rt's `scikit-build-core<0.10` requirement
into the isolated ruckig build. A separate legacy override passes exactly one
GELLO serial device.
See [Docker deployment](docs/docker-deployment.md) before hardware mode.

For an editable Python 3.11/Node.js 22 setup, see
[Installation](docs/installation.mdx) and [Development](docs/development.md).

## Python SDK

`CtrlPiClient` uses the same REST services and safety checks as the browser;
it has no privileged motion path. From an editable checkout:

```bash
python -m pip install -e ./backend
```

```python
from ctrl_pi import CtrlPiClient


with CtrlPiClient("http://127.0.0.1:8000") as ctrl:
    print(ctrl.health())
    print(ctrl.list_arms())
```

The typed client covers system status, YAM setup and bounded control,
recording, datasets, models, external and managed training, and inference. See
the [REST and Python SDK guide](docs/python-sdk.md) for lifecycle and cleanup
examples. Before delegating those workflows, give your automation the
[user-facing AI-agent integration guide](docs/ai-agent-integration.md).

## Safety and support boundary

- Mock mode supplies a four-arm `MockYAMDriver`, a synthetic camera, and Stub
  compute. It exercises two declared pairs but does not prove hardware or cloud
  behavior.
- Hardware mode fails closed: missing configuration or devices never fall back
  to mocks. Discovery and preflight do not open devices or start a controller.
- Connecting a real follower can energize/hold the arm, and calibrating a
  `linear_4310` or `crank_4310` follower moves the jaws. Teleop starts with synchronization disabled and
  performs no follower write; slow synchronization is a second explicit
  motion boundary.
- All-CAN inference Stop clears writes, safe-idles to gravity compensation,
  and releases the motion lease, but leaves the follower connected/holding.
  Explicit Settings Disconnect must reap its worker and return it to the
  de-energized/limp state.
- The all-CAN adapter supervises one pinned-checkout i2rt worker per connected
  arm. No physical YAM was available in this workspace. Container access,
  directions, offsets, limits, calibration, loop behavior, E-stop response,
  and disconnect recovery still require the ordered
  [H0–H7 field session](V1_2_FIELD_TEST_HANDOFF.md).
- A missing frame map means intentional identity mapping. Missing follower
  soft limits are shown as `NO SASH GUARD`; ctrl-π never invents limits.
- Managed training can allocate up to eight A100 or H100 GPUs in the user's
  Modal account. It is SDK-only, cost-acknowledged, limited to one nonterminal
  job, and terminal only after provider teardown is verified. Use
  `make modal-panic` if normal cleanup cannot converge.
- The V1 camera remains synthetic. Real OpenPI serving is unavailable and
  fails before provider deployment; real LeRobot policy serving on Modal is
  supported.

## Documentation

- [Introduction](docs/index.mdx)
- [Installation](docs/installation.mdx) and [mock quickstart](docs/quickstart.mdx)
- [YAM setup](docs/yam-setup.md) and [Robots](docs/arms.md)
- [Recording](docs/recording.md), [Datasets](docs/datasets.md),
  [Training](docs/training.md), [managed training](docs/managed-training.md),
  [Models](docs/models.md), and
  [Inference](docs/inference.md)
- [REST and Python SDK](docs/python-sdk.md)
- [Architecture](docs/architecture.md),
  [Docker deployment](docs/docker-deployment.md), and
  [Troubleshooting](docs/troubleshooting.md)
- [V1.2 physical field-test handoff](V1_2_FIELD_TEST_HANDOFF.md)
- [Current UI gallery](docs/screenshots.mdx) and
  [release notes](docs/release-notes.md)

Keep credentials in the backend-only, gitignored `.env`; the browser receives
status and the minimum workflow data, never service secrets.
