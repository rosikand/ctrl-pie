# ctrl-π

Self-hosted robot-learning operations for YAM arms: collect demonstrations,
inspect LeRobot datasets and models, track training, and execute policies from
one web console or typed Python client.

![ctrl-π Inference tab in deterministic mock mode](docs/assets/ctrl-pi-inference.png)

> [!WARNING]
> ctrl-π has no authentication in V1. Bind it to localhost or a trusted,
> firewalled LAN. Never expose it directly to the public Internet.

## Product

ctrl-π runs on the Linux machine attached to the robot. PostgreSQL stores
small control-plane records, Hugging Face remains the source of truth for
datasets and model artifacts, and Modal runs real inference workloads. There
is no hosted ctrl-π service.

The interface has six first-class workflows:

- **Arms** — inspect telemetry, bus health, joints, pose, gripper state, and
  loop diagnostics; send bounded manual jog commands.
- **Record / Teleop** — operate one leader/follower pair, capture episodes,
  and publish LeRobot v3 datasets.
- **Datasets** — browse namespace-scoped repositories and inspect immutable
  episodes, synchronized state/actions, and proxied video.
- **Training** — monitor metrics, bounded console output, configuration, and
  checkpoints reported by external trainer clients. This release does not
  launch managed training.
- **Models** — browse Hugging Face model cards, immutable revisions, and
  checkpoint metadata in a separate read-only catalog.
- **Inference** — deploy one pinned policy revision, explicitly start robot
  execution, and stop with provider teardown verification.

Settings provides passive YAM discovery, bounded configuration and
calibration-artifact preflight, durable single-rig setup, explicit connection,
and separately acknowledged opt-in restoration after boot or hot-plug.

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
service starts.

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
recording, datasets, models, externally reported training, and inference. See
the [REST and Python SDK guide](docs/python-sdk.md) for lifecycle and cleanup
examples.

## Safety and support boundary

- Mock mode supplies `MockYAMDriver`, a synthetic camera, and Stub compute. It
  exercises the product workflow but does not prove hardware or cloud behavior.
- Hardware mode fails closed: missing configuration or devices never fall back
  to mocks. Discovery and preflight do not open devices or start a controller.
- Connecting a real follower may energize motors and engage gravity
  compensation. The operator must clear the workspace, verify power and
  emergency-stop access, and deliberately acknowledge immediate or automatic
  connection.
- The real adapter has extensive fake-vendor coverage, but no physical YAM was
  available in this workspace. Directions, offsets, limits, calibration,
  model fidelity, bus behavior, E-stop response, and disconnect recovery still
  require validation on the target Ubuntu/YAM box before motion.
- The V1 camera remains synthetic. Real OpenPI serving is unavailable and
  fails before provider deployment; real LeRobot policy serving on Modal is
  supported.

## Documentation

- [Introduction](docs/index.mdx)
- [Installation](docs/installation.mdx) and [mock quickstart](docs/quickstart.mdx)
- [YAM setup](docs/yam-setup.md) and [Arms](docs/arms.md)
- [Recording](docs/recording.md), [Datasets](docs/datasets.md),
  [Training](docs/training.md), [Models](docs/models.md), and
  [Inference](docs/inference.md)
- [REST and Python SDK](docs/python-sdk.md)
- [Architecture](docs/architecture.md),
  [Docker deployment](docs/docker-deployment.md), and
  [Troubleshooting](docs/troubleshooting.md)
- [Current UI gallery](docs/screenshots.mdx) and
  [release notes](docs/release-notes.md)

Keep credentials in the backend-only, gitignored `.env`; the browser receives
status and the minimum workflow data, never service secrets.
