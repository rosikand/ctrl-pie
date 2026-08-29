# ctrl-π

Self-hosted robot-learning operations for YAM arms: collect demonstrations,
inspect LeRobot datasets, track external training, and execute revision-pinned
policies from one engineering console.

![ctrl-π Inference tab in deterministic mock mode](docs/assets/ctrl-pi-inference.png)

> [!WARNING]
> ctrl-π has no authentication in V1. Bind it to localhost or a trusted,
> firewalled LAN; never expose it directly to the public Internet.

## What it covers

- **Arms** — live YAM telemetry, diagnostics, and bounded manual jogs.
- **Record / Teleop** — leader/follower control, synchronized episodes, and
  LeRobot v3 upload.
- **Datasets** — namespace-scoped cards, revisions, private video, and synced
  observation/action playback.
- **Training** — runs and scalar metrics reported by external scripts, plus
  Hugging Face model discovery. ctrl-π does not run training.
- **Inference** — immutable policy deployment, explicit robot execution,
  optional recording, and verified Modal teardown.

The full workflow runs safely in mock mode with `MockYAMDriver`, a synthetic
camera, and Stub compute. PostgreSQL and Hugging Face remain the durable stores
you control; live robot state stays in one backend process.

## Quickstart

Requirements: Docker Engine with current Compose/Buildx plugins and a
PostgreSQL 14+ database reachable from the container.

```bash
git clone https://github.com/rosikand/ctrl-pie.git
cd ctrl-pie
cp .env.example .env
${EDITOR:-vi} .env
docker compose up --build -d --wait
curl --fail http://127.0.0.1:8000/api/health
```

Set `DATABASE_URL`, leave `CTRL_PI_MOCK_MODE=true`, and open
<http://127.0.0.1:8000>. Hugging Face and Modal credentials may stay blank for
the local mock workflow. Migrations run automatically before the service
starts.

## Documentation

- [Introduction](docs/index.mdx)
- [Installation](docs/installation.mdx) and [mock quickstart](docs/quickstart.mdx)
- [First-time setup](docs/setup.md) and [configuration](docs/configuration.md)
- [YAM onboarding](docs/yam-setup.md)
- [Trainer API](docs/trainer-api.md)
- [Inference and Modal teardown](docs/inference.md)
- [Docker deployment](docs/docker-deployment.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Architecture](docs/architecture.md)
- [Changelog](docs/release-notes.md)

The repository-hosted Mintlify site is configured by
[`docs/docs.json`](docs/docs.json). For source installation, contributor
commands, required gates, and local docs validation, see the
[development guide](docs/development.md).

## V1 boundaries

Real policy serving supports LeRobot 0.4.4 on Modal A10G, A100, or H100. Real
OpenPI serving is unavailable. Hardware mode is limited to a standard
SocketCAN YAM follower, calibrated GELLO leader, and `crank_4310` gripper; it
fails closed and requires physical Ubuntu/YAM validation before motion. The V1
camera is synthetic in both modes.
