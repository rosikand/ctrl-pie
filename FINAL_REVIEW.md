# ctrl-π V1 final review

Review date: 2026-08-28  
Branch: `milestone-spec-review`  
Base: `origin/main` (`0486e46d0710d8584cb9d77cb495b93bce9cc414`)

## Outcome

The V1 software implementation is complete through Milestone 14 and passes
the required mock-mode, persistence, build, smoke, and packaging gates. The
Milestone 13 preservation commit (`3ec9a73`) was not treated as completion;
Milestone 13 was completed by `e518d37` and reviewed in `455146f`. Milestone 14
was completed by `801b1e1`.

The real YAM boundary is implemented and fail-closed, but no physical robot was
available in this workspace. Nothing in this review claims successful physical
motion or hardware validation. The exact remaining Ubuntu/YAM-box checks are
listed below and in [docs/yam-driver.md](docs/yam-driver.md).

There is one owner-controlled publication blocker: this repository is private
and has no selected project license. The code can be reviewed and run by its
owner, but it must not be described as an open-source release until the owner
chooses a license, adds the matching package metadata, and makes the repository
public. ctrl-π also deliberately leaves user dataset-card licenses unset; the
operator must choose an appropriate data license.

## Final gate evidence

| Gate | Result |
| --- | --- |
| Frontend production build | PASS — TypeScript and Vite built 1,832 modules; JS 429.52 kB and CSS 31.81 kB before gzip. |
| Backend tests | PASS — 431 passed in 45.29 s; one upstream Starlette/httpx deprecation warning. |
| Python validation | PASS — `compileall`; `pip check` reported no broken requirements. |
| Diff validation | PASS — `git diff --check`; entrypoint shell syntax; documentation link, image, shell, JSON, and secret checks. |
| PostgreSQL migrations | PASS — one Alembic head; `upgrade head` and `current` succeeded against the configured PostgreSQL database at `0006`. |
| Development boot | PASS — backend on loopback `:8000`, Vite on loopback `:5173`, frontend API proxy, `/inference` deep link, redacted Settings response, and `/ws/arms` with two mock arms. Both processes shut down cleanly. |
| Required `make smoke` | PASS — real private Hugging Face upload/list/delete plus Stub inference and exact 100 mock-arm writes; evidence below. |
| Docker build | PASS — fresh image `sha256:9ff483ad0e37d7a014b9b997b5c0831f6d2231216998329eb6741ab370621cc5`, 2,926,816,409 bytes. |
| Docker runtime | PASS — UID/GID 10001, one Uvicorn worker, FFmpeg/libx264, CPU Torch, LeRobot 0.4.4, Modal 1.5.4, YAM plugin imports, SPA/deep links/API/WebSocket, writable staging, and no Node/npm/GCC in the final image. |
| Container smoke and teardown | PASS — fake-Hub smoke recorded 5.000 s/50 samples, ran 100 actions, and proved zero resources; scoped Compose teardown left 0 containers, 0 networks, and 0 volumes. |
| Credential scan | PASS — no configured credential value in tracked files, image metadata/history, built frontend, installed ctrl-π files, or Alembic assets. |
| YAM probe | PASS as a safe unavailable-hardware check — preflight returned `missing` with two disconnected arms and did not connect to a device. `--connect` was not used. |

The final production image also proved that operator utilities bypass automatic
database migration: an emergency cleanup command can run even while PostgreSQL
is unavailable. Normal Uvicorn container startup still applies migrations.

### Required real-HF smoke

`make smoke` used the configured backend-only Hugging Face credentials and
reported:

- recording: 5.001 s wall time, 5.000 s episode, 50 synchronized samples,
  9,548-byte H.264 MP4;
- private dataset:
  `rosikand/ctrl-pi-smoke-20260828232142-1342605b2d`;
- immutable revision:
  `bfbd6a394bb4ae4b531ba934763035d8829b8cf8`;
- discovery: the same repository and revision, one episode, 50 frames;
- inference: Stub target, Stub runtime, exactly 100 writes to `yam-follower`;
- teardown: recording 0, inference 0, compute 0, rig free, upload idle,
  repository deleted, workspace removed.

An independent Hub existence check after the gate confirmed that the temporary
repository was absent.

### Retained real Modal evidence

The final audit did not create another paid Modal resource. The retained
Milestone 11 proof is commit
`b34d9a63e16e9642f821e61d4e9a40a28d769e9b`:

- model:
  `rosikand/ctrl-pi-demo-act-aloha@7fe3b8c62b715f56907701faa00a0bf665873c9d`;
- A10G runtime: 115.6 s, 290 inference chunks, 5,778 mock-arm actions,
  119.70 ms average inference latency, 117.37 ms final latency, 49.45 Hz,
  zero dropped chunks, empty final queue;
- private evidence dataset:
  `rosikand/ctrl-pi-m11-live-modal-act-20260828@743f28ba80c737ea87ea3cc9447fa308b549b700`,
  one episode and 2,312 frames;
- Modal App `ap-ASxbec6ZI3XEtwe0a4Sbpl` stopped with zero running tasks;
  ownership-safe panic verification also found zero active resources, and the
  temporary proxy credential was deleted.

## Exact launch instructions

### Source development

Requirements: Python 3.11, Node.js 22+, PostgreSQL 14+, and FFmpeg with
libx264.

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

Alembic intentionally does not load `.env`; export `DATABASE_URL` in the shell
before running it. Start these commands in separate terminals:

```bash
PYTHONPATH=backend/src uvicorn ctrl_pi.main:app --reload --host 127.0.0.1 --port 8000
npm --prefix frontend run dev
```

Open <http://127.0.0.1:5173>.

### Production container

```bash
cp .env.example .env
$EDITOR .env
docker compose up --build -d --wait
curl --fail http://127.0.0.1:8000/api/health
```

Open <http://127.0.0.1:8000>. Stop the stack without deleting unuploaded
recordings:

```bash
docker compose down
```

Do not add `--volumes` while the `ctrl-pi-data` volume contains recordings
that have not reached a verified `uploaded` state.

### Acceptance smoke

The release gate uses real Hugging Face access but mock arms, camera, compute,
and policy execution:

```bash
export HF_TOKEN='<backend-only token>'
export HF_NAMESPACE='<exact user or organization>'
make smoke
```

For a network-free development check only:

```bash
PYTHONPATH=backend/src .venv/bin/python -m ctrl_pi.smoke --fake-hub
```

## Known limitations

- V1 has no authentication. Bind to loopback or a trusted, firewalled LAN;
  never expose the service directly to the public Internet.
- The project has no selected source-code license and the GitHub repository is
  private. License selection, package metadata, and repository visibility are
  owner actions required before calling it an open-source release.
- No physical YAM validation occurred. Real support is deliberately limited to
  the standard SocketCAN YAM follower, calibrated GELLO serial leader, and
  `crank_4310` gripper. Linear grippers, YAM Pro, and YAM Ultra are unsupported
  in V1.
- Vendor writes acknowledge submission to an in-memory command buffer, not
  confirmed physical motion. Sampling and worker-health checks fail closed,
  but they are not a motor-level acknowledgement.
- A stuck vendor I/O call can outlive the bounded application wait and delay
  device cleanup. The operator must use the physical emergency stop and make
  the rig safe rather than trusting process shutdown alone.
- Hardware mode still uses the synthetic V1 camera; a real camera driver is
  outside this release.
- Real OpenPI serving is unavailable. OpenPI has a deterministic mock adapter;
  real Modal deployment rejects it before resource creation.
- Finalized but unuploaded MP4, JSONL, and manifests exist only in
  `RECORDING_STAGING_DIR` or the `ctrl-pi-data` volume. Back up and preserve
  that storage until the verified Hub upload commits. A database or credential
  restore alone cannot recreate local raw artifacts.
- Dataset cards do not grant a license automatically. Public visibility and a
  suitable data license are separate operator decisions.
- Modal lifecycle inspection uses a narrow boundary pinned to Modal 1.5.4,
  including isolated private protocol calls required for exact task/lifecycle
  proof. Dependency upgrades require revalidation.
- There is no separate frontend unit/E2E runner. Frontend acceptance consists
  of the strict TypeScript/Vite production build plus API, browser, WebSocket,
  smoke, and container integration checks.

## Required physical Ubuntu/YAM validation

Before enabling motion on the target box, an operator with immediate access to
the emergency stop must complete and record all of these checks:

1. Verify the supported standard YAM/GELLO/`crank_4310` hardware and the exact
   pinned `0.1.1` YAM plugins with LeRobot 0.4.4.
2. Configure SocketCAN, bitrate, interface state, permissions, bus-off recovery,
   and physical CAN termination; verify container access if Docker is used.
3. Use a stable `/dev/serial/by-id/...` leader path, correct dialout/group
   permissions, and an existing noninteractive GELLO calibration file.
4. Validate the MuJoCo XML and referenced assets, `joint1`–`joint6`, and
   `grasp_site`; confirm the model corresponds to the physical arm.
5. Run `make yam-probe` first. With the rig secured, run the explicit connected
   probe from [docs/yam-driver.md](docs/yam-driver.md), observe bounded cleanup,
   and treat an incomplete shutdown as a blocker. This review did not run it.
6. Compare every reported joint with independent physical motion: zero offsets,
   signs, radians, wrist-flex/wrist-roll ordering, and FK position/orientation.
7. Verify the gripper boundary physically: vendor crank/GELLO `0=open` and
   `1=closed` must appear publicly as `1=open` and `0=closed`, including the
   inverted follower velocity sign and mounted GELLO endpoints.
8. Check follower velocity/effort units and signs, gravity-compensation and
   position-control transitions, safe/torque state, hard limits, current
   limits, collision behavior, and one-at-a-time low-speed jog bounds.
9. Validate teleop direction and rate only after the previous checks; then
   validate recording alignment while remembering that the camera is
   synthetic in V1.
10. Exercise leader disconnect, follower/CAN loss, worker death, bus-off,
    process termination, and emergency-stop recovery. Confirm no command is
    accepted after the software fault latch and make the hardware safe
    independently of application cleanup.

Until that checklist passes on the real Ubuntu/YAM box, use mock mode for all
acceptance and do not enable physical teleoperation or inference motion.
