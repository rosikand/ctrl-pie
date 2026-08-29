# ctrl-π V1.1 release review

Review date: 2026-08-29

Branch: `milestone-spec-review`

Implementation HEAD audited before this review artifact:
`8d3a62a0862f5771e989e855e58f63182f2ee07d`

## Outcome

ctrl-π V1.1 is implemented on top of the completed V1. The current product has
six first-class workflows, persisted and consent-gated YAM onboarding, one
typed public REST/Python SDK, repository-owned Mintlify documentation, managed
Modal training, and a canonical safety guide for user-owned AI agents.

The browser, `ctrl_pi.CtrlPiClient`, and the compatible
`ctrl_pi.trainer.Client` all use the same FastAPI services. Large artifacts
remain in the user's Hugging Face namespace, small lifecycle state remains in
the user's PostgreSQL database, robot/provider details stay behind backend
interfaces, and local raw recording artifacts remain in staging until a Hub
upload is verified.

Mock-mode implementation and automated acceptance are complete. No physical
YAM was available, and this review does **not** claim physical calibration,
motion, disconnect, bus, or emergency-stop validation. V1 retained a successful
real Modal inference demonstration and real private Hugging Face smoke. V1.1
real managed-training validation is incomplete: three controlled real Modal
attempts reached the pre-FunctionCall App/image-build window and exposed the
reconciliation defect. Commit `5a65d98` preserved a marked zero-task launch
settlement window; `72707ec` narrowed that behavior so any active task without
a durable FunctionCall is stopped promptly. A fourth attempt received a
transient deploy failure before any handle was persisted. None spawned a
FunctionCall or GPU task, and no successful end-to-end managed training run,
final model publication, or deployment of that managed artifact has been
demonstrated.

GitHub currently reports the repository as public, but the tree has no selected
source-code license and GitHub reports no license metadata. It is ready for
owner review and operation, but source visibility alone is not an open-source
license: do not represent it as open source until the owner selects a license
and aligns the repository and package metadata.

## Accepted V1.1 issues

### ROS-53 — Training and Models as first-class tabs

Commit: `e4294bf`

- The primary navigation now contains exactly Arms, Record / Teleop, Datasets,
  Training, Models, and Inference. Settings remains a gear route.
- Training preserves externally reported runs, bounded metrics, checkpoints,
  and console output.
- Models is a separate read-only Hugging Face catalog with immutable revisions,
  cards, and checkpoint paths.
- Managed-job observability was added later by ROS-52 without adding a browser
  launch or cancel control.

### ROS-58 — First-time YAM onboarding and restoration

Commit: `c5a95f9`; final safety hardening: `9f1771e`

- Settings provides passive discovery, non-opening preflight, bounded setup,
  one persisted physical leader/follower rig, and explicit connection.
- Manual connection and automatic boot/hot-plug restoration have separate,
  strict boolean motion-risk acknowledgements. Auto-restore is off by default.
- Saved consent and configuration are reapplied to the same `YAMDriver` used by
  Arms, recording, and inference. The shared rig lease serializes setup,
  manual control, teleoperation, and inference.
- Missing hardware is passively re-detected; active/vendor failures latch for
  manual review instead of being retried blindly.
- Mock onboarding remains deterministic and never overwrites or deletes the
  saved physical setup.
- Calibration readiness proves only the structure and readability of a bounded
  artifact. It is not physical calibration evidence.

### ROS-57 — Universal REST/Python SDK

Commit: `62e7dcd`

- `from ctrl_pi import CtrlPiClient` exposes typed synchronous methods for
  health/settings, YAM setup and arms, recording, datasets, models, external
  training, managed training, and inference.
- The hardened transport rejects ambiguous base URLs, userinfo, redirects,
  environment proxies, malformed typed responses, and JSON responses above the
  streamed 64 MiB ceiling. Errors expose a sanitized message and optional
  status code without retaining a raw body, credential, cause, or context.
- IDs, repository names, revisions, query values, and motion/configuration
  inputs are bounded before interpolation. Dangerous choices retain explicit
  visibility or acknowledgement parameters.
- `CtrlPiClient` is an idempotently closable context manager and the package
  ships `py.typed`.
- `ctrl_pi.trainer.Client` remains source-compatible on the shared transport.
- `GET /api/models` is canonical; deprecated `GET /api/trainer/models` uses the
  same model service. There is no raw-action or model-upload backdoor.
- WebSocket telemetry/state, MJPEG camera data, and byte-range dataset video
  remain documented low-level public read surfaces rather than sync JSON SDK
  wrappers.

### ROS-54 — Production Mintlify documentation

Commit: `d949b83`

- `docs/docs.json` owns production navigation for installation, quickstart,
  configuration, all six workflows, SDK/API, architecture, operations,
  screenshots, troubleshooting, and release notes.
- Existing detailed operational guides were retained and updated to the current
  product rather than replaced with stale V1 content.
- The preliminary `san-diego` branch at `7614a54` was inspected as a reference
  for structure, navigation, validation, and gallery organization; useful
  patterns were selectively reapplied against the current product, while its
  stale screenshots and product claims were not merged or cherry-picked.
- Link, anchor, asset, navigation, frontmatter, shell/JSON example, and secret
  validation is automated through `npm run docs:validate`.
- The README is a concise product/launch entry point. Current screenshots are
  explicitly labeled as deterministic mock-mode UI evidence, not hardware or
  cloud validation.

### ROS-52 — Managed Modal training through the Trainer API

Primary commits: `be2a424`, `16c6af5`; release hardening: `9f1771e`,
`90fb551`, `5a65d98`, `72707ec`, `8d3a62a`

- A durable `ManagedTrainingJob` is linked one-to-one with the existing
  `TrainingRun`; external reporting remains compatible.
- `POST /api/trainer/jobs` creates durable state before provider mutation and
  returns asynchronously. A caller-owned UUID plus canonical request hash makes
  exact replay idempotent and mismatched replay a conflict.
- A partial unique database index permits exactly one nonterminal managed job.
  This is a hard V1.1 aggregate cost guard, not a queue.
- The public request accepts only a fixed LeRobot 0.4.4 SmolVLA workflow,
  immutable dataset/base-model identities, a new exact output repository,
  bounded steps/workers/telemetry/checkpoints, a 1–1,440 minute deadline, and
  one acknowledged A10G/A100/H100 allocation. Lambda, Auto sizing, arbitrary
  code, arguments, environment, and callback URLs are excluded.
- The output repository is private by default. Public output requires a
  separate acknowledgement. ctrl-π writes and verifies an ownership marker,
  never chooses a model license, and retains ambiguous/partial repositories as
  evidence instead of destructively reusing them.
- Structured worker events reuse the bounded Training-run metric, console, and
  checkpoint stores. Sequence loss is surfaced as `event_gap` rather than
  silently hidden.
- Terminal job errors preserve the primary launch/execution/deadline cause
  across cleanup retries. Transient teardown diagnostics are removed once
  teardown is verified, so the UI never says cleanup is unverified beside a
  verified terminal state.
- Checkpoint and final results must match the exact job/repository, use verified
  immutable commits, and include the deployable policy root plus the pinned
  `.ctrl-pi-smolvlm/` dependency bundle required by the offline inference
  runtime.
- Completion, failure, cancellation, and deadline expiry do not become terminal
  until the exact owned App is stopped/absent with zero tasks. Startup
  reattaches only a proven persisted FunctionCall, never relaunches an
  ambiguous job, and watches only a marked, zero-task deterministic App through
  its deadline while provider deployment/image build settles. An active task
  without a durable FunctionCall is unobservable and is stopped promptly with
  verified teardown. Graceful shutdown preserves a deliberately running job
  after its exact handle is durably stored.
- `make modal-panic` independently discovers and verifies both inference and
  managed-training ownership namespaces.
- The Training UI observes managed lifecycle, output identity, event gaps, and
  teardown truth, but intentionally cannot launch or cancel paid compute.

### ROS-56 — User-facing AI-agent integration guide

Commit: `9c29ba9`

- `docs/ai-agent-integration.md` is the canonical policy for users' automation;
  it does not replace the repository developer instructions in `AGENTS.md`.
- Agents use only the same `CtrlPiClient` or documented REST API. No alternate
  identity, route, callback, raw action, direct provider path, or backdoor was
  added.
- The guide defaults agents to read-only and mock-first operation, requires
  fresh human confirmation for motion/cloud/public/reset/panic actions, treats
  returned metadata/logs/errors as untrusted data, and prohibits credential or
  backend-internal access.
- Examples reconcile uncertain mutations before retry, use caller-owned
  idempotency, bound polling, unwind episode before teleop, stop inference, and
  require managed/inference teardown verification.

## Architecture after V1.1

```text
Browser / CtrlPiClient / compatible Trainer Client
                         │
                         ▼
                 FastAPI REST + WebSockets
                         │
       ┌─────────────────┼────────────────────┐
       ▼                 ▼                    ▼
  YAMDriver + rig   PostgreSQL control    HF/Modal services
       lease          plane state          and artifacts
       │                                      │
       ▼                                      ├─ datasets/models
 MockYAMDriver or                            ├─ inference Apps
 fail-closed real YAM                       └─ managed-training Apps
```

Important ownership boundaries:

- `YAMDriver` is the only hardware boundary. Frontend and SDK code contain no
  CAN, serial, vendor, or raw action path.
- `RigLease` permits one process-local robot-control owner at a time.
- PostgreSQL owns lifecycle truth; it does not store video, weights, raw
  provider logs, credentials, or live telemetry.
- Hugging Face owns dataset/model artifacts and immutable revisions.
- Each inference or training deployment has a deterministic, separately tagged
  Modal App. Cleanup uses exact persisted identity and verified task counts.
- The production deployment is deliberately one backend process/Uvicorn worker
  because robot loops, rig ownership, staging sentinels, and supervision are
  process-local. Running multiple backend replicas against one rig/database is
  unsupported.
- ctrl-π has no authentication, authorization, RBAC, or multi-user ownership.
  The service must remain on loopback or a trusted, firewalled LAN.

## Test and gate evidence

The entries below distinguish final release evidence from retained earlier
milestone evidence. All final product gates used release code HEAD `8d3a62a`;
the documentation and diff validators were rerun after this review artifact
was finalized.

| Gate | Recorded evidence | Release status |
| --- | --- | --- |
| Frontend `npm run build` | PASS: TypeScript/Vite built 1,836 modules; JS 471.56 kB and CSS 34.06 kB before gzip. | Final release gate passed. |
| Full backend `.venv/bin/pytest -q backend/tests` | PASS: 671 passed in 79.53 s; one upstream Starlette/httpx deprecation warning. | Final release gate passed. |
| Focused public/API/SDK/docs/YAM/inference/deployment tests | PASS after final safety fixes: 179 passed with one upstream Starlette/httpx deprecation warning. | Supporting evidence, not a substitute for full pytest. |
| Targeted settings/YAM/rig/recording/upload/dataset/episode suite | PASS: 180 tests. | Supporting evidence. |
| Managed provider/runtime/UI focused suite | PASS at provider handoff: 168 tests; frontend build passed. | Supporting evidence; no real managed workload claim. |
| Python validation | PASS: `compileall`, `pip check`, and `git diff --check` after the final safety fixes and review-document updates. | Final release gate passed. |
| Mintlify `npm run docs:validate` | PASS after final documentation changes: repository checker, Mintlify build validation, links, and anchors. | Final release gate passed. |
| Alembic | PASS against the configured PostgreSQL database: one declared `0010` head, `upgrade head`, `current`, and `check` all clean with no model drift. | Final release gate passed. |
| Mock-mode backend/frontend boot | PASS: current Uvicorn and Vite servers started on loopback; the root plus all six workflow routes and Settings returned the SPA; health, settings, YAM setup, two-arm REST/WebSocket telemetry, reserved-path 404, strict-consent 422, and clean shutdown all passed. | Final release gate passed. |
| `make smoke` | PASS on final V1.1 code: 5.000 s / 50 frames, real private Hub upload and discovery, 100 mock-arm actions, exact teardown, Hub deletion, and workspace removal. | Final release gate passed; exact evidence is retained below. |
| Docker build/runtime | PASS: final image `sha256:19a84ad742b0e297c3bd7c5dc1bfd65aa391fe3e3970109e203e58be5bcf0ed6` (2,927,838,257 bytes) ran as UID/GID 10001 with zero effective capabilities and `no-new-privileges`; current routes/APIs/WebSocket, writable staging, utility DB bypass, fake-Hub smoke, graceful shutdown, and scoped cleanup all passed. | Final release gate passed. |
| Real physical YAM | Safe unavailable-hardware probe only; no device was opened. | **NOT VALIDATED**; complete the checklist below. |
| Real managed Modal training | Three attempts reached the App/image-build window and reproduced the pre-FunctionCall reconciliation bug; `5a65d98` preserved zero-task settlement and `72707ec` added the active-task cost bound. A fourth received a transient deploy failure. None spawned a FunctionCall or GPU task. | **NOT END-TO-END VALIDATED**; no successful train/publish/deploy result. |

### Final release commands to record

Run these against the exact commit being approved:

```bash
npm run build
.venv/bin/pytest -q
.venv/bin/python -m compileall -q backend/src backend/tests
.venv/bin/python -m pip check
git diff --check
npm run docs:validate

export DATABASE_URL='postgresql://ctrl_pi:password@localhost:5432/ctrl_pi'
.venv/bin/alembic -c alembic.ini heads
.venv/bin/alembic -c alembic.ini upgrade head
.venv/bin/alembic -c alembic.ini current
.venv/bin/alembic -c alembic.ini check

docker compose config -q
docker buildx build --check .
```

Then boot both development servers, exercise `/api/health`, `/settings`, all
six deep links, `/ws/arms`, and clean shutdown. Run the required smoke and the
relevant current Docker build/runtime/teardown checks described below.

### Final V1.1 real-Hugging-Face smoke

The required final `make smoke` run passed on 2026-08-29:

- Recording: 5.001 s wall time, 5.000 s episode, 50 synchronized frames, and a
  9,548-byte H.264 MP4.
- Upload/discovery: private
  `rosikand/ctrl-pi-smoke-20260829064906-48aed5e412` at immutable revision
  `3feb6a9e273f5fbecb5dc499165208015ee13d7a`, one episode, 50 frames.
- Inference: Stub target/runtime completed exactly 100 writes to
  `yam-follower`.
- Teardown: zero recording, inference, and compute resources; rig free; upload
  idle; Hub repository deleted; local workspace removed.

### Final V1.1 source and Docker runtime validation

The source-mode Uvicorn and Vite servers started without errors on loopback.
The root, Arms, Record / Teleop, Datasets, Training, Models, Inference, and
Settings routes all served the current SPA. Health reported mock mode; Settings
and YAM setup were truthful; REST and WebSocket arms telemetry returned the two
connected mock arms; a reserved API path returned 404; a coercive string YAM
motion acknowledgement returned 422. Both servers then shut down cleanly.

The final production image is
`sha256:19a84ad742b0e297c3bd7c5dc1bfd65aa391fe3e3970109e203e58be5bcf0ed6`
(2,927,838,257 bytes). `docker buildx build --check .` reported no warnings and
Compose configuration validation passed. The isolated audit container:

- became healthy with zero restarts as user/UID/GID `ctrl-pi`/10001/10001;
- had zero effective capabilities, `cap_drop=ALL`, `NoNewPrivs=1`, and an init
  process;
- served the root, all seven deep application routes, and a representative
  dataset detail route while keeping reserved API/asset/media/WebSocket paths
  at 404;
- returned mock health, Settings, YAM setup, two six-joint arms, and both arms
  over WebSocket, with writable confined staging;
- ran the in-image fake-Hub smoke for 5.000 seconds, 50 samples, one listed
  episode, exactly 100 follower actions, and zero retained recording,
  inference, compute, repository, or workspace resources;
- ran a utility command with an intentionally unreachable `DATABASE_URL` and
  no network, proving non-Uvicorn commands bypass the migration entrypoint;
- completed graceful application shutdown without OOM, after which the exact
  audit container and volume were removed and its loopback port was closed.

### Retained V1 external evidence

The historical details are in `FINAL_REVIEW.md` and remain evidence for the V1
baseline, not a replacement for V1.1 final gates:

- The real-HF smoke recorded a 5.001 s wall-time / 5.000 s episode with 50
  synchronized samples and a 9,548-byte H.264 MP4.
- It uploaded private dataset
  `rosikand/ctrl-pi-smoke-20260828232142-1342605b2d` at immutable revision
  `bfbd6a394bb4ae4b531ba934763035d8829b8cf8`, rediscovered it, ran exactly 100
  Stub inference writes, verified zero local resources, deleted the Hub repo,
  and removed the workspace.
- The retained Milestone 11 real Modal inference proof used
  `rosikand/ctrl-pi-demo-act-aloha@7fe3b8c62b715f56907701faa00a0bf665873c9d`,
  ran 115.6 s on A10G, completed 290 chunks and 5,778 mock-arm actions at
  49.45 Hz, stopped App `ap-ASxbec6ZI3XEtwe0a4Sbpl` with zero tasks, and
  deleted the temporary proxy credential.
- The V1 Docker image ran as UID/GID 10001 with one Uvicorn worker and verified
  FFmpeg/libx264, CPU Torch, LeRobot 0.4.4, Modal 1.5.4, YAM plugin imports,
  SPA/API/WebSocket access, writable staging, and scoped teardown.

### Managed Modal/HF evidence and remaining validation

V1.1 has strong fake-provider, fake-Hub, crash-window, cancellation, deadline,
startup, shutdown, artifact, and panic-cleanup coverage. During release review,
three real Modal attempts reached the pre-FunctionCall App/image-build window.
The then-current reconciler stopped those Apps prematurely; that specific
behavior is covered by `5a65d98`, which preserved a marked zero-task settlement
window, and `72707ec`, which fail-stops an active task lacking a durable
FunctionCall. A fourth attempt received a transient deploy error before any
handle was persisted. No attempt produced a FunctionCall or ran a GPU task.

The four durable audit job IDs were
`4a3ca788-1b09-427b-b08a-815012932408`,
`c6cd874c-4590-40ec-a1b9-a37c28ff8745`,
`701a56a7-5eeb-45e5-899c-1a779dbf2f7f`, and
`d8295d7c-8555-49de-a676-7e0cdf251c21`. The first three failed with verified
teardown after reproducing the premature image-build reconciliation. The
fourth held the single-job slot while the ambiguous launch settled, then became
terminal `failed` with teardown verified at 2026-08-29 06:36:25.688 UTC,
0.269527 seconds after its immutable deadline. A controlled, unsupported
second-backend concurrency audit against the real settling row neither
relaunched, adopted an unproven handle, nor published false teardown; the final
`running_tasks` discriminator observed the zero-task App and left settlement to
the original supervisor. Production still requires exactly one backend.

Final real-account cleanup reverified each job marker before deleting all four
private marker-only Hugging Face repositories. `make modal-panic` exited 0 with
zero active ctrl-π Apps to stop. Modal inspection reported zero active or
unverifiable inference/training Apps and zero running tasks; PostgreSQL reported
zero active managed jobs and zero active deployments. Four inert historical
training Apps remain visible to Modal as `stopped` with zero tasks. No
FunctionCall or GPU task was observed; billing was not independently reviewed.
Provider activity was limited to the deployment/image-build launch window.

The fourth row also exposed a final presentation defect: its verified terminal
state temporarily retained the earlier cleanup diagnostic “teardown is not yet
verified.” Commit `8d3a62a` now preserves primary failure causes and clears
transient cleanup text after verification. Because that already-terminal audit
row had lost its original provider error, it was repaired once to the truthful
generic “ended without an execution result” message; its outcome, timestamps,
provider identity, and teardown evidence were not changed.

Do not infer more from those attempts. Before describing managed training as
real-workload validated, run and retain evidence for all of the following:

1. Launch one short, private A10G job with exact immutable dataset and supported
   SmolVLA base-model commits through `CtrlPiClient`.
2. Confirm only one exact `ctrl-pi-training-<UUID>` App and matching ownership
   tag exist, the FunctionCall ID is persisted, and a concurrent launch returns
   409 without creating another App or repository.
3. Observe bounded logs, finite metrics, checkpoints, current step, provider
   state, and any event-gap signal through both the API and Training UI.
4. Verify every checkpoint and the final root in the exact output repository,
   private visibility, ownership marker, immutable lowercase commit, the four
   policy root files, and the complete pinned `.ctrl-pi-smolvlm/` manifest.
5. Deploy the final managed artifact with the existing offline LeRobot
   inference runtime and prove it loads without an undeclared nested Hub fetch.
6. Cancel a second job while active and verify the exact call/App is cancelled,
   stopped/absent, and reports zero tasks before terminal state.
7. Restart the backend during App-only image build, during an active persisted
   FunctionCall, and after provider completion. Confirm no relaunch, no early
   stop of an in-flight launch, exact reattachment only, and final cleanup.
8. Exercise deadline expiry and a transient provider/database outage; cleanup
   must remain loud and retryable until zero-task proof.
9. Run `make modal-panic` and record its final proof of zero active inference
   and managed-training tasks.
10. Confirm no credential, raw provider log, arbitrary code, unsafe artifact,
    or unbounded metric/log payload reached the request, database, UI, or image.

The product intentionally retains output repositories and ownership markers
from failed, cancelled, or ambiguous attempts. Review them manually; do not
delete them automatically or assume the same slug can be reused. The four
release-audit repositories were an explicit exception: all were private and
were manually deleted only after exact marker/ownership review and zero-task
cleanup proof. That operator cleanup does not change the product default.

## Exact launch instructions

### Source development in mock mode

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
${EDITOR:-vi} .env
export DATABASE_URL='postgresql://ctrl_pi:password@localhost:5432/ctrl_pi'
.venv/bin/alembic -c alembic.ini upgrade head
make seed
```

When Hub credentials are present, `make seed` creates or reuses the fixed
private seed repository as well as database rows. Leave the Hugging Face
credentials unset when a database-only local seed is intended.

Leave `CTRL_PI_MOCK_MODE=true`. Alembic deliberately does not load `.env`, so
`DATABASE_URL` must be exported in the Alembic shell. Start these in separate
terminals:

```bash
export DATABASE_URL='postgresql://ctrl_pi:password@localhost:5432/ctrl_pi'
PYTHONPATH=backend/src .venv/bin/uvicorn ctrl_pi.main:app \
  --reload --host 127.0.0.1 --port 8000
```

```bash
npm --prefix frontend run dev
```

Open <http://127.0.0.1:5173>. Do not bind an unauthenticated development
server to a public interface.

### Production Docker

```bash
cp .env.example .env
${EDITOR:-vi} .env
docker compose up --build -d --wait
curl --fail http://127.0.0.1:8000/api/health
```

Open <http://127.0.0.1:8000>. The entrypoint applies migrations when
`DATABASE_URL` is configured. That address must be reachable **from inside the
app container**; `localhost` there refers to the container itself, not a
PostgreSQL server on the host. Stop without deleting the staging volume:

```bash
docker compose down
```

Do **not** add `--volumes` while `ctrl-pi-data` contains recordings that have
not reached a verified `uploaded` state. Hardware-mode SocketCAN/USB setup and
device passthrough must follow `docs/docker-deployment.md` and the physical
validation checklist below.

### Documentation site

```bash
npm run docs:validate
npm run docs:dev
```

The development command serves the repository-owned Mintlify tree. Treat a
successful docs build as documentation validation, not product or hardware
validation.

### Python SDK

Install from the checkout, then use the fixed trusted origin:

```bash
. .venv/bin/activate
python -m pip install -e ./backend
```

```python
from ctrl_pi import CtrlPiClient


with CtrlPiClient("http://127.0.0.1:8000", timeout=10.0) as ctrl:
    print(ctrl.health())
    print(ctrl.get_system_status())
    print(ctrl.list_arms())
```

Closing the client does not stop teleop, an episode, inference, a deployment,
or managed training. Use the explicit stop/cancel methods in `finally`, then
verify state. See `docs/python-sdk.md` and `docs/ai-agent-integration.md` before
delegating mutations.

### Smoke tests

The required release smoke uses mock arms/camera/compute but a real private Hub
dataset lifecycle:

```bash
export DATABASE_URL='postgresql://ctrl_pi:password@localhost:5432/ctrl_pi'
export HF_TOKEN='<backend-only token>'
export HF_NAMESPACE='<exact user or organization>'
make smoke
```

For a network-free development check only:

```bash
export DATABASE_URL='postgresql://ctrl_pi:password@localhost:5432/ctrl_pi'
PYTHONPATH=backend/src .venv/bin/python -m ctrl_pi.smoke --fake-hub
```

The fake-Hub command is not the required release smoke and does not validate
Hugging Face or Modal.

## Known limitations and owner actions

- **No authentication:** any process that reaches ctrl-π can invoke every
  public operation, including motion and paid compute. Bind to loopback or a
  trusted, firewalled LAN. There is no RBAC or multi-user ownership.
- **One backend process:** robot control and lifecycle supervision are
  process-local. Use the supplied single-worker production layout; horizontal
  replicas against one rig/database are unsupported.
- **One managed job:** exactly one nonterminal managed-training job is allowed.
  There is no queue, Lambda, Auto sizing, BYOC, user code, or browser launch.
- **Managed training evidence:** the real App launch window was exercised, but
  a complete real train/checkpoint/final-artifact/deploy/cancel cycle remains to
  be validated as listed above.
- **Physical YAM:** no physical rig was validated. Real support is limited to
  the standard SocketCAN YAM follower, calibrated GELLO serial leader, and
  `crank_4310` gripper. YAM Pro/Ultra and linear grippers are unsupported.
- **Hardware acknowledgement:** a vendor write acknowledges submission to an
  in-memory command buffer, not confirmed physical motion. A stuck vendor I/O
  call can outlive the bounded application wait; use the physical E-stop and
  make the rig safe independently.
- **Camera:** the V1.1 hardware path still uses the synthetic camera.
- **OpenPI:** mock behavior is deterministic, but real OpenPI serving is
  unavailable and fails before resource creation.
- **Local staging:** finalized, unuploaded MP4/JSONL/manifests live only under
  `RECORDING_STAGING_DIR` or the Docker `ctrl-pi-data` volume. PostgreSQL, Hub,
  or credential restoration cannot recreate them. Back up and preserve that
  storage until the upload revision and local `uploaded` transition are both
  durable.
- **Preserved local recording:** this review deliberately retains ready,
  unuploaded recording `0be1b3e5-3fe0-4cb2-863e-b2c84ff1754a`: one 30.050-second
  episode with 601 samples, no Hub repository, and 2.72 MiB under
  `.ctrl-pi/recordings/0be1b3e5-3fe0-4cb2-863e-b2c84ff1754a/`. It is operator
  evidence, not disposable cache; preserve or back it up until deliberately
  uploaded or archived.
- **Artifact retention:** failed/ambiguous managed output repositories and
  markers are retained deliberately. Partial dataset uploads also retain local
  raw staging. Cleanup must not erase evidence or unrelated user artifacts.
- **Licensing:** the repository is public but has no source-code license or
  package license metadata. The owner must choose and add a license and align
  the repository/package metadata before describing ctrl-π as open source.
  Dataset/model cards do not infer a user artifact license; public visibility
  and licensing are separate operator decisions.
- **Modal internals:** exact lifecycle/task proof is pinned to Modal 1.5.4 and
  uses a narrow internal protocol boundary where the public SDK lacks the
  required inspection primitive. Revalidate before dependency upgrades.
- **Frontend testing:** there is no separate browser unit/E2E runner. The gate
  is strict TypeScript/Vite build plus SPA/API/WebSocket, smoke, and container
  integration coverage.

## Required physical Ubuntu/YAM validation

Before physical teleoperation, jog, auto-restore, or inference motion, an
operator with immediate emergency-stop access must complete and record every
item below on the target Ubuntu/YAM box:

1. Confirm the exact supported standard YAM/GELLO/`crank_4310` hardware and the
   pinned YAM `0.1.1` plugins with LeRobot 0.4.4.
2. Configure SocketCAN, bitrate, interface-up state, permissions, termination,
   and bus-off recovery. Verify the same access from the production container
   if Docker is used.
3. Select a stable `/dev/serial/by-id/...` leader device, correct
   dialout/group permissions, and an existing noninteractive GELLO calibration
   file.
4. Validate the MuJoCo XML and all referenced assets, `joint1`–`joint6`, and
   `grasp_site` against the actual mounted robot.
5. Use Settings discovery and preflight first. Confirm they do not open devices
   and treat `calibration_ready` only as artifact-structure evidence.
6. Run `make yam-probe` without `--connect`. Resolve every package, model,
   calibration, serial, CAN, and permission diagnostic before proceeding.
7. Secure the workspace, verify power and E-stop access, then run the explicit
   connected probe documented in `docs/yam-driver.md`. Confirm bounded cleanup;
   an incomplete shutdown is a blocker.
8. Compare every reported joint with independent low-speed physical motion:
   zero offsets, signs, radians, limits, and wrist-flex/wrist-roll ordering.
   Verify FK position/orientation against the physical end effector.
9. Verify the gripper conversion physically: vendor crank/GELLO `0=open`,
   `1=closed` must appear publicly as `1=open`, `0=closed`, including follower
   velocity sign and mounted leader endpoints.
10. Check follower velocity/effort units and signs, gravity-compensation and
    position-control transitions, torque/safe state, hard/current limits,
    collisions, and one-at-a-time low-speed jog bounds.
11. Validate leader-to-follower teleop direction/rate, then recording alignment;
    remember that the camera remains synthetic.
12. Explicitly test saved setup with auto-restore off, manual Connect, consented
    boot restore, unplugged boot plus hot-plug, consent revocation, and the
    failure latch. No unexpected connection or repeated retry is acceptable.
13. Exercise leader disconnect, follower/CAN loss, bus-off, vendor exception,
    worker death, backend termination, and E-stop recovery. Confirm commands
    fail closed and make the hardware safe independently of software cleanup.
14. Only after all previous checks, validate a short inference session with the
    operator at the E-stop and require provider plus robot-loop teardown.

Until this checklist passes, keep `CTRL_PI_MOCK_MODE=true` and do not claim
that ctrl-π has physically validated the YAM rig.

## Release sign-off checklist

- [x] Replace every final-rerun placeholder with exact final-HEAD results.
- [x] Confirm the committed release worktree is clean and `git diff --check`
      passes.
- [x] Confirm the final implementation, this review, and all final corrections
      are committed and pushed to `origin/milestone-spec-review`.
- [x] Update the draft PR with the V1.1 issue summary, gate evidence, known
      limitations, real-hardware checklist, and managed-training validation gap.
- [x] Run `make modal-panic` in the configured real Modal account and retain the
      final zero-active-task output before ending release validation.
- [x] Preserve or explicitly resolve every local staged recording and retained
      managed output repository/marker.
- [ ] Complete the physical YAM checklist before enabling or advertising real
      robot motion.
- [ ] Select a source-code license and align repository/package metadata before
      calling ctrl-π an open-source release.
