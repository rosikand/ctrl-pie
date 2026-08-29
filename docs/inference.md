---
title: "Inference"
description: "Deploy revision-pinned policies, run the robot loop, and verify complete Modal teardown."
icon: "radio-tower"
---

ctrl-π separates **deployment** from **robot motion**. Deploying creates and
verifies a compute endpoint, but it never moves an arm. Starting inference is
a second operation that selects one connected follower, acquires the shared
rig lease, re-verifies the deployed policy identity, and only then permits
actions. Stopping joins arm writes before tearing down the provider.

![Inference tab with a verified deterministic mock deployment](/assets/ctrl-pi-inference.png)

The screenshot shows mock mode. Mock mode exercises the same browser and
orchestration lifecycle without Hugging Face or Modal calls; it is not
evidence of a real GPU deployment.

## Supported combinations

| Target | Runtime | Compute | V1 behavior |
| --- | --- | --- | --- |
| Stub target (`CTRL_PI_MOCK_MODE=true`) | Stub | CPU | Deterministic in-process policy and compute lifecycle; no external call. |
| Stub target | LeRobot or OpenPI selection | Modal GPU label | Deterministic stub emulates that runtime's typed identity for UI/tests; no framework is loaded. |
| Modal target (`CTRL_PI_MOCK_MODE=false`) | Stub | CPU | Trivial CPU workload used to prove Modal lifecycle. |
| Modal target | LeRobot | `Modal: A10G`, `Modal: A100`, or `Modal: H100` | Real revision-pinned LeRobot 0.4.4 policy on one GPU container. |
| Modal target | OpenPI | Modal GPU label | Rejected explicitly. A real OpenPI runtime is unavailable in V1. |

Modal is the only real compute provider in V1. `ComputeTarget` keeps provider
operations behind a small deploy/health/inspect/stop boundary, but ctrl-π does
not implement BYOC, queues, warm pools, or another cloud target.

## Lifecycle and API

The browser uses these backend routes:

| Operation | Route | Effect |
| --- | --- | --- |
| Deploy | `POST /api/inference/deployments` | Resolve identity, create one provider resource, verify health and provider state, return a `running` deployment. |
| List/detail | `GET /api/inference/deployments` and `GET /api/inference/deployments/{id}` | Read durable control-plane state. |
| Start motion | `POST /api/inference/deployments/{id}/start` | Validate one connected follower, acquire the rig, verify runtime identity, optionally prepare passive recording, then start the robot loop. |
| Read state | `GET /api/inference/deployments/{id}/state` | Read deployment/session state and live metrics; idle readiness uses cached provider lifecycle inspection only. |
| Stream state | `WS /api/inference/deployments/{id}/stream` | Receive server-only state frames every 100 ms until stopped or failed. |
| Stop | `POST /api/inference/deployments/{id}/stop` | Stop and join robot writes, finalize optional recording, stop compute, and verify teardown. |

For a credential-free API exercise, start the backend in mock mode and run:

```bash
DEPLOYMENT_ID="$({
  curl --fail --silent --show-error \
    -H 'Content-Type: application/json' \
    -d '{
      "name":"Mock policy",
      "model_repo":"ctrl-pi/mock-policy",
      "checkpoint_revision":"0000000000000000000000000000000000000000",
      "runtime":"stub",
      "compute_size":"CPU"
    }' \
    http://127.0.0.1:8000/api/inference/deployments
} | python -c 'import json,sys; print(json.load(sys.stdin)["id"])')"

curl --fail -H 'Content-Type: application/json' \
  -d '{"arm_id":"yam-follower","task":"mock safety check"}' \
  "http://127.0.0.1:8000/api/inference/deployments/$DEPLOYMENT_ID/start"

curl --fail \
  "http://127.0.0.1:8000/api/inference/deployments/$DEPLOYMENT_ID/state"

curl --fail -X POST -H 'Content-Type: application/json' -d '{}' \
  "http://127.0.0.1:8000/api/inference/deployments/$DEPLOYMENT_ID/stop"
```

Deployment and session statuses are related but distinct. A provider can be
`running` while its robot session is `idle`. Treat `teardown_verified=true`
and a stopped deployment as the terminal success condition; a database status
alone is not proof that external compute stopped.

## Immutable model identity

For real LeRobot deployment, the requested repository must belong to the
exact configured `HF_NAMESPACE`. The backend uses its explicit `HF_TOKEN` to
resolve a branch, tag, or SHA through the Hub and accepts only the exact same
repository plus a lowercase 40-character commit SHA. It persists that SHA
before creating Modal resources. Browser-supplied credentials and repositories
outside the namespace are rejected.

The Modal image build receives `HF_TOKEN` only as a build secret and downloads
that one repository revision into `/opt/ctrl-pi/model`. It writes a small
identity marker containing the repository and SHA, then enables both
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for the serving image. Startup
fails closed if either offline flag, the identity marker, or required policy
files are missing or mismatched.

The selected Hub revision must therefore be a deployable LeRobot policy at
the **model repository root**. In particular, `config.json`,
`model.safetensors`, `policy_preprocessor.json`, and
`policy_postprocessor.json` must be root files, not only files below
`checkpoints/<step>`. The executable upload in the
[Trainer API](/trainer-api#end-to-end-lerobot-fine-tune-and-checkpoint-flow)
demonstrates and tests this layout.

## Runtime adapters

`InferenceRuntime` exposes `load`, `describe`, `reset_session`, `predict`, and
`close`. The common protocol describes named vector/image/language inputs, one
action feature, the maximum actions per chunk, runtime kind, model repository,
and immutable revision.

The real `LeRobotRuntime` preserves LeRobot 0.4.4 policy semantics locally. It
loads `PreTrainedConfig`, the registered policy class, and LeRobot's native
preprocessor and postprocessor from the pinned directory with
`local_files_only=True`; moves the policy to the requested device; calls
`policy.reset()` per transport session; and executes
`predict_action_chunk()` under PyTorch inference mode. Images are decoded as
bounded JPEGs and resized to the declared feature shape. The adapter never
loads a mutable Hub ref while serving.

ctrl-π intentionally does not expose LeRobot's pickle/gRPC serving protocol
to the network. Framework-native policy/pre/postprocessing sits behind a
provider-neutral, bounded JSON envelope so the robot loop does not contain
framework-specific logic. `OpenPIInferenceRuntime` is an explicit
`RuntimeUnavailableError` placeholder; mock mode's OpenPI choice is only the
deterministic stub contract.

## Modal identity and cost guardrails

Every deployment owns one deterministic App:

- App name `ctrl-pi-<canonical deployment UUID>`;
- Modal tag `ctrl-pi-deployment=<the same UUID>`; and
- a persisted provider App ID, target kind, canonical endpoint, model repo,
  runtime, and commit SHA.

Creation refuses a name collision, and all later inspection/stop operations
re-check provider ID, exact name, and ownership tag. A deployment persisted as
`stub` is never stopped through Modal after a configuration change, or vice
versa. Restart and deadline cleanup construct the adapter named by that
persisted kind. Missing Modal credentials leave a visible retryable failure;
they never turn cleanup intent into a falsely verified stop.

The resource policy enforces zero warm containers (both minimum and buffer
counts are zero), exactly one maximum container, idle scaledown no longer than
60 seconds, and a hard timeout from 1 to 30 minutes. Real LeRobot accepts only
A10G, A100, or H100.
Use the cheapest adequate GPU, keep real checks under ten minutes, and always
verify teardown. The project-wide development budget in the specification is
approximately $30 of Modal credit.

## Runtime transport boundary

Modal endpoints require a separate Proxy Token pair: an ID beginning `wk-`
and a secret beginning `ws-`. Modal API credentials (`ak-`/`as-`) cannot be
used for endpoint traffic. All four values stay in the backend environment;
none is accepted from or returned to the browser.

Only an HTTPS root URL on a canonical lowercase `*.modal.run` host is
accepted: no explicit port, user information, query, fragment, redirect, or
non-root path. The runtime HTTP client ignores environment-derived proxies.
The WebSocket is derived as WSS at `/ws`; that connector also disables proxy
discovery and compression and bounds its receive queue to one message. HTTP
and WebSocket transports do not follow redirects.

Requests and responses use strict discriminated JSON. Unknown fields,
duplicate object keys, non-finite numbers, invalid feature names, unsafe
dimensions, malformed base64/JPEG payloads, and identity/correlation
mismatches fail closed. An action response is capped at 2 MiB, 100 rows, and
256 finite scalars per row. WebSocket observations are capped at 2 MiB; a
larger otherwise-valid observation is routed to bounded POST only before a
WebSocket action establishes a stateful session.

The robot client permits one inference request in flight. Once an observation
may have been accepted over WebSocket, a send/receive failure is never replayed
over POST. Every returned chunk must echo its UUID request ID, observation
sequence, and expected model revision, and chunks must arrive in increasing
sequence order.

## Robot loop and recording

Start requires the deployment to be provider-verified `running` and the arm
to be connected with role `follower`. Before the first write, the loop:

1. acquires the application-scoped `RigLease` as `inference`;
2. calls runtime `describe` and compares runtime, repository, and SHA with the
   persisted deployment;
3. sends a random health nonce and verifies its echo and the same identity;
4. if requested, opens passive inference recording under that exact lease;
   and
5. schedules the 50 Hz action loop.

The loop maintains a bounded queue of 200 actions by default. It rejects an
entire response chunk if queue capacity would be exceeded, drops chunks or
remaining same-chunk actions older than two seconds, and stops on malformed,
non-finite, out-of-order, or mismatched output. A native YAM action must have
exactly seven values: six absolute joint targets plus gripper. Default
per-step limits are `0.35` radians and `0.20` gripper units, in addition to
global mock joint/gripper bounds. Non-seven-dimensional policies are projected
only when a `MockYAMDriver` is explicitly in use; real arms fail closed.

Inference recording does not start teleoperation and does not acquire a
second lease. Its start hook opens the camera/episode after runtime identity
verification but before the action task is scheduled, and its observer records
only actions already applied by the loop. On stop, the resulting recording is
finalized locally. Upload to Hugging Face remains a separate, explicit action
through the normal recording pipeline; see
[Record and teleoperate](/recording).

## Stop, restart, and emergency cleanup

Normal Stop is deliberately ordered:

1. signal the robot loop and join it, including any already in-flight network
   call;
2. clear queued actions, close the transport, and release the rig lease;
3. finalize or fail the optional local recording; and
4. in an unconditional cancellation-deferring path, stop the owned provider
   App and verify a stopped/absent lifecycle with zero running tasks.

If local cleanup fails, provider stop is still attempted. If provider proof
fails, the session/deployment remains failed and retryable; ctrl-π never
claims verified teardown based only on intent or a database update.

Startup never resumes robot motion or calls the runtime web endpoint. It stops
unattended providers through the adapter named by each row's persisted target
kind because no process-local inference loop survived the restart. The
watchdog retries failed cleanup while the app runs, including after switching
between mock and hardware mode. Shutdown uses the same stop ordering. If the
normal route or shutdown was interrupted, run the
ownership-safe procedure in [Modal operations](/modal-operations).
The panic command is a provider cleanup tool only; it does not finalize local
recordings or revoke Proxy Tokens.

## Retained Milestone 11 live evidence

The V1 real-GPU acceptance run was retained on 2026-08-28. This is distinct
from the recurring stub smoke gate in [Full mock smoke gate](/smoke-test).

| Evidence | Result |
| --- | --- |
| Policy | Private `rosikand/ctrl-pi-demo-act-aloha` at `7fe3b8c62b715f56907701faa00a0bf665873c9d` (206,785,023 bytes) |
| Source model | `qualia-robotics/act-aloha-sim-transfer-cube-human-a76db398@3c027cc108e266b1f790e7af5cb51697aa1aa1f6` |
| Offline proof | LeRobot 0.4.4 loaded on local CPU with both offline flags; a 14-value state plus three 480×640 visual inputs produced a 20×14 action chunk. |
| Modal deployment | Deployment `a8c268e2-e455-48fb-bff1-ea7fad44cfc4`, App `ap-ASxbec6ZI3XEtwe0a4Sbpl`, A10G |
| Run window | `2026-08-28T18:41:47.539772Z` through `2026-08-28T18:43:44.670598Z` (115.6 seconds recorded) |
| Action execution | 5,778 actions from 290 chunks; 49.452478447646975 Hz; 119.69564296203568 ms average and 117.37259899746277 ms final latency; zero dropped chunks; final queue depth zero |
| Dataset | Private [`rosikand/ctrl-pi-m11-live-modal-act-20260828`](https://huggingface.co/datasets/rosikand/ctrl-pi-m11-live-modal-act-20260828) at `743f28ba80c737ea87ea3cc9447fa308b549b700`: LeRobot v3, one episode, 2,312 frames at 20 fps |
| Teardown | Deployment stopped and provider-verified; Modal AppList showed the App stopped with zero tasks; `modal-panic` found zero active owned Apps; the ephemeral Proxy Token was deleted and later listing found zero matching tokens. |
| Milestone commit | `b34d9a63e16e9642f821e61d4e9a40a28d769e9b` |

The demo copied the source policy into an operator-owned private repository
and set the owned repo ID plus `pretrained_backbone_weights=null` in its
configuration, preventing an implicit torchvision download during strict
offline load. The dataset link requires access to that private namespace.
