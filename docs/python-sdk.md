---
title: "REST and Python SDK"
description: "Automate ctrl-π workflows through the typed client and shared REST control plane."
icon: "code-xml"
---

`CtrlPiClient` is the typed, synchronous Python interface to the same ctrl-π
REST services used by the web console. It covers the current system, YAM setup,
arm, recording, dataset, model, externally reported training, and inference
workflows. It does not create a second control plane and it never accepts Hugging
Face, PostgreSQL, Modal, or hardware credentials.

Managed training is not part of this version of the SDK. Training scripts run
on user-managed compute, upload weights to Hugging Face, and report their run
metadata through ctrl-π. The focused legacy `ctrl_pi.trainer.Client` remains
compatible; see the [Trainer API](trainer-api.md).

## Install and connect

Install the package from a checkout with Python 3.11 or newer:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ./backend
```

Start the backend, then connect to its root URL:

```python
from ctrl_pi import CtrlPiClient


with CtrlPiClient("http://127.0.0.1:8000", timeout=10.0) as ctrl:
    print(ctrl.health())
    print(ctrl.get_system_status())
```

ctrl-π V1.1 has no authentication layer. Keep the backend on a trusted local
network and never place credentials in the URL, settings payloads, training
config, recording metadata, logs, or task text. The client rejects URL
userinfo, queries, fragments, and path prefixes, ignores environment proxy
configuration, does not follow redirects, and reads JSON through a streamed
64 MiB ceiling. `CtrlPiError.status_code` contains an HTTP status when the
server responded; sanitized error text never includes the response body or
underlying transport exception.

```python
from ctrl_pi import CtrlPiClient, CtrlPiError


try:
    with CtrlPiClient("http://ctrl-pi-box:8000") as ctrl:
        print(ctrl.list_arms())
except CtrlPiError as error:
    print(error.status_code, str(error))
```

Responses use strict, frozen Pydantic models (nested lists and dictionaries
remain ordinary Python collections). Common models and Literal aliases are
exported from `ctrl_pi`, and the distribution includes `py.typed` for static
type checkers.

## Model to YAM inference

Hugging Face remains the artifact source of truth. ctrl-π discovers models;
it does not proxy arbitrary model-file uploads. Select an immutable model
revision, deploy it, start the robot-side inference session, and always stop it
in a `finally` block:

```python
import time

from ctrl_pi import CtrlPiClient


with CtrlPiClient("http://ctrl-pi-box:8000") as ctrl:
    models = ctrl.list_models(refresh=True)
    selected = models.models[0]
    if selected.revision is None:
        raise RuntimeError("The selected model has no immutable Hub revision")

    deployment = ctrl.deploy(
        "YAM block-pick policy",
        model_repo=selected.repo_id,
        checkpoint_revision=selected.revision,
        runtime="lerobot",
        compute_size="Modal: A10G",
    )
    try:
        state = ctrl.start_inference(
            deployment.id,
            arm_id="yam-follower",
            task="pick up the block",
            record_session=True,
            recording_name="policy block-pick evaluation",
        )
        while state.session_status == "running" and state.steps_executed < 100:
            time.sleep(0.5)
            state = ctrl.get_inference_state(deployment.id)
    finally:
        stopped = ctrl.stop_inference(deployment.id)
        if not stopped.teardown_verified:
            raise RuntimeError("Provider teardown was not verified")
```

Closing `CtrlPiClient` only closes local HTTP resources. It does **not** stop
teleoperation, finish an episode, stop arm inference, or tear down a Modal App.
Call the matching stop operation explicitly, including on exceptions. Never
retry a motion or lifecycle mutation blindly after an uncertain failure; read
current state first. If a deployment response is lost, compare
`list_deployments()` with the prior list and stop the newly owned deployment
before retrying. If the API cannot perform verified cleanup, follow
[`make modal-panic`](modal-operations.md) from the arm-connected box.

Mock mode uses the built-in `ctrl-pi/mock-policy` identity, the `stub` runtime,
and `CPU`. Real LeRobot serving uses a Modal GPU; a real-mode OpenPI choice is
rejected before deployment because that runtime is unavailable in V1.1.

## YAM setup and arm control

Discovery and preflight are passive/read-only. A physical setup can be saved
only after preflight. Both unattended restoration and explicit connection have
separate, required risk acknowledgements:

```python
from ctrl_pi import CtrlPiClient, YAMSetupConfig


config = YAMSetupConfig(
    can_interface="can0",
    leader_port="/dev/serial/by-id/usb-YAM_GELLO_1234",
    mujoco_xml_path="/opt/yam/yam.xml",
    leader_calibration_id="yam-leader",
    leader_calibration_dir="/var/lib/ctrl-pi/calibration",
)

with CtrlPiClient("http://ctrl-pi-box:8000") as ctrl:
    discovery = ctrl.discover_yam()
    preflight = ctrl.preflight_yam(config)
    if not preflight.ready:
        raise RuntimeError(preflight.diagnostic.detail)

    ctrl.save_yam_setup(
        config,
        auto_restore=False,
        acknowledge_automatic_motion_risk=False,
    )
    ctrl.connect_yam(acknowledge_hardware_motion_risk=True)
    follower = ctrl.get_arm("yam-follower")
    ctrl.jog_arm(
        follower.id,
        kind="joint",
        axis="shoulder_yaw",
        delta=0.05,
    )
```

Acknowledgement values never default to `True`. Set them only after the area is
clear, an emergency stop is reachable, the follower is supported, and the
operator understands that gravity compensation or a command can move hardware.
Schema-valid calibration proves file readiness only; it does not prove safe
directions, offsets, limits, model fidelity, or emergency-stop behavior. The
physical Ubuntu/YAM validation checklist remains in the
[YAM driver guide](yam-driver.md).

YAM and arm methods:

| Method | Purpose |
| --- | --- |
| `get_yam_setup()` | Read saved/applied/connected setup state. |
| `discover_yam()` | Passively enumerate bounded SocketCAN and stable serial candidates. |
| `preflight_yam(config)` | Read-only readiness and calibration-artifact check. |
| `save_yam_setup(config, *, auto_restore, acknowledge_automatic_motion_risk)` | Persist/apply one setup. |
| `connect_yam(*, acknowledge_hardware_motion_risk)` | Explicitly open the configured rig. |
| `reset_yam_setup()` | In hardware mode, disconnect and forget the saved physical setup; mock mode preserves any physical row. |
| `list_arms()` / `get_arm(arm_id)` | Read arm telemetry snapshots. |
| `jog_arm(arm_id, *, kind, axis, delta)` | Send one server-bounded relative jog. |

## Recording and datasets

```python
with CtrlPiClient("http://ctrl-pi-box:8000") as ctrl:
    recording = ctrl.create_recording(
        "block pickup demonstrations",
        task="pick up the red block",
        leader_robot_id="yam-leader",
        follower_robot_id="yam-follower",
    )
    try:
        ctrl.start_teleop(recording.id)
        ctrl.start_episode(recording.id, operator="Ada")
        try:
            # The operator performs the demonstration here.
            ctrl.stop_episode(recording.id, success=True, notes="clean pickup")
        except Exception:
            state = ctrl.get_recording_state(recording.id)
            if state.episode_active:
                ctrl.stop_episode(
                    recording.id,
                    success=False,
                    notes="aborted after a caller failure",
                )
            raise
    finally:
        state = ctrl.get_recording_state(recording.id)
        if state.episode_active:
            ctrl.stop_episode(recording.id, success=False, notes="aborted")
            state = ctrl.get_recording_state(recording.id)
        if state.teleop_active:
            ctrl.stop_teleop(recording.id)

    uploaded = ctrl.upload_recording(
        recording.id,
        repo_name="yam-block-pickup",
        private=True,
    )
    print(uploaded.repo_id, uploaded.revision)
```

Repository visibility is a required argument; use `private=True` unless the
operator has intentionally approved public release and chosen any appropriate
data license. `HF_TOKEN` stays in the backend.

| Method | Purpose |
| --- | --- |
| `list_recordings()` / `get_recording(id)` | Read persisted sessions. |
| `get_recording_state(id)` | Read live session/episode state. |
| `create_recording(...)` | Create one leader/follower recording session. |
| `start_teleop(id)` / `stop_teleop(id)` | Control the teleop lifecycle. |
| `start_episode(id, ...)` / `stop_episode(id, ...)` | Record and durably finalize an episode. |
| `upload_recording(id, *, repo_name, private)` | Upload a finalized LeRobot dataset to HF. |
| `list_datasets(...)` | Browse one bounded namespace page. |
| `list_dataset_episodes(repo_name)` | List episodes at a pinned revision. |
| `get_dataset_episode(repo_name, index, *, revision)` | Read the bounded synchronized timeline. |

`DatasetEpisode.video_url` is the same-origin, revision-pinned byte-range REST
endpoint for private video playback. Use a range-capable HTTP/media client
against the backend URL; do not append an HF token.

## Externally reported training

The universal client keeps the familiar trainer method names:

| Method | Purpose |
| --- | --- |
| `create_run(name, ...)` | Create an externally managed training run. |
| `list_runs(status=None)` / `get_run(id)` | Read reported runs. |
| `update_run(id, ...)` | Update status, step, config, and artifact metadata. |
| `log_metrics(id, *, step, metrics)` | Report finite scalar metrics. |
| `list_console_logs(id, *, after_sequence, limit)` | Page through the bounded sanitized log tail. |
| `log_console(id, *, line, source, step)` | Append one already-scrubbed console line. |
| `register_checkpoint(id, *, repo_id, revision, step)` | Register a checkpoint already uploaded to HF. |

Never forward environment dumps or raw subprocess output without scrubbing it.
The backend rejects common credential shapes, but the caller remains
responsible for removing secrets.

## Complete method and REST map

| Product surface | SDK methods | REST routes |
| --- | --- | --- |
| System | `health`, `get_system_status` | `GET /api/health`, `GET /api/settings/status` |
| Settings | `get_settings`, `update_settings` | `GET/PATCH /api/settings` |
| YAM | setup methods above | `/api/yam/setup` and its `discover`, `preflight`, `connect` actions |
| Arms | `list_arms`, `get_arm`, `jog_arm` | `GET /api/arms`, `GET/POST /api/arms/{id}[/jog]` |
| Recording | recording methods above | `/api/recordings` lifecycle routes |
| Datasets | dataset methods above | `GET /api/datasets...` |
| Models | `list_models` | `GET /api/models` |
| Training | reported-training methods above | `/api/trainer/runs...` |
| Inference | `deploy`, `list_deployments`, `get_deployment`, `get_inference_state`, `start_inference`, `stop_inference` | `/api/inference/deployments...` |

The public backend also exposes low-level streaming routes that the synchronous
JSON client deliberately does not wrap:

- `GET /api/camera/stream` — MJPEG camera stream;
- `GET /api/datasets/{repo_name}/episodes/{index}/video?revision={sha}` —
  revision-pinned byte-range episode video;
- `WS /ws/arms` — live arm telemetry;
- `WS /api/inference/deployments/{id}/stream` — live inference state.

Use those endpoints only on the same trusted network. The SDK polling methods
cover every state/control operation; streaming does not create a separate
motion, cleanup, or credential path.
