---
title: "YAM setup"
description: "Discover, preflight, save, and safely restore one supported YAM leader/follower pair."
icon: "cable"
---

ctrl-π supports one standard YAM follower over SocketCAN, one calibrated
GELLO/Dynamixel leader over a stable serial path, and the `crank_4310`
gripper. YAM Pro, YAM Ultra, linear grippers, other leaders, and multiple
pairs are outside this release. The camera remains synthetic in both modes.

<Warning>
  No physical YAM was available in the development environment. Fake-vendor
  and mock tests validate software behavior, not calibration accuracy, joint
  directions, controller state, bus behavior, limits, or the emergency stop.
  Complete the [Ubuntu/YAM validation checklist](/yam-driver#required-ubuntu-yam-validation)
  before normal motion.
</Warning>

## Before onboarding

Install the hardware extras and retain the versions pinned by ctrl-π:

- `lerobot==0.4.4`
- `lerobot-robot-yam==0.1.1`
- `lerobot-teleoperator-yam-gello==0.1.1`
- `yam-common==0.1.1`

Configure the selected SocketCAN interface on the Ubuntu host. Verify the
bitrate for your rig; this example uses 1 Mbit/s:

```bash
export YAM_CAN_INTERFACE=can0
sudo ip link set "$YAM_CAN_INTERFACE" down
sudo ip link set "$YAM_CAN_INTERFACE" type can bitrate 1000000
sudo ip link set "$YAM_CAN_INTERFACE" up
ip -details -statistics link show "$YAM_CAN_INTERFACE"
```

Use a stable leader path under `/dev/serial/by-id/` and grant the ctrl-π
process read/write permission through the appropriate device group. Provide
an absolute MuJoCo XML path whose relative assets are present and whose model
contains `joint1` through `joint6` and `grasp_site`.

Start the backend in hardware mode with a migrated PostgreSQL database:

```dotenv
CTRL_PI_MOCK_MODE=false
```

The `YAM_*` environment fields remain an optional bootstrap when no physical
setup row exists. They select a candidate configuration only: they do not
authorize startup to open a bus or engage a controller. Once a physical setup
is saved, that database row takes precedence. See [Configuration](/configuration).

## Create the leader calibration outside ctrl-π

Onboarding requires an existing LeRobot calibration JSON. ctrl-π never opens
an interactive calibration prompt and always constructs the leader with
`calibrate=False`.

Stop ctrl-π and every process that might own the leader port. Secure the
workspace, verify independent power and motion controls, keep an operator at
the emergency stop, and run the pinned LeRobot CLI with the same stable port,
ID, and directory you will enter in Settings:

```bash
.venv/bin/lerobot-calibrate \
  --teleop.type=yam_leader \
  --teleop.port=/dev/serial/by-id/REPLACE_WITH_LEADER \
  --teleop.id=yam-leader \
  --teleop.calibration_dir=/absolute/path/to/leader-calibration
```

For that example, onboarding expects
`/absolute/path/to/leader-calibration/yam-leader.json`. This interactive
command opens and configures the leader; it neither connects nor validates the
follower. Do not proceed if the detected device, prompts, or movement differ
from the reviewed YAM procedure.

## Onboard in Settings

Open **Settings → YAM onboarding** and follow the three numbered stages.

<Steps>
  <Step title="Discover host-visible devices">
    Choose **Discover devices**. The backend returns a bounded list of Linux
    network interfaces and stable serial candidates. Discovery does not
    import vendor modules, open a device, start a controller, or move an arm.
    ctrl-π never auto-selects an ambiguous list; identify the intended pair.
  </Step>
  <Step title="Configure and check readiness">
    Select one follower CAN interface and leader serial path. Enter the
    absolute MuJoCo XML path, calibration directory, and calibration ID, then
    choose **Run read-only preflight**. Paths resolve on the backend host, not
    in the browser.

    Preflight checks pinned package metadata, file bounds and readability,
    Linux ARPHRD_CAN type, serial character-device visibility, and the
    calibration JSON's pinned structural fields. It does not import vendor
    modules, open either bus, configure a motor, or prove that calibration is
    physically correct.
  </Step>
  <Step title="Save, then connect explicitly">
    Choose **Save YAM setup**. ctrl-π persists one bounded, non-secret setup
    row and applies it to the existing `YAMDriver` without opening hardware.
    A rejected change leaves the prior saved selection intact.

    Secure the rig, clear the workspace, verify independent power/motion
    controls, and place an operator at the emergency stop. Check the explicit
    motion-risk acknowledgment, then choose **Connect and verify one sample**.
    Connection can configure the leader bus and engage the follower's pinned
    gravity-compensation/control worker even though ctrl-π sends no jog or
    application action.
  </Step>
</Steps>

A changed saved configuration safely disconnects the old selection. Changing
only the automatic-restoration preference for the same connected setup does
not interrupt that connection.

## Automatic restoration and hot-plug

Automatic connection is off for every newly saved physical setup. Enabling it
requires a separate acknowledgment that a later boot or hot-plug may connect
the saved rig—and therefore engage the follower controller—without another
prompt.

With that consent enabled, the backend loads the saved configuration, performs
passive preflight while prerequisites are missing, and makes one serialized
connection attempt when readiness changes to ready. It never substitutes
`MockYAMDriver`, never opens devices during discovery or preflight, and never
blindly retries a latched runtime or vendor error. Settings refreshes the
sanitized waiting state, so a newly plugged-in saved rig becomes visible.

You can revoke future automatic-connection consent while the rig is absent:
uncheck it and choose **Disable automatic connection**. That no-I/O update
does not require preflight and does not disconnect an already connected rig.
Enabling it again or changing the selected configuration requires a current
passing preflight; re-enabling also requires the automatic-motion
acknowledgment again.

Use **Forget saved YAM setup** in hardware mode to disable restoration, close
any setup-owned connection, and delete the saved physical row. In mock mode,
setup and reset operations are deterministic but never overwrite or delete a
saved physical rig.

## Independent command-line probes

The onboarding preflight and this source-checkout command are non-opening:

```bash
make yam-probe
```

The command-line probe reads the process environment rather than the saved
PostgreSQL setup row. It checks required values, pinned packages, readable model/calibration
files, serial permissions, and network-interface existence. It does not prove
interface bitrate, import vendor modules, construct a device, or parse a
physical calibration for correctness.

Only after the passive checks pass and the rig is independently safe, run one
connected sample if your validation plan calls for it:

```bash
PYTHONPATH=backend/src \
  .venv/bin/python -m ctrl_pi.yam_probe --connect
```

The connected probe sends no application jog or action, but it is not
electrically read-only. It configures the leader serial bus, starts the
follower controller, reads one sample, and attempts bounded shutdown. If
vendor I/O does not return, make the hardware safe independently; process
output is not proof of electrical torque state.

For a container, first apply the scoped device/model override from
[Docker deployment](/docker-deployment), then run:

```bash
docker compose -f docker-compose.yml -f docker-compose.yam.yml \
  run --rm --no-deps app python -m ctrl_pi.yam_probe
```

## Failure and recovery

Configuration, import, calibration, connection, sampling, worker, and command
failures leave the API available but mark both stable arm IDs disconnected.
Motion fails before vendor mutation, and hardware mode never falls back to
mock arms.

- If saved auto-restore is waiting on missing prerequisites, correct or
  reconnect them; passive preflight can detect the ready transition once.
- If an explicit connection or runtime/vendor operation is latched as an
  error, inspect the rig and use manual **Connect** after correction, or
  restart the backend. There is no uncontrolled retry loop.
- If the setup fields changed, discover again, rerun read-only preflight, and
  save before connecting.

Before teleoperation, verify joint offsets, signs, units, wrist ordering,
MuJoCo forward kinematics, gripper inversion, firmware limits, disconnect and
shutdown behavior, and the emergency stop. Then issue one-at-a-time low-speed
joint and gripper jogs from **Arms**. Real Cartesian jog is unsupported.

## REST and Python access

The public setup endpoints are `GET`, `PUT`, and `DELETE /api/yam/setup` plus
`POST /api/yam/setup/discover`, `/preflight`, and `/connect`. The typed SDK
exposes `get_yam_setup()`, `discover_yam()`, `preflight_yam()`,
`save_yam_setup()`, `connect_yam()`, and `reset_yam_setup()` through the same
services. Both hardware connection paths require their acknowledgments in the
backend; bypassing the UI does not bypass the safety gates. See
[REST and Python SDK](/python-sdk).

For detailed mapping, limits, threading, cleanup, and the physical validation
checklist, continue to [YAM driver interface](/yam-driver).
