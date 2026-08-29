---
title: "YAM setup"
description: "Onboard the supported standard YAM follower and GELLO leader safely."
icon: "cable"
---

Hardware mode supports one standard YAM follower over SocketCAN, one
GELLO/Dynamixel leader over serial, and the `crank_4310` gripper. It uses the
exact `0.1.1` YAM plugin releases pinned by the backend package.

YAM Pro, YAM Ultra, linear grippers, alternate leader devices, and multiple
pairs are outside V1. The camera remains synthetic even when real arms are
selected.

<Warning>
  Cloud tests cover the adapter with fake vendor objects only. Before motion,
  secure the rig, clear the workspace, place an operator at an independent
  emergency stop, and complete the physical checklist in
  [YAM driver interface](/yam-driver).
</Warning>

## 1. Prepare the Ubuntu host

Install ctrl-π and confirm these packages remain at their pinned versions:

- `lerobot==0.4.4`
- `lerobot-robot-yam==0.1.1`
- `lerobot-teleoperator-yam-gello==0.1.1`
- `yam-common==0.1.1`

Configure the intended SocketCAN interface on the host. The example uses the
standard 1 Mbit/s bitrate; verify the correct value for the physical rig:

```bash
export YAM_CAN_INTERFACE=can0
sudo ip link set "$YAM_CAN_INTERFACE" down
sudo ip link set "$YAM_CAN_INTERFACE" type can bitrate 1000000
sudo ip link set "$YAM_CAN_INTERFACE" up
ip -details -statistics link show "$YAM_CAN_INTERFACE"
```

Use a stable leader path below `/dev/serial/by-id/` and grant the ctrl-π
process read/write access through the appropriate device group. Do not rely on
a changing `/dev/ttyUSB*` name.

## 2. Prepare model and calibration files

The follower needs an explicit MuJoCo XML with all referenced assets available
relative to it. The model must expose `joint1` through `joint6` and
`grasp_site` for gravity compensation and forward kinematics.

The leader needs an existing, noninteractive LeRobot calibration JSON. If the
calibration ID is `yam-leader`, the configured directory must contain
`yam-leader.json`. ctrl-π starts the plugin with `calibrate=False`; it never
opens a calibration prompt inside the server.

## 3. Configure hardware mode

```dotenv
CTRL_PI_MOCK_MODE=false
YAM_CAN_INTERFACE=can0
YAM_LEADER_PORT=/dev/serial/by-id/your-gello-leader
YAM_MUJOCO_XML_PATH=/absolute/path/to/yam_crank_4310.xml
YAM_GRIPPER_TYPE=crank_4310
YAM_LEADER_CALIBRATION_ID=yam-leader
YAM_LEADER_CALIBRATION_DIR=/absolute/path/to/calibration
```

Hardware mode also requires PostgreSQL, verified Hugging Face access, Modal
API credentials, and Modal Proxy Tokens for the complete Settings checklist.
See [Configuration and credentials](/configuration).

## 4. Run the non-opening preflight

From a source checkout:

```bash
make yam-probe
```

The default probe checks required values, exact package metadata, readable
model/calibration files, serial permissions, and whether the named network
interface exists. It does not import vendor modules, construct a device, open
a bus, parse model contents, or prove the interface bitrate.

For Docker, first create the scoped device/model override from
[Docker deployment](/docker-deployment), then run:

```bash
docker compose -f docker-compose.yml -f docker-compose.yam.yml \
  run --rm --no-deps app python -m ctrl_pi.yam_probe
```

## 5. Run one explicit connected probe

Only after the non-opening checks pass and the rig is physically safe:

```bash
PYTHONPATH=backend/src \
  .venv/bin/python -m ctrl_pi.yam_probe --connect
```

The connected probe sends no application jog or `apply_action`, but it is not
electrically read-only. The leader configures its serial bus and the follower
starts its gravity-compensation/control worker. The probe reads one safe
snapshot and always attempts bounded shutdown, including follower safe mode
and both device closes.

If vendor I/O does not return, make the hardware safe independently and treat
cleanup as failed. Do not infer electrical torque state from process output.

## 6. Validate mapping before teleoperation

With power and motion constrained, verify all of the following against the
physical pair:

- joint zero offsets, signs, radians, and wrist-flex/wrist-roll ordering;
- MuJoCo forward-kinematics direction and scale at the actual tool point;
- follower velocity/effort units and optional temperature fields;
- gripper inversion: ctrl-π exposes `0=closed` and `1=open`;
- current, torque, joint, collision, and firmware safety limits;
- emergency-stop operation and recovery;
- serial disconnect, CAN loss/bus-off, worker failure, and process shutdown;
- non-root device visibility inside the production container, if used.

Then issue one-at-a-time low-speed joint and gripper jogs from **Arms** before
starting teleoperation. Real Cartesian jog is not supported by the pinned
joint-space plugin.

## Fail-closed behavior

A configuration, import, calibration, connection, sample, worker, or command
failure leaves the API available but marks both stable arm IDs disconnected.
Motion calls fail before vendor mutation, and recovery requires a backend
restart. The application never replaces a failed real driver with
`MockYAMDriver`.

For units, limits, threading, and the complete operator checklist, continue to
[YAM driver interface](/yam-driver).
