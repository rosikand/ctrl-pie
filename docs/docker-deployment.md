# Docker deployment

The Docker option runs ctrl-pi as one non-root service. A Node build stage
compiles the Vite application, then the final Python image serves those static
files and the FastAPI control plane from one Uvicorn worker. HTTP, MJPEG, byte
ranges, and WebSockets therefore stay same-origin without a second runtime or
proxy process. The one worker owns the robot rig, recordings, and inference
teardown lifecycle.

PostgreSQL, Hugging Face, and Modal remain services in your accounts. Compose
does not install a second database, store credentials in an image, mount the
Docker socket, or run either service as privileged.

Use Docker Engine with the Compose plugin and BuildKit/Buildx 0.17 or newer.
The SocketCAN override below also requires Compose 2.24 or newer for its
`!reset` tag. Check the installed tooling with `docker compose version` and
`docker buildx version`. If Compose is paired with an older Buildx plugin but
a current standalone Buildx is installed, build the same production target
directly, then start without rebuilding:

```bash
docker buildx build --load --target runtime -t ctrl-pi:local .
docker compose up -d --wait --no-build
```

## Start in mock mode

Copy the example environment and replace its placeholders before starting:

```bash
cp .env.example .env
$EDITOR .env
docker compose build
docker compose up -d --wait
```

Open <http://127.0.0.1:8000>. `CTRL_PI_PORT` changes the host port and
`CTRL_PI_BIND_ADDRESS` changes its bind address:

```bash
CTRL_PI_BIND_ADDRESS=0.0.0.0 CTRL_PI_PORT=8000 docker compose up -d
```

ctrl-pi has no V1 authentication. Bind to a LAN address only on a trusted,
firewalled network; never expose it directly to the public Internet.

`DATABASE_URL` must name a PostgreSQL server reachable from the container.
Inside a container, `localhost` means the container itself, not the Ubuntu
host. Use the database's LAN/DNS address or a hosted PostgreSQL URL. If the
variable is omitted, the backend starts so Settings can show first-run status;
when it is present, the entrypoint runs `alembic upgrade head` before Uvicorn.

Useful checks are:

```bash
docker compose ps
docker compose logs -f app
curl --fail http://127.0.0.1:8000/api/health
docker compose exec app python -m ctrl_pi.smoke --fake-hub
```

The smoke command stays offline: it records a five-second mock episode,
packages it through LeRobot, runs 100 stub-policy actions, and verifies compute
teardown.

## Data and credentials

The named `ctrl-pi-data` volume is mounted at `/var/lib/ctrl-pi`. It holds
pre-upload recording artifacts, temporary LeRobot conversion output, and
download caches. PostgreSQL and Hugging Face remain the durable sources of
truth. A normal shutdown or `docker compose down` preserves the volume;
`docker compose down --volumes` permanently removes this local staging data.

The service runs as UID `10001`. If you replace the named volume with a bind
mount, make that directory writable by UID 10001 before starting. The image
contains FFmpeg with libx264 for MP4 finalization and CPU-only PyTorch; Node,
npm, compilers, and frontend source are absent from the runtime image.

Compose never bakes secrets into the image. By default they enter only through
the explicit environment keys loaded from `.env`; the frontend receives no HF
or Modal credential variables. Modal API credentials (`MODAL_TOKEN_ID` and
`MODAL_TOKEN_SECRET`) are distinct from endpoint proxy tokens
(`MODAL_PROXY_TOKEN_ID` and `MODAL_PROXY_TOKEN_SECRET`). To use Modal's local
profile instead of environment credentials, add a read-only bind mount for
the host file:

```yaml
services:
  app:
    volumes:
      - ctrl-pi-data:/var/lib/ctrl-pi
      - ${HOME}/.modal.toml:/home/ctrl-pi/.modal.toml:ro
```

## Graceful shutdown

Use Compose so the backend receives `SIGTERM` and its 90-second grace period:

```bash
docker compose stop
docker compose down
```

The one Uvicorn worker joins robot actions, finalizes or aborts FFmpeg safely,
verifies provider teardown, stops the YAM sampling loop, and then asks the
follower to enter its safe mode before closing both hardware connections. Do
not scale the app service: live rig leases and session state are deliberately
process-local. If Modal cleanup was interrupted, run:

```bash
docker compose run --rm app python -m ctrl_pi.modal_panic
```

## Real YAM device and model passthrough

The supported V1 hardware shape is one standard YAM follower with a crank 4310
gripper on SocketCAN and one calibrated GELLO-style leader on a Dynamixel
serial bus. Linear grippers and YAM Pro/Ultra variants are not claimed. Use a
stable `/dev/serial/by-id` name for the leader and pass only that device. Do
not mount all of `/dev`, add `NET_ADMIN`, or use `privileged: true`.

```bash
export YAM_LEADER_DEVICE=/dev/serial/by-id/your-gello-leader
export YAM_LEADER_GID="$(stat -Lc '%g' "$YAM_LEADER_DEVICE")"
export YAM_CAN_INTERFACE=can0
export YAM_CAN_BITRATE=1000000
export YAM_MODEL_DIR=/absolute/path/to/yam-model-assets
export YAM_MODEL_XML=yam_crank_4310_d405.xml
export YAM_CALIBRATION_DIR=/absolute/path/to/leader-calibration
export YAM_LEADER_CALIBRATION_ID=yam-leader
```

The model directory must contain the selected XML and every relative asset it
includes. The calibration directory must already contain
`yam-leader.json` (or the configured ID plus `.json`). Mount both read-only.
Create a local `docker-compose.yam.yml` override:

```yaml
services:
  app:
    network_mode: host
    ports: !reset []
    devices:
      - ${YAM_LEADER_DEVICE:?set YAM_LEADER_DEVICE}:/dev/yam-leader:rw
    group_add:
      - ${YAM_LEADER_GID:?set YAM_LEADER_GID}
    cap_add:
      - NET_RAW
    volumes:
      - ctrl-pi-data:/var/lib/ctrl-pi
      - ${YAM_MODEL_DIR:?set YAM_MODEL_DIR}:/opt/ctrl-pi/yam-model:ro
      - ${YAM_CALIBRATION_DIR:?set YAM_CALIBRATION_DIR}:/opt/ctrl-pi/yam-calibration:ro
    environment:
      CTRL_PI_MOCK_MODE: "false"
      YAM_CAN_INTERFACE: ${YAM_CAN_INTERFACE:?set YAM_CAN_INTERFACE}
      YAM_LEADER_PORT: /dev/yam-leader
      YAM_MUJOCO_XML_PATH: /opt/ctrl-pi/yam-model/${YAM_MODEL_XML:?set YAM_MODEL_XML}
      YAM_GRIPPER_TYPE: crank_4310
      YAM_LEADER_CALIBRATION_ID: ${YAM_LEADER_CALIBRATION_ID:-yam-leader}
      YAM_LEADER_CALIBRATION_DIR: /opt/ctrl-pi/yam-calibration
```

`NET_RAW` is the only capability added back after the base service drops all
capabilities; it permits the container to open the existing CAN raw socket.
The host retains `NET_ADMIN` and owns all interface configuration.

Configure SocketCAN on the Ubuntu host before starting the container:

```bash
sudo ip link set "$YAM_CAN_INTERFACE" down
sudo ip link set "$YAM_CAN_INTERFACE" type can bitrate "$YAM_CAN_BITRATE"
sudo ip link set "$YAM_CAN_INTERFACE" up
ip -details -statistics link show "$YAM_CAN_INTERFACE"
```

Validate the merged Compose definition and the non-opening hardware preflight:

```bash
docker compose -f docker-compose.yml -f docker-compose.yam.yml config -q
docker compose -f docker-compose.yml -f docker-compose.yam.yml \
  run --rm --no-deps app python -m ctrl_pi.yam_probe
```

The default probe validates configuration, pinned packages, model/calibration
paths, serial visibility, and the named network interface without opening the
leader or follower. A connected probe is an explicit physical operation:

```bash
docker compose -f docker-compose.yml -f docker-compose.yam.yml \
  run --rm --no-deps app python -m ctrl_pi.yam_probe --connect
```

Even with `calibrate=False`, the leader plugin configures its serial bus and
the follower plugin starts a gravity-compensation/control thread. Run the
connected probe only with an operator present, a clear workspace, the arm
secured, and an independent emergency stop. It sends no jog or policy action
and always attempts bounded driver shutdown in a `finally` path, but it is not
electrically read-only. If vendor I/O does not return, make the rig safe and
terminate the container rather than treating the probe as cleanly closed.

The probe's temporary connection is closed before it exits. After it succeeds,
start the one service and verify its safe status payload. A `YAM_*` environment
bootstrap selects configuration but deliberately leaves the devices closed;
complete the acknowledged [Settings onboarding flow](setup.md#yam-first-time-setup-and-automatic-restoration)
to preflight and save it, then choose explicit Connect or opt into automatic
connection.

```bash
docker compose -f docker-compose.yml -f docker-compose.yam.yml up -d --wait
curl --fail http://127.0.0.1:8000/api/health
curl --fail http://127.0.0.1:8000/api/settings/status
curl --fail http://127.0.0.1:8000/api/yam/setup
```

Host networking makes ctrl-π available directly on host port 8000 rather than
the mapped `CTRL_PI_PORT`; firewall it carefully. Restart the container after a
serial disconnect so Docker maps the current device node. The cloud test suite
uses fake vendor objects and cannot validate motor direction/offsets, the
wrist-flex/wrist-roll mapping, gravity compensation, physical limits, bus-off
behavior, non-root access on the target kernel, or emergency-stop behavior.
Complete the operator checklist in [YAM driver interface](yam-driver.md) before
allowing teleoperation or inference to move real hardware.
