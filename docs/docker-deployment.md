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
and verifies provider teardown during application shutdown. Do not scale the
app service: live rig leases and session state are deliberately
process-local. If Modal cleanup was interrupted, run:

```bash
docker compose run --rm app python -m ctrl_pi.modal_panic
```

## USB serial passthrough

Use stable `/dev/serial/by-id` names and pass only the required device. Do not
mount all of `/dev` and do not use `privileged: true`.

```bash
export YAM_USB_DEVICE=/dev/serial/by-id/your-yam-adapter
export YAM_USB_GID="$(stat -Lc '%g' "$YAM_USB_DEVICE")"
```

Create a local `docker-compose.usb.yml` override:

```yaml
services:
  app:
    devices:
      - ${YAM_USB_DEVICE:?set YAM_USB_DEVICE}:/dev/yam:rw
    group_add:
      - ${YAM_USB_GID:?set YAM_USB_GID}
```

Then start with both files and verify the non-root process can reach it:

```bash
docker compose -f docker-compose.yml -f docker-compose.usb.yml up -d --wait
docker compose -f docker-compose.yml -f docker-compose.usb.yml \
  exec app sh -c 'test -r /dev/yam && test -w /dev/yam'
```

Repeat the mapping with a distinct container path for each adapter when the
leader and follower use separate USB devices. Restart the container after a
USB disconnect/reconnect so Docker can map the current device node again.

## SocketCAN passthrough

A SocketCAN interface is a network interface, not a `/dev` node. Configure it
on the Ubuntu host using the interface name and bitrate from the YAM hardware
documentation; ctrl-pi does not guess either value:

```bash
export YAM_CAN_INTERFACE=can0
export YAM_CAN_BITRATE=REPLACE_WITH_HARDWARE_BITRATE
sudo ip link set "$YAM_CAN_INTERFACE" down
sudo ip link set "$YAM_CAN_INTERFACE" type can bitrate "$YAM_CAN_BITRATE"
sudo ip link set "$YAM_CAN_INTERFACE" up
ip -details -statistics link show "$YAM_CAN_INTERFACE"
```

To share that interface, create `docker-compose.socketcan.yml`:

```yaml
services:
  app:
    network_mode: host
    ports: !reset []
    environment:
      YAM_CAN_INTERFACE: ${YAM_CAN_INTERFACE:?set YAM_CAN_INTERFACE}
```

Start with the override, then confirm interface visibility without granting
the container `NET_ADMIN`:

```bash
docker compose -f docker-compose.yml -f docker-compose.socketcan.yml up -d --wait
docker compose -f docker-compose.yml -f docker-compose.socketcan.yml \
  exec app python -c \
  'import os,socket; name=os.environ["YAM_CAN_INTERFACE"]; print(socket.if_nametoindex(name))'
```

Add `YAM_CAN_INTERFACE` to `.env` so the probe and, in Milestone 14, the real
driver receive the same explicit name. In this mode ctrl-pi is available on
host port 8000 rather than the mapped `CTRL_PI_PORT`. Host networking exposes
that port directly through the host network namespace; firewall it carefully.
The host owns CAN setup, so the container keeps all Linux capabilities
dropped.

## Milestone boundary

Milestone 12 proves isolated packaging and narrowly scoped device visibility.
The current application still constructs `MockYAMDriver`; setting
`CTRL_PI_MOCK_MODE=false` selects real external compute behavior but does not
silently substitute an unfinished hardware driver. Real `YAMDriver` selection,
protocol configuration, and hardware validation land in Milestone 14.
