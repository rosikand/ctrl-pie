# ctrl-π V1.2 physical field-test handoff

This handoff is for the first physical acceptance of the V1.2 YAM cell path.
It is not evidence that V1.2 has been physically validated. All V1.2
implementation and release gates run in mock/fake-vendor mode; H0–H7 below
must be completed on the Ubuntu YAM workstation before making a hardware
support claim.

## Architecture being tested

- One persisted, YAM-specific `YAMCell` contains typed arm rows. Each arm has
  a stable logical ID, role, transport, durable physical identity,
  end-effector kind, and optional pair/group/side metadata.
- A SocketCAN arm persists its USB-CAN adapter serial. Passive discovery maps
  that serial to the current Linux interface. A `canN` name is runtime
  telemetry only and may change after any re-enumeration.
- The all-CAN adapter uses one supervised Python worker process per connected
  CAN arm. Only that worker constructs the i2rt YAM object and owns its nested
  control threads. Bounded local IPC carries identity handshakes, cached
  telemetry, latest-state actions, safe-idle, and shutdown. ctrl-π does not
  run the Lux shell launchers or expose a pair RPC server.
- The operator supplies an exact local i2rt checkout and full Git commit. The
  container build and ctrl-π preflight use local read-only Git inspection;
  they do not clone, fetch, resolve a branch, or substitute public upstream.
  The field-tested commit `a32cfa660151d75319751be69cfa580a1f8b344c`
  is not advertised by public upstream and must not be reconstructed by
  selecting `latest`.
- Connect is per arm and explicit. A `linear_4310` or `crank_4310` follower
  requires both the general hardware-motion acknowledgement and a separate
  acknowledgement that calibration moves the jaws. A connected follower is energized and may
  hold/resist motion until explicit disconnect completes.
- Teleop acquires one declared pair and starts observation-only with sync
  disabled. It performs no follower write. A second, freshly acknowledged
  action makes the approximately three-second correction toward the latest
  mapped leader pose; disabling sync stops writes while telemetry remains.
- Frame maps apply before soft limits. No frame-map artifact means intentional
  1:1 mapping. No soft-limit artifact means exactly `NO SASH GUARD`; ctrl-π
  does not invent or restore a bound.
- Inference accepts follower logical IDs only. Per-arm rig leases allow
  independent pairs while refusing a second writer to the same arm/bus.
- During all-CAN teleop or inference, the parent safe-idles after 125 ms
  without a refreshed target, before the child watchdog; the child refuses an
  action older than 250 ms. Motion resumes only through a fresh policy command
  boundary or fresh teleop synchronization boundary. These are command-age
  guards, not field-measured loop-rate thresholds, and one-shot jog is off.
- The retained legacy adapter supports one serial GELLO leader plus one CAN
  `crank_4310` follower through a separate Docker override.

### Worker failure and shutdown model

ctrl-π revokes new writes first, requests cooperative safe-idle and device
close, then joins the worker. It terminates, kills, and reaps after bounded
deadlines if cooperation fails. A crash, stale heartbeat/telemetry, stopped
inner loop, mismatched identity, or uncertain survivor latches the affected
arm/pair unavailable; there is no blind auto-respawn. An uncertain survivor
keeps that bus blocked until an operator makes the rig safe and reviews it.

Arms shows the observed i2rt/control-loop rate as a diagnostic. The earlier
Lux/i2rt session measured followers at roughly 263–275 Hz and leaders at
roughly 419–426 Hz with `bilateral_kp=0.0`. Those are reference observations,
not ctrl-π health thresholds.

## Software gate record

These are cloud/mock integration results only. They do not validate physical
hardware behavior; H0-H7 remain required on the field rig.

| Gate | Result |
| --- | --- |
| Frontend production build | PASS — `npm run build` |
| Full backend pytest | PASS — 791 tests, one existing Starlette deprecation warning |
| Clean PostgreSQL Alembic upgrade to head | PASS — PostgreSQL upgrade/no-op at revision `0011` |
| Backend and frontend development boot | PASS — mock Uvicorn health and Vite root answered on loopback; both were stopped afterward |
| Hardware-derived Docker image build | NOT COMPLETED in this cloud — exact-source/script preflight passed, but isolated daemon attempt 1 had no bridge and attempt 2 had no NAT, so dependency download failed and no image was produced |
| All-CAN and legacy Compose render | PASS — base, all-CAN, and legacy merged configurations plus shell syntax |
| `make smoke` | PASS — four arms/two pairs, 3.117 s explicit sync, 5.000 s/50 samples, real private Hub upload/list, 100 stub actions, verified compute/rig/upload cleanup and Hub repo deletion |
| Final branch / implementation commit | `v1.2-yam-cell` / milestone 17 commit recorded in the evidence follow-up |

No row in this table may be described as physical validation.

## Known risks and unvalidated assumptions

- Whether non-root UID 10001 can inspect the required read-only sysfs paths
  and open existing SocketCAN raw sockets with only `NET_RAW` is an H1 test on
  the reference host. The container has no `NET_ADMIN`, cannot configure or
  bounce links, and is not privileged.
- The exact mounted i2rt checkout, its dependencies in the derived image, and
  safe-idle/close behavior have not run in this cloud workspace against a YAM.
- Motor directions, offsets, gravity compensation, joint/gripper units,
  calibration motion, physical limits, bus-off behavior, E-stop response,
  and bounded shutdown remain physical acceptance items.
- The reference cell has no current soft-limit files. It has no sash guard and
  must remain out of in-hood operation until limits are measured for the
  current mount.
- Teaching-handle range checking is detection-only. It actively opens a
  read-only CAN diagnostic after acknowledgement, but never zeros an encoder.
  Re-zero remains an external i2rt CLI maintenance operation with the trigger
  mechanically released.
- Bimanual group identity and selection are present, but true multi-follower
  policy execution and bimanual episode recording are deferred.
- The camera is synthetic. H6 validates real robot state/action capture and
  the dataset pipeline, not training-data readiness.
- The approximately 15-degree H4 sanity check is a one-session operator
  heuristic, not a product limit and must not be encoded in software.

## Prepare the exact local i2rt source

On the Ubuntu box, point these variables at the operator-controlled checkout.
If any check fails, stop. Do not fetch, reset, clean, or switch the checkout
as part of the field procedure; obtain the reviewed exact source from its
owner.

```bash
export CTRL_PI_I2RT_CHECKOUT=/home/andre/i2rt
export CTRL_PI_I2RT_DEPENDENCY_COMMIT=a32cfa660151d75319751be69cfa580a1f8b344c

test "$(git -C "$CTRL_PI_I2RT_CHECKOUT" rev-parse --show-toplevel)" = \
  "$(realpath "$CTRL_PI_I2RT_CHECKOUT")"
test "$(git -C "$CTRL_PI_I2RT_CHECKOUT" rev-parse --verify 'HEAD^{commit}')" = \
  "$CTRL_PI_I2RT_DEPENDENCY_COMMIT"
test -z "$(git -C "$CTRL_PI_I2RT_CHECKOUT" status \
  --porcelain=v1 --ignored --untracked-files=all)"
```

The status check deliberately rejects modified, staged, untracked, **and
ignored** content. An ignored Python file can still shadow code from the
pinned commit. Do not clean it automatically; stop and obtain a reviewed
checkout from its owner.

Use internal port 8010 because the reference host already has a service on
8000:

```bash
export CTRL_PI_LISTEN_PORT=8010
export CTRL_PI_PORT=8010
```

The all-CAN service is rendered and built with:

```bash
docker compose -f docker-compose.yml -f docker-compose.yam-cell.yml config -q
make docker-yam-cell
```

The build consumes only the named local checkout context. Runtime mounts the
same checkout at `/opt/i2rt:ro`; enter `/opt/i2rt` and the exact expected
commit in Settings when defining the cell. The image retains that build
identity as `CTRL_PI_I2RT_DEPENDENCY_COMMIT`; preflight rejects a cell whose
configured/mounted commit differs. Rebuild the hardware image whenever the
reviewed checkout commit changes. The checked-in pip build constraint carries
i2rt's `scikit-build-core<0.10` requirement into the isolated `ruckig` build;
the verified Make target is the required hardware-image workflow.

The runtime read-only gate accepts only the kernel `ST_RDONLY` mount flag seen
inside the container. Host `chmod`, ownership, and effective-permission scans
are not proof. The Compose `/opt/i2rt:ro` bind is intended to supply the flag;
a missing flag is a hard preflight stop, not a reason to relax the verifier.

## H0–H7 physical acceptance

Stop immediately on unexpected motion, direction, speed, sound, controller
state, or cleanup. Keep independent power/E-stop access available for every
motion-capable gate.

### H0 — software only

1. Pull the reviewed `v1.2-yam-cell` commit recorded above; do not use an
   uncommitted field-box edit.
2. Set `CTRL_PI_MOCK_MODE=true`, `CTRL_PI_BIND_ADDRESS=127.0.0.1`,
   `CTRL_PI_PORT=8010`, and `CTRL_PI_LISTEN_PORT=8010`.
3. Build and start the ordinary production image:

   ```bash
   docker compose build
   docker compose up -d --wait
   curl --fail http://127.0.0.1:8010/api/health
   ```

4. Run `docker compose exec app python -m ctrl_pi.smoke` and require its final
   `[smoke] PASS` teardown line.
5. In Settings and Arms, verify the deterministic four-arm mock cell, two
   independent pairs, stable fake identities, handle health, and follower
   selection. Verify Postgres/HF/Modal status as needed for H6/H7.
6. Stop the mock service with `docker compose down`. Do not delete the data
   volume.

### H1 — passive real-cell discovery

1. With the hardware service still stopped, use the ordinary image for a
   read-only audit of both the V1.2 cell row and retained V1.1 row:

   ```bash
   docker compose run --rm --no-deps -e CTRL_PI_MOCK_MODE=true app python -c '
   import os
   import psycopg
   with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
       connection.execute("SET TRANSACTION READ ONLY")
       rows = connection.execute("""
           SELECT '"'"'yam_cells'"'"' AS source, id, auto_restore FROM yam_cells
           UNION ALL
           SELECT '"'"'yam_setups'"'"' AS source, id, auto_restore FROM yam_setups
           ORDER BY source, id
       """).fetchall()
   print("\n".join(f"{source}:{row_id} auto_restore={str(enabled).lower()}" for source, row_id, enabled in rows) or "no persisted YAM rows")
   if any(enabled for _, _, enabled in rows):
       raise SystemExit("BLOCKED: revoke every saved YAM auto_restore consent before hardware startup")
   '
   ```

   This command must exit zero and show only `false` (or no rows). If it is
   blocked, explicitly revoke both stored consents while still using the
   non-hardware one-shot container:

   ```bash
   docker compose run --rm --no-deps -e CTRL_PI_MOCK_MODE=true app python -c '
   import os
   import psycopg
   with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
       cells = connection.execute("UPDATE yam_cells SET auto_restore = false, updated_at = now() WHERE auto_restore IS TRUE RETURNING id").fetchall()
       legacy = connection.execute("UPDATE yam_setups SET auto_restore = false, updated_at = now() WHERE auto_restore IS TRUE RETURNING id").fetchall()
   print(f"revoked yam_cells={len(cells)} yam_setups={len(legacy)}")
   '
   ```

   Both commands replace the service command with a bounded database script,
   force mock mode, and never start FastAPI or construct a hardware driver.
   Rerun the read-only audit and require only `false` values or no rows before
   continuing. Do not start the hardware application on a failed or ambiguous
   result.
2. Confirm the host owns CAN configuration and that the intended links are
   already present. Do not run `ip link set`, a bringup script, or motor ping.
3. Verify the exact, clean local i2rt checkout with the commands above. Then
   prove the Compose bind reports the kernel read-only mount flag without
   starting FastAPI or a hardware driver:

   ```bash
   docker compose -f docker-compose.yml -f docker-compose.yam-cell.yml \
     run --rm --no-deps --entrypoint python app -c '
   import os
   flags = os.statvfs("/opt/i2rt").f_flag
   if not flags & os.ST_RDONLY:
       raise SystemExit("BLOCKED: /opt/i2rt lacks the kernel ST_RDONLY mount flag")
   print("/opt/i2rt kernel read-only mount: true")
   '
   ```

   Require the final `true` line. Host chmod or an effective-permission check
   is not acceptable proof; stop if this command fails.
4. Set `CTRL_PI_MOCK_MODE=false` and start the all-CAN override:

   ```bash
   docker compose -f docker-compose.yml -f docker-compose.yam-cell.yml \
     up -d --wait --no-build
   curl --fail http://127.0.0.1:8010/api/health
   ```

5. Open Settings → YAM Cell. Discover only; do not Connect.
6. Confirm all four fixture serials are present and assigned to the intended
   logical roles. The original session mapped:

   ```text
   2080377E45465008             right leader
   2058377F45465008             right follower
   004A001633354B0432333837     left leader
   208B336E594E5018             left follower
   ```

   Treat the displayed `canN` values as session-local; they may differ.
7. Confirm pair/group/side metadata, distinct pair ports, intentional identity
   frame maps, and the prominent `NO SASH GUARD` warning.
8. Confirm no arm moves, no jaw calibration runs, no motor is pinged/enabled,
   and no CAN link changes state. Inspect container capabilities and mounts:

   ```bash
   docker compose -f docker-compose.yml -f docker-compose.yam-cell.yml \
     exec app sh -c 'id; grep -E "(/sys|/opt/i2rt)" /proc/mounts'
   ```

9. Record whether UID 10001 plus `NET_RAW`, `/sys:ro`, and host networking are
   sufficient. Any requested `NET_ADMIN`, broad `/dev`, or privileged access
   is a blocker requiring a code/review round trip.

### H2 — passive leader health

1. Keep both followers disconnected.
2. Select the right CAN teaching-handle leader and start the explicit range
   check with the active-CAN diagnostic acknowledgement.
3. Move only the trigger released → fully squeezed → released. Confirm current
   value, button state, reachability, and a non-stuck range. Earlier evidence
   was approximately `0.000 .. 1.000`.
4. Repeat for the left leader. Earlier post-repair evidence was approximately
   `0.007 .. 0.973`.
5. Do not zero either handle from ctrl-π. A pinned constant such as
   `1.000 .. 1.000` is unhealthy even when the adapter is connected; stop and
   use the reviewed external encoder-manager maintenance procedure only after
   separate operator approval.

### H3 — one follower connect

Use the right follower first.

1. Secure/support the arm, clear its entire swept volume, empty the jaws, and
   place an operator at independent power/E-stop controls.
2. Select only the right follower. Read and accept both warnings: controller
   connection can energize/hold the arm, and configured 4310 gripper
   calibration (`linear_4310` here, or `crank_4310`) moves the jaws.
3. Submit both acknowledgements and Connect once.
4. Observe calibration without reaching into the workspace. Verify six joint
   values, normalized gripper, connected/energized/holding state, and live
   loop diagnostics. The prior i2rt follower rate of roughly 265–272 Hz is a
   reference observation only.
5. Explicitly Disconnect that follower. Wait for worker teardown and verify
   disconnected, not energized/holding, and safely limp as designed before
   touching or manually repositioning it. Uncertain cleanup is a blocker.

### H4 — right pair teleop

1. Reconnect the right follower with both acknowledgements, then connect only
   its declared right leader.
2. Create/select the right recording pair and Start Teleop. Sync must show
   disabled. Confirm live leader/follower telemetry and per-joint deltas.
   Confirm zero follower command/write occurred at teleop start.
3. Manually compare the displayed deltas. Earlier carefully homed pairA had a
   worst residual near 8.7 degrees, with most joints below 3 degrees.
4. With the full follower workspace clear, freshly acknowledge and Enable
   Sync. Expect only the slow approximately three-second correction covering
   roughly the deltas just inspected.
5. For this field session, treat approximately 15 degrees as a conservative
   operator sanity threshold only. Any unexpectedly large, fast, or
   wrong-direction correction means disable sync immediately and investigate
   zeroing/frame mapping. Do not wait to see where unexpected motion goes.
6. Make one tiny leader motion and verify the right follower tracks the right
   direction and correct pair only.
7. Disable Sync. Verify follower writes stop while live pair telemetry remains.
8. Stop Teleop, disconnect leader then follower, wait for clean worker reaping,
   and verify the follower is no longer holding. Never force a live follower
   by hand.

### H5 — left pair

Repeat every H4 step with the declared left pair. Confirm the right follower
does not move and cross-pair selection/routing is rejected. Earlier pairB
homing evidence had a worst residual near 7.1 degrees. Apply the same
session-only 15-degree sanity heuristic and the same stop-on-surprise rule.

### H6 — recording

1. Select only the right declared pair and start teleop observation-only.
2. Inspect deltas, explicitly enable sync, and verify stable tiny tracking
   before recording.
3. Record a short open-table episode, then stop/finalize and disable sync.
4. Upload to a new private repository in the configured HF namespace.
5. Re-open the immutable revision in Datasets. Verify six arm joints plus one
   normalized gripper for state/action, correct right-pair metadata, expected
   sample timing, and no left-pair samples.
6. Disconnect both arms cleanly.

The H6 camera remains synthetic. Passing this gate is not permission to
collect production training data. Real camera support, intervention masks,
policy-version tags, and graded outcomes are follow-on work outside V1.2.

### H7 — single-follower inference

Proceed only after H3–H6 establish action normalization and cleanup.

1. Use a compatible controlled policy/test policy with deliberately small,
   reviewed actions. Do not begin with an arbitrary high-amplitude model.
2. Deploy the exact immutable revision and verify its runtime identity before
   starting robot execution.
3. Select only the right follower logical ID; selecting a leader must fail.
4. Clear the workspace, connect the follower with both acknowledgements, then
   explicitly start inference and verify one small safe action path.
5. Stop execution. Verify local writes and queued actions stop, the arm lease
   is released, and the follower safe-idles into gravity compensation. The
   all-CAN worker remains connected and the follower remains energized/holding;
   do not try to reposition it by hand.
6. In **Settings → YAM Cell**, explicitly Disconnect the follower. Verify the
   worker is reaped and the follower is de-energized/limp before calling local
   hardware cleanup complete.
7. Require provider-verified Modal teardown with zero
   running tasks. If teardown is uncertain, keep reconciling or use the
   documented exact ownership-safe panic procedure; do not report success.

## Operations that may energize or move hardware

- Connecting a CAN leader opens its i2rt YAM object and may energize its
  controller even though follower synchronization is still disabled.
- Connecting a follower enables its controller/gravity-compensation path;
  initialization of a `linear_4310` or `crank_4310` gripper calibrates and
  physically moves the jaws.
- Enabling teleop sync performs the explicit approximately three-second
  follower correction, then permits tracking writes until sync is disabled.
- Inference writes follower targets. The supervised all-CAN adapter disables
  one-shot jog in V1.2; do not use Arms/SDK jog as a physical validation path.
  Bounded jog remains available in mock mode and the retained legacy adapter.
- Inference Stop clears writes/queues, safe-idles to gravity compensation, and
  releases its lease, but leaves an all-CAN follower connected and holding.
  Explicit Settings Disconnect is the worker-reap and de-energize/limp
  boundary.
- Disconnect/shutdown changes torque/control state while making the arm limp;
  the arm must be supported.

Passive discovery/preflight does none of those operations. The teaching-handle
range test opens only the acknowledged, read-only encoder diagnostic and does
not zero an encoder or command a motor.
