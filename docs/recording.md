---
title: "Recording and teleoperation"
description: "Teleoperate one YAM pair, capture durable episodes, and upload LeRobot datasets."
icon: "video"
---

The Record tab operates one leader/follower pair and collects demonstrations
as synchronized camera, observation, and action samples. The active lifecycle
is process-local; stable session metadata is in PostgreSQL; finalized artifact
bytes stay in local staging until explicitly uploaded as a LeRobot dataset.

Mock mode exercises this exact flow with `MockYAMDriver`, a synthetic camera,
real FFmpeg MP4 encoding, and LeRobot 0.4.4 conversion. The driver boundary is
described in [YAM driver interface](yam-driver.md).

## Before recording

Recording requires:

- a migrated PostgreSQL database;
- FFmpeg with the `libx264` encoder;
- one exact declared pair containing a connected `leader` and connected
  `follower`;
- a writable `RECORDING_STAGING_DIR` with enough free space; and
- free resource leases for those two logical arms.

The reference mock pairs are `yam-leader` / `yam-follower` and
`yam-leader-left` / `yam-follower-left`. Creating a session looks up both live
arms, rejects cross-pair routing, validates roles/connections, then
upserts their stable `driver_id` mapping into PostgreSQL. The recording row
stores robot UUID foreign keys while the API continues to show driver IDs.

Recording FPS is 1–60. The value saved on the Settings page takes precedence;
`RECORDING_FPS` is the fallback when no database setting exists. Teleoperation
runs at its own 50 Hz control cadence, while video/sample capture follows the
selected recording FPS.

## Operator lifecycle

Use the controls in this order:

1. Select the leader and follower, enter a bounded name and task, and create
   the session. Its stable status is `draft`.
2. Start teleoperation. The backend acquires leases for exactly that declared
   pair and starts observation-only. Synchronization is disabled and no
   follower write occurs. Inspect the live per-joint deltas. Status is
   `teleop`.
3. Clear the follower workspace, then separately **Enable Sync** with a fresh
   slow-motion acknowledgement. The backend maps the latest leader state,
   applies follower frame/limit guards, and interpolates the follower toward
   it over approximately three seconds before live latest-state tracking.
4. Start an episode only after synchronization completes. The backend opens
   exclusive partial files and FFmpeg,
   captures an initial synchronized sample, then reports `recording`.
5. Stop the episode with its success flag and optional notes. The backend
   stops accepting samples, closes FFmpeg outside the event loop, validates
   completion, atomically renames both partial files, publishes a durable
   bounded manifest, then commits aggregate metadata. Teleop remains active so
   another episode can begin.
6. Repeat episode start/stop as needed.
7. **Disable Sync** to stop follower writes while keeping pair telemetry, then
   stop teleoperation. The backend safe-idles before releasing the pair lease;
   stable status becomes
   `ready` once at least one episode exists.
8. Upload explicitly. Upload never starts implicitly when teleop or an episode
   stops.

<Warning>
  Teleop start is not a motion action. If the follower moves or receives a
  write before the separate sync acknowledgement, stop and treat it as a
  blocker. A connected follower may still be energized/holding independently
  of synchronization; never force it by hand.
</Warning>

An episode may include only optional trimmed `operator` and `notes` metadata.
The stop request accepts `success` and final notes. Recording creation reserves
the server-owned `episodes` and `upload` metadata keys.

The public statuses are:

| Status | Meaning |
| --- | --- |
| `draft` | Session exists but has no finalized episode and no live teleop. |
| `teleop` | Leader-to-follower control owns the rig; no episode is capturing. |
| `recording` | An episode is capturing or completing its atomic finalization. |
| `ready` | One or more finalized episodes are available for another teleop session or upload. |
| `uploading` | LeRobot conversion/Hub transfer owns this immutable session. |
| `uploaded` | Hub commit and required remote files were verified. |
| `failed` | Capture, persistence, teleop, or upload stopped safely; correct the safely reported cause and retry only where allowed. |

Transient fields such as `teleop_active`, `sync_enabled`, `sync_in_progress`,
`joint_deltas_radians`, `episode_active`, current episode index, and current
duration come from the in-memory manager. They are not durable database fields.

## Rig ownership and conflicts

Teleop owns the selected leader/follower resource set. Inference owns one
selected follower; mock/legacy jog can own one follower. Disjoint pairs can
operate independently; an
overlapping owner receives HTTP 409 and commands are never queued. Only the
exact lease token may release those resources.

Teleop start is rejected for an undeclared/cross pair or overlapping lease.
Episode start is rejected until slow sync completes. Stop teleop is rejected
while an episode is active/finalizing; stop the episode first. Starting another
episode while the previous FFmpeg finalizer is running is also rejected.

If a driver or teleop step fails, the task records a safe failure and releases
its lease. Episode finalization does not restore a stale `teleop` state when
the control loop has already failed. Application shutdown joins teleop tasks,
settles any non-cancellable FFmpeg finalizer, aborts open partial episodes, and
releases every owned lease.

## Staged episode format

Each episode has a directory below the configured root:

```text
RECORDING_STAGING_DIR/
  <recording UUID>/
    episode_000000/
      video.mp4
      samples.jsonl
      episode.json
```

Capture begins with `video.partial.mp4` and `samples.partial.jsonl`. A complete
stop closes the writer and sample stream, renames both files to their final
names, and atomically publishes `episode.json` only after both artifacts are
durable. Failed starts and finalization remove partial or mixed files so they
cannot be mistaken for an uploadable episode.

Each due sample pairs one camera frame with leader observation, the post-map
and post-limit absolute action actually routed to the follower, and the
resulting follower observation. Sample
indexes and timestamps are monotonic. The final episode metadata records its
relative artifact key, sample count, FPS, duration, success flag, notes, and
UTC start/end timestamps. Absolute host paths are never persisted or returned.

PostgreSQL stores only the episode aggregate metadata, total count, total
duration, and relative keys. It does not store MP4 or JSONL bytes. Preserve the
staging directory through a successful Hub upload; a database backup alone
cannot recover unuploaded data.

The production Docker named volume keeps staging across ordinary restarts and
`docker compose down`. Removing that volume permanently removes local
unuploaded artifacts. See [Docker deployment](docker-deployment.md).

## LeRobot conversion and Hub upload

Upload accepts a repository slug and a visibility flag. The backend always
prepends the environment's exact `HF_NAMESPACE`; the browser cannot select
another owner or provide a token. Visibility defaults to private.

Before any Hub mutation, conversion validates all staged paths remain below
the configured root and are not symlink escapes. It validates the JSONL frame
indexes/timestamps, joint order, finite vectors, FPS, image dimensions, and an
exact decoded MP4-frame/sample count. Mixed-FPS or misaligned episodes are
rejected instead of truncated.

LeRobot 0.4.4 creates the canonical v3 parquet, video, and metadata layout.
The backend reopens the finalized dataset and verifies it before transfer. It
uses an explicitly token-bound Hub client to create the target, defaults it to
private, uploads the folder, and verifies the returned commit and required
remote files before marking the recording `uploaded`.

Every ctrl-π dataset contains `.ctrl-pi.json` with its recording UUID. On a
first attempt, an existing repository is a conflict and is never claimed. If
an interrupted attempt created the repository, a retry may continue only when
the persisted server-owned intent and the remote marker match the same
recording. A markerless repository may be bootstrapped only while it is empty
apart from Hub's `.gitattributes`; an unrelated or mismatched repository is
left untouched.

Upload is one synchronous request backed by a worker thread—there is no queue
or progress service. Its in-process reservation remains held until terminal
database state is committed, even if the HTTP caller disconnects. After a
partial remote mutation, the session is immutable and retries may use only the
same repository target. Once uploaded, no new episodes can be added.

The temporary converted LeRobot copy is removed after the attempt. Original
staged episodes are retained on failure, cancellation, or an uncertain final
database commit. They are removed only after the Hub revision is verified and
the `uploaded` database transition commits. If that cleanup is interrupted,
startup removes the retained raw staging after observing the durable
`uploaded` row. Hugging Face is then the artifact source of truth.

ctrl-π does not choose or grant a license for captured user data. Generated
dataset cards leave the Hub license field unset; an operator who publishes a
dataset must select and document the appropriate data license.

## Inference recording

The Inference tab can passively record a policy session through the same
pipeline. It does not start teleop or acquire a second lease. Before the first
policy action is scheduled, recording verifies that the matching inference
session owns the rig and opens the episode. At each configured capture time it
pairs the follower observation, actual validated `ArmAction`, applied follower
snapshot, and camera frame.

Inference Stop first prevents and joins further arm writes, then finalizes and
persists the recording episode, and finally verifies provider teardown. A
successful stop returns the recording UUID and `ready` recording state. Upload
that recording later with the normal Record workflow; provider teardown never
waits on a Hub upload.

## Recovery behavior

Live teleop and capture never resume after a process restart. A persisted
`teleop` or `recording` badge is normalized to `draft`, `ready`, or `failed`
when no corresponding in-memory task exists. Valid episode manifests are
reconciled idempotently into PostgreSQL, including a manifest published before
a failed episode-metadata commit. Incomplete partial, mixed, or invalid episode
directories are removed. Valid manifests unknown to the connected database
and unrelated shared-root entries are preserved, so a fresh or wrong database
does not destroy finalized raw data. An interrupted `uploading` row is marked
failed when the uploader has no active reservation, preserving its original
target for a safe retry.

## Operational checklist

Before a real demonstration:

- verify exact pair/group/side metadata, stable identities, handle health, and
  live CAN state;
- inspect live joint deltas before the separately acknowledged slow sync;
- heed `NO SASH GUARD` and keep an unguarded cell out of the hood;
- confirm no inference or manual control owns the rig;
- confirm recording FPS and available staging disk;
- perform a short test episode and verify video playback;
- keep the host on a trusted network because V1 has no authentication; and
- keep `.env`, Hub credentials, and absolute cache paths out of task/episode
  metadata.

After the session:

- stop/finalize every episode, disable sync, then stop teleop;
- verify episode count and duration;
- upload private unless publication is intentional;
- record the returned repository and immutable revision; and
- do not manually remove local staging before the API reports the durable
  `uploaded` result; ctrl-π removes it after that commit.

## Related guides

- [Architecture](architecture.md)
- [Setup and configuration](setup.md)
- [YAM driver interface](yam-driver.md)
- [Compute and inference](inference.md)
- [Full mock smoke gate](smoke-test.md)
