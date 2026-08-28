# Recording and teleoperation

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
- a connected arm with role `leader` and a different connected arm with role
  `follower`;
- a writable `RECORDING_STAGING_DIR` with enough free space; and
- a free process-local robot rig.

The normal mock pair is `yam-leader` and `yam-follower`. Creating a session
looks up both live arms, validates their roles and connection flags, then
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
2. Start teleoperation. The backend acquires the shared rig lease, applies one
   initial leader action before reporting success, then continuously maps the
   leader's absolute six-joint and gripper state to the follower. Status is
   `teleop`.
3. Start an episode. The backend opens exclusive partial files and FFmpeg,
   captures an initial synchronized sample, then reports `recording`.
4. Stop the episode with its success flag and optional notes. The backend
   stops accepting samples, closes FFmpeg outside the event loop, validates
   completion, atomically renames both partial files, then commits aggregate
   metadata. Teleop remains active so another episode can begin.
5. Repeat episode start/stop as needed.
6. Stop teleoperation. The rig lease is released and the stable status becomes
   `ready` once at least one episode exists.
7. Upload explicitly. Upload never starts implicitly when teleop or an episode
   stops.

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

Transient flags such as `teleop_active`, `episode_active`, current episode
index, and current duration come from the in-memory manager. They are not
durable database fields.

## Rig ownership and conflicts

Manual jog, teleop, and inference share one non-blocking `RigLease`. A second
control mode receives HTTP 409; commands are never queued behind an owner.
Only the exact lease token may release the rig.

Teleop start is rejected when another recording or inference session owns the
rig. Stop teleop is rejected while an episode is active or finalizing; stop the
episode first so no sample is silently lost. Starting another episode while
the previous FFmpeg finalizer is running is also rejected.

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
```

Capture begins with `video.partial.mp4` and `samples.partial.jsonl`. A complete
stop closes the writer and sample stream and then renames both files to their
final names. Failed starts/finalization remove partial files so they cannot be
mistaken for an uploadable episode.

Each due sample pairs one camera frame with leader observation, the absolute
action sent to the follower, and the resulting follower observation. Sample
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
staged episodes remain on local disk; ctrl-π has no automatic retention or
delete API in V1. Hugging Face is the durable artifact source after remote
verification.

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
when no corresponding in-memory task exists. An interrupted `uploading` row is
marked failed when the uploader has no active reservation, preserving its
original target for a safe retry.

An open partial episode is not presented as finalized data. If the host or
container is lost before finalization, keep the files for inspection but do
not hand-rename them into an uploadable episode.

## Operational checklist

Before a real demonstration:

- verify the selected arm roles and live CAN state;
- confirm no inference or manual control owns the rig;
- confirm recording FPS and available staging disk;
- perform a short test episode and verify video playback;
- keep the host on a trusted network because V1 has no authentication; and
- keep `.env`, Hub credentials, and absolute cache paths out of task/episode
  metadata.

After the session:

- stop and finalize every episode before stopping teleop;
- verify episode count and duration;
- upload private unless publication is intentional;
- record the returned repository and immutable revision; and
- retain local staging until remote playback and metadata are verified.

## Related guides

- [Architecture](architecture.md)
- [Setup and configuration](setup.md)
- [YAM driver interface](yam-driver.md)
- [Compute and inference](inference.md)
- [Full mock smoke gate](smoke-test.md)
