---
title: "Full mock smoke gate"
description: "Run the deterministic end-to-end recording, Hub, inference, and teardown gate."
icon: "flame"
---

`make smoke` is the recurring Milestone 11+ acceptance gate. Robot arms,
camera source, policy runtime, and compute target stay in deterministic mock
mode. The default command deliberately crosses one real external boundary: it
creates, reads, lists, and safely deletes a private dataset in the configured
Hugging Face namespace.

It is not a real Modal test. The retained real-GPU proof is documented under
[Compute targets and inference](inference.md#retained-milestone-11-live-evidence).

## Prerequisites

Run from the repository root with the project virtual environment installed.
FFmpeg with an H.264 encoder and the pinned LeRobot dependencies must be
available. The default gate also requires:

- `HF_NAMESPACE` set to the exact user or organization that should own the
  temporary dataset; and
- backend-only `HF_TOKEN` access sufficient to create, read, and delete a
  private dataset in that namespace.

The gate does not need Modal credentials, a GPU, YAM hardware, the frontend,
or the application's configured PostgreSQL database. Its inference lifecycle
uses a temporary local database and Stub compute.

## Run the release gate

```bash
make smoke
```

The command performs these operations in order:

1. create a temporary workspace and one `MockYAMDriver` with four reference mock arms,
   `MockCamera`, resource-scoped `RigLease`, and `RecordingManager`; verify
   stable identities and independent declared pairs, including
   `yam-leader-left` and `yam-follower-left`;
2. start the right mock pair observation-only and assert zero follower writes,
   then perform explicit slow synchronization before recording at least five wall-clock
   seconds from `yam-leader` to `yam-follower` at 10 fps; disable sync and
   finalize a real non-empty H.264 MP4 with synchronized samples;
3. convert the episode through LeRobot 0.4.4 into a LeRobot v3 dataset;
4. create a unique `HF_NAMESPACE/ctrl-pi-smoke-<timestamp>-<nonce>` private
   dataset and upload it with ctrl-π's ownership marker;
5. read the exact returned 40-character commit SHA, verify the repository is
   private, and list the same repo/SHA/episode/frame count through the normal
   Datasets service;
6. deploy the deterministic Stub CPU policy, select only one logical mock
   follower, and execute exactly 100 successful arm writes without touching
   the other pair; and
7. stop the robot loop and compute target, then prove that no task, queued
   action, lease, upload, repository, or temporary workspace remains.

Successful output contains one evidence line for each phase:

```text
[smoke] topology: PASS arms=4 pairs=2
[smoke] record: PASS ...
[smoke] upload: PASS mode=real private=true ...
[smoke] datasets: PASS ...
[smoke] deploy: PASS target=stub runtime=stub status=running
[smoke] inference: PASS steps=100 arm=yam-follower
[smoke] teardown: PASS recording=0 inference=0 compute=0 rig=free upload=idle hub_repo=deleted workspace=removed
[smoke] PASS
```

Dynamic duration, byte, frame, repository, and revision values are printed as
evidence. The token itself is never printed.

## Ownership-safe Hub cleanup

The temporary repository is not deleted merely because its generated name
looks familiar. Cleanup:

1. reads a valid immutable current revision;
2. requires that it still equals the revision uploaded by this run;
3. verifies the remote `.ctrl-pi.json` marker belongs to the exact recording;
4. repeats the revision check immediately before deletion; and
5. deletes only that exact dataset repo, then verifies absence.

If ownership or revision changes, cleanup refuses deletion and the command
exits nonzero with the repository ID for manual inspection. This may
intentionally leave the repository recoverable rather than risk deleting data
the gate no longer owns.

All remaining cleanup runs from `finally` paths after a normal failure or
interrupt. An incomplete recording shutdown, held rig, active uploader,
retained inference task/queue, active Stub resource, refused Hub deletion, or
remaining temporary workspace makes the gate fail. A passing last line is
therefore the teardown proof, not just a statement that the 100th action ran.

## Offline development and tests

Tests and offline development must opt into the deterministic filesystem Hub:

```bash
PYTHONPATH=backend/src .venv/bin/python -m ctrl_pi.smoke --fake-hub
```

`--fake-hub` replaces only Hugging Face transport. It still waits five real
seconds, captures/finalizes MP4, converts with LeRobot, exercises immutable
listing, executes exactly 100 actions, and proves cleanup. The fake repository
exists only below the temporary workspace and makes no network call.

The actual `make smoke` target does not pass `--fake-hub`; do not substitute
the offline command when recording a release-gate result. Pytest covers the
fake boundary and failure cleanup without real credentials:

```bash
.venv/bin/pytest -q backend/tests/test_smoke.py
```
