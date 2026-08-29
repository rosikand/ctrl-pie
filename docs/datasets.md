---
title: "Datasets"
description: "Browse LeRobot repositories and inspect synchronized video, observations, and actions."
icon: "database"
---

The Datasets page is a read-only browser over LeRobot repositories owned by
the configured `HF_NAMESPACE`. The backend authenticates with `HF_TOKEN`,
post-filters results to the exact owner, and keeps private bytes away from the
browser.

## Requirements

Set both Hub values and restart the backend:

```dotenv
HF_NAMESPACE=my-user-or-org
HF_TOKEN=hf_replace_with_a_backend_token
```

Read access is enough to browse metadata and private video. Recording upload
requires write and repository-creation access. The Settings check must verify
that the token belongs to the exact user or organization namespace.

## Dataset catalog

Open **Datasets** to load up to 24 repositories at a time. Only exact-owner
repositories carrying the LeRobot tag appear. Cards show available:

- repository ID, visibility, gated status, and immutable revision;
- dataset-card title, description, license, tasks, and tags;
- LeRobot version, robot type, FPS, episode/frame/task counts, and feature
  count; and
- creation and last-modified timestamps.

Choose **Refresh** to bypass the short namespace and card caches. **Load more**
continues keyset pagination from the current last-modified time and repository
ID. A missing or malformed card degrades only that card rather than hiding the
rest of the catalog.

ctrl-π does not infer a license grant for recorded data. A card that has no
license is shown as **not specified**. Public visibility and an appropriate
data license are separate operator choices.

## Episode visualizer

Choose **Browse episodes** on a dataset card. The episode page pins the list,
timeline data, and video to one 40-character Hub commit. Select an episode from
the sidebar or use the previous/next controls.

The player provides:

- native MP4 controls and a separate episode-relative timeline slider;
- frame index, duration, FPS, task labels, and packed-video segment bounds;
- the observation-state vector nearest the selected time; and
- the corresponding action vector with feature names.

Private video is byte-range proxied through a same-origin backend URL. The
browser never receives `HF_TOKEN`, a presigned provider URL, or a local cache
path. Revision pinning prevents timeline rows from one commit being paired
with media from a newer commit.

## Bounded visualization

Episode detail returns at most 2,000 deterministic timeline samples and one
million state/action scalar values. Wide vectors may produce a lower sample
limit. The first and last frame are always included, and the UI labels a
sampled result with both returned and total frame counts.

Before decoding, the backend validates episode indexes, Parquet file/row-group
bounds, cumulative row counts, and scalar slots. Accepted row groups are read
in 1,024-row batches rather than materializing an entire episode table.

## Create a dataset

Datasets are produced through [Record / Teleop](/recording) or optional
[Inference](/inference) recording. Upload:

1. converts every finalized episode with LeRobot 0.4.4;
2. verifies MP4/sample alignment and the generated v3 layout;
3. creates a new private repository by default;
4. uploads with a server-owned recording marker;
5. verifies the returned commit and required remote files; and
6. marks the recording uploaded before removing local raw staging.

The browser supplies only a repository slug and visibility. The backend
prepends `HF_NAMESPACE` and refuses an unrelated existing target.

## Common failures

| Result | Meaning |
| --- | --- |
| HTTP 403 | The token cannot access or does not own the configured namespace. |
| HTTP 404 | The repository or episode is absent at the requested scope. |
| HTTP 409 | The mutable dataset revision changed; reload the episode list. |
| HTTP 422 | Cursor, LeRobot metadata, indexes, or bounds are invalid. |
| HTTP 502 | Hub enumeration or artifact retrieval failed safely. |
| HTTP 503 | `HF_TOKEN` or `HF_NAMESPACE` is missing. |

See [Troubleshooting](/troubleshooting) for recovery steps.
