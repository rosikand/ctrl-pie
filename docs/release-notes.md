---
title: "Changelog"
description: "Product and documentation changes for ctrl-π."
icon: "history"
---

This changelog summarizes user-visible ctrl-π changes. The Git history remains
the detailed engineering record.

## Unreleased

### Product

- Replaced the narrow single-pair hardware setup with one configurable,
  normalized YAM cell: stable USB-CAN identities, current-interface
  resolution, typed arm roles/end effectors, pair/group/side metadata, CAN
  teaching-handle leaders, calibrated `linear_4310`/`crank_4310` followers,
  and a deterministic four-arm/two-pair mock fixture. The serial-GELLO pair
  remains compatible.
- Added a supervised direct-i2rt worker per connected CAN arm. Hardware images
  and preflight require an exact clean operator checkout, baked dependency
  commit, and read-only runtime mount; ctrl-π never fetches public upstream or
  selects latest. Runtime faults latch without blind respawn.
- Made connect/disconnect and rig ownership per arm. General connection,
  unattended restoration, and calibrated-4310 jaw motion have distinct
  acknowledgements. CAN connectivity and teaching-handle range health are
  reported separately; handle zeroing remains an external maintenance action.
- Made teleop start observation-only with synchronization disabled and zero
  follower writes. Live deltas precede a separate acknowledged approximately
  three-second sync correction. Cross-pair routing fails closed; one follower
  remains the inference target.
- Added configurable Uvicorn listen port, a least-privileged host-network
  all-CAN Compose override, a separate single-device legacy GELLO override,
  and the exact H0–H7 physical field-test handoff.
- Promoted **Training** and **Models** into separate first-class routes and
  primary tabs. The application now has exactly six workflow tabs, with
  Settings remaining behind the gear icon.
- Added Settings-based YAM discovery, read-only preflight, one-row physical
  setup persistence, explicit connection, and consented boot/hot-plug
  restoration. Mock setup retains full behavior without mutating the saved
  physical rig.
- Added `ctrl_pi.CtrlPiClient`, a typed synchronous SDK over the same REST
  services as the UI, covering system/settings, YAM setup and arms, recording,
  datasets, models, external and managed training, and inference.
- Added canonical `GET /api/models`; the legacy Trainer model route remains a
  deprecated compatibility alias backed by the same service.
- Added bounded, sequenced training console records while preserving the
  legacy `ctrl_pi.trainer.Client` API.
- Added SDK/REST-launched managed SmolVLA training through LeRobot 0.4.4 on exact owned Modal Apps,
  all supported A10G/A100/H100 allocations, durable reconciliation, bounded
  observability, Hugging Face artifact ownership, cancellation, deadlines, and
  teardown verification. The Training UI observes but does not launch jobs.

Physical YAM behavior remains unvalidated until the target Ubuntu/YAM
H0–H7 checklist is completed. Earlier Lux/i2rt field evidence is behavioral
reference, not validation of this ctrl-π adapter.

### Documentation

- Added cell-centric setup/driver/Docker guidance, exact pinned local i2rt
  source handling, pair-safe sync lifecycle, `NO SASH GUARD` semantics, and a
  navigated physical acceptance guide.
- Added the repository-hosted Mintlify site and production navigation.
- Split installation, quickstart, first-run setup, credentials, YAM onboarding,
  and product workflows into focused guides.
- Added dedicated Arms, Datasets, Training, Managed training, Models, troubleshooting,
  screenshots, and changelog pages.
- Added a product-wide REST/Python SDK guide and current YAM onboarding and
  restoration procedures.
- Added a canonical user-facing AI-agent integration guide for the same public
  SDK/REST surface, with mock-first operation, fresh human approvals, hostile
  content boundaries, idempotent reconciliation, and verified cleanup.
- Kept the existing Trainer API, inference, architecture, Docker, recording,
  YAM driver, Modal operations, development, and smoke guides as the canonical
  deep references.
- Added local validation for Mintlify navigation, page metadata, internal
  links, assets, and duplicate routes.

## 0.1.0 — 2026-08-28

Initial V1 implementation through the canonical specification milestones.

### Product

- Added the original five-tab V1 React/Vite console plus mode-aware Settings
  checklist; V1.1 later separated Training and Models into six tabs.
- Added PostgreSQL/Alembic control-plane persistence and deterministic seed
  data.
- Added mock and fail-closed real standard-YAM driver boundaries, live
  WebSocket telemetry, bounded jogs, and one shared rig lease.
- Added leader/follower teleoperation, synthetic camera capture, durable
  episode manifests, and LeRobot v3 conversion/upload.
- Added exact-namespace Hugging Face dataset and model discovery, private media
  proxying, and a revision-pinned episode visualizer.
- Added the synchronous Python Trainer client and REST API for runs, scalar
  metrics, and checkpoint registration.
- Added Stub and Modal compute targets, LeRobot runtime serving, robot-side
  action execution, optional inference recording, lifetime watchdogs, and
  verified provider teardown.
- Added the single-process production Docker image, scoped YAM device
  passthrough guidance, emergency Modal cleanup, and the full mock smoke gate.

### Known V1 boundaries

- No ctrl-π authentication, multi-user collaboration, queue, Kubernetes, or
  hosted service.
- Training executes outside ctrl-π.
- Real OpenPI inference is unavailable; its mock adapter is deterministic.
- The hardware camera remains synthetic.
- Physical YAM validation is still required on the target Ubuntu box before
  motion.
