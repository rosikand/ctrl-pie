---
title: "Changelog"
description: "Product and documentation changes for ctrl-π."
icon: "history"
---

This changelog summarizes user-visible ctrl-π changes. The Git history remains
the detailed engineering record.

## Unreleased

### Documentation

- Added the repository-hosted Mintlify site and production navigation.
- Split installation, quickstart, first-run setup, credentials, YAM onboarding,
  and product workflows into focused guides.
- Added dedicated Arms, Datasets, Training, Models, troubleshooting,
  screenshots, and changelog pages.
- Kept the existing Trainer API, inference, architecture, Docker, recording,
  YAM driver, Modal operations, development, and smoke guides as the canonical
  deep references.
- Added local validation for Mintlify navigation, page metadata, internal
  links, assets, and duplicate routes.

## 0.1.0 — 2026-08-28

Initial V1 implementation through the canonical specification milestones.

### Product

- Added the five-tab React/Vite console plus mode-aware Settings checklist.
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
