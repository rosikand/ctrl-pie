---
title: "YAM cell field acceptance"
description: "Run the mock-first H0–H7 acceptance sequence without overstating physical validation."
icon: "clipboard-check"
---

V1.2's all-CAN YAM cell support was implemented and tested without physical
hardware in the development environment. Physical acceptance is deliberately
separate and must run in order on the Ubuntu YAM workstation.

The repository root contains the exact operator handoff:

<Card title="V1.2 physical field-test handoff" icon="file-check" href="https://github.com/rosikand/ctrl-pie/blob/v1.2-yam-cell/V1_2_FIELD_TEST_HANDOFF.md">
  Architecture, pinned local i2rt source checks, known risks, motion boundaries,
  and the complete H0–H7 procedure.
</Card>

The gate sequence is:

1. **H0:** production-shaped four-arm mock smoke only.
2. **H1:** passive real-cell discovery; no device open, link mutation, motor
   ping, motor enable, or gripper calibration.
3. **H2:** explicitly acknowledged read-only teaching-handle range checks; no
   encoder zeroing.
4. **H3:** one follower connect/calibrate/telemetry/disconnect.
5. **H4:** right-pair observation-only teleop, live delta inspection, separate
   slow-sync acknowledgement, tiny motion, clean disable.
6. **H5:** repeat for the left pair and prove pair isolation.
7. **H6:** short open-table right-pair recording and private Hub verification.
8. **H7:** one deliberately small, controlled follower inference path; Stop
   execution, explicitly Disconnect and prove worker reap plus
   de-energized/limp hardware, and verify provider teardown.

<Warning>
  The reference cell currently has no active soft-limit artifacts. `NO SASH
  GUARD` means it is not approved for in-hood motion. The H4 approximately
  15-degree check is a field-session operator heuristic, not a product limit.
</Warning>

Mock success, a passing passive preflight, and the earlier Lux/i2rt evidence
do not validate ctrl-π V1.2 on hardware. Record each gate's exact commit,
observations, cleanup proof, and any deviation before advancing.
