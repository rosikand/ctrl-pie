# Full mock smoke gate

`make smoke` is the recurring Milestone 11+ acceptance gate. It keeps all
robot, camera, policy-runtime, and compute behavior in mock mode, while using
the configured Hugging Face account for the dataset boundary.

The default command requires `HF_TOKEN` and `HF_NAMESPACE`. It records and
finalizes one real five-second mock-camera episode, converts it to LeRobot v3,
creates a uniquely named private dataset repository, uploads the artifacts,
and confirms that the repository appears through the Datasets service at the
exact uploaded revision. It then deploys the stub CPU policy, applies exactly
100 actions to `yam-follower`, and stops the robot loop and stub workload.

The smoke repository is ephemeral. Cleanup reads the ctrl-pi recording marker
at an immutable 40-character revision, verifies the repository still has that
same revision, and only then deletes the exact repository. If either identity
check fails, cleanup fails closed and reports the repository id for manual
inspection. The token is never included in evidence output.

Run the real gate from the repository root:

```sh
make smoke
```

Successful output reports evidence for `record`, `upload`, `datasets`,
`deploy`, `inference`, and `teardown`. The final teardown line verifies zero
recording tasks, episodes, inference tasks, queued actions, and active compute
resources; it also verifies a free rig lease, idle uploader, deleted Hub
repository, and removed temporary workspace.

Tests and offline development must opt into the deterministic filesystem Hub:

```sh
PYTHONPATH=backend/src .venv/bin/python -m ctrl_pi.smoke --fake-hub
```

`--fake-hub` still performs the real five-second recording and LeRobot
conversion. It replaces only the remote Hub transport and makes no external
calls. It is not used by `make smoke`.

This recurring gate deliberately uses stub compute. The separate Milestone 11
live demo is responsible for retained real-Modal latency and teardown
evidence.
