Read SPEC.md fully before any work. It is the canonical source of truth.

Work through the SPEC.md milestones in order, continuously, without
stopping to ask for approval. If something is ambiguous, make the choice
SPEC.md implies, log it under "Decisions" in SPEC.md, and continue.

Everything must run in mock mode (MockYAMDriver, mock camera, stubbed
Modal workload). Real HF calls are allowed via HF_TOKEN.

Gates before advancing to the next milestone:
- npm run build passes (frontend)
- pytest passes (backend)
- alembic upgrade head runs clean against DATABASE_URL
- both dev servers start without errors
- commit with message "milestone N: <name>"
- From milestone 11 on: `make smoke` passes. It runs the full mock
  loop end-to-end: record a 5-second episode, upload to HF, list it
  in Datasets, deploy the stub policy, execute 100 action steps on
  the mock arms, tear down.

Do not expand scope. No auth, queues, K8s, or abstractions not in
SPEC.md. Prefer boring implementations.
