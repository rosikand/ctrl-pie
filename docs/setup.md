---
title: "First-time setup"
description: "Use the Settings checklist to verify PostgreSQL, Hugging Face, Modal, and arm readiness."
icon: "list-checks"
---

ctrl-π starts even when required configuration is missing so the Settings page
can explain what needs attention. Other pages show a setup banner until every
service required by the active mode is ready.

Credentials are not entered in the browser. Put secrets in the backend's
gitignored `.env` file or the supported provider credential store, restart the
backend, and then use **Recheck** in Settings.

## Open the checklist

<Steps>
  <Step title="Start ctrl-π">
    Follow [Installation](/installation). A blank `DATABASE_URL` still allows
    the backend and setup checklist to start, but database-backed workflows
    return HTTP 503 until PostgreSQL is configured and migrated.
  </Step>
  <Step title="Open Settings">
    Choose the gear icon in the desktop sidebar or mobile header. The page
    reports the current **Mock** or **Hardware** mode and displays one card for
    PostgreSQL, Hugging Face, Modal, and arms.
  </Step>
  <Step title="Resolve required services">
    Edit `.env` on the backend host, apply any required migration, and restart
    the single backend process. Service details are sanitized and never expose
    connection strings, tokens, device paths, or provider responses.
  </Step>
  <Step title="Recheck readiness">
    Choose **Recheck**. The page shows **Setup complete** only when every
    service required by the active mode is connected or configured.
  </Step>
</Steps>

## What each mode requires

| Service | Mock mode | Hardware mode | What the check proves |
| --- | --- | --- | --- |
| PostgreSQL | Required | Required | The configured database accepts a bounded `SELECT 1`. |
| Hugging Face | Optional | Required | `HF_TOKEN` identifies the exact configured user or organization namespace. |
| Modal | Optional | Required | A complete local API credential pair/profile and proxy-token pair pass local validation. This is not a live provider call. |
| Arms | Required | Required | The selected driver reports its arms connected. Hardware mode never falls back to mocks. |

Mock mode is deliberately useful with Hugging Face and Modal blank. Dataset
upload/browsing and model discovery still require Hub credentials when you
choose those workflows.

## Settings page

The browser can edit only non-secret operational defaults stored in
PostgreSQL:

- **Recording FPS** — capture rate from 1 to 60.
- **Default runtime** — LeRobot or OpenPI. Real OpenPI execution is unavailable
  in V1 even though the mock adapter can emulate its contract.
- **Default compute** — Modal A10G, A100, or H100.
- **Deployment timeout** — immutable lifetime copied into each new deployment,
  from 1 to 30 minutes.

The Hugging Face namespace is displayed read-only because its source of truth
is `HF_NAMESPACE`. Settings never accepts or returns `DATABASE_URL`,
`HF_TOKEN`, Modal credentials, or YAM device paths.

## Verify outside the browser

```bash
curl --fail http://127.0.0.1:8000/api/health
curl --fail http://127.0.0.1:8000/api/settings/status
```

`/api/health` proves the FastAPI process is reachable. The Settings response
contains only booleans and safe status text. A configured Modal card means the
local credentials are structurally ready; it does not claim that Modal was
contacted.

## Change modes

`CTRL_PI_MOCK_MODE` selects both the arm and compute boundaries at process
startup:

```dotenv
CTRL_PI_MOCK_MODE=true
```

Set it to `false` only after completing [YAM setup](/yam-setup) and the Modal
sections of [Configuration and credentials](/configuration). Restart the
backend after changing the mode. A hardware startup failure leaves stable,
disconnected `yam-leader` and `yam-follower` identities with an actionable
diagnostic; it never substitutes `MockYAMDriver`.

## Next steps

- [Configuration and credentials](/configuration)
- [Run the mock quickstart](/quickstart)
- [Onboard YAM hardware](/yam-setup)
- [Troubleshooting](/troubleshooting)
