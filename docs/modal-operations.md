---
title: "Modal operations and emergency cleanup"
description: "Inspect, stop, and verify cleanup of ctrl-π-owned inference and managed-training resources."
icon: "cloud-off"
---

Use the Inference tab's Stop control for normal inference shutdown. It stops and joins
robot actions, finalizes an optional local recording, then stops the owned
Modal App and verifies a stopped/absent provider lifecycle with zero running
tasks. The full ordering and runtime security boundary are in
[Compute targets and inference](inference.md).

For managed training, use `cancel_managed_training_job` through the SDK/REST
API. Natural completion, failure, cancellation, and deadline expiry all stop
the exact training App and require the same zero-task proof before terminal
status. See [Managed training](/managed-training).

`make modal-panic` is the recovery command when a deploy, Stop request,
backend shutdown, or restart reconciliation was interrupted. It scans the
configured Modal environment and stops every App whose ctrl-π ownership can
be proven exactly.

## Credentials and environment

Run the command from the repository root with the same Modal account and
environment used by the backend:

```bash
make modal-panic
```

The command uses `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` when both are set.
They are Modal API credentials and must be configured as a complete pair.
When neither is set, the pinned `modal==1.5.4` SDK reads the active profile
created by the Modal CLI. The Docker equivalent, when those credentials or a
read-only profile are available inside the service, is:

```bash
docker compose run --rm app python -m ctrl_pi.modal_panic
```

Credentials are never printed or persisted by ctrl-π. Modal Proxy Tokens
(`MODAL_PROXY_TOKEN_ID` beginning `wk-` and
`MODAL_PROXY_TOKEN_SECRET` beginning `ws-`) authenticate inference endpoint
traffic but do not authorize App cleanup. Panic does not create, rotate, or
delete Proxy Tokens.

## Exact ownership rule

An inference App is owned only when both provider markers agree:

- its name is exactly `ctrl-pi-<canonical deployment UUID>`; and
- its tags contain `ctrl-pi-deployment=<the same UUID>`.

A managed-training App uses a disjoint rule:

- its name is exactly `ctrl-pi-training-<canonical job UUID>`; and
- its tags contain `ctrl-pi-training-job=<the same UUID>`.

The command evaluates both families independently and never selects resources
by a loose `ctrl-pi` prefix. A
near-prefix name, malformed UUID, or different ownership tag is left
untouched. An active exact-name candidate with a missing tag, unreadable tags,
or otherwise unverifiable identity is also not stopped and makes the command
fail closed. An already-stopped exact-name App with no running tasks needs no
action.

For every verified active candidate, panic stops the provider App ID, not a
mutable name. Immediately before the stop it re-resolves the name, provider
ID, and tag. It then polls Modal's raw App lifecycle and task listing until
the App is stopped or absent with zero tasks. Finally, it enumerates all exact
owned candidates again. These checks use the same AppList, lifecycle, and
AppStop semantics as the pinned Modal CLI.

Candidates are independent. One unverifiable or failed App does not prevent
cleanup of other exactly verified Apps.

## Interpreting the result

A successful run prints output in this form and exits `0`:

```text
modal-panic: verified zero active ctrl-pi Apps (1 stopped).
```

Exit status `0` means the final enumeration found no active inference or
managed-training App matching either exact ownership rule. It does not mean
that similarly named unowned Apps,
local recordings, PostgreSQL rows, Hub repositories, or Proxy Tokens were
removed.

A credential, listing, tag, stop, timeout, ownership, or final-verification
failure exits nonzero and reports only sanitized App names. Correct the
reported account/environment/access issue and rerun the same command. Do not
manually delete a candidate until its ownership has been independently
confirmed in the Modal dashboard.

## Restart behavior

Backend startup never resumes motion. It reconciles persisted inference rows
with their original target and stops any provider still running because the
process-local robot loop did not survive. A healthy managed training call is
different: startup may reattach only when the exact persisted App, FunctionCall,
name, tag, and request identity agree. It never relaunches an ambiguous attempt.
Completed, failed, cancelled, expired, or unreattachable calls converge through
exact App cleanup. Unknown ownership or provider state remains retryable rather
than being declared safe.

After a crash or machine restart:

1. run `make modal-panic` with the original Modal account/environment;
2. require exit status `0` and the zero-active confirmation;
3. start ctrl-π and inspect failed/stopped deployments plus managed jobs;
4. verify the shared rig is idle before creating a new inference deployment;
   and
5. do not relaunch a managed request until its prior idempotency/job state and
   final panic result are authoritative.

The Modal guardrails—zero warm/buffer containers, at most one container per
App, no provider retries, bounded idle scaledown, and immutable hard
deadlines—limit exposure but do not replace verified Stop/cancel or panic
cleanup. Inference deployments are capped at 30 minutes; managed jobs are
capped at 1,440 minutes and ctrl-π permits only one nonterminal managed job.
