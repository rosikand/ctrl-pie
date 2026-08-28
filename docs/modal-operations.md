# Modal operator cleanup

Milestone 9 uses one Modal App for each ctrl-pi deployment. Run the panic
command if a deploy or normal stop is interrupted:

```bash
make modal-panic
```

The command uses `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` when both are set.
When neither is set, the pinned Modal SDK reads the active profile created by
the Modal CLI. Credentials are never printed, persisted by ctrl-pi, or sent to
the browser.

## Safety boundary

An App is considered owned only when both markers match exactly:

- its name is `ctrl-pi-<canonical deployment UUID>`; and
- its Modal App tags contain `ctrl-pi-deployment=<the same UUID>`.

Near-prefix names, malformed UUIDs, and Apps with a missing or different tag
are left untouched. After that check, the command stops the verified provider
App ID (not a mutable name), waits for Modal's raw lifecycle to become stopped
with zero listed tasks, and enumerates owned active Apps again.

Candidates are checked independently. If one exact-name candidate cannot have
its tag read, it is not stopped, but other exactly verified Apps are still
cleaned up and the command exits nonzero with the unverifiable name.

Exit status `0` means no owned App remains active. A credentials, listing,
stop, timeout, ownership, or final-verification failure produces a nonzero
status. Cleanup continues across the other exactly owned Apps after an
individual failure, so it is safe to correct the reported issue and rerun the
same command.
