from __future__ import annotations

import sys
from typing import Protocol, TextIO, TypeAlias

from ctrl_pi.compute import ComputeTargetError, TargetState
from ctrl_pi.compute_modal import ModalComputeTarget
from ctrl_pi.config import get_config
from ctrl_pi.training_compute import ManagedTrainingTargetError, TrainingTargetState
from ctrl_pi.training_compute_modal import ModalTrainingTarget


_PanicState: TypeAlias = TargetState | TrainingTargetState


class _PanicTarget(Protocol):
    def list_owned_for_panic(self) -> tuple[list[_PanicState], list[str]]: ...

    def stop_owned(self, state: _PanicState) -> None: ...


class CombinedModalPanicTarget:
    """Aggregate exact inference and training owners without cross-stopping."""

    def __init__(
        self,
        inference: ModalComputeTarget,
        training: ModalTrainingTarget,
    ) -> None:
        self._inference = inference
        self._training = training

    def list_owned_for_panic(self) -> tuple[list[_PanicState], list[str]]:
        states: list[_PanicState] = []
        unverifiable: list[str] = []
        for label, target in (
            ("inference-enumeration", self._inference),
            ("training-enumeration", self._training),
        ):
            try:
                owned, target_unverifiable = target.list_owned_for_panic()
                states.extend(owned)
                unverifiable.extend(target_unverifiable)
            except (ComputeTargetError, ManagedTrainingTargetError):
                unverifiable.append(label)
            except Exception:
                unverifiable.append(label)
        return states, sorted(set(unverifiable))

    def stop_owned(self, state: _PanicState) -> None:
        if isinstance(state, TrainingTargetState):
            self._training.stop_owned(state)
            return
        self._inference.stop_owned(state)


def run_modal_panic(target: _PanicTarget, *, output: TextIO) -> int:
    """Stop and verify every exactly marked ctrl-pi Modal App."""

    try:
        states, unverifiable = target.list_owned_for_panic()
    except (ComputeTargetError, ManagedTrainingTargetError) as error:
        print(f"modal-panic: {error}", file=output)
        return 1
    except Exception:
        print("modal-panic: Modal Apps could not be enumerated.", file=output)
        return 1

    active_by_id = {
        state.provider_app_id: state
        for state in states
        if not state.stopped_verified
    }
    failures: list[str] = list(unverifiable)
    for state in active_by_id.values():
        try:
            target.stop_owned(state)
        except (ComputeTargetError, ManagedTrainingTargetError):
            failures.append(state.app_name)
        except Exception:
            failures.append(state.app_name)

    try:
        verified, final_unverifiable = target.list_owned_for_panic()
        remaining = [
            state
            for state in verified
            if not state.stopped_verified
        ]
        failures.extend(final_unverifiable)
    except (ComputeTargetError, ManagedTrainingTargetError):
        remaining = list(active_by_id.values())
        failures.append("verification")
    except Exception:
        remaining = list(active_by_id.values())
        failures.append("verification")

    if failures or remaining:
        names = sorted(
            {
                *failures,
                *(state.app_name for state in remaining),
            }
        )
        print(
            "modal-panic: cleanup could not be verified for " + ", ".join(names),
            file=output,
        )
        return 1

    print(
        f"modal-panic: verified zero active ctrl-pi Apps ({len(active_by_id)} stopped).",
        file=output,
    )
    return 0


def main() -> int:
    try:
        config = get_config()
        target = CombinedModalPanicTarget(
            ModalComputeTarget.from_config(config),
            ModalTrainingTarget.from_config(config),
        )
    except (ComputeTargetError, ManagedTrainingTargetError) as error:
        print(f"modal-panic: {error}", file=sys.stderr)
        return 1
    return run_modal_panic(target, output=sys.stdout)


if __name__ == "__main__":
    raise SystemExit(main())
