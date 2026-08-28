from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from ctrl_pi.config import AppConfig, get_config
from ctrl_pi.drivers.real_yam import RealYAMDriver
from ctrl_pi.drivers.yam import YAMDriver, YAMDriverDiagnostic


def _safe_report(
    driver: YAMDriver,
    *,
    connect_requested: bool,
    diagnostic: YAMDriverDiagnostic | None = None,
) -> dict[str, object]:
    diagnostic = diagnostic or driver.diagnostic()
    return {
        "connect_requested": connect_requested,
        "status": diagnostic.status,
        "detail": diagnostic.detail,
        "arms": [
            {
                "id": arm.id,
                "role": arm.role,
                "connected": arm.connected,
                "driver": arm.driver,
                "can_state": arm.can.state,
            }
            for arm in driver.list_arms()
        ],
    }


def run_probe(
    config: AppConfig,
    *,
    connect: bool,
    driver: YAMDriver | None = None,
) -> int:
    selected_driver = driver or RealYAMDriver.from_app_config(config)
    try:
        preflight = (
            selected_driver.preflight()
            if isinstance(selected_driver, RealYAMDriver)
            else selected_driver.diagnostic()
        )
        if connect:
            if preflight.status == "configured":
                selected_driver.startup()
            diagnostic = selected_driver.diagnostic()
        else:
            diagnostic = preflight
        report = _safe_report(
            selected_driver,
            connect_requested=connect,
            diagnostic=diagnostic,
        )
        print(json.dumps(report, sort_keys=True))
        arms = selected_driver.list_arms()
        if connect:
            return int(
                report["status"] != "connected"
                or not arms
                or not all(arm.connected for arm in arms)
            )
        return int(report["status"] not in {"configured", "connected"})
    finally:
        selected_driver.shutdown()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Preflight YAM config or opt in to one connect-and-sample probe."
    )
    parser.add_argument(
        "--connect",
        action="store_true",
        help="Open configured devices, take a sample, and close them safely.",
    )
    args = parser.parse_args(argv)
    return run_probe(get_config(), connect=args.connect)


if __name__ == "__main__":
    raise SystemExit(main())
