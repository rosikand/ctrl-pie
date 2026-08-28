from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from ctrl_pi.compute import ResourcePolicy

MODAL_HEALTH_FUNCTION = "health"
MODAL_OWNERSHIP_TAG_KEY = "ctrl-pi-deployment"


class _ModalFunctionDecorator(Protocol):
    def __call__(self, function: Any) -> Any: ...


class _ModalModule(Protocol):
    App: Any
    Image: Any

    def fastapi_endpoint(
        self,
        *,
        method: str,
        requires_proxy_auth: bool,
    ) -> _ModalFunctionDecorator: ...


@dataclass(frozen=True)
class ModalWorkload:
    app: Any
    health_function: Any


def health_echo(payload: dict[str, object]) -> dict[str, object]:
    """Tiny M9 proof-of-life endpoint; no model or credential access."""

    nonce = payload.get("nonce")
    healthy = isinstance(nonce, str) and 1 <= len(nonce) <= 128
    return {
        "healthy": healthy,
        "echo": nonce if healthy else "",
    }


def build_modal_workload(
    *,
    app_name: str,
    deployment_id: uuid.UUID,
    resources: ResourcePolicy,
    modal_module: _ModalModule | None = None,
) -> ModalWorkload:
    """Build, but do not deploy, the bounded CPU-only Modal echo App."""

    if modal_module is None:
        import modal

        modal_module = modal

    image = modal_module.Image.debian_slim(python_version="3.11").uv_pip_install(
        "fastapi==0.141.1"
    )
    app = modal_module.App(
        app_name,
        image=image,
        tags={MODAL_OWNERSHIP_TAG_KEY: str(deployment_id)},
    )
    web_function = modal_module.fastapi_endpoint(
        method="POST",
        requires_proxy_auth=False,
    )(health_echo)
    health_function = app.function(
        name=MODAL_HEALTH_FUNCTION,
        min_containers=resources.min_containers,
        buffer_containers=resources.buffer_containers,
        max_containers=resources.max_containers,
        scaledown_window=resources.scaledown_window_seconds,
        timeout=resources.timeout_seconds,
    )(web_function)
    return ModalWorkload(app=app, health_function=health_function)
