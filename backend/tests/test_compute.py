from __future__ import annotations

import uuid
from dataclasses import replace

import pytest

from ctrl_pi.compute import (
    ComputeOwnershipError,
    ComputeTargetError,
    DeploymentHandle,
    DeploymentSpec,
    ResourcePolicy,
    TargetState,
    deployment_app_name,
    deployment_ownership_tag,
)
from ctrl_pi.compute_stub import StubComputeTarget


def _spec(deployment_id: uuid.UUID | None = None) -> DeploymentSpec:
    deployment_id = deployment_id or uuid.uuid4()
    return DeploymentSpec(
        deployment_id=deployment_id,
        app_name=deployment_app_name(deployment_id),
        ownership_tag=deployment_ownership_tag(deployment_id),
        model_repo="acme/stub-policy",
        checkpoint_revision="a" * 40,
        runtime="stub",
        resources=ResourcePolicy(compute_size="CPU", timeout_seconds=1800),
    )


def test_stub_lifecycle_is_deterministic_and_nonce_verified() -> None:
    target = StubComputeTarget()
    spec = _spec()

    handle = target.deploy(spec)
    health = target.health(handle, "fixed-health-nonce")
    running = target.inspect(handle)
    target.stop(handle)
    stopped = target.inspect(handle)
    target.stop(handle)

    assert handle.provider_app_id == f"stub-{spec.deployment_id.hex}"
    assert handle.app_name == f"ctrl-pi-{spec.deployment_id}"
    assert health.healthy is True and health.echo == "fixed-health-nonce"
    assert running.running_verified is True
    assert stopped.stopped_verified is True
    assert stopped.running_tasks == 0
    assert target.list_owned() == [stopped]
    with pytest.raises(ComputeTargetError):
        target.deploy(spec)


def test_resource_policy_enforces_no_warm_pool_and_runaway_bounds() -> None:
    with pytest.raises(ValueError, match="warm"):
        ResourcePolicy(
            compute_size="CPU",
            timeout_seconds=60,
            min_containers=1,
        )
    with pytest.raises(ValueError, match="1800"):
        ResourcePolicy(compute_size="CPU", timeout_seconds=1801)
    with pytest.raises(ValueError, match="scaledown"):
        ResourcePolicy(
            compute_size="CPU",
            timeout_seconds=60,
            scaledown_window_seconds=61,
        )


def test_absence_is_not_stopped_when_provider_reports_active_lifecycle() -> None:
    spec = _spec()
    base = TargetState(
        deployment_id=spec.deployment_id,
        provider_app_id=f"stub-{spec.deployment_id.hex}",
        app_name=spec.app_name,
        ownership_tag=spec.ownership_tag,
        exists=False,
        lifecycle="running",
        running_tasks=0,
        endpoint_url=f"stub://ctrl-pi/{spec.deployment_id}",
    )

    assert base.stopped_verified is False
    assert replace(base, lifecycle="deploying").stopped_verified is False
    assert replace(base, lifecycle="stopping").stopped_verified is False
    assert replace(base, lifecycle="failed").stopped_verified is False
    assert replace(base, lifecycle="unknown").stopped_verified is True
    assert replace(base, lifecycle="stopped").stopped_verified is True


def test_stub_rejects_a_safe_but_wrong_provider_identity() -> None:
    target = StubComputeTarget()
    spec = _spec()
    handle = target.deploy(spec)
    wrong = DeploymentHandle(
        deployment_id=handle.deployment_id,
        provider_app_id="stub-wrong",
        app_name=handle.app_name,
        ownership_tag=handle.ownership_tag,
        endpoint_url=handle.endpoint_url,
    )

    with pytest.raises(ComputeOwnershipError):
        target.stop(wrong)


def test_identity_handle_can_exist_before_provider_publishes_endpoint_url() -> None:
    spec = _spec()
    handle = DeploymentHandle(
        deployment_id=spec.deployment_id,
        provider_app_id=f"provider-{spec.deployment_id.hex}",
        app_name=spec.app_name,
        ownership_tag=spec.ownership_tag,
        endpoint_url=None,
    )
    state = TargetState(
        deployment_id=handle.deployment_id,
        provider_app_id=handle.provider_app_id,
        app_name=handle.app_name,
        ownership_tag=handle.ownership_tag,
        exists=True,
        lifecycle="deploying",
        running_tasks=0,
        endpoint_url=None,
    )

    assert state.handle() == handle
