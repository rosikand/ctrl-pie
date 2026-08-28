from __future__ import annotations

import io
import traceback
import uuid
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from ctrl_pi.compute import (
    ComputeConfigurationError,
    ComputeOwnershipError,
    ComputeTargetError,
    DeploymentHandle,
    DeploymentSpec,
    ResourcePolicy,
    TargetState,
    deployment_app_name,
    deployment_ownership_tag,
)
from ctrl_pi.compute_modal import (
    ModalComputeTarget,
    _OfficialModalGateway,
    _ProviderApp,
    _ProviderDeployment,
    _ResolvedProviderApp,
)
from ctrl_pi.modal_panic import run_modal_panic
from ctrl_pi.modal_workload import (
    MODAL_HEALTH_FUNCTION,
    MODAL_OWNERSHIP_TAG_KEY,
    build_modal_workload,
    health_echo,
)

DEPLOYMENT_ID = uuid.UUID("11111111-1111-4111-8111-111111111111")
OTHER_DEPLOYMENT_ID = uuid.UUID("22222222-2222-4222-8222-222222222222")
PROVIDER_APP_ID = "ap-owned111111"
ENDPOINT_URL = "https://workspace--ctrl-pi-health.modal.run"
SECRET = "modal_super_secret_value"


def _spec(
    *,
    deployment_id: uuid.UUID = DEPLOYMENT_ID,
    runtime: str = "stub",
    compute_size: str = "CPU",
    scaledown_window_seconds: int = 60,
) -> DeploymentSpec:
    return DeploymentSpec(
        deployment_id=deployment_id,
        app_name=deployment_app_name(deployment_id),
        ownership_tag=deployment_ownership_tag(deployment_id),
        model_repo="ctrl-pi/m9-echo",
        checkpoint_revision=None,
        runtime=runtime,
        resources=ResourcePolicy(
            compute_size=compute_size,
            timeout_seconds=1800,
            scaledown_window_seconds=scaledown_window_seconds,
        ),
    )


def _handle() -> DeploymentHandle:
    return DeploymentHandle(
        deployment_id=DEPLOYMENT_ID,
        provider_app_id=PROVIDER_APP_ID,
        app_name=deployment_app_name(DEPLOYMENT_ID),
        ownership_tag=deployment_ownership_tag(DEPLOYMENT_ID),
        endpoint_url=ENDPOINT_URL,
    )


class FakeGateway:
    def __init__(self) -> None:
        self.apps: list[_ProviderApp] = []
        self.tags: dict[str, dict[str, str]] = {}
        self.lifecycles: dict[str, str | None] = {}
        self.resolved: dict[str, _ResolvedProviderApp] = {}
        self.endpoint_urls: dict[str, str | None] = {}
        self.deploy_result = _ProviderDeployment(PROVIDER_APP_ID, ENDPOINT_URL)
        self.deploy_error: Exception | None = None
        self.list_error: Exception | None = None
        self.tag_error: Exception | None = None
        self.lifecycle_error: Exception | None = None
        self.endpoint_error: Exception | None = None
        self.stop_error: Exception | None = None
        self.configure_owned_on_deploy = True
        self.stop_updates_state = True
        self.deploy_calls: list[DeploymentSpec] = []
        self.stop_calls: list[str] = []
        self.tags_read: list[str] = []
        self.list_calls = 0
        self.lifecycle_calls = 0

    def add_app(
        self,
        *,
        deployment_id: uuid.UUID = DEPLOYMENT_ID,
        app_id: str = PROVIDER_APP_ID,
        app_name: str | None = None,
        tag_value: str | None = None,
        lifecycle: str = "running",
        running_tasks: int = 1,
        endpoint_url: str | None = ENDPOINT_URL,
    ) -> None:
        name = app_name or deployment_app_name(deployment_id)
        self.apps.append(
            _ProviderApp(
                app_id=app_id,
                app_name=name,
                lifecycle=lifecycle,  # type: ignore[arg-type]
                running_tasks=running_tasks,
            )
        )
        self.tags[app_id] = {
            MODAL_OWNERSHIP_TAG_KEY: tag_value or str(deployment_id)
        }
        self.lifecycles[app_id] = lifecycle
        self.resolved[name] = _ResolvedProviderApp(
            app_id=app_id,
            lifecycle=lifecycle,  # type: ignore[arg-type]
        )
        self.endpoint_urls[name] = endpoint_url

    def deploy(self, spec: DeploymentSpec) -> _ProviderDeployment:
        self.deploy_calls.append(spec)
        if self.deploy_error is not None:
            raise self.deploy_error
        if self.configure_owned_on_deploy:
            self.add_app(
                deployment_id=spec.deployment_id,
                app_id=self.deploy_result.app_id,
                endpoint_url=self.deploy_result.endpoint_url,
            )
        return self.deploy_result

    def list_apps(self) -> list[_ProviderApp]:
        self.list_calls += 1
        if self.list_error is not None:
            raise self.list_error
        return list(self.apps)

    def resolve_name(self, app_name: str) -> _ResolvedProviderApp | None:
        return self.resolved.get(app_name)

    def get_lifecycle(self, app_id: str) -> str | None:
        self.lifecycle_calls += 1
        if self.lifecycle_error is not None:
            raise self.lifecycle_error
        return self.lifecycles.get(app_id)

    def get_tags(self, app_id: str) -> dict[str, str]:
        self.tags_read.append(app_id)
        if self.tag_error is not None:
            raise self.tag_error
        return dict(self.tags.get(app_id, {}))

    def get_endpoint_url(self, app_name: str) -> str | None:
        if self.endpoint_error is not None:
            raise self.endpoint_error
        return self.endpoint_urls.get(app_name)

    def stop(self, app_id: str) -> None:
        self.stop_calls.append(app_id)
        if self.stop_error is not None:
            raise self.stop_error
        if not self.stop_updates_state:
            return
        self.lifecycles[app_id] = "stopped"
        self.apps = [
            replace(app, lifecycle="stopped", running_tasks=0)
            if app.app_id == app_id
            else app
            for app in self.apps
        ]
        for name, resolved in list(self.resolved.items()):
            if resolved.app_id == app_id:
                self.resolved[name] = replace(resolved, lifecycle="stopped")


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _assert_secret_safe(error: BaseException) -> None:
    rendered = "".join(traceback.format_exception(error))
    assert SECRET not in str(error)
    assert SECRET not in rendered
    assert error.__cause__ is None
    assert error.__context__ is None


def test_workload_builder_applies_exact_tags_and_cost_guardrails_with_fakes() -> None:
    class FakeImage:
        def __init__(self, owner: "FakeModal") -> None:
            self.owner = owner

        def uv_pip_install(self, package: str) -> "FakeImage":
            self.owner.package = package
            return self

    class FakeImageAPI:
        def __init__(self, owner: "FakeModal") -> None:
            self.owner = owner

        def debian_slim(self, *, python_version: str) -> FakeImage:
            self.owner.python_version = python_version
            return FakeImage(self.owner)

    class FakeApp:
        def __init__(self, owner: "FakeModal") -> None:
            self.owner = owner

        def function(self, **kwargs: Any):
            self.owner.function_options = kwargs

            def decorate(function: Any) -> object:
                self.owner.web_function = function
                return self.owner.function_handle

            return decorate

    class FakeModal:
        def __init__(self) -> None:
            self.Image = FakeImageAPI(self)
            self.function_handle = object()
            self.app_name = ""
            self.tags: dict[str, str] = {}
            self.python_version = ""
            self.package = ""
            self.web_method = ""
            self.requires_proxy_auth = True
            self.function_options: dict[str, Any] = {}
            self.web_function: Any = None

        def App(self, app_name: str, *, image: Any, tags: dict[str, str]) -> FakeApp:
            self.app_name = app_name
            self.tags = tags
            assert isinstance(image, FakeImage)
            return FakeApp(self)

        def fastapi_endpoint(self, *, method: str, requires_proxy_auth: bool):
            self.web_method = method
            self.requires_proxy_auth = requires_proxy_auth
            return lambda function: function

    fake_modal = FakeModal()
    spec = _spec()
    workload = build_modal_workload(
        app_name=spec.app_name,
        deployment_id=spec.deployment_id,
        resources=spec.resources,
        modal_module=fake_modal,  # type: ignore[arg-type]
    )

    assert fake_modal.app_name == deployment_app_name(DEPLOYMENT_ID)
    assert fake_modal.tags == {MODAL_OWNERSHIP_TAG_KEY: str(DEPLOYMENT_ID)}
    assert fake_modal.python_version == "3.11"
    assert fake_modal.package == "fastapi==0.141.1"
    assert fake_modal.web_method == "POST"
    assert fake_modal.requires_proxy_auth is False
    assert fake_modal.function_options == {
        "name": MODAL_HEALTH_FUNCTION,
        "min_containers": 0,
        "buffer_containers": 0,
        "max_containers": 1,
        "scaledown_window": 60,
        "timeout": 1800,
    }
    assert workload.health_function is fake_modal.function_handle
    assert fake_modal.web_function is health_echo
    assert health_echo({"nonce": "proof"}) == {
        "healthy": True,
        "echo": "proof",
    }
    assert health_echo({"nonce": SECRET * 20}) == {
        "healthy": False,
        "echo": "",
    }


def test_deploy_uses_exact_identity_and_returns_verified_handle() -> None:
    gateway = FakeGateway()
    target = ModalComputeTarget(gateway=gateway)
    spec = _spec()

    handle = target.deploy(spec)

    assert gateway.deploy_calls == [spec]
    assert handle == _handle()
    assert gateway.stop_calls == []


@pytest.mark.parametrize(
    "spec",
    [
        _spec(runtime="lerobot"),
        _spec(compute_size="A10G"),
        _spec(scaledown_window_seconds=1),
    ],
)
def test_deploy_rejects_non_probe_or_unbounded_resources(spec: DeploymentSpec) -> None:
    gateway = FakeGateway()
    target = ModalComputeTarget(gateway=gateway)

    with pytest.raises(ComputeConfigurationError, match="guardrails|stub CPU"):
        target.deploy(spec)

    assert gateway.deploy_calls == []


def test_deploy_refuses_exact_name_collision_without_mutation() -> None:
    gateway = FakeGateway()
    gateway.add_app(tag_value=str(OTHER_DEPLOYMENT_ID))
    target = ModalComputeTarget(gateway=gateway)

    with pytest.raises(ComputeOwnershipError, match="not owned"):
        target.deploy(_spec())

    assert gateway.deploy_calls == []
    assert gateway.stop_calls == []


def test_deploy_refuses_listed_prepublish_name_collision_without_mutation() -> None:
    gateway = FakeGateway()
    gateway.apps.append(
        _ProviderApp(
            app_id=PROVIDER_APP_ID,
            app_name=deployment_app_name(DEPLOYMENT_ID),
            lifecycle="deploying",
            running_tasks=0,
        )
    )
    gateway.tags[PROVIDER_APP_ID] = {
        MODAL_OWNERSHIP_TAG_KEY: str(DEPLOYMENT_ID)
    }
    target = ModalComputeTarget(gateway=gateway)

    with pytest.raises(ComputeTargetError, match="already exists"):
        target.deploy(_spec())

    assert gateway.resolve_name(deployment_app_name(DEPLOYMENT_ID)) is None
    assert gateway.deploy_calls == []
    assert gateway.stop_calls == []


@pytest.mark.parametrize("mismatch", ["provider_id", "tag"])
def test_post_deploy_identity_mismatch_never_stops_unowned_app(mismatch: str) -> None:
    gateway = FakeGateway()
    gateway.configure_owned_on_deploy = False
    if mismatch == "provider_id":
        gateway.resolved[deployment_app_name(DEPLOYMENT_ID)] = _ResolvedProviderApp(
            app_id="ap-someone-elses-app",
            lifecycle="running",
        )
    else:
        gateway.resolved[deployment_app_name(DEPLOYMENT_ID)] = _ResolvedProviderApp(
            app_id=PROVIDER_APP_ID,
            lifecycle="running",
        )
        gateway.tags[PROVIDER_APP_ID] = {
            MODAL_OWNERSHIP_TAG_KEY: str(OTHER_DEPLOYMENT_ID)
        }
    target = ModalComputeTarget(gateway=gateway)

    with pytest.raises(ComputeOwnershipError):
        target.deploy(_spec())

    assert gateway.stop_calls == []


def test_missing_endpoint_after_create_retains_identity_for_verified_teardown() -> None:
    gateway = FakeGateway()
    gateway.deploy_result = _ProviderDeployment(PROVIDER_APP_ID, None)
    target = ModalComputeTarget(gateway=gateway)

    handle = target.deploy(_spec())

    assert handle.endpoint_url is None
    with pytest.raises(ComputeTargetError, match="health endpoint is unavailable"):
        target.health(handle, "nonce")
    target.stop(handle)

    assert gateway.stop_calls == [PROVIDER_APP_ID]
    assert gateway.list_calls >= 2
    assert gateway.lifecycle_calls >= 2
    assert gateway.lifecycles[PROVIDER_APP_ID] == "stopped"


def test_listed_tagged_initializing_orphan_without_name_resolution_can_stop() -> None:
    gateway = FakeGateway()
    gateway.configure_owned_on_deploy = False

    def deploy(spec: DeploymentSpec) -> _ProviderDeployment:
        gateway.deploy_calls.append(spec)
        gateway.apps.append(
            _ProviderApp(
                app_id=PROVIDER_APP_ID,
                app_name=spec.app_name,
                lifecycle="deploying",
                running_tasks=1,
            )
        )
        gateway.tags[PROVIDER_APP_ID] = {
            MODAL_OWNERSHIP_TAG_KEY: str(spec.deployment_id)
        }
        gateway.lifecycles[PROVIDER_APP_ID] = "deploying"
        return _ProviderDeployment(PROVIDER_APP_ID, None)

    gateway.deploy = deploy  # type: ignore[method-assign]
    target = ModalComputeTarget(gateway=gateway)

    handle = target.deploy(_spec())
    assert handle.endpoint_url is None
    assert gateway.resolve_name(handle.app_name) is None

    owned = target.list_owned()
    assert [state.provider_app_id for state in owned] == [PROVIDER_APP_ID]
    target.stop(handle)

    assert gateway.stop_calls == [PROVIDER_APP_ID]
    assert target.inspect(handle).stopped_verified


def test_list_owned_filters_exact_name_and_tag_and_skips_stopped_history() -> None:
    gateway = FakeGateway()
    gateway.add_app()
    gateway.add_app(
        deployment_id=OTHER_DEPLOYMENT_ID,
        app_id="ap-foreign-tag",
        tag_value=str(DEPLOYMENT_ID),
    )
    gateway.add_app(
        deployment_id=OTHER_DEPLOYMENT_ID,
        app_id="ap-stopped-history",
        lifecycle="stopped",
        running_tasks=0,
    )
    gateway.apps.append(
        _ProviderApp(
            app_id="ap-near-prefix",
            app_name=f"{deployment_app_name(DEPLOYMENT_ID)}-worker",
            lifecycle="running",
            running_tasks=1,
        )
    )
    gateway.tags["ap-near-prefix"] = {
        MODAL_OWNERSHIP_TAG_KEY: str(DEPLOYMENT_ID)
    }
    target = ModalComputeTarget(gateway=gateway)

    states = target.list_owned()

    assert len(states) == 1
    assert states[0].deployment_id == DEPLOYMENT_ID
    assert states[0].endpoint_url == ENDPOINT_URL
    assert states[0].handle() == _handle()
    assert "ap-near-prefix" not in gateway.tags_read


def test_list_owned_keeps_stop_identity_when_endpoint_recovery_fails() -> None:
    gateway = FakeGateway()
    gateway.add_app()
    gateway.endpoint_error = RuntimeError(f"endpoint failed with {SECRET}")
    target = ModalComputeTarget(gateway=gateway)

    states = target.list_owned()

    assert len(states) == 1
    assert states[0].provider_app_id == PROVIDER_APP_ID
    assert states[0].endpoint_url is None
    assert states[0].handle().endpoint_url is None


def test_panic_enumeration_continues_past_unverifiable_exact_candidate() -> None:
    gateway = FakeGateway()
    gateway.add_app()
    gateway.add_app(
        deployment_id=OTHER_DEPLOYMENT_ID,
        app_id="ap-unverifiable",
    )
    original_get_tags = gateway.get_tags

    def get_tags(app_id: str) -> dict[str, str]:
        if app_id == "ap-unverifiable":
            raise RuntimeError(f"tag RPC leaked {SECRET}")
        return original_get_tags(app_id)

    gateway.get_tags = get_tags  # type: ignore[method-assign]
    target = ModalComputeTarget(gateway=gateway)

    states, unverifiable = target.list_owned_for_panic()

    assert [state.provider_app_id for state in states] == [PROVIDER_APP_ID]
    assert unverifiable == [deployment_app_name(OTHER_DEPLOYMENT_ID)]
    with pytest.raises(ComputeTargetError, match="could not be verified") as caught:
        target.list_owned()
    _assert_secret_safe(caught.value)


def test_stop_uses_verified_provider_id_and_polls_lifecycle_and_task_count() -> None:
    gateway = FakeGateway()
    gateway.add_app()
    target = ModalComputeTarget(gateway=gateway)

    target.stop(_handle())

    assert gateway.stop_calls == [PROVIDER_APP_ID]
    assert gateway.list_calls >= 2
    assert gateway.lifecycle_calls >= 2
    assert target.inspect(_handle()).stopped_verified


def test_stop_and_public_inspect_accept_modal_dropping_tags_after_stop() -> None:
    gateway = FakeGateway()
    gateway.add_app()
    original_stop = gateway.stop

    def stop_and_drop_tags(app_id: str) -> None:
        original_stop(app_id)
        gateway.tags[app_id] = {}

    gateway.stop = stop_and_drop_tags  # type: ignore[method-assign]
    target = ModalComputeTarget(gateway=gateway)

    target.stop(_handle())
    stopped = target.inspect(_handle())

    assert gateway.stop_calls == [PROVIDER_APP_ID]
    assert stopped.stopped_verified
    assert stopped.lifecycle == "stopped"
    assert stopped.running_tasks == 0


def test_verified_stop_poll_allows_missing_tags_while_modal_is_stopping() -> None:
    gateway = FakeGateway()
    gateway.add_app()
    clock = FakeClock()

    def begin_stop(app_id: str) -> None:
        gateway.stop_calls.append(app_id)
        gateway.lifecycles[app_id] = "stopping"
        gateway.apps = [
            replace(app, lifecycle="stopping", running_tasks=0)
            if app.app_id == app_id
            else app
            for app in gateway.apps
        ]
        gateway.tag_error = RuntimeError(f"tags disappeared with {SECRET}")

    def finish_stop(seconds: float) -> None:
        clock.sleep(seconds)
        gateway.lifecycles[PROVIDER_APP_ID] = "stopped"
        gateway.apps = [
            replace(app, lifecycle="stopped", running_tasks=0)
            if app.app_id == PROVIDER_APP_ID
            else app
            for app in gateway.apps
        ]

    gateway.stop = begin_stop  # type: ignore[method-assign]
    target = ModalComputeTarget(
        gateway=gateway,
        clock=clock,
        sleeper=finish_stop,
    )

    target.stop(_handle())

    assert gateway.stop_calls == [PROVIDER_APP_ID]
    assert target.inspect(_handle()).stopped_verified


def test_stopped_lifecycle_and_app_list_absence_verify_without_tags() -> None:
    gateway = FakeGateway()
    gateway.add_app()
    gateway.apps = []
    gateway.resolved = {}
    gateway.lifecycles[PROVIDER_APP_ID] = "stopped"
    gateway.tags[PROVIDER_APP_ID] = {}
    target = ModalComputeTarget(gateway=gateway)

    stopped = target.inspect(_handle())
    target.stop(_handle())

    assert stopped.stopped_verified
    assert stopped.exists  # The immutable lifecycle still proves the historical App.
    assert stopped.running_tasks == 0
    assert gateway.stop_calls == []


def test_missing_live_ownership_tag_never_authorizes_stop() -> None:
    gateway = FakeGateway()
    gateway.add_app()
    gateway.tags[PROVIDER_APP_ID] = {}
    target = ModalComputeTarget(gateway=gateway)

    with pytest.raises(ComputeOwnershipError, match="ownership tag"):
        target.stop(_handle())

    assert gateway.stop_calls == []


def test_stop_timeout_is_retryable_and_sanitized() -> None:
    gateway = FakeGateway()
    gateway.add_app()
    gateway.stop_updates_state = False
    clock = FakeClock()
    target = ModalComputeTarget(
        gateway=gateway,
        stop_timeout_seconds=0.5,
        poll_interval_seconds=0.25,
        clock=clock,
        sleeper=clock.sleep,
    )

    with pytest.raises(ComputeTargetError, match="could not be verified") as caught:
        target.stop(_handle())

    assert gateway.stop_calls == [PROVIDER_APP_ID]
    assert gateway.lifecycles[PROVIDER_APP_ID] == "running"
    _assert_secret_safe(caught.value)


def test_health_nonce_round_trip_and_mismatch() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.content == b'{"nonce":"nonce-1"}':
            return httpx.Response(200, json={"healthy": True, "echo": "nonce-1"})
        return httpx.Response(200, json={"healthy": True, "echo": "different"})

    target = ModalComputeTarget(
        gateway=FakeGateway(),
        http_transport=httpx.MockTransport(handler),
    )

    assert target.health(_handle(), "nonce-1").healthy
    mismatch = target.health(_handle(), "nonce-2")
    assert not mismatch.healthy
    assert mismatch.echo == ""


def test_gateway_and_http_errors_never_leak_credentials_or_exception_chains() -> None:
    gateway = FakeGateway()
    gateway.deploy_error = RuntimeError(f"provider rejected {SECRET}")
    target = ModalComputeTarget(gateway=gateway)

    with pytest.raises(ComputeTargetError, match="could not be deployed") as deploy:
        target.deploy(_spec())
    _assert_secret_safe(deploy.value)

    def network_error(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"header contained {SECRET}", request=request)

    health_target = ModalComputeTarget(
        gateway=FakeGateway(),
        http_transport=httpx.MockTransport(network_error),
    )
    with pytest.raises(ComputeTargetError, match="could not be reached") as health:
        health_target.health(_handle(), "nonce")
    _assert_secret_safe(health.value)


@pytest.mark.parametrize("owned", [True, False])
def test_official_gateway_recovers_only_exact_owned_app_after_deploy_exception(
    monkeypatch: pytest.MonkeyPatch,
    owned: bool,
) -> None:
    class FailedApp:
        app_id = None

        def deploy(self, **kwargs: Any) -> None:
            raise RuntimeError(f"deploy failed with {SECRET}")

    gateway = _OfficialModalGateway(
        token_id=None,
        token_secret=None,
        environment_name=None,
    )
    gateway._modal_client = object()
    gateway.resolve_name = lambda app_name: _ResolvedProviderApp(  # type: ignore[method-assign]
        PROVIDER_APP_ID,
        "running",
    )
    gateway.list_apps = lambda: [  # type: ignore[method-assign]
        _ProviderApp(
            PROVIDER_APP_ID,
            deployment_app_name(DEPLOYMENT_ID),
            "running",
            0,
        )
    ]
    tag_value = str(DEPLOYMENT_ID if owned else OTHER_DEPLOYMENT_ID)
    gateway.get_tags = lambda app_id: {  # type: ignore[method-assign]
        MODAL_OWNERSHIP_TAG_KEY: tag_value
    }
    monkeypatch.setattr(
        "ctrl_pi.compute_modal.build_modal_workload",
        lambda **kwargs: SimpleNamespace(
            app=FailedApp(),
            health_function=object(),
        ),
    )

    if owned:
        assert gateway.deploy(_spec()) == _ProviderDeployment(
            PROVIDER_APP_ID,
            None,
        )
    else:
        with pytest.raises(RuntimeError, match="did not complete") as caught:
            gateway.deploy(_spec())
        _assert_secret_safe(caught.value)


def test_official_gateway_recovers_prepublish_initializing_app_from_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailedApp:
        app_id = None

        def deploy(self, **kwargs: Any) -> None:
            raise RuntimeError(f"deploy failed with {SECRET}")

    gateway = _OfficialModalGateway(
        token_id=None,
        token_secret=None,
        environment_name=None,
    )
    gateway._modal_client = object()
    gateway.resolve_name = lambda app_name: None  # type: ignore[method-assign]
    gateway.list_apps = lambda: [  # type: ignore[method-assign]
        _ProviderApp(
            PROVIDER_APP_ID,
            deployment_app_name(DEPLOYMENT_ID),
            "deploying",
            0,
        )
    ]
    gateway.get_tags = lambda app_id: {  # type: ignore[method-assign]
        MODAL_OWNERSHIP_TAG_KEY: str(DEPLOYMENT_ID)
    }
    monkeypatch.setattr(
        "ctrl_pi.compute_modal.build_modal_workload",
        lambda **kwargs: SimpleNamespace(
            app=FailedApp(),
            health_function=object(),
        ),
    )

    assert gateway.deploy(_spec()) == _ProviderDeployment(
        PROVIDER_APP_ID,
        None,
    )


def test_partial_explicit_credentials_are_deferred_and_fail_without_network() -> None:
    target = ModalComputeTarget(token_id=SECRET)

    with pytest.raises(ComputeConfigurationError, match="not configured") as caught:
        target.list_owned()

    _assert_secret_safe(caught.value)


class FakePanicTarget:
    def __init__(
        self,
        states: list[TargetState],
        failures: set[str] | None = None,
        unverifiable: list[str] | None = None,
    ) -> None:
        self.states = {state.provider_app_id: state for state in states}
        self.failures = failures or set()
        self.unverifiable = unverifiable or []
        self.stop_calls: list[str] = []

    def list_owned_for_panic(self) -> tuple[list[TargetState], list[str]]:
        return list(self.states.values()), list(self.unverifiable)

    def stop_owned(self, state: TargetState) -> None:
        self.stop_calls.append(state.provider_app_id)
        if state.provider_app_id in self.failures:
            raise RuntimeError(f"stop failed with {SECRET}")
        self.states.pop(state.provider_app_id, None)


def _active_state(
    deployment_id: uuid.UUID,
    provider_app_id: str,
) -> TargetState:
    return TargetState(
        deployment_id=deployment_id,
        provider_app_id=provider_app_id,
        app_name=deployment_app_name(deployment_id),
        ownership_tag=deployment_ownership_tag(deployment_id),
        exists=True,
        lifecycle="running",
        running_tasks=1,
        endpoint_url=ENDPOINT_URL,
    )


def test_modal_panic_stops_all_owned_apps_and_verifies_zero() -> None:
    target = FakePanicTarget(
        [
            _active_state(DEPLOYMENT_ID, PROVIDER_APP_ID),
            _active_state(OTHER_DEPLOYMENT_ID, "ap-owned222222"),
        ]
    )
    output = io.StringIO()

    status = run_modal_panic(target, output=output)

    assert status == 0
    assert target.stop_calls == [PROVIDER_APP_ID, "ap-owned222222"]
    assert "verified zero active" in output.getvalue()


def test_modal_panic_adapter_never_stops_near_prefix_or_foreign_tag_apps() -> None:
    gateway = FakeGateway()
    gateway.add_app()
    gateway.add_app(
        deployment_id=OTHER_DEPLOYMENT_ID,
        app_id="ap-foreign-tag",
        tag_value=str(DEPLOYMENT_ID),
    )
    gateway.apps.append(
        _ProviderApp(
            app_id="ap-near-prefix",
            app_name=f"{deployment_app_name(DEPLOYMENT_ID)}-worker",
            lifecycle="running",
            running_tasks=1,
        )
    )
    gateway.tags["ap-near-prefix"] = {
        MODAL_OWNERSHIP_TAG_KEY: str(DEPLOYMENT_ID)
    }
    output = io.StringIO()

    status = run_modal_panic(
        ModalComputeTarget(gateway=gateway),
        output=output,
    )

    assert status == 0
    assert gateway.stop_calls == [PROVIDER_APP_ID]
    assert any(app.app_id == "ap-foreign-tag" for app in gateway.apps)
    assert any(app.app_id == "ap-near-prefix" for app in gateway.apps)


def test_modal_panic_reports_missing_marker_on_active_candidate_but_skips_stopped_history() -> None:
    gateway = FakeGateway()
    gateway.add_app()
    gateway.tags[PROVIDER_APP_ID] = {}
    target = ModalComputeTarget(gateway=gateway)

    states, unverifiable = target.list_owned_for_panic()

    assert states == []
    assert unverifiable == [deployment_app_name(DEPLOYMENT_ID)]
    output = io.StringIO()
    assert run_modal_panic(target, output=output) == 1
    assert gateway.stop_calls == []
    assert deployment_app_name(DEPLOYMENT_ID) in output.getvalue()

    gateway.lifecycles[PROVIDER_APP_ID] = "stopped"
    gateway.apps = [
        replace(app, lifecycle="stopped", running_tasks=0)
        for app in gateway.apps
    ]
    assert target.list_owned_for_panic() == ([], [])


def test_modal_panic_continues_after_failure_and_exits_nonzero_without_leak() -> None:
    target = FakePanicTarget(
        [
            _active_state(DEPLOYMENT_ID, PROVIDER_APP_ID),
            _active_state(OTHER_DEPLOYMENT_ID, "ap-owned222222"),
        ],
        failures={PROVIDER_APP_ID},
    )
    output = io.StringIO()

    status = run_modal_panic(target, output=output)

    assert status == 1
    assert target.stop_calls == [PROVIDER_APP_ID, "ap-owned222222"]
    assert deployment_app_name(DEPLOYMENT_ID) in output.getvalue()
    assert SECRET not in output.getvalue()


def test_modal_panic_stops_verified_apps_but_fails_closed_on_unverifiable_candidate() -> None:
    candidate = deployment_app_name(OTHER_DEPLOYMENT_ID)
    target = FakePanicTarget(
        [_active_state(DEPLOYMENT_ID, PROVIDER_APP_ID)],
        unverifiable=[candidate],
    )
    output = io.StringIO()

    status = run_modal_panic(target, output=output)

    assert status == 1
    assert target.stop_calls == [PROVIDER_APP_ID]
    assert candidate in output.getvalue()
