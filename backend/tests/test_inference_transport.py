from __future__ import annotations

import base64
import json
import uuid
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from ctrl_pi.inference_runtime import (
    ActionChunk,
    EncodedImage,
    ObservationRequest,
    RuntimeDescriptor,
    RuntimeFeature,
    RuntimeLoadSpec,
    StubInferenceRuntime,
)
from ctrl_pi.inference_transport import (
    InferenceTransportConfigurationError,
    InferenceTransportError,
    InferenceTransportProtocolError,
    InProcessInferenceTransport,
    MAX_RUNTIME_RESPONSE_BYTES,
    MAX_RUNTIME_WEBSOCKET_MESSAGE_BYTES,
    ModalInferenceTransport,
)

ENDPOINT = "https://ctrl-pi-runtime--example.modal.run"
REPO_ID = "acme/mock-policy"
REVISION = "c" * 40
PROXY_ID = "wk-test-key"
PROXY_SECRET = "ws-test-secret"


def _descriptor() -> RuntimeDescriptor:
    return RuntimeDescriptor(
        runtime="lerobot",
        policy_type="act",
        model_repo=REPO_ID,
        revision=REVISION,
        inputs=[RuntimeFeature(name="observation.state", kind="state", shape=(7,))],
        action=RuntimeFeature(name="action", kind="action", shape=(7,)),
        actions_per_chunk=2,
    )


def _request(sequence: int = 4) -> ObservationRequest:
    return ObservationRequest(
        request_id=uuid.uuid4(),
        sequence=sequence,
        captured_at=datetime.now(UTC),
        vectors={"observation.state": [0.0] * 7},
    )


def _action_payload(request: ObservationRequest, *, revision: str = REVISION) -> dict[str, Any]:
    now = datetime.now(UTC)
    return ActionChunk(
        request_id=request.request_id,
        observation_sequence=request.sequence,
        revision=revision,
        actions=[[0.0] * 7, [0.01] * 7],
        server_received_at=now,
        server_completed_at=now,
    ).model_dump(mode="json")


def _runtime_http_handler(calls: list[dict[str, Any]]):
    descriptor = _descriptor()

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(
            {
                "url": str(request.url),
                "headers": dict(request.headers),
                "payload": payload,
            }
        )
        assert request.headers["Modal-Key"] == PROXY_ID
        assert request.headers["Modal-Secret"] == PROXY_SECRET
        if payload["type"] == "describe":
            body = descriptor.model_dump(mode="json")
        elif payload["type"] == "health":
            body = {
                "type": "health",
                "healthy": True,
                "echo": payload["nonce"],
                "runtime": descriptor.runtime,
                "model_repo": descriptor.model_repo,
                "revision": descriptor.revision,
            }
        else:
            observation = ObservationRequest.model_validate(payload)
            body = _action_payload(observation)
        return httpx.Response(200, json=body)

    return handler


def test_in_process_transport_runs_loaded_stub_without_network() -> None:
    runtime = StubInferenceRuntime(runtime="lerobot")
    descriptor = runtime.load(
        RuntimeLoadSpec(
            model_repo=REPO_ID,
            revision=REVISION,
            local_model_path=None,
            device="cpu",
            actions_per_chunk=2,
        )
    )
    transport = InProcessInferenceTransport(runtime)
    request = _request()

    health = transport.health("proof")
    chunk = transport.infer(request)

    assert transport.describe() == descriptor
    assert health.healthy and health.echo == "proof"
    assert (health.runtime, health.model_repo, health.revision) == (
        descriptor.runtime,
        descriptor.model_repo,
        descriptor.revision,
    )
    assert chunk.request_id == request.request_id
    transport.close()
    transport.close()
    with pytest.raises(InferenceTransportError, match="closed"):
        transport.describe()


class _CountingStubRuntime(StubInferenceRuntime):
    def __init__(self) -> None:
        super().__init__(runtime="stub")
        self.reset_count = 0

    def reset_session(self) -> None:
        super().reset_session()
        self.reset_count += 1


def test_in_process_transport_resets_once_before_first_inference() -> None:
    runtime = _CountingStubRuntime()
    runtime.load(
        RuntimeLoadSpec(
            model_repo=REPO_ID,
            revision=REVISION,
            local_model_path=None,
            device="cpu",
            actions_per_chunk=2,
        )
    )
    transport = InProcessInferenceTransport(runtime)
    request = _request()

    transport.describe()
    assert runtime.reset_count == 0
    transport.infer(request)
    transport.infer(request)
    assert runtime.reset_count == 1


@pytest.mark.parametrize(
    ("token_id", "token_secret"),
    [
        (None, None),
        (PROXY_ID, None),
        (None, PROXY_SECRET),
        ("ak-api-token", PROXY_SECRET),
        (PROXY_ID, "as-api-secret"),
    ],
)
def test_modal_transport_requires_a_distinct_complete_proxy_pair(
    token_id: str | None,
    token_secret: str | None,
) -> None:
    with pytest.raises(InferenceTransportConfigurationError):
        ModalInferenceTransport(
            ENDPOINT,
            proxy_token_id=token_id,
            proxy_token_secret=token_secret,
        )


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://ctrl-pi-runtime--example.modal.run",
        "https://user:password@ctrl-pi-runtime--example.modal.run",
        "https://ctrl-pi-runtime--example.modal.run?token=secret",
        "https://ctrl-pi-runtime--example.modal.run:443",
        "https://ctrl-pi-runtime--example.modal.run:",
        "https://ctrl-pi-runtime--example.modal.run/runtime",
        "https://ctrl-pi-runtime--example.modal.run/%2e%2e/runtime",
        "https://.modal.run/",
        "https://modal.run/",
        "https://bad_label.modal.run/",
        "https://example.com",
        "not a URL",
    ],
)
def test_modal_transport_rejects_unsafe_urls(endpoint: str) -> None:
    with pytest.raises(InferenceTransportConfigurationError):
        ModalInferenceTransport(
            endpoint,
            proxy_token_id=PROXY_ID,
            proxy_token_secret=PROXY_SECRET,
        )


def test_modal_http_transport_sends_proxy_headers_and_parses_all_envelopes() -> None:
    calls: list[dict[str, Any]] = []
    transport = ModalInferenceTransport(
        ENDPOINT,
        proxy_token_id=SecretStr(PROXY_ID),
        proxy_token_secret=SecretStr(PROXY_SECRET),
        http_transport=httpx.MockTransport(_runtime_http_handler(calls)),
        prefer_websocket=False,
    )
    request = _request()

    descriptor = transport.describe()
    health = transport.health("exact-nonce")
    chunk = transport.infer(request)

    assert descriptor == _descriptor()
    assert health.healthy and health.echo == "exact-nonce"
    assert chunk.request_id == request.request_id
    assert [call["payload"]["type"] for call in calls] == [
        "describe",
        "health",
        "observation",
    ]
    assert all(call["url"] == ENDPOINT + "/" for call in calls)
    assert PROXY_ID not in calls[0]["url"] and PROXY_SECRET not in calls[0]["url"]


def test_modal_http_transport_ignores_ambient_proxy_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    client_options: list[dict[str, Any]] = []
    original_client = httpx.Client

    def recording_client(*args: Any, **kwargs: Any) -> httpx.Client:
        client_options.append(kwargs.copy())
        return original_client(*args, **kwargs)

    monkeypatch.setattr(
        "ctrl_pi.inference_transport.httpx.Client",
        recording_client,
    )
    transport = ModalInferenceTransport(
        ENDPOINT,
        proxy_token_id=PROXY_ID,
        proxy_token_secret=PROXY_SECRET,
        http_transport=httpx.MockTransport(_runtime_http_handler(calls)),
        prefer_websocket=False,
    )

    assert transport.describe() == _descriptor()
    assert client_options and all(options["trust_env"] is False for options in client_options)


def test_large_valid_observation_uses_post_before_opening_websocket() -> None:
    calls: list[dict[str, Any]] = []
    connector_calls = 0

    def connector(*args: Any, **kwargs: Any) -> None:
        nonlocal connector_calls
        del args, kwargs
        connector_calls += 1
        raise AssertionError("an oversized WebSocket frame must not be attempted")

    transport = ModalInferenceTransport(
        ENDPOINT,
        proxy_token_id=PROXY_ID,
        proxy_token_secret=PROXY_SECRET,
        http_transport=httpx.MockTransport(_runtime_http_handler(calls)),
        websocket_connector=connector,
    )
    transport.describe()
    request = _request().model_copy(
        update={
            "images": {
                "observation.images.camera": EncodedImage(
                    data_base64=base64.b64encode(b"x" * 1_600_000).decode("ascii")
                )
            }
        }
    )
    assert len(json.dumps(request.model_dump(mode="json")).encode("utf-8")) > (
        MAX_RUNTIME_WEBSOCKET_MESSAGE_BYTES
    )

    chunk = transport.infer(request)

    assert chunk.request_id == request.request_id
    assert connector_calls == 0
    assert [call["payload"]["type"] for call in calls] == [
        "describe",
        "observation",
    ]


def test_modal_health_before_describe_is_supported_and_safe() -> None:
    calls: list[dict[str, Any]] = []
    transport = ModalInferenceTransport(
        ENDPOINT,
        proxy_token_id=PROXY_ID,
        proxy_token_secret=PROXY_SECRET,
        http_transport=httpx.MockTransport(_runtime_http_handler(calls)),
        prefer_websocket=False,
    )

    health = transport.health("first-health")

    assert health.healthy and health.echo == "first-health"
    assert [call["payload"]["type"] for call in calls] == ["health"]

    with pytest.raises(InferenceTransportProtocolError, match="Describe"):
        transport.infer(_request())
    assert [call["payload"]["type"] for call in calls] == ["health"]


def test_modal_health_and_action_identity_mismatches_are_rejected() -> None:
    descriptor = _descriptor()
    request = _request()
    response_kind = "describe"

    def handler(http_request: httpx.Request) -> httpx.Response:
        nonlocal response_kind
        payload = json.loads(http_request.content)
        if payload["type"] == "describe":
            response_kind = "health"
            return httpx.Response(200, json=descriptor.model_dump(mode="json"))
        if response_kind == "health":
            response_kind = "action"
            return httpx.Response(
                200,
                json={
                    "type": "health",
                    "healthy": True,
                    "echo": payload["nonce"],
                    "runtime": "lerobot",
                    "model_repo": "acme/wrong",
                    "revision": REVISION,
                },
            )
        return httpx.Response(200, json=_action_payload(request, revision="d" * 40))

    transport = ModalInferenceTransport(
        ENDPOINT,
        proxy_token_id=PROXY_ID,
        proxy_token_secret=PROXY_SECRET,
        http_transport=httpx.MockTransport(handler),
        prefer_websocket=False,
    )
    transport.describe()
    with pytest.raises(InferenceTransportProtocolError, match="identity"):
        transport.health("nonce")
    with pytest.raises(InferenceTransportProtocolError, match="descriptor"):
        transport.infer(request)


def test_modal_transport_never_follows_redirects_or_exposes_remote_content() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            307,
            headers={"location": "https://evil.invalid/hf_supersecret"},
            text="hf_raw_remote_secret",
        )

    transport = ModalInferenceTransport(
        ENDPOINT,
        proxy_token_id=PROXY_ID,
        proxy_token_secret=PROXY_SECRET,
        http_transport=httpx.MockTransport(handler),
        prefer_websocket=False,
    )
    with pytest.raises(InferenceTransportError) as captured:
        transport.describe()

    assert calls == 1
    assert "evil" not in str(captured.value)
    assert "supersecret" not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


def test_modal_transport_sanitizes_network_json_and_size_failures() -> None:
    def network_failure(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("proxy ws-test-secret failed")

    failing = ModalInferenceTransport(
        ENDPOINT,
        proxy_token_id=PROXY_ID,
        proxy_token_secret=PROXY_SECRET,
        http_transport=httpx.MockTransport(network_failure),
        prefer_websocket=False,
    )
    with pytest.raises(InferenceTransportError) as captured:
        failing.describe()
    assert "ws-test-secret" not in str(captured.value)
    assert captured.value.__cause__ is captured.value.__context__ is None

    invalid_json = ModalInferenceTransport(
        ENDPOINT,
        proxy_token_id=PROXY_ID,
        proxy_token_secret=PROXY_SECRET,
        http_transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, content=b"not-json")
        ),
        prefer_websocket=False,
    )
    with pytest.raises(InferenceTransportProtocolError, match="invalid"):
        invalid_json.describe()

    too_large = ModalInferenceTransport(
        ENDPOINT,
        proxy_token_id=PROXY_ID,
        proxy_token_secret=PROXY_SECRET,
        http_transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                headers={"content-length": str(MAX_RUNTIME_RESPONSE_BYTES + 1)},
                content=b"{}",
            )
        ),
        prefer_websocket=False,
    )
    with pytest.raises(InferenceTransportProtocolError, match="safe limit"):
        too_large.describe()


@pytest.mark.parametrize(
    "body",
    [
        b'{"type":"runtime","type":"runtime"}',
        b'{"type":"runtime","actions_per_chunk":NaN}',
    ],
)
def test_modal_transport_strictly_rejects_duplicate_and_nonfinite_json(
    body: bytes,
) -> None:
    transport = ModalInferenceTransport(
        ENDPOINT,
        proxy_token_id=PROXY_ID,
        proxy_token_secret=PROXY_SECRET,
        http_transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, content=body)
        ),
        prefer_websocket=False,
    )
    with pytest.raises(InferenceTransportProtocolError, match="invalid"):
        transport.describe()


class _FakeWebSocket:
    def __init__(self, *, fail_send: bool = False, fail_receive: bool = False) -> None:
        self.fail_send = fail_send
        self.fail_receive = fail_receive
        self.sent: list[str] = []
        self.closed = False

    def send(self, payload: str) -> None:
        if self.fail_send:
            raise OSError("ws-secret-send")
        self.sent.append(payload)

    def recv(self, timeout: float) -> str:
        assert timeout > 0
        if self.fail_receive:
            raise OSError("ws-secret-receive")
        request = ObservationRequest.model_validate(json.loads(self.sent[-1]))
        return json.dumps(_action_payload(request))

    def close(self) -> None:
        self.closed = True


def test_modal_websocket_uses_proxy_headers_and_same_envelopes() -> None:
    calls: list[dict[str, Any]] = []
    websocket = _FakeWebSocket()
    connector_calls: list[tuple[str, dict[str, Any]]] = []

    def connector(url: str, **kwargs: Any) -> _FakeWebSocket:
        connector_calls.append((url, kwargs))
        return websocket

    transport = ModalInferenceTransport(
        ENDPOINT,
        proxy_token_id=PROXY_ID,
        proxy_token_secret=PROXY_SECRET,
        http_transport=httpx.MockTransport(_runtime_http_handler(calls)),
        websocket_connector=connector,
    )
    transport.describe()
    request = _request()

    chunk = transport.infer(request)
    transport.close()

    assert chunk.request_id == request.request_id
    assert json.loads(websocket.sent[0])["type"] == "observation"
    url, kwargs = connector_calls[0]
    assert url == "wss://ctrl-pi-runtime--example.modal.run/ws"
    assert kwargs["additional_headers"] == {
        "Modal-Key": PROXY_ID,
        "Modal-Secret": PROXY_SECRET,
    }
    assert kwargs["proxy"] is None and kwargs["compression"] is None
    assert websocket.closed is True
    assert [call["payload"]["type"] for call in calls] == ["describe"]


def test_established_websocket_never_switches_to_post_for_large_observation() -> None:
    calls: list[dict[str, Any]] = []
    websocket = _FakeWebSocket()
    transport = ModalInferenceTransport(
        ENDPOINT,
        proxy_token_id=PROXY_ID,
        proxy_token_secret=PROXY_SECRET,
        http_transport=httpx.MockTransport(_runtime_http_handler(calls)),
        websocket_connector=lambda _url, **_kwargs: websocket,
    )
    transport.describe()
    small_request = _request()

    assert transport.infer(small_request).request_id == small_request.request_id

    large_request = _request(sequence=small_request.sequence + 1).model_copy(
        update={
            "images": {
                "observation.images.camera": EncodedImage(
                    data_base64=base64.b64encode(b"x" * 1_600_000).decode("ascii")
                )
            }
        }
    )
    assert len(json.dumps(large_request.model_dump(mode="json")).encode("utf-8")) > (
        MAX_RUNTIME_WEBSOCKET_MESSAGE_BYTES
    )

    with pytest.raises(InferenceTransportProtocolError, match="active WebSocket"):
        transport.infer(large_request)

    assert len(websocket.sent) == 1
    assert websocket.closed is False
    assert [call["payload"]["type"] for call in calls] == ["describe"]


def test_websocket_falls_back_only_before_an_observation_is_accepted() -> None:
    calls: list[dict[str, Any]] = []

    def failed_connector(_url: str, **_kwargs: Any) -> None:
        raise OSError("connect-secret")

    transport = ModalInferenceTransport(
        ENDPOINT,
        proxy_token_id=PROXY_ID,
        proxy_token_secret=PROXY_SECRET,
        http_transport=httpx.MockTransport(_runtime_http_handler(calls)),
        websocket_connector=failed_connector,
    )
    request = _request()
    transport.describe()
    assert transport.infer(request).request_id == request.request_id
    assert [call["payload"]["type"] for call in calls] == [
        "describe",
        "observation",
    ]

    received_then_failed = _FakeWebSocket(fail_receive=True)
    second_calls: list[dict[str, Any]] = []
    second = ModalInferenceTransport(
        ENDPOINT,
        proxy_token_id=PROXY_ID,
        proxy_token_secret=PROXY_SECRET,
        http_transport=httpx.MockTransport(_runtime_http_handler(second_calls)),
        websocket_connector=lambda _url, **_kwargs: received_then_failed,
    )
    second.describe()
    with pytest.raises(InferenceTransportError, match="after accepting") as captured:
        second.infer(request)
    assert PROXY_SECRET not in str(captured.value)
    assert [call["payload"]["type"] for call in second_calls] == ["describe"]

    ambiguously_failed_send = _FakeWebSocket(fail_send=True)
    third_calls: list[dict[str, Any]] = []
    third = ModalInferenceTransport(
        ENDPOINT,
        proxy_token_id=PROXY_ID,
        proxy_token_secret=PROXY_SECRET,
        http_transport=httpx.MockTransport(_runtime_http_handler(third_calls)),
        websocket_connector=lambda _url, **_kwargs: ambiguously_failed_send,
    )
    third.describe()
    with pytest.raises(InferenceTransportError, match="after accepting"):
        third.infer(request)
    assert [call["payload"]["type"] for call in third_calls] == ["describe"]
