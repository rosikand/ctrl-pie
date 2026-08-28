from __future__ import annotations

import json
import re
import threading
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import SecretStr, ValidationError

from ctrl_pi.inference_runtime import (
    ActionChunk,
    DescribeRequest,
    HealthRequest,
    InferenceRuntime,
    ObservationRequest,
    RuntimeDescriptor,
    RuntimeHealth,
    RuntimeProtocolError,
    validate_action_chunk,
)

MAX_RUNTIME_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_RUNTIME_WEBSOCKET_MESSAGE_BYTES = 2 * 1024 * 1024
_MODAL_ENDPOINT_HOST = re.compile(
    r"(?=.{1,253}\Z)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+modal\.run"
)


class InferenceTransportError(RuntimeError):
    """A safe robot-to-runtime transport error."""


class InferenceTransportConfigurationError(InferenceTransportError):
    """The backend-only runtime transport is not configured safely."""


class InferenceTransportProtocolError(InferenceTransportError):
    """The remote runtime returned a malformed or mismatched response."""


@runtime_checkable
class InferenceTransport(Protocol):
    def health(self, nonce: str) -> RuntimeHealth: ...

    def describe(self) -> RuntimeDescriptor: ...

    def infer(self, request: ObservationRequest) -> ActionChunk: ...

    def close(self) -> None: ...


class InProcessInferenceTransport:
    """Credential-free transport for a loaded in-process mock runtime."""

    def __init__(self, runtime: InferenceRuntime) -> None:
        self._runtime = runtime
        self._closed = False
        self._session_started = False
        self._lock = threading.RLock()

    def health(self, nonce: str) -> RuntimeHealth:
        request = HealthRequest(nonce=nonce)
        with self._lock:
            self._require_open()
            descriptor = self._runtime.describe()
            return RuntimeHealth(
                healthy=True,
                echo=request.nonce,
                runtime=descriptor.runtime,
                model_repo=descriptor.model_repo,
                revision=descriptor.revision,
            )

    def describe(self) -> RuntimeDescriptor:
        with self._lock:
            self._require_open()
            return self._runtime.describe()

    def infer(self, request: ObservationRequest) -> ActionChunk:
        with self._lock:
            self._require_open()
            if not self._session_started:
                self._runtime.reset_session()
                self._session_started = True
            return self._runtime.predict(request)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._runtime.close()

    def _require_open(self) -> None:
        if self._closed:
            raise InferenceTransportError("The inference transport is closed.")


class ModalInferenceTransport:
    """Strict proxy-authenticated HTTP/WS transport for one Modal runtime URL."""

    def __init__(
        self,
        endpoint_url: str,
        *,
        proxy_token_id: SecretStr | str | None,
        proxy_token_secret: SecretStr | str | None,
        timeout_seconds: float = 60.0,
        http_transport: httpx.BaseTransport | None = None,
        websocket_connector: Any | None = None,
        prefer_websocket: bool = True,
    ) -> None:
        self._endpoint_url, self._websocket_url = self._validated_urls(endpoint_url)
        token_id = self._unwrap_secret(proxy_token_id)
        token_secret = self._unwrap_secret(proxy_token_secret)
        if (token_id is None) != (token_secret is None):
            raise InferenceTransportConfigurationError(
                "Modal proxy credentials must be configured as a complete pair."
            )
        if token_id is None or token_secret is None:
            raise InferenceTransportConfigurationError(
                "Modal proxy credentials are required for runtime inference."
            )
        if not self._valid_proxy_token(token_id, "wk-") or not self._valid_proxy_token(
            token_secret, "ws-"
        ):
            raise InferenceTransportConfigurationError(
                "Modal proxy credentials are invalid."
            )
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not 0 < float(timeout_seconds) <= 180
        ):
            raise InferenceTransportConfigurationError(
                "Inference timeout must be between 0 and 180 seconds."
            )
        self._proxy_token_id = SecretStr(token_id)
        self._proxy_token_secret = SecretStr(token_secret)
        self._timeout_seconds = float(timeout_seconds)
        self._http_transport = http_transport
        self._websocket_connector = websocket_connector
        self._prefer_websocket = prefer_websocket
        self._websocket: Any | None = None
        self._websocket_disabled = not prefer_websocket
        self._websocket_action_received = False
        self._closed = False
        self._descriptor: RuntimeDescriptor | None = None
        self._lock = threading.RLock()

    def health(self, nonce: str) -> RuntimeHealth:
        request = HealthRequest(nonce=nonce)
        payload = self._post(request.model_dump(mode="json"))
        health = self._parse(RuntimeHealth, payload)
        descriptor = self._descriptor
        if descriptor is not None and (
            health.runtime != descriptor.runtime
            or health.model_repo != descriptor.model_repo
            or health.revision != descriptor.revision
        ):
            raise InferenceTransportProtocolError(
                "The remote runtime identity does not match its descriptor."
            )
        return health

    def describe(self) -> RuntimeDescriptor:
        request = DescribeRequest()
        payload = self._post(request.model_dump(mode="json"))
        descriptor = self._parse(RuntimeDescriptor, payload)
        self._descriptor = descriptor
        return descriptor

    def infer(self, request: ObservationRequest) -> ActionChunk:
        payload = request.model_dump(mode="json")
        with self._lock:
            self._require_open()
            if self._descriptor is None:
                raise InferenceTransportProtocolError(
                    "Describe the remote runtime before requesting inference."
                )
            if not self._websocket_disabled:
                chunk = self._infer_websocket(payload)
                if chunk is not None:
                    return self._validated_chunk(request, chunk)
            response = self._post_locked(payload)
        return self._validated_chunk(request, self._parse(ActionChunk, response))

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            websocket = self._websocket
            self._websocket = None
            if websocket is not None:
                try:
                    websocket.close()
                except Exception:
                    pass

    def _post(self, payload: dict[str, Any]) -> Any:
        with self._lock:
            self._require_open()
            return self._post_locked(payload)

    def _post_locked(self, payload: dict[str, Any]) -> Any:
        response_payload: bytes | None = None
        response_status: int | None = None
        response_redirect = False
        failed = False
        try:
            with httpx.Client(
                transport=self._http_transport,
                timeout=self._timeout_seconds,
                follow_redirects=False,
                trust_env=False,
            ) as client:
                with client.stream(
                    "POST",
                    self._endpoint_url,
                    json=payload,
                    headers=self._headers(),
                ) as response:
                    response_status = response.status_code
                    response_redirect = response.is_redirect
                    content_length = response.headers.get("content-length")
                    if content_length is not None:
                        try:
                            declared_length = int(content_length)
                        except ValueError:
                            declared_length = -1
                        if (
                            declared_length < 0
                            or declared_length > MAX_RUNTIME_RESPONSE_BYTES
                        ):
                            raise InferenceTransportProtocolError(
                                "The remote inference response exceeds the safe limit."
                            )
                    body = bytearray()
                    for chunk in response.iter_bytes():
                        body.extend(chunk)
                        if len(body) > MAX_RUNTIME_RESPONSE_BYTES:
                            raise InferenceTransportProtocolError(
                                "The remote inference response exceeds the safe limit."
                            )
                    response_payload = bytes(body)
        except InferenceTransportProtocolError:
            raise
        except Exception:
            failed = True
        if failed or response_payload is None or response_status is None:
            raise InferenceTransportError(
                "The remote inference endpoint could not be reached."
            )
        if response_redirect or response_status != 200:
            raise InferenceTransportError(
                "The remote inference endpoint rejected the request."
            )
        parsed: Any = None
        invalid = False
        try:
            parsed = self._strict_json_loads(response_payload)
        except Exception:
            invalid = True
        if invalid:
            raise InferenceTransportProtocolError(
                "The remote inference response is invalid."
            )
        return parsed

    def _infer_websocket(self, payload: dict[str, Any]) -> ActionChunk | None:
        serialized = json.dumps(payload, separators=(",", ":"))
        if len(serialized.encode("utf-8")) > MAX_RUNTIME_WEBSOCKET_MESSAGE_BYTES:
            # Modal accepts larger POST bodies than RFC6455 messages. This is
            # known before the first observation is committed, so POST is safe.
            # Once a WebSocket action has been returned, switching transports
            # would split one stateful policy session across two connections.
            if self._websocket_action_received:
                raise InferenceTransportProtocolError(
                    "The observation exceeds the active WebSocket session limit."
                )
            self._discard_websocket()
            self._websocket_disabled = True
            return None
        if self._websocket is None:
            connected = self._connect_websocket()
            if not connected:
                self._websocket_disabled = True
                return None
        websocket = self._websocket
        committed = False
        raw: object | None = None
        failed = False
        try:
            # Once a socket exists, send failure is ambiguous: the server may
            # have accepted the bytes before the client observed the error.
            # Never replay that stateful observation over HTTP.
            committed = True
            websocket.send(serialized)
            raw = websocket.recv(timeout=self._timeout_seconds)
        except Exception:
            failed = True
        if failed:
            self._discard_websocket()
            if not committed:
                self._websocket_disabled = True
                return None
            raise InferenceTransportError(
                "The remote inference WebSocket failed after accepting an observation."
            )
        if not isinstance(raw, str) or len(raw.encode("utf-8")) > MAX_RUNTIME_RESPONSE_BYTES:
            raise InferenceTransportProtocolError(
                "The remote inference response exceeds the safe limit."
            )
        decoded: object | None = None
        invalid = False
        try:
            decoded = self._strict_json_loads(raw)
        except (TypeError, ValueError):
            invalid = True
        if invalid:
            self._discard_websocket()
            raise InferenceTransportProtocolError(
                "The remote inference response is invalid."
            )
        chunk = self._parse(ActionChunk, decoded)
        self._websocket_action_received = True
        return chunk

    def _connect_websocket(self) -> bool:
        connector = self._websocket_connector
        if connector is None:
            try:
                from websockets.sync.client import connect

                connector = connect
            except Exception:
                return False
        websocket: Any | None = None
        failed = False
        try:
            websocket = connector(
                self._websocket_url,
                additional_headers=self._headers(),
                open_timeout=self._timeout_seconds,
                close_timeout=min(self._timeout_seconds, 10.0),
                max_size=MAX_RUNTIME_RESPONSE_BYTES,
                max_queue=1,
                compression=None,
                proxy=None,
            )
        except Exception:
            failed = True
        if failed or websocket is None:
            return False
        self._websocket = websocket
        return True

    def _discard_websocket(self) -> None:
        websocket = self._websocket
        self._websocket = None
        if websocket is not None:
            try:
                websocket.close()
            except Exception:
                pass

    def _headers(self) -> dict[str, str]:
        return {
            "Modal-Key": self._proxy_token_id.get_secret_value(),
            "Modal-Secret": self._proxy_token_secret.get_secret_value(),
        }

    def _require_open(self) -> None:
        if self._closed:
            raise InferenceTransportError("The inference transport is closed.")

    def _validated_chunk(
        self,
        request: ObservationRequest,
        chunk: ActionChunk,
    ) -> ActionChunk:
        descriptor = self._descriptor
        if descriptor is None:
            raise InferenceTransportProtocolError(
                "Describe the remote runtime before requesting inference."
            )
        failed = False
        try:
            validate_action_chunk(request, chunk, descriptor)
        except RuntimeProtocolError:
            failed = True
        if failed:
            raise InferenceTransportProtocolError(
                "The remote action chunk violates its runtime descriptor."
            )
        return chunk

    @staticmethod
    def _parse(model: Any, payload: Any) -> Any:
        parsed: Any = None
        failed = False
        try:
            parsed = model.model_validate(payload)
        except (ValidationError, TypeError, ValueError):
            failed = True
        if failed:
            raise InferenceTransportProtocolError(
                "The remote inference response is invalid."
            )
        return parsed

    @staticmethod
    def _unwrap_secret(value: SecretStr | str | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, SecretStr):
            return value.get_secret_value()
        if isinstance(value, str):
            return value
        raise InferenceTransportConfigurationError(
            "Modal proxy credentials are invalid."
        )

    @staticmethod
    def _valid_proxy_token(value: str, prefix: str) -> bool:
        return (
            value.startswith(prefix)
            and 4 <= len(value) <= 256
            and not any(character.isspace() or ord(character) < 32 for character in value)
        )

    @staticmethod
    def _strict_json_loads(value: bytes | str) -> Any:
        def reject_constant(_value: str) -> None:
            raise ValueError("non-finite JSON constant")

        def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, item in pairs:
                if key in result:
                    raise ValueError("duplicate JSON key")
                result[key] = item
            return result

        return json.loads(
            value,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )

    @staticmethod
    def _validated_urls(endpoint_url: str) -> tuple[str, str]:
        invalid = False
        try:
            parsed = urlsplit(endpoint_url)
            parsed_port = None if parsed is None else parsed.port
        except (TypeError, ValueError):
            parsed = None
            parsed_port = None
            invalid = True
        if (
            invalid
            or parsed is None
            or parsed.scheme != "https"
            or parsed.hostname is None
            or _MODAL_ENDPOINT_HOST.fullmatch(parsed.hostname) is None
            or parsed_port is not None
            or parsed.netloc.endswith(":")
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise InferenceTransportConfigurationError(
                "The Modal runtime endpoint URL is invalid."
            )
        root_path = "/"
        http_url = urlunsplit(("https", parsed.netloc, root_path, "", ""))
        websocket_path = "/ws"
        websocket_url = urlunsplit(("wss", parsed.netloc, websocket_path, "", ""))
        return http_url, websocket_url
