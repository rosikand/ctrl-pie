from __future__ import annotations

import math
from types import TracebackType
from typing import Any, Protocol

import httpx


MAX_JSON_RESPONSE_BYTES = 64 * 1024 * 1024


class _ClientError(Protocol):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
    ) -> None: ...


class SafeHttpClient:
    """Private synchronous JSON transport shared by the public SDK clients."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float,
        error_type: type[_ClientError],
        _transport: httpx.BaseTransport | None = None,
    ) -> None:
        parsed_url: httpx.URL | None = None
        try:
            parsed_url = httpx.URL(base_url)
        except Exception:
            pass
        if (
            parsed_url is None
            or parsed_url.scheme not in ("http", "https")
            or not parsed_url.host
            or bool(parsed_url.userinfo)
            or bool(parsed_url.query)
            or bool(parsed_url.fragment)
            or parsed_url.path not in ("", "/")
        ):
            raise error_type("ctrl-pi base URL is invalid.") from None
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise error_type("ctrl-pi client timeout is invalid.") from None

        client: httpx.Client | None = None
        try:
            client = httpx.Client(
                base_url=str(parsed_url).rstrip("/"),
                timeout=timeout,
                transport=_transport,
                headers={"Accept": "application/json"},
                follow_redirects=False,
                trust_env=False,
            )
        except Exception:
            pass
        if client is None:
            raise error_type("ctrl-pi client configuration is invalid.") from None
        self._client = client
        self._error_type = error_type

    @property
    def is_closed(self) -> bool:
        return self._client.is_closed

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> SafeHttpClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any = None,
        json_supplied: bool = False,
        status_messages: dict[int, str] | None = None,
    ) -> bytes:
        request_kwargs: dict[str, Any] = {}
        if params:
            request_kwargs["params"] = params
        if json_supplied:
            request_kwargs["json"] = json

        status_code: int | None = None
        invalid_response = False
        oversized_response = False
        transport_failed = False
        body: bytes | None = None
        try:
            with self._client.stream(method, path, **request_kwargs) as response:
                if not 200 <= response.status_code < 300:
                    status_code = response.status_code
                else:
                    content_length = response.headers.get("content-length")
                    declared_length: int | None = None
                    if content_length is not None:
                        try:
                            declared_length = int(content_length)
                        except ValueError:
                            invalid_response = True
                    if declared_length is not None and (
                        declared_length < 0
                        or declared_length > MAX_JSON_RESPONSE_BYTES
                    ):
                        oversized_response = True

                    if not invalid_response and not oversized_response:
                        buffer = bytearray()
                        for chunk in response.iter_bytes():
                            if len(buffer) + len(chunk) > MAX_JSON_RESPONSE_BYTES:
                                oversized_response = True
                                break
                            buffer.extend(chunk)
                        if not oversized_response:
                            body = bytes(buffer)
        except Exception:
            transport_failed = True

        if transport_failed:
            raise self._error_type("Could not reach the ctrl-pi API.") from None
        if status_code is not None:
            raise self._error_type(
                (status_messages or {}).get(
                    status_code, self._status_message(status_code)
                ),
                status_code=status_code,
            ) from None
        if oversized_response:
            raise self._error_type(
                "ctrl-pi returned a response that is too large."
            ) from None
        if invalid_response or body is None:
            raise self._error_type("ctrl-pi returned an invalid response.") from None
        return body

    @staticmethod
    def _status_message(status_code: int) -> str:
        messages = {
            400: "The ctrl-pi API rejected the request.",
            403: "The ctrl-pi API denied the request.",
            404: "The requested ctrl-pi resource was not found.",
            409: "The ctrl-pi operation conflicts with current state.",
            422: "The ctrl-pi request failed validation.",
            429: "The ctrl-pi API is temporarily rate limited.",
            502: "A ctrl-pi upstream service is unavailable.",
            503: "The ctrl-pi service or configuration is unavailable.",
        }
        return messages.get(status_code, "The ctrl-pi API request failed.")
