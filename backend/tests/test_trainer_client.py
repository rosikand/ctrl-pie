from __future__ import annotations

import json

import httpx
import pytest

from ctrl_pi.trainer import Client, TrainerClientError

RUN_ID = "11111111-1111-4111-8111-111111111111"


def _run(run_id: str = RUN_ID) -> dict:
    return {
        "id": run_id,
        "name": "ACT demo",
        "status": "running",
        "current_step": 10,
        "dataset_repo": "acme/data",
        "base_model": "lerobot/act-base",
        "runtime": "lerobot",
        "framework": "pytorch",
        "output_model_repo": None,
        "checkpoint_revision": None,
        "config": {},
        "metrics": {},
        "checkpoints": [],
        "created_at": "2026-08-28T00:00:00Z",
        "updated_at": "2026-08-28T00:00:00Z",
    }


def test_client_methods_send_the_documented_requests() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET" and request.url.path == "/api/trainer/runs":
            return httpx.Response(200, json={"runs": [_run()]})
        if request.url.path.endswith("/logs"):
            return httpx.Response(
                201,
                json={
                    "sequence": 1,
                    "source": "stderr",
                    "line": "checkpoint pending",
                    "step": 10,
                    "timestamp": "2026-08-29T00:00:00Z",
                },
            )
        return httpx.Response(200, json=_run())

    transport = httpx.MockTransport(handler)
    with Client("http://ctrl-pi.local/", timeout=3, _transport=transport) as client:
        client.create_run("ACT demo", dataset_repo="acme/data", config={"lr": 1e-4})
        client.list_runs(status="running")
        client.get_run(RUN_ID)
        client.update_run(RUN_ID, status="running", output_model_repo=None)
        client.log_metrics(RUN_ID, step=10, metrics={"loss": 0.5})
        logged = client.log_console(
            RUN_ID,
            source="stderr",
            line="checkpoint pending",
            step=10,
        )
        client.register_checkpoint(
            RUN_ID, repo_id="acme/model", revision="a" * 40, step=10
        )
        assert not client.is_closed
    assert client.is_closed
    assert logged["sequence"] == 1
    assert logged["timestamp"] == "2026-08-29T00:00:00Z"

    assert [(request.method, request.url.path) for request in requests] == [
        ("POST", "/api/trainer/runs"),
        ("GET", "/api/trainer/runs"),
        ("GET", f"/api/trainer/runs/{RUN_ID}"),
        ("PATCH", f"/api/trainer/runs/{RUN_ID}"),
        ("POST", f"/api/trainer/runs/{RUN_ID}/metrics"),
        ("POST", f"/api/trainer/runs/{RUN_ID}/logs"),
        ("POST", f"/api/trainer/runs/{RUN_ID}/checkpoints"),
    ]
    assert dict(requests[1].url.params) == {"status": "running"}
    assert json.loads(requests[3].content) == {
        "status": "running",
        "output_model_repo": None,
    }
    assert json.loads(requests[4].content) == {
        "step": 10,
        "metrics": {"loss": 0.5},
    }
    assert json.loads(requests[5].content) == {
        "source": "stderr",
        "line": "checkpoint pending",
        "step": 10,
    }


def test_client_returns_typed_error_without_echoing_server_body() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            422,
            json={"detail": "invalid hf_super_secret"},
        )
    )
    client = Client("http://ctrl-pi.local", _transport=transport)

    with pytest.raises(TrainerClientError) as caught:
        client.create_run("bad")

    assert caught.value.status_code == 422
    assert str(caught.value) == "The trainer request failed validation."
    assert "hf_super_secret" not in str(caught.value)
    client.close()


def test_client_sanitizes_transport_and_invalid_response_errors() -> None:
    def network_error(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("leaked credential", request=request)

    client = Client(
        "http://ctrl-pi.local",
        _transport=httpx.MockTransport(network_error),
    )
    with pytest.raises(TrainerClientError, match="Could not reach") as network:
        client.get_run(RUN_ID)
    assert "credential" not in str(network.value)
    assert network.value.__cause__ is None
    assert network.value.__context__ is None
    client.close()

    invalid = Client(
        "http://ctrl-pi.local",
        _transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=b"not-json")
        ),
    )
    with pytest.raises(TrainerClientError, match="invalid response") as malformed:
        invalid.get_run(RUN_ID)
    assert malformed.value.__cause__ is None
    assert malformed.value.__context__ is None
    invalid.close()


def test_client_rejects_non_uuid_run_ids_before_request() -> None:
    calls: list[httpx.Request] = []
    client = Client(
        "http://ctrl-pi.local",
        _transport=httpx.MockTransport(
            lambda request: (calls.append(request), httpx.Response(200, json={}))[1]
        ),
    )

    with pytest.raises(TrainerClientError, match="ID is invalid") as caught:
        client.get_run("../models")

    assert calls == []
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    client.close()


@pytest.mark.parametrize(
    "base_url",
    [
        "http://[::1",
        "http://user:supersecret@ctrl-pi.local",
        "ctrl-pi.local:8000",
        "http://ctrl-pi.local?token=supersecret",
    ],
)
def test_client_rejects_unsafe_base_urls_without_raw_exceptions(base_url: str) -> None:
    with pytest.raises(TrainerClientError, match="base URL is invalid") as caught:
        Client(base_url)

    assert "supersecret" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
