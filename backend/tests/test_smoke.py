from __future__ import annotations

import json
from pathlib import Path

import pytest

from ctrl_pi import smoke


@pytest.mark.asyncio
async def test_full_smoke_loop_uses_fake_hub_offline(tmp_path: Path) -> None:
    messages: list[str] = []

    evidence = await smoke.run_smoke(
        fake_hub=True,
        temp_parent=tmp_path,
        output=messages.append,
    )

    assert evidence.hub_mode == "fake"
    assert evidence.mock_arm_ids == (
        "yam-leader",
        "yam-follower",
        "yam-leader-left",
        "yam-follower-left",
    )
    assert evidence.mock_pair_ids == ("left", "right")
    assert evidence.teleop_sync_seconds >= 2.7
    assert evidence.private is True
    assert evidence.recording_wall_seconds >= 5.0
    assert evidence.episode_duration_seconds >= 5.0
    assert evidence.episode_samples >= 50
    assert evidence.video_bytes > 0
    assert evidence.listed_in_datasets is True
    assert evidence.listed_episodes == 1
    assert evidence.listed_frames == evidence.episode_samples
    assert evidence.deployment_status == "stopped"
    assert evidence.action_steps == 100
    assert evidence.recording_tasks == 0
    assert evidence.recording_episodes == 0
    assert evidence.inference_tasks == 0
    assert evidence.inference_queue_depth == 0
    assert evidence.active_compute_resources == 0
    assert evidence.rig_released is True
    assert evidence.upload_released is True
    assert evidence.hub_repo_deleted is True
    assert evidence.workspace_removed is True
    assert list(tmp_path.iterdir()) == []

    report = "\n".join(messages)
    for stage in (
        "topology",
        "record",
        "upload",
        "datasets",
        "deploy",
        "inference",
        "teardown",
    ):
        assert f"[smoke] {stage}: PASS" in report
    assert "[smoke] PASS" in report
    assert "arms=4 pairs=2 group=bimanual mode=mock" in report
    assert "sync=" in report
    assert "steps=100" in report
    assert smoke._FAKE_TOKEN not in report


def test_cli_defaults_to_real_hub_and_requires_explicit_fake_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []

    async def fake_run_smoke(*, fake_hub: bool = False, **_kwargs):
        calls.append(fake_hub)

    monkeypatch.setattr(smoke, "run_smoke", fake_run_smoke)

    assert smoke.main([]) == 0
    assert smoke.main(["--fake-hub"]) == 0
    assert calls == [False, True]


def test_cleanup_refuses_repository_with_different_recording_marker(
    tmp_path: Path,
) -> None:
    api = smoke._LocalHubApi(
        tmp_path / "hub",
        namespace=smoke._FAKE_NAMESPACE,
        token=smoke._FAKE_TOKEN,
    )
    repo_id = f"{smoke._FAKE_NAMESPACE}/ctrl-pi-smoke-marker-check"
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=True,
        exist_ok=False,
    )
    marker = tmp_path / "marker.json"
    marker.write_text(
        json.dumps(
            {
                "schema": "ctrl-pi.recording-dataset",
                "version": 1,
                "recording_id": "different-recording",
            }
        ),
        encoding="utf-8",
    )
    api.upload_file(
        path_or_fileobj=marker,
        path_in_repo=".ctrl-pi.json",
        repo_id=repo_id,
        repo_type="dataset",
        token=smoke._FAKE_TOKEN,
        commit_message="wrong owner",
    )
    uploader = smoke.HFDatasetUploader(
        tmp_path / "recordings",
        hub_api_factory=api.factory,
    )

    with pytest.raises(smoke.SmokeFailure, match="ownership marker"):
        smoke._safe_delete_dataset(
            api=api,
            uploader=uploader,
            repo_id=repo_id,
            recording_id="expected-recording",
            token=smoke._FAKE_TOKEN,
            expected_revision=None,
        )

    assert api.repo_exists(
        repo_id=repo_id,
        repo_type="dataset",
        token=smoke._FAKE_TOKEN,
    )
