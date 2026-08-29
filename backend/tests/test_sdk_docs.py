from __future__ import annotations

import ast
from pathlib import Path

from ctrl_pi import CtrlPiClient


DOCUMENT = Path(__file__).parents[2] / "docs" / "python-sdk.md"


def test_python_sdk_examples_are_syntactically_valid() -> None:
    source = DOCUMENT.read_text(encoding="utf-8")
    blocks = source.split("```python\n")[1:]
    assert len(blocks) >= 5
    for index, block in enumerate(blocks):
        ast.parse(block.split("```", 1)[0], filename=f"python-sdk.md:{index}")


def test_python_sdk_guide_documents_every_public_method_and_stream_boundary() -> None:
    source = DOCUMENT.read_text(encoding="utf-8")
    public_methods = {
        name
        for name, value in vars(CtrlPiClient).items()
        if callable(value) and not name.startswith("_") and name not in {"close"}
    }
    assert public_methods <= set(source.replace("`", "").split()) or all(
        method in source for method in public_methods
    )
    for required in (
        "64 MiB",
        "does **not** stop",
        "acknowledge_hardware_motion_risk",
        "acknowledge_automatic_motion_risk",
        "private=True",
        "mock mode preserves any physical row",
        "nested lists and dictionaries",
        "WS /ws/arms",
        "WS /api/inference/deployments/{id}/stream",
        "GET /api/camera/stream",
        "video_url",
        "teardown_verified",
        "launch_managed_training",
        "cancel_managed_training_job",
        "Only one managed job may be nonterminal",
        "Closing the client does not cancel a job.",
    ):
        assert required in source
