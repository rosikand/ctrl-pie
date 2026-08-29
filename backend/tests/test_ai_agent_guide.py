from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re

from ctrl_pi import CtrlPiClient


REPOSITORY_ROOT = Path(__file__).parents[2]
GUIDE = REPOSITORY_ROOT / "docs" / "ai-agent-integration.md"
DOCS_CONFIG = REPOSITORY_ROOT / "docs" / "docs.json"
DEVELOPER_AGENTS = REPOSITORY_ROOT / "AGENTS.md"
DEVELOPER_AGENTS_SHA256 = (
    "8ef93d21b1f5be5fec1861a87236d750a32bfd02cc6874642c620874f76edd5f"
)
FENCE = re.compile(
    r"^```(?P<language>[A-Za-z0-9_-]+)\s*\n(?P<body>.*?)^```\s*$",
    re.MULTILINE | re.DOTALL,
)


def _page_references(value: object) -> list[str]:
    references: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "pages" and isinstance(child, list):
                references.extend(item for item in child if isinstance(item, str))
            references.extend(_page_references(child))
    elif isinstance(value, list):
        for child in value:
            references.extend(_page_references(child))
    return references


def _guide_source() -> str:
    assert GUIDE.is_file(), "the user-facing AI-agent integration guide is required"
    return GUIDE.read_text(encoding="utf-8")


def _assert_terms(source: str, terms: tuple[str, ...], contract: str) -> None:
    normalized = re.sub(r"\s+", " ", source).casefold()
    missing = [
        term
        for term in terms
        if re.sub(r"\s+", " ", term).casefold() not in normalized
    ]
    assert missing == [], f"{contract} is missing: {', '.join(missing)}"


def test_ai_agent_guide_is_a_navigated_user_document_without_replacing_agents() -> None:
    source = _guide_source()
    config = json.loads(DOCS_CONFIG.read_text(encoding="utf-8"))
    references = _page_references(config["navigation"])

    assert references.count("ai-agent-integration") == 1
    assert re.match(
        r'\A---\n(?=[\s\S]*^title:\s*["\'].+["\']\s*$)'
        r'(?=[\s\S]*^description:\s*["\'].+["\']\s*$)[\s\S]*?\n---\n',
        source,
        flags=re.MULTILINE,
    )
    assert hashlib.sha256(DEVELOPER_AGENTS.read_bytes()).hexdigest() == (
        DEVELOPER_AGENTS_SHA256
    ), "ROS-56 must not replace or rewrite the repository developer AGENTS.md"
    _assert_terms(
        source,
        (
            "your AI or automation agent",
            "AGENTS.md",
            "developer",
            "no agent-specific",
        ),
        "agent/developer boundary",
    )


def test_ai_agent_python_examples_parse_and_use_only_the_public_client() -> None:
    source = _guide_source()
    python_blocks = [
        match.group("body")
        for match in FENCE.finditer(source)
        if match.group("language").casefold() in {"python", "py"}
    ]
    assert len(python_blocks) >= 2

    public_methods = {
        name
        for name, value in vars(CtrlPiClient).items()
        if callable(value) and not name.startswith("_")
    }
    observed_methods: set[str] = set()
    forbidden_imports = {
        "can",
        "huggingface_hub",
        "httpx",
        "lerobot",
        "modal",
        "playwright",
        "requests",
        "selenium",
    }
    has_context_manager = False
    has_finally = False
    polling_loops: list[ast.While] = []
    for index, block in enumerate(python_blocks, start=1):
        tree = ast.parse(block, filename=f"docs/ai-agent-integration.md:{index}")
        has_context_manager = has_context_manager or any(
            isinstance(node, (ast.With, ast.AsyncWith)) for node in ast.walk(tree)
        )
        has_finally = has_finally or any(
            isinstance(node, ast.Try) and bool(node.finalbody) for node in ast.walk(tree)
        )
        polling_loops.extend(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.While)
            and any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and isinstance(child.func.value, ast.Name)
                and child.func.value.id in {"ctrl", "client"}
                for child in ast.walk(node)
            )
        )
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                    "ctrl_pi."
                ):
                    raise AssertionError(
                        "agent examples must not import ctrl-pi backend internals"
                    )
                if isinstance(node, ast.Import) and any(
                    alias.name.startswith("ctrl_pi.") for alias in node.names
                ):
                    raise AssertionError(
                        "agent examples must not import ctrl-pi backend internals"
                    )
                names = (
                    [alias.name.split(".", 1)[0] for alias in node.names]
                    if isinstance(node, ast.Import)
                    else [(node.module or "").split(".", 1)[0]]
                )
                assert forbidden_imports.isdisjoint(names), (
                    "agent examples must not bypass ctrl-pi through hardware, "
                    "provider, browser, or raw HTTP libraries"
                )
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in {"ctrl", "client"}
            ):
                observed_methods.add(node.func.attr)
                assert node.func.attr in public_methods
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg in {
                        "acknowledge_automatic_motion_risk",
                        "acknowledge_compute_cost",
                        "acknowledge_hardware_motion_risk",
                        "acknowledge_public_model_risk",
                    }:
                        assert not (
                            isinstance(keyword.value, ast.Constant)
                            and keyword.value.value is True
                        ), "an agent example must not synthesize a human risk acknowledgement"

    assert has_context_manager
    assert has_finally
    for loop in polling_loops:
        calls = {
            child.func.attr
            for child in ast.walk(loop)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
        }
        assert {"monotonic", "sleep"}.issubset(calls), (
            "agent polling must use a monotonic deadline and bounded backoff"
        )
    assert {
        "get_system_status",
        "get_yam_setup",
        "get_recording_state",
        "stop_episode",
        "stop_teleop",
        "get_managed_training_job",
        "cancel_managed_training_job",
        "list_deployments",
        "get_inference_state",
        "stop_inference",
    }.issubset(observed_methods)


def test_ai_agent_guide_documents_current_sdk_and_rest_boundaries() -> None:
    source = _guide_source()
    _assert_terms(
        source,
        (
            "CtrlPiClient",
            "documented REST API",
            "/api/health",
            "/api/settings/status",
            "/api/yam/setup",
            "/api/arms",
            "/api/recordings",
            "/api/datasets",
            "/api/models",
            "/api/trainer/runs",
            "/api/trainer/jobs",
            "/api/inference/deployments",
        ),
        "public SDK/REST map",
    )
    _assert_terms(
        source,
        (
            "get_system_status",
            "discover_yam",
            "preflight_yam",
            "connect_yam",
            "jog_arm",
            "get_recording_state",
            "list_datasets",
            "list_models",
            "launch_managed_training",
            "get_managed_training_job",
            "cancel_managed_training_job",
            "list_deployments",
            "get_inference_state",
            "stop_inference",
        ),
        "current typed client surface",
    )


def test_ai_agent_guide_requires_state_reconciliation_and_verified_cleanup() -> None:
    source = _guide_source()
    _assert_terms(
        source,
        (
            "Read current state",
            "blindly retry",
            "finally",
            "Closing `CtrlPiClient`",
            "does not cancel managed training",
            "episode_active",
            "teleop_active",
            "stop_episode",
            "stop_teleop",
            "idempotency_key",
            "persisted idempotency UUID",
            "identical payload",
            "server compares",
            "stored canonical request hash",
            "get_managed_training_job",
            "cancel_managed_training_job",
            "list_deployments",
            "stop_inference",
            "teardown_verified",
            "zero running tasks",
            "monotonic",
            "bounded",
            "make modal-panic",
        ),
        "failure reconciliation and cleanup contract",
    )


def test_ai_agent_guide_enforces_human_safety_credentials_and_prohibitions() -> None:
    source = _guide_source()
    _assert_terms(
        source,
        (
            "mock mode first",
            "fresh human approval",
            "never synthesize",
            "emergency stop",
            "workspace",
            "physical motion",
            "auto-restore",
            "paid cloud",
            "public",
            "reset_yam_setup",
            "panic",
            "trusted",
            "LAN",
            "no authentication",
            "HF_TOKEN",
            "DATABASE_URL",
            "Modal credentials",
            "prompt",
            "log",
            "browser automation",
            "backend internals",
            "direct CAN",
            "raw action",
            "Hugging Face/Modal SDKs",
        ),
        "human/safety/credential boundary",
    )
    _assert_terms(
        source,
        (
            "calibration readiness",
            "does not prove",
            "mock",
            "claim",
            "Ubuntu/YAM",
        ),
        "physical-validation truth boundary",
    )

    prohibited = source.split("## Prohibited behavior", 1)
    assert len(prohibited) == 2
    prohibited_body = prohibited[1].split("\n## ", 1)[0]
    _assert_terms(
        prohibited_body,
        (
            ".env",
            "query PostgreSQL directly",
            "Hugging Face/Modal SDKs",
            "YAMDriver",
            "open direct CAN/serial devices",
            "raw actions",
            "browser automation",
            "_transport",
            "acknowledgement",
            "concurrent robot-control",
            "teardown_verified",
            "claim that mock mode",
        ),
        "explicitly prohibited behavior",
    )


def test_ai_agent_guide_treats_retrieved_content_as_untrusted_and_avoids_heuristics() -> None:
    source = _guide_source()
    _assert_terms(
        source,
        (
            "untrusted data",
            "model and dataset cards or metadata",
            "task",
            "notes",
            "console",
            "diagnostic",
            "API error",
            "not instructions",
            "authorize",
            "ambiguous",
            "another operator",
            "ask the human",
        ),
        "untrusted-content and ambiguous-ownership boundary",
    )
