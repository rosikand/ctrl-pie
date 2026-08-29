from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

from PIL import Image, ImageChops


REPOSITORY_ROOT = Path(__file__).parents[2]
DOCS_ROOT = REPOSITORY_ROOT / "docs"
README = REPOSITORY_ROOT / "README.md"
SCREENSHOTS = {
    "ctrl-pi-arms.png": ("arms", "joint", "gripper"),
    "ctrl-pi-record.png": ("record", "session", "camera"),
    "ctrl-pi-datasets.png": ("datasets", "cards", "episode"),
    "ctrl-pi-training.png": ("training", "run", "metric"),
    "ctrl-pi-models.png": ("models", "model", "revision"),
    "ctrl-pi-inference.png": ("inference", "deployment", "robot"),
    "ctrl-pi-yam-setup.png": ("settings", "yam", "readiness"),
}

_MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\((?P<target><[^>]+>|[^\s)]+)")
_FENCE = re.compile(
    r"^```(?P<language>[A-Za-z0-9_-]+)\s*\n(?P<body>.*?)^```\s*$",
    re.MULTILINE | re.DOTALL,
)
_SECRET_LITERALS = (
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\b(?:ak|as|wk|ws)-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


def _markdown_files() -> list[Path]:
    assert README.is_file(), "the top-level README is required"
    files = [
        README,
        *sorted(DOCS_ROOT.rglob("*.md")),
        *sorted(DOCS_ROOT.rglob("*.mdx")),
    ]
    assert len(files) >= 11
    return files


def test_mintlify_configuration_frontmatter_navigation_and_links_validate() -> None:
    result = subprocess.run(
        [sys.executable, str(REPOSITORY_ROOT / "scripts" / "validate_docs.py")],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def _relative_target(document: Path, raw_target: str) -> Path | None:
    target = raw_target[1:-1] if raw_target.startswith("<") else raw_target
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc or not parsed.path or parsed.path.startswith("/"):
        return None
    return (document.parent / unquote(parsed.path)).resolve()


def test_documentation_relative_links_resolve() -> None:
    failures: list[str] = []
    for document in _markdown_files():
        source = document.read_text(encoding="utf-8")
        for match in _MARKDOWN_LINK.finditer(source):
            target = _relative_target(document, match.group("target"))
            if target is not None and not target.exists():
                failures.append(
                    f"{document.relative_to(REPOSITORY_ROOT)} -> "
                    f"{match.group('target')}"
                )
    assert failures == []


def test_screenshot_gallery_assets_are_real_and_version_control_visible() -> None:
    gallery = (DOCS_ROOT / "screenshots.mdx").read_text(encoding="utf-8")
    image_tags = re.findall(r"<img\b[^>]*>", gallery, flags=re.DOTALL)
    referenced_assets: set[str] = set()
    for tag in image_tags:
        source_match = re.search(r'\bsrc="/assets/(?P<source>[^"]+)"', tag)
        alt_match = re.search(r'\balt="(?P<alt>[^"]+)"', tag)
        assert source_match is not None, f"gallery image is missing a docs asset: {tag}"
        assert alt_match is not None, f"gallery image is missing alt text: {tag}"

        source = source_match.group("source")
        assert source not in referenced_assets, f"duplicate gallery image: {source}"
        referenced_assets.add(source)
        required_alt_words = SCREENSHOTS.get(source)
        assert required_alt_words is not None, f"unexpected gallery image: {source}"
        alt = alt_match.group("alt").casefold()
        assert all(word in alt for word in required_alt_words), (
            f"gallery alt text for {source} does not describe the visible product state"
        )

    assert referenced_assets == set(SCREENSHOTS)
    screenshot_paths = [DOCS_ROOT / "assets" / name for name in SCREENSHOTS]
    for screenshot in screenshot_paths:
        assert screenshot.is_file()
        assert screenshot.stat().st_size >= 50_000
        with Image.open(screenshot) as image:
            image.verify()
        with Image.open(screenshot) as image:
            assert image.format == "PNG"
            assert image.width >= 1_200
            assert image.height >= 700
            rgb = image.convert("RGB")
            extrema = rgb.getextrema()
            assert all(high - low >= 32 for low, high in extrema)
            background = Image.new("RGB", rgb.size, rgb.getpixel((0, 0)))
            content_bounds = ImageChops.difference(rgb, background).getbbox()
            assert content_bounds is not None
            assert content_bounds[1] < image.height // 4

    relative_paths = [
        screenshot.relative_to(REPOSITORY_ROOT).as_posix()
        for screenshot in screenshot_paths
    ]
    visible = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            *relative_paths,
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert set(relative_paths).issubset(visible), (
        "one or more gallery screenshots are ignored and would not be committed"
    )
    assert "docs/assets/ctrl-pi-inference.png" in README.read_text(encoding="utf-8")
    assert "assets/ctrl-pi-inference.png" in (
        DOCS_ROOT / "inference.md"
    ).read_text(encoding="utf-8")


def test_documentation_shell_json_and_python_examples_parse() -> None:
    failures: list[str] = []
    for document in _markdown_files():
        source = document.read_text(encoding="utf-8")
        for index, match in enumerate(_FENCE.finditer(source), start=1):
            language = match.group("language").casefold()
            body = match.group("body")
            if language in {"bash", "sh"}:
                result = subprocess.run(
                    ["bash", "-n"],
                    input=body,
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    failures.append(
                        f"{document.relative_to(REPOSITORY_ROOT)} "
                        f"shell block {index}: {result.stderr.strip()}"
                    )
            elif language == "json":
                try:
                    json.loads(body)
                except (TypeError, ValueError) as error:
                    failures.append(
                        f"{document.relative_to(REPOSITORY_ROOT)} "
                        f"JSON block {index}: {error}"
                    )
            elif language in {"python", "py"}:
                try:
                    ast.parse(
                        body,
                        filename=(
                            f"{document.relative_to(REPOSITORY_ROOT)}:"
                            f"python-block-{index}"
                        ),
                    )
                except SyntaxError as error:
                    failures.append(
                        f"{document.relative_to(REPOSITORY_ROOT)} "
                        f"Python block {index}: {error}"
                    )
    assert failures == []


def test_documentation_contains_no_secret_literals_or_placeholders() -> None:
    failures: list[str] = []
    for document in _markdown_files():
        source = document.read_text(encoding="utf-8")
        for pattern in _SECRET_LITERALS:
            for match in pattern.finditer(source):
                candidate = match.group(0).casefold()
                if not any(word in candidate for word in ("replace", "example")):
                    failures.append(
                        f"{document.relative_to(REPOSITORY_ROOT)}: "
                        f"possible secret matching {pattern.pattern}"
                    )
        if re.search(r"\b(?:TODO|TBD|FIXME)\b", source, re.IGNORECASE):
            failures.append(
                f"{document.relative_to(REPOSITORY_ROOT)}: unfinished placeholder"
            )
    assert failures == []


def test_container_entrypoint_does_not_gate_operator_tools_on_database(
    tmp_path: Path,
) -> None:
    """Only the normal server command runs migrations before exec."""

    entrypoint = REPOSITORY_ROOT / "docker" / "entrypoint.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "calls"
    for executable in ("alembic", "uvicorn"):
        script = fake_bin / executable
        script.write_text(
            "#!/bin/sh\nprintf '%s\\n' \"$0 $*\" >> \"$ENTRYPOINT_CALLS\"\n",
            encoding="utf-8",
        )
        script.chmod(0o755)

    environment = {
        **os.environ,
        "DATABASE_URL": "postgresql://unreachable.invalid/ctrl_pi",
        "ENTRYPOINT_CALLS": str(calls),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
    }
    operator = subprocess.run(
        [str(entrypoint), "sh", "-c", "exit 0"],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert operator.returncode == 0
    assert not calls.exists(), "operator command unexpectedly invoked migrations"

    server = subprocess.run(
        [str(entrypoint), "uvicorn", "ctrl_pi.main:app"],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert server.returncode == 0
    invocations = calls.read_text(encoding="utf-8").splitlines()
    assert [Path(line.split()[0]).name for line in invocations] == [
        "alembic",
        "uvicorn",
    ]


def test_milestone_13_reference_contracts_are_documented() -> None:
    yam = (DOCS_ROOT / "yam-driver.md").read_text(encoding="utf-8")
    for required in (
        "list_arms",
        "get_arm",
        "jog",
        "apply_action",
        "thread-safe",
        "RigLease",
        "process-local",
        "50 Hz",
        "RealYAMDriver",
        "crank_4310",
        "yam_probe --connect",
    ):
        assert required in yam

    inference = (DOCS_ROOT / "inference.md").read_text(encoding="utf-8")
    for required in (
        "model repository root",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "predict_action_chunk()",
        "real OpenPI runtime is unavailable in V1",
        "zero warm containers",
        "one inference request in flight",
        "5,778 actions",
        "act-aloha-sim-transfer-cube-human-a76db398",
        "743f28ba80c737ea87ea3cc9447fa308b549b700",
    ):
        assert required in inference

    operations = (DOCS_ROOT / "modal-operations.md").read_text(encoding="utf-8")
    assert "make modal-panic" in operations
    assert "ctrl-pi-deployment=<the same UUID>" in operations
    assert re.search(r"zero\s+running\s+tasks", operations)
    assert re.search(
        r"Panic\s+does\s+not\s+create,\s+rotate,\s+or\s+delete\s+Proxy\s+Tokens",
        operations,
    )

    smoke = (DOCS_ROOT / "smoke-test.md").read_text(encoding="utf-8")
    assert "make smoke" in smoke
    assert "[smoke] topology: PASS" in smoke
    assert "four reference mock arms" in smoke
    assert "observation-only" in smoke
    assert "explicit slow synchronization" in smoke
    assert "yam-leader-left" in smoke
    assert "yam-follower-left" in smoke
    assert "exactly 100 successful arm writes" in smoke
    assert "--fake-hub" in smoke
    assert "actual `make smoke` target does not pass `--fake-hub`" in smoke


def test_ros58_onboarding_documents_external_leader_calibration_boundary() -> None:
    setup = (DOCS_ROOT / "setup.md").read_text(encoding="utf-8")
    for required in (
        ".venv/bin/lerobot-calibrate",
        "--teleop.type=yam_leader",
        "--teleop.port=/dev/serial/by-id/REPLACE_WITH_LEADER",
        "--teleop.id=yam-leader",
        "--teleop.calibration_dir=/absolute/path/to/leader-calibration",
        "outside the ctrl-π web application and backend",
        "physical operator at the emergency stop",
        "It does not connect, calibrate, or\nvalidate the follower",
        "have not been executed on a physical YAM",
    ):
        assert required in setup

    yam = (DOCS_ROOT / "yam-driver.md").read_text(encoding="utf-8")
    assert "setup.md#creating-the-leader-calibration-artifact" in yam
    for required in (
        "Settings discovery and preflight",
        "automatic connection remains off by default",
        "boot once with the rig\n    unplugged, hot-plug it",
        "without background retries",
        "revoke automatic connection without\n    forgetting the setup",
        "physical PostgreSQL row was neither overwritten nor deleted",
        "serial-device remap",
        "have not been performed on a physical\nUbuntu/YAM box",
    ):
        assert required in yam
