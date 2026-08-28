from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

from PIL import Image, ImageChops


REPOSITORY_ROOT = Path(__file__).parents[2]
DOCS_ROOT = REPOSITORY_ROOT / "docs"
README = REPOSITORY_ROOT / "README.md"
SCREENSHOT = DOCS_ROOT / "assets" / "ctrl-pi-inference.png"

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
    files = [README, *sorted(DOCS_ROOT.rglob("*.md"))]
    assert len(files) >= 11
    return files


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


def test_inference_screenshot_is_real_and_version_control_visible() -> None:
    assert SCREENSHOT.is_file()
    assert SCREENSHOT.stat().st_size >= 50_000
    with Image.open(SCREENSHOT) as image:
        image.verify()
    with Image.open(SCREENSHOT) as image:
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

    relative = SCREENSHOT.relative_to(REPOSITORY_ROOT).as_posix()
    visible = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            relative,
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert relative in visible, "screenshot is ignored and would not be committed"
    assert "docs/assets/ctrl-pi-inference.png" in README.read_text(encoding="utf-8")
    assert "assets/ctrl-pi-inference.png" in (
        DOCS_ROOT / "inference.md"
    ).read_text(encoding="utf-8")


def test_documentation_shell_and_json_examples_parse() -> None:
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
    assert "exactly 100 successful arm writes" in smoke
    assert "--fake-hub" in smoke
    assert "actual `make smoke` target does not pass `--fake-hub`" in smoke
