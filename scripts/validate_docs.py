#!/usr/bin/env python3
"""Validate the repository-hosted Mintlify documentation without dependencies."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path, PurePosixPath
import re
import sys
from urllib.parse import unquote, urlsplit


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPOSITORY_ROOT / "docs"
CONFIG_PATH = DOCS_ROOT / "docs.json"

REQUIRED_ROUTES = {
    "index",
    "installation",
    "quickstart",
    "setup",
    "configuration",
    "yam-setup",
    "arms",
    "recording",
    "datasets",
    "training",
    "models",
    "inference",
    "python-sdk",
    "trainer-api",
    "architecture",
    "yam-driver",
    "docker-deployment",
    "development",
    "modal-operations",
    "smoke-test",
    "troubleshooting",
    "screenshots",
    "release-notes",
}

FRONTMATTER = re.compile(
    r"\A---\r?\n(?P<body>.*?)\r?\n---(?:\r?\n|\Z)", re.DOTALL
)
FRONTMATTER_FIELD = re.compile(
    r'^\s*(?P<key>title|description):\s*["\'](?P<value>.+?)["\']\s*$',
    re.MULTILINE,
)
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\((?P<target><[^>]+>|[^\s)]+)")
HTML_LINK = re.compile(
    r"\b(?:href|src)\s*=\s*(?P<quote>[\"\'])(?P<target>.*?)(?P=quote)"
)
FENCED_BLOCK = re.compile(r"^```.*?^```\s*$", re.MULTILINE | re.DOTALL)
HEADING = re.compile(r"^#{1,6}\s+(?P<title>.+?)\s*#*\s*$", re.MULTILINE)
EXPLICIT_ID = re.compile(r'\bid=["\'](?P<id>[^"\']+)["\']')
ROUTE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_./-]*[A-Za-z0-9_-])?")


def page_references(value: object) -> list[str]:
    """Return every string leaf under a Mintlify ``pages`` collection."""

    references: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "pages" and isinstance(child, list):
                references.extend(item for item in child if isinstance(item, str))
            references.extend(page_references(child))
    elif isinstance(value, list):
        for child in value:
            references.extend(page_references(child))
    return references


def safe_route(route: str) -> bool:
    if (
        not route
        or route.startswith("/")
        or "\\" in route
        or "%" in route
        or ROUTE.fullmatch(route) is None
        or Path(route).suffix in {".md", ".mdx"}
    ):
        return False
    parts = PurePosixPath(route).parts
    return bool(parts) and all(part not in {"", ".", ".."} for part in parts)


def page_source(route: str) -> Path | None:
    if not safe_route(route):
        return None
    candidates = [DOCS_ROOT / f"{route}.md", DOCS_ROOT / f"{route}.mdx"]
    existing = [candidate for candidate in candidates if candidate.is_file()]
    if len(existing) != 1:
        return None
    try:
        existing[0].resolve().relative_to(DOCS_ROOT.resolve())
    except ValueError:
        return None
    return existing[0]


def heading_slug(title: str) -> str:
    value = re.sub(r"<[^>]+>", "", title)
    value = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", value)
    value = value.replace("`", "").casefold()
    value = re.sub(r"[^\w\s-]", "", value)
    return re.sub(r"[-\s]+", "-", value).strip("-")


def page_anchors(text: str) -> set[str]:
    visible = FENCED_BLOCK.sub("", text)
    anchors: set[str] = {
        unquote(match.group("id")) for match in EXPLICIT_ID.finditer(visible)
    }
    counts: Counter[str] = Counter()
    for match in HEADING.finditer(visible):
        base = heading_slug(match.group("title"))
        if not base:
            continue
        duplicate = counts[base]
        anchors.add(base if duplicate == 0 else f"{base}-{duplicate}")
        counts[base] += 1
    return anchors


def route_and_anchor(target: str) -> tuple[str, str | None]:
    parsed = urlsplit(target)
    route = unquote(parsed.path).strip("/") or "index"
    return route, unquote(parsed.fragment) or None


def content_files() -> list[Path]:
    return sorted(
        path
        for pattern in ("*.md", "*.mdx")
        for path in DOCS_ROOT.rglob(pattern)
        if not {"node_modules", ".mintlify"}.intersection(path.parts)
    )


def _config_errors(config: object) -> list[str]:
    if not isinstance(config, dict):
        return ["docs/docs.json: top-level value must be an object"]

    errors: list[str] = []
    required: dict[str, object | None] = {
        "$schema": "https://mintlify.com/docs.json",
        "name": None,
        "theme": None,
        "colors": None,
        "navigation": None,
    }
    for key, expected in required.items():
        if key not in config:
            errors.append(f"docs/docs.json: missing required key {key!r}")
        elif expected is not None and config[key] != expected:
            errors.append(f"docs/docs.json: {key!r} must equal {expected!r}")
        elif expected is None and config[key] in (None, "", {}, []):
            errors.append(f"docs/docs.json: {key!r} must not be empty")
    return errors


def _asset_error(source: Path, raw_path: str) -> str | None:
    relative = unquote(raw_path).lstrip("/")
    candidate = (DOCS_ROOT / relative).resolve()
    try:
        candidate.relative_to(DOCS_ROOT.resolve())
    except ValueError:
        return f"{source.relative_to(REPOSITORY_ROOT)}: asset escapes docs root {raw_path}"
    if not candidate.is_file():
        return f"{source.relative_to(REPOSITORY_ROOT)}: missing asset {raw_path}"
    return None


def validate() -> list[str]:
    errors: list[str] = []
    try:
        config: object = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return [f"docs/docs.json: {error}"]

    errors.extend(_config_errors(config))
    if not isinstance(config, dict):
        return errors

    references = page_references(config.get("navigation"))
    if not references:
        errors.append("docs/docs.json: navigation must contain at least one page")
    invalid_references = sorted({route for route in references if not safe_route(route)})
    if invalid_references:
        errors.append(
            "docs/docs.json: unsafe page routes: " + ", ".join(invalid_references)
        )
    duplicate_references = sorted(
        route for route, count in Counter(references).items() if count > 1
    )
    if duplicate_references:
        errors.append(
            "docs/docs.json: duplicate page routes: " + ", ".join(duplicate_references)
        )
    missing_required = sorted(REQUIRED_ROUTES.difference(references))
    if missing_required:
        errors.append(
            "docs/docs.json: missing required product routes: "
            + ", ".join(missing_required)
        )

    route_sources: dict[str, Path] = {}
    for route in references:
        source = page_source(route)
        if source is None:
            errors.append(
                f"docs/docs.json: route {route!r} must resolve to exactly one safe .md or .mdx file"
            )
        else:
            route_sources[route] = source

    navigated_files = {source.resolve() for source in route_sources.values()}
    for source in content_files():
        if source.resolve() not in navigated_files:
            errors.append(
                f"{source.relative_to(REPOSITORY_ROOT)}: page is missing from docs.json navigation"
            )

    anchors: dict[str, set[str]] = {}
    file_routes: dict[Path, str] = {}
    page_text: dict[str, str] = {}
    for route, source in route_sources.items():
        text = source.read_text(encoding="utf-8")
        page_text[route] = text
        frontmatter = FRONTMATTER.match(text)
        if frontmatter is None:
            errors.append(f"{source.relative_to(REPOSITORY_ROOT)}: missing YAML frontmatter")
        else:
            fields = [
                (match.group("key"), match.group("value").strip())
                for match in FRONTMATTER_FIELD.finditer(frontmatter.group("body"))
            ]
            field_counts = Counter(key for key, _value in fields)
            metadata = dict(fields)
            for field in ("title", "description"):
                if not metadata.get(field):
                    errors.append(
                        f"{source.relative_to(REPOSITORY_ROOT)}: missing quoted {field} frontmatter"
                    )
                elif field_counts[field] != 1:
                    errors.append(
                        f"{source.relative_to(REPOSITORY_ROOT)}: frontmatter must contain one {field}"
                    )
        anchors[route] = page_anchors(text)
        file_routes[source.resolve()] = route

    known_routes = set(route_sources)
    for route, source in route_sources.items():
        text = FENCED_BLOCK.sub("", page_text[route])
        targets = [
            *(match.group("target") for match in MARKDOWN_LINK.finditer(text)),
            *(match.group("target") for match in HTML_LINK.finditer(text)),
        ]
        for raw_target in targets:
            target = raw_target[1:-1] if raw_target.startswith("<") else raw_target
            try:
                parsed = urlsplit(target)
            except ValueError:
                errors.append(
                    f"{source.relative_to(REPOSITORY_ROOT)}: malformed link {target}"
                )
                continue
            if parsed.scheme:
                if parsed.scheme.casefold() not in {"http", "https", "mailto"}:
                    errors.append(
                        f"{source.relative_to(REPOSITORY_ROOT)}: unsupported link scheme {parsed.scheme}"
                    )
                continue
            if parsed.netloc:
                continue

            destination_route: str | None = None
            if not parsed.path and parsed.fragment:
                destination_route = route
            elif parsed.path.startswith("/"):
                destination_route, _anchor = route_and_anchor(target)
                if destination_route.startswith("assets/"):
                    asset_error = _asset_error(source, parsed.path)
                    if asset_error:
                        errors.append(asset_error)
                    continue
                if destination_route not in known_routes:
                    errors.append(
                        f"{source.relative_to(REPOSITORY_ROOT)}: unknown route {parsed.path}"
                    )
                    continue
            elif parsed.path:
                relative = (source.parent / unquote(parsed.path)).resolve()
                if not relative.is_file():
                    errors.append(
                        f"{source.relative_to(REPOSITORY_ROOT)}: missing relative target {parsed.path}"
                    )
                    continue
                destination_route = file_routes.get(relative)

            anchor = unquote(parsed.fragment)
            if (
                anchor
                and destination_route is not None
                and anchor not in anchors.get(destination_route, set())
            ):
                errors.append(
                    f"{source.relative_to(REPOSITORY_ROOT)}: unknown anchor #{anchor} on /{destination_route}"
                )

    favicon = config.get("favicon")
    if isinstance(favicon, str) and favicon.startswith("/"):
        asset_error = _asset_error(CONFIG_PATH, favicon)
        if asset_error:
            errors.append(asset_error)
    else:
        errors.append("docs/docs.json: favicon must be an absolute docs asset path")
    logo = config.get("logo")
    if not isinstance(logo, dict):
        errors.append("docs/docs.json: logo must define light and dark assets")
    else:
        for mode in ("light", "dark"):
            value = logo.get(mode)
            if not isinstance(value, str) or not value.startswith("/"):
                errors.append(f"docs/docs.json: logo.{mode} must be an absolute asset path")
                continue
            asset_error = _asset_error(CONFIG_PATH, value)
            if asset_error:
                errors.append(asset_error)

    return errors


def main() -> int:
    errors = validate()
    if errors:
        print("Documentation validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Documentation validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
