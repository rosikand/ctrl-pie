#!/usr/bin/env python3
"""Validate the repository-hosted Mintlify documentation without dependencies."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPOSITORY_ROOT / "docs"
CONFIG_PATH = DOCS_ROOT / "docs.json"

FRONTMATTER = re.compile(r"\A---\n(?P<body>.*?)\n---(?:\n|\Z)", re.DOTALL)
FRONTMATTER_FIELD = re.compile(
    r'^\s*(?P<key>title|description):\s*["\'](?P<value>.+?)["\']\s*$',
    re.MULTILINE,
)
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\((?P<target><[^>]+>|[^\s)]+)")
HTML_LINK = re.compile(r'\b(?:href|src)="(?P<target>[^"]+)"')
HEADING = re.compile(r"^#{2,6}\s+(?P<title>.+?)\s*$", re.MULTILINE)


def page_references(value: object) -> list[str]:
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


def page_source(route: str) -> Path | None:
    candidates = [DOCS_ROOT / f"{route}.md", DOCS_ROOT / f"{route}.mdx"]
    existing = [candidate for candidate in candidates if candidate.is_file()]
    return existing[0] if len(existing) == 1 else None


def heading_slug(title: str) -> str:
    value = re.sub(r"<[^>]+>", "", title)
    value = value.replace("`", "").casefold()
    value = re.sub(r"[^\w\s-]", "", value)
    return re.sub(r"[-\s]+", "-", value).strip("-")


def route_and_anchor(target: str) -> tuple[str, str | None]:
    parsed = urlsplit(target)
    route = unquote(parsed.path).strip("/") or "index"
    return route, unquote(parsed.fragment) or None


def validate() -> list[str]:
    errors: list[str] = []
    try:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return [f"docs/docs.json: {error}"]

    required = {
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
            errors.append(
                f"docs/docs.json: {key!r} must equal {expected!r}"
            )

    references = page_references(config.get("navigation"))
    duplicate_references = sorted(
        route for route, count in Counter(references).items() if count > 1
    )
    if duplicate_references:
        errors.append(
            "docs/docs.json: duplicate page routes: "
            + ", ".join(duplicate_references)
        )

    route_sources: dict[str, Path] = {}
    for route in references:
        source = page_source(route)
        if source is None:
            errors.append(
                f"docs/docs.json: route {route!r} must resolve to exactly one .md or .mdx file"
            )
        else:
            route_sources[route] = source

    content_files = sorted(
        [*DOCS_ROOT.rglob("*.md"), *DOCS_ROOT.rglob("*.mdx")]
    )
    navigated_files = set(route_sources.values())
    for source in content_files:
        if source not in navigated_files:
            errors.append(
                f"{source.relative_to(REPOSITORY_ROOT)}: page is missing from docs.json navigation"
            )

    headings: dict[str, set[str]] = {}
    for route, source in route_sources.items():
        text = source.read_text(encoding="utf-8")
        frontmatter = FRONTMATTER.match(text)
        if frontmatter is None:
            errors.append(
                f"{source.relative_to(REPOSITORY_ROOT)}: missing YAML frontmatter"
            )
        else:
            metadata = {
                match.group("key"): match.group("value").strip()
                for match in FRONTMATTER_FIELD.finditer(frontmatter.group("body"))
            }
            for field in ("title", "description"):
                if not metadata.get(field):
                    errors.append(
                        f"{source.relative_to(REPOSITORY_ROOT)}: missing quoted {field} frontmatter"
                    )
        headings[route] = {
            heading_slug(match.group("title")) for match in HEADING.finditer(text)
        }

    known_routes = set(route_sources)
    for route, source in route_sources.items():
        text = source.read_text(encoding="utf-8")
        targets = [
            *(match.group("target") for match in MARKDOWN_LINK.finditer(text)),
            *(match.group("target") for match in HTML_LINK.finditer(text)),
        ]
        for raw_target in targets:
            target = raw_target[1:-1] if raw_target.startswith("<") else raw_target
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or target.startswith("mailto:"):
                continue
            if not parsed.path and parsed.fragment:
                destination_route = route
            elif parsed.path.startswith("/"):
                destination_route, _ = route_and_anchor(target)
                if destination_route.startswith("assets/"):
                    asset = DOCS_ROOT / destination_route
                    if not asset.is_file():
                        errors.append(
                            f"{source.relative_to(REPOSITORY_ROOT)}: missing asset {parsed.path}"
                        )
                    continue
                if destination_route not in known_routes:
                    errors.append(
                        f"{source.relative_to(REPOSITORY_ROOT)}: unknown route {parsed.path}"
                    )
                    continue
            else:
                relative = (source.parent / unquote(parsed.path)).resolve()
                if parsed.path and not relative.is_file():
                    errors.append(
                        f"{source.relative_to(REPOSITORY_ROOT)}: missing relative target {parsed.path}"
                    )
                continue

            anchor = unquote(parsed.fragment)
            if anchor and anchor not in headings.get(destination_route, set()):
                errors.append(
                    f"{source.relative_to(REPOSITORY_ROOT)}: unknown anchor #{anchor} on /{destination_route}"
                )

    for asset_key in ("favicon",):
        value = config.get(asset_key)
        if isinstance(value, str) and value.startswith("/"):
            asset = DOCS_ROOT / value.lstrip("/")
            if not asset.is_file():
                errors.append(f"docs/docs.json: missing {asset_key} asset {value}")
    logo = config.get("logo")
    if isinstance(logo, dict):
        for mode in ("light", "dark"):
            value = logo.get(mode)
            if not isinstance(value, str) or not (DOCS_ROOT / value.lstrip("/")).is_file():
                errors.append(f"docs/docs.json: missing logo.{mode} asset")

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
