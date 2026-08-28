from __future__ import annotations

from pathlib import Path, PurePosixPath

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


_RESERVED_PREFIXES = frozenset({"api", "assets", "ws"})


def install_spa(application: FastAPI, dist_dir: Path | None) -> bool:
    """Serve a built Vite app when a complete distribution is configured.

    The routes are intentionally installed after every API and WebSocket
    router.  The catch-all serves only extensionless browser navigation paths;
    missing API, socket, asset, and media paths must retain normal 404s.
    """

    application.state.frontend_serving_enabled = False
    application.state.frontend_dist_dir = None
    if dist_dir is None:
        return False

    candidate = Path(dist_dir).expanduser()
    index_path = candidate / "index.html"
    if not index_path.is_file():
        return False

    assets_path = candidate / "assets"
    if assets_path.is_dir():
        application.mount(
            "/assets",
            StaticFiles(directory=str(assets_path), html=False),
            name="frontend-assets",
        )

    def index_response() -> FileResponse:
        # The Vite index names immutable hashed assets but must itself be
        # revalidated when a new image is deployed.
        return FileResponse(
            index_path,
            media_type="text/html",
            headers={"Cache-Control": "no-cache"},
        )

    @application.api_route(
        "/", methods=["GET", "HEAD"], include_in_schema=False
    )
    async def frontend_index() -> FileResponse:
        return index_response()

    @application.api_route(
        "/{frontend_path:path}", methods=["GET", "HEAD"], include_in_schema=False
    )
    async def frontend_fallback(frontend_path: str) -> FileResponse:
        segments = tuple(part for part in frontend_path.split("/") if part)
        first_segment = segments[0].casefold() if segments else ""
        filename = PurePosixPath(frontend_path).name
        is_dataset_detail = (
            len(segments) == 2 and first_segment == "datasets"
        )
        if first_segment in _RESERVED_PREFIXES or (
            "." in filename and not is_dataset_detail
        ):
            raise HTTPException(status_code=404, detail="Not Found")
        return index_response()

    application.state.frontend_serving_enabled = True
    application.state.frontend_dist_dir = candidate
    return True
