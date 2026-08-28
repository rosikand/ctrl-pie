from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path as ApiPath, Query, Request, Response
from pydantic import BaseModel, ConfigDict
from starlette.responses import StreamingResponse

from ctrl_pi.config import AppConfig, get_config
from ctrl_pi.hf_datasets import (
    DatasetAuthenticationError,
    DatasetCursorError,
    DatasetHubError,
    HFDatasetBrowser,
)
from ctrl_pi.hf_episodes import (
    ByteRangeError,
    EpisodeAuthenticationError,
    EpisodeDatasetNotFoundError,
    EpisodeFormatError,
    EpisodeHubError,
    EpisodeNotFoundError,
    HFEpisodeBrowser,
)

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


class DatasetCardRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    title: str | None
    description: str | None
    license: str | None
    task_categories: list[str]


class LeRobotDatasetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    codebase_version: str | None
    robot_type: str | None
    fps: float | None
    total_episodes: int | None
    total_frames: int | None
    total_tasks: int | None
    features: list[str]


class DatasetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    repo_id: str
    name: str
    revision: str | None
    hub_url: str
    private: bool
    gated: bool
    created_at: datetime | None
    last_modified: datetime | None
    tags: list[str]
    card: DatasetCardRead | None
    lerobot: LeRobotDatasetRead | None


class DatasetsRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    namespace: str
    datasets: list[DatasetRead]
    total: int
    next_cursor: str | None
    fetched_at: datetime


class EpisodeSummaryRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    episode_index: int
    tasks: list[str]
    frame_count: int
    duration_seconds: float
    dataset_from_index: int
    dataset_to_index: int
    video_from_timestamp: float | None
    video_to_timestamp: float | None


class EpisodesRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    repo_id: str
    revision: str
    fps: float
    state_names: list[str]
    action_names: list[str]
    video_key: str | None
    total_episodes: int
    episodes: list[EpisodeSummaryRead]


class TimelineFrameRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    timestamp: float
    frame_index: int
    state: list[float]
    action: list[float]


class EpisodeDetailRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    repo_id: str
    revision: str
    fps: float
    state_names: list[str]
    action_names: list[str]
    video_key: str | None
    episode: EpisodeSummaryRead
    frames: list[TimelineFrameRead]
    sampled_frame_count: int
    frames_truncated: bool
    video_url: str | None


def get_dataset_browser(request: Request) -> HFDatasetBrowser:
    return request.app.state.hf_dataset_browser


def get_episode_browser(request: Request) -> HFEpisodeBrowser:
    return request.app.state.hf_episode_browser


def _hub_credentials(config: AppConfig) -> tuple[str, str]:
    namespace = (config.hf_namespace or "").strip()
    if not namespace:
        raise HTTPException(
            status_code=503,
            detail="Set HF_NAMESPACE before browsing datasets.",
        )
    if config.hf_token is None or not config.hf_token.get_secret_value().strip():
        raise HTTPException(
            status_code=503,
            detail="Set HF_TOKEN before browsing datasets.",
        )
    return namespace, config.hf_token.get_secret_value().strip()


def _episode_http_error(error: Exception) -> HTTPException:
    if isinstance(error, EpisodeAuthenticationError):
        return HTTPException(status_code=403, detail=str(error))
    if isinstance(error, (EpisodeDatasetNotFoundError, EpisodeNotFoundError)):
        return HTTPException(status_code=404, detail=str(error))
    if isinstance(error, EpisodeFormatError):
        return HTTPException(status_code=422, detail=str(error))
    return HTTPException(status_code=502, detail=str(error))


@router.get("", response_model=DatasetsRead)
async def list_datasets(
    response: Response,
    limit: Annotated[int, Query(ge=1, le=100)] = 24,
    cursor: Annotated[str | None, Query(max_length=1024)] = None,
    refresh: bool = False,
    config: AppConfig = Depends(get_config),
    browser: HFDatasetBrowser = Depends(get_dataset_browser),
) -> DatasetsRead:
    namespace, token = _hub_credentials(config)
    try:
        page = await asyncio.to_thread(
            browser.list_namespace,
            namespace=namespace,
            token=token,
            limit=limit,
            cursor=cursor,
            refresh=refresh,
        )
    except DatasetCursorError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except DatasetAuthenticationError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
    except DatasetHubError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    response.headers["Cache-Control"] = (
        "private, no-store" if refresh else "private, max-age=30"
    )
    return DatasetsRead.model_validate(page)


@router.get("/{repo_name}/episodes", response_model=EpisodesRead)
async def list_dataset_episodes(
    repo_name: Annotated[str, ApiPath(min_length=1, max_length=96)],
    config: AppConfig = Depends(get_config),
    browser: HFEpisodeBrowser = Depends(get_episode_browser),
) -> EpisodesRead:
    namespace, token = _hub_credentials(config)
    try:
        episodes = await asyncio.to_thread(
            browser.list_episodes,
            namespace=namespace,
            repo_name=repo_name,
            token=token,
        )
    except (
        EpisodeAuthenticationError,
        EpisodeDatasetNotFoundError,
        EpisodeFormatError,
        EpisodeHubError,
    ) as error:
        raise _episode_http_error(error) from error
    return EpisodesRead.model_validate(episodes)


@router.get("/{repo_name}/episodes/{episode_index}", response_model=EpisodeDetailRead)
async def get_dataset_episode(
    repo_name: Annotated[str, ApiPath(min_length=1, max_length=96)],
    episode_index: Annotated[int, ApiPath(ge=0)],
    revision: Annotated[str, Query(pattern=r"^[0-9a-f]{40}$")],
    config: AppConfig = Depends(get_config),
    browser: HFEpisodeBrowser = Depends(get_episode_browser),
) -> EpisodeDetailRead:
    namespace, token = _hub_credentials(config)
    try:
        episode = await asyncio.to_thread(
            browser.episode_detail,
            namespace=namespace,
            repo_name=repo_name,
            episode_index=episode_index,
            revision=revision,
            token=token,
        )
    except (
        EpisodeAuthenticationError,
        EpisodeDatasetNotFoundError,
        EpisodeNotFoundError,
        EpisodeFormatError,
        EpisodeHubError,
    ) as error:
        raise _episode_http_error(error) from error
    return EpisodeDetailRead.model_validate(episode)


@router.get("/{repo_name}/episodes/{episode_index}/video")
@router.head(
    "/{repo_name}/episodes/{episode_index}/video",
    include_in_schema=False,
)
async def get_dataset_episode_video(
    request: Request,
    repo_name: Annotated[str, ApiPath(min_length=1, max_length=96)],
    episode_index: Annotated[int, ApiPath(ge=0)],
    revision: Annotated[str, Query(pattern=r"^[0-9a-f]{40}$")],
    config: AppConfig = Depends(get_config),
    browser: HFEpisodeBrowser = Depends(get_episode_browser),
) -> Response:
    namespace, token = _hub_credentials(config)
    try:
        asset = await asyncio.to_thread(
            browser.video_asset,
            namespace=namespace,
            repo_name=repo_name,
            episode_index=episode_index,
            revision=revision,
            token=token,
        )
    except (
        EpisodeAuthenticationError,
        EpisodeDatasetNotFoundError,
        EpisodeNotFoundError,
        EpisodeFormatError,
        EpisodeHubError,
    ) as error:
        raise _episode_http_error(error) from error

    base_headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "private, max-age=31536000, immutable",
        "ETag": asset.etag,
    }
    try:
        selected_range = browser.parse_range(request.headers.get("range"), asset.size)
    except ByteRangeError:
        return Response(
            status_code=416,
            headers={
                **base_headers,
                "Content-Range": f"bytes */{asset.size}",
                "Content-Length": "0",
            },
            media_type="video/mp4",
        )

    if selected_range is None:
        start, end, status_code = 0, asset.size - 1, 200
    else:
        start, end = selected_range
        status_code = 206
    length = end - start + 1
    headers = {**base_headers, "Content-Length": str(length)}
    if status_code == 206:
        headers["Content-Range"] = f"bytes {start}-{end}/{asset.size}"
    if request.method == "HEAD":
        return Response(
            status_code=status_code,
            headers=headers,
            media_type="video/mp4",
        )
    return StreamingResponse(
        browser.stream_file(asset.path, start=start, length=length),
        status_code=status_code,
        headers=headers,
        media_type="video/mp4",
    )
