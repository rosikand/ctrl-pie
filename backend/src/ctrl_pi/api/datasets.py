from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, ConfigDict

from ctrl_pi.config import AppConfig, get_config
from ctrl_pi.hf_datasets import (
    DatasetAuthenticationError,
    DatasetCursorError,
    DatasetHubError,
    HFDatasetBrowser,
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


def get_dataset_browser(request: Request) -> HFDatasetBrowser:
    return request.app.state.hf_dataset_browser


@router.get("", response_model=DatasetsRead)
async def list_datasets(
    response: Response,
    limit: Annotated[int, Query(ge=1, le=100)] = 24,
    cursor: Annotated[str | None, Query(max_length=1024)] = None,
    refresh: bool = False,
    config: AppConfig = Depends(get_config),
    browser: HFDatasetBrowser = Depends(get_dataset_browser),
) -> DatasetsRead:
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
    token = config.hf_token.get_secret_value().strip()
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
