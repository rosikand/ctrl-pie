from __future__ import annotations

import asyncio
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict

from ctrl_pi.config import AppConfig, get_config
from ctrl_pi.hf_models import (
    HFModelBrowser,
    ModelAuthenticationError,
    ModelHubError,
)

router = APIRouter(prefix="/api/models", tags=["models"])
legacy_router = APIRouter(prefix="/api/trainer", tags=["trainer"])


class ModelCardRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    description: str | None
    base_model: list[str]
    datasets: list[str]


class ModelRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    repo_id: str
    name: str
    revision: str | None
    hub_url: str
    private: bool
    gated: bool
    last_modified: datetime | None
    pipeline_tag: str | None
    library_name: str | None
    tags: list[str]
    card: ModelCardRead | None
    checkpoints: list[str]


class ModelsRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    namespace: str
    models: list[ModelRead]
    total: int
    fetched_at: datetime


def get_model_browser(request: Request) -> HFModelBrowser:
    return request.app.state.hf_model_browser


@legacy_router.get(
    "/models",
    response_model=ModelsRead,
    deprecated=True,
    operation_id="list_trainer_models_legacy",
)
@router.get("", response_model=ModelsRead, operation_id="list_models")
async def list_models(
    response: Response,
    refresh: bool = False,
    config: AppConfig = Depends(get_config),
    browser: HFModelBrowser = Depends(get_model_browser),
) -> ModelsRead:
    """List revision-pinned models in the configured Hugging Face namespace."""

    namespace = (config.hf_namespace or "").strip()
    if not namespace:
        raise HTTPException(
            status_code=503,
            detail="Set HF_NAMESPACE before browsing models.",
        )
    if config.hf_token is None or not config.hf_token.get_secret_value().strip():
        raise HTTPException(
            status_code=503,
            detail="Set HF_TOKEN before browsing models.",
        )
    token = config.hf_token.get_secret_value().strip()
    try:
        page = await asyncio.to_thread(
            browser.list_namespace,
            namespace=namespace,
            token=token,
            refresh=refresh,
        )
    except ModelAuthenticationError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
    except ModelHubError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    response.headers["Cache-Control"] = (
        "private, no-store" if refresh else "private, max-age=30"
    )
    return ModelsRead.model_validate(page)
