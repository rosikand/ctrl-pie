from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from ctrl_pi.db import Base

json_type = JSON().with_variant(JSONB(), "postgresql")


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class Robot(TimestampMixin, Base):
    __tablename__ = "robots"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    driver_id: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False, default="follower")
    driver: Mapped[str] = mapped_column(String(64), nullable=False, default="mock")
    can_interface: Mapped[str | None] = mapped_column(String(64))
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    config: Mapped[dict[str, Any]] = mapped_column(json_type, default=dict, nullable=False)


class Recording(TimestampMixin, Base):
    __tablename__ = "recordings"
    __table_args__ = (Index("ix_recordings_created_at", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    task: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="draft")
    leader_robot_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("robots.id", ondelete="SET NULL")
    )
    follower_robot_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("robots.id", ondelete="SET NULL")
    )
    episode_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    hf_repo_id: Mapped[str | None] = mapped_column(String(255))
    recording_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", json_type, default=dict, nullable=False
    )


class TrainingRun(TimestampMixin, Base):
    __tablename__ = "training_runs"
    __table_args__ = (Index("ix_training_runs_status", "status"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="created")
    current_step: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    dataset_repo: Mapped[str | None] = mapped_column(String(255))
    base_model: Mapped[str | None] = mapped_column(String(255))
    runtime: Mapped[str | None] = mapped_column(String(64))
    framework: Mapped[str | None] = mapped_column(String(64))
    output_model_repo: Mapped[str | None] = mapped_column(String(255))
    checkpoint_revision: Mapped[str | None] = mapped_column(String(255))
    config: Mapped[dict[str, Any]] = mapped_column(json_type, default=dict, nullable=False)
    metrics: Mapped[dict[str, list[dict[str, float | int]]]] = mapped_column(
        json_type, default=dict, server_default="{}", nullable=False
    )
    checkpoints: Mapped[list[dict[str, Any]]] = mapped_column(
        json_type, default=list, server_default="[]", nullable=False
    )
    console_logs: Mapped[list[dict[str, Any]]] = mapped_column(
        json_type, default=list, server_default="[]", nullable=False
    )


class InferenceEndpoint(TimestampMixin, Base):
    __tablename__ = "inference_endpoints"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    runtime: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="stopped")
    endpoint_url: Mapped[str | None] = mapped_column(Text)
    provider_app_id: Mapped[str | None] = mapped_column(String(255))
    last_heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Deployment(TimestampMixin, Base):
    __tablename__ = "deployments"
    __table_args__ = (
        Index("ix_deployments_status", "status"),
        CheckConstraint(
            "target_kind IN ('stub', 'modal')",
            name="ck_deployments_target_kind",
        ),
        CheckConstraint(
            "timeout_seconds BETWEEN 1 AND 1800",
            name="ck_deployments_timeout_seconds",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    endpoint_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("inference_endpoints.id", ondelete="SET NULL")
    )
    robot_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("robots.id", ondelete="SET NULL")
    )
    recording_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("recordings.id", ondelete="SET NULL")
    )
    model_repo: Mapped[str] = mapped_column(String(255), nullable=False)
    checkpoint_revision: Mapped[str | None] = mapped_column(String(255))
    runtime: Mapped[str] = mapped_column(String(64), nullable=False)
    compute_size: Mapped[str] = mapped_column(String(64), nullable=False)
    target_kind: Mapped[str] = mapped_column(
        String(16), nullable=False, default="stub", server_default="stub"
    )
    timeout_seconds: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1800, server_default=text("1800")
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="created")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    stopped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    latency_ms: Mapped[float | None] = mapped_column(Float)
    frequency_hz: Mapped[float | None] = mapped_column(Float)
    record_session: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class AppSetting(TimestampMixin, Base):
    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String(120), primary_key=True)
    value: Mapped[Any] = mapped_column(json_type, nullable=False)
