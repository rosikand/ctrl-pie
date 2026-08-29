from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
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
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

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


class YAMSetup(TimestampMixin, Base):
    """The one persisted, non-secret YAM rig setup supported by V1.1."""

    __tablename__ = "yam_setups"
    __table_args__ = (
        CheckConstraint("id = 'primary'", name="ck_yam_setups_singleton"),
        CheckConstraint("mode IN ('mock', 'hardware')", name="ck_yam_setups_mode"),
    )

    id: Mapped[str] = mapped_column(String(16), primary_key=True, default="primary")
    mode: Mapped[str] = mapped_column(String(16), nullable=False)
    can_interface: Mapped[str] = mapped_column(String(15), nullable=False)
    leader_port: Mapped[str] = mapped_column(String(200), nullable=False)
    mujoco_xml_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    leader_calibration_id: Mapped[str] = mapped_column(String(64), nullable=False)
    leader_calibration_dir: Mapped[str] = mapped_column(String(1024), nullable=False)
    auto_restore: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text("false")
    )
    last_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_connected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class YAMCell(TimestampMixin, Base):
    """The one persisted physical YAM cell supported by V1.2.

    Mock topology remains driver-owned and is deliberately not stored here. Runtime
    SocketCAN names are likewise absent: an arm's durable USB adapter identity lives
    on :class:`YAMCellArm` and resolves to the current ``canN`` only in live state.
    """

    __tablename__ = "yam_cells"
    __table_args__ = (
        CheckConstraint("id = 'primary'", name="ck_yam_cells_singleton"),
        CheckConstraint("length(i2rt_commit) = 40", name="ck_yam_cells_i2rt_commit"),
    )

    id: Mapped[str] = mapped_column(String(16), primary_key=True, default="primary")
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    i2rt_root: Mapped[str] = mapped_column(String(1_024), nullable=False)
    i2rt_commit: Mapped[str] = mapped_column(String(40), nullable=False)
    pair_ports: Mapped[dict[str, int]] = mapped_column(
        json_type, default=dict, server_default=text("'{}'"), nullable=False
    )
    auto_restore: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text("false")
    )
    last_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_connected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    arms: Mapped[list[YAMCellArm]] = relationship(
        back_populates="cell",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="YAMCellArm.position",
    )


class YAMCellArm(TimestampMixin, Base):
    """One typed, durable arm assignment within the primary physical cell."""

    __tablename__ = "yam_cell_arms"
    __table_args__ = (
        CheckConstraint("position >= 0", name="ck_yam_cell_arms_position"),
        CheckConstraint(
            "role IN ('leader', 'follower')", name="ck_yam_cell_arms_role"
        ),
        CheckConstraint(
            "transport_kind IN ('socketcan', 'serial')",
            name="ck_yam_cell_arms_transport_kind",
        ),
        CheckConstraint(
            "end_effector_kind IN ('yam_teaching_handle', 'linear_4310', "
            "'crank_4310', 'gello', 'none')",
            name="ck_yam_cell_arms_end_effector_kind",
        ),
        CheckConstraint(
            "(calibration_id IS NULL AND calibration_dir IS NULL) OR "
            "(calibration_id IS NOT NULL AND calibration_dir IS NOT NULL)",
            name="ck_yam_cell_arms_calibration_pair",
        ),
        CheckConstraint(
            "(transport_kind = 'socketcan' AND role = 'leader' AND "
            "end_effector_kind = 'yam_teaching_handle') OR "
            "(transport_kind = 'socketcan' AND role = 'follower' AND "
            "end_effector_kind IN ('linear_4310', 'crank_4310')) OR "
            "(transport_kind = 'serial' AND role = 'leader' AND "
            "end_effector_kind = 'gello')",
            name="ck_yam_cell_arms_supported_shape",
        ),
        UniqueConstraint(
            "cell_id", "position", name="uq_yam_cell_arms_position"
        ),
        UniqueConstraint(
            "cell_id", "logical_id", name="uq_yam_cell_arms_logical_id"
        ),
        UniqueConstraint("cell_id", "name", name="uq_yam_cell_arms_name"),
        UniqueConstraint(
            "cell_id",
            "transport_kind",
            "stable_identity",
            name="uq_yam_cell_arms_stable_identity",
        ),
        Index("ix_yam_cell_arms_pair", "cell_id", "pair_id"),
        Index("ix_yam_cell_arms_group", "cell_id", "group_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    cell_id: Mapped[str] = mapped_column(
        String(16), ForeignKey("yam_cells.id", ondelete="CASCADE"), nullable=False
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    logical_id: Mapped[str] = mapped_column(String(120), nullable=False)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    pair_id: Mapped[str | None] = mapped_column(String(120))
    group_id: Mapped[str | None] = mapped_column(String(120))
    side: Mapped[str | None] = mapped_column(String(64))
    transport_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    stable_identity: Mapped[str] = mapped_column(String(256), nullable=False)
    end_effector_kind: Mapped[str] = mapped_column(String(64), nullable=False)
    frame_map_path: Mapped[str | None] = mapped_column(String(1_024))
    soft_limits_path: Mapped[str | None] = mapped_column(String(1_024))
    mujoco_xml_path: Mapped[str | None] = mapped_column(String(1_024))
    calibration_id: Mapped[str | None] = mapped_column(String(64))
    calibration_dir: Mapped[str | None] = mapped_column(String(1_024))
    config: Mapped[dict[str, Any]] = mapped_column(
        json_type, default=dict, server_default=text("'{}'"), nullable=False
    )
    cell: Mapped[YAMCell] = relationship(back_populates="arms")


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


class ManagedTrainingJob(TimestampMixin, Base):
    """Durable ownership and teardown state for one managed training run."""

    __tablename__ = "managed_training_jobs"
    __table_args__ = (
        Index("ix_managed_training_jobs_status", "status"),
        Index("ix_managed_training_jobs_deadline", "status", "deadline_at"),
        Index(
            "uq_managed_training_jobs_one_active",
            text("(1)"),
            unique=True,
            postgresql_where=text(
                "status IN ('created', 'launching', 'running', 'finalizing', 'cancelling')"
            ),
            sqlite_where=text(
                "status IN ('created', 'launching', 'running', 'finalizing', 'cancelling')"
            ),
        ),
        CheckConstraint(
            "status IN ('created', 'launching', 'running', 'finalizing', "
            "'cancelling', 'completed', 'failed', 'cancelled')",
            name="ck_managed_training_jobs_status",
        ),
        CheckConstraint(
            "outcome IN ('pending', 'succeeded', 'failed', 'cancelled')",
            name="ck_managed_training_jobs_outcome",
        ),
        CheckConstraint(
            "target_kind IN ('stub', 'modal')",
            name="ck_managed_training_jobs_target_kind",
        ),
        CheckConstraint(
            "provider_state IN ('pending', 'running', 'succeeded', 'failed', "
            "'cancelled', 'stopping', 'stopped', 'unknown')",
            name="ck_managed_training_jobs_provider_state",
        ),
        CheckConstraint(
            "length(request_hash) = 64",
            name="ck_managed_training_jobs_request_hash",
        ),
        CheckConstraint(
            "compute_size IN ('Modal: A10G', 'Modal: A100', 'Modal: 2xA100', "
            "'Modal: 4xA100', 'Modal: 8xA100', 'Modal: H100', "
            "'Modal: 2xH100', 'Modal: 4xH100', 'Modal: 8xH100')",
            name="ck_managed_training_jobs_compute_size",
        ),
        CheckConstraint(
            "(status = 'completed' AND outcome = 'succeeded' AND "
            "teardown_verified_at IS NOT NULL) OR "
            "(status = 'failed' AND outcome = 'failed' AND "
            "teardown_verified_at IS NOT NULL) OR "
            "(status = 'cancelled' AND outcome = 'cancelled' AND "
            "teardown_verified_at IS NOT NULL) OR "
            "status NOT IN ('completed', 'failed', 'cancelled')",
            name="ck_managed_training_jobs_terminal_truth",
        ),
        CheckConstraint(
            "runtime = 'lerobot'", name="ck_managed_training_jobs_runtime"
        ),
        CheckConstraint(
            "timeout_seconds BETWEEN 60 AND 86400",
            name="ck_managed_training_jobs_timeout",
        ),
        CheckConstraint(
            "max_steps BETWEEN 1 AND 2147483647",
            name="ck_managed_training_jobs_max_steps",
        ),
        CheckConstraint(
            "batch_size BETWEEN 1 AND 4096",
            name="ck_managed_training_jobs_batch_size",
        ),
        CheckConstraint(
            "log_every BETWEEN 1 AND max_steps",
            name="ck_managed_training_jobs_log_every",
        ),
        CheckConstraint(
            "save_every BETWEEN 1 AND max_steps",
            name="ck_managed_training_jobs_save_every",
        ),
        CheckConstraint(
            "seed BETWEEN 0 AND 2147483647",
            name="ck_managed_training_jobs_seed",
        ),
        CheckConstraint(
            "num_workers BETWEEN 0 AND 64",
            name="ck_managed_training_jobs_num_workers",
        ),
        CheckConstraint(
            "last_event_sequence BETWEEN 0 AND 9007199254740991",
            name="ck_managed_training_jobs_event_sequence",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    training_run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("training_runs.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    idempotency_key: Mapped[uuid.UUID] = mapped_column(
        Uuid, unique=True, nullable=False
    )
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="created", server_default="created"
    )
    outcome: Mapped[str] = mapped_column(
        String(16), nullable=False, default="pending", server_default="pending"
    )
    target_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    provider_state: Mapped[str] = mapped_column(
        String(16), nullable=False, default="pending", server_default="pending"
    )
    compute_size: Mapped[str] = mapped_column(String(32), nullable=False)
    runtime: Mapped[str] = mapped_column(
        String(32), nullable=False, default="lerobot", server_default="lerobot"
    )
    requested_dataset_revision: Mapped[str | None] = mapped_column(String(128))
    dataset_revision: Mapped[str | None] = mapped_column(String(40))
    requested_base_model_revision: Mapped[str | None] = mapped_column(String(128))
    base_model_revision: Mapped[str | None] = mapped_column(String(40))
    dataset_repo: Mapped[str] = mapped_column(String(255), nullable=False)
    base_model: Mapped[str] = mapped_column(String(255), nullable=False)
    output_model_repo: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False
    )
    output_private: Mapped[bool] = mapped_column(Boolean, nullable=False)
    output_marker_revision: Mapped[str | None] = mapped_column(String(40))
    output_revision: Mapped[str | None] = mapped_column(String(40))
    max_steps: Mapped[int] = mapped_column(Integer, nullable=False)
    batch_size: Mapped[int] = mapped_column(Integer, nullable=False)
    log_every: Mapped[int] = mapped_column(Integer, nullable=False)
    save_every: Mapped[int] = mapped_column(Integer, nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    num_workers: Mapped[int] = mapped_column(Integer, nullable=False)
    timeout_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    deadline_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    provider_app_id: Mapped[str | None] = mapped_column(String(255))
    provider_function_call_id: Mapped[str | None] = mapped_column(String(255))
    last_event_sequence: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, server_default="0"
    )
    event_gap: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text("false")
    )
    launch_attempted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    provider_launch_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    execution_finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    cancel_requested_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    teardown_verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    last_error: Mapped[str | None] = mapped_column(String(240))


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
