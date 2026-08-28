"""Add control-plane persistence schema.

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-28
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def timestamps() -> list[sa.Column]:
    return [
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    ]


def upgrade() -> None:
    op.create_table(
        "robots",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("driver", sa.String(length=64), nullable=False),
        sa.Column("can_interface", sa.String(length=64), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column(
            "config", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        *timestamps(),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_table(
        "recordings",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("task", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("leader_robot_id", sa.Uuid(), nullable=True),
        sa.Column("follower_robot_id", sa.Uuid(), nullable=True),
        sa.Column("episode_count", sa.Integer(), nullable=False),
        sa.Column("duration_seconds", sa.Float(), nullable=False),
        sa.Column("hf_repo_id", sa.String(length=255), nullable=True),
        sa.Column(
            "metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        *timestamps(),
        sa.ForeignKeyConstraint(["follower_robot_id"], ["robots.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["leader_robot_id"], ["robots.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_recordings_created_at", "recordings", ["created_at"])
    op.create_table(
        "training_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("current_step", sa.Integer(), nullable=False),
        sa.Column("dataset_repo", sa.String(length=255), nullable=True),
        sa.Column("base_model", sa.String(length=255), nullable=True),
        sa.Column("runtime", sa.String(length=64), nullable=True),
        sa.Column("framework", sa.String(length=64), nullable=True),
        sa.Column("output_model_repo", sa.String(length=255), nullable=True),
        sa.Column("checkpoint_revision", sa.String(length=255), nullable=True),
        sa.Column("config", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        *timestamps(),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_training_runs_status", "training_runs", ["status"])
    op.create_table(
        "inference_endpoints",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("runtime", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("endpoint_url", sa.Text(), nullable=True),
        sa.Column("provider_app_id", sa.String(length=255), nullable=True),
        sa.Column("last_heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        *timestamps(),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "settings",
        sa.Column("key", sa.String(length=120), nullable=False),
        sa.Column("value", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        *timestamps(),
        sa.PrimaryKeyConstraint("key"),
    )
    op.create_table(
        "deployments",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("endpoint_id", sa.Uuid(), nullable=True),
        sa.Column("robot_id", sa.Uuid(), nullable=True),
        sa.Column("recording_id", sa.Uuid(), nullable=True),
        sa.Column("model_repo", sa.String(length=255), nullable=False),
        sa.Column("checkpoint_revision", sa.String(length=255), nullable=True),
        sa.Column("runtime", sa.String(length=64), nullable=False),
        sa.Column("compute_size", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("stopped_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("frequency_hz", sa.Float(), nullable=True),
        sa.Column("record_session", sa.Boolean(), nullable=False),
        *timestamps(),
        sa.ForeignKeyConstraint(["endpoint_id"], ["inference_endpoints.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["recording_id"], ["recordings.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["robot_id"], ["robots.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_deployments_status", "deployments", ["status"])


def downgrade() -> None:
    op.drop_index("ix_deployments_status", table_name="deployments")
    op.drop_table("deployments")
    op.drop_table("settings")
    op.drop_table("inference_endpoints")
    op.drop_index("ix_training_runs_status", table_name="training_runs")
    op.drop_table("training_runs")
    op.drop_index("ix_recordings_created_at", table_name="recordings")
    op.drop_table("recordings")
    op.drop_table("robots")

