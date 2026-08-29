"""Add durable managed training jobs.

Revision ID: 0009
Revises: 0008
Create Date: 2026-08-29
"""

from alembic import op
import sqlalchemy as sa

revision = "0009"
down_revision = "0008"
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
        "managed_training_jobs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("training_run_id", sa.Uuid(), nullable=False),
        sa.Column("idempotency_key", sa.Uuid(), nullable=False),
        sa.Column("request_hash", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), server_default="created", nullable=False),
        sa.Column("outcome", sa.String(length=16), server_default="pending", nullable=False),
        sa.Column("target_kind", sa.String(length=16), nullable=False),
        sa.Column("provider_state", sa.String(length=16), server_default="pending", nullable=False),
        sa.Column("compute_size", sa.String(length=32), nullable=False),
        sa.Column("runtime", sa.String(length=32), server_default="lerobot", nullable=False),
        sa.Column("requested_dataset_revision", sa.String(length=128)),
        sa.Column("dataset_revision", sa.String(length=40)),
        sa.Column("requested_base_model_revision", sa.String(length=128)),
        sa.Column("base_model_revision", sa.String(length=40)),
        sa.Column("dataset_repo", sa.String(length=255), nullable=False),
        sa.Column("base_model", sa.String(length=255), nullable=False),
        sa.Column("output_model_repo", sa.String(length=255), nullable=False),
        sa.Column("output_private", sa.Boolean(), nullable=False),
        sa.Column("output_marker_revision", sa.String(length=40)),
        sa.Column("output_revision", sa.String(length=40)),
        sa.Column("max_steps", sa.Integer(), nullable=False),
        sa.Column("batch_size", sa.Integer(), nullable=False),
        sa.Column("log_every", sa.Integer(), nullable=False),
        sa.Column("save_every", sa.Integer(), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("num_workers", sa.Integer(), nullable=False),
        sa.Column("timeout_seconds", sa.Integer(), nullable=False),
        sa.Column("deadline_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("provider_app_id", sa.String(length=255)),
        sa.Column("provider_function_call_id", sa.String(length=255)),
        sa.Column("last_event_sequence", sa.BigInteger(), server_default="0", nullable=False),
        sa.Column("event_gap", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("launch_attempted_at", sa.DateTime(timezone=True)),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("execution_finished_at", sa.DateTime(timezone=True)),
        sa.Column("cancel_requested_at", sa.DateTime(timezone=True)),
        sa.Column("teardown_verified_at", sa.DateTime(timezone=True)),
        sa.Column("last_error", sa.String(length=240)),
        *timestamps(),
        sa.CheckConstraint(
            "status IN ('created', 'launching', 'running', 'finalizing', "
            "'cancelling', 'completed', 'failed', 'cancelled')",
            name="ck_managed_training_jobs_status",
        ),
        sa.CheckConstraint(
            "outcome IN ('pending', 'succeeded', 'failed', 'cancelled')",
            name="ck_managed_training_jobs_outcome",
        ),
        sa.CheckConstraint("target_kind IN ('stub', 'modal')", name="ck_managed_training_jobs_target_kind"),
        sa.CheckConstraint(
            "provider_state IN ('pending', 'running', 'succeeded', 'failed', "
            "'cancelled', 'stopping', 'stopped', 'unknown')",
            name="ck_managed_training_jobs_provider_state",
        ),
        sa.CheckConstraint("length(request_hash) = 64", name="ck_managed_training_jobs_request_hash"),
        sa.CheckConstraint(
            "compute_size IN ('Modal: A10G', 'Modal: A100', 'Modal: 2xA100', "
            "'Modal: 4xA100', 'Modal: 8xA100', 'Modal: H100', "
            "'Modal: 2xH100', 'Modal: 4xH100', 'Modal: 8xH100')",
            name="ck_managed_training_jobs_compute_size",
        ),
        sa.CheckConstraint(
            "(status = 'completed' AND outcome = 'succeeded' AND "
            "teardown_verified_at IS NOT NULL) OR "
            "(status = 'failed' AND outcome = 'failed' AND "
            "teardown_verified_at IS NOT NULL) OR "
            "(status = 'cancelled' AND outcome = 'cancelled' AND "
            "teardown_verified_at IS NOT NULL) OR "
            "status NOT IN ('completed', 'failed', 'cancelled')",
            name="ck_managed_training_jobs_terminal_truth",
        ),
        sa.CheckConstraint("runtime = 'lerobot'", name="ck_managed_training_jobs_runtime"),
        sa.CheckConstraint("timeout_seconds BETWEEN 60 AND 86400", name="ck_managed_training_jobs_timeout"),
        sa.CheckConstraint("max_steps BETWEEN 1 AND 2147483647", name="ck_managed_training_jobs_max_steps"),
        sa.CheckConstraint("batch_size BETWEEN 1 AND 4096", name="ck_managed_training_jobs_batch_size"),
        sa.CheckConstraint("log_every BETWEEN 1 AND max_steps", name="ck_managed_training_jobs_log_every"),
        sa.CheckConstraint("save_every BETWEEN 1 AND max_steps", name="ck_managed_training_jobs_save_every"),
        sa.CheckConstraint("seed BETWEEN 0 AND 2147483647", name="ck_managed_training_jobs_seed"),
        sa.CheckConstraint("num_workers BETWEEN 0 AND 64", name="ck_managed_training_jobs_num_workers"),
        sa.CheckConstraint(
            "last_event_sequence BETWEEN 0 AND 9007199254740991",
            name="ck_managed_training_jobs_event_sequence",
        ),
        sa.ForeignKeyConstraint(["training_run_id"], ["training_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("training_run_id"),
        sa.UniqueConstraint("idempotency_key"),
        sa.UniqueConstraint("output_model_repo"),
    )
    op.create_index("ix_managed_training_jobs_status", "managed_training_jobs", ["status"])
    op.create_index(
        "ix_managed_training_jobs_deadline",
        "managed_training_jobs",
        ["status", "deadline_at"],
    )
    op.create_index(
        "uq_managed_training_jobs_one_active",
        "managed_training_jobs",
        [sa.literal_column("(1)")],
        unique=True,
        postgresql_where=sa.text(
            "status IN ('created', 'launching', 'running', 'finalizing', 'cancelling')"
        ),
        sqlite_where=sa.text(
            "status IN ('created', 'launching', 'running', 'finalizing', 'cancelling')"
        ),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_managed_training_jobs_one_active", table_name="managed_training_jobs"
    )
    op.drop_index("ix_managed_training_jobs_deadline", table_name="managed_training_jobs")
    op.drop_index("ix_managed_training_jobs_status", table_name="managed_training_jobs")
    op.drop_table("managed_training_jobs")
