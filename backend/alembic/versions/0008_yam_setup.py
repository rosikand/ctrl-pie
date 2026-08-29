"""Persist one non-secret YAM rig setup.

Revision ID: 0008
Revises: 0007
Create Date: 2026-08-29
"""

from alembic import op
import sqlalchemy as sa

revision = "0008"
down_revision = "0007"
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
        "yam_setups",
        sa.Column("id", sa.String(length=16), nullable=False),
        sa.Column("mode", sa.String(length=16), nullable=False),
        sa.Column("can_interface", sa.String(length=15), nullable=False),
        sa.Column("leader_port", sa.String(length=200), nullable=False),
        sa.Column("mujoco_xml_path", sa.String(length=1024), nullable=False),
        sa.Column("leader_calibration_id", sa.String(length=64), nullable=False),
        sa.Column("leader_calibration_dir", sa.String(length=1024), nullable=False),
        sa.Column(
            "auto_restore",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column("last_attempt_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_connected_at", sa.DateTime(timezone=True), nullable=True),
        *timestamps(),
        sa.CheckConstraint("id = 'primary'", name="ck_yam_setups_singleton"),
        sa.CheckConstraint(
            "mode IN ('mock', 'hardware')", name="ck_yam_setups_mode"
        ),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("yam_setups")
