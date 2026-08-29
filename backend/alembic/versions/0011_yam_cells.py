"""Persist a normalized multi-arm YAM cell.

Revision ID: 0011
Revises: 0010
Create Date: 2026-08-29
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


json_type = sa.JSON().with_variant(
    postgresql.JSONB(astext_type=sa.Text()), "postgresql"
)


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
    # The V1.1 yam_setups topology remains readable through the legacy
    # compatibility adapter until an operator explicitly saves a V1.2 cell.
    # Historical unattended-motion consent is not carried across the V1.2
    # hardware architecture boundary. Operators may explicitly re-enable it
    # after reviewing the retained legacy topology in the new control plane.
    op.execute(
        sa.text("UPDATE yam_setups SET auto_restore = false WHERE auto_restore = true")
    )
    op.create_table(
        "yam_cells",
        sa.Column("id", sa.String(length=16), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("i2rt_root", sa.String(length=1024), nullable=False),
        sa.Column("i2rt_commit", sa.String(length=40), nullable=False),
        sa.Column(
            "pair_ports",
            json_type,
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
        sa.Column(
            "auto_restore",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column("last_attempt_at", sa.DateTime(timezone=True)),
        sa.Column("last_connected_at", sa.DateTime(timezone=True)),
        *timestamps(),
        sa.CheckConstraint("id = 'primary'", name="ck_yam_cells_singleton"),
        sa.CheckConstraint(
            "length(i2rt_commit) = 40", name="ck_yam_cells_i2rt_commit"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "yam_cell_arms",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("cell_id", sa.String(length=16), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("logical_id", sa.String(length=120), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("pair_id", sa.String(length=120)),
        sa.Column("group_id", sa.String(length=120)),
        sa.Column("side", sa.String(length=64)),
        sa.Column("transport_kind", sa.String(length=16), nullable=False),
        sa.Column("stable_identity", sa.String(length=256), nullable=False),
        sa.Column("end_effector_kind", sa.String(length=64), nullable=False),
        sa.Column("frame_map_path", sa.String(length=1024)),
        sa.Column("soft_limits_path", sa.String(length=1024)),
        sa.Column("mujoco_xml_path", sa.String(length=1024)),
        sa.Column("calibration_id", sa.String(length=64)),
        sa.Column("calibration_dir", sa.String(length=1024)),
        sa.Column(
            "config",
            json_type,
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
        *timestamps(),
        sa.CheckConstraint("position >= 0", name="ck_yam_cell_arms_position"),
        sa.CheckConstraint(
            "role IN ('leader', 'follower')", name="ck_yam_cell_arms_role"
        ),
        sa.CheckConstraint(
            "transport_kind IN ('socketcan', 'serial')",
            name="ck_yam_cell_arms_transport_kind",
        ),
        sa.CheckConstraint(
            "end_effector_kind IN ('yam_teaching_handle', 'linear_4310', "
            "'crank_4310', 'gello', 'none')",
            name="ck_yam_cell_arms_end_effector_kind",
        ),
        sa.CheckConstraint(
            "(calibration_id IS NULL AND calibration_dir IS NULL) OR "
            "(calibration_id IS NOT NULL AND calibration_dir IS NOT NULL)",
            name="ck_yam_cell_arms_calibration_pair",
        ),
        sa.CheckConstraint(
            "(transport_kind = 'socketcan' AND role = 'leader' AND "
            "end_effector_kind = 'yam_teaching_handle') OR "
            "(transport_kind = 'socketcan' AND role = 'follower' AND "
            "end_effector_kind IN ('linear_4310', 'crank_4310')) OR "
            "(transport_kind = 'serial' AND role = 'leader' AND "
            "end_effector_kind = 'gello')",
            name="ck_yam_cell_arms_supported_shape",
        ),
        sa.ForeignKeyConstraint(
            ["cell_id"], ["yam_cells.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "cell_id", "position", name="uq_yam_cell_arms_position"
        ),
        sa.UniqueConstraint(
            "cell_id", "logical_id", name="uq_yam_cell_arms_logical_id"
        ),
        sa.UniqueConstraint("cell_id", "name", name="uq_yam_cell_arms_name"),
        sa.UniqueConstraint(
            "cell_id",
            "transport_kind",
            "stable_identity",
            name="uq_yam_cell_arms_stable_identity",
        ),
    )
    op.create_index(
        "ix_yam_cell_arms_pair", "yam_cell_arms", ["cell_id", "pair_id"]
    )
    op.create_index(
        "ix_yam_cell_arms_group", "yam_cell_arms", ["cell_id", "group_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_yam_cell_arms_group", table_name="yam_cell_arms")
    op.drop_index("ix_yam_cell_arms_pair", table_name="yam_cell_arms")
    op.drop_table("yam_cell_arms")
    op.drop_table("yam_cells")
