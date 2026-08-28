"""Persist the compute provider kind for safe lifecycle recovery.

Revision ID: 0005
Revises: 0004
Create Date: 2026-08-28
"""

from alembic import op
import sqlalchemy as sa

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "deployments",
        sa.Column(
            "target_kind",
            sa.String(length=16),
            nullable=False,
            server_default=sa.text("'stub'"),
        ),
    )
    op.create_check_constraint(
        "ck_deployments_target_kind",
        "deployments",
        "target_kind IN ('stub', 'modal')",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_deployments_target_kind",
        "deployments",
        type_="check",
    )
    op.drop_column("deployments", "target_kind")
