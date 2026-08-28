"""Persist each deployment's hard lifetime backstop.

Revision ID: 0006
Revises: 0005
Create Date: 2026-08-28
"""

from alembic import op
import sqlalchemy as sa

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "deployments",
        sa.Column(
            "timeout_seconds",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("1800"),
        ),
    )
    op.create_check_constraint(
        "ck_deployments_timeout_seconds",
        "deployments",
        "timeout_seconds BETWEEN 1 AND 1800",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_deployments_timeout_seconds",
        "deployments",
        type_="check",
    )
    op.drop_column("deployments", "timeout_seconds")
