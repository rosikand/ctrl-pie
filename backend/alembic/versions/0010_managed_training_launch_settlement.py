"""Track ambiguous managed provider launch settlement.

Revision ID: 0010
Revises: 0009
Create Date: 2026-08-29
"""

from alembic import op
import sqlalchemy as sa


revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "managed_training_jobs",
        sa.Column("provider_launch_started_at", sa.DateTime(timezone=True)),
    )


def downgrade() -> None:
    op.drop_column("managed_training_jobs", "provider_launch_started_at")
