"""Add stable hardware driver identity to robots.

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-28
"""

from alembic import op
import sqlalchemy as sa

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("robots", sa.Column("driver_id", sa.String(length=120), nullable=True))
    op.execute("UPDATE robots SET driver_id = 'legacy:' || CAST(id AS text)")
    op.alter_column("robots", "driver_id", nullable=False)
    op.create_unique_constraint("uq_robots_driver_id", "robots", ["driver_id"])


def downgrade() -> None:
    op.drop_constraint("uq_robots_driver_id", "robots", type_="unique")
    op.drop_column("robots", "driver_id")
