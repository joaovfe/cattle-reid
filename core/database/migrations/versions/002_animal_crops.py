"""Add animal_crops table."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "animal_crops",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("animal_id", sa.Integer(), nullable=False),
        sa.Column("tracklet_id", sa.Integer(), nullable=True),
        sa.Column("source_path", sa.String(length=768), nullable=False),
        sa.Column("frame_index", sa.Integer(), nullable=True),
        sa.Column("bbox", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["animal_id"], ["animals.id"]),
        sa.ForeignKeyConstraint(["tracklet_id"], ["tracklets.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_animal_crops_animal_id", "animal_crops", ["animal_id"], unique=False)
    op.create_index("ix_animal_crops_tracklet_id", "animal_crops", ["tracklet_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_animal_crops_tracklet_id", table_name="animal_crops")
    op.drop_index("ix_animal_crops_animal_id", table_name="animal_crops")
    op.drop_table("animal_crops")

