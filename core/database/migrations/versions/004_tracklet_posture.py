"""Add posture_label and posture_score to tracklets."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("tracklets", sa.Column("posture_label", sa.String(64), nullable=True))
    op.add_column("tracklets", sa.Column("posture_score", sa.Float, nullable=True))


def downgrade() -> None:
    op.drop_column("tracklets", "posture_score")
    op.drop_column("tracklets", "posture_label")
