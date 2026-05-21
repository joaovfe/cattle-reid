"""Enable pgvector and persist embeddings in embeddings table."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

EMBEDDING_DIMENSION = 384


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.add_column("embeddings", sa.Column("embedding", Vector(EMBEDDING_DIMENSION), nullable=True))


def downgrade() -> None:
    op.drop_column("embeddings", "embedding")
    op.execute("DROP EXTENSION IF EXISTS vector")
