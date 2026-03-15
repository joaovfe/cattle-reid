"""Initial schema: animals, embeddings, tracklets, events, image_metadata."""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "animals",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("external_id", sa.String(128), nullable=True),
        sa.Column("name", sa.String(256), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_animals_external_id", "animals", ["external_id"], unique=False)
    op.create_table(
        "embeddings",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("animal_id", sa.Integer(), nullable=False),
        sa.Column("source_image_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["animal_id"], ["animals.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_embeddings_animal_id", "embeddings", ["animal_id"], unique=False)
    op.create_table(
        "tracklets",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("video_source", sa.String(512), nullable=True),
        sa.Column("start_frame", sa.Integer(), nullable=True),
        sa.Column("end_frame", sa.Integer(), nullable=True),
        sa.Column("track_id", sa.Integer(), nullable=False),
        sa.Column("resolved_animal_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["resolved_animal_id"], ["animals.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_tracklets_resolved_animal_id", "tracklets", ["resolved_animal_id"], unique=False)
    op.create_index("ix_tracklets_track_id", "tracklets", ["track_id"], unique=False)
    op.create_table(
        "events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("animal_id", sa.Integer(), nullable=True),
        sa.Column("tracklet_id", sa.Integer(), nullable=True),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("timestamp", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["animal_id"], ["animals.id"]),
        sa.ForeignKeyConstraint(["tracklet_id"], ["tracklets.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_events_animal_id", "events", ["animal_id"], unique=False)
    op.create_index("ix_events_event_type", "events", ["event_type"], unique=False)
    op.create_index("ix_events_timestamp", "events", ["timestamp"], unique=False)
    op.create_index("ix_events_tracklet_id", "events", ["tracklet_id"], unique=False)
    op.create_table(
        "image_metadata",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("source_path", sa.String(512), nullable=True),
        sa.Column("frame_index", sa.Integer(), nullable=True),
        sa.Column("video_id", sa.String(256), nullable=True),
        sa.Column("timestamp", sa.DateTime(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("image_metadata")
    op.drop_index("ix_events_tracklet_id", "events")
    op.drop_index("ix_events_timestamp", "events")
    op.drop_index("ix_events_event_type", "events")
    op.drop_index("ix_events_animal_id", "events")
    op.drop_table("events")
    op.drop_index("ix_tracklets_track_id", "tracklets")
    op.drop_index("ix_tracklets_resolved_animal_id", "tracklets")
    op.drop_table("tracklets")
    op.drop_index("ix_embeddings_animal_id", "embeddings")
    op.drop_table("embeddings")
    op.drop_index("ix_animals_external_id", "animals")
    op.drop_table("animals")
