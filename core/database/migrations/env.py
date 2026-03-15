import os
from logging.config import fileConfig

from sqlalchemy.engine import Connection
from alembic import context

from core.database.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)
target_metadata = Base.metadata


def get_url():
    return os.getenv("DATABASE_URL", "postgresql+asyncpg://cattle:cattle@localhost:5432/cattle_reid").replace(
        "postgresql+asyncpg", "postgresql"
    )


def run_migrations_offline() -> None:
    context.configure(
        url=get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    from sqlalchemy import create_engine
    url = get_url()
    sync_engine = create_engine(url)
    with sync_engine.connect() as connection:
        do_run_migrations(connection)
    sync_engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
