"""
FastAPI application: enrollment, inference, lookup, events, health.
"""
from pathlib import Path

from core.config import load_repo_dotenv
from fastapi import FastAPI

load_repo_dotenv()

from src.api.routes import animals, inference, events, health, tracklets


async def lifespan(app: FastAPI):
    from core.config import load_config
    from core.storage.minio_storage import get_minio_storage

    app.state.config = load_config()
    app.state.minio = None
    try:
        storage = get_minio_storage()
        if storage is not None:
            storage.ensure_bucket()
            storage.ensure_bucket(storage.crops_bucket)
            storage.ensure_bucket(storage.videos_bucket)
            app.state.minio = storage
    except Exception:
        # API segue sem MinIO (ex.: dev sem container); uploads usam disco local.
        app.state.minio = None
    root = Path(__file__).resolve().parent.parent.parent
    faiss_path = root / "core" / "data" / "faiss_index"
    if not faiss_path.exists():
        faiss_path = root / "data" / "faiss_index"
    if faiss_path.exists():
        try:
            from src.ai.reid.faiss_store import FAISSStore
            app.state.faiss_store = FAISSStore(index_path=faiss_path)
            app.state.faiss_store.load()
        except Exception:
            app.state.faiss_store = FAISSStore()
    else:
        from src.ai.reid.faiss_store import FAISSStore
        app.state.faiss_store = FAISSStore()
    yield
    app.state.faiss_store.save()


app = FastAPI(
    title="Novus API",
    description="Identificação de bovinos por drone: enrollment, inferência, eventos.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router, tags=["health"])
app.include_router(animals.router, prefix="/animals", tags=["animals"])
app.include_router(inference.router, prefix="/inference", tags=["inference"])
app.include_router(events.router, prefix="/events", tags=["events"])
app.include_router(tracklets.router, prefix="/tracklets", tags=["tracklets"])


@app.get("/")
async def root():
    return {"service": "novus", "docs": "/docs"}
