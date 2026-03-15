from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/models")
async def models_status(request: Request):
    store = getattr(request.app.state, "faiss_store", None)
    if store is None:
        return {"faiss": "not_loaded"}
    return {"faiss": "loaded", "num_vectors": len(store)}
