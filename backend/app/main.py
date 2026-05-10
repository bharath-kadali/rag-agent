from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import BadRequestError
from qdrant_client.http.exceptions import ResponseHandlingException

from .rag import index_document, retrieve, answer_with_groq, warmup_embedding_model
from .settings import Settings
from .storage import MetadataStore


load_dotenv()
settings = Settings.load()
store = MetadataStore(settings.app_data_dir)

# Render / Fly / etc.: localhost Qdrant is wrong — use Qdrant Cloud URL + API key in env.
_QDRANT_UNREACHABLE = (
    "Cannot reach Qdrant (connection refused). On cloud hosting, set QDRANT_URL to your "
    "Qdrant Cloud HTTPS URL (from cloud.qdrant.io) and QDRANT_API_KEY. "
    "Never use http://localhost:6333 on the server — that points at the app container, not your database."
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    preload = os.getenv("SKIP_EMBED_WARMUP", "").strip().lower() not in ("1", "true", "yes")
    if preload:
        warmup_embedding_model()
    yield


app = FastAPI(title="NotebookLM RAG (Groq + Qdrant)", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    indexed_chunks: int


class DocumentItem(BaseModel):
    document_id: str
    filename: str
    content_type: str
    created_at: float


class ChatRequest(BaseModel):
    document_id: str
    question: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=12)


class Citation(BaseModel):
    score: float
    text: str
    filename: str | None = None
    page: int | None = None
    row: int | None = None
    chunk_index: int | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True}


@app.get("/api/documents", response_model=list[DocumentItem])
def list_documents() -> list[DocumentItem]:
    return [DocumentItem(**d.__dict__) for d in store.list()]


@app.post("/api/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    meta = store.create(filename=file.filename, content_type=file.content_type or "application/octet-stream")

    try:
        res = index_document(
            qdrant_url=settings.qdrant_url,
            qdrant_api_key=settings.qdrant_api_key,
            document_id=meta.document_id,
            filename=meta.filename,
            content_type=meta.content_type,
            raw_bytes=raw,
        )
    except ResponseHandlingException as e:
        raise HTTPException(status_code=503, detail=_QDRANT_UNREACHABLE) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}") from e

    return UploadResponse(document_id=meta.document_id, filename=meta.filename, indexed_chunks=int(res["indexed_chunks"]))


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    meta = store.get(req.document_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Unknown document_id")

    try:
        contexts = retrieve(
            qdrant_url=settings.qdrant_url,
            qdrant_api_key=settings.qdrant_api_key,
            document_id=req.document_id,
            query=req.question,
            k=req.k,
        )
    except ResponseHandlingException as e:
        raise HTTPException(status_code=503, detail=_QDRANT_UNREACHABLE) from e
    if not settings.groq_api_key:
        raise HTTPException(
            status_code=500,
            detail="Missing GROQ_API_KEY. Copy backend/.env.example to backend/.env and set GROQ_API_KEY.",
        )

    try:
        answer = answer_with_groq(
            groq_api_key=settings.groq_api_key,
            groq_model=settings.groq_model,
            question=req.question,
            contexts=contexts,
        )
    except BadRequestError as e:
        # Commonly triggered by decommissioned/invalid models.
        raise HTTPException(status_code=400, detail=str(e)) from e
    return ChatResponse(answer=answer, citations=[Citation(**c) for c in contexts])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)

