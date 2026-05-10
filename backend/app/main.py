from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import BadRequestError

from .rag import index_document, retrieve, answer_with_groq
from .settings import Settings
from .storage import MetadataStore


load_dotenv()
settings = Settings.load()
store = MetadataStore(settings.app_data_dir)

app = FastAPI(title="NotebookLM RAG (Groq + Qdrant)")
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

    res = index_document(
        qdrant_url=settings.qdrant_url,
        qdrant_api_key=settings.qdrant_api_key,
        document_id=meta.document_id,
        filename=meta.filename,
        content_type=meta.content_type,
        raw_bytes=raw,
    )

    return UploadResponse(document_id=meta.document_id, filename=meta.filename, indexed_chunks=int(res["indexed_chunks"]))


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    meta = store.get(req.document_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Unknown document_id")

    contexts = retrieve(
        qdrant_url=settings.qdrant_url,
        qdrant_api_key=settings.qdrant_api_key,
        document_id=req.document_id,
        query=req.question,
        k=req.k,
    )
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

