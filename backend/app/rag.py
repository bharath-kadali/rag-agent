from __future__ import annotations

import io
import os
import re
import uuid
from typing import Any, Iterable

import pandas as pd
from fastembed import TextEmbedding
from openai import OpenAI
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


COLLECTION_NAME = "notebooklm_chunks"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def chunk_text(text: str, *, chunk_size: int = 900, chunk_overlap: int = 150) -> list[str]:
    """
    Simple character-based chunking with overlap.
    """
    text = _clean_text(text)
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - chunk_overlap)
    return chunks


def parse_pdf(data: bytes) -> list[dict[str, Any]]:
    reader = PdfReader(io.BytesIO(data))
    pages: list[dict[str, Any]] = []
    for idx, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        txt = _clean_text(txt)
        if txt:
            pages.append({"text": txt, "page": idx + 1})
    return pages


def parse_txt(data: bytes) -> list[dict[str, Any]]:
    txt = data.decode("utf-8", errors="ignore")
    txt = _clean_text(txt)
    return [{"text": txt, "page": None}] if txt else []


def parse_csv(data: bytes, *, max_rows: int = 2000) -> list[dict[str, Any]]:
    df = pd.read_csv(io.BytesIO(data))
    if len(df) > max_rows:
        df = df.head(max_rows)
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(df.itertuples(index=False), start=1):
        parts = []
        for col, val in zip(df.columns, row):
            if pd.isna(val):
                continue
            parts.append(f"{col}: {val}")
        text = _clean_text("\n".join(parts))
        if text:
            rows.append({"text": text, "row": i})
    return rows


def ensure_collection(client: QdrantClient, vector_size: int) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME in existing:
        return
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
    )


def _iter_embeddings(embedding_model: TextEmbedding, texts: Iterable[str]) -> Iterable[list[float]]:
    for vec in embedding_model.embed(texts):
        yield list(vec)


def index_document(
    *,
    qdrant_url: str,
    document_id: str,
    filename: str,
    content_type: str,
    raw_bytes: bytes,
) -> dict[str, Any]:
    client = QdrantClient(url=qdrant_url)
    embedding_model = TextEmbedding(model_name=EMBED_MODEL_NAME)
    vector_size = len(next(_iter_embeddings(embedding_model, ["dimension_probe"])))
    ensure_collection(client, vector_size)

    if content_type == "application/pdf" or filename.lower().endswith(".pdf"):
        units = parse_pdf(raw_bytes)
        unit_kind = "page"
    elif content_type.startswith("text/") or filename.lower().endswith(".txt"):
        units = parse_txt(raw_bytes)
        unit_kind = "text"
    elif content_type in ("text/csv", "application/csv") or filename.lower().endswith(".csv"):
        units = parse_csv(raw_bytes)
        unit_kind = "row"
    else:
        # try as text fallback
        units = parse_txt(raw_bytes)
        unit_kind = "text"

    points: list[qm.PointStruct] = []
    all_texts: list[str] = []
    payloads: list[dict[str, Any]] = []

    chunk_index = 0
    for u in units:
        base_text = u["text"]
        base_meta = {k: v for k, v in u.items() if k != "text"}
        for chunk in chunk_text(base_text):
            chunk_index += 1
            all_texts.append(chunk)
            payloads.append(
                {
                    "doc_id": document_id,
                    "filename": filename,
                    "content_type": content_type,
                    "unit_kind": unit_kind,
                    **base_meta,
                    "chunk_index": chunk_index,
                    "text": chunk,
                }
            )

    if not all_texts:
        return {"indexed_chunks": 0}

    # Qdrant point IDs must be an unsigned int or a UUID. Use a deterministic UUID per (doc_id, chunk_idx).
    ns = uuid.uuid5(uuid.NAMESPACE_URL, "notebooklm_chunks")
    for idx, (vec, pl) in enumerate(zip(_iter_embeddings(embedding_model, all_texts), payloads), start=1):
        pid = uuid.uuid5(ns, f"{document_id}:{idx}")
        points.append(qm.PointStruct(id=str(pid), vector=vec, payload=pl))

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return {"indexed_chunks": len(points)}


def retrieve(
    *,
    qdrant_url: str,
    document_id: str,
    query: str,
    k: int = 5,
) -> list[dict[str, Any]]:
    client = QdrantClient(url=qdrant_url)
    embedding_model = TextEmbedding(model_name=EMBED_MODEL_NAME)
    query_vec = list(next(_iter_embeddings(embedding_model, [query])))
    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec,
        limit=k,
        query_filter=qm.Filter(must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=document_id))]),
        with_payload=True,
    )
    out: list[dict[str, Any]] = []
    for h in hits:
        p = h.payload or {}
        out.append(
            {
                "score": float(h.score),
                "text": p.get("text", ""),
                "filename": p.get("filename"),
                "page": p.get("page"),
                "row": p.get("row"),
                "chunk_index": p.get("chunk_index"),
            }
        )
    return out


def answer_with_groq(
    *,
    groq_api_key: str,
    groq_model: str,
    question: str,
    contexts: list[dict[str, Any]],
) -> str:
    client = OpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")

    context_blob = "\n\n".join(
        [
            f"[{i+1}] (file={c.get('filename')}, page={c.get('page')}, row={c.get('row')}, chunk={c.get('chunk_index')})\n{c.get('text','')}"
            for i, c in enumerate(contexts)
        ]
    )

    system = (
        "You are a RAG assistant. Answer ONLY using the provided context.\n"
        "If the context is insufficient, say you don't know based on the document.\n"
        "Cite sources inline like [1], [2] corresponding to the context blocks.\n"
        "Do not use external knowledge.\n\n"
        f"CONTEXT:\n{context_blob}"
    )

    resp = client.chat.completions.create(
        model=groq_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()

