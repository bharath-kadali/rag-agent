# NotebookLM-style RAG (Groq + Qdrant)

Web app that lets you upload **PDF / TXT / CSV** and then **chat with the document** using a full RAG pipeline:
ingestion → chunking → embedding → vector DB storage → retrieval → grounded generation.

## Prereqs

- Docker (for Qdrant)
- Python 3.10+ (backend)
- Node.js (frontend)

## 1) Start Qdrant (vector DB)

From the repo root:

```bash
docker compose up -d
```

Qdrant UI will be on `http://localhost:6333/dashboard`.

## 2) Backend (FastAPI)

```bash
cd backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `backend/.env` and set:

- `GROQ_API_KEY=...`
- `GROQ_MODEL=llama-3.3-70b-versatile` (optional; this is the default)

Run:

```bash
python -m uvicorn app.main:app --reload --port 8000
```

Backend health: `http://localhost:8000/api/health`

## 3) Frontend (Vite)

```bash
cd frontend
copy .env.example .env
npm install
npm run dev
```

Open the UI at `http://localhost:5173`.

## Notes

- **Chunking**: character-based chunking with overlap (see `backend/app/rag.py::chunk_text`).
- **Embeddings**: local embeddings via `fastembed` (no external embedding API required).
- **Grounding**: the Groq prompt instructs the model to answer *only* from retrieved context and include citations like `[1]`, `[2]`.

