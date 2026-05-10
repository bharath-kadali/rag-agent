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
- **Qdrant Cloud**: `QDRANT_URL` = your cluster URL (e.g. `https://xxxx.cloud.qdrant.io`), and `QDRANT_API_KEY` = key from the Qdrant Cloud console. Leave `QDRANT_API_KEY` empty when using local Docker Qdrant.

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

## Deploying backend (e.g. Render) + Qdrant

The API must reach a **real** Qdrant instance over the network.

- **Do not** set `QDRANT_URL=http://localhost:6333` on Render. On the server, `localhost` is the Render container itself — nothing is listening → `Connection refused` (errno 111).
- Create a cluster on [Qdrant Cloud](https://cloud.qdrant.io), then on Render set:
  - **`QDRANT_URL`** = cluster URL (usually `https://…cloud.qdrant.io`, from the dashboard).
  - **`QDRANT_API_KEY`** = API key from the same dashboard.
- Redeploy the backend after changing env vars.

## Notes

- **Chunking**: character-based chunking with overlap (see `backend/app/rag.py::chunk_text`).
- **Embeddings**: local embeddings via `fastembed` (no external embedding API required).
- **Grounding**: the Groq prompt instructs the model to answer *only* from retrieved context and include citations like `[1]`, `[2]`.

