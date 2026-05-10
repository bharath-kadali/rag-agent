import './style.css'

type DocumentItem = {
  document_id: string
  filename: string
  content_type: string
  created_at: number
}

type Citation = {
  score: number
  text: string
  filename?: string | null
  page?: number | null
  row?: number | null
  chunk_index?: number | null
}

type ChatResponse = {
  answer: string
  citations: Citation[]
}

const API_BASE =
  localStorage.getItem('apiBase') ??
  (import.meta.env.VITE_API_BASE as string | undefined) ??
  'http://localhost:8000'

const el = (selector: string) => document.querySelector(selector) as HTMLElement
const escapeHtml = (s: string) =>
  s.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')

function setStatus(msg: string, kind: 'ok' | 'warn' | 'err' | 'info' = 'info') {
  const box = el('#status')
  box.className = `status ${kind}`
  box.textContent = msg
}

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init)
  const text = await res.text()
  if (!res.ok) {
    let detail = text
    try {
      detail = JSON.parse(text)?.detail ?? text
    } catch {
      // ignore
    }
    throw new Error(detail || `Request failed: ${res.status}`)
  }
  return (text ? JSON.parse(text) : {}) as T
}

function render() {
  el('#app').innerHTML = `
  <div class="shell">
    <header class="topbar">
      <div>
        <div class="title">NotebookLM RAG</div>
        <div class="subtitle">Upload PDF/TXT/CSV → chat with grounded answers + citations</div>
      </div>
      <div class="api">
        <label>API</label>
        <input id="apiBase" type="text" value="${escapeHtml(API_BASE)}" />
        <button id="saveApi" class="btn secondary">Save</button>
      </div>
    </header>

    <div id="status" class="status info">Ready.</div>

    <main class="grid">
      <section class="card">
        <h2>1) Upload a document</h2>
        <p class="muted">Supported: PDF, TXT, CSV. CSV rows are converted to text and indexed.</p>
        <div class="row">
          <input id="file" type="file" accept=".pdf,.txt,.csv,application/pdf,text/plain,text/csv" />
          <button id="upload" class="btn">Upload & Index</button>
        </div>
      </section>

      <section class="card">
        <h2>2) Pick a document</h2>
        <div class="row">
          <button id="refreshDocs" class="btn secondary">Refresh</button>
        </div>
        <div id="docs" class="docs"></div>
      </section>

      <section class="card chat">
        <h2>3) Ask questions</h2>
        <div class="row">
          <input id="question" type="text" placeholder="Ask a question about the selected document..." />
          <button id="ask" class="btn" disabled>Ask</button>
        </div>
        <div id="answer" class="answer"></div>
        <details class="citations">
          <summary>Citations</summary>
          <div id="cites"></div>
        </details>
      </section>
    </main>
  </div>
  `
}

let selectedDocId: string | null = null

async function loadDocs() {
  const docs = await api<DocumentItem[]>('/api/documents')
  const wrap = el('#docs')
  if (!docs.length) {
    wrap.innerHTML = `<div class="muted">No documents uploaded yet.</div>`
    return
  }
  wrap.innerHTML = docs
    .map((d) => {
      const active = d.document_id === selectedDocId ? 'active' : ''
      return `
      <button class="doc ${active}" data-id="${escapeHtml(d.document_id)}">
        <div class="docTitle">${escapeHtml(d.filename)}</div>
        <div class="docMeta">${escapeHtml(d.content_type)}</div>
      </button>`
    })
    .join('')

  wrap.querySelectorAll('button.doc').forEach((b) => {
    b.addEventListener('click', () => {
      selectedDocId = (b as HTMLButtonElement).dataset.id || null
      ;(el('#ask') as HTMLButtonElement).disabled = !selectedDocId
      loadDocs().catch(() => {})
      setStatus(selectedDocId ? `Selected document: ${selectedDocId}` : 'No document selected.', 'info')
    })
  })
}

async function upload() {
  const input = el('#file') as HTMLInputElement
  const file = input.files?.[0]
  if (!file) throw new Error('Choose a file first.')
  const fd = new FormData()
  fd.append('file', file)

  setStatus('Uploading and indexing…', 'info')
  const res = await api<{ document_id: string; filename: string; indexed_chunks: number }>(
    '/api/documents/upload',
    {
      method: 'POST',
      body: fd,
    },
  )
  selectedDocId = res.document_id
  ;(el('#ask') as HTMLButtonElement).disabled = false
  setStatus(`Indexed ${res.indexed_chunks} chunks for "${res.filename}".`, 'ok')
  await loadDocs()
}

async function ask() {
  if (!selectedDocId) throw new Error('Select a document first.')
  const q = (el('#question') as HTMLInputElement).value.trim()
  if (!q) throw new Error('Enter a question.')

  ;(el('#ask') as HTMLButtonElement).disabled = true
  setStatus('Retrieving context and generating answer…', 'info')
  const res = await api<ChatResponse>('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ document_id: selectedDocId, question: q, k: 5 }),
  })

  el('#answer').innerHTML = `<div class="answerText">${escapeHtml(res.answer).replaceAll('\n', '<br/>')}</div>`
  el('#cites').innerHTML =
    res.citations
      .map((c, i) => {
        const loc = c.page ? `page ${c.page}` : c.row ? `row ${c.row}` : 'location n/a'
        const head = `${i + 1}. ${c.filename ?? ''} • ${loc} • score ${c.score.toFixed(3)}`
        return `
        <div class="cite">
          <div class="citeHead">${escapeHtml(head)}</div>
          <div class="citeBody">${escapeHtml(c.text).replaceAll('\n', '<br/>')}</div>
        </div>`
      })
      .join('') || `<div class="muted">No citations returned.</div>`

  setStatus('Done.', 'ok')
  ;(el('#ask') as HTMLButtonElement).disabled = false
}

function wire() {
  el('#refreshDocs').addEventListener('click', () => loadDocs().catch((e) => setStatus(String(e), 'err')))
  el('#upload').addEventListener('click', () => upload().catch((e) => setStatus(String(e), 'err')))
  el('#ask').addEventListener('click', () => ask().catch((e) => setStatus(String(e), 'err')))
  el('#question').addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter') ask().catch((e) => setStatus(String(e), 'err'))
  })
  el('#saveApi').addEventListener('click', () => {
    const v = (el('#apiBase') as HTMLInputElement).value.trim()
    if (v) {
      localStorage.setItem('apiBase', v)
      location.reload()
    }
  })
}

render()
wire()
api('/api/health')
  .then(() => setStatus('Backend reachable. Upload a document to begin.', 'ok'))
  .catch(() => setStatus('Backend not reachable. Start backend on http://localhost:8000.', 'warn'))
  .finally(() => loadDocs().catch(() => {}))
