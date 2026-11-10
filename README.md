# AI Customer Support Agent

A runnable Streamlit application that implements a four-stage support workflow with AutoGen, hybrid FAISS/BM25 retrieval, a persistent provenance graph, confidence-based escalation, CSV audit logs, optional Google Sheets mirroring, and guarded long-term memory.

## Architecture

```text
Terminal 1: mcp_server.py (Streamable HTTP at /mcp)
    ├── health_check
    ├── search_product_scope
    ├── search_support_knowledge
    ├── get_knowledge_stats
    ├── get_recent_ticket_summary
    ├── get_graph_stats
    └── search_knowledge_graph
                │
                ▼
Terminal 2: Streamlit app.py + AutoGen workflow
    │
    ▼
1. Customer Query Agent
   MCP product-scope search → category, urgency, sentiment, priority, summary
    │
    ▼
2. Knowledge Base Agent
   MCP support search → policies, FAQs, catalogs, and auto-resolved cases
    │
    ▼
3. Resolution Agent
   Grounded answer + exact source filenames + confidence score
    │
    ├── confidence ≥ 70%, grounded, in scope, non-high-priority → RESOLVED
    └── otherwise → ESCALATED (SMTP only when explicitly enabled)
    │
    ▼
4. Log Agent
   CSV audit record; only resolved grounded cases are added to support FAISS
```

The orchestration mechanism is an explicit, linear AutoGen `GraphFlow` topology — not an open-ended group chat. `support_system/agents.py::build_support_graphflow` wires the three AutoGen `AssistantAgent` instances (Agents 1–3) through `DiGraphBuilder` with a single directed edge per hop (query → knowledge → resolution) and a `SourceMatchTermination` condition that halts the run immediately after the resolution agent's single turn — no looping, no round robin. This guarantees the requested fixed agent order and a deterministic halting point using AutoGen's own orchestration primitive. `SupportWorkflow.process()` still invokes the three agents' `classify`/`retrieve`+`synthesize`/`resolve` methods directly in that same fixed order (rather than driving the GraphFlow team inline), because the full unit-test suite exercises `SupportWorkflow` with plain Python test doubles that are not AutoGen `ChatAgent` objects and cannot be scheduled inside a GraphFlow team; this keeps determinism and test coverage intact while GraphFlow remains the proven, AutoGen-native topology for the real agents. Agent 4 performs deterministic state changes so the LLM cannot fabricate a successful log or memory write, and is never wired into the GraphFlow team.

## Main features

- **Agent 1:** product-scope RAG, classification, sentiment, urgency, and priority.
- **Agent 2:** separate support vector DB for policies, FAQs, catalogs, and clearly labeled auto-resolved history.
- **Agent 3:** evidence-only response, confidence score, source filenames, and 70% gate.
- **Agent 4:** UTC timestamped CSV logging, escalation decision, and long-term memory.
- Persistent local FAISS indexes under `data/`, combined with Okapi BM25 using deterministic Reciprocal Rank Fusion (RRF).
- Stable chunk IDs and score provenance (`vector`, `BM25`, `fusion`) are retained for inspection and citation validation.
- Semantic embeddings default to a small, fully local `sentence-transformers` model (no API key, no network call); an OpenAI-compatible embeddings endpoint or lightweight lexical hashing can be configured instead. The app and MCP settings must match.
- The existing Knowledge Base synthesis call performs bounded LLM reranking using only known chunk IDs, with deterministic fallback order.
- A safe JSON-persisted NetworkX provenance graph links documents, pages, chunks, and document kinds; it never replaces source evidence.
- A separate read-only MCP server exposes seven narrowly scoped support and graph tools over Streamable HTTP.
- Optional Google Sheets mirroring uses a durable local JSONL outbox, masked allowlisted fields, RAW cell input, and an explicit admin sync action. Local CSV remains authoritative.
- Per-ticket total and stage latency are persisted for operational review.
- The Streamlit app verifies MCP health and uses MCP retrieval for the AutoGen workflow.
- PDF page, filename, kind, and chunk metadata are preserved for citations.
- Streamlit includes the SupportIQ operations dashboard and a localhost-only Admin Console with ticket state, latency, retrieval, graph, escalation, and Sheets-sync viewers.
- SMTP sends only when explicitly enabled and fully configured; the provided environment template routes support escalation to `support@example.com` (replace with your real support mailbox).
- Retrieved PDF text is treated as untrusted data and cannot override agent instructions.

## Prerequisites

- Python **3.10–3.12 recommended**
- An OpenAI API key, or an OpenAI-compatible local endpoint that supports tool calling and structured JSON output
- Text-based PDFs. Scanned PDFs must be OCR-processed before upload.

## Installation

```bash
cd /root/.hermes/DURAI/customer-support-agent
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

If `python3.12 -m venv` reports that `ensurepip` is unavailable on Debian/Ubuntu, install `python3.12-venv`, or create the environment with `virtualenv`.

### OpenAI

Edit `.env`:

```dotenv
OPENAI_API_KEY=your-key-from-a-secret-store
MODEL_NAME=gpt-4o-mini
```

Do not commit `.env`.

### Ollama or another OpenAI-compatible server

The selected model must support function/tool calling and structured JSON output:

```dotenv
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=local-model
MODEL_NAME=qwen2.5:7b
```

Compatibility varies by model and server. If structured output or tool calling is unsupported, use an OpenAI model or a compatible local model.

### Hybrid retrieval and semantic embeddings

Every search gates vector candidates by their raw similarity before combining eligible FAISS and positive-evidence BM25 rankings with RRF. `RETRIEVAL_MIN_SCORE` defaults to `0.20`; this is a safer packaged starting point, not a universal cutoff. Calibrate it against representative queries whenever the embedding model or provider changes. The returned `score` is gated relevance in `[0,1]`, while `fusion_score` is retained separately for ordering provenance.

Three embedding backends are supported via `EMBEDDING_BACKEND`:

- `local` (default): loads `sentence-transformers/all-MiniLM-L6-v2` once per process and embeds locally — real semantic retrieval, fully offline, no API key, no network call after the model is cached.
- `hashing`: deterministic lexical hashing. Lightweight and dependency-light, but not semantic. Useful as a minimal fallback if `sentence-transformers`/`torch` cannot be installed.
- `openai-compatible`: delegates to a remote OpenAI-compatible `/embeddings` endpoint, for example:

```dotenv
EMBEDDING_BACKEND=openai-compatible
EMBEDDING_BASE_URL=http://127.0.0.1:11434/v1
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSION=768
EMBEDDING_API_KEY=local-only
EMBEDDING_TIMEOUT_SECONDS=30
ENABLE_LLM_RERANK=true
```

The embedding model and dimension must be identical in the MCP and Streamlit processes. Changing them rebuilds the persisted FAISS projection from stored chunk text. The `local` backend always reports a fixed 384-dimensional vector regardless of `EMBEDDING_DIMENSION`; that setting only applies to `openai-compatible`. Confirm the exact model dimension from the selected provider before enabling it.

## Run: MCP first, Streamlit second

The application intentionally uses two processes. Keep both terminals open.

### Windows PowerShell

**Terminal 1 — MCP server:**

```powershell
cd .\customer-support-agent
.\run_mcp_server.ps1
```

**Terminal 2 — Streamlit:**

```powershell
cd .\customer-support-agent
.\run_app.ps1
```

If PowerShell blocks local scripts, use the direct commands instead:

```powershell
# Terminal 1
.\.venv\Scripts\python.exe .\mcp_server.py

# Terminal 2
.\.venv\Scripts\python.exe -m scripts.check_mcp
.\.venv\Scripts\python.exe -m streamlit run .\app.py --server.address 127.0.0.1
```

### Linux/macOS

Make the helpers executable once, then start each process in a separate terminal:

```bash
chmod +x run_mcp_server.sh run_app.sh
./run_mcp_server.sh   # Terminal 1
./run_app.sh          # Terminal 2
```

Direct commands are equivalent:

```bash
# Terminal 1
.venv/bin/python mcp_server.py

# Terminal 2
.venv/bin/python -m scripts.check_mcp
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1
```

The MCP endpoint is `http://127.0.0.1:8765/mcp` and Streamlit opens at `http://127.0.0.1:8501`. The dashboard shows **MCP Tool Server · Online** when the connection is ready.

> **Local-only security boundary:** the bundled MCP server deliberately rejects non-loopback binds, and the startup scripts bind Streamlit to `127.0.0.1`. The dashboard has no multi-user authentication or tenant isolation. Do not expose either process to a LAN or the internet; add a reviewed authentication/TLS gateway and application authorization before any shared deployment.

Then:

1. Upload product manuals, policies, FAQs, catalogs, and resolved-issue PDFs.
2. Click **Ingest PDFs**.
3. Enter a customer ID and query.
4. Click **Run four-agent workflow**.
5. Review the answer, confidence, sources, agent timeline, and decision.

To try the included non-PDF sample:

```bash
PYTHONPATH=. python scripts/seed_sample.py
# Then use the two-terminal MCP + Streamlit commands above.
```

Example query: `I bought my first subscription 10 days ago. When will an approved refund arrive?`

## Human review and escalation email

Escalated and open tickets appear in **Overview → Human support review**. A reviewer can inspect the ticket, edit the AI draft, choose whether the ticket becomes **Resolved** or remains **Open**, and explicitly click **Approve reply and send escalation email**. The email is sent only after that button is pressed; a successful send updates the CSV review fields and ticket status.

The supplied `.env.example` routes escalation notifications and approved review messages to the placeholder `support@example.com`. Complete the private `.env` with your SMTP sender configuration:

```dotenv
ENABLE_ESCALATION_EMAIL=true
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-sender@gmail.com
SMTP_PASSWORD=load-a-Google-App-Password-from-your-private-env
SMTP_SENDER=your-sender@gmail.com
SUPPORT_EMAIL=support@example.com
```

Do not use your normal Gmail password and do not commit `.env`. For Gmail, enable two-step verification and create an App Password. Test with a non-production ticket first. Automatic workflow escalation also uses this mailbox when SMTP is fully configured. Email delivery transfers the ticket ID, support context, and approved reply to the configured mailbox; common email, phone, and payment-number patterns are deterministically redacted before transmission.

## Optional Google Sheets mirror

Google Sheets is disabled by default. The local CSV remains authoritative, and ticket processing never waits for Google. When enabled, every durable ticket state change creates an idempotent, privacy-reduced event in `data/google_sheets_outbox.jsonl`. A local administrator must explicitly approve and run synchronization from **Admin Console → Google Sheets synchronization**.

1. Create a dedicated Google Cloud service account and enable the Google Sheets API.
2. Create a spreadsheet with tabs named exactly `Ticket Events` and `Current Tickets`; the application initializes their headers when they are empty.
3. Share only that spreadsheet with the service-account email as editor.
4. Store the downloaded service-account JSON outside the project and ZIP.
5. Configure the private `.env`:

```dotenv
ENABLE_GOOGLE_SHEETS_SYNC=true
GOOGLE_SERVICE_ACCOUNT_FILE=C:/secure/supportiq-sheets-service-account.json
GOOGLE_SHEETS_SPREADSHEET_ID=your-spreadsheet-id
```

Only allowlisted operational fields are mirrored: ticket ID, masked customer ID, category, priority, status, confidence, escalation state, timestamps, and latency. Raw queries, answers, review replies, contact/payment data, delivery IDs, and credentials are excluded. Formula prefixes are neutralized and Sheets writes use `RAW` input mode. Failed events remain local for inspection; no real Google write occurs during automated tests.

## Admin Console

The Admin Console shows ticket state, escalation and review delivery state, total and per-stage latency, hybrid retrieval provenance, bounded graph evidence, and Sheets outbox state. Retrieval inspection displays source/page/chunk IDs and component rankings instead of raw numeric vectors. Graph previews are redacted and bounded. The console has no multi-user authentication, so the packaged app must remain bound to `127.0.0.1` unless a reviewed authentication, authorization, and TLS layer is added.

## Data and memory policy

- `data/tickets.csv` stores customer IDs, queries, classifications, answers, sources, decisions, email-delivery state, and human-review audit fields. Treat it as personal/customer data. The dashboard and downloadable CSV mask customer IDs and common email, phone, and payment-number patterns by default.
- `data/product_scope_faiss/` is Agent 1's product-scope store.
- `data/support_knowledge_faiss/` is Agent 2's support and long-term-memory store.
- `data/knowledge_graph.json` is the locked, JSON-only NetworkX provenance projection. It stores typed document/page/chunk/kind relationships and redacted previews, never pickled objects.
- `data/google_sheets_outbox.jsonl` exists only when Sheets mirroring is completely configured. It contains allowlisted masked snapshots and delivery state, not raw ticket conversations.
- Only resolved, grounded, in-scope, non-high-priority cases at or above the confidence threshold create an `auto_resolved_case` marker. The marker contains only fixed decision metadata and validated evidence-source names—not the customer query, answer, customer ID, or ticket ID.
- Auto-resolved markers tell later runs to retrieve the current indexed evidence again; generated answer text is never promoted into shared RAG memory.
- Escalated or low-confidence answers are never learned. This prevents uncertain output from recursively poisoning future retrieval.
- Escalated drafts are withheld from customers; the UI shows a neutral human-review acknowledgement instead.
- Retrieval uses a minimum relevance score, and cited filenames must match retrieved evidence before auto-resolution.
- CSV formula prefixes are neutralized, PDF size/page/text/chunk limits are enforced, and local index writes use file locks.
- For production, add retention rules, access control, encryption, stronger entity redaction, deletion workflows, and preferably human approval before promoting learned cases.

## Tests and checks

```bash
source .venv/bin/activate
pytest -q
python -m compileall -q app.py support_system scripts tests
```

The test suite covers hybrid FAISS/BM25/RRF retrieval, stable chunk IDs, embedding migrations, bounded reranking, graph persistence/redaction, Sheets outbox idempotency, latency instrumentation, FAISS persistence, CSV formula neutralization, the exact 70% boundary, escalation, citation validation, PDF limits, and the memory-write gate.

## Project layout

```text
app.py                         SupportIQ dashboard and local Admin Console
mcp_server.py                  Standalone Streamable HTTP MCP server (seven tools)
run_mcp_server.ps1/.sh         Start the MCP process
run_app.ps1/.sh                Verify MCP port, then start Streamlit
support_system/mcp_client.py   AutoGen-compatible typed MCP workbench client
support_system/agents.py       Four agent classes
support_system/workflow.py     Sequential orchestration and decision gates
support_system/rag.py          Hybrid FAISS/BM25/RRF retrieval and embedders
support_system/knowledge_graph.py Safe JSON-persisted NetworkX provenance graph
support_system/ingestion.py    PDF extraction, chunking, and metadata
support_system/storage.py      CSV audit logger
support_system/sheets_mirror.py Locked outbox and optional Google Sheets transport
support_system/email_service.py SMTP adapter with disabled-by-default gate
support_system/config.py       Environment configuration and model client
knowledge/                     Sample knowledge
scripts/seed_sample.py         Offline sample seeding
tests/                         Automated tests
```

## Troubleshooting

### `MCP Tool Server: Offline` or connection refused

Start `mcp_server.py` in a separate terminal before Streamlit. Confirm the MCP process prints `SupportIQ MCP ready at http://127.0.0.1:8765/mcp`. Keep `MCP_SERVER_URL`, `MCP_HOST`, and `MCP_PORT` aligned. The bundled unauthenticated server accepts loopback binds only; remote deployment requires a separately reviewed authenticated/TLS architecture.

### `search_product_scope is not strict`

AutoGen structured output requires tools to be declared as strict function tools. The current project wraps both RAG tools with `FunctionTool(..., strict=True)`. If this message appears, you are running an older copy of `support_system/agents.py`; download the latest archive, reinstall its requirements, and restart Streamlit.

## Current limitations

- The default `local` embedder (`sentence-transformers/all-MiniLM-L6-v2`) is a small, genuinely semantic sentence encoder. For larger/hosted models, configure the supplied OpenAI-compatible embedder backend; for a dependency-light fallback with no semantic understanding, use `hashing`.
- The NetworkX graph is a bounded provenance viewer, not an unrestricted LLM-extracted business-relationship truth layer.
- Google Sheets is a manually synchronized reporting mirror, not a transactional database or source of truth.
- Image-only PDFs need OCR before ingestion.
- The real LLM path requires a configured model endpoint and was intentionally not invoked by automated tests, so tests do not incur API cost or send data externally.
