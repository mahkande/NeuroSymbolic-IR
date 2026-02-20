# NeuroGraph OS

NeuroGraph OS is a deterministic IR + graph reasoning layer for LLM systems.  
It compiles natural language into ISA-compliant IR, writes validated facts to graph memory, and applies evidence-first verification before persistence.

The goal is to reduce "LLM as pure next-token prediction" risk by adding:
- structured IR
- deterministic validators
- backward/evidence verification
- contradiction checks
- measurable quality and release gates

## 1. Clone
```bash
git clone https://github.com/mahkande/NeuroSymbolic-IR.git
cd NeuroSymbolic-IR
```

## 2. One-Command Docker Start (Neo4j + Dashboard)
```bash
docker compose up -d
```
Prerequisite: Docker Desktop must be installed and running.

Open:
- Dashboard: `http://localhost:8501`
- Neo4j Browser: `http://localhost:7474`

Stop:
```bash
docker compose down
```

This mode is the easiest for new users because Neo4j and dashboard are started together with working defaults.

## 3. Local Setup (Alternative)
### Windows (PowerShell, full setup)
```powershell
.\scripts\setup.ps1
```

### macOS/Linux (full setup)
```bash
bash scripts/setup.sh
```

### Optional lightweight setup (no Jina/provider extras)
```powershell
.\scripts\setup.ps1 -Lite
```
```bash
bash scripts/setup.sh lite
```

What these scripts do:
- create `.venv` if missing
- install dependencies from `requirements.txt` (or `requirements-lite.txt`)
- download required NLTK resources

### Environment file
Copy `.env.example` to your local env file/process variables and set your keys/backend values.

## 4. Run Dashboard (Local Mode)
```bash
streamlit run ui/dashboard.py
```

Open: `http://localhost:8501`

## 5. LLM Providers
The dashboard supports:
- Local (Ollama)
- Groq
- OpenAI
- Anthropic
- Google Gemini
- Mistral
- Together

Provider client packages can still be auto-installed from the dashboard when selected.

## 6. Graph / Vector Backends
### Neo4j (recommended)
```bash
docker compose -f docker-compose.neo4j.yml up -d
set COGNITIVE_GRAPH_BACKEND=neo4j
set NEO4J_URI=bolt://localhost:7687
set NEO4J_USER=neo4j
set NEO4J_PASSWORD=neo4j_password
set NEO4J_DATABASE=neo4j
```

### JSON fallback
```bash
set COGNITIVE_GRAPH_BACKEND=json
```

### Qdrant vector backend
```bash
set COGNITIVE_VECTOR_BACKEND=qdrant
set QDRANT_URL=http://localhost:6333
set COGNITIVE_VECTOR_COLLECTION=cognitive_graph
set COGNITIVE_EMBED_DIM=128
```

### Local vector fallback
```bash
set COGNITIVE_VECTOR_BACKEND=local
set COGNITIVE_VECTOR_LOCAL_PATH=memory/vector_index.jsonl
```

## 7. Jina Embeddings and Reranker
Implemented:
- `jina-embeddings-v3` integration for semantic embeddings
- `jina-reranker` integration for retrieval reranking

Used in:
- `core/embedding_pipeline.py`
- `core/hybrid_retriever.py`

Environment variables:
```bash
set COGNITIVE_EMBED_PROVIDER=jina
set COGNITIVE_JINA_EMBED_MODEL=jinaai/jina-embeddings-v3
set COGNITIVE_USE_RERANKER=1
set COGNITIVE_RERANK_PROVIDER=jina
set COGNITIVE_JINA_RERANK_MODEL=jinaai/jina-reranker-v2-base-multilingual
```

Important behavior:
- Repository clone alone does **not** install dependencies.
- Running setup scripts installs Python packages, including Jina stack in full mode.
- If Jina dependencies are missing, embedding falls back to hash embedding and reranker is skipped.
- If dependencies are installed, model weights are downloaded by Hugging Face on first use (internet required), then cached locally.

To enable Jina fully, install:
```bash
pip install sentence-transformers torch
```

## 8. Chat Ingestion, Bridge, and Intervention Layer
### VSCode/Codex chat ingestion
- Chat messages are ingested from session `.jsonl` files (not from direct chat API stream).
- Bridge status is visible in dashboard sidebar.
- Critical warning appears if bridge is stopped.

### Auto-start behavior
- Bridge + shadow listener can auto-start with:
```bash
set COGNITIVE_BRIDGE_AUTOSTART=1
```

### Run as standalone daemon (recommended for always-on ingestion)
```bash
python scripts/bridge_daemon.py
```

### Intervention layer
- Analyzes forwarded user chat
- Detects high-risk/contradiction signals
- Injects intervention hints into shadow flow
- Logs to `memory/interventions.jsonl`
- Outbox: `memory/intervention_outbox.txt`

Toggle:
```bash
set COGNITIVE_INTERVENTION_ENABLED=1
```

## 9. Major Implemented Changes (This Cycle)
- Strict IR schema parsing and schema rejection
- Unsupported-claim blocking
- Proof object generation (`claim -> supports -> verdict`)
- Backward verifier before memory write
- Trace persistence for claim proofs
- Retriever v2 dynamic graph scoring (confidence/provenance/depth)
- Evidence recall evaluation (`Recall@K`, `MRR`)
- Z3 profile + timeout controls + expanded constraints
- Counterexample-style reject messages
- Fallback KPI tracking (opcode distribution + useful-edge ratio)
- Fallback-only gold evaluation
- Release quality gates expanded with reliability metrics

## 10. Useful Scripts
```bash
python scripts/release_check.py
python scripts/load_test.py --requests 24 --workers 6 --listener-blocks 20
python scripts/eval_pipeline.py --dataset data/gold/gold_set_v1.jsonl --output reports/eval_report_latest.json
python scripts/ablation_runner.py --dataset data/gold/gold_set_v1.jsonl --report reports/retriever_ablation_report.json --strategy-out spec/retrieval_strategy.json
python scripts/eval_evidence_recall.py --dataset data/gold/gold_set_v1.jsonl --top-k 8 --out reports/evidence_recall_report.json
python scripts/eval_fallback_only.py --dataset data/gold/gold_set_v1.jsonl --out reports/fallback_eval_report.json
python scripts/offline_benchmark_v9.py --dataset data/gold/gold_set_v1.jsonl --out reports/offline_benchmark_v9.json
python scripts/threshold_gates.py --thresholds spec/release_thresholds.json
python scripts/canary_rollout.py --policy spec/canary_policy.json
python scripts/bridge_daemon.py
```

## 11. Project Structure (Key Files)
- `main.py`: main cognitive pipeline
- `ui/dashboard.py`: live dashboard
- `core/model_bridge.py`: provider bridge (local + APIs)
- `core/parser.py`: strict IR parser/schema checks
- `core/deterministic_verifier.py`: deterministic gate
- `core/backward_verifier.py`: evidence-first backward checks
- `core/evidence.py`: proof object builder
- `core/hybrid_retriever.py`: hybrid retrieval + Jina rerank
- `core/retrieval_scoring.py`: dynamic graph evidence scoring
- `core/intervention_layer.py`: chat intervention logic
- `core/vscode_chat_bridge.py`: session file bridge
- `memory/graph_store.py`: graph persistence
- `memory/vector_store.py`: vector persistence
- `TASKS.md`: implementation roadmap/status
