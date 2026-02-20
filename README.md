# NeuroGraph OS

A deterministic IR + Graph cognitive operating layer. It compiles natural language into ISA-compliant IR, writes it into persistent graph memory, validates it with logic/verifier gates, and exposes full runtime observability in the dashboard.

This project is designed to go beyond the classic `prompt -> source code -> token prediction` pipeline by connecting human intent to a machine-near semantic representation (LLVM-IR-like ISA). Instead of a black-box response generator, it provides a controllable reasoning system with deterministic verification, contradiction management, inverse/abductive reasoning, measurable quality metrics, and production safeguards (release gates, threshold checks, canary rollback). Use it when you need traceable, auditable, and operationally reliable reasoning flows in real systems.

## Clone and Setup
1. Clone the repository:
```bash
git clone https://github.com/mahkande/NeuroSymbolic-IR.git
cd NeuroSymbolic-IR
```
2. Prepare Python 3.11+.
3. Create and activate a virtual environment:
```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
```
4. Install core dependencies:
```bash
pip install -U pip
pip install networkx streamlit streamlit-agraph watchdog z3-solver zeyrek nltk neo4j
```
Note: API client packages (`groq`, `openai`, `anthropic`, `google-generativeai`, `mistralai`, `together`) can be auto-installed from the dashboard when a provider is selected.

## Access the Dashboard
Run:
```bash
streamlit run ui/dashboard.py
```
Then open Streamlit in your browser (usually `http://localhost:8501`).

## LLM Provider Selection (Dashboard)
The sidebar includes an `LLM Provider` menu.

Supported providers:
- `Local (Ollama)`
- `Groq`
- `OpenAI`
- `Anthropic`
- `Google Gemini`
- `Mistral`
- `Together`

Behavior:
- If `Local` is selected, no API key is required (uses Ollama endpoint via `OLLAMA_URL`, default `http://localhost:11434`).
- If an API provider is selected:
  - API key input is shown.
  - Model name can be configured.
  - Required client package can be installed automatically.

## Backend Configuration
### Graph Backend
Neo4j is recommended:
```bash
docker compose -f docker-compose.neo4j.yml up -d
set COGNITIVE_GRAPH_BACKEND=neo4j
set NEO4J_URI=bolt://localhost:7687
set NEO4J_USER=neo4j
set NEO4J_PASSWORD=neo4j_password
set NEO4J_DATABASE=neo4j
```
If Neo4j is unavailable:
```bash
set COGNITIVE_GRAPH_BACKEND=json
```

### Vector Backend
Qdrant:
```bash
set COGNITIVE_VECTOR_BACKEND=qdrant
set QDRANT_URL=http://localhost:6333
set COGNITIVE_VECTOR_COLLECTION=cognitive_graph
set COGNITIVE_EMBED_DIM=128
```
Local fallback:
```bash
set COGNITIVE_VECTOR_BACKEND=local
set COGNITIVE_VECTOR_LOCAL_PATH=memory/vector_index.jsonl
```

## Run via CLI
```bash
python main.py
```

## Latest Updates (TASKS Phase 7)
Phase 7 is completed:
- Step 1: Inference noise control
  - Transitive guard
  - Low-value pruning
  - Confidence policy
- Step 2: Gold dataset and real evaluation
  - Gold set v1
  - Eval pipeline (precision/recall/F1)
  - Regression suite
- Step 3: Verifier gate hardening
  - Strict gate mode
  - Conflict risk score
  - Policy profiles (`safe`, `balanced`, `aggressive`)
- Step 4: Retriever ablation and impact analysis
  - `vector-only`, `graph-only`, `hybrid` ablation runner
  - KPI report (quality + latency + token cost)
  - Default retrieval strategy selection
- Step 5: CI/CD release gate
  - CI workflow: `compile + release_check + load_test_smoke + eval_pipeline`
  - Threshold gate (quality/latency cutoffs)
  - Canary rollout with automatic rollback decision

## Useful Scripts
- `python scripts/release_check.py`
- `python scripts/load_test.py --requests 24 --workers 6 --listener-blocks 20`
- `python scripts/eval_pipeline.py --dataset data/gold/gold_set_v1.jsonl --output reports/eval_report_latest.json`
- `python scripts/ablation_runner.py --dataset data/gold/gold_set_v1.jsonl --report reports/retriever_ablation_report.json --strategy-out spec/retrieval_strategy.json`
- `python scripts/threshold_gates.py --thresholds spec/release_thresholds.json`
- `python scripts/canary_rollout.py --policy spec/canary_policy.json`

## Project Structure (Summary)
- `main.py`: main pipeline
- `ui/dashboard.py`: dashboard
- `core/model_bridge.py`: local/API provider bridge
- `core/deterministic_verifier.py`: verifier gate
- `core/inference_engine.py`: inference + noise controls
- `memory/graph_store.py`: graph persistence
- `memory/vector_store.py`: vector persistence

Detailed task tracking: `TASKS.md`
