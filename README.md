# NeuroGraph OS

Deterministic IR + Graph tabanli bilissel isletim sistemi. Metni ISA uyumlu IR'e cevirir, graph hafizaya yazar, mantik/verifier katmanlarindan gecirir ve dashboard uzerinden izlenebilir hale getirir.

## Clone ve Kurulum
1. Repoyu klonlayin:
```bash
git clone https://github.com/mahkande/NeuroSymbolic-IR.git
cd NeuroSymbolic-IR
```
2. Python 3.11+ hazirlayin.
3. Sanal ortam olusturun ve aktif edin:
```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
```
4. Temel paketleri kurun:
```bash
pip install -U pip
pip install networkx streamlit streamlit-agraph watchdog z3-solver zeyrek nltk neo4j
```
Not: API istemci paketleri (groq, openai, anthropic, google-generativeai, mistralai, together) dashboarddan provider secince arka planda otomatik kurulabilir.

## Dashboard'a Ulasim
Dashboard'i baslatin:
```bash
streamlit run ui/dashboard.py
```
Ardindan tarayicida Streamlit adresine gidin (genelde `http://localhost:8501`).

## LLM Provider Secimi (Dashboard)
Sidebar'da `LLM Provider` menusu vardir.

Desteklenen secenekler:
- `Local (Ollama)`
- `Groq`
- `OpenAI`
- `Anthropic`
- `Google Gemini`
- `Mistral`
- `Together`

Davranis:
- `Local` secilirse API key gerekmez, Ollama endpointi kullanilir (`OLLAMA_URL`, varsayilan `http://localhost:11434`).
- API provider secilirse:
  - API key alani acilir.
  - Model adi girilebilir.
  - Gerekli istemci paketi otomatik kurulabilir.

## Backend Ayarlari
### Graph Backend
Neo4j onerilir:
```bash
docker compose -f docker-compose.neo4j.yml up -d
set COGNITIVE_GRAPH_BACKEND=neo4j
set NEO4J_URI=bolt://localhost:7687
set NEO4J_USER=neo4j
set NEO4J_PASSWORD=neo4j_password
set NEO4J_DATABASE=neo4j
```
Neo4j yoksa:
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
Fallback local:
```bash
set COGNITIVE_VECTOR_BACKEND=local
set COGNITIVE_VECTOR_LOCAL_PATH=memory/vector_index.jsonl
```

## Calistirma (CLI)
```bash
python main.py
```

## Latest Updates (TASKS Faz 7)
Faz 7 tamamlandi:
- Adim 1: Inference noise kontrolu
  - Transitive guard
  - Low-value pruning
  - Confidence policy
- Adim 2: Gold dataset ve gercek degerlendirme
  - Gold set v1
  - Eval pipeline (precision/recall/F1)
  - Regression suite
- Adim 3: Verifier gate sertlestirme
  - Strict gate mode
  - Conflict risk score
  - Policy profiles (`safe`, `balanced`, `aggressive`)
- Adim 4: Retriever ablation ve etki analizi
  - `vector-only`, `graph-only`, `hybrid` ablation runner
  - KPI raporu (kalite + latency + token maliyeti)
  - Default retrieval strategy secimi
- Adim 5: CI/CD release gate
  - CI workflow: `compile + release_check + load_test_smoke + eval_pipeline`
  - Threshold gate (quality/latency esikleri)
  - Canary rollout + otomatik rollback karari

## Faydalı Scriptler
- `python scripts/release_check.py`
- `python scripts/load_test.py --requests 24 --workers 6 --listener-blocks 20`
- `python scripts/eval_pipeline.py --dataset data/gold/gold_set_v1.jsonl --output reports/eval_report_latest.json`
- `python scripts/ablation_runner.py --dataset data/gold/gold_set_v1.jsonl --report reports/retriever_ablation_report.json --strategy-out spec/retrieval_strategy.json`
- `python scripts/threshold_gates.py --thresholds spec/release_thresholds.json`
- `python scripts/canary_rollout.py --policy spec/canary_policy.json`

## Proje Dosyalari (Ozet)
- `main.py`: ana pipeline
- `ui/dashboard.py`: dashboard
- `core/model_bridge.py`: local/API provider bridge
- `core/deterministic_verifier.py`: verifier gate
- `core/inference_engine.py`: inference + noise kontrol
- `memory/graph_store.py`: graph persistence
- `memory/vector_store.py`: vector persistence

Detayli gorev takibi: `TASKS.md`
