# NeuroGraph OS

## 1. Project Goal

The goal of this project is to move beyond the classical **prompt -> source code -> token prediction** pipeline by building an AI system that is:

- deterministic,
- analyzable,
- and capable of inverse reasoning,

through a bridge between **human intent** and a **machine-near semantic representation (LLVM IR)**.

---

## 2. High-Level Concept

Classical LLM systems:
- treat code as token sequences,
- have limited determinism and traceability.

This project instead:
- handles programs at the **LLVM IR** level,
- represents IR as **graphs (CFG / DFG)**,
- converts graph structures into **embedding vectors**,
- performs **reasoning and inverse reasoning** over these structures.

## What It Does
- Compiles input text into ISA-schema-compliant IR instructions.
- Writes IR chains into persistent graph memory.
- Runs consistency/conflict checks (safe mode + optional native Z3).
- Provides a live dashboard for graph state, opcode distribution, and runtime events.
- Supports file/session-based ingestion through shadow/listener pipelines.
- Uses chunking for longer inputs to improve IR coverage.

## IR Categories
- `ONTOLOGICAL`: `DEF_ENTITY`, `DEF_CONCEPT`, `ISA`, `EQUIV`, `ATTR`
- `EPISTEMIC`: `KNOW`, `BELIEVE`, `DOUBT`, `WONDER`, `ASSUME`
- `TELEOLOGICAL`: `WANT`, `AVOID`, `GOAL`, `INTEND`, `EVAL`, `DO`
- `CAUSAL_LOGIC`: `CAUSE`, `PREVENT`, `IMPLY`, `OPPOSE`, `TRIGGER`
- `DEONTIC`: `MUST`, `MAY`, `FORBID`, `CAN`
- `TEMPORAL`: `BEFORE`, `WHILE`, `START`, `END`
- `META_COGNITIVE`: `REFLECT`, `CORRECT`, `ANALOGY`

ISA schema: `spec/isa_v1.json`

## Architecture Overview
- `core/model_bridge.py`: LLM client, prompting, fallback compilation
- `core/nlp_utils.py`: normalization, grammar filter, relation rebalance
- `core/logic_engine.py`: consistency checking
- `memory/knowledge_graph.py`: graph persistence and access rules
- `main.py`: main pipeline (chunking, cache, validation, persistence)
- `ui/dashboard.py`: live control panel

## Deterministic Reasoning Flow
1. A text claim is submitted.
2. An IR chain is generated with ISA opcodes.
3. IR is written to graph memory.
4. Consistency and contradiction checks are applied.
5. Output is retained with a verifiable reasoning trace.

## Local Setup
1. Prepare Python 3.11+.
2. Create and activate a virtual environment:
```bash
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```
3. Install dependencies:
```bash
pip install -U networkx streamlit streamlit-agraph watchdog groq z3-solver zeyrek nltk
```
4. Set environment variables:
```bash
set GROQ_API_KEY=YOUR_KEY
set GROQ_MODEL=llama-3.3-70b-versatile
```

## Run Dashboard
```bash
streamlit run ui/dashboard.py
```

After launch:
- Use `Clear Memory` and `Clear IR Cache` in the sidebar for a clean run.
- Paste text into the input box and press Enter to trigger the IR -> Graph pipeline.

## Run via CLI (Alternative)
```bash
python main.py
```

## Push to GitHub
From the project directory:
```bash
git init
git branch -M main
git add .
git commit -m "Initial commit: NeuroGraph OS"
git remote add origin https://github.com/mahkande/NeuroSymbolic-IR.git
git push -u origin main
```

If remote already exists:
```bash
git remote set-url origin https://github.com/mahkande/NeuroSymbolic-IR.git
git push -u origin main
```

## Design Notes
- Long inputs are processed in chunks to avoid single-pass bottlenecks.
- The graph is configured to keep multiple opcode edges between the same node pair.
- Shadow/bridge channels include filtering to reduce runtime noise.

## Security and Privacy
- Use environment variables for API keys instead of hardcoding secrets.
- Runtime graph/log/cache artifacts should stay out of version control via `.gitignore`.

## Roadmap
- Per-opcode precision/recall metrics dashboard
- Subgraph quality scoring and automatic alert thresholds
- Dataset-driven relation classifier training (hybrid pipeline)
