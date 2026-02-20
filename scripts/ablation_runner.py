#!/usr/bin/env python
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.embedding_pipeline import GraphEmbeddingPipeline
from core.hybrid_retriever import HybridRetriever
from main import load_global_graph
from memory.vector_store import VectorStore


MODES = ("vector-only", "graph-only", "hybrid")


def _safe_div(a, b):
    return (a / b) if b else 0.0


def _token_count(text: str) -> int:
    return len(str(text or "").split())


def load_dataset(path: str):
    rows = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row.setdefault("id", f"row_{i}")
            row.setdefault("text", "")
            row.setdefault("gold", [])
            rows.append(row)
    return rows


def _gold_ops(row):
    ops = set()
    for instr in row.get("gold", []):
        if isinstance(instr, dict):
            op = str(instr.get("op", "")).upper().strip()
            if op:
                ops.add(op)
    return ops


def _pred_ops_from_evidence(evidence):
    ops = set()
    for e in evidence or []:
        meta = e.get("meta", {}) or {}
        rel = str(meta.get("relation", "")).upper().strip()
        if rel:
            ops.add(rel)
            continue
        txt = str(e.get("text", ""))
        parts = txt.split()
        if len(parts) >= 4 and parts[0] == "edge":
            ops.add(parts[2].upper())
    return ops


def evaluate_mode(rows, retriever, mode: str, top_k: int):
    tp = fp = fn = 0
    latencies = []
    query_tokens = 0
    evidence_tokens = 0
    sample = []

    for row in rows:
        q = row.get("text", "")
        focus_terms = q.lower().split()[:2]
        t0 = time.perf_counter()
        evidence = retriever.retrieve(q, focus_terms=focus_terms, top_k=top_k, mode=mode)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat_ms)

        gold_ops = _gold_ops(row)
        pred_ops = _pred_ops_from_evidence(evidence)
        inter = gold_ops & pred_ops
        tp += len(inter)
        fp += len(pred_ops - gold_ops)
        fn += len(gold_ops - pred_ops)

        query_tokens += _token_count(q)
        evidence_tokens += sum(_token_count(e.get("text", "")) for e in evidence)
        if len(sample) < 20:
            sample.append(
                {
                    "id": row.get("id"),
                    "gold_ops": sorted(gold_ops),
                    "pred_ops": sorted(pred_ops),
                    "hit": bool(inter),
                    "latency_ms": round(lat_ms, 2),
                }
            )

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    avg_latency = _safe_div(sum(latencies), len(latencies))
    p95_latency = sorted(latencies)[int(max(0, len(latencies) * 0.95) - 1)] if latencies else 0.0
    total_cost_tokens = query_tokens + evidence_tokens

    return {
        "mode": mode,
        "quality": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        },
        "latency_ms": {
            "avg": round(avg_latency, 2),
            "p95": round(p95_latency, 2),
        },
        "token_cost": {
            "query_tokens": query_tokens,
            "evidence_tokens": evidence_tokens,
            "total": total_cost_tokens,
        },
        "samples": sample,
    }


def choose_default(results):
    if not results:
        return "hybrid"
    best_f1 = max(float(r["quality"]["f1"]) for r in results)
    # Keep quality first: candidate must be within 95% of best F1.
    floor = best_f1 * 0.95
    candidates = [r for r in results if float(r["quality"]["f1"]) >= floor]
    if not candidates:
        candidates = list(results)

    scored = []
    for row in candidates:
        l = row["latency_ms"]
        c = row["token_cost"]
        # Lower is better for latency and cost.
        tie_score = (_safe_div(1.0, 1.0 + l["avg"]) * 0.6) + (_safe_div(1.0, 1.0 + c["total"] / 10000.0) * 0.4)
        scored.append((tie_score, row["mode"]))
    scored.sort(reverse=True)
    return scored[0][1]


def main():
    parser = argparse.ArgumentParser(description="Retriever ablation runner: vector-only vs graph-only vs hybrid.")
    parser.add_argument("--dataset", default="data/gold/gold_set_v1.jsonl")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--report", default="reports/retriever_ablation_report.json")
    parser.add_argument("--strategy-out", default="spec/retrieval_strategy.json")
    args = parser.parse_args()

    rows = load_dataset(args.dataset)
    graph = load_global_graph()
    vec = VectorStore(dim=int(os.getenv("COGNITIVE_EMBED_DIM", "128")))
    pipe = GraphEmbeddingPipeline(vec, dim=vec.dim)
    pipe.index_graph(graph, focus="", max_subgraphs=48)
    retriever = HybridRetriever(graph, vec, dim=vec.dim)

    results = [evaluate_mode(rows, retriever, mode=m, top_k=max(1, args.top_k)) for m in MODES]
    default_mode = choose_default(results)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": args.dataset,
        "dataset_size": len(rows),
        "results": results,
        "default_strategy": default_mode,
        "selection_note": "Selected by weighted score over F1, latency, and token cost.",
    }

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    strategy = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "default_mode": default_mode,
        "candidates": list(MODES),
        "source_report": str(out).replace("\\", "/"),
    }
    strategy_path = Path(args.strategy_out)
    strategy_path.parent.mkdir(parents=True, exist_ok=True)
    strategy_path.write_text(json.dumps(strategy, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"default_strategy": default_mode, "report": str(out), "strategy": str(strategy_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
