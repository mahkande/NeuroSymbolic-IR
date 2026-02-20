#!/usr/bin/env python
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.embedding_pipeline import GraphEmbeddingPipeline
from core.hybrid_retriever import HybridRetriever
from main import load_global_graph
from memory.vector_store import VectorStore


def _load_jsonl(path: str):
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    for i, line in enumerate(p.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        row.setdefault("id", f"row_{i}")
        row.setdefault("text", "")
        row.setdefault("gold", [])
        rows.append(row)
    return rows


def _tok(text: str):
    return [t for t in str(text or "").lower().replace("_", " ").split() if t]


def _evidence_match(instr: dict, ev: dict) -> bool:
    op = str(instr.get("op", "")).upper().strip()
    args = [str(a) for a in (instr.get("args", []) or [])]
    meta = ev.get("meta", {}) or {}
    rel = str(meta.get("relation", "")).upper().strip()
    text = str(ev.get("text", "")).lower()
    if rel and op and rel != op:
        return False
    arg_hits = 0
    for a in args:
        toks = _tok(a)
        if not toks:
            continue
        if any(t in text for t in toks):
            arg_hits += 1
    return arg_hits >= 1


def evaluate(rows, retriever, top_k: int):
    total = 0
    hit = 0
    rr_sum = 0.0
    details = []
    for row in rows:
        q = row.get("text", "")
        evidence = retriever.retrieve(q, focus_terms=_tok(q)[:2], top_k=top_k, mode="hybrid")
        for instr in row.get("gold", []):
            if not isinstance(instr, dict):
                continue
            total += 1
            rank = None
            for i, ev in enumerate(evidence, start=1):
                if _evidence_match(instr, ev):
                    rank = i
                    break
            if rank is not None:
                hit += 1
                rr_sum += 1.0 / float(rank)
            if len(details) < 64:
                details.append(
                    {
                        "row_id": row.get("id"),
                        "op": instr.get("op"),
                        "rank": rank,
                        "hit": rank is not None,
                    }
                )
    recall = (hit / total) if total else 0.0
    mrr = (rr_sum / total) if total else 0.0
    return {
        "total_claims": total,
        "hits": hit,
        "recall_at_k": round(recall, 4),
        "mrr": round(mrr, 4),
        "samples": details,
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate evidence retrieval quality (recall@k + MRR).")
    ap.add_argument("--dataset", default="data/gold/gold_set_v1.jsonl")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--out", default="reports/evidence_recall_report.json")
    args = ap.parse_args()

    rows = _load_jsonl(args.dataset)
    graph = load_global_graph()
    vec = VectorStore(dim=int(os.getenv("COGNITIVE_EMBED_DIM", "128")))
    GraphEmbeddingPipeline(vec, dim=vec.dim).index_graph(graph, focus="", max_subgraphs=48)
    retriever = HybridRetriever(graph, vec, dim=vec.dim)

    metrics = evaluate(rows, retriever, top_k=max(1, args.top_k))
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "top_k": args.top_k,
        "metrics": metrics,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

