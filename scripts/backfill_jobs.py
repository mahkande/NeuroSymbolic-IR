#!/usr/bin/env python
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.embedding_pipeline import GraphEmbeddingPipeline
from core.inference_engine import InferenceEngine
from core.rule_engine import DeterministicRuleEngine
from core.truth_maintenance import TruthMaintenanceEngine
from main import load_global_graph, save_global_graph
from memory.vector_store import VectorStore


def chunks(seq, size):
    size = max(1, int(size))
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def run_backfill(batch_size: int = 25, max_batches: int = 100, sleep_ms: int = 50):
    graph = load_global_graph()
    rule = DeterministicRuleEngine(graph)
    inf = InferenceEngine(graph)
    tm = TruthMaintenanceEngine(graph)
    vec = VectorStore(dim=int(os.getenv("COGNITIVE_EMBED_DIM", "128")))
    pipe = GraphEmbeddingPipeline(vec, dim=vec.dim)

    nodes = list(graph.nodes)
    batches = list(chunks(nodes, batch_size))[: max(1, max_batches)]
    stats = {
        "batches": len(batches),
        "initial_nodes": graph.number_of_nodes(),
        "initial_edges": graph.number_of_edges(),
        "rule_added": 0,
        "attr_inherit_added": 0,
        "abduced_added": 0,
        "removed_conflicts": 0,
        "vector_upserts": 0,
    }

    for idx, batch in enumerate(batches, start=1):
        stats["rule_added"] += rule.run(max_iterations=2, max_new=400)
        stats["attr_inherit_added"] += inf.infer_isa_attr_inheritance(max_new=300)
        before = graph.number_of_edges()
        for n in batch:
            inf.abductive_reasoning(str(n), max_results=3, persist=True)
        after = graph.number_of_edges()
        stats["abduced_added"] += max(0, after - before)
        inf.transitive_discovery(focus_nodes=batch, cooldown=0)
        stats["removed_conflicts"] += tm.resolve()
        _, _, upserts = pipe.index_graph(graph, focus=str(batch[0]) if batch else "", max_subgraphs=24)
        stats["vector_upserts"] += upserts
        save_global_graph(graph, merge_existing=False)
        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)
        print(f"[backfill] batch {idx}/{len(batches)} done")

    stats["final_nodes"] = graph.number_of_nodes()
    stats["final_edges"] = graph.number_of_edges()
    stats["edge_growth"] = stats["final_edges"] - stats["initial_edges"]
    stats["batch_size"] = batch_size
    stats["max_batches"] = max_batches
    stats["throughput_edges_per_batch"] = round(stats["edge_growth"] / max(1, len(batches)), 2)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Run background inference backfill jobs in batches.")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--sleep-ms", type=int, default=50)
    parser.add_argument("--report", default="memory/backfill_report.json")
    args = parser.parse_args()

    t0 = time.time()
    stats = run_backfill(batch_size=args.batch_size, max_batches=args.max_batches, sleep_ms=args.sleep_ms)
    stats["duration_sec"] = round(time.time() - t0, 2)

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Backfill report written: {out}")


if __name__ == "__main__":
    main()
