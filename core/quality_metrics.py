import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_QUALITY_LOG = Path("memory/opcode_quality.jsonl")


def edge_type_counts(graph) -> Dict[str, int]:
    counts = Counter()
    if graph.__class__.__name__ == "MultiDiGraph":
        for _, _, _, attrs in graph.edges(data=True, keys=True):
            rel = (attrs or {}).get("relation") or (attrs or {}).get("label")
            if rel:
                counts[str(rel)] += 1
    else:
        for _, _, attrs in graph.edges(data=True):
            rel = (attrs or {}).get("relation") or (attrs or {}).get("label")
            if rel:
                counts[str(rel)] += 1
    return dict(counts)


def edge_diversity_metrics(graph, known_opcodes: Iterable[str] = ()):
    counts = edge_type_counts(graph)
    total = sum(counts.values())
    unique = len(counts)

    entropy = 0.0
    if total > 0:
        for c in counts.values():
            p = c / total
            entropy -= p * math.log2(p)

    max_entropy = math.log2(unique) if unique > 1 else 1.0
    norm_entropy = round(entropy / max_entropy, 4) if total > 0 else 0.0

    known = set(str(o) for o in known_opcodes if o)
    coverage = round((unique / len(known)), 4) if known else 0.0
    dominance_ratio = round((max(counts.values()) / total), 4) if total > 0 else 0.0
    cause_ratio = round((counts.get("CAUSE", 0) / total), 4) if total > 0 else 0.0

    return {
        "total_edges": total,
        "unique_edge_types": unique,
        "entropy": round(entropy, 4),
        "norm_entropy": norm_entropy,
        "coverage": coverage,
        "dominance_ratio": dominance_ratio,
        "cause_ratio": cause_ratio,
        "counts": counts,
    }


def provenance_coverage(graph):
    required = ("source", "created_at", "confidence", "inference_rule")
    total = 0
    ok = 0

    try:
        rows = list(graph.edges(data=True, keys=True))
        unpack = lambda r: r[3]
    except TypeError:
        rows = list(graph.edges(data=True))
        unpack = lambda r: r[2]

    for row in rows:
        attrs = unpack(row) or {}
        total += 1
        if all(k in attrs and attrs.get(k) not in (None, "") for k in required):
            ok += 1

    ratio = round((ok / total), 4) if total else 1.0
    return {"total_edges": total, "complete_edges": ok, "coverage": ratio}


def drift_alerts(graph, known_opcodes: Iterable[str] = (), cause_threshold: float = 0.45, min_edges: int = 40):
    metrics = edge_diversity_metrics(graph, known_opcodes=known_opcodes)
    alerts = []
    if metrics["total_edges"] >= min_edges and metrics["cause_ratio"] >= cause_threshold:
        alerts.append(
            f"CAUSE dominance drift: ratio={metrics['cause_ratio']:.2f} "
            f"(threshold={cause_threshold:.2f}, edges={metrics['total_edges']})"
        )
    if metrics["total_edges"] >= min_edges and metrics["dominance_ratio"] >= 0.65:
        alerts.append(
            f"Single-edge-type dominance high: ratio={metrics['dominance_ratio']:.2f} "
            f"(edges={metrics['total_edges']})"
        )
    return alerts, metrics


def record_opcode_quality(predicted_ops: List[str], final_ops: List[str], path: Path = DEFAULT_QUALITY_LOG):
    path.parent.mkdir(parents=True, exist_ok=True)
    predicted = Counter(str(op) for op in (predicted_ops or []))
    final = Counter(str(op) for op in (final_ops or []))
    payload = {"predicted": dict(predicted), "final": dict(final)}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def aggregate_opcode_quality(path: Path = DEFAULT_QUALITY_LOG, max_rows: int = 2000):
    if not path.exists():
        return {}

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    rows = path.read_text(encoding="utf-8").splitlines()[-max_rows:]
    for line in rows:
        try:
            row = json.loads(line)
        except Exception:
            continue
        pred = Counter((row or {}).get("predicted", {}))
        fin = Counter((row or {}).get("final", {}))
        ops = set(pred.keys()) | set(fin.keys())
        for op in ops:
            p = int(pred.get(op, 0))
            f = int(fin.get(op, 0))
            tp[op] += min(p, f)
            fp[op] += max(0, p - f)
            fn[op] += max(0, f - p)

    out = {}
    for op in sorted(set(tp.keys()) | set(fp.keys()) | set(fn.keys())):
        tpi = tp[op]
        fpi = fp[op]
        fni = fn[op]
        precision = (tpi / (tpi + fpi)) if (tpi + fpi) > 0 else 0.0
        recall = (tpi / (tpi + fni)) if (tpi + fni) > 0 else 0.0
        out[op] = {
            "precision_proxy": round(precision, 4),
            "recall_proxy": round(recall, 4),
            "tp": tpi,
            "fp": fpi,
            "fn": fni,
        }
    return out
