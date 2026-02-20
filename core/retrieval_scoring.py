from typing import List


def extract_edge_attrs(edge_data):
    if isinstance(edge_data, dict):
        if "relation" in edge_data or "label" in edge_data:
            return edge_data
        first = next(iter(edge_data.values()), {})
        if isinstance(first, dict):
            return first
    return {}


def score_graph_evidence(attrs: dict, depth: int = 1) -> float:
    try:
        conf = float((attrs or {}).get("confidence", 0.5))
    except Exception:
        conf = 0.5
    conf = max(0.0, min(1.0, conf))
    rel = str((attrs or {}).get("relation") or (attrs or {}).get("label") or "")
    has_prov = bool((attrs or {}).get("source") or (attrs or {}).get("inference_rule"))
    score = 0.45 + (0.42 * conf)
    if has_prov:
        score += 0.06
    if rel in {"CAUSE", "PREVENT", "OPPOSE", "GOAL", "IMPLY"}:
        score += 0.03
    score -= 0.07 * max(0, depth - 1)
    return round(max(0.0, min(1.0, score)), 3)


def build_graph_evidence(graph, terms: List[str], limit: int = 8) -> List[dict]:
    out = []
    seen = set()
    for t in terms:
        if not graph.has_node(t):
            continue

        # Depth-1 edges.
        for n in list(graph.successors(t))[: max(2, limit)]:
            edge_data = graph.get_edge_data(t, n) or {}
            attrs = extract_edge_attrs(edge_data)
            rel = str(attrs.get("relation") or attrs.get("label") or "")
            key = (t, rel, n, 1)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "kind": "graph",
                    "score": score_graph_evidence(attrs, depth=1),
                    "text": f"{t} -[{rel}]-> {n}",
                    "meta": {"u": t, "v": n, "relation": rel, "depth": 1},
                }
            )

        # Lightweight depth-2 context.
        for mid in list(graph.successors(t))[: max(2, limit // 2)]:
            for dst in list(graph.successors(mid))[:2]:
                edge_data = graph.get_edge_data(mid, dst) or {}
                attrs = extract_edge_attrs(edge_data)
                rel = str(attrs.get("relation") or attrs.get("label") or "")
                key = (t, mid, rel, dst, 2)
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    {
                        "kind": "graph",
                        "score": score_graph_evidence(attrs, depth=2),
                        "text": f"{t} -> {mid} -[{rel}]-> {dst}",
                        "meta": {"anchor": t, "u": mid, "v": dst, "relation": rel, "depth": 2},
                    }
                )
    return out[:limit]

