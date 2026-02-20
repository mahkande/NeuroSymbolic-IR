import re
from typing import List

from core.embedding_pipeline import hash_embed


class HybridRetriever:
    def __init__(self, graph, vector_store, dim: int = 128):
        self.graph = graph
        self.store = vector_store
        self.dim = max(16, int(dim))

    @staticmethod
    def _tokens(text: str) -> List[str]:
        return re.findall(r"[\wçğıöşü]+", (text or "").lower())

    def _graph_evidence(self, terms: List[str], limit: int = 8):
        out = []
        for t in terms:
            if not self.graph.has_node(t):
                continue
            for n in list(self.graph.successors(t))[:limit]:
                edge_data = self.graph.get_edge_data(t, n) or {}
                rel = ""
                if isinstance(edge_data, dict):
                    if "relation" in edge_data:
                        rel = edge_data.get("relation", "")
                    else:
                        first = next(iter(edge_data.values()), {})
                        if isinstance(first, dict):
                            rel = first.get("relation", "")
                out.append(
                    {
                        "kind": "graph",
                        "score": 0.72,
                        "text": f"{t} -[{rel}]-> {n}",
                        "meta": {"u": t, "v": n, "relation": rel},
                    }
                )
        return out[:limit]

    def retrieve(self, query_text: str, focus_terms: List[str] = None, top_k: int = 8, mode: str = "hybrid"):
        focus_terms = focus_terms or []
        mode = (mode or "hybrid").strip().lower()
        if mode not in {"vector-only", "graph-only", "hybrid"}:
            mode = "hybrid"

        qvec = hash_embed(query_text, self.dim)
        vec_hits = self.store.query(qvec, top_k=top_k) if mode in {"vector-only", "hybrid"} else []
        vec_out = [
            {
                "kind": "vector",
                "score": float(h.get("score", 0.0)),
                "text": h.get("text", ""),
                "meta": h.get("meta", {}) or {},
            }
            for h in vec_hits
        ]

        terms = list(dict.fromkeys((focus_terms or []) + self._tokens(query_text)[:4]))
        graph_out = self._graph_evidence(terms, limit=max(3, top_k // 2)) if mode in {"graph-only", "hybrid"} else []
        if mode == "vector-only":
            merged = vec_out
        elif mode == "graph-only":
            merged = graph_out
        else:
            merged = vec_out + graph_out
        merged.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return merged[:top_k]
