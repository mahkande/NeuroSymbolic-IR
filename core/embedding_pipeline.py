import hashlib
import math
import re
from typing import List, Tuple


def hash_embed(text: str, dim: int = 128) -> List[float]:
    dim = max(16, int(dim))
    vec = [0.0] * dim
    toks = re.findall(r"[\wçğıöşü]+", (text or "").lower())
    if not toks:
        return vec
    for tok in toks:
        h = hashlib.sha1(tok.encode("utf-8")).hexdigest()
        idx = int(h[:8], 16) % dim
        sign = 1.0 if (int(h[8:10], 16) % 2 == 0) else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 1e-12:
        return vec
    return [v / norm for v in vec]


class GraphEmbeddingPipeline:
    def __init__(self, vector_store, dim: int = 128):
        self.store = vector_store
        self.dim = max(16, int(dim))

    @staticmethod
    def _edge_iter(graph):
        try:
            return graph.edges(data=True, keys=True)
        except TypeError:
            return graph.edges(data=True)

    def _node_rows(self, graph):
        rows = []
        for n, attrs in graph.nodes(data=True):
            txt = f"node {n} type {attrs.get('type', '')} label {attrs.get('label', '')}".strip()
            rows.append({"id": f"node::{n}", "text": txt, "vector": hash_embed(txt, self.dim), "meta": {"kind": "node", "node": n}})
        return rows

    def _edge_rows(self, graph):
        rows = []
        try:
            for u, v, k, attrs in self._edge_iter(graph):
                rel = (attrs or {}).get("relation") or (attrs or {}).get("label") or "RELATED"
                txt = f"edge {u} {rel} {v}"
                rows.append(
                    {
                        "id": f"edge::{u}::{rel}::{v}::{k}",
                        "text": txt,
                        "vector": hash_embed(txt, self.dim),
                        "meta": {"kind": "edge", "u": u, "v": v, "relation": rel},
                    }
                )
        except ValueError:
            for u, v, attrs in self._edge_iter(graph):
                rel = (attrs or {}).get("relation") or (attrs or {}).get("label") or "RELATED"
                txt = f"edge {u} {rel} {v}"
                rows.append(
                    {
                        "id": f"edge::{u}::{rel}::{v}",
                        "text": txt,
                        "vector": hash_embed(txt, self.dim),
                        "meta": {"kind": "edge", "u": u, "v": v, "relation": rel},
                    }
                )
        return rows

    def _subgraph_rows(self, graph, focus: str = "", max_rows: int = 64):
        rows = []
        if not focus or not graph.has_node(focus):
            return rows
        neighbors = list(graph.successors(focus))[:max_rows]
        for n in neighbors:
            txt = f"subgraph {focus} neighbor {n}"
            rows.append(
                {
                    "id": f"subgraph::{focus}::{n}",
                    "text": txt,
                    "vector": hash_embed(txt, self.dim),
                    "meta": {"kind": "subgraph", "focus": focus, "neighbor": n},
                }
            )
        return rows

    def index_graph(self, graph, focus: str = "", max_subgraphs: int = 64) -> Tuple[int, int, int]:
        node_rows = self._node_rows(graph)
        edge_rows = self._edge_rows(graph)
        sub_rows = self._subgraph_rows(graph, focus=focus, max_rows=max_subgraphs)
        total = self.store.upsert(node_rows + edge_rows + sub_rows)
        return len(node_rows), len(edge_rows), total
