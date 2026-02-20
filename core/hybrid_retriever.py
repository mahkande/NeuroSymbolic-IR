import os
import re
from typing import List

from core.embedding_pipeline import embed_text
from core.retrieval_scoring import build_graph_evidence


class HybridRetriever:
    def __init__(self, graph, vector_store, dim: int = 128):
        self.graph = graph
        self.store = vector_store
        self.dim = max(16, int(dim))
        self._reranker = None
        self._reranker_error = ""

    @staticmethod
    def _tokens(text: str) -> List[str]:
        return re.findall(r"[\wÃ§ÄŸÄ±Ã¶ÅŸÃ¼]+", (text or "").lower())

    def _graph_evidence(self, terms: List[str], limit: int = 8) -> List[dict]:
        return build_graph_evidence(self.graph, terms, limit=limit)

    def _load_jina_reranker(self):
        if self._reranker is not None:
            return self._reranker
        if self._reranker_error:
            return None
        model_name = os.getenv("COGNITIVE_JINA_RERANK_MODEL", "jinaai/jina-reranker-v2-base-multilingual").strip()
        try:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(model_name)
        except Exception as exc:
            self._reranker_error = str(exc)
        return self._reranker

    def _rerank(self, query_text: str, hits: List[dict]) -> List[dict]:
        use_reranker = os.getenv("COGNITIVE_USE_RERANKER", "1").strip().lower() in {"1", "true", "yes", "on"}
        provider = os.getenv("COGNITIVE_RERANK_PROVIDER", "jina").strip().lower()
        if not use_reranker or provider != "jina" or not hits:
            return hits

        reranker = self._load_jina_reranker()
        if reranker is None:
            return hits

        try:
            pairs = [(query_text or "", str(h.get("text", ""))) for h in hits]
            scores = reranker.predict(pairs)
            for hit, score in zip(hits, list(scores)):
                hit["rerank_score"] = float(score)
            hits.sort(key=lambda x: float(x.get("rerank_score", x.get("score", 0.0))), reverse=True)
            return hits
        except Exception:
            return hits

    def retrieve(self, query_text: str, focus_terms: List[str] = None, top_k: int = 8, mode: str = "hybrid"):
        focus_terms = focus_terms or []
        mode = (mode or "hybrid").strip().lower()
        if mode not in {"vector-only", "graph-only", "hybrid"}:
            mode = "hybrid"

        qvec = embed_text(query_text, self.dim)
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

        terms = list(dict.fromkeys((focus_terms or []) + self._tokens(query_text)[:6]))
        graph_out = self._graph_evidence(terms, limit=max(4, top_k)) if mode in {"graph-only", "hybrid"} else []
        if mode == "vector-only":
            merged = vec_out
        elif mode == "graph-only":
            merged = graph_out
        else:
            merged = vec_out + graph_out
        merged.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        merged = self._rerank(query_text, merged)
        return merged[:top_k]

