import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional


class VectorStore:
    """
    Vector backend selection:
    - qdrant (default): COGNITIVE_VECTOR_BACKEND=qdrant
    - local fallback: COGNITIVE_VECTOR_BACKEND=local
    """

    def __init__(self, dim: int = 128):
        self.dim = max(16, int(dim))
        self.backend = (os.getenv("COGNITIVE_VECTOR_BACKEND", "qdrant") or "qdrant").strip().lower()
        self.collection = os.getenv("COGNITIVE_VECTOR_COLLECTION", "cognitive_graph")
        self.local_path = Path(os.getenv("COGNITIVE_VECTOR_LOCAL_PATH", "memory/vector_index.jsonl"))
        self._driver = None
        self._local_index: Dict[str, dict] = {}
        self._init_backend()

    def _init_backend(self):
        if self.backend == "qdrant":
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.http.models import Distance, VectorParams

                url = os.getenv("QDRANT_URL", "http://localhost:6333")
                api_key = os.getenv("QDRANT_API_KEY")
                client = QdrantClient(url=url, api_key=api_key)
                client.get_collections()
                try:
                    client.get_collection(self.collection)
                except Exception:
                    client.create_collection(
                        collection_name=self.collection,
                        vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
                    )
                self._driver = client
                return
            except Exception:
                pass
        self.backend = "local"
        self._load_local()

    def backend_name(self) -> str:
        return self.backend

    def backend_status(self) -> dict:
        return {
            "configured_backend": (os.getenv("COGNITIVE_VECTOR_BACKEND", "qdrant") or "qdrant").strip().lower(),
            "active_backend": self.backend,
            "collection": self.collection,
            "qdrant_url": os.getenv("QDRANT_URL", "http://localhost:6333"),
            "local_path": str(self.local_path),
        }

    def _load_local(self):
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.local_path.exists():
            return
        for line in self.local_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                rid = str(row.get("id", "")).strip()
                if rid:
                    self._local_index[rid] = row
            except Exception:
                continue

    def _save_local(self):
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        with self.local_path.open("w", encoding="utf-8") as f:
            for row in self._local_index.values():
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def upsert(self, rows: List[dict]):
        clean = []
        for row in rows or []:
            rid = str(row.get("id", "")).strip()
            vec = row.get("vector", [])
            if not rid or not isinstance(vec, list) or len(vec) != self.dim:
                continue
            payload = {
                "id": rid,
                "vector": [float(v) for v in vec],
                "text": str(row.get("text", "")),
                "meta": row.get("meta", {}) or {},
            }
            clean.append(payload)
        if not clean:
            return 0

        if self.backend == "qdrant" and self._driver is not None:
            from qdrant_client.http.models import PointStruct

            points = [
                PointStruct(
                    id=idx + 1 + abs(hash(r["id"])) % 10_000_000,
                    vector=r["vector"],
                    payload={"rid": r["id"], "text": r["text"], "meta": r["meta"]},
                )
                for idx, r in enumerate(clean)
            ]
            self._driver.upsert(collection_name=self.collection, points=points)
            return len(clean)

        for r in clean:
            self._local_index[r["id"]] = r
        self._save_local()
        return len(clean)

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na <= 1e-12 or nb <= 1e-12:
            return 0.0
        return dot / (na * nb)

    def query(self, vector: List[float], top_k: int = 8) -> List[dict]:
        if not vector or len(vector) != self.dim:
            return []
        top_k = max(1, int(top_k))

        if self.backend == "qdrant" and self._driver is not None:
            hits = self._driver.search(collection_name=self.collection, query_vector=vector, limit=top_k)
            out = []
            for h in hits:
                p = h.payload or {}
                out.append(
                    {
                        "id": p.get("rid", ""),
                        "score": float(h.score),
                        "text": p.get("text", ""),
                        "meta": p.get("meta", {}) or {},
                    }
                )
            return out

        scored = []
        for row in self._local_index.values():
            score = self._cosine(vector, row.get("vector", []))
            scored.append(
                {
                    "id": row.get("id", ""),
                    "score": float(score),
                    "text": row.get("text", ""),
                    "meta": row.get("meta", {}) or {},
                }
            )
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
