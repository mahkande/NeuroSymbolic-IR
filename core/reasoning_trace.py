import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx


class ReasoningTraceStore:
    """
    Persists bidirectional reasoning traces:
    - forward: root -> ... via typed graph edges
    - inverse: target <- hypothesized causes (abduction)
    """

    def __init__(self, path: str = "memory/reasoning_traces.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    @staticmethod
    def _edge_iter(graph, u, v):
        data = graph.get_edge_data(u, v) or {}
        if isinstance(data, dict) and "relation" in data:
            yield data
            return
        if isinstance(data, dict):
            for _, attrs in data.items():
                if isinstance(attrs, dict):
                    yield attrs

    @staticmethod
    def _normalize_record(record):
        return json.dumps(record, ensure_ascii=False, sort_keys=True)

    def _append_unique(self, record):
        payload = self._normalize_record(record)
        digest = hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()
        line = {
            "id": digest,
            "ts": datetime.now(timezone.utc).isoformat(),
            **record,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        return digest

    def record_forward(self, graph, root, max_depth: int = 2, max_records: int = 20):
        root = str(root).strip()
        if not root or not graph.has_node(root):
            return 0

        allow_rel = {
            "ISA",
            "ATTR",
            "BEFORE",
            "CAUSE",
            "TRIGGER",
            "IMPLY",
            "GOAL",
            "WANT",
            "DO",
            "INFERRED",
            "ABDUCED_CAUSE",
        }
        queue = [(root, [root], [], 0)]
        seen = set()
        written = 0

        while queue and written < max_records:
            node, path_nodes, rel_chain, depth = queue.pop(0)
            if depth >= max_depth:
                continue

            if isinstance(graph, nx.MultiDiGraph):
                edge_iter = graph.out_edges(node, data=True, keys=True)
                for _, nxt, _, attrs in edge_iter:
                    rel = (attrs or {}).get("relation") or (attrs or {}).get("label")
                    if rel not in allow_rel:
                        continue
                    new_nodes = path_nodes + [nxt]
                    new_rel_chain = rel_chain + [rel]
                    rec = {
                        "direction": "forward",
                        "root": root,
                        "path_nodes": new_nodes,
                        "path_relations": new_rel_chain,
                    }
                    key = self._normalize_record(rec)
                    if key not in seen:
                        seen.add(key)
                        self._append_unique(rec)
                        written += 1
                        if written >= max_records:
                            break
                    queue.append((nxt, new_nodes, new_rel_chain, depth + 1))
            else:
                edge_iter = graph.out_edges(node, data=True)
                for _, nxt, attrs in edge_iter:
                    rel = (attrs or {}).get("relation") or (attrs or {}).get("label")
                    if rel not in allow_rel:
                        continue
                    new_nodes = path_nodes + [nxt]
                    new_rel_chain = rel_chain + [rel]
                    rec = {
                        "direction": "forward",
                        "root": root,
                        "path_nodes": new_nodes,
                        "path_relations": new_rel_chain,
                    }
                    key = self._normalize_record(rec)
                    if key not in seen:
                        seen.add(key)
                        self._append_unique(rec)
                        written += 1
                        if written >= max_records:
                            break
                    queue.append((nxt, new_nodes, new_rel_chain, depth + 1))
        return written

    def record_inverse(self, target, hypotheses, max_records: int = 20):
        target = str(target).strip()
        if not target or not hypotheses:
            return 0

        written = 0
        for h in hypotheses[:max_records]:
            cause = str(h.get("cause", "")).strip()
            path = h.get("path") or []
            if not cause or not path:
                continue
            rec = {
                "direction": "inverse",
                "target": target,
                "cause": cause,
                "support": h.get("support"),
                "mode": h.get("mode"),
                "score": h.get("score"),
                "path_nodes": path,
            }
            self._append_unique(rec)
            written += 1
        return written

    def record_claim_proofs(self, root: str, proofs, max_records: int = 128):
        root = str(root or "").strip()
        if not root or not proofs:
            return 0
        written = 0
        for p in (proofs or [])[:max_records]:
            if not isinstance(p, dict):
                continue
            claim = p.get("claim", {}) or {}
            rec = {
                "direction": "claim_proof",
                "root": root,
                "claim_id": p.get("claim_id"),
                "claim_op": claim.get("op"),
                "claim_args": claim.get("args", []),
                "verdict": p.get("verdict"),
                "supports": p.get("supports", []),
            }
            self._append_unique(rec)
            written += 1
        return written
