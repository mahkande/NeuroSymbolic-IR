from datetime import datetime

import networkx as nx


class TruthMaintenanceEngine:
    """
    Resolves conflicting facts with deterministic preferences.
    Current policies:
    - ATTR conflicts: keep strongest value for (subject, key), remove weaker ones.
    - OPPOSE vs ISA: if both exist for same pair, keep ISA and drop OPPOSE.
    """

    def __init__(self, graph):
        self.graph = graph

    @staticmethod
    def _to_float(value, default=0.5):
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _to_ts(value):
        if not value:
            return datetime.min
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except Exception:
            return datetime.min

    def _edge_records(self):
        if isinstance(self.graph, nx.MultiDiGraph):
            for u, v, key, attrs in self.graph.edges(data=True, keys=True):
                yield u, v, key, attrs or {}
        else:
            for u, v, attrs in self.graph.edges(data=True):
                yield u, v, None, attrs or {}

    @staticmethod
    def _score(attrs):
        confidence = TruthMaintenanceEngine._to_float(attrs.get("confidence"), default=0.5)
        inferred_penalty = -0.15 if bool(attrs.get("inferred")) else 0.0
        source_bonus = 0.1 if str(attrs.get("source", "")).lower() in {"user", "runtime"} else 0.0
        return confidence + inferred_penalty + source_bonus

    def resolve(self):
        removed = 0
        removed += self._resolve_attr_conflicts()
        removed += self._resolve_oppose_isa_conflicts()
        return removed

    def _resolve_attr_conflicts(self):
        by_subject_key = {}
        for u, v, key, attrs in self._edge_records():
            rel = attrs.get("relation") or attrs.get("label")
            if rel != "ATTR":
                continue
            akey = attrs.get("key", "ozellik")
            by_subject_key.setdefault((u, akey), []).append((u, v, key, attrs))

        removed = 0
        for (_, _), rows in by_subject_key.items():
            distinct_values = {v for _, v, _, _ in rows}
            if len(distinct_values) <= 1:
                continue

            rows_sorted = sorted(
                rows,
                key=lambda r: (self._score(r[3]), self._to_ts(r[3].get("created_at"))),
                reverse=True,
            )
            keep = rows_sorted[0]
            for u, v, key, _ in rows_sorted[1:]:
                if isinstance(self.graph, nx.MultiDiGraph):
                    if key is not None and self.graph.has_edge(u, v, key=key):
                        self.graph.remove_edge(u, v, key=key)
                        removed += 1
                else:
                    if self.graph.has_edge(u, v):
                        self.graph.remove_edge(u, v)
                        removed += 1

            # Ensure winner carries provenance.
            ku, kv, kk, ka = keep
            if isinstance(self.graph, nx.MultiDiGraph) and kk is not None and self.graph.has_edge(ku, kv, key=kk):
                edge_attrs = self.graph[ku][kv][kk]
                edge_attrs.setdefault("resolved_by", "TRUTH_MAINTENANCE")
            elif self.graph.has_edge(ku, kv):
                edge_attrs = self.graph[ku][kv]
                edge_attrs.setdefault("resolved_by", "TRUTH_MAINTENANCE")
        return removed

    def _resolve_oppose_isa_conflicts(self):
        isa_pairs = set()
        oppose_rows = []
        for u, v, key, attrs in self._edge_records():
            rel = attrs.get("relation") or attrs.get("label")
            if rel == "ISA":
                isa_pairs.add((u, v))
            elif rel == "OPPOSE":
                oppose_rows.append((u, v, key))

        removed = 0
        for u, v, key in oppose_rows:
            if (u, v) not in isa_pairs and (v, u) not in isa_pairs:
                continue
            if isinstance(self.graph, nx.MultiDiGraph):
                if key is not None and self.graph.has_edge(u, v, key=key):
                    self.graph.remove_edge(u, v, key=key)
                    removed += 1
            else:
                if self.graph.has_edge(u, v):
                    self.graph.remove_edge(u, v)
                    removed += 1
        return removed
