import networkx as nx
from datetime import datetime, timezone


class DeterministicRuleEngine:
    """
    Deterministic forward-chaining rules over opcode graph.
    Adds inferred edges with provenance metadata.
    """

    def __init__(self, graph):
        self.graph = graph

    def _edge_iter(self):
        if isinstance(self.graph, nx.MultiDiGraph):
            for u, v, _, attrs in self.graph.edges(data=True, keys=True):
                yield u, v, attrs or {}
        else:
            for u, v, attrs in self.graph.edges(data=True):
                yield u, v, attrs or {}

    def _has_relation(self, u, v, relation):
        data = self.graph.get_edge_data(u, v)
        if not data:
            return False
        if isinstance(data, dict) and "relation" in data:
            return (data.get("relation") or data.get("label")) == relation
        if isinstance(data, dict):
            for _, attrs in data.items():
                if not isinstance(attrs, dict):
                    continue
                if (attrs.get("relation") or attrs.get("label")) == relation:
                    return True
        return False

    def _add_inferred(self, u, v, relation, rule, **attrs):
        if self._has_relation(u, v, relation):
            return False
        self.graph.add_node(u)
        self.graph.add_node(v)
        payload = {
            "relation": relation,
            "inferred": True,
            "inference_rule": rule,
            "source": "rule_engine",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "confidence": 0.7,
        }
        payload.update(attrs)
        self.graph.add_edge(u, v, **payload)
        return True

    def _collect(self):
        isa = set()
        before = set()
        oppose = set()
        attr = []
        want = set()
        goal = set()

        for u, v, attrs in self._edge_iter():
            rel = attrs.get("relation") or attrs.get("label")
            if rel == "ISA":
                isa.add((u, v))
            elif rel == "BEFORE":
                before.add((u, v))
            elif rel == "OPPOSE":
                oppose.add((u, v))
            elif rel == "ATTR":
                attr.append((u, attrs.get("key", "ozellik"), v))
            elif rel == "WANT":
                want.add((u, v))
            elif rel == "GOAL":
                goal.add((u, v))
        return isa, before, oppose, attr, want, goal

    def run(self, max_iterations=3, max_new=1000):
        total_new = 0

        for _ in range(max(1, int(max_iterations))):
            isa, before, oppose, attr, want, goal = self._collect()
            added_this_round = 0

            # Rule 1: ISA transitivity
            for a, b in list(isa):
                for x, c in list(isa):
                    if b != x or a == c:
                        continue
                    if self._add_inferred(a, c, "ISA", "ISA_TRANSITIVE"):
                        added_this_round += 1
                        total_new += 1
                        if total_new >= max_new:
                            return total_new

            # Rule 2: ATTR inheritance over ISA
            for child, parent in list(isa):
                for subj, key, value in list(attr):
                    if subj != parent:
                        continue
                    if self._add_inferred(child, value, "ATTR", "ISA_ATTR_INHERIT", key=key):
                        added_this_round += 1
                        total_new += 1
                        if total_new >= max_new:
                            return total_new

            # Rule 3: BEFORE transitivity
            for a, b in list(before):
                for x, c in list(before):
                    if b != x or a == c:
                        continue
                    if self._add_inferred(a, c, "BEFORE", "BEFORE_TRANSITIVE"):
                        added_this_round += 1
                        total_new += 1
                        if total_new >= max_new:
                            return total_new

            # Rule 4: OPPOSE symmetry
            for a, b in list(oppose):
                if self._add_inferred(b, a, "OPPOSE", "OPPOSE_SYMMETRIC"):
                    added_this_round += 1
                    total_new += 1
                    if total_new >= max_new:
                        return total_new

            # Rule 5: WANT -> GOAL bootstrap (only if missing)
            for agent, state in list(want):
                if (agent, state) in goal:
                    continue
                if self._add_inferred(agent, state, "GOAL", "WANT_TO_GOAL", priority="medium"):
                    added_this_round += 1
                    total_new += 1
                    if total_new >= max_new:
                        return total_new

            if added_this_round == 0:
                break

        return total_new
