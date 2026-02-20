import datetime
import re

import networkx as nx

from core.logic_engine import LogicEngine
from memory.conceptnet_service import ConceptNetService


class InferenceEngine:
    TRANSITIVE_REL_WHITELIST = {"DO", "CAUSE", "TRIGGER", "IMPLY", "BEFORE", "GOAL"}
    LOW_VALUE_TOKENS = {
        "",
        "ve",
        "veya",
        "ile",
        "de",
        "da",
        "bir",
        "bu",
        "su",
        "o",
        "the",
        "a",
        "an",
    }

    def __init__(self, memory_graph):
        self.graph = memory_graph
        self.z3 = LogicEngine()
        self.conceptnet = ConceptNetService(language="tr")
        self.log_lines = []
        self.min_inferred_confidence = 0.45
        self.max_transitive_branching = 24

    def log(self, msg, highlight=False):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if highlight:
            msg = f"\033[93m[INFERENCE]: {msg}\033[0m"
        line = f"[{ts}] {msg}"
        print(line)
        self.log_lines.append(line)

    @staticmethod
    def _prov(source: str, rule: str, confidence: float):
        return {
            "source": source,
            "inference_rule": rule,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "confidence": confidence,
        }

    def _edge_attr_variants(self, u, v):
        """Return edge attribute dicts for DiGraph and MultiDiGraph uniformly."""
        data = self.graph.get_edge_data(u, v)
        if not data:
            return []

        # DiGraph style: {"relation": "ISA", ...}
        if isinstance(data, dict) and "relation" in data:
            return [data]

        # MultiDiGraph style: {key: {"relation": "ISA", ...}, ...}
        out = []
        if isinstance(data, dict):
            for _, attrs in data.items():
                if isinstance(attrs, dict):
                    out.append(attrs)
        return out

    def _has_relation(self, u, v, relation):
        for attrs in self._edge_attr_variants(u, v):
            if (attrs.get("relation") or attrs.get("label")) == relation:
                return True
        return False

    def _relations_between(self, u, v):
        rels = set()
        for attrs in self._edge_attr_variants(u, v):
            rel = attrs.get("relation") or attrs.get("label")
            if rel:
                rels.add(str(rel))
        return rels

    def _relation_count_between(self, u, v, relation):
        rel_name = str(relation or "")
        count = 0
        for attrs in self._edge_attr_variants(u, v):
            rel = attrs.get("relation") or attrs.get("label")
            if str(rel or "") == rel_name:
                count += 1
        return count

    @staticmethod
    def _clean_token(val):
        return str(val or "").strip().lower()

    def _node_semantic_type(self, node):
        attrs = self.graph.nodes.get(node, {}) if self.graph.has_node(node) else {}
        semantic = attrs.get("semantic_type") or attrs.get("type") or attrs.get("node_type")
        token = self._clean_token(node)
        if semantic:
            return str(semantic).lower()
        if re.fullmatch(r"\d+", token):
            return "number"
        if re.fullmatch(r"[^\w]+", token):
            return "punct"
        return "entity"

    def _is_low_value_node(self, node):
        token = self._clean_token(node)
        if not token:
            return True
        if token in self.LOW_VALUE_TOKENS:
            return True
        if len(token) <= 1:
            return True
        if re.fullmatch(r"[^\w]+", token):
            return True
        return False

    def _should_prune_edge(self, u, v, relation):
        rel = str(relation or "").upper()
        if str(u) == str(v):
            return True
        if self._is_low_value_node(u) or self._is_low_value_node(v):
            return True
        return rel in {"INFERRED", "ABDUCED_CAUSE", "CN_ENRICH"} and self._relation_count_between(u, v, rel) > 1

    def prune_low_value_edges(self):
        if isinstance(self.graph, nx.MultiDiGraph):
            rows = list(self.graph.edges(data=True, keys=True))
            for u, v, k, attrs in rows:
                rel = (attrs or {}).get("relation") or (attrs or {}).get("label")
                if self._should_prune_edge(u, v, rel) and self.graph.has_edge(u, v, key=k):
                    self.graph.remove_edge(u, v, key=k)
            return

        rows = list(self.graph.edges(data=True))
        for u, v, attrs in rows:
            rel = (attrs or {}).get("relation") or (attrs or {}).get("label")
            if self._should_prune_edge(u, v, rel) and self.graph.has_edge(u, v):
                self.graph.remove_edge(u, v)

    def _dynamic_penalty(self, src, mid):
        src_branch = max(1, int(self.graph.out_degree(src)))
        mid_branch = max(1, int(self.graph.out_degree(mid)))
        branch_pressure = min(0.22, (max(0, src_branch - 4) * 0.01) + (max(0, mid_branch - 4) * 0.012))
        return round(branch_pressure, 3)

    def _confidence_after_penalty(self, base_conf, src, mid):
        conf = float(base_conf) - self._dynamic_penalty(src, mid)
        return round(max(0.0, min(1.0, conf)), 3)

    def _passes_transitive_guard(self, node, n1, n2):
        rel_1 = self._relations_between(node, n1)
        rel_2 = self._relations_between(n1, n2)
        if not rel_1 or not rel_2:
            return False
        if not (rel_1 & self.TRANSITIVE_REL_WHITELIST) or not (rel_2 & self.TRANSITIVE_REL_WHITELIST):
            return False

        t0 = self._node_semantic_type(node)
        t1 = self._node_semantic_type(n1)
        t2 = self._node_semantic_type(n2)
        if "punct" in {t0, t1, t2}:
            return False
        if t0 != t2 and "entity" not in {t0, t2}:
            return False

        if int(self.graph.out_degree(node)) > self.max_transitive_branching:
            return False
        if int(self.graph.out_degree(n1)) > self.max_transitive_branching:
            return False
        return True

    def _incoming_with_rel(self, node, rel_set):
        out = []
        if isinstance(self.graph, nx.MultiDiGraph):
            for u, _, _, attrs in self.graph.in_edges(node, data=True, keys=True):
                rel = attrs.get("relation") or attrs.get("label")
                if rel in rel_set:
                    out.append((u, rel))
        else:
            for u, _, attrs in self.graph.in_edges(node, data=True):
                rel = attrs.get("relation") or attrs.get("label")
                if rel in rel_set:
                    out.append((u, rel))
        return out

    def _graph_to_ir(self):
        ir = []
        if isinstance(self.graph, nx.MultiDiGraph):
            for u, v, _, attrs in self.graph.edges(data=True, keys=True):
                op = attrs.get("relation") or attrs.get("label")
                if op:
                    ir.append({"op": op, "args": [u, v]})
        else:
            for u, v, attrs in self.graph.edges(data=True):
                op = attrs.get("relation") or attrs.get("label")
                if op:
                    ir.append({"op": op, "args": [u, v]})
        return ir

    def infer_isa_attr_inheritance(self, max_new=500):
        """
        Deduction rule:
        ISA(A, B) and ATTR(B, key, C) => ATTR(A, key, C)

        Returns number of newly added inferred ATTR edges.
        """
        new_count = 0

        isa_pairs = set()
        attr_triples = []  # (subject, key, value)

        if isinstance(self.graph, nx.MultiDiGraph):
            edge_iter = self.graph.edges(data=True, keys=True)
            for u, v, _, attrs in edge_iter:
                rel = attrs.get("relation") or attrs.get("label")
                if rel == "ISA":
                    isa_pairs.add((u, v))
                elif rel == "ATTR":
                    attr_triples.append((u, attrs.get("key", "ozellik"), v))
        else:
            for u, v, attrs in self.graph.edges(data=True):
                rel = attrs.get("relation") or attrs.get("label")
                if rel == "ISA":
                    isa_pairs.add((u, v))
                elif rel == "ATTR":
                    attr_triples.append((u, attrs.get("key", "ozellik"), v))

        if not isa_pairs or not attr_triples:
            return 0

        history_ir = self._graph_to_ir()

        for child, parent in isa_pairs:
            for subj, key, value in attr_triples:
                if subj != parent:
                    continue

                # Skip if already present.
                if self._has_relation(child, value, "ATTR"):
                    continue

                candidate_ir = {"op": "ATTR", "args": [child, key or "ozellik", value]}
                is_ok, _ = self.z3.verify_consistency(history_ir + [candidate_ir])
                if not is_ok:
                    continue

                self.graph.add_node(child)
                self.graph.add_node(value)
                conf = self._confidence_after_penalty(0.72, child, parent)
                if conf < self.min_inferred_confidence:
                    continue
                self.graph.add_edge(
                    child,
                    value,
                    relation="ATTR",
                    key=key or "ozellik",
                    inferred=True,
                    **self._prov("inference_engine", "ISA_ATTR_INHERITANCE", conf),
                )
                new_count += 1
                history_ir.append(candidate_ir)
                if new_count >= max_new:
                    self.log(f"Inheritance inference limit reached ({max_new}).")
                    self.prune_low_value_edges()
                    return new_count

        if new_count > 0:
            self.log(f"ISA+ATTR inheritance added {new_count} new ATTR edges.", highlight=True)
        self.prune_low_value_edges()
        return new_count

    def abductive_reasoning(self, target_event, max_results=10, persist=False):
        """
        Inverse reasoning (abduction):
        Given a target event/state, return plausible causes from graph evidence.

        Rules:
        1) Direct: CAUSE(X, target) / TRIGGER(X, target) / IMPLY(X, target)
        2) Two-hop: CAUSE(X, Y) and CAUSE(Y, target)  => X (weaker hypothesis)
        """
        target = str(target_event).strip()
        if not target or not self.graph.has_node(target):
            return []

        rel_causal = {"CAUSE", "TRIGGER", "IMPLY"}
        hypotheses = []
        seen = set()

        # Direct hypotheses have higher score.
        for cause, rel in self._incoming_with_rel(target, rel_causal):
            key = (cause, rel, "direct")
            if key in seen:
                continue
            seen.add(key)
            hypotheses.append(
                {
                    "cause": cause,
                    "target": target,
                    "support": rel,
                    "path": [cause, target],
                    "score": 1.0,
                    "mode": "direct",
                }
            )

        # Two-hop hypotheses (weaker).
        mids = [u for u, _ in self._incoming_with_rel(target, rel_causal)]
        for mid in mids:
            for cause, rel in self._incoming_with_rel(mid, rel_causal):
                key = (cause, rel, "two_hop", mid)
                if key in seen:
                    continue
                seen.add(key)
                hypotheses.append(
                    {
                        "cause": cause,
                        "target": target,
                        "support": rel,
                        "path": [cause, mid, target],
                        "score": 0.65,
                        "mode": "two_hop",
                    }
                )

        hypotheses.sort(key=lambda x: (x["score"], x["cause"]), reverse=True)
        hypotheses = hypotheses[:max_results]

        if persist:
            for h in hypotheses:
                c = h["cause"]
                if not self._has_relation(c, target, "ABDUCED_CAUSE"):
                    conf = self._confidence_after_penalty(h["score"], c, target)
                    if conf < self.min_inferred_confidence:
                        continue
                    self.graph.add_edge(
                        c,
                        target,
                        relation="ABDUCED_CAUSE",
                        inferred=True,
                        score=h["score"],
                        **self._prov("inference_engine", "ABDUCTION", conf),
                    )

        if hypotheses:
            self.log(
                f"Abductive reasoning produced {len(hypotheses)} hypotheses for target '{target}'.",
                highlight=True,
            )
        self.prune_low_value_edges()
        return hypotheses

    def transitive_discovery(self, focus_nodes=None, cooldown=10):
        # Lightweight transitive closure over graph neighborhood.
        import time

        if not hasattr(self, "_last_inference"):
            self._last_inference = 0
        now = time.time()
        if now - self._last_inference < cooldown:
            return
        self._last_inference = now

        nodes = set(focus_nodes) if focus_nodes else set(self.graph.nodes)
        for node in nodes:
            neighbors1 = set(self.graph.successors(node))
            for n1 in neighbors1:
                neighbors2 = set(self.graph.successors(n1))
                for n2 in neighbors2:
                    if self.graph.has_edge(node, n2):
                        continue
                    if self._should_prune_edge(node, n2, "INFERRED"):
                        continue
                    if not self._passes_transitive_guard(node, n1, n2):
                        continue

                    # Infer a weak link only if local chain is consistent.
                    chain = [
                        {"op": "DO", "args": [node, n1]},
                        {"op": "DO", "args": [n1, n2]},
                    ]
                    is_consistent, _ = self.z3.verify_consistency(chain)
                    if is_consistent:
                        conf = self._confidence_after_penalty(0.6, node, n1)
                        if conf < self.min_inferred_confidence:
                            continue
                        self.graph.add_edge(
                            node,
                            n2,
                            relation="INFERRED",
                            **self._prov("inference_engine", "TRANSITIVE_DISCOVERY", conf),
                        )
                        self.log(f"Yeni bir baglanti kesfedildi! {node} -> {n2}", highlight=True)
        self.prune_low_value_edges()

    def conceptnet_enrichment(self, concept, topn=3, cooldown=5):
        # Concept enrichment with throttle.
        import time

        if not hasattr(self, "_last_enrich"):
            self._last_enrich = 0
        now = time.time()
        if now - self._last_enrich < cooldown:
            return
        self._last_enrich = now

        data = self.conceptnet.query(concept)
        irs = self.conceptnet.extract_facts(data)
        count = 0
        for ir in irs:
            if count >= topn:
                break
            if ir["op"] == "DEF_ENTITY":
                self.graph.add_edge(
                    ir["args"][0],
                    ir["args"][1],
                    relation="CN_ENRICH",
                    **self._prov("conceptnet", "CONCEPTNET_ENRICH", 0.62),
                )
                self.log(f"ConceptNet enrichment: {ir['args'][0]} -> {ir['args'][1]}")
                count += 1
            elif ir["op"] == "ATTR":
                self.graph.add_node(ir["args"][0])
                self.graph.add_node(ir["args"][2])
                self.graph.add_edge(
                    ir["args"][0],
                    ir["args"][2],
                    relation="CN_ENRICH",
                    key=ir["args"][1],
                    **self._prov("conceptnet", "CONCEPTNET_ENRICH", 0.62),
                )
                self.log(f"ConceptNet enrichment: {ir['args'][0]} -> {ir['args'][2]}")
                count += 1
            elif ir["op"] == "PART_OF":
                self.graph.add_edge(
                    ir["args"][0],
                    ir["args"][1],
                    relation="CN_ENRICH",
                    **self._prov("conceptnet", "CONCEPTNET_ENRICH", 0.62),
                )
                self.log(f"ConceptNet enrichment: {ir['args'][0]} -> {ir['args'][1]}")
                count += 1
