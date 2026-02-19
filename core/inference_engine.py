import networkx as nx
from core.logic_engine import LogicEngine
from memory.conceptnet_service import ConceptNetService
import datetime

class InferenceEngine:
    def __init__(self, memory_graph):
        self.graph = memory_graph
        self.z3 = LogicEngine()
        self.conceptnet = ConceptNetService(language="tr")
        self.log_lines = []

    def log(self, msg, highlight=False):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if highlight:
            msg = f"\033[93m[INFERENCE]: {msg}\033[0m"  # Sarı renk
        line = f"[{ts}] {msg}"
        print(line)
        self.log_lines.append(line)

    def transitive_discovery(self, focus_nodes=None, cooldown=10):
        # Sadece yeni eklenen düğüm ve 2 derinlikli komşuları üzerinde çalış
        import time
        if not hasattr(self, '_last_inference'):
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
                    if not self.graph.has_edge(node, n2):
                        ir_chain = [
                            {"op": self.graph[node][n1].get("label", "REL"), "args": [node, n1]},
                            {"op": self.graph[n1][n2].get("label", "REL"), "args": [n1, n2]}
                        ]
                        is_consistent, msg = self.z3.verify_consistency(ir_chain)
                        if is_consistent:
                            self.graph.add_edge(node, n2, label="INFERRED")
                            self.log(f"Yeni bir bağlantı keşfedildi! {node} -> {n2}", highlight=True)

    def conceptnet_enrichment(self, concept, topn=3, cooldown=5):
        # Sadece izole düğümler için ve cooldown ile
        import time
        if not hasattr(self, '_last_enrich'):
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
                self.graph.add_edge(ir["args"][0], ir["args"][1], label="CN_ENRICH")
                self.log(f"ConceptNet enrichment: {ir['args'][0]} -> {ir['args'][1]}")
                count += 1
            elif ir["op"] == "ATTR":
                self.graph.add_node(ir["args"][0])
                self.graph.add_node(ir["args"][2])
                self.graph.add_edge(ir["args"][0], ir["args"][2], label="CN_ENRICH")
                self.log(f"ConceptNet enrichment: {ir['args'][0]} -> {ir['args'][2]}")
                count += 1
            elif ir["op"] == "PART_OF":
                self.graph.add_edge(ir["args"][0], ir["args"][1], label="CN_ENRICH")
                self.log(f"ConceptNet enrichment: {ir['args'][0]} -> {ir['args'][1]}")
                count += 1
