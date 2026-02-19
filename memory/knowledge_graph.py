import matplotlib.pyplot as plt
import networkx as nx


class CognitiveMemory:
    CONTEXT_NODE = "__context__"

    def __init__(self):
        # Directed graph for relation storage.
        self.graph = nx.MultiDiGraph()

    def _ensure_node(self, node_id, **attrs):
        if node_id is None:
            return
        if not self.graph.has_node(node_id):
            self.graph.add_node(node_id, **attrs)

    def add_ir_chain(self, ir_chain):
        """Write a validated IR chain into graph memory."""
        for instr in ir_chain:
            op = instr.get("op")
            args = instr.get("args", [])
            if not isinstance(args, list):
                continue

            if op in ["DEF_ENTITY", "DEF_CONCEPT"] and len(args) >= 2:
                node_id = args[0]
                node_type = args[1]
                self.graph.add_node(node_id, type=node_type, label=node_id)
                continue

            if not args:
                continue

            # Unary opcodes (MUST, MAY, FORBID, START, END, ASSUME, ...)
            if len(args) == 1:
                u = self.CONTEXT_NODE
                v = args[0]
                self._ensure_node(u, type="SYSTEM", label="context")
                self._ensure_node(v)
                self.graph.add_edge(u, v, relation=op)
                continue

            # Opcode-specific edge targets for semantic correctness.
            if op == "ATTR" and len(args) >= 3:
                u, v = args[0], args[2]
                edge_attrs = {"relation": op, "key": args[1]}
            elif op == "BELIEVE" and len(args) >= 3:
                u, v = args[0], args[1]
                edge_attrs = {"relation": op, "confidence": args[2]}
            elif op == "GOAL" and len(args) >= 3:
                u, v = args[0], args[1]
                edge_attrs = {"relation": op, "priority": args[2]}
            else:
                u, v = args[0], args[1]
                edge_attrs = {"relation": op}
                if len(args) > 2:
                    edge_attrs["extra_args"] = args[2:]

            self._ensure_node(u)
            self._ensure_node(v)
            self.graph.add_edge(u, v, **edge_attrs)

    def find_conflicts_with_history(self, new_ir):
        """
        Check whether incoming IR conflicts with prior history.
        Example: if ISA(A,B) exists, OPPOSE(A,B) should be flagged.
        """
        for instr in new_ir:
            if instr["op"] == "OPPOSE":
                u, v = instr["args"][0], instr["args"][1]
                if self.graph.has_edge(u, v) or self.graph.has_edge(v, u):
                    rel = "UNKNOWN"
                    src_u, src_v = (u, v) if self.graph.has_edge(u, v) else (v, u)
                    rel_map = self.graph.get_edge_data(src_u, src_v) or {}
                    if isinstance(rel_map, dict):
                        # MultiDiGraph: {key: {attrs}}
                        first_val = next(iter(rel_map.values()), None)
                        if isinstance(first_val, dict) and "relation" in first_val:
                            rel = first_val.get("relation", rel)
                        # DiGraph: {attrs}
                        elif "relation" in rel_map:
                            rel = rel_map.get("relation", rel)
                    return True, f"Tarihsel Celiski: {u} ve {v} arasinda zaten '{rel}' bagi var, 'OPPOSE' edilemez!"
        return False, "Gecmisle uyumlu."

    def visualize(self):
        """Visualize memory (debug use)."""
        pos = nx.spring_layout(self.graph)
        labels = nx.get_edge_attributes(self.graph, "relation")
        nx.draw(self.graph, pos, with_labels=True, node_color="lightblue", node_size=2000)
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=labels)
        plt.show()

    def get_relevant_context(self, node_id, depth=1):
        """Return near-neighbor relation context for a node."""
        if not self.graph.has_node(node_id):
            return []
        neighbors = list(self.graph.neighbors(node_id))
        context = []
        for n in neighbors:
            edge_bundle = self.graph.get_edge_data(node_id, n) or {}
            if isinstance(edge_bundle, dict):
                first_val = next(iter(edge_bundle.values()), None)
                if isinstance(first_val, dict) and "relation" in first_val:
                    # MultiDiGraph
                    for _, attrs in edge_bundle.items():
                        rel = attrs.get("relation")
                        context.append({"from": node_id, "to": n, "relation": rel})
                else:
                    # DiGraph
                    rel = edge_bundle.get("relation")
                    context.append({"from": node_id, "to": n, "relation": rel})
        return context


if __name__ == "__main__":
    memory = CognitiveMemory()

    past_data = [
        {"op": "DEF_ENTITY", "args": ["insan", "canli"]},
        {"op": "DEF_ENTITY", "args": ["olumluluk", "durum"]},
        {"op": "ISA", "args": ["insan", "olumluluk"]},
    ]
    memory.add_ir_chain(past_data)

    new_data = [{"op": "OPPOSE", "args": ["insan", "olumluluk"]}]
    conflict, msg = memory.find_conflicts_with_history(new_data)

    if conflict:
        print(f"DUR! {msg}")
    else:
        memory.add_ir_chain(new_data)
        print("Bilgi hafizaya eklendi.")
