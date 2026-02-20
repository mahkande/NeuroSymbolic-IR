import json
import os
import re
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph


DEFAULT_GRAPH_PATH = Path(__file__).resolve().parent / "global_graph.json"
DEFAULT_BACKUP_PATH = Path(__file__).resolve().parent / "global_graph.backup.json"


class GraphStore:
    """
    Graph persistence layer.
    Backend selection:
    - neo4j (default): COGNITIVE_GRAPH_BACKEND=neo4j
    - json fallback:   COGNITIVE_GRAPH_BACKEND=json
    """

    def __init__(self):
        self.backend = (os.getenv("COGNITIVE_GRAPH_BACKEND", "neo4j") or "neo4j").strip().lower()
        self.path = Path(os.getenv("COGNITIVE_GRAPH_JSON_PATH", str(DEFAULT_GRAPH_PATH)))
        self.backup_path = Path(os.getenv("COGNITIVE_GRAPH_JSON_BACKUP_PATH", str(DEFAULT_BACKUP_PATH)))

    def backend_name(self) -> str:
        if self.backend == "neo4j" and self._neo4j_driver() is not None:
            return "neo4j"
        return "json"

    def backend_status(self) -> dict:
        neo4j_ok = self._neo4j_driver() is not None
        active = "neo4j" if (self.backend == "neo4j" and neo4j_ok) else "json"
        fallback = self.backend == "neo4j" and not neo4j_ok
        return {
            "configured_backend": self.backend,
            "active_backend": active,
            "neo4j_connected": neo4j_ok,
            "neo4j_uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            "neo4j_database": os.getenv("NEO4J_DATABASE", "neo4j"),
            "json_path": str(self.path),
            "json_backup_path": str(self.backup_path),
            "fallback_to_json": fallback,
        }

    def load_graph(self, path: Path | None = None) -> nx.MultiDiGraph:
        if self.backend == "neo4j":
            graph = self._load_neo4j_graph()
            if graph is not None:
                return graph
        return self._load_json_graph(path or self.path)

    def save_graph(self, graph: nx.MultiDiGraph, path: Path | None = None) -> bool:
        if self.backend == "neo4j":
            if self._save_neo4j_graph(graph):
                return True
        self._save_json_graph(graph, path or self.path)
        return True

    def clear_graph(self, path: Path | None = None):
        if self.backend == "neo4j":
            if self._clear_neo4j_graph():
                return
        self._clear_json_graph(path or self.path)

    def _load_json_graph(self, path: Path) -> nx.MultiDiGraph:
        backup_path = self.backup_path if path == self.path else Path(str(path) + ".backup")

        def _read_graph(p: Path) -> nx.MultiDiGraph:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            g = json_graph.node_link_graph(
                data,
                directed=bool(data.get("directed", True)),
                multigraph=bool(data.get("multigraph", True)),
            )
            if isinstance(g, nx.MultiDiGraph):
                return g
            mg = nx.MultiDiGraph()
            for node, attrs in g.nodes(data=True):
                mg.add_node(node, **attrs)
            for u, v, attrs in g.edges(data=True):
                mg.add_edge(u, v, **attrs)
            return mg

        if not path.exists() and not backup_path.exists():
            return nx.MultiDiGraph()

        for candidate in [path, backup_path]:
            if not candidate.exists():
                continue
            try:
                return _read_graph(candidate)
            except Exception:
                continue
        return nx.MultiDiGraph()

    def _save_json_graph(self, graph: nx.MultiDiGraph, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        backup_path = self.backup_path if path == self.path else Path(str(path) + ".backup")
        data = json_graph.node_link_data(graph)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        if path.exists():
            try:
                path.replace(backup_path)
            except Exception:
                pass
        tmp_path.replace(path)

    def _clear_json_graph(self, path: Path):
        if path.exists():
            path.unlink()
        backup_path = self.backup_path if path == self.path else Path(str(path) + ".backup")
        if backup_path.exists():
            backup_path.unlink()

    @staticmethod
    def _safe_rel_type(rel: str) -> str:
        rel = (rel or "RELATED").upper()
        rel = re.sub(r"[^A-Z0-9_]", "_", rel)
        if not rel:
            rel = "RELATED"
        if rel[0].isdigit():
            rel = f"R_{rel}"
        return rel

    @staticmethod
    def _neo4j_driver():
        try:
            from neo4j import GraphDatabase
        except Exception:
            return None
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "neo4j")
        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            driver.verify_connectivity()
            return driver
        except Exception:
            return None

    def _load_neo4j_graph(self) -> nx.MultiDiGraph | None:
        driver = self._neo4j_driver()
        if driver is None:
            return None
        database = os.getenv("NEO4J_DATABASE", "neo4j")
        graph = nx.MultiDiGraph()
        try:
            with driver.session(database=database) as session:
                rows = session.run(
                    "MATCH (n) "
                    "OPTIONAL MATCH (n)-[r]->(m) "
                    "RETURN n, r, m"
                )
                for row in rows:
                    n = row["n"]
                    src = n.get("id")
                    if src is None:
                        continue
                    n_attrs = {k: v for k, v in dict(n).items() if k != "id"}
                    n_attrs.setdefault("label", src)
                    graph.add_node(src, **n_attrs)

                    rel = row["r"]
                    m = row["m"]
                    if rel is None or m is None:
                        continue
                    dst = m.get("id")
                    if dst is None:
                        continue
                    m_attrs = {k: v for k, v in dict(m).items() if k != "id"}
                    m_attrs.setdefault("label", dst)
                    graph.add_node(dst, **m_attrs)

                    edge_attrs = dict(rel)
                    if "relation" not in edge_attrs:
                        edge_attrs["relation"] = rel.type
                    graph.add_edge(src, dst, **edge_attrs)
            return graph
        except Exception:
            return None
        finally:
            driver.close()

    def _save_neo4j_graph(self, graph: nx.MultiDiGraph) -> bool:
        driver = self._neo4j_driver()
        if driver is None:
            return False
        database = os.getenv("NEO4J_DATABASE", "neo4j")
        nodes = []
        for node, attrs in graph.nodes(data=True):
            node_id = str(node)
            props = {}
            for k, v in (attrs or {}).items():
                if v is None:
                    continue
                props[str(k)] = str(v) if isinstance(v, (list, dict, set, tuple)) else v
            nodes.append({"id": node_id, "props": props})

        edge_groups = {}
        edge_key = 0
        for u, v, attrs in graph.edges(data=True):
            edge_key += 1
            rel = (attrs or {}).get("relation", "RELATED")
            rel_type = self._safe_rel_type(rel)
            props = {}
            for k, val in (attrs or {}).items():
                if val is None:
                    continue
                props[str(k)] = str(val) if isinstance(val, (list, dict, set, tuple)) else val
            props["_ekey"] = edge_key
            edge_groups.setdefault(rel_type, []).append({"u": str(u), "v": str(v), "props": props})

        try:
            with driver.session(database=database) as session:
                session.run("MATCH (n) DETACH DELETE n")
                session.run("CREATE CONSTRAINT node_id_unique IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE")
                if nodes:
                    session.run(
                        "UNWIND $nodes AS row "
                        "MERGE (n:Entity {id: row.id}) "
                        "SET n += row.props",
                        nodes=nodes,
                    )
                for rel_type, rel_rows in edge_groups.items():
                    cypher = (
                        "UNWIND $rows AS row "
                        "MATCH (a:Entity {id: row.u}) "
                        "MATCH (b:Entity {id: row.v}) "
                        f"CREATE (a)-[r:{rel_type}]->(b) "
                        "SET r += row.props"
                    )
                    session.run(cypher, rows=rel_rows)
            return True
        except Exception:
            return False
        finally:
            driver.close()

    def _clear_neo4j_graph(self) -> bool:
        driver = self._neo4j_driver()
        if driver is None:
            return False
        database = os.getenv("NEO4J_DATABASE", "neo4j")
        try:
            with driver.session(database=database) as session:
                session.run("MATCH (n) DETACH DELETE n")
            return True
        except Exception:
            return False
        finally:
            driver.close()
