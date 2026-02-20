import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory.graph_store import GraphStore


def main():
    parser = argparse.ArgumentParser(description="Migrate graph JSON snapshot into Neo4j backend.")
    parser.add_argument("--json", default="memory/global_graph.json", help="Path to source graph json")
    parser.add_argument("--backup", default="memory/global_graph.backup.json", help="Optional backup path")
    args = parser.parse_args()

    source = GraphStore()
    source.backend = "json"
    source.path = Path(args.json).resolve()
    source.backup_path = Path(args.backup).resolve()
    graph = source.load_graph(path=source.path)

    target = GraphStore()
    target.backend = "neo4j"
    ok = target.save_graph(graph)
    active_backend = target.backend_name()
    if not ok or active_backend != "neo4j":
        raise SystemExit("Neo4j migration failed. Check NEO4J_* environment variables and connectivity.")

    print(f"Migration completed to neo4j. nodes={graph.number_of_nodes()} edges={graph.number_of_edges()}")


if __name__ == "__main__":
    if not os.getenv("COGNITIVE_GRAPH_BACKEND"):
        os.environ["COGNITIVE_GRAPH_BACKEND"] = "neo4j"
    main()
