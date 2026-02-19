import requests
import networkx as nx
import time

class ConceptNetService:
    def __init__(self, language="tr", cache_ttl=3600):
        self.language = language
        self.cache = {}
        self.cache_ttl = cache_ttl  # seconds

    def _cache_key(self, concept):
        return f"{self.language}:{concept.lower()}"

    def _is_cached(self, key):
        return key in self.cache and (time.time() - self.cache[key]["time"]) < self.cache_ttl

    def query(self, concept):
        key = self._cache_key(concept)
        if self._is_cached(key):
            return self.cache[key]["data"]
        url = f"http://api.conceptnet.io/c/{self.language}/{concept.lower()}"
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            self.cache[key] = {"data": data, "time": time.time()}
            return data
        except Exception as e:
            return None

    def extract_facts(self, data):
        """ConceptNet API yanıtından hızlıca IR üretilebilecek ilişkileri çıkarır."""
        if not data or "edges" not in data:
            return []
        irs = []
        for edge in data["edges"]:
            rel = edge["rel"]["label"]
            start = edge["start"]["label"]
            end = edge["end"]["label"]
            # Sık kullanılan ilişkiler için IR öner
            if rel == "IsA":
                irs.append({"op": "DEF_ENTITY", "args": [start, end], "source": "CommonSense"})
            elif rel == "HasProperty":
                irs.append({"op": "ATTR", "args": [start, "özellik", end], "source": "CommonSense"})
            elif rel == "PartOf":
                irs.append({"op": "PART_OF", "args": [start, end], "source": "CommonSense"})
        return irs

    def add_to_graph(self, graph, irs):
        for ir in irs:
            if ir["op"] == "DEF_ENTITY":
                graph.add_edge(ir["args"][0], ir["args"][1], label="IsA", source="CommonSense")
            elif ir["op"] == "ATTR":
                graph.add_node(ir["args"][0])
                graph.add_node(ir["args"][2])
                graph.add_edge(ir["args"][0], ir["args"][2], label="HasProperty", source="CommonSense")
            elif ir["op"] == "PART_OF":
                graph.add_edge(ir["args"][0], ir["args"][1], label="PartOf", source="CommonSense")
