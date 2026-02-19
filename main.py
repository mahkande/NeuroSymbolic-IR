from pathlib import Path
import json
import re
import threading
import time

import networkx as nx
from networkx.readwrite import json_graph

from core.inference_engine import InferenceEngine
from core.logic_engine import LogicEngine
from core.memory_manager import MemoryManager
from core.model_bridge import NanbeigeBridge
from core.nlp_utils import grammar_filter_ir, rebalance_relations
from core.parser import IRParser
from core.validator import CognitiveValidator
from memory.conceptnet_service import ConceptNetService
from memory.knowledge_graph import CognitiveMemory


GLOBAL_GRAPH_PATH = Path(__file__).resolve().parent / "memory" / "global_graph.json"
GLOBAL_GRAPH_BACKUP_PATH = Path(__file__).resolve().parent / "memory" / "global_graph.backup.json"
_GRAPH_IO_LOCK = threading.Lock()


def _merge_graphs(target, source):
    for node, attrs in source.nodes(data=True):
        target.add_node(node, **attrs)
    if isinstance(target, nx.MultiDiGraph):
        for u, v, attrs in source.edges(data=True):
            exists = False
            if target.has_edge(u, v):
                edge_map = target.get_edge_data(u, v) or {}
                for _, prev_attrs in edge_map.items():
                    if prev_attrs == attrs:
                        exists = True
                        break
            if not exists:
                target.add_edge(u, v, **attrs)
    else:
        for u, v, attrs in source.edges(data=True):
            target.add_edge(u, v, **attrs)


def load_global_graph(path=GLOBAL_GRAPH_PATH):
    path = Path(path)
    backup_path = GLOBAL_GRAPH_BACKUP_PATH if path == GLOBAL_GRAPH_PATH else Path(str(path) + ".backup")

    def _read_graph(p):
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


def save_global_graph(graph, path=GLOBAL_GRAPH_PATH, merge_existing=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    backup_path = GLOBAL_GRAPH_BACKUP_PATH if path == GLOBAL_GRAPH_PATH else Path(str(path) + ".backup")

    with _GRAPH_IO_LOCK:
        to_save = graph.copy()
        if merge_existing and path.exists():
            existing = load_global_graph(path)
            _merge_graphs(existing, to_save)
            to_save = existing

        data = json_graph.node_link_data(to_save)
        tmp_path = path.with_suffix(path.suffix + ".tmp")

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        if path.exists():
            try:
                path.replace(backup_path)
            except Exception:
                pass
        tmp_path.replace(path)


def clear_global_graph(path=GLOBAL_GRAPH_PATH):
    path = Path(path)
    if path.exists():
        path.unlink()
    backup_path = GLOBAL_GRAPH_BACKUP_PATH if path == GLOBAL_GRAPH_PATH else Path(str(path) + ".backup")
    if backup_path.exists():
        backup_path.unlink()


def update_global_graph(ir_chain, base_graph=None):
    graph = base_graph.copy() if base_graph is not None else load_global_graph()
    if ir_chain:
        temp_memory = CognitiveMemory()
        temp_memory.graph = graph
        temp_memory.add_ir_chain(ir_chain)
        graph = temp_memory.graph
    save_global_graph(graph)
    return graph


def run_cognitive_os(user_input):
    parser = IRParser()
    validator = CognitiveValidator()
    memory = CognitiveMemory()
    memory.graph = load_global_graph()

    z3_engine = LogicEngine()
    bridge = NanbeigeBridge()
    conceptnet = ConceptNetService(language="tr")
    inference = InferenceEngine(memory.graph)
    cache = MemoryManager()

    logs = []
    isa_schema = validator.isa

    def split_input_chunks(text: str, max_words: int = 280):
        raw = (text or "").strip()
        if not raw:
            return []
        words = raw.split()
        if len(words) <= max_words:
            return [raw]
        sentence_like = re.split(r"(?<=[\.\!\?\n])\s+", raw)
        chunks = []
        buf = []
        count = 0
        for part in sentence_like:
            p = part.strip()
            if not p:
                continue
            w = p.split()
            if count + len(w) > max_words and buf:
                chunks.append(" ".join(buf).strip())
                buf = [p]
                count = len(w)
            else:
                buf.append(p)
                count += len(w)
        if buf:
            chunks.append(" ".join(buf).strip())
        return chunks

    def finalize(z3_status):
        save_global_graph(memory.graph)
        return {"graph": memory.graph, "log": logs, "z3_status": z3_status}

    main_entity = user_input.split()[0]
    context = memory.get_relevant_context(main_entity)
    concept_data = conceptnet.query(main_entity)
    conceptnet_irs = conceptnet.extract_facts(concept_data)

    hints = []
    if conceptnet_irs:
        for ir in conceptnet_irs:
            if ir["op"] == "DEF_ENTITY":
                hints.append(f"IsA({ir['args'][0]}, {ir['args'][1]})")
            elif ir["op"] == "ATTR":
                hints.append(f"HasProperty({ir['args'][0]}, {ir['args'][2]})")
            elif ir["op"] == "PART_OF":
                hints.append(f"PartOf({ir['args'][0]}, {ir['args'][1]})")
        conceptnet.add_to_graph(memory.graph, conceptnet_irs)
        logs.append(f"HINT: {', '.join(hints)}")

    cached_ir = cache.get_ir(user_input)
    if cached_ir:
        ir_chain = cached_ir
        logs.append("[CACHE] IR cache'den yuklendi.")
    else:
        ir_chain = []
        chunks = split_input_chunks(user_input, max_words=280)
        if len(chunks) > 1:
            logs.append(f"[CHUNKING] Uzun metin {len(chunks)} parcaya bolundu.")
        for idx, chunk in enumerate(chunks, start=1):
            llm_raw_response = bridge.compile_to_ir(
                chunk,
                isa_schema,
                conceptnet_hints=", ".join(hints) if hints else None,
                memory_terms=list(memory.graph.nodes),
            )
            if isinstance(llm_raw_response, dict) and "error" in llm_raw_response:
                logs.append(f"LLM Cikti Hatasi (chunk {idx}/{len(chunks)}): {llm_raw_response['error']}")
                continue

            parsed_chunk = parser.parse_raw_output(llm_raw_response)
            if isinstance(parsed_chunk, dict) and "error" in parsed_chunk:
                logs.append(f"LLM Parse Hatasi (chunk {idx}/{len(chunks)}): {parsed_chunk['error']}")
                continue
            if isinstance(parsed_chunk, list):
                ir_chain.extend(parsed_chunk)

        if not ir_chain:
            return finalize("Hata")

    # Dil bilgisel denetim: POS filtre + lemma normalize (zeyrek)
    ir_chain, gf_logs, attr_augment = grammar_filter_ir(ir_chain, known_entities=list(memory.graph.nodes))
    logs.extend(gf_logs)
    if attr_augment:
        logs.append(f"[GRAMMAR_FILTER] Ek ATTR sayisi: {len(attr_augment)}")
        ir_chain.extend(attr_augment)
    ir_chain, rb_logs = rebalance_relations(ir_chain, user_input)
    logs.extend(rb_logs)
    if not ir_chain:
        logs.append("[GRAMMAR_FILTER] Tum IR baglari elendi; hafizaya yazilmadi.")
        return finalize("Hata")

    if not cached_ir:
        cache.add_ir(user_input, ir_chain)

    for instr in ir_chain:
        is_valid, msg = validator.validate_instruction(instr["op"], instr["args"])
        if not is_valid:
            logs.append(f"Sozdizimi Hatasi: {msg}")
            corrected = bridge.feedback_correction(
                user_input,
                isa_schema,
                msg,
                memory_terms=list(memory.graph.nodes),
            )
            ir_chain = parser.parse_raw_output(corrected)
            if isinstance(ir_chain, dict) and "error" in ir_chain:
                logs.append(f"LLM Cikti Hatasi: {ir_chain['error']}")
                return finalize("Hata")
            break

    max_retries = 2
    retry_count = 0

    sub_nodes = {main_entity}
    if main_entity in memory.graph:
        sub_nodes.update(memory.graph.neighbors(main_entity))

    subgraph_ir = []
    for instr in ir_chain:
        if all(arg in sub_nodes for arg in instr.get("args", [])):
            subgraph_ir.append(instr)
    if not subgraph_ir:
        subgraph_ir = ir_chain

    is_consistent, logic_msg = z3_engine.verify_consistency(subgraph_ir)
    logs.append(f"Z3: {logic_msg}")

    while not is_consistent and retry_count < max_retries:
        logs.append(f"Mantik Hatasi: {logic_msg}")
        unsat_core = z3_engine.find_minimal_unsat_core(ir_chain)

        old_ir = []
        for node in unsat_core:
            if memory.graph.has_node(node["args"][0]):
                old_ir.append({"op": node["op"], "args": node["args"]})

        revision_raw = bridge.request_revision(unsat_core, old_ir, isa_schema)
        revision_ir = parser.parse_raw_output(revision_raw)
        if isinstance(revision_ir, dict) and "error" in revision_ir:
            logs.append(f"LLM Cikti Hatasi: {revision_ir['error']}")
            return finalize("Hata")

        temp_memory = CognitiveMemory()
        temp_memory.graph = memory.graph.copy()
        temp_memory.add_ir_chain(revision_ir)
        is_consistent, logic_msg = z3_engine.verify_consistency(revision_ir)
        retry_count += 1

        if is_consistent:
            memory.graph = temp_memory.graph
            ir_chain = revision_ir
            logs.append("Hafiza guncellendi.")
            break
        logs.append(f"Celiski Cozumleme Sonrasi Mantik Hatasi: {logic_msg}")

    if not is_consistent:
        logs.append("Celiski cozumleme basarisiz, hafiza degismedi.")
        return finalize("UNSAT")

    is_conflict, conflict_msg = memory.find_conflicts_with_history(ir_chain)
    if is_conflict:
        logs.append(f"Hafiza Hatasi: {conflict_msg}")
        corrected = bridge.feedback_correction(
            user_input,
            isa_schema,
            conflict_msg,
            memory_terms=list(memory.graph.nodes),
        )
        ir_chain = parser.parse_raw_output(corrected)
        if isinstance(ir_chain, dict) and "error" in ir_chain:
            logs.append(f"LLM Cikti Hatasi: {ir_chain['error']}")
            return finalize("Hata")
        is_conflict, conflict_msg = memory.find_conflicts_with_history(ir_chain)
        if is_conflict:
            logs.append(f"Duzeltme Sonrasi Hafiza Hatasi: {conflict_msg}")
            return finalize("UNSAT")

    memory.graph = update_global_graph(ir_chain, base_graph=memory.graph)
    logs.append("Dusunce basariyla dogrulandi ve hafizaya islendi.")
    logs.append(f"Baglamsal Hafiza: {context}")

    def background_inference():
        inference.transitive_discovery()
        save_global_graph(memory.graph)

    def background_enrichment():
        isolated = [n for n in memory.graph.nodes if memory.graph.degree[n] == 0]
        for node in isolated:
            inference.conceptnet_enrichment(node)
        time.sleep(5)
        save_global_graph(memory.graph)

    threading.Thread(target=background_inference, daemon=True).start()
    threading.Thread(target=background_enrichment, daemon=True).start()

    return finalize(logic_msg)


def stress_test():
    print("\n--- A Senaryosu: Mantiksal Tutarlilik Testi ---")
    run_cognitive_os("Ates sicaktir.")
    run_cognitive_os("Ates soguktur.")

    print("\n--- B Senaryosu: Geciskenlik (Transitivity) Testi ---")
    run_cognitive_os("Limon bir meyvedir.")
    run_cognitive_os("Meyveler bitkiseldir.")

    memory = CognitiveMemory()
    memory.graph = load_global_graph()
    has_nodes = memory.graph.has_node("limon") and memory.graph.has_node("bitki")
    path = nx.shortest_path(memory.graph, source="limon", target="bitki") if has_nodes else None
    if path:
        print(f"Limon -> bitki yolu: {path}. Sonuc: Evet.")
    else:
        print("Limon -> bitki yolu bulunamadi.")

    print("\n--- C Senaryosu: Celiski Cozumleme (Update) Testi ---")
    run_cognitive_os("Yazilim hatasizdir.")
    run_cognitive_os("Yazilimda buglar olabilir.")


if __name__ == "__main__":
    stress_test()
