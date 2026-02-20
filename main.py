import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx

from core.inference_engine import InferenceEngine
from core.deterministic_verifier import DeterministicVerifierGate
from core.embedding_pipeline import GraphEmbeddingPipeline
from core.fallback_rules import semantic_fallback_ir
from core.hybrid_semantics import RelationClassifier, RuleBasedExtractorV2, fuse_relations
from core.hybrid_retriever import HybridRetriever
from core.logic_engine import LogicEngine
from core.memory_manager import MemoryManager
from core.model_bridge import NanbeigeBridge
from core.nlp_utils import grammar_filter_ir, rebalance_relations
from core.parser import IRParser
from core.quality_metrics import drift_alerts, record_opcode_quality
from core.quality_metrics import record_fallback_usage
from core.reasoning_trace import ReasoningTraceStore
from core.rule_engine import DeterministicRuleEngine
from core.rule_learner import propose_llm_rule_candidate
from core.truth_maintenance import TruthMaintenanceEngine
from core.validator import CognitiveValidator
from core.evidence import build_proof_objects
from core.backward_verifier import BackwardVerifier
from memory.conceptnet_service import ConceptNetService
from memory.graph_store import GraphStore
from memory.knowledge_graph import CognitiveMemory
from memory.vector_store import VectorStore


_GRAPH_IO_LOCK = threading.Lock()
_GRAPH_STORE = GraphStore()
_VECTOR_STORE = VectorStore(dim=int(os.getenv("COGNITIVE_EMBED_DIM", "128")))
_VOLATILE_EDGE_KEYS = {"_ekey", "created_at", "updated_at", "ts", "timestamp"}


def _stable_value(val):
    if isinstance(val, dict):
        return tuple(sorted((str(k), _stable_value(v)) for k, v in val.items()))
    if isinstance(val, (list, tuple, set)):
        return tuple(_stable_value(v) for v in val)
    return val


def _edge_signature(u, v, attrs):
    clean = {}
    for k, v_ in (attrs or {}).items():
        if k in _VOLATILE_EDGE_KEYS:
            continue
        clean[str(k)] = _stable_value(v_)
    return str(u), str(v), tuple(sorted(clean.items()))


def _ensure_edge_provenance(graph):
    now = datetime.now(timezone.utc).isoformat()
    if isinstance(graph, nx.MultiDiGraph):
        for u, v, k, attrs in graph.edges(data=True, keys=True):
            edge = graph[u][v][k]
            edge.setdefault("source", "runtime")
            edge.setdefault("created_at", now)
            edge.setdefault("confidence", 0.5)
            if "inference_rule" not in edge:
                edge["inference_rule"] = "LEGACY_BACKFILL" if edge.get("inferred") else "DIRECT_INPUT"
    else:
        for u, v, attrs in graph.edges(data=True):
            edge = graph[u][v]
            edge.setdefault("source", "runtime")
            edge.setdefault("created_at", now)
            edge.setdefault("confidence", 0.5)
            if "inference_rule" not in edge:
                edge["inference_rule"] = "LEGACY_BACKFILL" if edge.get("inferred") else "DIRECT_INPUT"


def _merge_graphs(target, source):
    for node, attrs in source.nodes(data=True):
        target.add_node(node, **attrs)
    if isinstance(target, nx.MultiDiGraph):
        seen = set()
        for u, v, prev_attrs in target.edges(data=True):
            seen.add(_edge_signature(u, v, prev_attrs))
        for u, v, attrs in source.edges(data=True):
            sig = _edge_signature(u, v, attrs)
            if sig not in seen:
                target.add_edge(u, v, **attrs)
                seen.add(sig)
    else:
        for u, v, attrs in source.edges(data=True):
            target.add_edge(u, v, **attrs)


def load_global_graph(path=None):
    graph = _GRAPH_STORE.load_graph(path=path)
    if isinstance(graph, nx.MultiDiGraph):
        return graph
    mg = nx.MultiDiGraph()
    for node, attrs in graph.nodes(data=True):
        mg.add_node(node, **attrs)
    for u, v, attrs in graph.edges(data=True):
        mg.add_edge(u, v, **attrs)
    return mg


def save_global_graph(graph, path=None, merge_existing=True):
    with _GRAPH_IO_LOCK:
        to_save = graph.copy()
        if merge_existing:
            existing = load_global_graph(path)
            _merge_graphs(existing, to_save)
            to_save = existing
        _ensure_edge_provenance(to_save)
        _GRAPH_STORE.save_graph(to_save, path=path)


def clear_global_graph(path=None):
    with _GRAPH_IO_LOCK:
        _GRAPH_STORE.clear_graph(path=path)


def get_graph_backend():
    return _GRAPH_STORE.backend_name()


def get_graph_backend_status():
    return _GRAPH_STORE.backend_status()


def get_vector_backend():
    return _VECTOR_STORE.backend_name()


def get_vector_backend_status():
    return _VECTOR_STORE.backend_status()


def get_default_retrieval_mode():
    env_mode = os.getenv("COGNITIVE_RETRIEVAL_MODE", "").strip().lower()
    valid = {"vector-only", "graph-only", "hybrid"}
    if env_mode in valid:
        return env_mode

    strategy_path = Path("spec/retrieval_strategy.json")
    if strategy_path.exists():
        try:
            import json

            payload = json.loads(strategy_path.read_text(encoding="utf-8-sig"))
            file_mode = str(payload.get("default_mode", "")).strip().lower()
            if file_mode in valid:
                return file_mode
        except Exception:
            pass
    return "hybrid"


def update_global_graph(ir_chain, base_graph=None):
    graph = base_graph.copy() if base_graph is not None else load_global_graph()
    if ir_chain:
        temp_memory = CognitiveMemory()
        temp_memory.graph = graph
        temp_memory.add_ir_chain(ir_chain)
        graph = temp_memory.graph
    save_global_graph(graph)
    return graph


def compact_global_graph(path=None):
    with _GRAPH_IO_LOCK:
        graph = load_global_graph(path=path)
        compacted = nx.MultiDiGraph()
        for node, attrs in graph.nodes(data=True):
            compacted.add_node(node, **attrs)

        seen = set()
        for u, v, attrs in graph.edges(data=True):
            sig = _edge_signature(u, v, attrs)
            if sig in seen:
                continue
            seen.add(sig)
            compacted.add_edge(u, v, **attrs)

        _GRAPH_STORE.save_graph(compacted, path=path)
        return graph.number_of_edges(), compacted.number_of_edges()


def run_cognitive_os(user_input):
    parser = IRParser()
    validator = CognitiveValidator()
    memory = CognitiveMemory()
    memory.graph = load_global_graph()

    z3_engine = LogicEngine()
    bridge = NanbeigeBridge()
    conceptnet = ConceptNetService(language="tr")
    inference = InferenceEngine(memory.graph)
    verifier_profile = os.getenv("COGNITIVE_VERIFIER_PROFILE", "balanced").strip().lower()
    verifier_strict = os.getenv("COGNITIVE_VERIFIER_STRICT", "0").strip().lower() in {"1", "true", "yes", "on"}
    verifier = DeterministicVerifierGate(
        validator,
        z3_engine,
        profile=verifier_profile,
        strict_mode=verifier_strict,
    )
    embed_pipeline = GraphEmbeddingPipeline(_VECTOR_STORE, dim=_VECTOR_STORE.dim)
    retriever = HybridRetriever(memory.graph, _VECTOR_STORE, dim=_VECTOR_STORE.dim)
    retrieval_mode = get_default_retrieval_mode()
    trace_store = ReasoningTraceStore()
    backward_verifier = BackwardVerifier()
    rule_engine = DeterministicRuleEngine(memory.graph)
    truth_maintenance = TruthMaintenanceEngine(memory.graph)
    rb_v2 = RuleBasedExtractorV2()
    classifier = RelationClassifier()
    cache = MemoryManager()

    logs = []
    isa_schema = validator.isa
    allowed_ops = set(validator.opcodes.keys())
    proof_objects = []

    def is_do_only(chain):
        rows = [r for r in (chain or []) if isinstance(r, dict)]
        if not rows:
            return False
        return all(str(r.get("op", "")).upper() == "DO" for r in rows)

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
        return {"graph": memory.graph, "log": logs, "z3_status": z3_status, "proofs": proof_objects}

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
    base_predicted_ops = []
    if cached_ir:
        ir_chain = cached_ir
        base_predicted_ops = [str(i.get("op", "")) for i in (ir_chain or []) if isinstance(i, dict)]
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
                fallback_chunk = semantic_fallback_ir(chunk, include_do=True)
                if fallback_chunk:
                    ir_chain.extend(fallback_chunk)
                    record_fallback_usage("main", chunk, fallback_chunk, reason="compile_error")
                    logs.append(f"[FALLBACK] Chunk {idx}/{len(chunks)} semantik fallback ile {len(fallback_chunk)} bag uretildi.")
                continue

            parsed_chunk = parser.parse_raw_output(llm_raw_response, allowed_ops=allowed_ops, strict_schema=True)
            if isinstance(parsed_chunk, dict) and "error" in parsed_chunk:
                logs.append(f"LLM Parse Hatasi (chunk {idx}/{len(chunks)}): {parsed_chunk['error']}")
                fallback_chunk = semantic_fallback_ir(chunk, include_do=True)
                if fallback_chunk:
                    ir_chain.extend(fallback_chunk)
                    record_fallback_usage("main", chunk, fallback_chunk, reason="parse_error")
                    logs.append(f"[FALLBACK] Parse hatasi sonrasi chunk {idx}/{len(chunks)} icin {len(fallback_chunk)} semantik bag eklendi.")
                continue
            if isinstance(parsed_chunk, list):
                ir_chain.extend(parsed_chunk)

        if not ir_chain:
            fallback_all = semantic_fallback_ir(user_input, include_do=True)
            if fallback_all:
                ir_chain.extend(fallback_all)
                record_fallback_usage("main", user_input, fallback_all, reason="all_chunks_empty")
                logs.append(f"[FALLBACK] Tum chunklar bos kaldi, tum metinden {len(fallback_all)} semantik bag uretildi.")
            else:
                return finalize("Hata")
        if not ir_chain:
            return finalize("Hata")
        base_predicted_ops = [str(i.get("op", "")) for i in (ir_chain or []) if isinstance(i, dict)]

    # Faz 3 hibrit semantik katmani: rule-v2 + classifier + belirsizlikte LLM hakemligi.
    rule_candidates = rb_v2.extract(user_input)
    base_predicted_ops.extend(str(c.get("op", "")) for c in rule_candidates if isinstance(c, dict))
    cls_probs = classifier.predict(user_input)

    def arbitration_call(text):
        arb_raw = bridge.compile_to_ir(
            text,
            isa_schema,
            conceptnet_hints=", ".join(hints) if hints else None,
            max_retries=1,
            memory_terms=list(memory.graph.nodes),
        )
        if isinstance(arb_raw, dict) and "error" in arb_raw:
            logs.append(f"[HYBRID] LLM arbitration hatasi: {arb_raw['error']}")
            return []
        parsed = parser.parse_raw_output(arb_raw, allowed_ops=allowed_ops, strict_schema=True)
        return parsed if isinstance(parsed, list) else []

    ir_chain, hybrid_logs = fuse_relations(
        user_input,
        ir_chain,
        rule_candidates,
        cls_probs,
        arbitration_cb=arbitration_call if not cached_ir else None,
    )
    logs.extend(hybrid_logs)

    # Dil bilgisel denetim: POS filtre + lemma normalize (zeyrek)
    ir_chain, gf_logs, attr_augment = grammar_filter_ir(ir_chain, known_entities=list(memory.graph.nodes))
    logs.extend(gf_logs)
    if attr_augment:
        logs.append(f"[GRAMMAR_FILTER] Ek ATTR sayisi: {len(attr_augment)}")
        ir_chain.extend(attr_augment)
    ir_chain, rb_logs = rebalance_relations(ir_chain, user_input)
    logs.extend(rb_logs)
    if not ir_chain:
        proposal = propose_llm_rule_candidate(
            user_input,
            validator,
            provider=bridge.provider,
            model_name=bridge.model_name,
            api_key=bridge.api_key,
            source="main_llm_rule_bootstrap",
        )
        if proposal and proposal.get("ir"):
            ir_chain = proposal.get("ir", [])
            logs.append("[FALLBACK] Bos IR sonrasi LLM rule bootstrap denendi.")
            if proposal.get("question"):
                logs.append(f"[RULE_BOOTSTRAP] Soru: {proposal.get('question')}")
        if not ir_chain:
            logs.append("[GRAMMAR_FILTER] Tum IR baglari elendi; hafizaya yazilmadi.")
            return finalize("Hata")

    if is_do_only(ir_chain):
        proposal = propose_llm_rule_candidate(
            user_input,
            validator,
            provider=bridge.provider,
            model_name=bridge.model_name,
            api_key=bridge.api_key,
            source="main_llm_rule_bootstrap",
        )
        if proposal:
            if proposal.get("question"):
                logs.append(f"[RULE_BOOTSTRAP] Soru: {proposal.get('question')}")
            if proposal.get("auto_apply"):
                cand_ir = proposal.get("ir", [])
                if cand_ir and not is_do_only(cand_ir):
                    ir_chain = cand_ir
                    logs.append("[FALLBACK] DO-only IR yerine LLM rule bootstrap IR kullanildi.")

    history_ir_for_gate = []
    if isinstance(memory.graph, nx.MultiDiGraph):
        for u, v, _, attrs in memory.graph.edges(data=True, keys=True):
            op = (attrs or {}).get("relation") or (attrs or {}).get("label")
            if op in validator.opcodes:
                history_ir_for_gate.append({"op": op, "args": [u, v]})
    else:
        for u, v, attrs in memory.graph.edges(data=True):
            op = (attrs or {}).get("relation") or (attrs or {}).get("label")
            if op in validator.opcodes:
                history_ir_for_gate.append({"op": op, "args": [u, v]})

    prewrite_evidence = retriever.retrieve(user_input, focus_terms=[main_entity], top_k=8, mode=retrieval_mode)
    proof_objects, ir_chain = build_proof_objects(ir_chain, prewrite_evidence)
    supported = sum(1 for p in proof_objects if p.get("verdict") == "supported")
    if proof_objects:
        logs.append(f"[PROOF] supported={supported}/{len(proof_objects)}")
        trace_store.record_claim_proofs(main_entity, proof_objects, max_records=96)

    if os.getenv("COGNITIVE_USE_BACKWARD_VERIFIER", "1").strip().lower() in {"1", "true", "yes", "on"}:
        bwd_ok, bwd_msg, bwd_report = backward_verifier.verify(ir_chain, history_ir_for_gate, proof_objects=proof_objects)
        logs.append(f"[BACKWARD_VERIFIER] {bwd_msg}")
        if not bwd_ok:
            unsupported = [r for r in (bwd_report or []) if not r.get("supported")]
            logs.append(f"[BACKWARD_VERIFIER] unsupported examples: {str(unsupported[:2])}")
            return finalize("UNSAT")

    gate_ok, gate_msg, gated_ir = verifier.verify(ir_chain, history_ir=history_ir_for_gate)
    logs.append(f"[VERIFIER] {gate_msg}")
    if not gate_ok:
        return finalize("UNSAT")
    ir_chain = gated_ir

    if not cached_ir:
        cache.add_ir(user_input, ir_chain)

    for instr in ir_chain:
        is_valid, msg = validator.validate_instruction(
            instr["op"],
            instr["args"],
            known_entities=list(memory.graph.nodes),
        )
        if not is_valid:
            logs.append(f"Sozdizimi Hatasi: {msg}")
            corrected = bridge.feedback_correction(
                user_input,
                isa_schema,
                msg,
                memory_terms=list(memory.graph.nodes),
            )
            ir_chain = parser.parse_raw_output(corrected, allowed_ops=allowed_ops, strict_schema=True)
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
        revision_ir = parser.parse_raw_output(revision_raw, allowed_ops=allowed_ops, strict_schema=True)
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
        ir_chain = parser.parse_raw_output(corrected, allowed_ops=allowed_ops, strict_schema=True)
        if isinstance(ir_chain, dict) and "error" in ir_chain:
            logs.append(f"LLM Cikti Hatasi: {ir_chain['error']}")
            return finalize("Hata")
        is_conflict, conflict_msg = memory.find_conflicts_with_history(ir_chain)
        if is_conflict:
            logs.append(f"Duzeltme Sonrasi Hafiza Hatasi: {conflict_msg}")
            return finalize("UNSAT")

    memory.graph = update_global_graph(ir_chain, base_graph=memory.graph)
    final_ops = [str(i.get("op", "")) for i in (ir_chain or []) if isinstance(i, dict)]
    record_opcode_quality(base_predicted_ops, final_ops)

    removed_edges = truth_maintenance.resolve()
    if removed_edges:
        logs.append(f"[TRUTH_MAINTENANCE] Removed {removed_edges} conflicting edges.")

    drift_msgs, _ = drift_alerts(memory.graph, known_opcodes=list(validator.opcodes.keys()))
    for msg in drift_msgs:
        logs.append(f"[DRIFT_ALERT] {msg}")

    n_nodes, n_edges, n_upserts = embed_pipeline.index_graph(memory.graph, focus=main_entity, max_subgraphs=48)
    logs.append(f"[VECTOR] Indexed graph (nodes={n_nodes}, edges={n_edges}, upserts={n_upserts}, backend={_VECTOR_STORE.backend_name()}).")
    evidence = retriever.retrieve(user_input, focus_terms=[main_entity], top_k=6, mode=retrieval_mode)
    if evidence:
        head = "; ".join(e.get("text", "") for e in evidence[:3])
        logs.append(f"[RETRIEVER] mode={retrieval_mode} evidence sample: {head}")
    logs.append("Dusunce basariyla dogrulandi ve hafizaya islendi.")
    logs.append(f"Baglamsal Hafiza: {context}")

    def background_inference():
        added_rules = rule_engine.run(max_iterations=3, max_new=1200)
        if added_rules:
            logs.append(f"[RULE_ENGINE] Deterministic inference added {added_rules} edges.")
        inference.infer_isa_attr_inheritance(max_new=800)
        inference.transitive_discovery()
        trace_forward = trace_store.record_forward(memory.graph, main_entity, max_depth=2, max_records=25)
        if trace_forward:
            logs.append(f"[TRACE] Recorded {trace_forward} forward reasoning chains.")
        hypotheses = inference.abductive_reasoning(main_entity, max_results=5, persist=True)
        if hypotheses:
            logs.append(f"[ABDUCTION] Added {len(hypotheses)} abductive hypotheses for '{main_entity}'.")
        trace_inverse = trace_store.record_inverse(main_entity, hypotheses, max_records=25)
        if trace_inverse:
            logs.append(f"[TRACE] Recorded {trace_inverse} inverse reasoning chains.")
        removed_bg = truth_maintenance.resolve()
        if removed_bg:
            logs.append(f"[TRUTH_MAINTENANCE] Background removed {removed_bg} edges.")
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
