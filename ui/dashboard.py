import json
import os
import re
import sys
from pathlib import Path

import networkx as nx
import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.listener import get_shadow_listener_status, read_recent_shadow_events, start_shadow_listener
from core.model_bridge import PROVIDER_SPECS, ensure_provider_client
from core.rule_guard import approve_candidate, auto_review_candidates, get_review_queue, get_rule_stats, reject_candidate
from core.quality_metrics import (
    aggregate_fallback_kpi,
    aggregate_opcode_quality,
    drift_alerts,
    edge_diversity_metrics,
    provenance_coverage,
)
from core.vscode_chat_bridge import get_vscode_chat_bridge_status, start_vscode_chat_bridge
from core.memory_manager import MemoryManager
try:
    from main import clear_global_graph, compact_global_graph, get_graph_backend, load_global_graph, run_cognitive_os
except ImportError:
    from main import clear_global_graph, get_graph_backend, load_global_graph, run_cognitive_os

    def compact_global_graph(path=None):
        graph = load_global_graph(path=path)
        return graph.number_of_edges(), graph.number_of_edges()
from core.validator import CognitiveValidator

try:
    from main import get_graph_backend_status
except ImportError:
    def get_graph_backend_status():
        active = get_graph_backend()
        return {
            "configured_backend": os.getenv("COGNITIVE_GRAPH_BACKEND", "neo4j").strip().lower(),
            "active_backend": active,
            "neo4j_connected": active == "neo4j",
            "neo4j_uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            "neo4j_database": os.getenv("NEO4J_DATABASE", "neo4j"),
            "json_path": os.getenv("COGNITIVE_GRAPH_JSON_PATH", "memory/global_graph.json"),
            "json_backup_path": os.getenv("COGNITIVE_GRAPH_JSON_BACKUP_PATH", "memory/global_graph.backup.json"),
            "fallback_to_json": active == "json",
        }

try:
    from main import get_vector_backend_status
except ImportError:
    def get_vector_backend_status():
        return {
            "configured_backend": os.getenv("COGNITIVE_VECTOR_BACKEND", "qdrant").strip().lower(),
            "active_backend": "local",
            "collection": os.getenv("COGNITIVE_VECTOR_COLLECTION", "cognitive_graph"),
            "qdrant_url": os.getenv("QDRANT_URL", "http://localhost:6333"),
            "local_path": os.getenv("COGNITIVE_VECTOR_LOCAL_PATH", "memory/vector_index.jsonl"),
        }


st.set_page_config(layout="wide", page_title="Cognitive OS Dashboard")
st.title("Cognitive OS - Live Dashboard")
st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 100% !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


if "graph" not in st.session_state:
    st.session_state["graph"] = load_global_graph()
if "decision_log" not in st.session_state:
    st.session_state["decision_log"] = []
if "z3_status" not in st.session_state:
    st.session_state["z3_status"] = "Unknown"
if "last_ir" not in st.session_state:
    st.session_state["last_ir"] = ""
if "last_processed_input" not in st.session_state:
    st.session_state["last_processed_input"] = ""
if "prompt_input" not in st.session_state:
    st.session_state["prompt_input"] = ""
if "prompt_upload_nonce" not in st.session_state:
    st.session_state["prompt_upload_nonce"] = 0
if "active_terms" not in st.session_state:
    st.session_state["active_terms"] = []
if "alerts" not in st.session_state:
    st.session_state["alerts"] = []
if "shadow_listener_booted" not in st.session_state:
    st.session_state["shadow_listener_booted"] = False
if "vscode_bridge_booted" not in st.session_state:
    st.session_state["vscode_bridge_booted"] = False
if "llm_provider" not in st.session_state:
    st.session_state["llm_provider"] = os.getenv("COGNITIVE_LLM_PROVIDER", "groq").strip().lower()
if "llm_model" not in st.session_state:
    st.session_state["llm_model"] = os.getenv("COGNITIVE_LLM_MODEL", "")
if "llm_api_key" not in st.session_state:
    st.session_state["llm_api_key"] = os.getenv("COGNITIVE_LLM_API_KEY", "")


def edge_label(data):
    return data.get("label") or data.get("relation") or ""


def merge_graphs(target, source):
    for node, attrs in source.nodes(data=True):
        target.add_node(node, **attrs)
    for u, v, attrs in source.edges(data=True):
        target.add_edge(u, v, **attrs)


def sync_graph_from_disk():
    st.session_state["graph"] = load_global_graph()


def extract_terms(text):
    return [tok for tok in re.findall(r"[\w]+", text.lower()) if tok]


def compute_active_region(graph, terms):
    active = set()
    for term in terms:
        if graph.has_node(term):
            active.add(term)
            active.update(graph.successors(term))
            active.update(graph.predecessors(term))
    return active


def split_text_for_queue(text: str, max_words: int = 1200):
    raw = (text or "").replace("\r", "\n").strip()
    if not raw:
        return []
    words = raw.split()
    if len(words) <= max_words:
        return [raw]

    parts = []
    for block in raw.split("\n\n"):
        chunk = block.strip()
        if chunk:
            parts.append(chunk)

    out = []
    buf = []
    count = 0
    for part in parts:
        w = part.split()
        if count + len(w) > max_words and buf:
            out.append("\n\n".join(buf).strip())
            buf = [part]
            count = len(w)
        else:
            buf.append(part)
            count += len(w)
    if buf:
        out.append("\n\n".join(buf).strip())
    return out


def process_prompt_payload(payload_text: str):
    sync_graph_from_disk()
    result = run_cognitive_os(payload_text)
    st.session_state["graph"] = load_global_graph()
    st.session_state["decision_log"].extend(result["log"])
    for entry in result["log"]:
        if "[DASHBOARD_ALERT]" in str(entry):
            st.session_state["alerts"].append(str(entry))
    st.session_state["z3_status"] = result["z3_status"]
    st.session_state["last_processed_input"] = payload_text
    st.session_state["active_terms"] = extract_terms(payload_text)
    st.session_state["last_ir"] = json.dumps(result["log"][-1] if result["log"] else "", ensure_ascii=False, indent=2)
    return result


st.sidebar.header("Controls")
st.sidebar.subheader("LLM Provider")
provider_options = list(PROVIDER_SPECS.keys())
provider_labels = {k: v.get("label", k) for k, v in PROVIDER_SPECS.items()}
default_provider = st.session_state["llm_provider"] if st.session_state["llm_provider"] in provider_options else "groq"
provider = st.sidebar.selectbox(
    "Model Source",
    options=provider_options,
    index=provider_options.index(default_provider),
    format_func=lambda x: provider_labels.get(x, x),
)
st.session_state["llm_provider"] = provider

default_model = st.session_state["llm_model"] or os.getenv("COGNITIVE_LLM_MODEL", "")
model_name = st.sidebar.text_input("Model Name", value=default_model, help="Provider model id (local icin Ollama modeli).")
st.session_state["llm_model"] = model_name

if provider == "local":
    st.sidebar.caption("Local mode: Ollama kullanilir. OLLAMA_URL ve model adini kontrol edin.")
    st.session_state["llm_api_key"] = ""
else:
    key = st.sidebar.text_input("API Key", value=st.session_state["llm_api_key"], type="password")
    st.session_state["llm_api_key"] = key
    auto_install = st.sidebar.toggle("Auto-install client package", value=True)
    ok_pkg, pkg_msg = ensure_provider_client(provider, auto_install=auto_install)
    if ok_pkg:
        st.sidebar.success(f"Client ready ({pkg_msg})")
    else:
        st.sidebar.error(f"Client error: {pkg_msg}")

os.environ["COGNITIVE_LLM_PROVIDER"] = provider
os.environ["COGNITIVE_LLM_MODEL"] = model_name.strip()
os.environ["COGNITIVE_LLM_API_KEY"] = st.session_state.get("llm_api_key", "").strip()
if provider != "local":
    env_key = PROVIDER_SPECS.get(provider, {}).get("env_key")
    if env_key:
        os.environ[env_key] = st.session_state.get("llm_api_key", "").strip()

if st.sidebar.button("Clear Memory"):
    clear_global_graph()
    st.session_state["graph"] = nx.MultiDiGraph()
    st.session_state["decision_log"] = []
    st.session_state["z3_status"] = "Unknown"
    st.session_state["last_ir"] = ""
    st.session_state["last_processed_input"] = ""
    st.session_state["active_terms"] = []
    st.rerun()

if st.sidebar.button("Compact Duplicate Edges"):
    before, after = compact_global_graph()
    st.session_state["graph"] = load_global_graph()
    st.sidebar.success(f"Edges compacted: {before} -> {after}")
    st.rerun()

if st.sidebar.button("Clear IR Cache"):
    MemoryManager().clear_cache()
    st.session_state["last_processed_input"] = ""
    st.sidebar.success("IR cache temizlendi.")

if not st.session_state["shadow_listener_booted"]:
    try:
        start_shadow_listener(force_restart=True)
    except TypeError:
        # Backward compatibility for older listener signatures.
        start_shadow_listener()
    st.session_state["shadow_listener_booted"] = True
if not st.session_state["vscode_bridge_booted"]:
    start_vscode_chat_bridge()
    st.session_state["vscode_bridge_booted"] = True

shadow_status = get_shadow_listener_status()
vscode_bridge_status = get_vscode_chat_bridge_status()
shadow_events = read_recent_shadow_events(limit=25)
rule_stats = get_rule_stats()
rule_queue = get_review_queue(limit=10)
for event in shadow_events:
    msg = str(event.get("message", ""))
    if "[DASHBOARD_ALERT]" in msg and msg not in st.session_state["alerts"]:
        st.session_state["alerts"].append(msg)

# Keep graph aligned with background shadow updates.
sync_graph_from_disk()


st.subheader("Prompt Input")
queue_word_limit = st.slider("Batch Size (words per queue item)", min_value=300, max_value=3000, value=1200, step=100)
with st.form("prompt_form", clear_on_submit=False):
    st.text_area(
        "Prompt",
        key="prompt_input",
        height=130,
        placeholder="Type a sentence or paste long text...",
    )
    uploaded_txt = st.file_uploader(
        "📎 Upload Text File (.txt, .md)",
        type=["txt", "md"],
        key=f"prompt_upload_{st.session_state['prompt_upload_nonce']}",
    )
    submit_prompt = st.form_submit_button("Process")

if submit_prompt:
    prompt_text = (st.session_state.get("prompt_input") or "").strip()
    file_text = ""
    if uploaded_txt is not None:
        try:
            file_text = uploaded_txt.read().decode("utf-8", errors="ignore").strip()
        except Exception:
            file_text = ""

    combined_input = "\n\n".join([x for x in [prompt_text, file_text] if x]).strip()
    if not combined_input:
        st.warning("Prompt or text file is required.")
    else:
        queue_items = split_text_for_queue(combined_input, max_words=queue_word_limit)
        if not queue_items:
            st.warning("Input could not be segmented.")
        else:
            progress = st.progress(0.0)
            status = st.empty()
            total = len(queue_items)
            for idx, chunk in enumerate(queue_items, start=1):
                status.info(f"Processing chunk {idx}/{total}...")
                process_prompt_payload(chunk)
                progress.progress(float(idx) / float(total))
            status.success(f"Completed. {total} chunk(s) processed.")
            st.session_state["prompt_input"] = ""
            st.session_state["prompt_upload_nonce"] += 1
            st.rerun()


st.sidebar.subheader("Manual IR Editor")
manual_ir = st.sidebar.text_area("Edit IR as JSON:", st.session_state["last_ir"], height=150)
if st.sidebar.button("Apply Manual IR"):
    try:
        json.loads(manual_ir)
        st.session_state["decision_log"].append(f"[MANUAL IR]: {manual_ir}")
        st.sidebar.success("Manual IR log added.")
    except Exception as e:
        st.sidebar.error(f"Invalid JSON: {e}")

st.sidebar.subheader("Shadow Listener")
z3_mode = os.getenv("COGNITIVE_USE_Z3", "0").strip().lower()
st.sidebar.write("**Z3 Mode:**", "Native" if z3_mode in {"1", "true", "yes", "on"} else "Safe (No native check)")
backend_status = get_graph_backend_status()
st.sidebar.subheader("Graph Storage")
st.sidebar.write("**Configured Backend:**", backend_status.get("configured_backend", "-"))
st.sidebar.write("**Active Backend:**", backend_status.get("active_backend", "-"))
st.sidebar.write("**Neo4j Connected:**", "Yes" if backend_status.get("neo4j_connected") else "No")
if backend_status.get("fallback_to_json"):
    st.sidebar.warning("Neo4j baglantisi yok; sistem JSON fallback kullaniyor.")
if backend_status.get("active_backend") == "json":
    st.sidebar.caption(f"JSON Path: {backend_status.get('json_path', '-')}")
else:
    st.sidebar.caption(
        f"Neo4j: {backend_status.get('neo4j_uri', '-')}/{backend_status.get('neo4j_database', '-')}"
    )
st.sidebar.write("**Graph Backend:**", get_graph_backend())
edge_rel = {}
for _, _, attrs in st.session_state["graph"].edges(data=True):
    rel = edge_label(attrs)
    if not rel:
        continue
    edge_rel[rel] = edge_rel.get(rel, 0) + 1
if edge_rel:
    st.sidebar.write("**Edge Type Counts:**")
    for rel, cnt in sorted(edge_rel.items(), key=lambda x: x[1], reverse=True)[:8]:
        st.sidebar.write(f"- {rel}: {cnt}")

validator_for_metrics = CognitiveValidator()
div = edge_diversity_metrics(st.session_state["graph"], known_opcodes=list(validator_for_metrics.opcodes.keys()))
prov = provenance_coverage(st.session_state["graph"])
alerts, _ = drift_alerts(st.session_state["graph"], known_opcodes=list(validator_for_metrics.opcodes.keys()))

st.sidebar.subheader("Quality Metrics")
st.sidebar.write(f"Entropy (norm): {div.get('norm_entropy', 0.0):.3f}")
st.sidebar.write(f"Coverage: {div.get('coverage', 0.0):.3f}")
st.sidebar.write(f"Dominance ratio: {div.get('dominance_ratio', 0.0):.3f}")
st.sidebar.write(f"CAUSE ratio: {div.get('cause_ratio', 0.0):.3f}")
st.sidebar.write(f"Provenance coverage: {prov.get('coverage', 0.0):.3f}")

fallback_kpi = aggregate_fallback_kpi()
st.sidebar.write(f"Fallback rows: {fallback_kpi.get('rows', 0)}")
st.sidebar.write(f"Fallback useful-edge ratio: {float(fallback_kpi.get('avg_useful_ratio', 0.0)):.3f}")
fb_ops = fallback_kpi.get("opcode_counts", {}) or {}
if fb_ops:
    st.sidebar.write("**Fallback Opcode Dist.:**")
    for op, cnt in sorted(fb_ops.items(), key=lambda x: x[1], reverse=True)[:6]:
        st.sidebar.write(f"- {op}: {cnt}")

if alerts:
    st.sidebar.warning("Drift alarms:")
    for a in alerts[:3]:
        st.sidebar.write(f"- {a}")

opcode_q = aggregate_opcode_quality()
if opcode_q:
    st.sidebar.write("**Per-opcode quality (proxy):**")
    ranked = sorted(opcode_q.items(), key=lambda x: (x[1].get("tp", 0), x[1].get("precision_proxy", 0.0)), reverse=True)
    for op, m in ranked[:8]:
        st.sidebar.write(f"- {op}: P={m['precision_proxy']:.2f} R={m['recall_proxy']:.2f}")

vstatus = get_vector_backend_status()
st.sidebar.subheader("Vector Storage")
st.sidebar.write("**Configured Backend:**", vstatus.get("configured_backend", "-"))
st.sidebar.write("**Active Backend:**", vstatus.get("active_backend", "-"))
st.sidebar.write("**Collection:**", vstatus.get("collection", "-"))
if vstatus.get("active_backend") == "qdrant":
    st.sidebar.caption(f"Qdrant: {vstatus.get('qdrant_url', '-')}")
else:
    st.sidebar.caption(f"Local Index: {vstatus.get('local_path', '-')}")

st.sidebar.write("**Status:**", "Running" if shadow_status.get("running") else "Stopped")
st.sidebar.write("**Processed Blocks:**", shadow_status.get("processed_blocks", 0))
st.sidebar.write("**Produced IR:**", shadow_status.get("produced_ir", 0))
st.sidebar.write("**Rejected IR:**", shadow_status.get("rejected_ir", 0))
watch_files = shadow_status.get("watch_files") or []
if watch_files:
    st.sidebar.write("**Watch Files:**")
    for wf in watch_files[:4]:
        st.sidebar.code(str(Path(wf).name))
if shadow_status.get("last_error"):
    st.sidebar.warning(shadow_status.get("last_error"))

st.sidebar.subheader("VSCode Chat Bridge")
st.sidebar.write("**Status:**", "Running" if vscode_bridge_status.get("running") else "Stopped")
if not vscode_bridge_status.get("running"):
    st.sidebar.error("CRITICAL: VSCode Chat Bridge is stopped. Chat messages will not be ingested.")
st.sidebar.write("**Watched Session Files:**", vscode_bridge_status.get("watched_files", 0))
st.sidebar.write("**Forwarded User Msg:**", vscode_bridge_status.get("forwarded_user", 0))
st.sidebar.write("**Forwarded Assistant Msg:**", vscode_bridge_status.get("forwarded_assistant", 0))
for p in (vscode_bridge_status.get("watched_paths") or [])[:3]:
    st.sidebar.code(str(Path(p).name))
if vscode_bridge_status.get("last_error"):
    st.sidebar.warning(vscode_bridge_status.get("last_error"))

intervention_log = Path("memory/interventions.jsonl")
if intervention_log.exists():
    try:
        last_rows = intervention_log.read_text(encoding="utf-8").splitlines()[-20:]
    except Exception:
        last_rows = []
    if last_rows:
        st.sidebar.subheader("Intervention Layer")
        st.sidebar.write("**Enabled:**", "Yes" if os.getenv("COGNITIVE_INTERVENTION_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"} else "No")
        st.sidebar.write("**Recent Interventions:**", len(last_rows))
        try:
            tail = json.loads(last_rows[-1])
            st.sidebar.caption(str(tail.get("message", ""))[:160])
        except Exception:
            pass

st.sidebar.subheader("Rule Review")
st.sidebar.write("**Active Rules:**", rule_stats.get("active_rules", 0))
st.sidebar.write("**Pending Candidates:**", rule_stats.get("pending_candidates", 0))
st.sidebar.write("**Need Human Review:**", rule_stats.get("pending_human_review", 0))
if st.sidebar.button("Auto Review (>=3 hits)"):
    result = auto_review_candidates(CognitiveValidator(), min_hits=3)
    st.sidebar.success(f"Auto review done. approved={result.get('approved', 0)} rejected={result.get('rejected', 0)}")
if rule_queue:
    for cand in rule_queue[:5]:
        cid = cand.get("id")
        conf = cand.get("confidence")
        conf_txt = f"{float(conf):.2f}" if conf is not None else "-"
        st.sidebar.write(
            f"`{cid}` hits={cand.get('hits', 0)} op={cand.get('op')} conf={conf_txt} human={bool(cand.get('require_human'))}"
        )
        st.sidebar.code(str(cand.get("example_text", ""))[:120])
        if cand.get("clarification_question"):
            st.sidebar.caption(f"Q: {cand.get('clarification_question')}")
        c1, c2 = st.sidebar.columns(2)
        if c1.button(f"Approve {cid}", key=f"approve_{cid}"):
            ok = approve_candidate(cid, CognitiveValidator())
            if ok:
                st.sidebar.success(f"{cid} approved")
            else:
                st.sidebar.error(f"{cid} rejected by safety gate")
        if c2.button(f"Reject {cid}", key=f"reject_{cid}"):
            reject_candidate(cid, reason="dashboard_manual")
            st.sidebar.warning(f"{cid} rejected")


st.subheader("Knowledge Graph")
G = st.session_state["graph"]
active_nodes = compute_active_region(G, st.session_state["active_terms"])
use_focus = len(active_nodes) > 0
physics_enabled = st.sidebar.toggle("Graph Physics", value=False)
max_edges = st.sidebar.slider("Max Render Edges", min_value=100, max_value=3000, value=900, step=100)
max_nodes = st.sidebar.slider("Max Render Nodes", min_value=50, max_value=1200, value=350, step=50)

render_nodes = set(G.nodes)
if use_focus:
    render_nodes = set(active_nodes)
elif len(render_nodes) > max_nodes:
    rank = sorted(G.degree, key=lambda x: x[1], reverse=True)
    render_nodes = {n for n, _ in rank[:max_nodes]}

nodes = []
for n in render_nodes:
    if use_focus and n not in active_nodes:
        nodes.append(Node(id=n, label=n, color="#cfd4dc", size=12))
    else:
        nodes.append(Node(id=n, label=n, color="#1f77b4", size=18))

edges = []
rendered_edge_count = 0
for u, v, data in G.edges(data=True):
    if u not in render_nodes or v not in render_nodes:
        continue
    if rendered_edge_count >= max_edges:
        break
    label = edge_label(data)
    if use_focus and not (u in active_nodes and v in active_nodes):
        edges.append(Edge(source=u, target=v, label=label, color="#d9dde3"))
    else:
        edges.append(Edge(source=u, target=v, label=label, color="#4a5568"))
    rendered_edge_count += 1

if G.number_of_edges() > max_edges:
    st.caption(f"Rendering first {max_edges} edges out of {G.number_of_edges()} for stability.")
graph_width_px = min(3200, max(1200, int(len(render_nodes) * 4.5)))
graph_height_px = min(1000, max(560, int(len(render_nodes) * 1.4)))
config = Config(
    width=graph_width_px,
    height=graph_height_px,
    directed=True,
    physics=physics_enabled,
    hierarchical=False,
)
agraph(nodes=nodes, edges=edges, config=config)


st.sidebar.header("Decision Log and Z3")
st.sidebar.write("**Z3 Status:**", st.session_state["z3_status"])
if st.session_state["alerts"]:
    st.sidebar.write("---")
    st.sidebar.write("**Alerts:**")
    for alert in st.session_state["alerts"][-5:][::-1]:
        st.sidebar.error(alert.replace("[DASHBOARD_ALERT]", "").strip())
st.sidebar.write("---")
st.sidebar.write("**Recent Logs:**")
for log in st.session_state["decision_log"][-20:][::-1]:
    st.sidebar.write(log)

if shadow_events:
    st.sidebar.write("---")
    st.sidebar.write("**Shadow Events:**")
    for ev in shadow_events[-8:][::-1]:
        ev_ts = ev.get("ts", "")
        ev_src = ev.get("source", "-")
        ev_msg = ev.get("message", "")
        if str(ev.get("type", "")).startswith("shadow_"):
            st.sidebar.write(f"[{ev_ts}] {ev_src}: {ev_msg}")
