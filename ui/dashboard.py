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
from core.rule_guard import approve_candidate, auto_review_candidates, get_review_queue, get_rule_stats, reject_candidate
from core.vscode_chat_bridge import get_vscode_chat_bridge_status, start_vscode_chat_bridge
from core.memory_manager import MemoryManager
from main import clear_global_graph, load_global_graph, run_cognitive_os
from core.validator import CognitiveValidator


st.set_page_config(layout="wide", page_title="Cognitive OS Dashboard")
st.title("Cognitive OS - Live Dashboard")


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
if "active_terms" not in st.session_state:
    st.session_state["active_terms"] = []
if "alerts" not in st.session_state:
    st.session_state["alerts"] = []
if "shadow_listener_booted" not in st.session_state:
    st.session_state["shadow_listener_booted"] = False
if "vscode_bridge_booted" not in st.session_state:
    st.session_state["vscode_bridge_booted"] = False


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


st.sidebar.header("Controls")
if st.sidebar.button("Clear Memory"):
    clear_global_graph()
    st.session_state["graph"] = nx.DiGraph()
    st.session_state["decision_log"] = []
    st.session_state["z3_status"] = "Unknown"
    st.session_state["last_ir"] = ""
    st.session_state["last_processed_input"] = ""
    st.session_state["active_terms"] = []
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


user_input = st.text_input("Enter a sentence and press Enter:", "")

if user_input and user_input != st.session_state["last_processed_input"]:
    sync_graph_from_disk()
    result = run_cognitive_os(user_input)

    # main.py persists to disk; keep disk graph as source of truth.
    st.session_state["graph"] = load_global_graph()

    st.session_state["decision_log"].extend(result["log"])
    for entry in result["log"]:
        if "[DASHBOARD_ALERT]" in str(entry):
            st.session_state["alerts"].append(str(entry))
    st.session_state["z3_status"] = result["z3_status"]
    st.session_state["last_processed_input"] = user_input
    st.session_state["active_terms"] = extract_terms(user_input)
    st.session_state["last_ir"] = json.dumps(result["log"][-1] if result["log"] else "", ensure_ascii=False, indent=2)


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
st.sidebar.write("**Watched Session Files:**", vscode_bridge_status.get("watched_files", 0))
st.sidebar.write("**Forwarded User Msg:**", vscode_bridge_status.get("forwarded_user", 0))
st.sidebar.write("**Forwarded Assistant Msg:**", vscode_bridge_status.get("forwarded_assistant", 0))
for p in (vscode_bridge_status.get("watched_paths") or [])[:3]:
    st.sidebar.code(str(Path(p).name))
if vscode_bridge_status.get("last_error"):
    st.sidebar.warning(vscode_bridge_status.get("last_error"))

st.sidebar.subheader("Rule Review")
st.sidebar.write("**Active Rules:**", rule_stats.get("active_rules", 0))
st.sidebar.write("**Pending Candidates:**", rule_stats.get("pending_candidates", 0))
if st.sidebar.button("Auto Review (>=3 hits)"):
    result = auto_review_candidates(CognitiveValidator(), min_hits=3)
    st.sidebar.success(f"Auto review done. approved={result.get('approved', 0)} rejected={result.get('rejected', 0)}")
if rule_queue:
    for cand in rule_queue[:5]:
        cid = cand.get("id")
        st.sidebar.write(f"`{cid}` hits={cand.get('hits', 0)} op={cand.get('op')}")
        st.sidebar.code(str(cand.get("example_text", ""))[:120])
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

nodes = []
for n in G.nodes:
    if use_focus and n not in active_nodes:
        nodes.append(Node(id=n, label=n, color="#cfd4dc", size=12))
    else:
        nodes.append(Node(id=n, label=n, color="#1f77b4", size=18))

edges = []
for u, v, data in G.edges(data=True):
    label = edge_label(data)
    if use_focus and not (u in active_nodes and v in active_nodes):
        edges.append(Edge(source=u, target=v, label=label, color="#d9dde3"))
    else:
        edges.append(Edge(source=u, target=v, label=label, color="#4a5568"))

config = Config(width=1000, height=560, directed=True, physics=True, hierarchical=False)
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
