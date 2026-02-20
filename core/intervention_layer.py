import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from core.fallback_rules import semantic_fallback_ir
from core.logic_engine import LogicEngine
from memory.graph_store import GraphStore


INTERVENTION_LOG = Path("memory/interventions.jsonl")
INTERVENTION_OUTBOX = Path(os.getenv("COGNITIVE_INTERVENTION_OUTBOX", "memory/intervention_outbox.txt"))


def _write_event(row: dict):
    INTERVENTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with INTERVENTION_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_outbox(text: str):
    INTERVENTION_OUTBOX.parent.mkdir(parents=True, exist_ok=True)
    with INTERVENTION_OUTBOX.open("a", encoding="utf-8") as f:
        f.write(text.strip() + "\n\n")


def _rule_based_risk(text: str) -> Optional[str]:
    low = (text or "").lower()
    high_risk_terms = [
        "sil ",
        "delete",
        "drop database",
        "reset --hard",
        "production",
        "şifre",
        "password",
        "api key",
    ]
    for t in high_risk_terms:
        if t in low:
            return f"high_risk_term:{t.strip()}"
    return None


def analyze_intervention(text: str, source: str = "chat"):
    payload = (text or "").strip()
    if len(payload) < 12:
        return None

    reason = _rule_based_risk(payload)
    fallback_ir = semantic_fallback_ir(payload, include_do=True)

    # Lightweight contradiction check against current graph + candidate fallback IR.
    contradiction = False
    contradiction_msg = ""
    try:
        graph = GraphStore().load_graph()
        hist = []
        for u, v, attrs in graph.edges(data=True):
            op = (attrs or {}).get("relation") or (attrs or {}).get("label")
            if op:
                hist.append({"op": op, "args": [u, v]})
        is_ok, msg = LogicEngine().verify_consistency(hist + fallback_ir)
        contradiction = not is_ok
        contradiction_msg = msg if not is_ok else ""
    except Exception:
        contradiction = False
        contradiction_msg = ""

    if not reason and not contradiction:
        return None

    msg_parts = ["[INTERVENTION]"]
    if reason:
        msg_parts.append(f"Risk detected ({reason}).")
    if contradiction:
        msg_parts.append(f"Potential contradiction: {contradiction_msg}")
    msg_parts.append("Please provide explicit evidence/constraints before continuing.")
    out_msg = " ".join(msg_parts)

    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "reason": reason or "",
        "contradiction": contradiction,
        "contradiction_msg": contradiction_msg,
        "message": out_msg,
        "input_preview": payload[:240],
    }
    _write_event(row)
    _append_outbox(out_msg)
    return row

