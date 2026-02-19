import re
from typing import List

from core.rule_guard import register_candidate_from_example


def propose_candidates_from_text(text: str, ir_chain: List[dict], source: str = "listener_fallback"):
    payload = (text or "").strip()
    if not payload or not ir_chain:
        return

    # Exact-text rule candidate for low-risk promotion path.
    register_candidate_from_example(payload, ir_chain, source=source)

    # Optional lightweight generalization candidates.
    cleaned = re.sub(r"\s+", " ", payload.lower()).strip()

    # Generic intent pattern candidate (if GOAL is produced).
    if any(i.get("op") == "GOAL" for i in ir_chain):
        pattern = r"\b(\w+mak|\w+mek)\s+icin\s+\w+"
        for i in ir_chain:
            if i.get("op") == "GOAL":
                register_candidate_from_example(cleaned, [i], source=f"{source}:goal")
                break

    # Generic gerund cause candidate (if CAUSE is produced).
    if any(i.get("op") == "CAUSE" for i in ir_chain):
        pattern = r"\b\w+(ip|ıp|up|üp)\s+\w+"
        for i in ir_chain:
            if i.get("op") == "CAUSE":
                register_candidate_from_example(cleaned, [i], source=f"{source}:cause")
                break
