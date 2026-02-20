import json
import re
from typing import Dict, List, Optional

from core.model_bridge import NanbeigeBridge
from core.rule_guard import (
    materialize_args_template,
    register_candidate_from_example,
    register_candidate_rule,
)


def _extract_json_object(raw: str) -> Optional[Dict]:
    text = str(raw or "").strip()
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _clean_payload(text: str) -> str:
    payload = (text or "").strip().lower()
    payload = re.sub(r"\s+", " ", payload)
    return payload


def _non_do_edges(ir_chain: List[dict]) -> List[dict]:
    out = []
    for row in ir_chain or []:
        if not isinstance(row, dict):
            continue
        op = str(row.get("op", "")).upper()
        args = row.get("args", [])
        if op == "DO":
            continue
        if not isinstance(args, list) or not args:
            continue
        out.append({"op": op, "args": [str(a) for a in args]})
    return out


def propose_candidates_from_text(text: str, ir_chain: List[dict], source: str = "listener_fallback"):
    payload = (text or "").strip()
    if not payload or not ir_chain:
        return

    # Avoid promoting DO-only chains into permanent rules.
    useful_ir = _non_do_edges(ir_chain)
    if not useful_ir:
        return

    # Exact-text candidate (safe baseline).
    register_candidate_from_example(payload, useful_ir, source=source)

    # Optional generalized candidates for common constructs.
    for i in useful_ir:
        if i.get("op") == "GOAL":
            register_candidate_rule(
                pattern=r"\b([a-z0-9_]+(?:mak|mek))\s+icin\s+([a-z0-9_]+)\b",
                op="GOAL",
                args=i.get("args", []),
                example_text=payload,
                source=f"{source}:goal_generic",
                args_template=["$2", "$1", "medium"],
                require_human=True,
            )
            break

    for i in useful_ir:
        if i.get("op") == "CAUSE":
            register_candidate_rule(
                pattern=r"\b([a-z0-9_]+(?:ip|up))\s+([a-z0-9_]+)\b",
                op="CAUSE",
                args=i.get("args", []),
                example_text=payload,
                source=f"{source}:cause_generic",
                args_template=["$1", "$2"],
                require_human=True,
            )
            break


def propose_llm_rule_candidate(
    text: str,
    validator,
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    source: str = "llm_rule_bootstrap",
) -> Optional[dict]:
    payload = (text or "").strip()
    if len(payload) < 8:
        return None

    bridge = NanbeigeBridge(model_name=model_name, api_key=api_key, provider=provider)
    if bridge._init_error:
        return None

    opcodes = ", ".join(sorted(validator.opcodes.keys()))
    prompt = (
        "You generate reusable rule candidates for a deterministic IR system.\n"
        "Return only one JSON object with this exact shape:\n"
        "{"
        "\"should_create_rule\": true,"
        "\"pattern\": \"regex pattern\","
        "\"op\": \"ISA_OPCODE\","
        "\"args_template\": [\"$1\", \"$2\"],"
        "\"args_preview\": [\"arg1\", \"arg2\"],"
        "\"clarification_question\": \"short question\","
        "\"confidence\": 0.0"
        "}\n"
        f"Allowed opcodes: {opcodes}\n"
        "Rules:\n"
        "- Prefer high-value relations; avoid DO unless absolutely necessary.\n"
        "- pattern must be reusable, not full exact sentence.\n"
        "- args_preview must match opcode argument count.\n"
        "- If ambiguous, set should_create_rule=false and provide clarification_question.\n"
        f"Input sentence: {payload}"
    )

    try:
        raw = bridge._chat(prompt, max_tokens=320)
    except Exception:
        return None

    obj = _extract_json_object(raw)
    if not obj:
        return None

    should_create = bool(obj.get("should_create_rule", True))
    question = str(obj.get("clarification_question", "")).strip()
    confidence = 0.0
    try:
        confidence = float(obj.get("confidence", 0.0))
    except Exception:
        confidence = 0.0

    if not should_create:
        return {
            "candidate": None,
            "ir": [],
            "question": question,
            "confidence": confidence,
            "auto_apply": False,
        }

    op = str(obj.get("op", "")).upper().strip()
    pattern = str(obj.get("pattern", "")).strip()
    args_template = obj.get("args_template", [])
    args_preview = obj.get("args_preview", [])
    if not op or not pattern:
        return None
    if not isinstance(args_template, list):
        args_template = []
    if not isinstance(args_preview, list):
        args_preview = []

    payload_norm = _clean_payload(payload)
    try:
        compiled = re.compile(pattern)
    except Exception:
        return None
    match = compiled.search(payload_norm)

    if not args_preview and args_template and match:
        args_preview = materialize_args_template(args_template, match)
    args_preview = [str(a) for a in args_preview]
    if not args_preview:
        return None

    ok, _ = validator.validate_instruction(op, args_preview)
    if not ok:
        return None

    cand = register_candidate_rule(
        pattern=pattern,
        op=op,
        args=args_preview,
        example_text=payload,
        source=source,
        args_template=args_template if args_template else None,
        confidence=confidence,
        clarification_question=question,
        require_human=True,
    )

    auto_apply = bool(op != "DO" and confidence >= 0.82 and not question)
    return {
        "candidate": cand,
        "ir": [{"op": op, "args": args_preview}],
        "question": question,
        "confidence": confidence,
        "auto_apply": auto_apply,
    }