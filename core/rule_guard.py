import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


RULES_PATH = Path("rules/learned_fallback_rules.json")
CANDIDATES_PATH = Path("memory/rule_candidates.json")
UNKNOWN_PATH = Path("memory/rule_unknown_buffer.jsonl")
DECISIONS_PATH = Path("memory/rule_decisions.jsonl")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return default


def _save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _fingerprint(pattern: str, op: str, args: List[str]) -> str:
    return f"{pattern}|{op}|{'|'.join(args)}"


def _ensure_stores():
    if not RULES_PATH.exists():
        _save_json(RULES_PATH, {"version": 1, "rules": []})
    if not CANDIDATES_PATH.exists():
        _save_json(CANDIDATES_PATH, {"version": 1, "candidates": []})


def get_rules() -> List[dict]:
    _ensure_stores()
    data = _load_json(RULES_PATH, {"rules": []})
    return data.get("rules", [])


def get_active_rules() -> List[dict]:
    return [r for r in get_rules() if r.get("status") == "active"]


def get_review_queue(limit: int = 50) -> List[dict]:
    _ensure_stores()
    data = _load_json(CANDIDATES_PATH, {"candidates": []})
    cands = [c for c in data.get("candidates", []) if c.get("status", "pending") == "pending"]
    cands.sort(key=lambda x: (x.get("hits", 0), x.get("last_seen", "")), reverse=True)
    return cands[:limit]


def get_rule_stats() -> dict:
    rules = get_rules()
    queue = get_review_queue(limit=10000)
    return {
        "total_rules": len(rules),
        "active_rules": len([r for r in rules if r.get("status") == "active"]),
        "disabled_rules": len([r for r in rules if r.get("status") == "disabled"]),
        "pending_candidates": len(queue),
    }


def register_unknown_pattern(text: str, source: str = "", reason: str = ""):
    payload = (text or "").strip()
    if not payload:
        return
    _append_jsonl(
        UNKNOWN_PATH,
        {"ts": _now(), "text": payload, "source": source, "reason": reason},
    )


def register_candidate_rule(pattern: str, op: str, args: List[str], example_text: str, source: str = "auto") -> dict:
    _ensure_stores()
    data = _load_json(CANDIDATES_PATH, {"version": 1, "candidates": []})
    candidates = data.get("candidates", [])
    fp = _fingerprint(pattern, op, args)

    for c in candidates:
        if c.get("fingerprint") == fp and c.get("status") != "rejected":
            c["hits"] = int(c.get("hits", 0)) + 1
            c["last_seen"] = _now()
            c["example_text"] = example_text or c.get("example_text", "")
            _save_json(CANDIDATES_PATH, data)
            return c

    cid = f"cand_{abs(hash(fp))}_{len(candidates)+1}"
    cand = {
        "id": cid,
        "fingerprint": fp,
        "pattern": pattern,
        "op": op,
        "args": args,
        "status": "pending",
        "hits": 1,
        "source": source,
        "example_text": example_text,
        "created_at": _now(),
        "last_seen": _now(),
    }
    candidates.append(cand)
    data["candidates"] = candidates
    _save_json(CANDIDATES_PATH, data)
    return cand


def register_candidate_from_example(text: str, ir_chain: List[dict], source: str = "auto"):
    payload = (text or "").strip()
    if not payload:
        return
    norm = re.sub(r"\s+", " ", payload.lower()).strip()
    pattern = r"^" + re.escape(norm) + r"$"
    for instr in ir_chain or []:
        op = instr.get("op")
        args = instr.get("args", [])
        if isinstance(op, str) and isinstance(args, list) and args:
            register_candidate_rule(pattern, op, [str(a) for a in args], payload, source=source)


def _is_rule_safe(rule: dict, validator) -> bool:
    try:
        re.compile(rule.get("pattern", ""))
    except Exception:
        return False
    op = rule.get("op")
    args = rule.get("args", [])
    if not isinstance(args, list):
        return False
    ok, _ = validator.validate_instruction(op, args)
    return bool(ok)


def _activate_candidate(cand: dict):
    rules_data = _load_json(RULES_PATH, {"version": 1, "rules": []})
    rules = rules_data.get("rules", [])
    if not any(r.get("fingerprint") == cand.get("fingerprint") for r in rules):
        rules.append(
            {
                "id": f"rule_{cand.get('id')}",
                "fingerprint": cand.get("fingerprint"),
                "pattern": cand.get("pattern"),
                "op": cand.get("op"),
                "args": cand.get("args"),
                "status": "active",
                "priority": 100,
                "created_at": _now(),
                "source": cand.get("source", "auto"),
            }
        )
        rules_data["rules"] = rules
        _save_json(RULES_PATH, rules_data)


def approve_candidate(candidate_id: str, validator) -> bool:
    _ensure_stores()
    data = _load_json(CANDIDATES_PATH, {"candidates": []})
    for c in data.get("candidates", []):
        if c.get("id") == candidate_id:
            if not _is_rule_safe(c, validator):
                c["status"] = "rejected"
                _save_json(CANDIDATES_PATH, data)
                _append_jsonl(DECISIONS_PATH, {"ts": _now(), "candidate_id": candidate_id, "decision": "rejected", "reason": "safety_gate_failed"})
                return False
            c["status"] = "approved"
            c["approved_at"] = _now()
            _save_json(CANDIDATES_PATH, data)
            _activate_candidate(c)
            _append_jsonl(DECISIONS_PATH, {"ts": _now(), "candidate_id": candidate_id, "decision": "approved"})
            return True
    return False


def reject_candidate(candidate_id: str, reason: str = "manual_reject") -> bool:
    _ensure_stores()
    data = _load_json(CANDIDATES_PATH, {"candidates": []})
    for c in data.get("candidates", []):
        if c.get("id") == candidate_id:
            c["status"] = "rejected"
            c["rejected_at"] = _now()
            _save_json(CANDIDATES_PATH, data)
            _append_jsonl(DECISIONS_PATH, {"ts": _now(), "candidate_id": candidate_id, "decision": "rejected", "reason": reason})
            return True
    return False


def auto_review_candidates(validator, min_hits: int = 3) -> dict:
    queue = get_review_queue(limit=10000)
    approved = 0
    rejected = 0
    for c in queue:
        if int(c.get("hits", 0)) < int(min_hits):
            continue
        if approve_candidate(c.get("id"), validator):
            approved += 1
        else:
            rejected += 1
    return {"approved": approved, "rejected": rejected}


def apply_active_rules(text: str, validator) -> List[dict]:
    payload = (text or "").strip().lower()
    payload = re.sub(r"\s+", " ", payload)
    if not payload:
        return []

    out = []
    for rule in get_active_rules():
        pattern = rule.get("pattern", "")
        if not pattern:
            continue
        try:
            if not re.search(pattern, payload):
                continue
        except Exception:
            continue

        op = rule.get("op")
        args = rule.get("args", [])
        if not isinstance(args, list):
            continue
        ok, _ = validator.validate_instruction(op, args)
        if not ok:
            continue
        out.append({"op": op, "args": args})

    return out
