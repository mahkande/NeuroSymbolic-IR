import re
from typing import List, Tuple


def _tok(text: str) -> List[str]:
    return re.findall(r"[\w]+", str(text or "").lower())


def _instr_terms(instr: dict) -> List[str]:
    args = instr.get("args", []) if isinstance(instr, dict) else []
    terms = []
    for arg in args if isinstance(args, list) else []:
        terms.extend(_tok(arg))
    # Keep terms reasonably focused.
    return [t for t in terms if len(t) > 1][:24]


def _hit_text(hit: dict) -> str:
    if not isinstance(hit, dict):
        return ""
    text = str(hit.get("text", "") or "")
    meta = hit.get("meta", {}) or {}
    if isinstance(meta, dict):
        extra = " ".join(str(meta.get(k, "")) for k in ("u", "v", "relation", "node", "kind"))
        text = f"{text} {extra}".strip()
    return text


def _support_score(instr_terms: List[str], hit_terms: List[str]) -> float:
    if not instr_terms or not hit_terms:
        return 0.0
    hs = set(hit_terms)
    overlap = sum(1 for t in instr_terms if t in hs)
    return overlap / max(1, len(set(instr_terms)))


def build_proof_objects(ir_chain: List[dict], evidence_hits: List[dict], min_score: float = 0.34) -> Tuple[List[dict], List[dict]]:
    proofs = []
    enriched = []
    for idx, instr in enumerate(ir_chain or []):
        if not isinstance(instr, dict):
            continue
        terms = _instr_terms(instr)
        supports = []
        for eidx, hit in enumerate(evidence_hits or []):
            htxt = _hit_text(hit)
            hterms = _tok(htxt)
            score = _support_score(terms, hterms)
            if score < min_score:
                continue
            supports.append(
                {
                    "id": f"ev::{eidx}",
                    "kind": str(hit.get("kind", "")),
                    "score": round(float(score), 3),
                    "text": str(hit.get("text", ""))[:220],
                    "meta": hit.get("meta", {}) or {},
                }
            )

        instr_copy = dict(instr)
        instr_copy["evidence_ids"] = [s["id"] for s in supports]
        enriched.append(instr_copy)

        proofs.append(
            {
                "claim_id": f"claim::{idx}",
                "claim": {"op": instr_copy.get("op"), "args": instr_copy.get("args", [])},
                "supports": supports,
                "verdict": "supported" if supports else "unsupported",
            }
        )
    return proofs, enriched

