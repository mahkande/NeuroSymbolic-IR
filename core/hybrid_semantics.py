import re
from typing import Callable, Dict, List, Optional, Tuple


def _to_token(text: str) -> str:
    token = (text or "").strip().lower()
    token = re.sub(r"[^\w\sçğıöşüÇĞİÖŞÜ-]", " ", token)
    token = re.sub(r"\s+", " ", token).strip()
    return token.replace(" ", "_")


def _dedupe_key(instr: dict) -> Tuple[str, Tuple[str, ...]]:
    op = str(instr.get("op", "")).upper()
    args = tuple(str(a) for a in instr.get("args", []))
    return op, args


class RuleBasedExtractorV2:
    """
    Hybrid semantic extraction candidates with explicit lexical pattern coverage.
    """

    def extract(self, text: str) -> List[dict]:
        raw = (text or "").strip().lower()
        if not raw:
            return []

        out: List[dict] = []

        def add(op: str, args: List[str], score: float, signal: str):
            if not args or any(not a for a in args):
                return
            cand = {
                "op": op,
                "args": [str(a) for a in args],
                "score": round(float(score), 3),
                "source": "rule_v2",
                "signal": signal,
            }
            out.append(cand)

        for m in re.finditer(r"\b([\wçğıöşü]+)\s+bir\s+([\wçğıöşü]+)(?:dir|dır|dur|dür|tir|tır|tur|tür)\b", raw):
            add("ISA", [_to_token(m.group(1)), _to_token(m.group(2))], 0.93, "isa_bir")

        for m in re.finditer(r"\b([\wçğıöşü]+)\s+([\wçğıöşü]+)(?:dir|dır|dur|dür|tir|tır|tur|tür)\b", raw):
            if m.group(2) != "bir":
                add("ATTR", [_to_token(m.group(1)), "ozellik", _to_token(m.group(2))], 0.73, "attr_dir")

        for m in re.finditer(r"\b([\wçğıöşü]+)\s+(?:cunku|çünkü|dolayisiyla|dolayısıyla|bu_yuzden)\s+([\wçğıöşü]+)\b", raw):
            add("CAUSE", [_to_token(m.group(2)), _to_token(m.group(1))], 0.78, "cause_connector")

        m = re.search(r"(.+?)\s+(?:olursa|ise)\s+(.+?)\s+olur\b", raw)
        if m:
            add("CAUSE", [_to_token(m.group(1)), _to_token(m.group(2))], 0.86, "cause_conditional")

        m = re.search(r"\b([\wçğıöşü]+)\s+[\wçğıöşü\s]{0,25}ist(?:iyor|edi|er)\b", raw)
        if m:
            add("WANT", [_to_token(m.group(1)), "hedef"], 0.71, "want_intent")

        m = re.search(r"\b([\wçğıöşü]+)\s+[\wçğıöşü\s]{0,20}inan(?:iyor|ıyordu|di|dı|ir)\b", raw)
        if m:
            add("BELIEVE", [_to_token(m.group(1)), "oneri", "0.7"], 0.69, "believe_signal")

        m = re.search(r"\b([\wçğıöşü]+)\s+[\wçğıöşü\s]{0,20}biliyor\b", raw)
        if m:
            add("KNOW", [_to_token(m.group(1)), "bilgi"], 0.67, "know_signal")

        m = re.search(r"\b([\wçğıöşü]+)\s+icin\s+([\wçğıöşü]+)\b", raw)
        if m:
            add("GOAL", [_to_token(m.group(2)), _to_token(m.group(1)), "medium"], 0.72, "goal_icin")

        m = re.search(r"\bonce\s+([\wçğıöşü]+)\s+.*\bsonra\s+([\wçğıöşü]+)\b", raw)
        if m:
            add("BEFORE", [_to_token(m.group(1)), _to_token(m.group(2))], 0.82, "before_sequence")

        m = re.search(r"\b([\wçğıöşü]+)\s+(ama|fakat|ancak)\s+([\wçğıöşü]+)\b", raw)
        if m:
            add("OPPOSE", [_to_token(m.group(1)), _to_token(m.group(3))], 0.75, "oppose_connector")

        m = re.search(r"\b([\wçğıöşü]+)\s+(?:engeller|onler|önler)\s+([\wçğıöşü]+)\b", raw)
        if m:
            add("PREVENT", [_to_token(m.group(1)), _to_token(m.group(2))], 0.77, "prevent_signal")

        m = re.search(r"\b([\wçğıöşü]+)\s+(?:iyi|kotu|guzel|cirkin)\b", raw)
        if m:
            score = "1.0" if m.group(0).split()[-1] in {"iyi", "guzel"} else "-1.0"
            add("EVAL", [_to_token(m.group(1)), score], 0.74, "eval_adj")

        unique: Dict[Tuple[str, Tuple[str, ...]], dict] = {}
        for cand in out:
            key = _dedupe_key(cand)
            prev = unique.get(key)
            if prev is None or cand["score"] > prev["score"]:
                unique[key] = cand
        return list(unique.values())

    def extract_ir(self, text: str) -> List[dict]:
        return [{"op": c["op"], "args": c["args"]} for c in self.extract(text)]


class RelationClassifier:
    """
    Lightweight multi-label opcode scorer from lexical features.
    """

    _FEATURES: Dict[str, Dict[str, float]] = {
        "CAUSE": {"cunku": 1.0, "çünkü": 1.0, "dolayisiyla": 0.9, "dolayısıyla": 0.9, "olursa": 0.8, "ise": 0.5},
        "GOAL": {"icin": 1.0, "için": 1.0, "amac": 0.9, "amaç": 0.9, "hedef": 0.8},
        "WANT": {"istiyor": 1.0, "ister": 0.8, "istedi": 0.8, "arzu": 0.8},
        "BELIEVE": {"inaniyor": 1.0, "inanıyor": 1.0, "sanir": 0.7, "sanır": 0.7},
        "KNOW": {"biliyor": 1.0, "bilir": 0.8},
        "ISA": {"bir": 0.7, "tur": 0.5, "tür": 0.5},
        "OPPOSE": {"ama": 1.0, "fakat": 1.0, "ancak": 0.9},
        "BEFORE": {"once": 1.0, "önce": 1.0, "sonra": 0.9},
        "EVAL": {"iyi": 1.0, "kotu": 1.0, "kötü": 1.0, "guzel": 1.0, "güzel": 1.0, "cirkin": 1.0, "çirkin": 1.0},
        "PREVENT": {"engeller": 1.0, "onler": 1.0, "önler": 1.0},
        "DO": {"yap": 0.5, "et": 0.5},
    }

    def predict(self, text: str) -> Dict[str, float]:
        raw = (text or "").lower()
        toks = re.findall(r"[\wçğıöşü]+", raw)
        if not toks:
            return {}

        scores: Dict[str, float] = {op: 0.0 for op in self._FEATURES}
        for tok in toks:
            for op, fmap in self._FEATURES.items():
                if tok in fmap:
                    scores[op] += fmap[tok]

        if " bir " in f" {raw} " and re.search(r"\b(?:dir|dır|dur|dür|tir|tır|tur|tür)\b", raw):
            scores["ISA"] += 1.2
        if " icin " in f" {raw} " or " için " in raw:
            scores["GOAL"] += 1.1
        if re.search(r"\b(?:olursa|ise)\b", raw):
            scores["CAUSE"] += 0.7

        max_score = max(scores.values()) if scores else 0.0
        if max_score <= 0:
            return {"DO": 0.55}

        probs: Dict[str, float] = {}
        for op, score in scores.items():
            if score <= 0:
                continue
            probs[op] = round(min(0.97, 0.20 + (score / max_score) * 0.77), 3)
        return probs

    def top_labels(self, text: str, threshold: float = 0.45) -> List[Tuple[str, float]]:
        probs = self.predict(text)
        pairs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return [p for p in pairs if p[1] >= threshold]


def fuse_relations(
    text: str,
    llm_ir: List[dict],
    rule_candidates: List[dict],
    classifier_probs: Dict[str, float],
    arbitration_cb: Optional[Callable[[str], List[dict]]] = None,
) -> Tuple[List[dict], List[str]]:
    logs: List[str] = []
    fused: List[dict] = []
    dedupe: Dict[Tuple[str, Tuple[str, ...]], dict] = {}

    rule_index: Dict[Tuple[str, Tuple[str, ...]], float] = {}
    for cand in rule_candidates:
        key = _dedupe_key(cand)
        rule_index[key] = max(rule_index.get(key, 0.0), float(cand.get("score", 0.0)))

    uncertain_keys: List[Tuple[str, Tuple[str, ...]]] = []

    for instr in llm_ir or []:
        op = str(instr.get("op", "")).upper()
        args = [str(a) for a in instr.get("args", [])]
        base = float(instr.get("confidence", 0.66))
        cls = float(classifier_probs.get(op, 0.0))
        rs = float(rule_index.get((op, tuple(args)), 0.0))
        score = round((base * 0.55) + (cls * 0.30) + (rs * 0.15), 3)
        merged = {"op": op, "args": args, "confidence": score, "provenance": "llm+hybrid"}
        key = _dedupe_key(merged)
        dedupe[key] = merged
        if 0.45 <= score < 0.62:
            uncertain_keys.append(key)

    for cand in rule_candidates:
        op = str(cand.get("op", "")).upper()
        args = [str(a) for a in cand.get("args", [])]
        rule_score = float(cand.get("score", 0.0))
        cls = float(classifier_probs.get(op, 0.0))
        if rule_score >= 0.74 and cls >= 0.45:
            key = (op, tuple(args))
            if key not in dedupe:
                merged = {
                    "op": op,
                    "args": args,
                    "confidence": round((rule_score * 0.65) + (cls * 0.35), 3),
                    "provenance": "rule_v2+classifier",
                }
                dedupe[key] = merged
                logs.append(f"[HYBRID] Rule+Classifier edge eklendi: {op} {args}")

    if uncertain_keys and arbitration_cb:
        judged = arbitration_cb(text) or []
        if judged:
            judged_map = {_dedupe_key(i): i for i in judged if isinstance(i, dict)}
            replaced = 0
            for key in uncertain_keys:
                if key in judged_map:
                    cand = judged_map[key]
                    op = str(cand.get("op", "")).upper()
                    args = [str(a) for a in cand.get("args", [])]
                    dedupe[key] = {
                        "op": op,
                        "args": args,
                        "confidence": 0.72,
                        "provenance": "llm_arbitration",
                    }
                    replaced += 1
            if replaced:
                logs.append(f"[HYBRID] LLM arbitration ile {replaced} belirsiz bag guncellendi.")

    fused.extend(dedupe.values())
    fused.sort(key=lambda i: float(i.get("confidence", 0.0)), reverse=True)
    return fused, logs
