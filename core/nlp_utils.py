import importlib
import io
import re
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from typing import Iterable, List, Sequence, Tuple


def _ensure_zeyrek():
    try:
        return importlib.import_module("zeyrek")
    except Exception:
        pass

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "zeyrek"])
        return importlib.import_module("zeyrek")
    except Exception as exc:
        raise RuntimeError(
            "zeyrek zorunludur. Otomatik kurulum basarisiz oldu. "
            "Lutfen 'pip install zeyrek' ile kurulumu tamamlayin."
        ) from exc


def _ensure_nltk_resource(name: str):
    import nltk

    try:
        nltk.data.find(name)
        return
    except LookupError:
        pass

    parts = [p for p in name.split("/") if p]
    download_key = parts[1] if len(parts) > 1 else parts[0]
    nltk.download(download_key, quiet=True)


_ZEYREK = _ensure_zeyrek()
_ensure_nltk_resource("tokenizers/punkt_tab/turkish/")


def _build_lemmatizer():
    try:
        return _ZEYREK.MorphAnalyzer()
    except Exception as exc:
        raise RuntimeError("zeyrek MorphAnalyzer baslatilamadi.") from exc


_LEMMATIZER = _build_lemmatizer()


DISALLOWED_POS = {"Conj", "Det", "Postp", "Part", "Pron"}
LOGICAL_TOKEN_MAP = {
    "ve": "AND",
    "veya": "OR",
    "degil": "NOT",
    "değil": "NOT",
}
CAUSAL_TOKEN_TO_OPCODE = {
    "icin": "GOAL",
    "için": "GOAL",
    "cunku": "CAUSE",
    "çünkü": "CAUSE",
    "dolayisiyla": "CAUSE",
    "dolayısıyla": "CAUSE",
}
CASE_MORPHEME_TO_ATTR = {
    "Abl": "ayrilma",   # -den/-dan/-ten/-tan
    "Dat": "yonelme",   # -e/-a
    "Loc": "bulunma",   # -de/-da
    "Acc": "belirtme",  # -(y)i
    "Gen": "tamlayan",  # -in/-in
}
VERB_SUFFIX_HINTS = (
    "mak", "mek", "yor", "yoru", "yorum", "yoruz", "yorsun", "yorsunuz",
    "acak", "ecek", "ti", "tı", "tu", "tü", "di", "dı", "du", "dü",
    "acak", "ecek", "irim", "arım", "erim", "urum", "ürüm",
)
GERUND_SUFFIXES = ("ip", "ıp", "up", "üp")
COMMON_SUBJECT_STOPWORDS = {"bu", "su", "şu", "o", "bir", "ve", "ile", "icin", "için"}


def clean_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^\w\sçğıöşü]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t for t in text.split(" ") if t]


def _analyze_silent(token: str):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return _LEMMATIZER.analyze(token)


def _lemmatize_silent(token: str):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return _LEMMATIZER.lemmatize(token)


def _first_parse_obj(token: str):
    try:
        analyses = _analyze_silent(token)
    except Exception:
        return None
    if not analyses:
        return None

    first_group = analyses[0]
    if isinstance(first_group, list) and first_group:
        return first_group[0]
    return None


def _lemma_with_zeyrek(token: str):
    parse = _first_parse_obj(token)
    if parse is not None:
        lemma = getattr(parse, "lemma", None)
        if isinstance(lemma, str) and lemma.strip():
            return lemma.strip().lower()

    try:
        analyses = _lemmatize_silent(token)
    except Exception:
        return None

    if not analyses:
        return None

    for item in analyses:
        if isinstance(item, tuple) and item:
            first = item[0]
            if isinstance(first, str) and first.strip():
                return first.strip().lower()
            if isinstance(first, (list, tuple)) and first:
                inner = first[0]
                if isinstance(inner, str) and inner.strip():
                    return inner.strip().lower()
        elif isinstance(item, str) and item.strip():
            return item.strip().lower()
    return None


def _safe_lemma(token: str) -> str:
    lemma = _lemma_with_zeyrek(token)
    if not lemma:
        return token
    if lemma.lower() in {"unk", "unknown", "bilinmeyen"}:
        return token
    return lemma


def _pos_tags(token: str) -> List[str]:
    tags = []
    try:
        analyses = _analyze_silent(token)
    except Exception:
        return tags
    for group in analyses or []:
        if isinstance(group, list):
            for parse in group:
                pos = getattr(parse, "pos", None)
                if isinstance(pos, str) and pos:
                    tags.append(pos)
    return tags


def _morphemes(token: str) -> List[str]:
    out = []
    try:
        analyses = _analyze_silent(token)
    except Exception:
        return out
    for group in analyses or []:
        if isinstance(group, list):
            for parse in group:
                ms = getattr(parse, "morphemes", None)
                if isinstance(ms, list):
                    out.extend([str(m) for m in ms])
    return out


def lemmatize_tokens(tokens: Sequence[str]) -> List[str]:
    out = []
    for t in tokens:
        if not t:
            continue
        lemma = _safe_lemma(t)
        out.append(lemma if lemma else t)
    return out


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]


def levenshtein_similarity(a: str, b: str) -> float:
    a = a or ""
    b = b or ""
    if not a and not b:
        return 1.0
    dist = levenshtein_distance(a, b)
    base = 1.0 - (dist / max(len(a), len(b), 1))
    if dist == 1 and abs(len(a) - len(b)) <= 1:
        return min(0.95, base + 0.20)
    return base


def build_memory_vocabulary(memory_terms: Iterable[str]) -> List[str]:
    vocab = set()
    for term in memory_terms or []:
        cleaned = clean_text(str(term))
        if not cleaned:
            continue
        parts = tokenize(cleaned)
        lemmas = lemmatize_tokens(parts)
        for p in lemmas:
            vocab.add(p)
    return sorted(vocab)


def fuzzy_match_tokens(
    tokens: Sequence[str],
    memory_terms: Iterable[str],
    threshold: float = 0.90,
) -> Tuple[List[str], List[dict]]:
    vocab = build_memory_vocabulary(memory_terms)
    if not vocab:
        return list(tokens), []

    corrected = []
    corrections = []
    for tok in tokens:
        best = tok
        best_score = 0.0
        for cand in vocab:
            score = levenshtein_similarity(tok, cand)
            if score > best_score:
                best = cand
                best_score = score
        if best != tok and best_score >= threshold:
            corrected.append(best)
            corrections.append({"from": tok, "to": best, "confidence": round(best_score, 3)})
        else:
            corrected.append(tok)
    return corrected, corrections


def normalize_and_match(text: str, memory_terms: Iterable[str] = None, threshold: float = 0.90) -> dict:
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    lemmas = lemmatize_tokens(tokens)
    corrected, corrections = fuzzy_match_tokens(lemmas, memory_terms or [], threshold=threshold)
    return {
        "cleaned": cleaned,
        "tokens": tokens,
        "lemmas": lemmas,
        "corrected_tokens": corrected,
        "corrected_text": " ".join(corrected).strip(),
        "corrections": corrections,
        "lemmatizer": "zeyrek",
    }


def _normalize_arg_text(arg: str) -> str:
    cleaned = clean_text(arg)
    if not cleaned:
        return ""
    toks = tokenize(cleaned)
    if not toks:
        return ""
    lemmas = lemmatize_tokens(toks)
    return "_".join(lemmas)


def is_pos_blocked(token: str, known_entities: Iterable[str] = None) -> Tuple[bool, str]:
    t = clean_text(token)
    if not t:
        return True, "Empty"

    known = {clean_text(k) for k in (known_entities or []) if str(k).strip()}
    tags = _pos_tags(t)
    if not tags:
        return False, "Unknown"

    if "Pron" in tags and t in known:
        return False, "PronAllowedByEntity"

    for tag in tags:
        if tag in DISALLOWED_POS:
            return True, tag
    return False, tags[0]


def _suffix_attrs_for_token(token: str, lemma: str) -> List[dict]:
    attrs = []
    morphs = set(_morphemes(token))
    for morph, value in CASE_MORPHEME_TO_ATTR.items():
        if morph in morphs:
            attrs.append({"op": "ATTR", "args": [lemma, "hal", value]})
    return attrs


def _is_verb_token(token: str) -> bool:
    tags = _pos_tags(token)
    if "Verb" in tags:
        return True
    return any(token.endswith(sfx) for sfx in VERB_SUFFIX_HINTS)


def _first_verb(tokens: Sequence[str]) -> str:
    for tok in tokens:
        if _is_verb_token(tok):
            return _safe_lemma(tok) or tok
    return ""


def _intent_goal_ir(raw_text: str):
    cleaned = clean_text(raw_text)
    toks = tokenize(cleaned)
    if not toks:
        return None
    m = re.search(r"(\w+?(?:mak|mek))\s+icin\s+(\w+)", cleaned)
    if m:
        target_raw = m.group(1)
        action_raw = m.group(2)
        target = _safe_lemma(target_raw[:-3]) or target_raw[:-3]
        action = _safe_lemma(action_raw) or action_raw
        if action and target:
            return {"op": "GOAL", "args": [action, target, "medium"]}
    try:
        idx = toks.index("icin")
    except ValueError:
        return None
    if idx <= 0:
        return None

    # Target candidate from the infinitive clause before 'icin' (e.g. kir-mak).
    target = ""
    for prev in reversed(toks[:idx]):
        if prev.endswith("mak") or prev.endswith("mek"):
            root = prev[:-3]
            target = _safe_lemma(root) or root
            break
    if not target:
        target = _safe_lemma(toks[idx - 1]) or toks[idx - 1]

    action = _first_verb(toks[idx + 1 :])
    if not action:
        return None

    return {"op": "GOAL", "args": [action, target, "medium"]}


def _gerund_cause_ir(raw_text: str):
    cleaned = clean_text(raw_text)
    toks = tokenize(cleaned)
    if not toks:
        return None
    m = re.search(r"(\w+?)(?:ip|ıp|up|üp)\s+(\w+)", cleaned)
    if m:
        first_action = _safe_lemma(m.group(1)) or m.group(1)
        second_action = _safe_lemma(m.group(2)) or m.group(2)
        if first_action and second_action:
            return {"op": "CAUSE", "args": [first_action, second_action]}

    gerund_idx = -1
    first_action = ""
    for i, tok in enumerate(toks):
        for suf in GERUND_SUFFIXES:
            if tok.endswith(suf) and len(tok) > len(suf):
                base = tok[: -len(suf)]
                first_action = _safe_lemma(base) or base
                gerund_idx = i
                break
        if gerund_idx != -1:
            break
    if gerund_idx == -1:
        return None

    second_action = _first_verb(toks[gerund_idx + 1 :])
    if not first_action or not second_action:
        return None

    return {"op": "CAUSE", "args": [first_action, second_action]}


def _dynamic_do_ir(raw_text: str):
    cleaned = clean_text(raw_text)
    toks = tokenize(cleaned)
    if not toks:
        return None

    action = _first_verb(toks)
    if not action:
        action = _safe_lemma(toks[-1]) or toks[-1]

    subject = ""
    for tok in toks:
        if tok in COMMON_SUBJECT_STOPWORDS:
            continue
        if tok == action:
            continue
        subject = _safe_lemma(tok) or tok
        break
    if not subject:
        subject = _safe_lemma(toks[0]) or toks[0] or "agent"

    return {"op": "DO", "args": [subject or "agent", action or "eylem"]}


def grammar_filter_ir(
    ir_chain: List[dict],
    known_entities: Iterable[str] = None,
) -> Tuple[List[dict], List[str], List[dict]]:
    filtered = []
    logs = []
    generated_attrs = []
    known_ops = {
        "DEF_ENTITY", "DEF_CONCEPT", "ISA", "EQUIV", "ATTR", "KNOW", "BELIEVE", "DOUBT",
        "WONDER", "ASSUME", "WANT", "AVOID", "GOAL", "INTEND", "EVAL", "CAUSE", "PREVENT",
        "IMPLY", "OPPOSE", "TRIGGER", "MUST", "MAY", "FORBID", "CAN", "BEFORE", "WHILE",
        "START", "END", "REFLECT", "CORRECT", "ANALOGY", "DO",
    }
    for instr in ir_chain or []:
        op = str(instr.get("op", ""))
        args = instr.get("args", [])
        raw_text = " ".join([a for a in args if isinstance(a, str)]).strip()
        if not isinstance(args, list):
            logs.append(f"[GRAMMAR_FILTER] Gecersiz args yapisi: {instr}")
            continue

        blocked = False
        new_args = []
        requested_opcode = None
        for arg in args:
            if not isinstance(arg, str):
                new_args.append(arg)
                continue

            cleaned = clean_text(arg)
            toks = tokenize(cleaned)
            if not toks:
                blocked = True
                logs.append(f"[GRAMMAR_FILTER] Bos arg nedeniyle IR reddedildi: {instr}")
                break

            out_toks = []
            for tok in toks:
                if tok in LOGICAL_TOKEN_MAP:
                    out_toks.append(f"__{LOGICAL_TOKEN_MAP[tok]}__")
                    continue

                if tok in CAUSAL_TOKEN_TO_OPCODE:
                    requested_opcode = CAUSAL_TOKEN_TO_OPCODE[tok]
                    continue

                is_blocked, reason = is_pos_blocked(tok, known_entities=known_entities)
                if is_blocked:
                    blocked = True
                    logs.append(f"[GRAMMAR_FILTER] POS={reason} nedeniyle IR reddedildi: {instr}")
                    break

                lemma = _safe_lemma(tok) or tok
                out_toks.append(lemma)
                generated_attrs.extend(_suffix_attrs_for_token(tok, lemma))

            if blocked:
                break

            if out_toks:
                new_args.append("_".join(out_toks))

        if blocked:
            # Even if POS filter blocks a token, try intent/result fallback rules to avoid data loss.
            goal_ir = _intent_goal_ir(raw_text)
            cause_ir = _gerund_cause_ir(raw_text)
            do_ir = _dynamic_do_ir(raw_text)
            for cand, label in (
                (goal_ir, "Intent"),
                (cause_ir, "Zincirleme"),
                (do_ir, "Dinamik DO"),
            ):
                if cand and cand not in filtered:
                    filtered.append(cand)
                    logs.append(f"[GRAMMAR_FILTER] {label} fallback uygulandi (POS blok sonrasi): {cand}")
            continue

        final_op = op
        # Dynamic opcode mapping: unknown ops are coerced into DO(subject, action).
        if op not in known_ops:
            do_ir = _dynamic_do_ir(raw_text)
            filtered.append(do_ir if do_ir else {"op": "DO", "args": ["agent", "eylem"]})
            logs.append(f"[GRAMMAR_FILTER] Bilinmeyen opcode '{op}' -> DO fallback uygulandi.")
            continue

        if requested_opcode == "GOAL":
            if len(new_args) >= 2:
                final_op = "GOAL"
                if len(new_args) == 2:
                    new_args.append("medium")
                    logs.append(f"[GRAMMAR_FILTER] 'icin' algilandi, GOAL icin varsayilan oncelik eklendi: {instr}")
        elif requested_opcode == "CAUSE":
            final_op = "CAUSE"

        out_instr = {"op": final_op, "args": new_args}
        if "confidence" in instr:
            out_instr["confidence"] = instr["confidence"]
        if "provenance" in instr:
            out_instr["provenance"] = instr["provenance"]
        if "source" in instr:
            out_instr["source"] = instr["source"]
        filtered.append(out_instr)

        goal_ir = _intent_goal_ir(raw_text)
        if goal_ir and goal_ir not in filtered:
            filtered.append(goal_ir)
            logs.append(f"[GRAMMAR_FILTER] Intent kurali uygulandi: {goal_ir}")

        cause_ir = _gerund_cause_ir(raw_text)
        if cause_ir and cause_ir not in filtered:
            filtered.append(cause_ir)
            logs.append(f"[GRAMMAR_FILTER] Zincirleme neden-sonuc kurali uygulandi: {cause_ir}")

        # Dynamic opcode fallback: no reliable opcode signal -> keep data as temporary DO.
        if op not in known_ops and requested_opcode is None:
            do_ir = _dynamic_do_ir(raw_text)
            if do_ir and do_ir not in filtered:
                filtered.append(do_ir)
                logs.append(f"[GRAMMAR_FILTER] Dinamik opcode fallback (DO) uygulandi: {do_ir}")

    return filtered, logs, generated_attrs


def rebalance_relations(ir_chain: List[dict], source_text: str) -> Tuple[List[dict], List[str]]:
    """
    Reduce CAUSE overfitting by adding non-causal ISA opcodes inferred from explicit lexical cues.
    """
    out = list(ir_chain or [])
    logs = []
    raw = clean_text(source_text or "")
    if not raw:
        return out, logs

    cause_count = sum(1 for i in out if i.get("op") == "CAUSE")
    total = max(1, len(out))
    cause_ratio = cause_count / total

    # Only activate when CAUSE is dominant or text is long/narrative.
    if cause_ratio < 0.45 and len(raw.split()) < 20:
        return out, logs

    def add_once(instr, reason):
        if instr not in out:
            out.append(instr)
            logs.append(f"[REL_REBALANCE] {reason}: {instr}")

    # ISA: X bir Y'dir
    for m in re.finditer(r"\b([\wçğıöşü]+)\s+bir\s+([\wçğıöşü]+)(?:dir|dır|dur|dür|tir|tır|tur|tür)\b", raw):
        add_once({"op": "ISA", "args": [_safe_lemma(m.group(1)), _safe_lemma(m.group(2))]}, "ISA kalibi")

    # ATTR: X Y'dir (bir ...dir disi)
    disallowed_subjects = {"ama", "fakat", "ancak", "ve", "veya", "ile", "bir"}
    for m in re.finditer(r"\b([\wçğıöşü]+)\s+([\wçğıöşü]+)(?:dir|dır|dur|dür|tir|tır|tur|tür)\b", raw):
        if m.group(2) != "bir" and m.group(1) not in disallowed_subjects:
            add_once({"op": "ATTR", "args": [_safe_lemma(m.group(1)), "ozellik", _safe_lemma(m.group(2))]}, "ATTR kalibi")

    # WANT
    m = re.search(r"\b([\wçğıöşü]+)\s+[\wçğıöşü\s]{0,25}ist(?:iyor|edi|er)\b", raw)
    if m:
        subj = _safe_lemma(m.group(1))
        add_once({"op": "WANT", "args": [subj, "hedef"]}, "WANT kalibi")

    # BELIEVE
    m = re.search(r"\b([\wçğıöşü]+)\s+[\wçğıöşü\s]{0,20}inan(?:iyor|ıyordu|di|dı|ir)\b", raw)
    if m:
        subj = _safe_lemma(m.group(1))
        add_once({"op": "BELIEVE", "args": [subj, "oneri", "0.7"]}, "BELIEVE kalibi")

    # GOAL: ... icin ...
    if " icin " in f" {raw} " or " için " in source_text.lower():
        toks = tokenize(raw)
        if len(toks) >= 2:
            add_once({"op": "GOAL", "args": [_safe_lemma(toks[0]), _safe_lemma(toks[-1]), "medium"]}, "GOAL baglaci")

    # BEFORE: once ... sonra ...
    m = re.search(r"\bonce\s+([\wçğıöşü]+)\s+.*\bsonra\s+([\wçğıöşü]+)\b", raw)
    if m:
        add_once({"op": "BEFORE", "args": [_safe_lemma(m.group(1)), _safe_lemma(m.group(2))]}, "BEFORE kalibi")

    # OPPOSE
    m = re.search(r"\b([\wçğıöşü]+)\s+(ama|fakat|ancak)\s+([\wçğıöşü]+)\b", raw)
    if m:
        left = _safe_lemma(m.group(1))
        right = _safe_lemma(m.group(3))
        if left not in disallowed_subjects and right not in disallowed_subjects:
            add_once({"op": "OPPOSE", "args": [left, right]}, "OPPOSE baglaci")

    # EVAL
    m = re.search(r"\b([\wçğıöşü]+)\s+(iyi|kotu|guzel|cirkin)\b", raw)
    if m:
        score = "1.0" if m.group(2) in {"iyi", "guzel"} else "-1.0"
        add_once({"op": "EVAL", "args": [_safe_lemma(m.group(1)), score]}, "EVAL kalibi")

    return out, logs

