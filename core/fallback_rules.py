import re
import unicodedata
from typing import List, Tuple


_VERB_SUFFIXES = (
    "mak",
    "mek",
    "yor",
    "iyor",
    "uyor",
    "di",
    "dim",
    "dik",
    "du",
    "dum",
    "duk",
    "ti",
    "tim",
    "tik",
    "tu",
    "tum",
    "tuk",
)

_VERB_HINTS = {
    "yap",
    "et",
    "duzelt",
    "bul",
    "coz",
    "kur",
    "yukle",
    "baglan",
    "ac",
    "kapat",
    "dene",
    "anlat",
    "sor",
}

_STOP_SUBJECT = {
    "ve",
    "veya",
    "ama",
    "fakat",
    "ancak",
    "icin",
    "bir",
    "bu",
    "su",
    "o",
    "de",
    "da",
    "ile",
    "mi",
    "mu",
    "ne",
    "neden",
    "hangi",
    "nasil",
    "kim",
}

_PROBLEM_HINTS = {"sorun", "hata", "baglanti", "problem", "error", "ariza", "issue"}


def _ascii_fold(text: str) -> str:
    folded = str(text or "").lower()
    folded = folded.replace("\u0131", "i")
    folded = unicodedata.normalize("NFKD", folded)
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return folded


def _clean_text(text: str) -> str:
    txt = _ascii_fold(text).strip()
    txt = re.sub(r"[^a-z0-9_\s\.\-]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def _token(text: str) -> str:
    txt = _clean_text(text)
    txt = txt.replace(" ", "_")
    txt = re.sub(r"_+", "_", txt).strip("_")
    return txt


def _split_clauses(raw: str, sep_re: str) -> Tuple[str, str]:
    match = re.search(sep_re, raw)
    if not match:
        return "", ""
    left = raw[: match.start()].strip()
    right = raw[match.end() :].strip()
    return left, right


def _is_verbish(tok: str) -> bool:
    if not tok:
        return False
    if tok in _VERB_HINTS:
        return True
    if any(tok.endswith(sfx) for sfx in _VERB_SUFFIXES):
        return True
    return False


def _find_subject(tokens: List[str], action: str) -> str:
    if any(tok in {"ben", "biz"} for tok in tokens):
        return "ben"
    for tok in tokens:
        if tok == action:
            continue
        if tok in _STOP_SUBJECT:
            continue
        if len(tok) <= 1:
            continue
        return tok
    return "agent"


def _add(out: List[dict], seen: set, op: str, args: List[str]):
    if not op or not isinstance(args, list):
        return
    if any(not str(a).strip() for a in args):
        return
    row = {"op": str(op).upper(), "args": [str(a) for a in args]}
    key = (row["op"], tuple(row["args"]))
    if key in seen:
        return
    seen.add(key)
    out.append(row)


def semantic_fallback_ir(text: str, include_do: bool = True) -> List[dict]:
    """
    Deterministic fallback rules that prioritize high-value graph relations.
    DO is emitted only when no richer relation can be extracted.
    """
    raw = _clean_text(text)
    if not raw:
        return []

    out: List[dict] = []
    seen = set()

    # ISA: X bir Y(dir)
    for m in re.finditer(r"\b([a-z0-9_.-]+)\s+bir\s+([a-z0-9_.-]+)\s*(?:dir|dur|tir|tur)?\b", raw):
        _add(out, seen, "ISA", [_token(m.group(1)), _token(m.group(2))])

    # ATTR: X Y(dir) excluding "bir"
    for m in re.finditer(r"\b([a-z0-9_.-]+)\s+([a-z0-9_.-]+)\s*(?:dir|dur|tir|tur)\b", raw):
        if m.group(2) != "bir":
            _add(out, seen, "ATTR", [_token(m.group(1)), "ozellik", _token(m.group(2))])

    # GOAL: ... icin ...
    goal_match = re.search(r"\b([a-z0-9_.-]+)\s+icin\s+([a-z0-9_.-]+)\b", raw)
    if goal_match:
        _add(out, seen, "GOAL", [_token(goal_match.group(2)), _token(goal_match.group(1)), "medium"])

    # BEFORE: once ... sonra ...
    before_match = re.search(r"\bonce\s+([a-z0-9_.-]+)\s+.*\bsonra\s+([a-z0-9_.-]+)\b", raw)
    if before_match:
        _add(out, seen, "BEFORE", [_token(before_match.group(1)), _token(before_match.group(2))])

    # OPPOSE: A ama/fakat/ancak B
    oppose_match = re.search(r"\b([a-z0-9_.-]+)\s+(?:ama|fakat|ancak)\s+([a-z0-9_.-]+)\b", raw)
    if oppose_match:
        _add(out, seen, "OPPOSE", [_token(oppose_match.group(1)), _token(oppose_match.group(2))])

    # PREVENT: A engeller/onler B
    prevent_match = re.search(r"\b([a-z0-9_.-]+)\s+(?:engeller|onler)\s+([a-z0-9_.-]+)\b", raw)
    if prevent_match:
        _add(out, seen, "PREVENT", [_token(prevent_match.group(1)), _token(prevent_match.group(2))])

    # EVAL: X iyi/kotu/guzel/cirkin
    eval_match = re.search(r"\b([a-z0-9_.-]+)\s+(iyi|kotu|guzel|cirkin)\b", raw)
    if eval_match:
        score = "1.0" if eval_match.group(2) in {"iyi", "guzel"} else "-1.0"
        _add(out, seen, "EVAL", [_token(eval_match.group(1)), score])

    # WANT / KNOW / BELIEVE lexical signals.
    want_match = re.search(r"\b([a-z0-9_.-]+)\s+(?:[a-z0-9_.-]+\s+){0,5}ist(?:iyor|edi|er|iyorum|iyoruz)\b", raw)
    if want_match:
        _add(out, seen, "WANT", [_token(want_match.group(1)), "hedef"])

    know_match = re.search(r"\b([a-z0-9_.-]+)\s+(?:[a-z0-9_.-]+\s+){0,4}bil(?:iyor|ir)\b", raw)
    if know_match:
        _add(out, seen, "KNOW", [_token(know_match.group(1)), "bilgi"])

    believe_match = re.search(r"\b([a-z0-9_.-]+)\s+(?:[a-z0-9_.-]+\s+){0,4}inan(?:iyor|di|ir)\b", raw)
    if believe_match:
        _add(out, seen, "BELIEVE", [_token(believe_match.group(1)), "oneri", "0.7"])

    # CAUSE: "effect cunku cause"
    left, right = _split_clauses(raw, r"\b(cunku|nedeniyle|yuzunden|dolayisiyla)\b")
    if left and right:
        _add(out, seen, "CAUSE", [_token(right), _token(left)])

    # CAUSE: A ise/olursa B
    cond_match = re.search(r"\b(.+?)\s+(?:olursa|ise)\s+(.+?)(?:\s+olur)?$", raw)
    if cond_match:
        _add(out, seen, "CAUSE", [_token(cond_match.group(1)), _token(cond_match.group(2))])

    # Diagnostic pattern: "<env> icinde <pkg> paketi yoktu"
    missing_match = re.search(r"\b([a-z0-9_.-]+)\s+icinde\s+([a-z0-9_.-]+)\s+paketi\s+yok(?:tu|tur)?\b", raw)
    if missing_match:
        env_name = _token(missing_match.group(1))
        pkg_name = _token(missing_match.group(2))
        missing_state = f"{pkg_name}_missing"
        _add(out, seen, "ATTR", [env_name, "package_state", missing_state])

        if any(hint in raw for hint in _PROBLEM_HINTS):
            _add(out, seen, "CAUSE", [missing_state, "system_problem"])

    # Action chain for "bulup duzelttim" style diagnostics.
    if re.search(r"\bbulup\b.*\bduzelt(?:tim|tik|ti|ildi)\b", raw):
        _add(out, seen, "DO", ["ben", "root_cause_analysis"])
        _add(out, seen, "DO", ["ben", "issue_fix"])
        _add(out, seen, "BEFORE", ["root_cause_analysis", "issue_fix"])

    # Last resort DO only if richer relations were not found.
    if include_do and not out:
        toks = re.findall(r"[a-z0-9_.-]+", raw)
        if toks:
            action = ""
            for tok in toks:
                if _is_verbish(tok):
                    action = tok
                    break
            if not action:
                action = toks[-1]
            subject = _find_subject(toks, action)
            _add(out, seen, "DO", [_token(subject), _token(action)])

    return out


def merge_fallback(primary: List[dict], secondary: List[dict]) -> List[dict]:
    out: List[dict] = []
    seen = set()
    for row in (primary or []) + (secondary or []):
        if not isinstance(row, dict):
            continue
        op = str(row.get("op", "")).upper().strip()
        args = [str(a) for a in row.get("args", [])]
        if not op or not args:
            continue
        key = (op, tuple(args))
        if key in seen:
            continue
        seen.add(key)
        out.append({"op": op, "args": args})
    return out