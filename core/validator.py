import json
import os
import re
import unicodedata
from typing import Iterable

from core.isa_versioning import get_active_isa_path, get_active_version


class CognitiveValidator:
    PRONOUNS = {
        "ben",
        "biz",
        "sen",
        "siz",
        "o",
        "onlar",
        "bu",
        "su",
    }
    ALLOWED_AGENT_PRONOUNS = {"ben", "biz"}
    GENERIC_ENTITY_STOPWORDS = {"bir", "ve", "ile", "icin", "de", "da", "ki"}

    def __init__(self, isa_path=None):
        self.isa_path = isa_path or get_active_isa_path()
        self.isa_version = get_active_version()
        self.isa = self.load_isa()
        self.opcodes = self.flatten_opcodes()
        self.validation = self._load_validation_rules()

    def load_isa(self):
        if not os.path.exists(self.isa_path):
            raise FileNotFoundError(f"Hata: {self.isa_path} bulunamadi!")
        with open(self.isa_path, "r", encoding="utf-8-sig") as f:
            return json.load(f)

    def flatten_opcodes(self):
        flat_list = {}
        for _, content in self.isa.get("categories", {}).items():
            for op, details in content.get("opcodes", {}).items():
                flat_list[op] = details
        return flat_list

    def _load_validation_rules(self):
        payload = self.isa.get("validation", {})
        if isinstance(payload, dict):
            return payload
        return {}

    @staticmethod
    def _ascii_fold(text: str) -> str:
        value = str(text or "").strip().lower()
        value = value.replace("\u0131", "i")
        value = unicodedata.normalize("NFKD", value)
        value = "".join(ch for ch in value if not unicodedata.combining(ch))
        return value

    @classmethod
    def _norm_token(cls, text: str) -> str:
        value = cls._ascii_fold(text)
        value = re.sub(r"[^a-z0-9_\s\.-]", " ", value)
        value = re.sub(r"\s+", "_", value).strip("_")
        return value

    @classmethod
    def _looks_like_token(cls, text: str) -> bool:
        token = cls._norm_token(text)
        if not token:
            return False
        if len(token) > 128:
            return False
        return bool(re.fullmatch(r"[a-z0-9][a-z0-9_\.-]*", token))

    @classmethod
    def _is_number_in_range(cls, value, low: float, high: float) -> bool:
        try:
            v = float(value)
        except Exception:
            return False
        return low <= v <= high

    @classmethod
    def _known_entities(cls, known_entities: Iterable[str]):
        return {cls._norm_token(k) for k in (known_entities or []) if str(k).strip()}

    def _validate_arg_type(self, arg_type: str, value, known_entities: Iterable[str] = None):
        t = str(arg_type or "").strip().lower()
        sval = str(value or "").strip()
        token = self._norm_token(sval)
        known = self._known_entities(known_entities)

        if t in {"entity", "entity_or_concept", "concept", "type", "label", "relation_key"}:
            if not self._looks_like_token(sval):
                return False, f"arg tipi '{arg_type}' token degil: {value}"
            if token in self.GENERIC_ENTITY_STOPWORDS:
                return False, f"arg tipi '{arg_type}' anlamsiz stopword: {value}"
            if token in self.PRONOUNS and token not in self.ALLOWED_AGENT_PRONOUNS and token not in known:
                return False, f"arg tipi '{arg_type}' coreference gerektiriyor (zamir): {value}"
            return True, "ok"

        if t in {"fact", "state", "action", "event", "process", "ability", "question", "ir_chain_ref", "attr_value"}:
            if not sval:
                return False, f"arg tipi '{arg_type}' bos olamaz"
            if len(sval) > 240:
                return False, f"arg tipi '{arg_type}' cok uzun"
            return True, "ok"

        if t == "priority":
            enums = self.validation.get("enums", {}) if isinstance(self.validation, dict) else {}
            allowed = [self._ascii_fold(x) for x in enums.get("priority", ["low", "medium", "high", "critical"])]
            if self._ascii_fold(sval) not in allowed:
                return False, f"priority gecersiz: {value}"
            return True, "ok"

        if t == "confidence":
            if not self._is_number_in_range(value, 0.0, 1.0):
                return False, f"confidence [0,1] araliginda olmali: {value}"
            return True, "ok"

        if t == "score":
            if not self._is_number_in_range(value, -1.0, 1.0):
                return False, f"score [-1,1] araliginda olmali: {value}"
            return True, "ok"

        if t == "opcode":
            op = str(value or "").strip().upper()
            if op not in self.opcodes:
                return False, f"opcode referansi gecersiz: {value}"
            return True, "ok"

        # Unknown type => non-empty string guard.
        if not sval:
            return False, f"arg tipi '{arg_type}' icin bos deger"
        return True, "ok"

    def validate_instruction(self, op, args, known_entities: Iterable[str] = None):
        if op not in self.opcodes:
            return False, f"Gecersiz Opcode: {op}"

        expected_args_count = len(self.opcodes[op].get("args", []))
        if len(args) != expected_args_count:
            return False, f"{op} icin beklenen arguman sayisi {expected_args_count}, alinan {len(args)}"

        op_rules = (self.validation.get("op_constraints", {}) or {}).get(op, {}) if isinstance(self.validation, dict) else {}
        arg_types = op_rules.get("arg_types", []) if isinstance(op_rules, dict) else []

        if isinstance(arg_types, list) and len(arg_types) == expected_args_count:
            for idx, (arg, at) in enumerate(zip(args, arg_types)):
                ok, msg = self._validate_arg_type(at, arg, known_entities=known_entities)
                if not ok:
                    return False, f"{op} arg#{idx} hata: {msg}"

        # Semantic constraints: distinct args for selected opcodes.
        distinct_pairs = op_rules.get("distinct_pairs", []) if isinstance(op_rules, dict) else []
        for pair in distinct_pairs:
            if not isinstance(pair, list) or len(pair) != 2:
                continue
            i, j = int(pair[0]), int(pair[1])
            if i < 0 or j < 0 or i >= len(args) or j >= len(args):
                continue
            if self._norm_token(args[i]) == self._norm_token(args[j]):
                return False, f"{op} icin arg#{i} ve arg#{j} ayni olamaz"

        return True, "Gecerli"

    def check_logical_conflicts(self, ir_chain):
        causes = set()
        opposes = set()

        for instr in ir_chain:
            op = instr.get("op")
            args = instr.get("args")

            if op == "CAUSE":
                causes.add(tuple(args))
            elif op == "OPPOSE":
                opposes.add(tuple(sorted(args)))

        for cause in causes:
            if tuple(sorted(cause)) in opposes:
                return False, f"Mantiksal Celiski: {cause[0]} hem {cause[1]}'e sebep oluyor hem de onunla celisiyor!"

        return True, "Mantiksal tutarlilik onaylandi."


if __name__ == "__main__":
    validator = CognitiveValidator()

    sample_ir = [
        {"op": "DEF_ENTITY", "args": ["yagmur", "hava_durumu"]},
        {"op": "DEF_ENTITY", "args": ["piknik", "aktivite"]},
        {"op": "CAUSE", "args": ["yagmur", "piknik"]},
        {"op": "OPPOSE", "args": ["yagmur", "piknik"]},
    ]

    print("--- Sozdizimi Kontrolu ---")
    for instr in sample_ir:
        valid, msg = validator.validate_instruction(instr["op"], instr["args"])
        print(f"{instr['op']}: {valid} ({msg})")

    print("\n--- Mantiksal Denetim ---")
    consistent, logic_msg = validator.check_logical_conflicts(sample_ir)
    print(f"Sonuc: {consistent} | Mesaj: {logic_msg}")