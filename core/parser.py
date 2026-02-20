import json
import os
import re
from typing import Iterable, List, Tuple


class IRParser:
    REQUIRED_KEYS = {"op", "args"}
    OPTIONAL_KEYS = {"confidence", "source_span", "evidence_ids", "provenance", "source"}

    def __init__(self):
        self.json_pattern = re.compile(r"\[\s*\{.*\}\s*\]", re.DOTALL)
        self.strict_schema_default = os.getenv("COGNITIVE_STRICT_IR_SCHEMA", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    @staticmethod
    def _extract_json_array_text(raw: str):
        text = str(raw or "")
        start = text.find("[")
        if start < 0:
            return None
        depth = 0
        in_str = False
        esc = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
        return None

    def _validate_row(self, row: dict, allowed_ops: Iterable[str] = None, strict_schema: bool = True) -> Tuple[bool, str, dict]:
        if not isinstance(row, dict):
            return False, "IR satiri obje degil.", {}

        keys = set(row.keys())
        if not self.REQUIRED_KEYS.issubset(keys):
            return False, "IR satirinda zorunlu anahtar eksik (op,args).", {}
        if strict_schema:
            allowed = self.REQUIRED_KEYS | self.OPTIONAL_KEYS
            unknown = [k for k in keys if k not in allowed]
            if unknown:
                return False, f"Schema disi alan bulundu: {unknown}", {}

        op = str(row.get("op", "")).strip().upper()
        if not op:
            return False, "op bos olamaz.", {}
        if allowed_ops and op not in set(allowed_ops):
            return False, f"Bilinmeyen opcode: {op}", {}

        args = row.get("args", [])
        if not isinstance(args, list):
            return False, "args list olmalidir.", {}
        clean_args = [str(a).strip() for a in args]
        if any(not a for a in clean_args):
            return False, "args bos deger iceremez.", {}

        clean = {"op": op, "args": clean_args}

        if "confidence" in row:
            try:
                conf = float(row.get("confidence"))
            except Exception:
                return False, "confidence sayisal olmali.", {}
            clean["confidence"] = conf

        if "source_span" in row:
            span = str(row.get("source_span", "")).strip()
            if not span:
                return False, "source_span bos olamaz.", {}
            clean["source_span"] = span[:500]

        if "evidence_ids" in row:
            ev = row.get("evidence_ids", [])
            if not isinstance(ev, list):
                return False, "evidence_ids list olmalidir.", {}
            clean["evidence_ids"] = [str(x).strip() for x in ev if str(x).strip()]

        if "provenance" in row:
            pv = row.get("provenance")
            if not isinstance(pv, dict):
                return False, "provenance obje olmalidir.", {}
            clean["provenance"] = pv

        if "source" in row:
            clean["source"] = str(row.get("source", "")).strip()[:120]

        return True, "ok", clean

    def parse_raw_output(self, llm_output, allowed_ops: Iterable[str] = None, strict_schema: bool = None):
        strict_schema = self.strict_schema_default if strict_schema is None else bool(strict_schema)
        if isinstance(llm_output, dict):
            if "error" in llm_output:
                return llm_output
            return {"error": "Beklenen format liste, gelen dict.", "raw": llm_output}
        if isinstance(llm_output, list):
            rows = llm_output
        else:
            json_str = self._extract_json_array_text(llm_output)
            if not json_str:
                return {"error": "Metin icerisinde gecerli bir IR listesi bulunamadi.", "raw": llm_output}
            try:
                rows = json.loads(json_str)
            except json.JSONDecodeError as exc:
                return {"error": f"JSON cozumleme hatasi: {exc}", "raw": json_str}

        if not isinstance(rows, list):
            return {"error": "IR kok nesnesi liste olmali.", "raw": rows}

        clean_rows: List[dict] = []
        for idx, row in enumerate(rows):
            ok, msg, clean = self._validate_row(row, allowed_ops=allowed_ops, strict_schema=strict_schema)
            if not ok:
                return {"error": f"IR schema hatasi (index={idx}): {msg}", "raw": row}
            clean_rows.append(clean)
        return clean_rows

    @staticmethod
    def stringify_ir(ir_chain):
        lines = []
        for i, instr in enumerate(ir_chain or []):
            op = instr.get("op", "UNKNOWN")
            args = ", ".join(instr.get("args", []))
            lines.append(f"{i + 1}. {op}({args})")
        return "\n".join(lines)
