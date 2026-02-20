#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.isa_versioning import get_active_version, migrate_ir_chain


def _load_json_or_jsonl(path: Path):
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line in text.splitlines():
            if not line.strip():
                continue
            rows.append(json.loads(line))
        return rows, "jsonl"
    return json.loads(text), "json"


def _save_json_or_jsonl(path: Path, data, mode: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "jsonl":
        with path.open("w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _migrate_payload(payload, from_v: str, to_v: str):
    # Supported input forms:
    # 1) pure ir chain: [{"op":"...", "args":[...]}]
    # 2) wrapped: {"ir":[...], ...}
    # 3) jsonl rows where each row has "ir" or direct instruction list
    if isinstance(payload, list):
        if payload and isinstance(payload[0], dict) and "op" in payload[0]:
            return migrate_ir_chain(payload, from_v, to_v)
        out = []
        for row in payload:
            if isinstance(row, dict) and "ir" in row and isinstance(row["ir"], list):
                row = dict(row)
                row["ir"] = migrate_ir_chain(row["ir"], from_v, to_v)
            out.append(row)
        return out
    if isinstance(payload, dict) and isinstance(payload.get("ir"), list):
        out = dict(payload)
        out["ir"] = migrate_ir_chain(out["ir"], from_v, to_v)
        return out
    return payload


def main():
    parser = argparse.ArgumentParser(description="Migrate IR payloads between ISA versions.")
    parser.add_argument("--input", required=True, help="Input JSON or JSONL path")
    parser.add_argument("--output", required=True, help="Output JSON or JSONL path")
    parser.add_argument("--from-version", default="1.0")
    parser.add_argument("--to-version", default=None)
    args = parser.parse_args()

    to_v = args.to_version or get_active_version()
    src = Path(args.input)
    dst = Path(args.output)
    payload, mode = _load_json_or_jsonl(src)
    migrated = _migrate_payload(payload, args.from_version, to_v)
    _save_json_or_jsonl(dst, migrated, mode)
    print(f"ISA migration done: {src} -> {dst} ({args.from_version} -> {to_v})")


if __name__ == "__main__":
    main()
