#!/usr/bin/env python
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.fallback_rules import semantic_fallback_ir
from core.model_bridge import NanbeigeBridge
from core.parser import IRParser
from core.validator import CognitiveValidator
from main import run_cognitive_os


MODES = ("llm-only", "fallback-only", "hybrid", "backward-verifier")


def _load_jsonl(path: str):
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    for i, line in enumerate(p.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        row.setdefault("id", f"row_{i}")
        row.setdefault("text", "")
        row.setdefault("gold", [])
        rows.append(row)
    return rows


def _gold_ops(row):
    out = set()
    for instr in row.get("gold", []):
        if isinstance(instr, dict):
            op = str(instr.get("op", "")).upper().strip()
            if op:
                out.add(op)
    return out


def _pred_ops_from_ir(ir_chain):
    out = set()
    for instr in ir_chain or []:
        if isinstance(instr, dict):
            op = str(instr.get("op", "")).upper().strip()
            if op:
                out.add(op)
    return out


def _pred_ops_from_result(res):
    proofs = (res or {}).get("proofs", []) or []
    ops = set()
    for p in proofs:
        op = str(p.get("claim", {}).get("op", "")).upper().strip()
        if op:
            ops.add(op)
    return ops


def _safe_div(a, b):
    return (a / b) if b else 0.0


def run_mode(rows, mode: str):
    validator = CognitiveValidator()
    parser = IRParser()
    bridge = NanbeigeBridge()

    tp = fp = fn = 0
    lat = []
    errors = 0
    unsupported_claims = 0
    total_claims = 0

    prev = os.getenv("COGNITIVE_USE_BACKWARD_VERIFIER")
    try:
        for row in rows:
            text = row.get("text", "")
            gold = _gold_ops(row)
            t0 = time.perf_counter()
            pred = set()

            if mode == "fallback-only":
                ir = semantic_fallback_ir(text, include_do=True)
                pred = _pred_ops_from_ir(ir)
            elif mode == "llm-only":
                raw = bridge.compile_to_ir(text, validator.isa, max_retries=1, memory_terms=[])
                ir = parser.parse_raw_output(raw, allowed_ops=set(validator.opcodes.keys()), strict_schema=True)
                if isinstance(ir, list):
                    pred = _pred_ops_from_ir(ir)
                else:
                    errors += 1
            else:
                os.environ["COGNITIVE_USE_BACKWARD_VERIFIER"] = "1" if mode == "backward-verifier" else "0"
                res = run_cognitive_os(text)
                pred = _pred_ops_from_result(res)
                z3 = str(res.get("z3_status", "")).strip().lower()
                if z3 in {"hata", "unsat"}:
                    errors += 1
                proofs = res.get("proofs", []) or []
                total_claims += len(proofs)
                unsupported_claims += sum(1 for p in proofs if str(p.get("verdict", "")).lower() != "supported")

            lat.append((time.perf_counter() - t0) * 1000.0)
            tp += len(pred & gold)
            fp += len(pred - gold)
            fn += len(gold - pred)

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
        lat_s = sorted(lat)
        p95_idx = max(0, min(len(lat_s) - 1, int(len(lat_s) * 0.95) - 1)) if lat_s else 0
        return {
            "mode": mode,
            "quality": {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            },
            "runtime": {
                "avg_latency_ms": round(_safe_div(sum(lat), len(lat)), 2),
                "p95_latency_ms": round(lat_s[p95_idx], 2) if lat_s else 0.0,
                "error_rate": round(_safe_div(errors, len(rows)), 4),
            },
            "reliability": {
                "unsupported_claim_rate": round(_safe_div(unsupported_claims, total_claims), 4) if total_claims else 0.0,
                "total_claims": total_claims,
            },
        }
    finally:
        if prev is None:
            os.environ.pop("COGNITIVE_USE_BACKWARD_VERIFIER", None)
        else:
            os.environ["COGNITIVE_USE_BACKWARD_VERIFIER"] = prev


def main():
    ap = argparse.ArgumentParser(description="Offline benchmark: llm-only vs fallback-only vs hybrid vs backward-verifier.")
    ap.add_argument("--dataset", default="data/gold/gold_set_v1.jsonl")
    ap.add_argument("--out", default="reports/offline_benchmark_v9.json")
    args = ap.parse_args()

    rows = _load_jsonl(args.dataset)
    results = [run_mode(rows, m) for m in MODES]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "dataset_size": len(rows),
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

