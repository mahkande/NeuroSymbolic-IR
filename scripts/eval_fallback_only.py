#!/usr/bin/env python
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from core.fallback_rules import semantic_fallback_ir


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


def _ops(chain):
    out = set()
    for row in chain or []:
        if isinstance(row, dict):
            op = str(row.get("op", "")).upper().strip()
            if op:
                out.add(op)
    return out


def _safe_div(a, b):
    return (a / b) if b else 0.0


def evaluate(rows):
    tp = fp = fn = 0
    samples = []
    for row in rows:
        pred = _ops(semantic_fallback_ir(row.get("text", ""), include_do=True))
        gold = _ops(row.get("gold", []))
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
        if len(samples) < 40:
            samples.append(
                {
                    "id": row.get("id"),
                    "pred_ops": sorted(pred),
                    "gold_ops": sorted(gold),
                    "hit": bool(pred & gold),
                }
            )
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "samples": samples,
    }


def main():
    ap = argparse.ArgumentParser(description="Fallback-only gold evaluation.")
    ap.add_argument("--dataset", default="data/gold/gold_set_v1.jsonl")
    ap.add_argument("--out", default="reports/fallback_eval_report.json")
    args = ap.parse_args()

    rows = _load_jsonl(args.dataset)
    metrics = evaluate(rows)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "dataset_size": len(rows),
        "metrics": metrics,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

