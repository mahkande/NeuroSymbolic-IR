#!/usr/bin/env python
import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.hybrid_semantics import RuleBasedExtractorV2


def _sig(instr):
    op = str(instr.get("op", "")).upper().strip()
    args = tuple(str(a).strip().lower() for a in instr.get("args", []))
    return op, args


def _safe_div(a, b):
    return (a / b) if b else 0.0


def load_dataset(path: str):
    rows = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row.setdefault("id", f"row_{i}")
            row.setdefault("gold", [])
            row.setdefault("text", "")
            rows.append(row)
    return rows


def evaluate(dataset_path: str):
    rows = load_dataset(dataset_path)
    ext = RuleBasedExtractorV2()

    per = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "support": 0})
    sentence_exact = 0
    total_gold = 0
    total_pred = 0
    total_tp = 0
    detailed = []

    for row in rows:
        gold_set = {_sig(x) for x in row.get("gold", []) if isinstance(x, dict)}
        pred_set = {_sig(x) for x in ext.extract_ir(row.get("text", ""))}
        sentence_exact += int(gold_set == pred_set)

        for op, _ in gold_set:
            per[op]["support"] += 1

        tp = gold_set & pred_set
        fp = pred_set - gold_set
        fn = gold_set - pred_set

        total_tp += len(tp)
        total_gold += len(gold_set)
        total_pred += len(pred_set)

        for op, _ in tp:
            per[op]["tp"] += 1
        for op, _ in fp:
            per[op]["fp"] += 1
        for op, _ in fn:
            per[op]["fn"] += 1

        detailed.append(
            {
                "id": row.get("id"),
                "text": row.get("text", ""),
                "gold": [{"op": x[0], "args": list(x[1])} for x in sorted(gold_set)],
                "pred": [{"op": x[0], "args": list(x[1])} for x in sorted(pred_set)],
                "tp": len(tp),
                "fp": len(fp),
                "fn": len(fn),
            }
        )

    per_opcode = {}
    macro_p = 0.0
    macro_r = 0.0
    macro_f1 = 0.0
    ops = sorted(per.keys())
    for op in ops:
        tp = per[op]["tp"]
        fp = per[op]["fp"]
        fn = per[op]["fn"]
        p = _safe_div(tp, tp + fp)
        r = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * p * r, p + r) if (p + r) else 0.0
        per_opcode[op] = {
            "support": per[op]["support"],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
        }
        macro_p += p
        macro_r += r
        macro_f1 += f1

    n_ops = max(1, len(ops))
    micro_p = _safe_div(total_tp, total_pred)
    micro_r = _safe_div(total_tp, total_gold)
    micro_f1 = _safe_div(2 * micro_p * micro_r, micro_p + micro_r) if (micro_p + micro_r) else 0.0

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": dataset_path,
        "dataset_size": len(rows),
        "metrics": {
            "micro_precision": round(micro_p, 4),
            "micro_recall": round(micro_r, 4),
            "micro_f1": round(micro_f1, 4),
            "macro_precision": round(macro_p / n_ops, 4),
            "macro_recall": round(macro_r / n_ops, 4),
            "macro_f1": round(macro_f1 / n_ops, 4),
            "sentence_exact_match": round(_safe_div(sentence_exact, len(rows)), 4),
        },
        "counts": {
            "gold_total": total_gold,
            "pred_total": total_pred,
            "tp_total": total_tp,
        },
        "per_opcode": per_opcode,
        "samples": detailed[:80],
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate opcode extraction on gold dataset.")
    parser.add_argument("--dataset", default="data/gold/gold_set_v1.jsonl")
    parser.add_argument("--output", default="reports/eval_report_latest.json")
    args = parser.parse_args()

    report = evaluate(args.dataset)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"Eval report written: {out}")


if __name__ == "__main__":
    main()
