#!/usr/bin/env python
import argparse
import json
import subprocess
import sys
from pathlib import Path


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _metric(report, key):
    return float(report.get("metrics", {}).get(key, 0.0))


def _compare(curr, base, min_support=5, max_drop=0.03):
    issues = []
    for k in ("micro_f1", "macro_f1", "sentence_exact_match"):
        drop = _metric(base, k) - _metric(curr, k)
        if drop > max_drop:
            issues.append(f"{k} dropped by {drop:.4f} (baseline={_metric(base, k):.4f}, current={_metric(curr, k):.4f})")

    curr_ops = curr.get("per_opcode", {})
    base_ops = base.get("per_opcode", {})
    for op, bm in base_ops.items():
        if int(bm.get("support", 0)) < min_support:
            continue
        if op not in curr_ops:
            issues.append(f"{op} missing in current report")
            continue
        b_f1 = float(bm.get("f1", 0.0))
        c_f1 = float(curr_ops[op].get("f1", 0.0))
        if b_f1 - c_f1 > max_drop:
            issues.append(f"{op}.f1 dropped by {b_f1 - c_f1:.4f} (baseline={b_f1:.4f}, current={c_f1:.4f})")
    return issues


def main():
    parser = argparse.ArgumentParser(description="Regression suite for eval metrics.")
    parser.add_argument("--dataset", default="data/gold/gold_set_v1.jsonl")
    parser.add_argument("--report", default="reports/eval_report_latest.json")
    parser.add_argument("--baseline", default="reports/eval_baseline.json")
    parser.add_argument("--min-support", type=int, default=5)
    parser.add_argument("--max-drop", type=float, default=0.03)
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args()

    cmd = [sys.executable, "scripts/eval_pipeline.py", "--dataset", args.dataset, "--output", args.report]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise SystemExit(proc.returncode)

    report_path = Path(args.report)
    baseline_path = Path(args.baseline)
    current = _load_json(report_path)

    if args.update_baseline or not baseline_path.exists():
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Baseline updated: {baseline_path}")
        return

    baseline = _load_json(baseline_path)
    issues = _compare(current, baseline, min_support=max(1, args.min_support), max_drop=max(0.0, args.max_drop))
    if issues:
        print("REGRESSION: FAIL")
        for item in issues:
            print(f"- {item}")
        raise SystemExit(2)

    print("REGRESSION: PASS")
    print(json.dumps(current.get("metrics", {}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
