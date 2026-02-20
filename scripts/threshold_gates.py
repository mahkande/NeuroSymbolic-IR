#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def _load_json(path: str):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8-sig"))


def _f(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def main():
    parser = argparse.ArgumentParser(description="Release threshold gates for quality/latency.")
    parser.add_argument("--thresholds", default="spec/release_thresholds.json")
    parser.add_argument("--eval-report", default="reports/eval_report_latest.json")
    parser.add_argument("--load-report", default="memory/load_test_smoke_report.json")
    parser.add_argument("--ablation-report", default="reports/retriever_ablation_report.json")
    parser.add_argument("--release-report", default="memory/release_check_report.json")
    parser.add_argument("--out", default="reports/threshold_gate_report.json")
    args = parser.parse_args()

    thresholds = _load_json(args.thresholds) or {}
    eval_r = _load_json(args.eval_report) or {}
    load_r = _load_json(args.load_report) or {}
    abl_r = _load_json(args.ablation_report) or {}
    rel_r = _load_json(args.release_report) or {}

    failed = []
    checks = []

    q = thresholds.get("quality", {})
    e_m = eval_r.get("metrics", {})
    if eval_r:
        micro_f1 = _f(e_m.get("micro_f1"))
        macro_f1 = _f(e_m.get("macro_f1"))
        sem = _f(e_m.get("sentence_exact_match"))
        min_micro = _f(q.get("min_micro_f1"), 0.0)
        min_macro = _f(q.get("min_macro_f1"), 0.0)
        min_sem = _f(q.get("min_sentence_exact"), 0.0)
        checks.extend(
            [
                {"name": "micro_f1", "value": micro_f1, "threshold": min_micro, "ok": micro_f1 >= min_micro},
                {"name": "macro_f1", "value": macro_f1, "threshold": min_macro, "ok": macro_f1 >= min_macro},
                {"name": "sentence_exact_match", "value": sem, "threshold": min_sem, "ok": sem >= min_sem},
            ]
        )
    else:
        failed.append("missing_eval_report")

    l = thresholds.get("latency", {})
    if load_r:
        p95 = _f(((load_r.get("parallel", {}) or {}).get("latency", {}) or {}).get("p95_ms"))
        max_p95 = _f(l.get("max_parallel_p95_ms"), 1e12)
        checks.append({"name": "parallel_p95_ms", "value": p95, "threshold": max_p95, "ok": p95 <= max_p95, "direction": "max"})
    else:
        failed.append("missing_load_report")

    if rel_r:
        ok = bool(rel_r.get("ok", False))
        checks.append({"name": "release_check_ok", "value": ok, "threshold": True, "ok": ok})
    else:
        failed.append("missing_release_report")

    if abl_r:
        mode = str(abl_r.get("default_strategy", "")).strip().lower()
        allowed = set(thresholds.get("retrieval", {}).get("allowed_default_modes", ["hybrid", "vector-only", "graph-only"]))
        mode_ok = mode in allowed
        checks.append({"name": "default_retrieval_mode", "value": mode, "threshold": sorted(allowed), "ok": mode_ok})
    else:
        failed.append("missing_ablation_report")

    failed.extend([c["name"] for c in checks if not c.get("ok")])
    report = {
        "ok": len(failed) == 0,
        "failed": failed,
        "checks": checks,
        "inputs": {
            "thresholds": args.thresholds,
            "eval_report": args.eval_report,
            "load_report": args.load_report,
            "ablation_report": args.ablation_report,
            "release_report": args.release_report,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
