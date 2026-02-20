#!/usr/bin/env python
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main import run_cognitive_os


CANARY_INPUTS = [
    "Limon bir meyvedir.",
    "Yagmur artarsa trafik yogunlugu artar.",
    "Ali ders calismak istiyor.",
    "Planlama yapilirsa hata orani duser.",
    "Kalite guvencesi release icin zorunludur.",
    "Once analiz sonra uygulama gelir.",
]


def _f(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return float(default)


def run_canary(samples: int = 6):
    rows = []
    errors = 0
    unsupported_total = 0
    proof_total = 0
    contradictions = 0
    lat = []
    for i in range(max(1, samples)):
        text = CANARY_INPUTS[i % len(CANARY_INPUTS)]
        t0 = time.perf_counter()
        res = run_cognitive_os(text)
        ms = (time.perf_counter() - t0) * 1000.0
        z3 = str(res.get("z3_status", "")).strip().lower()
        ok = z3 not in {"hata", "unsat"}
        if z3 in {"unsat"}:
            contradictions += 1
        if not ok:
            errors += 1
        lat.append(ms)
        proofs = res.get("proofs", []) or []
        proof_total += len(proofs)
        unsupported_total += sum(1 for p in proofs if str(p.get("verdict", "")).lower() != "supported")
        rows.append({"i": i, "text": text, "ok": ok, "latency_ms": round(ms, 2), "z3_status": res.get("z3_status")})
    lat_sorted = sorted(lat)
    p95_idx = max(0, min(len(lat_sorted) - 1, int(len(lat_sorted) * 0.95) - 1))
    return {
        "samples": rows,
        "summary": {
            "total": len(rows),
            "errors": errors,
            "error_rate": round(errors / max(1, len(rows)), 4),
            "unsupported_claim_rate": round(unsupported_total / max(1, proof_total), 4),
            "contradiction_rate": round(contradictions / max(1, len(rows)), 4),
            "avg_latency_ms": round(sum(lat) / max(1, len(lat)), 2),
            "p95_latency_ms": round(lat_sorted[p95_idx], 2) if lat_sorted else 0.0,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Canary rollout evaluator with automatic rollback decision.")
    parser.add_argument("--policy", default="spec/canary_policy.json")
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--out", default="reports/canary_rollout_report.json")
    args = parser.parse_args()

    policy_path = Path(args.policy)
    policy = json.loads(policy_path.read_text(encoding="utf-8-sig")) if policy_path.exists() else {}
    max_error_rate = _f(policy.get("max_error_rate"), 0.1)
    max_p95 = _f(policy.get("max_p95_latency_ms"), 3000.0)
    min_pass = _f(policy.get("min_pass_rate"), 0.9)
    max_unsupported = _f(policy.get("max_unsupported_claim_rate"), 0.20)
    max_contradiction = _f(policy.get("max_contradiction_rate"), 0.10)
    traffic = int(policy.get("traffic_percent", 5))

    canary = run_canary(samples=max(1, args.samples))
    summ = canary["summary"]
    pass_rate = 1.0 - _f(summ.get("error_rate"), 1.0)
    err_rate = _f(summ.get("error_rate"))
    p95 = _f(summ.get("p95_latency_ms"))
    unsupported = _f(summ.get("unsupported_claim_rate"))
    contradiction = _f(summ.get("contradiction_rate"))

    hard_fail = (err_rate > max_error_rate) or (p95 > max_p95) or (pass_rate < min_pass)
    reliability_fail = (unsupported > max_unsupported) or (contradiction > max_contradiction)
    should_rollback = hard_fail or (reliability_fail and unsupported > (max_unsupported * 1.5))
    should_escalate = (not should_rollback) and reliability_fail

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "traffic_percent": traffic,
        "policy": {
            "max_error_rate": max_error_rate,
            "max_p95_latency_ms": max_p95,
            "min_pass_rate": min_pass,
            "max_unsupported_claim_rate": max_unsupported,
            "max_contradiction_rate": max_contradiction,
        },
        "canary": canary,
        "decision": {
            "action": "rollback" if should_rollback else ("escalate" if should_escalate else "promote"),
            "reason": "threshold_violation" if should_rollback else ("reliability_warning" if should_escalate else "healthy_canary"),
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if should_rollback:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
