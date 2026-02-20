#!/usr/bin/env python
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.quality_metrics import provenance_coverage
from main import get_graph_backend_status, get_vector_backend_status, load_global_graph, run_cognitive_os


def _run(cmd):
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "cmd": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "")[-1200:],
        "stderr": (proc.stderr or "")[-1200:],
        "duration_sec": round(time.time() - t0, 2),
    }


def main():
    report = {"ts": time.time(), "checks": []}

    report["checks"].append(
        _run([sys.executable, "-m", "compileall", "core", "memory", "scripts", "main.py", "ui/dashboard.py"])
    )

    smoke = run_cognitive_os("Yazilim testi kalite guvencesi icin gereklidir.")
    report["smoke"] = {
        "z3_status": smoke.get("z3_status"),
        "log_size": len(smoke.get("log", [])),
        "has_verifier": any("[VERIFIER]" in str(x) for x in smoke.get("log", [])),
        "has_vector": any("[VECTOR]" in str(x) for x in smoke.get("log", [])),
    }

    graph = load_global_graph()
    report["graph_status"] = get_graph_backend_status()
    report["vector_status"] = get_vector_backend_status()
    report["graph_size"] = {"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges()}
    report["provenance"] = provenance_coverage(graph)

    failed = []
    if report["checks"][0]["returncode"] != 0:
        failed.append("compileall")
    if str(report["smoke"]["z3_status"]).strip().lower() == "hata":
        failed.append("smoke")
    if not report["smoke"]["has_verifier"]:
        failed.append("verifier_missing")
    if report["provenance"].get("coverage", 0.0) < 0.95:
        failed.append("provenance_coverage")

    report["failed"] = failed
    report["ok"] = len(failed) == 0

    out = Path("memory/release_check_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Release check report written: {out}")
    if not report["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
