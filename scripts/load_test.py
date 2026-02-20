#!/usr/bin/env python
import argparse
import json
import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.file_listener import FileListenerService
from main import clear_global_graph, load_global_graph, run_cognitive_os


def _latency_stats(values):
    if not values:
        return {"count": 0, "mean_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    vals = sorted(values)
    p95_idx = max(0, min(len(vals) - 1, int(len(vals) * 0.95) - 1))
    return {
        "count": len(vals),
        "mean_ms": round(statistics.mean(vals), 2),
        "p95_ms": round(vals[p95_idx], 2),
        "max_ms": round(vals[-1], 2),
    }


def run_long_text_case(words: int = 1200):
    sentence = "Enerji verimliligi icin planli bakim gerekir ve veri analizi karar kalitesini artirir."
    text = " ".join([sentence] * max(1, words // len(sentence.split())))
    t0 = time.perf_counter()
    res = run_cognitive_os(text)
    dt = (time.perf_counter() - t0) * 1000
    return {
        "name": "long_text",
        "latency_ms": round(dt, 2),
        "z3_status": res.get("z3_status"),
        "log_size": len(res.get("log", [])),
    }


def run_parallel_case(requests: int = 24, workers: int = 6):
    inputs = [
        "Limon bir meyvedir ve vitamin icin faydalidir.",
        "Yagmur artarsa trafik yogunlugu artar.",
        "Ali ders calismak istiyor ama yoruldu.",
        "Planlama yapilirsa hata orani duser.",
        "Bir urun kaliteli ise memnuniyet artar.",
        "Risk analizi proje basarisi icin onemlidir.",
    ]
    latencies = []
    errors = 0
    lock = threading.Lock()

    def _task(i):
        txt = inputs[i % len(inputs)]
        t0 = time.perf_counter()
        res = run_cognitive_os(txt)
        ms = (time.perf_counter() - t0) * 1000
        ok = res.get("z3_status") not in {"Hata"}
        return ms, ok

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(_task, i) for i in range(max(1, requests))]
        for f in as_completed(futs):
            ms, ok = f.result()
            with lock:
                latencies.append(ms)
                if not ok:
                    errors += 1

    return {
        "name": "parallel_requests",
        "requests": requests,
        "workers": workers,
        "errors": errors,
        "latency": _latency_stats(latencies),
    }


def run_listener_case(blocks: int = 20, timeout_sec: float = 20.0):
    conv = Path("memory/loadtest_conversation.txt")
    proc = Path("memory/loadtest_processed_logs.jsonl")
    conv.parent.mkdir(parents=True, exist_ok=True)
    conv.write_text("", encoding="utf-8")
    proc.write_text("", encoding="utf-8")

    svc = FileListenerService(
        conversation_path=str(conv),
        processed_logs_path=str(proc),
        debounce_sec=0.2,
    )
    svc.start()
    try:
        payload = []
        for i in range(max(1, blocks)):
            payload.append(f"Blok {i} icin test cumlesi: verimlilik artarsa maliyet azalir.")
        conv.write_text("\n\n".join(payload), encoding="utf-8")

        deadline = time.time() + max(5.0, timeout_sec)
        processed = 0
        while time.time() < deadline:
            lines = [ln for ln in proc.read_text(encoding="utf-8").splitlines() if ln.strip()]
            processed = sum(1 for ln in lines if '"kind": "consumed_block"' in ln)
            if processed >= blocks:
                break
            time.sleep(0.25)
    finally:
        svc.stop()

    return {
        "name": "listener_ingest",
        "target_blocks": blocks,
        "processed_blocks": processed,
        "success": processed >= blocks,
    }


def main():
    parser = argparse.ArgumentParser(description="Production load test scenarios for Cognitive OS")
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--long-words", type=int, default=1200)
    parser.add_argument("--listener-blocks", type=int, default=20)
    parser.add_argument("--reset-memory", action="store_true")
    parser.add_argument("--report", default="memory/load_test_report.json")
    args = parser.parse_args()

    if args.reset_memory:
        clear_global_graph()

    started = time.time()
    long_case = run_long_text_case(words=args.long_words)
    parallel_case = run_parallel_case(requests=args.requests, workers=args.workers)
    listener_case = run_listener_case(blocks=args.listener_blocks)
    graph = load_global_graph()

    report = {
        "started_at": started,
        "duration_sec": round(time.time() - started, 2),
        "long_text": long_case,
        "parallel": parallel_case,
        "listener": listener_case,
        "graph": {"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges()},
    }
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Load test report written: {out}")


if __name__ == "__main__":
    main()
