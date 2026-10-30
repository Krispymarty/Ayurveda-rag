"""Inspect isolated readiness preview and repeat the historical retrieval set."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import time

import requests

from ipindia_corpus.llm_experiment import ROOT, percentile, save_artifact, sha256, timestamp
from ipindia_corpus.passage_evaluation import matches


def verify(*, restart=False):
    if restart:
        subprocess.run(["docker", "restart", "ayurip-readiness-preview"], capture_output=True, check=True)
    probes, concurrent = [], []
    started = time.monotonic()
    session = requests.Session()
    session.trust_env = False
    while time.monotonic() - started < 70:
        try:
            before = time.monotonic()
            response = session.get("http://127.0.0.1:8002/health", timeout=(1, 2))
            probes.append({"at_ms": (time.monotonic() - started) * 1000, "http_ms": (time.monotonic() - before) * 1000,
                           "status": response.status_code, "body": response.json()})
            if response.status_code == 503 and not concurrent:
                def probe(path):
                    current = requests.Session()
                    current.trust_env = False
                    t = time.monotonic()
                    r = current.get("http://127.0.0.1:8002" + path, timeout=3)
                    return {"path": path, "status": r.status_code, "ms": (time.monotonic() - t) * 1000}
                with ThreadPoolExecutor(max_workers=5) as pool:
                    concurrent = list(pool.map(probe, ["/health"] * 3 + ["/search?q=Patents%20Act", "/live"]))
            if response.status_code == 200:
                break
        except requests.RequestException:
            pass
        time.sleep(.5)
    else:
        raise ValueError("preview_did_not_become_ready")
    startup = {"at": timestamp(), "probes": probes, "concurrent_probes": concurrent,
               "ready": response.json(), "observed_initializing_503": any(p["status"] == 503 for p in probes)}
    save_artifact(ROOT, "reports/reliability/readiness_startup.json", startup)
    path = ROOT / "data/benchmarks/ipindia-reviewed-20261015-v1/expanded.json"
    raw = path.read_bytes()
    dataset = json.loads(raw)
    queries = dataset["queries"]
    rows = []
    # Pace outside the measured HTTP interval to obey the main API limiter.
    last = time.monotonic() - 2
    for number in range(4):
        for item in queries:
            pause = 1.05 - (time.monotonic() - last)
            if pause > 0:
                time.sleep(pause)
            last = time.monotonic()
            r = session.get("http://127.0.0.1:8002/search", params={"q": item["query"], "top_k": 5}, timeout=(3, 10))
            elapsed = (time.monotonic() - last) * 1000
            r.raise_for_status()
            body = r.json()
            if body["generation_id"] != dataset["frozen_generation_id"] or body["ranking_version"] != "v2":
                raise ValueError("release_changed")
            recalled = [any(matches(h, target) for h in body["hits"]) for target in item["targets"]]
            if number:
                rows.append({"id": item["id"], "pass": number, "recall_at_5": sum(recalled) / len(recalled),
                             "ms": elapsed, "hits": [h["chunk_id"] for h in body["hits"]]})
        print(json.dumps({"retrieval_pass": number, "queries": len(queries)}), flush=True)
    if path.read_bytes() != raw:
        raise ValueError("historical_labels_changed")
    recall = sum(r["recall_at_5"] for r in rows) / len(rows)
    p95 = percentile([r["ms"] for r in rows], .95)
    report = {"at": timestamp(), "generation_id": dataset["frozen_generation_id"], "label_sha256": sha256(raw),
              "observations": len(rows), "recall_at_5": recall, "warm_p95_ms": p95, "rows": rows,
              "no_historical_recall_regression": recall >= .9666666666666667,
              "latency_pass": p95 <= 2000, "startup_ready_within_deadline": startup["ready"]["readiness"]["initialization_ms"] <= 60000,
              "public_answer_enabled": False, "pacing_excluded_from_http_latency": True}
    save_artifact(ROOT, "reports/reliability/readiness_retrieval.json", report)
    print(json.dumps({k: report[k] for k in report if k != "rows"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restart-preview", action="store_true")
    args = parser.parse_args()
    verify(restart=args.restart_preview)
