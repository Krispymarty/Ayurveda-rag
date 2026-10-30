"""Local regression evidence. No source capture, promotion or Docker reset."""
import argparse
import io
import json
import os
from importlib.metadata import version
import subprocess
import sys
import time
from urllib.parse import urlsplit, urlunsplit
import uuid

from ipindia_corpus.llm_experiment import ROOT, percentile, timestamp
from .common import DEST, GENERATION, protected, read, sha, write


def command(args, environment=None):
    started = time.monotonic()
    p = subprocess.run(args, cwd=ROOT, env=environment, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return {"command": args[1:], "exit_code": p.returncode, "ms": (time.monotonic() - started) * 1000,
            "output": p.stdout + p.stderr}


def offline():
    test = command([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
    lint = command([sys.executable, "-m", "ruff", "check", "ipindia_corpus", "tools", "tests", "--output-format", "concise"])
    result = {"at": timestamp(), "passed": test["exit_code"] == lint["exit_code"] == 0,
              "tests": test, "lint": lint, "scope": "offline structural/regression tests, not independent legal relevance review"}
    write("offline_checks.json", result)
    # Retain specific test identities rather than claiming the entire suite
    # automatically exercised every requested security property.
    names = [line.strip() for line in test["output"].splitlines() if line.startswith("test_") and any(token in line.casefold() for token in
             ("attack", "host", "redirect", "mutation", "changed_archive", "unicode", "concurrent", "rate", "forged", "malformed", "incomplete", "provider", "policy", "subset", "overflow"))]
    write("security_regression.json", {"offline_passed": test["exit_code"] == 0, "recorded_security_related_tests": names,
                                      "public_security_certification": False, "scope": "finite local tests; model resistance beyond these cases is unestablished",
                                      "new_security_regressions_observed": None if test["exit_code"] else False})
    return {"passed": result["passed"], "tests_exit": test["exit_code"], "lint_exit": lint["exit_code"]}


def sync():
    import unittest
    sys.path.insert(0, str(ROOT / "tests"))
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_milestone3_sync.py")
    stream = io.StringIO()
    class Result(unittest.TextTestResult):
        records = []
        def addSuccess(self, test):
            super().addSuccess(test)
            self.records.append({"id": test.id(), "passed": True})
        def addFailure(self, test, err):
            super().addFailure(test, err)
            self.records.append({"id": test.id(), "passed": False})
        def addError(self, test, err):
            super().addError(test, err)
            self.records.append({"id": test.id(), "passed": False})
    result = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=Result).run(suite)
    report = {"passed": result.wasSuccessful(), "executed": result.testsRun, "scenarios": Result.records,
              "output": stream.getvalue(), "live_requests": 0,
              "event_aliases": {"VERSION_REPLACED": "ledger NEW_VERSION with explicit verified version", "FETCH_FAILED": "capture FAILED/partial", "PARSE_FAILED": "quarantine + FAILED/partial", "PARTIAL_CAPTURE": "partial; no accepted snapshot"}}
    write("sync_offline_validation.json", report)
    return report


def postgres():
    import psycopg
    from psycopg import sql
    dsn = os.environ["DATABASE_URL"]
    parsed = urlsplit(dsn)
    name = "ayurip_check_m3_" + uuid.uuid4().hex
    with psycopg.connect(dsn, autocommit=True) as db:
        before = dict(db.execute("SELECT key,value FROM app_state").fetchall())
        db.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        try:
            environment = dict(os.environ, AYURIP_TEST_DATABASE_URL=urlunsplit((parsed.scheme, parsed.netloc, "/" + name, parsed.query, parsed.fragment)))
            checked = command([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_postgres_integration.py", "-v"], environment)
            after = dict(db.execute("SELECT key,value FROM app_state").fetchall())
        finally:
            db.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
    result = {"passed": checked["exit_code"] == 0 and before == after, "database": name,
              "production_unchanged": before == after, "checks": checked,
              "scope": "disposable indexing-failure and rollback checks; no production mutation"}
    write("postgres_checks.json", result)
    return result


def retrieval():
    from tools.reliability_v2_runtime import session
    from ipindia_corpus.passage_evaluation import matches
    client = session()
    # Host port 8002 is the isolated database, not the active production service.
    manifest = read(ROOT / "data/benchmarks/ipindia-reviewed-20261015-v1/manifest.json")
    bundle = manifest["sets"]["expanded"]
    raw = (ROOT / bundle["path"]).read_bytes()
    if sha(raw) != bundle["sha256"]:
        raise ValueError("labels_changed")
    questions = json.loads(raw)["queries"]
    client.get("http://127.0.0.1:8002/search", params={"q": questions[0]["query"], "top_k": 5}, timeout=20).raise_for_status()
    rows = []
    for n in range(1, 4):
        for q in questions:
            started = time.monotonic()
            response = client.get("http://127.0.0.1:8002/search", params={"q": q["query"], "top_k": 5}, timeout=20)
            elapsed = (time.monotonic() - started) * 1000
            try:
                response.raise_for_status()
                data = response.json()
                if data["generation_id"] != GENERATION:
                    raise ValueError("generation_changed")
                recall = sum(any(matches(h, target) for h in data["hits"]) for target in q["targets"]) / len(q["targets"])
                row = {"id": q["id"], "pass": n, "ms": elapsed, "recall_at_5": recall, "hit_ids": [h["chunk_id"] for h in data["hits"]]}
            except Exception as exc:
                row = {"id": q["id"], "pass": n, "ms": elapsed, "recall_at_5": 0, "error_kind": type(exc).__name__}
            rows.append(row)
    recall = sum(r["recall_at_5"] for r in rows) / len(rows)
    p95 = percentile([r["ms"] for r in rows], .95)
    result = {"passed": recall >= 29 / 30 and p95 <= 2000, "recall_at_5": recall, "warm_p95_ms": p95,
              "previous_p95_ms": 538.0, "rows": rows, "label_sha256": bundle["sha256"], "generation_id": GENERATION,
              "ranking_version": "v2", "failed_attempts_retried": False}
    write("retrieval_checks.json", result)
    return {k: v for k, v in result.items() if k != "rows"}


def final_state():
    import requests
    client = requests.Session()
    client.trust_env = False
    health = client.get("http://127.0.0.1:8000/health", timeout=10)
    answer = client.post("http://127.0.0.1:8000/answer", json={"question": "Quote an archived provision."}, timeout=10)
    status = client.get("http://127.0.0.1:8000/sources", timeout=10)
    result = {"at": timestamp(), "protected_inputs": protected(), "health_http": health.status_code, "health": health.json(),
              "public_answer_http": answer.status_code, "public_answer_reason": answer.json().get("reason"),
              "sources_http": status.status_code, "sources": status.json()}
    db = command(["docker", "compose", "exec", "-T", "db", "psql", "-U", "ayurip", "-d", "ayurip", "-At", "-c", "SELECT key,value FROM app_state ORDER BY key;"])
    result["database_binding"] = db
    result["production_binding_unchanged"] = db["exit_code"] == 0 and db["output"].strip() == read(ROOT / DEST / "preflight.json")["database_binding"]["output"]
    volume = command(["docker", "volume", "inspect", "sih_pgdata", "--format", "{{json .CreatedAt}}"])
    result["database_volume"] = {"name": "sih_pgdata", **volume}
    result["volume_preserved"] = volume["exit_code"] == 0 and volume["output"].strip() == read(ROOT / DEST / "preflight.json")["database_volume"]["created_at"]
    result["services"] = command(["docker", "ps", "-a", "--format", "{{.Names}} {{.State}}"])
    result["passed"] = (result["protected_inputs"]["passed"] and health.status_code == 200
                        and result["health"].get("generation_id") == GENERATION
                        and result["production_binding_unchanged"] and result["volume_preserved"]
                        and answer.status_code == 503 and result["public_answer_reason"] == "three_source_not_certified")
    write("final_state.json", result)
    return result


def runtime():
    import platform
    from tools.reliability_v2_runtime import session
    packages = {}
    for name in ("fastapi", "pydantic", "requests", "psutil", "PyMuPDF"):
        try:
            packages[name] = version(name)
        except Exception:
            packages[name] = None
    client = session()
    commands = {
        "hardware": ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json -Compress"],
        "gpu": ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        "docker": ["docker", "info", "--format", '{{json .NCPU}} {{json .MemTotal}} {{json .ServerVersion}}'],
        "ollama": ["ollama", "--version"]}
    result = {"at": timestamp(), "python": platform.python_version(), "packages": packages,
              "commands": {key: command(args) for key, args in commands.items()},
              "loaded_models": client.get("http://127.0.0.1:11434/api/ps", timeout=5).json(),
              "configuration": read(ROOT / "reports/milestone3/candidate.json"),
              "embedding_model": read(ROOT / "data/model.lock.json"),
              "ranking_code_sha256": sha((ROOT / "ipindia_corpus/retrieval.py").read_bytes()),
              "resource_sampler_actual_interval_ms": 2000,
              "legacy_resource_summary_interval_field": "Experiment summaries inherited a fixed 500ms label from resource_summary; the actual sampler interval in frozen experiment source is 2000ms. Sample values are unchanged."}
    write("runtime.json", result)
    return {"at": result["at"], "python": result["python"], "packages": packages}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("offline", "sync", "postgres", "retrieval", "final-state", "runtime"))
    args = parser.parse_args()
    result = {"offline": offline, "sync": sync, "postgres": postgres, "retrieval": retrieval, "final-state": final_state, "runtime": runtime}[args.action]()
    print(json.dumps(result, ensure_ascii=True))
