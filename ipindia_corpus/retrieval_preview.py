"""Isolated PostgreSQL retrieval preview; production state is never indexed here."""
import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from urllib.parse import urlsplit, urlunsplit

import psycopg
import requests
from psycopg import sql

from .indexer import SCHEMA, index
from .sync_runner import ROOT

STATE = ROOT / "reports/retrieval_preview.json"


def preview_dsn(name):
    parts = urlsplit(os.environ["DATABASE_URL"])
    if not name.startswith("ayurip_preview_") or not all(c in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in name):
        raise ValueError("unsafe_preview_database")
    if name == parts.path.lstrip("/"):
        raise ValueError("preview_must_be_separate")
    return urlunsplit((parts.scheme, parts.netloc, "/" + name, parts.query, parts.fragment))


@contextmanager
def environment(dsn):
    previous = os.environ["DATABASE_URL"]
    os.environ["DATABASE_URL"] = dsn
    try:
        yield
    finally:
        os.environ["DATABASE_URL"] = previous


def prepare(identifier):
    production = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    previous_state = json.loads(STATE.read_text()) if STATE.exists() else None
    reuse = previous_state and not previous_state.get("cleaned_up")
    name = previous_state["database"] if reuse else "ayurip_preview_" + uuid.uuid4().hex
    dsn = preview_dsn(name)
    before = production.execute("SELECT value FROM app_state WHERE key='active_generation'").fetchone()[0]
    if not reuse:
        production.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    try:
        with psycopg.connect(dsn, autocommit=True) as candidate:
            candidate.execute(SCHEMA)
            if not reuse:
                with production.cursor().copy("COPY embedding_cache TO STDOUT WITH (FORMAT BINARY)") as source:
                    with candidate.cursor().copy("COPY embedding_cache FROM STDIN WITH (FORMAT BINARY)") as destination:
                        for block in source:
                            destination.write(block)
        with environment(dsn):
            result = index(generation_id=identifier, publish_hint=False, record_events=False)
        with psycopg.connect(dsn, autocommit=True) as candidate:
            candidate.execute("INSERT INTO app_state(key,value) VALUES ('ranking_version','v2') ON CONFLICT(key) DO UPDATE SET value=excluded.value")
        after = production.execute("SELECT value FROM app_state WHERE key='active_generation'").fetchone()[0]
        if before != after:
            raise ValueError("production_changed_during_preview")
        state = {"database": name, "generation_id": identifier, "baseline_generation": before,
                 "production_unchanged": True, "index": result}
        STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
        return state
    except BaseException:
        if not reuse:
            production.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
        raise
    finally:
        production.close()


def evaluate_preview(benchmark_bundle=None):
    from .passage_evaluation import evaluate
    from .ranking_diagnostics import diagnose
    state = json.loads(STATE.read_text())
    dsn = preview_dsn(state["database"])
    environment_values = dict(os.environ, DATABASE_URL=dsn, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", AYURIP_RATE_LIMIT_ENABLED="0")
    reviewed = None
    if benchmark_bundle:
        from .benchmark_review import verify_bundle
        reviewed = verify_bundle(ROOT, benchmark_bundle)
    log_path = ROOT / "reports/retrieval_preview_server.log"
    with log_path.open("a", encoding="utf-8") as log:
        server = subprocess.Popen([sys.executable, "-m", "uvicorn", "ipindia_corpus.search_api:app",
                                   "--host", "127.0.0.1", "--port", "8001", "--no-proxy-headers"], env=environment_values,
                                  stdout=log, stderr=log, cwd=ROOT)
        try:
            for _ in range(50):
                if server.poll() is not None:
                    raise ValueError("preview_server_failed")
                try:
                    response = requests.get("http://127.0.0.1:8001/health", timeout=2)
                    if response.status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(.2)
            else:
                raise ValueError("preview_server_not_ready")
            reports = []
            reports.append(evaluate(api_url="http://127.0.0.1:8001", partial=not bool(reviewed),
                                    labels_path=ROOT / reviewed['sets']['original']['path'] if reviewed else None,
                                    report_name="reviewed_retrieval_original.json" if reviewed else "retrieval_preview_original.json"))
            reports.append(evaluate(api_url="http://127.0.0.1:8001", labels_path=ROOT / reviewed['sets']['expanded']['path'] if reviewed else ROOT / "tests/ipindia_retrieval_expanded.json",
                                    report_name="reviewed_retrieval_expanded.json" if reviewed else "retrieval_preview_expanded.json"))
            with environment(dsn):
                items = json.loads((ROOT / reviewed['sets']['original']['path'] if reviewed else ROOT / "tests/retrieval_queries.json").read_text())["queries"]
                details = [diagnose(q["query"], targets=q["targets"], version="v2")
                           for q in items if q["id"] in ("ipindia-08", "ipindia-09", "ipindia-10")]
            (ROOT / "reports" / ("ranking_reviewed.json" if reviewed else "ranking_candidate.json")).write_text(json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8")
            with psycopg.connect(os.environ["DATABASE_URL"]) as production:
                unchanged = production.execute("SELECT value FROM app_state WHERE key='active_generation'").fetchone()[0] == state["baseline_generation"]
            summary = {"production_unchanged": unchanged,
                       "evaluations": [{key: r[key] for key in ("generation_id", "queries", "recall_at_5", "p95_ms", "scope_pass", "ranking_version")} for r in reports]}
            (ROOT / "reports" / ("reviewed_retrieval_result.json" if reviewed else "retrieval_preview_result.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
            return summary
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)


def cleanup():
    state = json.loads(STATE.read_text())
    preview_dsn(state["database"])
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as production:
        production.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(state["database"])))
    state["cleaned_up"] = True
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "evaluate", "cleanup"))
    parser.add_argument("--generation-id")
    parser.add_argument("--benchmark-bundle", type=str)
    args = parser.parse_args()
    if args.action == "prepare" and not args.generation_id:
        parser.error("--generation-id required")
    result = prepare(args.generation_id) if args.action == "prepare" else evaluate_preview(args.benchmark_bundle) if args.action == "evaluate" else cleanup()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
