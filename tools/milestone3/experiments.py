"""Fixed development-only HTTP experiments with irreversible attempt intents."""
import argparse
from collections import defaultdict
import json
import os
import socket
import subprocess
import sys
import time

from ipindia_corpus.llm_experiment import ROOT, ResourceSampler, canonical, percentile, resource_summary, timestamp
from ipindia_corpus.locking import file_lock
from ipindia_corpus.paths import immutable_write
from ipindia_corpus.private_preview import ArchiveVerifier
from ipindia_corpus.reliable_answers import SpanProvider, configuration
from tools.benchmark_reliability import summarize
from tools.reliability_v2 import cases
from tools.reliability_v2_runtime import audit_citations, score_result, session
from .candidate import VARIANTS, answer
from .common import DATA, DEST, GENERATION, protected, read, sha, write


def fingerprint():
    checked = protected()
    if not checked["passed"]:
        raise ValueError("protected_input_changed")
    paths = ["tools/milestone3/" + name + ".py" for name in ("__init__", "common", "references", "candidate", "experiments")]
    paths += ["tests/test_milestone3.py"]
    preflight = read(ROOT / DEST / "preflight.json")
    sources = {name: sha((ROOT / name).read_bytes()) for name in paths}
    payload = {"version": "milestone3-development-2-admission", "sources": sources,
               "protected_snapshot_sha256": preflight["snapshot_sha256"], "generation_id": GENERATION,
               "ranking_version": "v2", "configuration": configuration(), "variants": VARIANTS,
               "top_k": 5, "normalize": False, "expand": False, "split": "development",
               "development_sha256": sha(canonical(cases()["development"])),
               "promotion_allowed": False, "independent_evaluation": False}
    return {**payload, "candidate_sha256": sha(canonical(payload))}


def freeze():
    receipt = fingerprint()
    folder = ROOT / DATA / "candidates" / receipt["candidate_sha256"]
    immutable_write(folder / "candidate.json", canonical(receipt))
    for name in receipt["sources"]:
        immutable_write(folder / "sources" / name, (ROOT / name).read_bytes())
    write("candidate.json", receipt)
    return receipt


def dependency_health(client):
    response = client.get("http://127.0.0.1:8002/health", timeout=(2, 3))
    if response.status_code != 200 or response.json().get("generation_id") != GENERATION:
        raise ValueError("isolated_search_not_ready")
    return response.json()


def serve(variant):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    import uvicorn
    from ipindia_corpus.http_security import configure_security
    from ipindia_corpus.search_api import AnswerRequest
    os.environ["AYURIP_RATE_LIMIT_ENABLED"] = "0"  # this opt-in benchmark process only
    receipt = fingerprint()
    verifier, provider, client = ArchiveVerifier(ROOT), SpanProvider(), session()
    app = FastAPI(docs_url=None, redoc_url=None)
    configure_security(app)

    def search(question, top_k=5):
        response = client.get("http://127.0.0.1:8002/search", params={"q": question, "top_k": top_k}, timeout=(3, 15))
        response.raise_for_status()
        return response.json()

    @app.get("/health")
    def health():
        if fingerprint() != receipt:
            return JSONResponse(status_code=503, content={"reason": "candidate_configuration_changed"})
        try:
            dependency_health(client)
        except Exception:
            return JSONResponse(status_code=503, content={"reason": "isolated_search_not_ready"})
        return {"status": "private_evaluation", "candidate_sha256": receipt["candidate_sha256"], "variant": variant}

    @app.post("/answer")
    def respond(request: AnswerRequest):
        if request.top_k != 5 or request.source is not None or request.domain is not None:
            return JSONResponse(status_code=400, content={"reason": "frozen_unfiltered_experiment"})
        # Do not hash hundreds of reports on each timed request; immutable source
        # bindings and generation archive checks are enforced independently.
        if any(sha((ROOT / name).read_bytes()) != expected for name, expected in receipt["sources"].items()):
            return JSONResponse(status_code=503, content={"reason": "candidate_configuration_changed"})
        result = answer(request.question, search, verifier, variant=variant, provider=provider)
        result["candidate_sha256"] = receipt["candidate_sha256"]
        return JSONResponse(status_code=503 if result["status"] == "unavailable" else 200, content=result)

    uvicorn.run(app, host="127.0.0.1", port=8001, proxy_headers=False)


def raw_selection_score(case, result):
    from ipindia_corpus.span_selection import build_plan, resolve_selection
    from ipindia_corpus.llm_scoring import score_case
    from .review import resolved_stages
    observation = next((r for r in result.get("model_observations", []) if r["stage"] == "selection"), None)
    if not observation:
        return {"provider_called": False, "passed": None, "backend_reason": result.get("reason")}
    packed = resolved_stages(result)["packed"]
    try:
        resolved = resolve_selection(observation["raw"], packed, build_plan(packed))
        return {"provider_called": True, **score_case({**case, "evidence": packed}, canonical(resolved["expanded_selection"]).decode(), done_reason="stop")}
    except (ValueError, TypeError, KeyError):
        return {"provider_called": True, "passed": False, "reason": "malformed_selection"}


def evaluate(variant, passes=3):
    receipt = read(ROOT / DEST / "candidate.json")
    if fingerprint() != receipt:
        raise ValueError("candidate_not_frozen")
    items = cases()["development"]
    folder = ROOT / DATA / "observations" / receipt["candidate_sha256"] / variant
    plan = {"candidate_sha256": receipt["candidate_sha256"], "variant": variant, "passes": passes,
            "cases_sha256": sha(canonical(items)), "unique_questions": len(items), "created_at": timestamp()}
    if (folder / "plan.json").exists():
        old = read(folder / "plan.json")
        if any(old[key] != plan[key] for key in ("candidate_sha256", "variant", "passes", "cases_sha256")):
            raise ValueError("experiment_plan_changed")
    else:
        immutable_write(folder / "plan.json", canonical(plan))
    client, verifier = session(), ArchiveVerifier(ROOT)
    dependency_health(client)
    # Warmup is separately retained and never included in timed slots.
    if not (folder / "warmup.json").exists():
        immutable_write(folder / "warmup-intent.json", canonical({"at": timestamp()}))
        started = time.monotonic()
        try:
            response = client.post("http://127.0.0.1:8001/answer", json={"question": items[0]["question"]}, timeout=(5, 60))
            warm = {"status_code": response.status_code, "ms": (time.monotonic() - started) * 1000, "result": response.json()}
        except Exception as exc:
            warm = {"error_kind": type(exc).__name__}
        immutable_write(folder / "warmup.json", canonical(warm))
    warm = read(folder / "warmup.json")
    if warm.get("status_code") != 200 or warm.get("result", {}).get("status") == "unavailable":
        raise ValueError("warmup_dependency_failed_no_timed_admission")
    slots = [(n, c) for n in range(1, passes + 1) for c in items]
    with ResourceSampler(interval=2) as sampler:
        for n, case in slots:
            slot = folder / "slots" / f"{n}-{case['id']}"
            if slot.with_suffix(".intent.json").exists():
                continue  # missing result is permanently failed/interrupted, never retried
            immutable_write(slot.with_suffix(".intent.json"), canonical({"at": timestamp(), "case_sha256": sha(canonical(case))}))
            started = time.monotonic()
            try:
                response = client.post("http://127.0.0.1:8001/answer", json={"question": case["question"]}, timeout=(5, 60))
                elapsed = (time.monotonic() - started) * 1000
                result = response.json()
                result["evaluation_citation_audit"] = audit_citations(result, verifier)
                row = {"case_id": case["id"], "pass": n, "at": timestamp(), "http_status": response.status_code,
                       "candidate_sha256": receipt["candidate_sha256"], "result": result,
                       "http_or_pipeline_ms": elapsed, "system_score": score_result(case, result),
                       "model_called": bool(result.get("model_observations")), "raw_selection_score": raw_selection_score(case, result)}
            except Exception as exc:
                row = {"case_id": case["id"], "pass": n, "http_or_pipeline_ms": (time.monotonic() - started) * 1000,
                       "result": {"status": "unavailable", "claims": [], "reason": "interrupted_or_dependency_failure"},
                       "system_score": {"passed": False, "failures": ["ATTEMPT_FAILED"], "target_coverage": 0},
                       "error_kind": type(exc).__name__, "model_called": None, "raw_selection_score": None}
            immutable_write(slot.with_suffix(".row.json"), canonical(row))
            print(json.dumps({"variant": variant, "pass": n, "case": case["id"], "passed": row["system_score"]["passed"]}), flush=True)
    rows = []
    for n, case in slots:
        path = folder / "slots" / f"{n}-{case['id']}.row.json"
        rows.append(read(path) if path.exists() else {"case_id": case["id"], "pass": n, "result": {"claims": [], "reason": "interrupted_attempt"},
                    "system_score": {"passed": False, "failures": ["INTERRUPTED_ATTEMPT"], "target_coverage": 0}, "model_called": None})
    summary = summarize(rows, items)
    by_question = defaultdict(list)
    for row in rows:
        by_question[row["case_id"]].append(row)
    summary.update(unique_question_all_attempts_success=sum(all(r["system_score"]["passed"] for r in rs) for rs in by_question.values()) / len(by_question),
                   required_evidence_coverage=sum(r["system_score"].get("target_coverage", 0) for r in rows if next(c for c in items if c["id"] == r["case_id"])["expected"]["policy"] == "quote_targets") / (15 * passes),
                   citation_support_correctness=None, citation_redundancy=None, unsupported_citation_rate=None,
                   semantic_metric_reason="pending_independent_human_review", stage_latency_ms={})
    for key in ("retrieval", "selection", "verification", "complete"):
        values = [r["result"].get("timings_ms", {}).get(key) for r in rows]
        values = [v for v in values if isinstance(v, (int, float))]
        summary["stage_latency_ms"][key] = {"p50": percentile(values, .5), "p95": percentile(values, .95)}
    invoked = [r["http_or_pipeline_ms"] for r in rows if r.get("model_called")]
    summary["model_invoking_http_ms"] = {"p50": percentile(invoked, .5), "p95": percentile(invoked, .95)}
    report = {"candidate": receipt, "variant": variant, "split": "development", "passes": passes,
              "rows": rows, "summary": summary, "resources": resource_summary(sampler.samples),
              "warmup": read(folder / "warmup.json"), "independent_certification": False}
    write(f"experiment-{variant}.json", report)
    return summary


def matrix(passes=3):
    freeze()
    with file_lock(ROOT / "data/llm/citation_selection.lock"):
        for variant in VARIANTS:
            with socket.socket() as check:
                if check.connect_ex(("127.0.0.1", 8001)) == 0:
                    raise ValueError("private_preview_port_already_owned")
            log_path = ROOT / DEST / f"preview-{variant}.log"
            with log_path.open("ab") as log:
                started = time.monotonic()
                child = subprocess.Popen([sys.executable, "-m", "tools.milestone3.experiments", "serve", "--variant", variant],
                                         cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                try:
                    client = session()
                    while time.monotonic() - started < 60:
                        if child.poll() is not None:
                            raise ValueError("owned_preview_startup_failed")
                        try:
                            response = client.get("http://127.0.0.1:8001/health", timeout=2)
                            if response.ok and response.json().get("variant") == variant:
                                break
                        except Exception:
                            pass
                        time.sleep(.5)
                    else:
                        raise ValueError("owned_preview_initialization_deadline")
                    write(f"startup-{variant}.json", {"ms": (time.monotonic() - started) * 1000, "separate_from_warm_requests": True})
                    print(json.dumps({"variant": variant, "summary": evaluate(variant, passes)}), flush=True)
                finally:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "serve", "matrix"))
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="A")
    parser.add_argument("--passes", type=int, choices=(1, 3), default=3)
    args = parser.parse_args()
    if args.command == "serve":
        serve(args.variant)
    elif args.command == "matrix":
        matrix(args.passes)
    else:
        print(json.dumps(freeze(), ensure_ascii=True))
