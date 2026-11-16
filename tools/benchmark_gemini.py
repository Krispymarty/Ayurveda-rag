"""Opt-in, synthetic local-HTTP provider comparison; never a legal certification.

Uses the existing private Passport with real hybrid retrieval and archive checks.
No service replacement, model download, corpus mutation or production promotion.
Provider HTTP calls happen only after explicit --live. Every attempted request,
including warmup, failure and interruption, has a unique retained artifact.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import statistics
import subprocess
import threading
import time

from tools import passport_demo as demo

ROOT = Path(__file__).resolve().parents[1]
ACTIVE = "20261014T204940358650Z"
PORT = 18003
ORIGIN = f"http://127.0.0.1:{PORT}"
VERSION = "gemini-passport-http-benchmark-v1"
CASES = ("traditional_ashwagandha_extract", "new_formulation", "novel_process",
         "abs_request", "hindi_intake", "unclear_description")
NOTICE = ("Synthetic, legally unreviewed functional/performance observations. "
          "Repeated attempts do not increase independent question count. "
          "Small-sample empirical percentiles are not latency/legal/security certification.")
HISTORICAL = {"source": "reports/next_stage/performance_results.md",
              "label": "historical Qwen serial observation, not a fresh paired control",
              "attempts": 18, "unique_questions": 6, "median_http_ms": 9017.04,
              "p95_http_ms": 10498.24}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def selected_scenarios(ids):
    catalog, digest = demo.load_scenarios()
    chosen = list(CASES) if ids is None else list(ids)
    known = {case["id"]: case for case in catalog["cases"]}
    if (not chosen or len(chosen) > 6 or len(chosen) != len(set(chosen))
            or any(identifier not in CASES or identifier not in known for identifier in chosen)):
        raise ValueError("choose_unique_frozen_representative_case_ids")
    return [known[identifier] for identifier in chosen], digest


def output_directory(value, *, root=ROOT):
    root = Path(root).resolve()
    path = Path(value).resolve()
    if path.exists() or not path.is_relative_to(root / "data/gemini_20261116"):
        raise ValueError("fresh_output_directory_under_data_gemini_20261116_required")
    return path


def nearest_rank(values, percentile):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile / 100 * len(ordered)) - 1)]


def distribution(values):
    values = [float(value) for value in values if type(value) in (int, float) and math.isfinite(value)]
    return {"n": len(values), "median": statistics.median(values) if values else None,
            "p95": nearest_rank(values, 95), "p99": nearest_rank(values, 99),
            "min": min(values) if values else None, "max": max(values) if values else None,
            "percentile_method": "empirical nearest-rank; median uses statistics.median"}


def model_observation(body, expected):
    model = body.get("model") if isinstance(body, dict) else None
    model = model if isinstance(model, dict) else {}
    name = model.get("name", model.get("model"))
    provider = model.get("provider", "ollama")
    completed = model.get("calls_completed")
    actual = type(completed) is int and completed > 0 and name == expected.get("model")
    if expected.get("provider") == "gemini":
        version = model.get("model_version", model.get("version"))
        actual = (actual and provider == "gemini" and isinstance(version, str)
                  and (version == name or version.startswith(name + "-")))
    else:
        digest = model.get("digest", model.get("model_digest"))
        actual = actual and provider == "ollama" and digest == expected.get("model_digest")
    return {"actual_model_run": bool(actual), "provider": provider, "model": name,
            "model_version": model.get("model_version", model.get("version")),
            "model_digest": model.get("digest", model.get("model_digest")),
            "completed_calls": completed,
            "basis": "server-reported completed calls and provider-specific identity; no Gemini weights digest asserted"}


def validate_response(body, case, verifier, expected):
    checked = demo.validate_result(body, case, verifier)
    # The retained demo validator deliberately requires the historical Qwen
    # digest. Replace ONLY that compatibility check, never a citation/quote or
    # structural failure, with the hosted provider's truthful identity contract.
    errors = [error for error in checked["errors"]
              if error != "research_case_has_no_completed_pinned_model_observation"]
    observed = model_observation(body, expected)
    if case["structural_expectations"]["requires_verified_quote"] and not observed["actual_model_run"]:
        errors.append("research_case_has_no_completed_provider_identity_observation")
    # Independent post-response original-byte check. The backend already checks
    # before/after inference; this benchmark check is outside measured HTTP time.
    if body.get("verified_evidence"):
        try:
            from ipindia_corpus.paths import archive_path
            documents, _ = verifier.generation(body.get("generation_id"))
            checked_originals = set()
            for item in body["verified_evidence"]:
                document = documents[item["document_id"]]
                identity = (document["raw_path"], document["sha256"])
                if identity not in checked_originals:
                    if sha(archive_path(verifier.root, document["raw_path"]).read_bytes()) != document["sha256"]:
                        errors.append("post_response_original_hash_mismatch")
                    checked_originals.add(identity)
        except Exception:
            errors.append("post_response_original_verification_unavailable")
    checked.update(errors=sorted(set(errors)), structural_and_integrity_checks_passed=not errors,
                   model_observation=observed)
    checked["outcome"] = ("dependency_or_model_unavailable" if body.get("status") == "unavailable"
                          else "validation_failed" if errors else "observed_with_verified_evidence"
                          if body.get("verified_evidence") else "safe_abstention_observed")
    return checked


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value)


def stage_totals(body, header_timings=None):
    timings = body.get("timings_ms", {})
    stages = {name: timings[key] for name, key in (
        ("intake_validation", "api_intake_validation"), ("intake_policy", "intake_policy"),
        ("readiness", "api_readiness"), ("planning", "issue_identification"),
        ("final_passport_archive", "final_passport_archive"), ("complete_service", "complete"))
        if numeric(timings.get(key))}
    performance = body.get("performance", {})
    for name, key in (("retrieval", "retrieval"), ("initial_archive", "archive_verification"),
                      ("packing", "packing"), ("selection", "selection"),
                      ("verification", "relevance_verification"), ("final_extract_archive", "final_citations")):
        values = [item.get("timings_ms", {}).get(key) for item in performance.get("issues", [])]
        if any(numeric(value) for value in values):
            stages[name] = sum(value for value in values if numeric(value))
    stages.update({key: value for key, value in (header_timings or {}).items() if numeric(value)})
    provider_totals = Counter()
    for item in performance.get("model_requests", []):
        for key, value in item.get("provider_timing", {}).items():
            if numeric(value):
                provider_totals[key] += value
    return stages, dict(provider_totals)


def aggregate(rows):
    measured = [row for row in rows if not row.get("warmup")]
    success = [row for row in measured if row.get("structural_pass") and row.get("http_status") == 200]
    questions = {row["case_id"] for row in measured}
    stages = sorted({key for row in measured for key in row.get("stages_ms", {})})
    tokens = sorted({key for row in measured for key in row.get("provider_totals", {})})
    invoking = [row for row in measured if row.get("calls_attempted", 0) > 0]
    window = None
    if measured and measured[0].get("started_at") and measured[-1].get("finished_at"):
        window = (datetime.fromisoformat(measured[-1]["finished_at"])
                  - datetime.fromisoformat(measured[0]["started_at"])).total_seconds()
    return {"attempts": len(measured), "unique_questions": len(questions),
            "warmup_attempts": len(rows) - len(measured),
            "structural_pass_attempts": len(success),
            "unique_questions_all_attempts_passed": sum(
                all(row.get("structural_pass") and row.get("http_status") == 200
                    for row in measured if row["case_id"] == question) for question in questions),
            "http_ms": distribution([row.get("http_ms") for row in measured]),
            "successful_http_ms": distribution([row.get("http_ms") for row in success]),
            "model_invoking_http_ms": distribution([row.get("http_ms") for row in invoking]),
            "failures": dict(Counter(row.get("outcome", "unknown") for row in measured if row not in success)),
            "calls_attempted": sum(row.get("calls_attempted", 0) for row in measured),
            "calls_completed": sum(row.get("calls_completed", 0) for row in measured),
            "observed_window_seconds_including_pacing": window,
            "throughput_requests_per_second_including_pacing": len(measured) / window if window and window > 0 else None,
            "per_question": {question: {
                "attempts": sum(row["case_id"] == question for row in measured),
                "structural_passes": sum(row["case_id"] == question for row in success),
                "http_ms": distribution([row.get("http_ms") for row in measured if row["case_id"] == question]),
                "evidence_counts": [row.get("evidence_count", 0) for row in measured if row["case_id"] == question],
                "outcomes": dict(Counter(row.get("outcome", "unknown") for row in measured if row["case_id"] == question))}
                for question in sorted(questions)},
            "stage_ms": {key: distribution([row.get("stages_ms", {}).get(key) for row in measured]) for key in stages},
            "provider_numeric_totals_per_request": {
                key: distribution([row.get("provider_totals", {}).get(key) for row in measured]) for key in tokens},
            "five_second_empirical_gate": bool(measured) and len(success) == len(measured)
                and nearest_rank([row["http_ms"] for row in measured], 95) <= 5000,
            "independent_quality_certification": False, "notice": NOTICE}


def server_timing(value):
    result = {}
    for part in (value or "").split(","):
        name, separator, duration = part.strip().partition(";dur=")
        if separator and name in {"intake", "readiness", "serialization"}:
            try:
                number = float(duration)
                if math.isfinite(number) and number >= 0:
                    result[name] = number
            except ValueError:
                pass
    return result


def runtime_metadata(root):
    paths = ("ipindia_corpus/retrieval.py", "ipindia_corpus/reliable_answers.py",
             "ipindia_corpus/inference.py", "ipindia_corpus/passport/service.py",
             "data/model.lock.json")
    hashes = {name: sha((root / name).read_bytes()) for name in paths if (root / name).is_file()}
    revision = None
    try:
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                           stderr=subprocess.DEVNULL, timeout=5).decode().strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return {"python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(), "logical_cpus": os.cpu_count(),
            "git_revision": revision, "file_hashes": hashes,
            "hardware_limit": "Provider hardware is remote/unknown; local GPU/RAM are not inferred from availability."}


class BudgetProvider:
    """One provider-wide chat budget includes warmup and failed attempts."""
    def __init__(self, provider, maximum=140):
        self.provider, self.maximum, self.attempted = provider, maximum, 0

    @property
    def metadata(self):
        from ipindia_corpus.inference import provider_identity
        return provider_identity(self.provider)

    def chat_with_deadline(self, fields, deadline):
        if self.attempted >= self.maximum:
            from ipindia_corpus.llm_experiment import ExperimentFailure
            raise ExperimentFailure('timeout')
        self.attempted += 1
        bounded = getattr(self.provider, 'chat_with_deadline', None)
        return bounded(fields, deadline) if callable(bounded) else self.provider.chat(fields)

    def chat(self, fields):
        from ipindia_corpus.llm_experiment import ExperimentFailure
        if self.attempted >= self.maximum:
            raise ExperimentFailure("benchmark_call_budget_exhausted")
        self.attempted += 1
        return self.provider.chat(fields)


@contextmanager
def private_server(engine):
    import uvicorn
    from ipindia_corpus.passport.api import create_app
    # Reserve the exact port first. Never replace/stop an existing service.
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind(("127.0.0.1", PORT))
        listener.listen(16)
    except BaseException:
        listener.close()
        raise
    app = create_app(engine=engine, enabled=True, port=PORT)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PORT, access_log=False,
                                         log_level="error", log_config=None))
    worker = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True,
                              name="gemini-benchmark-private-http")
    worker.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and worker.is_alive() and time.monotonic() < deadline:
            time.sleep(.02)
        if not server.started:
            raise RuntimeError("isolated_private_http_start_failed")
        yield ORIGIN
    finally:
        server.should_exit = True
        worker.join(10)
        listener.close()
        if worker.is_alive():
            raise RuntimeError("isolated_private_http_did_not_stop")


def request_attempt(session, origin, token, case, *, provider, pass_number, warmup, folder, verifier):
    identifier = ("warmup" if warmup else f"pass-{pass_number}") + "-" + case["id"]
    started = time.perf_counter()
    write_new(folder / f"{identifier}.started.json", {"started_at": now(), "case_id": case["id"],
               "warmup": warmup, "pass": pass_number, "state": "attempted", "retries": 0})
    row = {"case_id": case["id"], "started_at": now(), "warmup": warmup, "pass": pass_number, "http_status": None,
           "http_ms": None, "structural_pass": False, "calls_attempted": 0, "calls_completed": 0,
           "stages_ms": {}, "provider_totals": {}, "outcome": "transport_unavailable"}
    body = None
    interrupted = False
    try:
        payload = {**case["intake"], "acknowledged_external_processing": provider["external_processing"]}
        response = session.post(origin + "/api/passport", json=payload,
            headers={"Origin": origin, "X-Passport-Token": token}, timeout=(5, 180), allow_redirects=False)
        row["http_ms"] = round((time.perf_counter() - started) * 1000, 3)
        row["http_status"] = response.status_code
        if len(response.content) > 2_000_000:
            raise ValueError("benchmark_response_too_large")
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError("benchmark_response_not_object")
        if "overall_status" not in body:
            row.update(outcome="api_request_unavailable", reason=body.get("reason") if isinstance(body.get("reason"), str) else "unknown")
        else:
            checked = validate_response(body, case, verifier, provider)
            row.update(outcome=checked["outcome"], structural_pass=checked["structural_and_integrity_checks_passed"],
                       checks=checked, response_status=body.get("status"),
                       model=body.get("model"), generation_id=body.get("generation_id"),
                       evidence_count=len(body.get("verified_evidence", [])),
                       issue_ids=[item.get("issue_id") for item in body.get("issues", [])],
                       evidence_bindings=[{"document_id": item.get("document_id"), "chunk_id": item.get("chunk_id"),
                                           "raw_sha256": item.get("raw_sha256"),
                                           "quote_sha256": sha(item.get("exact_quote", "").encode())}
                                          for item in body.get("verified_evidence", [])],
                       calls_attempted=body.get("model", {}).get("calls_attempted", 0),
                       calls_completed=body.get("model", {}).get("calls_completed", 0))
            stages, provider_totals = stage_totals(body, server_timing(response.headers.get("Server-Timing")))
            row.update(stages_ms=stages, provider_totals=provider_totals,
                       configuration=body.get("performance", {}).get("configuration", {}),
                       model_request_bindings=body.get("performance", {}).get("model_requests", []))
    except KeyboardInterrupt:
        row["outcome"], interrupted = "interrupted", True
    except Exception as exc:
        # Never persist transport exceptions: URLs/headers can contain secrets.
        row["error_type"] = type(exc).__name__
    finally:
        if row["http_ms"] is None:
            row["http_ms"] = round((time.perf_counter() - started) * 1000, 3)
        row["finished_at"] = now()
        if body is not None:
            # Explicit synthetic-only artifacts, not ordinary input logging.
            raw = json.dumps(body, ensure_ascii=False, sort_keys=True).encode()
            row["response_sha256"] = sha(raw)
            write_new(folder / f"{identifier}.response.json", body)
        write_new(folder / f"{identifier}.result.json", row)
    if interrupted:
        raise KeyboardInterrupt
    return row


def run_provider(kind, cases, passes, folder):
    import requests
    from ipindia_corpus.inference import make_provider, provider_identity
    from ipindia_corpus.passport.service import LocalSearch, PassportEngine, PublishedVerifier
    result = {"provider": kind, "state": "initializing", "attempts": [], "started_at": now()}
    write_new(folder / "started.json", {"provider": kind, "started_at": result["started_at"]})
    initialized = time.monotonic()
    try:
        wrapped = BudgetProvider(make_provider(kind))
        identity = provider_identity(wrapped)
        result["identity"] = identity
        search = LocalSearch()
        gid = search.status()["generation_id"]
        if gid != ACTIVE:
            raise ValueError("protected_active_generation_required")
        engine = PassportEngine(search, PublishedVerifier(ROOT, gid), provider=wrapped)
        verifier = demo.CitationVerifier(ROOT)
        verifier.generation(gid)
        result["initialization_ms"] = round((time.monotonic() - initialized) * 1000, 3)
        with private_server(engine) as origin, requests.Session() as session:
            session.trust_env = False
            response = session.get(origin + "/api/session", headers={"Origin": origin}, timeout=(3, 10), allow_redirects=False)
            response.raise_for_status()
            state = response.json()
            if state.get("ready") is not True or not isinstance(state.get("csrf_token"), str):
                raise ValueError("private_session_not_ready")
            token = state["csrf_token"]
            schedule = [(0, cases[0], True)] + [(number, case, False) for number in range(1, passes + 1) for case in cases]
            last_started = None
            failed = 0
            result["state"] = "running"
            for number, case, warmup in schedule:
                if last_started is not None:
                    # Preserve the ordinary 10/min private rate limit. Delays
                    # are outside measured HTTP latency and are recorded below.
                    time.sleep(max(0, 6.1 - (time.monotonic() - last_started)))
                last_started = time.monotonic()
                row = request_attempt(session, origin, token, case, provider=identity,
                                      pass_number=number, warmup=warmup, folder=folder, verifier=verifier)
                result["attempts"].append(row)
                failed = failed + 1 if row["http_status"] != 200 or row.get("response_status") == "unavailable" else 0
                if failed >= 3 or wrapped.attempted >= 140:
                    result["state"] = "stopped_bounded_provider_failures_or_call_budget"
                    break
            else:
                result["state"] = "completed"
            result["planned_http_attempts"] = len(schedule)
            result["unexecuted_http_attempts"] = len(schedule) - len(result["attempts"])
            result["provider_calls_attempted_including_warmup"] = wrapped.attempted
            result["final_identity"] = provider_identity(wrapped)
    except KeyboardInterrupt:
        result["state"] = "interrupted"
    except Exception as exc:
        result.update(state="blocked", blocker_type=type(exc).__name__)
        # Stable configuration/provider reasons are safe; arbitrary exception
        # text is not. No API key or URL error details enter the report.
        if str(exc) in {"gemini_key_missing", "gemini_configuration_invalid", "protected_active_generation_required"}:
            result["blocker"] = str(exc)
    result["summary"] = aggregate(result["attempts"])
    result["finished_at"] = now()
    write_new(folder / "summary.json", result)
    return result


def markdown(report, output):
    lines = [f"\n## {report['started_at']} — {output.name}", "", NOTICE, "",
             "Actual loopback HTTP; serial planner/selection/verification; unchanged ranker v2 and archived IP India corpus.",
             "Rate-limit pacing: >=6.1 seconds between request starts, outside measured latency. No retries.",
             "Sequential provider order can include cache/time/thermal effects. Stage medians do not sum to request median.", "",
             "| Provider | State | Unique questions | Attempts | Structural pass | HTTP median ms | HTTP empirical p95 ms | Calls completed/attempted |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for provider in report["providers"]:
        s = provider["summary"]
        lines.append(f"| {provider['provider']} | {provider['state']} | {s['unique_questions']} | {s['attempts']} | {s['structural_pass_attempts']} | {s['http_ms']['median']} | {s['http_ms']['p95']} | {s['calls_completed']}/{s['calls_attempted']} |")
    lines += ["", "Historical Qwen control (separate retained run): 18 attempts / six questions, median 9,017.04 ms, empirical p95 10,498.24 ms. This is not a fresh paired baseline.",
              "", "### Stage medians (ms; request-level sums across topics)", "",
              "| Stage | " + " | ".join(p["provider"] for p in report["providers"]) + " |",
              "|---|" + "---:|" * len(report["providers"])]
    keys = sorted({key for provider in report["providers"] for key in provider["summary"]["stage_ms"]})
    for key in keys:
        lines.append("| " + key + " | " + " | ".join(str(p["summary"]["stage_ms"].get(key, {}).get("median", "unavailable")) for p in report["providers"]) + " |")
    lines += ["", f"Raw observations: `{output.relative_to(ROOT).as_posix()}`. All attempts, warmups and failures are retained individually.",
              "The five-second target is not certified by a small-sample pass; failed target measurements remain failures. Human legal relevance remains unreviewed.", ""]
    return "\n".join(lines)


def run_benchmark(*, provider="gemini", passes=3, case_ids=None, output=None, live=False):
    if live is not True:
        raise ValueError("explicit_live_opt_in_required")
    if provider not in {"gemini", "ollama", "both"} or type(passes) is not int or not 1 <= passes <= 3:
        raise ValueError("bounded_provider_and_one_to_three_passes_required")
    cases, scenario_sha = selected_scenarios(case_ids)
    target = output or ROOT / "data/gemini_20261116" / datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S%fZ")
    target = output_directory(target)
    target.mkdir(parents=True)
    report = {"version": VERSION, "started_at": now(), "notice": NOTICE,
              "synthetic": True, "legal_relevance_review": "unreviewed", "retries": 0,
              "scenario_sha256": scenario_sha, "ordered_case_ids": [case["id"] for case in cases],
              "passes": passes, "runtime": runtime_metadata(ROOT), "providers": [],
              "historical_secondary_control": HISTORICAL, "provider_order": ["ollama", "gemini"] if provider == "both" else [provider]}
    write_new(target / "protocol.json", {key: value for key, value in report.items() if key != "providers"})
    for kind in report["provider_order"]:
        result = run_provider(kind, cases, passes, target / kind)
        report["providers"].append(result)
        if result["state"] == "interrupted":
            break
    report["finished_at"] = now()
    write_new(target / "comparison.json", report)
    report_path = ROOT / "reports/gemini_20261116/performance.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(markdown(report, target))
    return target, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="explicit synthetic external-inference opt-in")
    parser.add_argument("--provider", choices=("gemini", "ollama", "both"), default="gemini")
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        output, report = run_benchmark(**vars(args))
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({"output": str(output), "providers": [
        {"provider": item["provider"], "state": item["state"], "summary": item["summary"]}
        for item in report["providers"]]}, ensure_ascii=False))
    return 0 if all(item["state"] == "completed" for item in report["providers"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
