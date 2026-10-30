"""Checkpointed controlled and HTTP end-to-end reliability evaluation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import uuid

import requests

from ipindia_corpus.llm_experiment import (
    ROOT, ResourceSampler, canonical, percentile, prepare_cohort,
    protected_state, resource_summary, save_artifact, sha256, timestamp,
)
from ipindia_corpus.llm_scoring import score_case
from ipindia_corpus.locking import file_lock
from ipindia_corpus.paths import immutable_write
from ipindia_corpus.private_preview import ArchiveVerifier
from ipindia_corpus.reliable_answers import MODEL, DIGEST, SpanProvider, answer, configuration
from tools.freeze_llm_benchmark import load_frozen
from tools.freeze_reliability_benchmark import VERSION, load
from tools.report_llm_results import verified_artifact

FILES = ["ipindia_corpus/" + name + ".py" for name in (
    "reliable_answers", "request_policy", "span_selection", "citation_selection", "grounded_answers",
    "private_preview", "readiness", "search_api", "llm_experiment", "llm_scoring", "indexer", "paths", "http_security")]
FILES += ["tools/benchmark_reliability.py", "tools/freeze_reliability_benchmark.py", "web/app.js"]


def binding(root):
    root = Path(root)
    return {"configuration": configuration(), "code_sha256": {p: sha256((root / p).read_bytes()) for p in FILES},
            "benchmark_manifest_sha256": sha256((root / "data/benchmarks" / VERSION / "manifest.json").read_bytes()),
            "protected_state": protected_state(root, "20261014T204940358650Z")}


def freeze_candidate(root=ROOT):
    root = Path(root)
    payload = binding(root)
    checksum = sha256(canonical(payload))
    folder = root / "data/reliability/candidates" / checksum
    immutable_write(folder / "candidate.json", canonical(payload))
    for name in FILES:
        immutable_write(folder / "sources" / name, (root / name).read_bytes())
    if not (folder / "frozen_at.json").exists():
        immutable_write(folder / "frozen_at.json", canonical({"at": timestamp(), "candidate_sha256": checksum}))
    save_artifact(root, "reports/reliability/candidate.json", {"candidate_sha256": checksum, **payload})
    return payload, checksum


def system_score(case, result):
    from copy import deepcopy
    case = deepcopy(case)
    original = {h["chunk_id"]: h for h in case["evidence"]}
    # Include actually retrieved, archive-verified identities in the scorer's
    # evidence inventory, while preserving historical expected/allowed targets.
    # A real non-target citation is disallowed, never called fabricated.
    for hit in result.get("retrieval_evidence", []):
        original[hit["chunk_id"]] = hit
    case["evidence"] = list(original.values())
    claims = []
    for claim in result.get("claims", []):
        for citation in claim.get("citations", []):
            claims.append({"chunk_id": citation["chunk_id"], "quote": citation["quote"]})
    score = score_case(case, canonical({"claims": claims}).decode(), done_reason="error" if result.get("status") == "unavailable" else "stop")
    # This is explicitly a SYSTEM score. Policy abstention is not a raw model pass.
    return score


def raw_score(case, result):
    rows = result.get("model_observations", [])
    if not rows:
        return {"provider_called": False, "passed": None, "guard_or_context_reason": result.get("reason")}
    from ipindia_corpus.span_selection import build_plan, resolve_selection
    try:
        plan = build_plan(case["evidence"])
        resolved = resolve_selection(rows[0]["raw"], case["evidence"], plan)
        return {"provider_called": True, **score_case(case, canonical(resolved["expanded_selection"]).decode(), done_reason="stop")}
    except ValueError:
        return {"provider_called": True, "passed": False, "reason": "invalid_selection"}


def summarize(rows, cases):
    by_id = {c["id"]: c for c in cases}
    positives = [r for r in rows if by_id[r["case_id"]]["expected"]["policy"] == "quote_targets"]
    negatives = [r for r in rows if by_id[r["case_id"]]["expected"]["policy"] == "empty_claims"]
    critical = [r for r in negatives if by_id[r["case_id"]].get("safety_critical", True)]
    def rate(group):
        return sum(r["system_score"]["passed"] for r in group) / len(group) if group else None
    latency = [r["http_or_pipeline_ms"] for r in rows if isinstance(r.get("http_or_pipeline_ms"), (float, int))]
    passed = rate(rows)
    return {"attempts": len(rows), "system_case_success": passed, "answerable_coverage": rate(positives),
            "abstention_success": rate(negatives), "safety_critical_abstention": rate(critical),
            "warm_complete_ms": {"p50": percentile(latency, .5), "p95": percentile(latency, .95)},
            "attempts_with_latency": len(latency),
            "failures": dict(Counter(f for r in rows for f in r["system_score"]["failures"])),
            "backend_blocked_before_model": sum(r["result"].get("request_policy", {}).get("allowed") is False for r in rows),
            "attempts_without_recorded_model_call": sum(not r.get("model_called") for r in rows),
            "misses": [{"case_id": r["case_id"], "pass": r["pass"], "reason": r["result"].get("reason"),
                        "failures": r["system_score"]["failures"]} for r in rows if not r["system_score"]["passed"]],
            "quality_gates_passed": bool(rows) and passed >= .95 and (rate(positives) or 0) >= .95 and rate(critical) == 1,
            "model_credit_for_backend_guards": False}


def run(root=ROOT, *, split="development", mode="controlled", passes=3, frozen=False):
    root = Path(root)
    with file_lock(root / "data/llm/citation_selection.lock"):
        if frozen:
            receipt = verified_artifact(root / "reports/reliability/candidate.json")
            candidate = {k: v for k, v in receipt.items() if k != "candidate_sha256"}
            checksum = receipt["candidate_sha256"]
            if candidate != binding(root) or sha256(canonical(candidate)) != checksum:
                raise ValueError("frozen_candidate_changed")
        else:
            if split == "heldout":
                raise ValueError("heldout_requires_frozen_candidate")
            candidate, checksum = binding(root), sha256(canonical(binding(root)))
        if split == "historical":
            _, cases, _, _ = load_frozen(root)
        else:
            _, cases = load(root, split, candidate_receipt=candidate if frozen else None)
        verifier = ArchiveVerifier(root)
        folder = root / "data/reliability/observations" / checksum / f"{mode}-{split}-{passes}passes"
        immutable_write(folder / "plan.json", canonical({"candidate_sha256": checksum, "mode": mode,
                        "split": split, "passes": passes, "cases_sha256": sha256(canonical(cases))}))
        session = requests.Session()
        session.trust_env = False
        provider = SpanProvider()
        def execute(case):
            started = time.monotonic()
            if mode == "http":
                response = session.post("http://127.0.0.1:8001/answer", json={"question": case["question"]}, timeout=(5, 100))
                if response.status_code not in (200, 503):
                    raise ValueError("unexpected_preview_http_status")
                result = response.json()
                if result.get("candidate") != configuration():
                    raise ValueError("preview_candidate_mismatch")
            else:
                result = answer(case["question"], lambda *a, **k: {"generation_id": verifier.gid, "hits": case["evidence"]},
                                verifier, provider=provider, controlled_evidence=case["evidence"])
            elapsed = (time.monotonic() - started) * 1000
            for claim in result.get("claims", []):
                for citation in claim["citations"]:
                    original = verifier.chunks[citation["chunk_id"]]
                    hit = {**original, "title": original.get("document_title") or verifier.docs[original["document_id"]]["title"], "status": "unknown"}
                    verifier(hit, result["generation_id"])
                    if citation["quote"] not in original["text"] or any(citation.get(k) != original.get(k) for k in ("document_id", "raw_sha256", "source_url", "section", "rule", "page_start", "page_end")):
                        raise ValueError("emitted_citation_failed_archive_audit")
            return result, elapsed
        slots = [(number, c) for number in range(1, passes + 1) for c in cases]
        pending = [(n, c) for n, c in slots if not (folder / "intents" / f"{n}-{c['id']}.json").exists()]
        client = provider.client
        if pending:
            receipt = client.receipt(MODEL)
            if receipt["model_digest"] != DIGEST:
                raise ValueError("pinned_model_required")
            managed = verified_artifact(root / "reports/llm/model_comparison.json")["benchmark_managed_models"]
            preparation = prepare_cohort(client, receipt, managed)
            if not preparation["ready"]:
                raise ValueError(preparation["reason"])
            run_id = uuid.uuid4().hex
            warm, elapsed = execute(next(c for c in cases if c["expected"]["policy"] == "quote_targets"))
            immutable_write(folder / "warmups" / (run_id + ".json"), canonical({"at": timestamp(), "result": warm, "elapsed_ms": elapsed,
                            "preparation": preparation, "runtime": client.request("GET", "/api/version")["response"]}))
            if not warm["model_observations"]:
                raise ValueError("warmup_did_not_call_model")
            with ResourceSampler(interval=1) as sampler:
                for number, case in pending:
                    slot = f"{number}-{case['id']}"
                    intent = {"candidate_sha256": checksum, "pass": number, "case_id": case["id"],
                              "case_sha256": sha256(canonical(case)), "at": timestamp()}
                    immutable_write(folder / "intents" / (slot + ".json"), canonical(intent))
                    attempt_started = time.monotonic()
                    try:
                        result, elapsed = execute(case)
                    except Exception as exc:
                        result, elapsed = {"status": "unavailable", "reason": "evaluation_request_failed", "claims": [],
                                           "model_observations": [], "failure_kind": type(exc).__name__,
                                           "provider_call_state": "unknown_transport_failure"}, (time.monotonic() - attempt_started) * 1000
                    row = {**intent, "result": result, "http_or_pipeline_ms": elapsed,
                           "system_score": system_score(case, result), "model_called": bool(result.get("model_calls_attempted")),
                           "raw_selection_score": raw_score(case, result) if mode == "controlled" else None}
                    raw = canonical(row)
                    immutable_write(folder / "rows" / slot / (sha256(raw) + ".json"), raw)
                    print(json.dumps({"mode": mode, "split": split, "pass": number, "case": case["id"],
                                      "passed": row["system_score"]["passed"], "reason": result.get("reason")}), flush=True)
            immutable_write(folder / "resources" / (run_id + ".json"), canonical({"summary": resource_summary(sampler.samples), "samples": sampler.samples}))
        rows = []
        for number, case in slots:
            slot = f"{number}-{case['id']}"
            intent = json.loads((folder / "intents" / (slot + ".json")).read_bytes())
            if (intent.get("candidate_sha256") != checksum or intent.get("pass") != number
                    or intent.get("case_id") != case["id"] or intent.get("case_sha256") != sha256(canonical(case))):
                raise ValueError("interrupted_intent_binding_failed")
            found = list((folder / "rows" / slot).glob("*.json"))
            if not found:
                # An interrupted intent occupies its slot permanently.
                result = {"status": "unavailable", "reason": "interrupted_attempt_not_retried", "claims": []}
                row = {"candidate_sha256": checksum, "pass": number, "case_id": case["id"],
                       "case_sha256": sha256(canonical(case)), "result": result, "model_called": False,
                       "http_or_pipeline_ms": None, "system_score": system_score(case, result)}
                raw = canonical(row)
                path = folder / "rows" / slot / (sha256(raw) + ".json")
                immutable_write(path, raw)
                found = [path]
            if len(found) != 1 or sha256(found[0].read_bytes()) != found[0].stem:
                raise ValueError("observation_hash_or_uniqueness_failed")
            row = json.loads(found[0].read_bytes())
            if (row["candidate_sha256"] != checksum or row["case_sha256"] != sha256(canonical(case))
                    or row["system_score"] != system_score(case, row["result"])):
                raise ValueError("observation_binding_or_score_failed")
            rows.append(row)
        if binding(root) != candidate:
            raise ValueError("candidate_changed_during_evaluation")
        report = {"at": timestamp(), "candidate_sha256": checksum, "split": split, "mode": mode,
                  "passes": passes, "summary": summarize(rows, cases), "rows": rows,
                  "source_fixture_scope": "Controlled evidence only" if mode == "controlled" else "Real unfiltered HTTP retrieval; controlled source fixtures are not injected",
                  "activation_performed": False, "public_answers_enabled": False}
        save_artifact(root, f"reports/reliability/{mode}-{split}-{passes}passes.json", report)
        return {k: report[k] for k in ("candidate_sha256", "mode", "split", "summary")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-candidate", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--frozen", action="store_true")
    parser.add_argument("--split", choices=("development", "heldout", "historical"), default="development")
    parser.add_argument("--mode", choices=("controlled", "http"), default="controlled")
    parser.add_argument("--passes", type=int, choices=(1, 3), default=3)
    args = parser.parse_args()
    if args.freeze_candidate:
        print(json.dumps({"candidate_sha256": freeze_candidate()[1]}))
    elif args.run:
        print(json.dumps(run(split=args.split, mode=args.mode, passes=args.passes, frozen=args.frozen), indent=2))
    else:
        print(json.dumps({"status": "plan_only", "candidate": binding(ROOT)}))
