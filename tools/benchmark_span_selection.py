"""Freeze and measure backend-copied span selection without application activation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time
import uuid

from ipindia_corpus.indexer import generation
from ipindia_corpus.llm_experiment import (
    ROOT, ExperimentFailure, OllamaExperimentClient, ResourceSampler, canonical,
    code_bindings, cohort_placement, prepare_cohort, protected_state,
    resource_snapshot, resource_summary, save_artifact, sha256, store_receipt,
    summarize, timestamp, timing_metadata,
)
from ipindia_corpus.llm_scoring import normalized, score_case
from ipindia_corpus.locking import file_lock
from ipindia_corpus.paths import immutable_write
from ipindia_corpus.span_selection import (
    SYSTEM, VARIANT, build_plan, request_fields, resolve_selection,
)
from tools.freeze_llm_benchmark import _evidence, load_frozen
from tools.report_llm_results import verified_artifact

MODEL = "qwen3.5:4b"
DIGEST = "d8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04"
PROFILE = {"num_predict": 200, "num_ctx": 4096, "keep_alive": "30m", "think": False}


def experiment_code(root):
    return {**code_bindings(), **{name: sha256((Path(root) / name).read_bytes()) for name in (
        "ipindia_corpus/span_selection.py", "ipindia_corpus/citation_selection.py",
        "tools/benchmark_span_selection.py", "ipindia_corpus/corpus_certification.py",
        "ipindia_corpus/indexer.py", "ipindia_corpus/locking.py", "ipindia_corpus/paths.py")}}


def coverage_audit(case, plan):
    by_id = {hit["chunk_id"]: hit for hit in case["evidence"]}
    groups = []
    for group in case["expected"].get("target_groups", []):
        matches = [identifier for identifier, span in plan["spans"].items()
                   if span["chunk_id"] in group["required_chunk_ids"]
                   and normalized(group["required_contains"]) in normalized(
                       by_id[span["chunk_id"]]["text"][span["start"]:span["end"]])]
        groups.append({"target": group, "matching_span_ids": matches})
    return {"case_id": case["id"], "all_groups_representable": all(g["matching_span_ids"] for g in groups),
            "groups": groups, "forced_empty": not plan["spans"]}


def freeze(root=ROOT):
    root = Path(root)
    manifest, cases, _, _ = load_frozen(root)
    _, documents, chunks = generation(root, generation_id=manifest["generation_id"])
    by_chunk = {c["chunk_id"]: c for c in chunks}
    by_doc = {d["document_id"]: d for d in documents}
    entries = []
    for case in cases:
        for hit in case["evidence"]:
            if hit.get("source") == "synthetic_fixture":
                continue
            chunk = by_chunk.get(hit["chunk_id"])
            if chunk is None or hit != _evidence(chunk, by_doc[chunk["document_id"]]):
                raise ValueError("frozen_evidence_does_not_match_archived_generation")
        plan = build_plan(case["evidence"])
        inputs = request_fields(case["question"], plan)
        entries.append({"case": case, "plan": plan, "request_fields": inputs,
                        "input_sha256": sha256(canonical(inputs)), "plan_sha256": sha256(canonical(plan)),
                        "coverage_audit": coverage_audit(case, plan)})
    bundle = {"variant": VARIANT, "frozen_benchmark": manifest, "system": SYSTEM,
              "entries": entries, "code_sha256": experiment_code(root),
              "protected_state": protected_state(root, manifest["generation_id"]),
              "offset_unit": "Python Unicode codepoints, half-open [start,end)",
              "model": MODEL, "model_digest": DIGEST, "profile": PROFILE, "passes": 3}
    checksum = sha256(canonical(bundle))
    path = root / "data/benchmarks" / (VARIANT + "-" + checksum[:20]) / "bundle.json"
    immutable_write(path, canonical(bundle))
    immutable_write(path.with_suffix(".sha256"), (checksum + "\n").encode("ascii"))
    save_artifact(root, "reports/llm/span_selection_plan.json", {
        "status": "frozen_not_measured", "variant": VARIANT, "bundle_path": path.relative_to(root).as_posix(),
        "bundle_sha256": checksum, "cases": len(cases), "model": MODEL, "profile": PROFILE,
        "coverage_audit": [entry["coverage_audit"] for entry in entries],
        "activation_performed": False, "public_answer_route_enabled": False})
    return bundle, path, checksum


def score_selection(case, plan, raw, reason):
    try:
        resolved = resolve_selection(raw, case["evidence"], plan)
    except ValueError as exc:
        score = score_case(case, None, done_reason=reason)
        score["failures"].append("INVALID_SPAN_SELECTION")
        return {"raw_selection_valid": False, "selection_error": str(exc), "score": score,
                "expanded_selection": None, "backend_result": None}
    score = score_case(case, canonical(resolved["expanded_selection"]).decode("utf-8"), done_reason=reason)
    return {"raw_selection_valid": True, "score": score, **resolved}


def observe_span(client, receipt, entry, *, root, run_id, number, code, warmup=False):
    case, plan = entry["case"], entry["plan"]
    started = time.monotonic()
    row = {"run_id": run_id, "captured_at": timestamp(), "case_id": case["id"], "pass": number,
           "phase": "warmup" if warmup else "timed", "category": case["category"],
           "model": MODEL, "model_digest": DIGEST, "profile": PROFILE, "code_sha256": code,
           "model_input_sha256": entry["input_sha256"], "plan_sha256": entry["plan_sha256"],
           "evidence_sha256": case["evidence_sha256"], "provider_called": False,
           "backend_guard_credit": False, "status": "failed"}
    with ResourceSampler() as sampler:
        try:
            row["placement_before"] = client.placement()
            row["cohort_before"] = cohort_placement(row["placement_before"], MODEL, allow_empty=warmup)
            if not row["cohort_before"]["cohort_safe"]:
                raise ExperimentFailure("unexpected_model_concurrency")
            client.verify_identity(receipt)
            payload = {**entry["request_fields"], "model": MODEL, "stream": False, "think": False,
                       "keep_alive": PROFILE["keep_alive"], "options": {
                           "temperature": 0, "num_ctx": PROFILE["num_ctx"], "num_predict": PROFILE["num_predict"]}}
            row["provider_called"] = True
            chat_started = time.monotonic()
            try:
                result = client.request("POST", "/api/chat", payload)
            finally:
                row["chat_attempt_wall_ms"] = (time.monotonic() - chat_started) * 1000
            response = result["response"]
            if response.get("model") != MODEL:
                raise ExperimentFailure("model_identity_mismatch")
            reason = response.get("done_reason")
            if response.get("done") is not True or reason not in (None, "stop", "length"):
                reason = "error"
            row.update(status="completed" if reason in (None, "stop") else "incomplete",
                       response_json=response, response_bytes_sha256=result["response_bytes_sha256"],
                       scoring_raw_output=response.get("message", {}).get("content"), scoring_done_reason=reason,
                       chat_http_wall_ms=result["http_wall_ms"], timings=timing_metadata(response),
                       **score_selection(case, plan, response.get("message", {}).get("content"), reason))
        except Exception as exc:
            reason = str(exc) if isinstance(exc, ExperimentFailure) else "span_observation_failed"
            row.update(status="timeout" if reason == "timeout" else "failed", failure_reason=reason,
                       scoring_raw_output=None, scoring_done_reason="timeout" if reason == "timeout" else "error",
                       **score_selection(case, plan, None, "timeout" if reason == "timeout" else "error"))
    try:
        row["placement_after"] = client.placement()
    except ExperimentFailure:
        row["placement_after"] = {}
    row["cohort_after"] = cohort_placement(row["placement_after"], MODEL)
    if not row["cohort_after"]["cohort_safe"]:
        row.update(status="failed", failure_reason="unexpected_model_concurrency")
        row["score"]["passed"] = False
        row["score"]["failures"].append("UNEXPECTED_MODEL_CONCURRENCY")
    row["observation_wall_ms"] = (time.monotonic() - started) * 1000
    row["resource_samples"] = sampler.samples
    row["resource_summary"] = resource_summary(sampler.samples)
    return row


def store_row(folder, slot, row):
    raw = canonical(row)
    path = folder / "observations" / slot / (sha256(raw) + ".json")
    immutable_write(path, raw)
    return path


def recover_rows(folder, entries, key, code):
    """Interrupted intents count once as failed attempts, never as fresh slots."""
    rows = []
    for intent_path in sorted((folder / "intents").glob("*.json")):
        intent = json.loads(intent_path.read_bytes())
        slot = intent_path.stem
        case_id, number = intent.get("case_id"), intent.get("pass")
        if (case_id not in entries or type(number) is not int or not 1 <= number <= 3
                or intent.get("checkpoint_key") != key
                or slot != f"{number}-{case_id}"):
            raise ValueError("invalid_span_attempt_intent")
        entry = entries[case_id]
        paths = list((folder / "observations" / slot).glob("*.json"))
        if len(paths) > 1:
            raise ValueError("duplicate_span_attempt_observations")
        if not paths:
            row = {"case_id": case_id, "pass": number, "model": MODEL, "model_digest": DIGEST,
                   "profile": PROFILE, "code_sha256": code, "phase": "timed", "run_id": intent["run_id"],
                   "model_input_sha256": entry["input_sha256"], "plan_sha256": entry["plan_sha256"],
                   "evidence_sha256": entry["case"]["evidence_sha256"], "status": "interrupted",
                   "provider_called": True, "provider_call_state": "unknown_interrupted",
                   "failure_reason": "interrupted_attempt_not_retried", "backend_guard_credit": False,
                   "scoring_raw_output": None, "scoring_done_reason": "error",
                   **score_selection(entry["case"], entry["plan"], None, "cancelled")}
            row["score"]["failures"].append("INTERRUPTED_ATTEMPT")
            paths = [store_row(folder, slot, row)]
        path = paths[0]
        raw = path.read_bytes()
        if path.stem != sha256(raw):
            raise ValueError("span_observation_hash_mismatch")
        row = json.loads(raw)
        expected = {"case_id": case_id, "pass": number, "model": MODEL, "model_digest": DIGEST,
                    "profile": PROFILE, "code_sha256": code, "model_input_sha256": entry["input_sha256"],
                    "plan_sha256": entry["plan_sha256"], "evidence_sha256": entry["case"]["evidence_sha256"]}
        if any(row.get(field) != value for field, value in expected.items()):
            raise ValueError("span_observation_binding_mismatch")
        # Recompute scores from retained raw output; never trust a stored pass flag.
        reason = row["scoring_done_reason"]
        if row["status"] in {"completed", "incomplete"} and row["scoring_raw_output"] != row.get("response_json", {}).get("message", {}).get("content"):
            raise ValueError("span_raw_output_mismatch")
        recomputed = score_selection(entry["case"], entry["plan"],
                                     row["scoring_raw_output"], reason)
        if row["status"] == "interrupted":
            recomputed["score"]["failures"].append("INTERRUPTED_ATTEMPT")
        if row.get("failure_reason") == "unexpected_model_concurrency":
            recomputed["score"]["passed"] = False
            recomputed["score"]["failures"].append("UNEXPECTED_MODEL_CONCURRENCY")
        if any(row.get(field) != value for field, value in recomputed.items()):
            raise ValueError("span_observation_score_mismatch")
        row.update(immutable_observation_path=path.as_posix(), immutable_observation_sha256=sha256(raw))
        rows.append(row)
    return rows


def metrics(rows, entries):
    summary = summarize(rows)
    summary["system_case_pass_rate"] = summary.pop("raw_model_case_pass_rate")
    raw_valid = [r for r in rows if r.get("raw_selection_valid")]
    claims = [claim for row in raw_valid for claim in row["expanded_selection"]["claims"]]
    abstentions = [r for r in rows if entries[r["case_id"]]["case"]["expected"]["policy"] == "empty_claims"]
    nonforced = [r for r in abstentions if entries[r["case_id"]]["plan"]["spans"]]
    return {"summary": summary, "raw_schema_valid_attempts": len(raw_valid), "backend_copied_claims": len(claims),
            "expected_abstention_attempts": len(abstentions),
            "correct_abstention_attempts": sum(r["score"]["correct_abstention"] for r in abstentions),
            "nonforced_abstention_attempts": len(nonforced),
            "correct_nonforced_abstention_attempts": sum(r["score"]["correct_abstention"] for r in nonforced),
            "copy_fidelity_scope": "Backend copies bound archived slices; this is structural system fidelity, not model quotation skill.",
            "misses": [{"case_id": r["case_id"], "pass": r["pass"], "failures": r["score"]["failures"]}
                       for r in rows if not r["score"]["passed"]],
            "by_category": {category: {"attempts": len(group), "passes": sum(r["score"]["passed"] for r in group)}
                            for category in sorted({e["case"]["category"] for e in entries.values()})
                            if (group := [r for r in rows if entries[r["case_id"]]["case"]["category"] == category])}}


def benchmark(root=ROOT):
    root = Path(root)
    with file_lock(root / "data/llm/citation_selection.lock"):
        bundle, bundle_path, key = freeze(root)
        code = bundle["code_sha256"]
        folder = root / "data/llm/span_selection_checkpoints" / key
        immutable_write(folder / "plan.json", canonical({"bundle_sha256": key, "bundle_path": bundle_path.as_posix()}))
        entries = {e["case"]["id"]: e for e in bundle["entries"]}
        rows = recover_rows(folder, entries, key, code)
        done = {(r["pass"], r["case_id"]) for r in rows}
        pending = [(number, entry) for number in range(1, 4) for entry in entries.values()
                   if (number, entry["case"]["id"]) not in done]
        client = OllamaExperimentClient()
        if pending:
            receipt = client.receipt(MODEL)
            if receipt["model_digest"] != DIGEST or not receipt["thinking_capable"]:
                raise ValueError("pinned_thinking_capable_qwen_required")
            main = verified_artifact(root / "reports/llm/model_comparison.json")
            preparation = prepare_cohort(client, receipt, main["benchmark_managed_models"])
            if not preparation["ready"]:
                raise ValueError(preparation["reason"])
            run_id = "span-" + uuid.uuid4().hex
            segment = {"run_id": run_id, "started_at": timestamp(), "preparation": preparation,
                       "resources": resource_snapshot(), "receipt_path": store_receipt(root, receipt),
                       "missing_at_start": len(pending), "runtime_version": client.request("GET", "/api/version")["response"]}
            immutable_write(folder / "segments" / (run_id + ".json"), canonical(segment))
            warmup = observe_span(client, receipt, bundle["entries"][0], root=root,
                                  run_id=run_id, number=0, code=code, warmup=True)
            immutable_write(folder / "warmups" / (run_id + ".json"), canonical(warmup))
            if warmup["status"] != "completed" or not warmup["cohort_after"]["cohort_safe"]:
                raise ValueError("span_warmup_failed")
            for number, entry in pending:
                case_id = entry["case"]["id"]
                slot = f"{number}-{case_id}"
                if not re.fullmatch(r"[0-9]-[a-zA-Z0-9-]+", slot):
                    raise ValueError("unsafe_span_slot")
                immutable_write(folder / "intents" / (slot + ".json"), canonical({
                    "checkpoint_key": key, "case_id": case_id, "pass": number,
                    "run_id": run_id, "started_at": timestamp()}))
                row = observe_span(client, receipt, entry, root=root, run_id=run_id, number=number, code=code)
                store_row(folder, slot, row)
                print(json.dumps({"pass": number, "case": case_id, "status": row["status"],
                                  "passed": row["score"]["passed"]}), flush=True)
                if row.get("failure_reason") == "unexpected_model_concurrency":
                    raise ValueError("unexpected_model_concurrency")
        rows = recover_rows(folder, entries, key, code)
        if len(rows) != 3 * len(entries):
            raise ValueError("incomplete_span_benchmark")
        unchanged = (protected_state(root, bundle["frozen_benchmark"]["generation_id"]) == bundle["protected_state"]
                     and experiment_code(root) == code)
        load_frozen(root)
        result = {"variant": VARIANT, "captured_at": timestamp(), "status": "measured" if unchanged else "protected_state_changed",
                  "bundle_path": bundle_path.relative_to(root).as_posix(), "bundle_sha256": key,
                  "model": MODEL, "model_digest": DIGEST, "profile": PROFILE, "passes": 3,
                  "metrics": metrics(rows, entries), "observations": rows, "code_sha256": code,
                  "segments": [json.loads(p.read_bytes()) for p in sorted((folder / "segments").glob("*.json"))],
                  "protected_state_unchanged": unchanged, "activation_performed": False,
                  "public_answer_route_enabled": False,
                  "limitations": ["Different prompt, representation and output schema from quote-generating variants.",
                                  "Original questions, evidence, order and target labels remain frozen.",
                                  "Literal fidelity is backend copying, not intrinsic model quotation accuracy.",
                                  "All expected target groups remain required; extra decoys fail the original scorer.",
                                  "No pre-inference guard receives model credit. Structurally forced empty outputs reported separately.",
                                  "Developer labels lack independent legal relevance review; substring checks do not prove complete legal meaning.",
                                  "Latency measures model HTTP calls, not end-to-end retrieval or user answers."]}
        save_artifact(root, "reports/llm/span_selection_comparison.json", result)
        return {"status": result["status"], "variant": VARIANT, "metrics": result["metrics"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Run only missing candidate attempts; retain all failures")
    args = parser.parse_args()
    if args.run:
        print(json.dumps(benchmark(), indent=2))
    else:
        bundle, path, checksum = freeze()
        print(json.dumps({"status": "frozen_not_measured", "bundle": path.as_posix(), "sha256": checksum,
                          "cases": len(bundle["entries"]), "unrepresentable": [e["case"]["id"] for e in bundle["entries"]
                          if not e["coverage_audit"]["all_groups_representable"]]}, indent=2))
