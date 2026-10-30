"""Plan or explicitly run the chunk-ID enum ablation, never production activation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import uuid

from ipindia_corpus.citation_selection import VARIANT, derive_case
from ipindia_corpus.llm_experiment import (
    ROOT, OllamaExperimentClient, ExperimentFailure, _managed_models, code_bindings,
    canonical, cohort_placement, observe, prepare_cohort, protected_state, save_artifact,
    store_receipt, summarize, timestamp, validate_model,
)
from ipindia_corpus.locking import file_lock
from ipindia_corpus.paths import immutable_write
from tools.freeze_llm_benchmark import load_frozen


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def variant_code(root):
    return {name: digest(Path(root) / name) for name in (
        "ipindia_corpus/citation_selection.py", "tools/benchmark_citation_selection.py")}


def plan(root=ROOT):
    root = Path(root)
    manifest, cases, system, base_schema = load_frozen(root)
    derived = [derive_case(case, system, base_schema) for case in cases]
    report = {"variant": VARIANT, "status": "planned_not_measured", "created_at": timestamp(),
              "benchmark_id": manifest["id"], "benchmark_files": manifest["files"],
              "generation_id": manifest["generation_id"], "cases": len(cases),
              "changed_variable": "chunk_id enum from every eligible official input hit; maxItems0 if no eligible IDs",
              "unchanged": ["question", "system", "evidence text and metadata", "evidence ordering", "exact quote field", "strict backend scorer"],
              "case_bindings": [binding for _, _, binding in derived],
              "schema_forced_empty_case_ids": [binding["case_id"] for _, _, binding in derived if binding["forced_empty_selection"]],
              "code_sha256": variant_code(root), "public_answer_route_enabled": False,
              "model_calls_performed": False, "activation_performed": False,
              "profile": {"temperature": 0, "num_ctx": 4096, "num_predict": 200, "keep_alive": "30m",
                          "think": "false only where model receipt verifies thinking capability"},
              "limitations": ["This is a different schema/input variant, not the original frozen model input.",
                              "All eligible decoys remain in the enum; expected targets are never used to construct it.",
                              "Grammar-enforced IDs are a system constraint, not evidence of intrinsic model citation reliability.",
                              "Cases without eligible official citations are forced empty structurally, not credited as learned abstention.",
                              "No model-quality or latency improvement has been measured for this candidate."]}
    save_artifact(root, "reports/llm/citation_selection_plan.json", report)
    return report, derived, system


def _completed_comparison(root):
    path = Path(root) / "reports/llm/model_comparison.json"
    if (not path.is_file() or not path.with_suffix(".json.sha256").is_file()
            or digest(path) != path.with_suffix(".json.sha256").read_text(encoding="ascii").strip()):
        raise ValueError("hash_verified_main_model_comparison_required")
    result = json.loads(path.read_text(encoding="utf-8"))
    measured_tags = {row.get("model") for row in result.get("models", [])}
    expected_count = result.get("cases_per_pass", 0) * result.get("passes", 0)
    if (result.get("status") != "completed" or not result.get("protected_state_unchanged")
            or measured_tags != {"ministral-3:3b", "qwen3.5:4b", "gemma3:4b"}
            or len(result.get("models", [])) != 3 or expected_count <= 0
            or any(row.get("status") != "measured" or len(row.get("observations", [])) != expected_count
                   or len({(r["pass"], r["case_id"]) for r in row["observations"]}) != expected_count
                   for row in result["models"])):
        raise ValueError("finish_main_three_model_comparison_before_ablation")
    return result, digest(path)


def benchmark(root=ROOT, *, model, passes=3, client=None):
    with file_lock(Path(root) / "data/llm/citation_selection.lock"):
        return _benchmark_locked(root, model=model, passes=passes, client=client)


def checkpoint_rows(root, run_ids, cases, *, model, model_digest, profile):
    rows, warmups = [], []
    for run_id in run_ids:
        if not re.fullmatch(r"citation-enum-[a-f0-9]{32}", run_id):
            raise ValueError("invalid_candidate_checkpoint_run")
        for path in sorted((Path(root) / "reports/llm/observations" / run_id).glob("*.json")):
            raw = path.read_bytes()
            checksum = hashlib.sha256(raw).hexdigest()
            if not path.stem.endswith("-" + checksum):
                raise ValueError("candidate_observation_hash_mismatch")
            row = json.loads(raw)
            case = cases.get(row.get("case_id"))
            if (not case or row.get("model") != model or row.get("model_digest") != model_digest
                    or row.get("run_id") != run_id or row.get("profile") != profile
                    or row.get("model_input_sha256") != case["model_input_sha256"]
                    or row.get("evidence_sha256") != case["evidence_sha256"]
                    or row.get("code_sha256") != code_bindings()):
                raise ValueError("candidate_checkpoint_binding_mismatch")
            row["immutable_observation_path"] = path.relative_to(root).as_posix()
            row["immutable_observation_sha256"] = checksum
            if row["phase"] == "citation_enum_warm":
                if not row.get("cohort_before", {}).get("cohort_safe") or not row.get("cohort_after", {}).get("cohort_safe"):
                    raise ValueError("unsafe_candidate_checkpoint_cohort")
                rows.append(row)
            elif row["phase"] == "citation_enum_warmup":
                warmups.append(row)
            else:
                raise ValueError("unknown_candidate_checkpoint_phase")
    keys = [(row["pass"], row["case_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate_candidate_checkpoint_attempt")
    return rows, warmups


def _benchmark_locked(root=ROOT, *, model, passes=3, client=None):
    root = Path(root)
    validate_model(model)
    if type(passes) is not int or not 1 <= passes <= 3:
        raise ValueError("passes_must_be_1_to_3")
    original, original_hash = _completed_comparison(root)
    planned, derived, system = plan(root)
    if (original.get("benchmark_id") != planned["benchmark_id"]
            or original.get("generation_id") != planned["generation_id"]
            or original.get("cases_per_pass") != len(derived)):
        raise ValueError("base_and_candidate_benchmark_identity_must_match")
    before = protected_state(root, planned["generation_id"])
    code_before = {**code_bindings(), **variant_code(root)}
    baseline = json.loads((root / "reports/llm/baseline_performance.json").read_text(encoding="utf-8"))
    client = client or OllamaExperimentClient()
    receipt = client.receipt(model)
    baseline_model = next((row for row in original["models"] if row.get("model") == model), None)
    if not baseline_model or baseline_model.get("model_digest") != receipt["model_digest"]:
        raise ValueError("exact_model_digest_must_match_completed_base_comparison")
    profile = {"num_predict": 200, "num_ctx": 4096, "keep_alive": "30m"}
    if receipt["thinking_capable"]:
        profile["think"] = False
    if baseline_model.get("profile") != profile:
        raise ValueError("base_and_enum_profiles_must_match")
    receipt_path = store_receipt(root, receipt)
    checkpoint = {"base_comparison_sha256": original_hash, "model": model,
                  "model_digest": receipt["model_digest"], "profile": profile, "passes": passes,
                  "case_bindings": planned["case_bindings"], "code_sha256": code_before,
                  "protected_state_anchor": before}
    key = hashlib.sha256(canonical(checkpoint)).hexdigest()
    directory = root / "data/llm/citation_checkpoints" / key
    immutable_write(directory / "plan.json", canonical(checkpoint))
    markers = []
    for path in sorted((directory / "segments").glob("*.json")):
        raw = path.read_bytes()
        marker = json.loads(raw)
        if path.stem != hashlib.sha256(raw).hexdigest() or marker.get("checkpoint_key") != key:
            raise ValueError("candidate_segment_checkpoint_mismatch")
        markers.append(marker)
    run_ids = [marker["run_id"] for marker in markers]
    lookup = {case["id"]: case for case, _, _ in derived}
    rows, warmups = checkpoint_rows(root, run_ids, lookup, model=model,
                                   model_digest=receipt["model_digest"], profile=profile)
    if any(type(row["pass"]) is not int or not 1 <= row["pass"] <= passes for row in rows):
        raise ValueError("invalid_candidate_checkpoint_pass")
    done = {(row["pass"], row["case_id"]) for row in rows}
    pending = [(number, case, schema) for number in range(1, passes + 1) for case, schema, _ in derived
               if (number, case["id"]) not in done]
    abort_reason = None
    preparation = markers[-1]["preparation"] if markers else None
    if pending:
        preparation = prepare_cohort(client, receipt, _managed_models(root, baseline))
        if not preparation["ready"]:
            raise ExperimentFailure(preparation["reason"])
        run_id = "citation-enum-" + uuid.uuid4().hex
        marker = {"run_id": run_id, "started_at": timestamp(), "checkpoint_key": key,
                  "preparation": preparation, "missing_at_start": len(pending)}
        raw = canonical(marker)
        immutable_write(directory / "segments" / (hashlib.sha256(raw).hexdigest() + ".json"), raw)
        markers.append(marker)
        run_ids.append(run_id)
        first, first_schema, _ = derived[0]
        warmup = observe(client, receipt, first, system, first_schema, profile, root=root, run_id=run_id,
                         phase="citation_enum_warmup", enforce_single_model=True, allow_unloaded=True)
        warmups.append(warmup)
        if warmup["status"] != "completed" or not cohort_placement(client.placement(), model)["single_model_verified"]:
            abort_reason = "warmup_incomplete_or_nonexclusive"
        else:
            for repetition, case, schema in pending:
                row = observe(client, receipt, case, system, schema, profile, root=root, run_id=run_id,
                              phase="citation_enum_warm", pass_number=repetition, enforce_single_model=True)
                print(json.dumps({"model": model, "pass": repetition, "case": case["id"], "status": row["status"]}), flush=True)
                if row.get("failure_reason") == "unexpected_model_concurrency":
                    abort_reason = "unexpected_model_concurrency"
                    break
    rows, warmups = checkpoint_rows(root, run_ids, lookup, model=model,
                                   model_digest=receipt["model_digest"], profile=profile)
    order = {case["id"]: index for index, (case, _, _) in enumerate(derived)}
    rows.sort(key=lambda row: (row["pass"], order[row["case_id"]]))
    if not abort_reason and len(rows) != passes * len(derived):
        raise ValueError("candidate_checkpoint_still_incomplete")
    after = protected_state(root, planned["generation_id"])
    code_after = {**code_bindings(), **variant_code(root)}
    # Re-verify the unchanged base bundle and bind derived inputs again.
    _, final_cases, final_system, final_schema = load_frozen(root)
    final_bindings = [derive_case(case, final_system, final_schema)[2] for case in final_cases]
    unchanged = (before == after and code_before == code_after and final_bindings == planned["case_bindings"]
                 and digest(root / "reports/llm/model_comparison.json") == original_hash)
    forced_empty = set(planned["schema_forced_empty_case_ids"])
    report = {"variant": VARIANT, "run_id": "citation-checkpoint-" + key[:24], "captured_at": timestamp(),
              "status": "failed_protected_state_changed" if not unchanged else "aborted" if abort_reason else "measured",
              "abort_reason": abort_reason, "model": model, "model_digest": receipt["model_digest"],
              "model_receipt_path": receipt_path, "profile": profile, "benchmark_id": planned["benchmark_id"],
              "cases_per_pass": len(derived), "passes": passes, "case_bindings": planned["case_bindings"],
              "base_comparison_sha256": original_hash, "base_model_summary": baseline_model.get("summary"),
              "summary": summarize(rows),
              "summary_scope": "grammar-constrained system behavior, not intrinsic unconstrained model reliability",
              "schema_forced_empty_case_ids": sorted(forced_empty),
              "nonempty_eligible_evidence_summary": summarize([row for row in rows if row["case_id"] not in forced_empty]),
              "schema_forced_empty_summary": summarize([row for row in rows if row["case_id"] in forced_empty]),
              "category_summaries": {
                  category: summarize([row for row in rows if row["category"] == category])
                  for category in sorted({row["category"] for row in rows})},
              "cohort_preparation": preparation, "warmup": warmups[-1], "warmups": warmups, "observations": rows,
              "checkpoint_key": key, "segment_checkpoints": markers,
              "execution_mode": "checkpointed; saved failed attempts are never retried",
              "protected_state_before": before, "protected_state_after": after,
              "protected_state_unchanged": unchanged, "code_sha256_before": code_before, "code_sha256_after": code_after,
              "limitations": [item for item in planned["limitations"] if not item.startswith("No model-quality")]
                  + ["Separate or resumed sessions can differ in background load; latency is descriptive, not causal optimization proof."],
              "public_answer_route_enabled": False,
              "activation_performed": False, "model_downloads_performed": False, "winner": None}
    save_artifact(root, "reports/llm/citation_selection_comparison.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Explicitly permit local inference after the main comparison")
    parser.add_argument("--model", default="ministral-3:3b")
    parser.add_argument("--passes", type=int, default=3)
    args = parser.parse_args()
    try:
        result = benchmark(model=args.model, passes=args.passes) if args.run else plan()[0]
        print(json.dumps({key: result[key] for key in ("variant", "status", "public_answer_route_enabled", "activation_performed")}, indent=2))
    except Exception:
        print(json.dumps({"status": "unavailable", "reason": "candidate_ablation_prerequisite_or_runtime_unavailable"}))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
