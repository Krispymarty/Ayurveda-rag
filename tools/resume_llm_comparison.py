"""Resume missing frozen model calls from immutable observations; never retry failures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import uuid

from ipindia_corpus.llm_experiment import (
    ROOT, OllamaExperimentClient, ExperimentFailure, canonical, code_bindings,
    cohort_placement, observe, prepare_cohort, protected_state, save_artifact,
    sha256, store_receipt, summarize, timestamp,
)
from ipindia_corpus.locking import file_lock
from ipindia_corpus.paths import immutable_write
from tools.freeze_llm_benchmark import load_frozen

MODELS = ("ministral-3:3b", "qwen3.5:4b", "gemma3:4b")
DEFAULT_BASE = "compare-57ab3251502d461586cf8296353a6583"


def read_rows(root, run_ids, cases, bindings):
    """Reject changed inputs, code, profiles, duplicates and unsafe cohorts."""
    rows, inventory = [], []
    for run_id in run_ids:
        if not re.fullmatch(r"(?:compare|resume)-[a-f0-9]{32}", run_id):
            raise ValueError("invalid_observation_run_id")
        folder = Path(root) / "reports/llm/observations" / run_id
        for path in sorted(folder.glob("*.json")):
            raw = path.read_bytes()
            digest = sha256(raw)
            if not path.stem.endswith("-" + digest):
                raise ValueError("observation_hash_mismatch")
            row = json.loads(raw)
            case = cases.get(row.get("case_id"))
            if (case is None or row.get("model") not in MODELS or row.get("run_id") != run_id
                    or row.get("model_input_sha256") != case["model_input_sha256"]
                    or row.get("evidence_sha256") != case["evidence_sha256"]
                    or row.get("category") != case["category"] or row.get("code_sha256") != bindings):
                raise ValueError("observation_input_or_code_mismatch")
            profile = {"num_predict": 200, "num_ctx": 4096, "keep_alive": "30m"}
            if row["model"] == "qwen3.5:4b":
                profile["think"] = False
            if row.get("profile") != profile:
                raise ValueError("observation_profile_mismatch")
            if not row.get("cohort_before", {}).get("cohort_safe") or not row.get("cohort_after", {}).get("cohort_safe"):
                raise ValueError("unsafe_model_cohort_cannot_be_reused")
            row["immutable_observation_path"] = path.relative_to(root).as_posix()
            row["immutable_observation_sha256"] = digest
            rows.append(row)
            inventory.append({"path": path.relative_to(root).as_posix(), "sha256": digest})
    timed = [r for r in rows if r["phase"] in {"comparison_warm", "resumed_comparison_warm"}]
    keys = [(r["model"], r["pass"], r["case_id"]) for r in timed]
    if len(keys) != len(set(keys)) or any(type(r["pass"]) is not int or not 1 <= r["pass"] <= 3 for r in timed):
        raise ValueError("duplicate_or_invalid_case_attempt")
    return rows, inventory


def pending_cases(rows, model, cases):
    done = {(r["pass"], r["case_id"]) for r in rows if r["model"] == model
            and r["phase"] in {"comparison_warm", "resumed_comparison_warm"}}
    return [(number, case) for number in range(1, 4) for case in cases if (number, case["id"]) not in done]


def resume(root=ROOT, *, base_run=DEFAULT_BASE, run=False, client=None):
    root = Path(root)
    with file_lock(root / "data/llm/comparison_resume.lock"):
        return _resume_locked(root, base_run=base_run, run=run, client=client)


def _resume_locked(root, *, base_run, run, client):
    manifest, cases, system, schema = load_frozen(root)
    by_id = {c["id"]: c for c in cases}
    bindings = code_bindings()
    wrapper_hash = sha256(Path(__file__).read_bytes())
    before = protected_state(root, manifest["generation_id"])
    baseline_path = root / "reports/llm/baseline_performance.json"
    baseline_raw = baseline_path.read_bytes()
    if sha256(baseline_raw) != baseline_path.with_suffix(".json.sha256").read_text(encoding="ascii").strip():
        raise ValueError("baseline_report_hash_mismatch")
    baseline = json.loads(baseline_raw)
    if not baseline.get("protected_state_unchanged") or baseline["protected_state_after"] != before:
        raise ValueError("protected_state_changed_since_baseline")
    base_rows, base_inventory = read_rows(root, [base_run], by_id, bindings)
    model_digests = {}
    for row in base_rows:
        if row["model"] in model_digests and model_digests[row["model"]] != row["model_digest"]:
            raise ValueError("mixed_model_digests")
        model_digests[row["model"]] = row["model_digest"]
    if set(model_digests) != set(MODELS):
        raise ValueError("base_run_must_identify_all_three_models")
    folder = root / "data/llm/comparison_resume"
    plan_path = folder / "plan.json"
    expected = {"base_run": base_run, "benchmark_id": manifest["id"],
                "benchmark_files": manifest["files"], "generation_id": manifest["generation_id"],
                "base_inventory": base_inventory, "model_digests": model_digests,
                "code_sha256": bindings, "wrapper_sha256": wrapper_hash, "protected_state_anchor": before}
    if plan_path.exists():
        plan = json.loads(plan_path.read_bytes())
        if any(plan.get(key) != value for key, value in expected.items()):
            raise ValueError("resume_plan_changed")
    else:
        plan = {**expected, "created_at": timestamp(), "comparison_id": "resumed-" + uuid.uuid4().hex}
        immutable_write(plan_path, canonical(plan))
    plan_hash = sha256(plan_path.read_bytes())
    markers = []
    for path in sorted((folder / "segments").glob("*.json")):
        raw = path.read_bytes()
        marker = json.loads(raw)
        if path.stem != sha256(raw) or marker.get("plan_sha256") != plan_hash:
            raise ValueError("segment_checkpoint_hash_mismatch")
        markers.append(marker)
    run_ids = [base_run] + [marker["run_id"] for marker in markers]
    rows, _ = read_rows(root, run_ids, by_id, bindings)
    counts = {model: len(pending_cases(rows, model, cases)) for model in MODELS}
    if not run:
        return {"status": "plan_only", "missing_attempts": counts, "model_calls_performed": False,
                "preserved_failures_are_not_retried": True, "plan_sha256": plan_hash}
    client = client or OllamaExperimentClient()
    receipts = {model: client.receipt(model) for model in MODELS}
    if any(receipts[model]["model_digest"] != model_digests[model] for model in MODELS):
        raise ValueError("current_model_digest_differs_from_saved_attempts")
    receipt_paths = {model: store_receipt(root, receipts[model]) for model in MODELS}
    for model in MODELS:
        missing = pending_cases(rows, model, cases)
        if not missing:
            continue
        preparation = prepare_cohort(client, receipts[model], model_digests)
        if not preparation["ready"]:
            raise ExperimentFailure(preparation["reason"])
        profile = {"num_predict": 200, "num_ctx": 4096, "keep_alive": "30m"}
        if model == "qwen3.5:4b":
            if not receipts[model]["thinking_capable"]:
                raise ValueError("saved_thinking_profile_no_longer_supported")
            profile["think"] = False
        run_id = "resume-" + uuid.uuid4().hex
        marker = {"run_id": run_id, "started_at": timestamp(), "plan_sha256": plan_hash,
                  "model": model, "model_digest": model_digests[model], "remaining_at_start": len(missing),
                  "preparation": preparation, "ollama_version": client.request("GET", "/api/version")["response"]}
        marker_raw = canonical(marker)
        immutable_write(folder / "segments" / (sha256(marker_raw) + ".json"), marker_raw)
        markers.append(marker)
        run_ids.append(run_id)
        warmup = observe(client, receipts[model], cases[0], system, schema, profile,
                         root=root, run_id=run_id, phase="resumed_comparison_warmup",
                         enforce_single_model=True, allow_unloaded=True)
        rows.append(warmup)
        if warmup["status"] != "completed" or not cohort_placement(client.placement(), model)["single_model_verified"]:
            raise ExperimentFailure("resumed_model_warmup_failed")
        for number, case in missing:
            row = observe(client, receipts[model], case, system, schema, profile, root=root,
                          run_id=run_id, phase="resumed_comparison_warm", pass_number=number,
                          enforce_single_model=True)
            rows.append(row)
            print(json.dumps({"model": model, "pass": number, "case": case["id"], "status": row["status"]}), flush=True)
            if row.get("failure_reason") == "unexpected_model_concurrency":
                raise ExperimentFailure("unexpected_model_concurrency")
    # Reload immutable bytes, avoiding mutable sampler lists from a live return value.
    rows, _ = read_rows(root, run_ids, by_id, bindings)
    if any(pending_cases(rows, model, cases) for model in MODELS):
        raise ValueError("comparison_still_incomplete")
    after = protected_state(root, manifest["generation_id"])
    unchanged = before == after and code_bindings() == bindings and sha256(Path(__file__).read_bytes()) == wrapper_hash
    load_frozen(root)
    results = []
    case_order = {case["id"]: n for n, case in enumerate(cases)}
    for model in MODELS:
        selected = sorted([r for r in rows if r["model"] == model and r["phase"] in {
            "comparison_warm", "resumed_comparison_warm"}], key=lambda r: (r["pass"], case_order[r["case_id"]]))
        warmups = [r for r in rows if r["model"] == model and r["phase"].endswith("warmup")]
        results.append({"model": model, "model_digest": model_digests[model], "status": "measured",
                        "profile": selected[0]["profile"], "model_receipt_path": receipt_paths[model],
                        "warmup": warmups[0], "warmups": warmups, "summary": summarize(selected),
                        "category_summaries": {category: summarize([r for r in selected if r["category"] == category])
                                               for category in sorted({r["category"] for r in selected})},
                        "observations": selected, "session_ranges": {identifier: {
                            "first_at": min(r["captured_at"] for r in selected if r["run_id"] == identifier),
                            "last_at": max(r["captured_at"] for r in selected if r["run_id"] == identifier)}
                            for identifier in sorted({r["run_id"] for r in selected})}})
    result = {"captured_at": timestamp(), "run_id": plan["comparison_id"], "benchmark_id": manifest["id"],
              "benchmark_files": manifest["files"], "generation_id": manifest["generation_id"],
              "cases_per_pass": len(cases), "passes": 3, "models": results,
              "protected_state_before": before, "protected_state_after": after,
              "protected_state_unchanged": unchanged, "code_sha256_before": bindings, "code_sha256_after": code_bindings(),
              "code_unchanged": bindings == code_bindings(), "resume_wrapper_sha256": wrapper_hash,
              "resume_plan_sha256": plan_hash, "execution_mode": "verified_resumed_segments",
              "segment_checkpoints": markers, "benchmark_managed_models": model_digests,
              "public_answer_route_enabled": False, "winner": None, "activation_performed": False,
              "model_downloads_performed": False, "raw_model_and_backend_guards_separate": True,
              "status": "completed" if unchanged else "failed_protected_state_changed",
              "limitations": ["Warm cohorts span interrupted sessions and dates; every resumed segment receives an unscored warmup.",
                              "Latency is descriptive of retained session conditions, not a controlled causal ranking across dates.",
                              "Protected-file baseline and current hashes match; no continuous monitor existed during the old interruption.",
                              "Every saved timed failure is retained and counted once; it is never retried for a better score.",
                              "Other interrupted diagnostic runs are retained separately and are not pooled."]}
    save_artifact(root, "reports/llm/model_comparison.json", result)
    return {"status": result["status"], "models": len(results), "timed_attempts": sum(len(r["observations"]) for r in results),
            "protected_state_unchanged": unchanged, "winner": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Perform only missing local model calls")
    parser.add_argument("--base-run", default=DEFAULT_BASE)
    args = parser.parse_args()
    print(json.dumps(resume(base_run=args.base_run, run=args.run), indent=2))


if __name__ == "__main__":
    main()
