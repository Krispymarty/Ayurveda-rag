"""One-variable 200-to-600-token follow-up to the measured chunk-ID enum."""
from __future__ import annotations

import json
from pathlib import Path
import uuid

from ipindia_corpus.llm_experiment import (
    ROOT, OllamaExperimentClient, canonical, code_bindings, cohort_placement,
    observe, prepare_cohort, protected_state, save_artifact, sha256, store_receipt, summarize, timestamp,
)
from ipindia_corpus.locking import file_lock
from ipindia_corpus.paths import immutable_write
from tools.benchmark_citation_selection import checkpoint_rows, variant_code
from tools.freeze_llm_benchmark import load_frozen
from ipindia_corpus.citation_selection import derive_case
from tools.report_llm_results import verified_artifact


def budget_profile(parent):
    allowed = {"num_predict", "num_ctx", "keep_alive", "think"}
    if (set(parent) - allowed or parent.get("num_predict") != 200 or parent.get("num_ctx") != 4096
            or parent.get("keep_alive") != "30m" or "think" in parent and parent["think"] is not False):
        raise ValueError("verified_common_parent_profile_required")
    return {**parent, "num_predict": 600}


def benchmark(root=ROOT, *, client=None):
    root = Path(root)
    with file_lock(root / "data/llm/citation_selection.lock"):
        return _run(root, client=client)


def _run(root, *, client):
    parent_path = root / "reports/llm/citation_selection_comparison.json"
    parent = verified_artifact(parent_path)
    if (parent.get("status") != "measured" or not parent.get("protected_state_unchanged")
            or parent.get("passes") != 3 or len(parent.get("observations", [])) != 135):
        raise ValueError("completed_200_token_enum_parent_required")
    manifest, cases, system, schema = load_frozen(root)
    derived = [derive_case(case, system, schema) for case in cases]
    if [binding for _, _, binding in derived] != parent["case_bindings"]:
        raise ValueError("parent_derived_inputs_changed")
    base_path = root / "reports/llm/model_comparison.json"
    main = verified_artifact(base_path)
    if sha256(base_path.read_bytes()) != parent["base_comparison_sha256"]:
        raise ValueError("main_comparison_changed")
    before = protected_state(root, manifest["generation_id"])
    if before != parent["protected_state_after"]:
        raise ValueError("protected_state_changed_since_parent")
    code = {**code_bindings(), **variant_code(root), "tools/benchmark_citation_budget.py": sha256(Path(__file__).read_bytes())}
    profile = budget_profile(parent["profile"])
    model = parent["model"]
    client = client or OllamaExperimentClient()
    receipt = client.receipt(model)
    if receipt["model_digest"] != parent["model_digest"] or profile.get("think") is False and not receipt["thinking_capable"]:
        raise ValueError("parent_model_identity_or_capability_changed")
    receipt_path = store_receipt(root, receipt)
    checkpoint = {"parent_sha256": sha256(parent_path.read_bytes()), "base_sha256": sha256(base_path.read_bytes()),
                  "profile": profile, "model_digest": receipt["model_digest"], "code_sha256": code,
                  "case_bindings": parent["case_bindings"], "protected_state": before}
    key = sha256(canonical(checkpoint))
    folder = root / "data/llm/citation_budget_checkpoints" / key
    immutable_write(folder / "plan.json", canonical(checkpoint))
    markers = []
    for path in sorted((folder / "segments").glob("*.json")):
        raw = path.read_bytes()
        marker = json.loads(raw)
        if path.stem != sha256(raw) or marker.get("checkpoint_key") != key:
            raise ValueError("budget_segment_checkpoint_mismatch")
        markers.append(marker)
    run_ids = [marker["run_id"] for marker in markers]
    lookup = {case["id"]: case for case, _, _ in derived}
    rows, warmups = checkpoint_rows(root, run_ids, lookup, model=model, model_digest=receipt["model_digest"], profile=profile)
    done = {(row["pass"], row["case_id"]) for row in rows}
    if any(type(number) is not int or not 1 <= number <= 3 for number, _ in done):
        raise ValueError("invalid_budget_pass")
    pending = [(number, case, schema) for number in range(1, 4) for case, schema, _ in derived
               if (number, case["id"]) not in done]
    if pending:
        preparation = prepare_cohort(client, receipt, main["benchmark_managed_models"])
        if not preparation["ready"]:
            raise ValueError(preparation["reason"])
        run_id = "citation-enum-" + uuid.uuid4().hex
        marker = {"run_id": run_id, "started_at": timestamp(), "checkpoint_key": key,
                  "preparation": preparation, "missing_at_start": len(pending)}
        raw = canonical(marker)
        immutable_write(folder / "segments" / (sha256(raw) + ".json"), raw)
        markers.append(marker)
        run_ids.append(run_id)
        first, first_schema, _ = derived[0]
        warmup = observe(client, receipt, first, system, first_schema, profile, root=root, run_id=run_id,
                         phase="citation_enum_warmup", enforce_single_model=True, allow_unloaded=True)
        if warmup["status"] != "completed" or not cohort_placement(client.placement(), model)["single_model_verified"]:
            raise ValueError("budget_warmup_failed")
        for number, case, schema in pending:
            row = observe(client, receipt, case, system, schema, profile, root=root, run_id=run_id,
                          phase="citation_enum_warm", pass_number=number, enforce_single_model=True)
            print(json.dumps({"model": model, "output_cap": 600, "pass": number, "case": case["id"], "status": row["status"]}), flush=True)
            if row.get("failure_reason") == "unexpected_model_concurrency":
                raise ValueError("unexpected_model_concurrency")
    rows, warmups = checkpoint_rows(root, run_ids, lookup, model=model, model_digest=receipt["model_digest"], profile=profile)
    if len(rows) != 135:
        raise ValueError("budget_comparison_incomplete")
    order = {case["id"]: index for index, case in enumerate(cases)}
    rows.sort(key=lambda r: (r["pass"], order[r["case_id"]]))
    after = protected_state(root, manifest["generation_id"])
    unchanged = (before == after and checkpoint["parent_sha256"] == sha256(parent_path.read_bytes())
                 and checkpoint["base_sha256"] == sha256(base_path.read_bytes())
                 and code == {**code_bindings(), **variant_code(root), "tools/benchmark_citation_budget.py": sha256(Path(__file__).read_bytes())})
    load_frozen(root)
    result = {"variant": "citation-chunk-id-enum-v1-output600", "status": "measured" if unchanged else "failed_protected_state_changed",
              "captured_at": timestamp(), "run_id": "citation-budget-" + key[:24], "model": model,
              "model_digest": receipt["model_digest"], "model_receipt_path": receipt_path,
              "changed_variable": "num_predict: 200 to 600; messages, schema and all other options unchanged",
              "profile": profile, "parent_enum_sha256": checkpoint["parent_sha256"],
              "base_comparison_sha256": checkpoint["base_sha256"], "benchmark_id": manifest["id"],
              "cases_per_pass": 45, "passes": 3, "case_bindings": parent["case_bindings"],
              "observations": rows, "warmups": warmups, "summary": summarize(rows),
              "protected_state_before": before, "protected_state_after": after,
              "protected_state_unchanged": unchanged, "code_sha256": code,
              "segment_checkpoints": markers, "activation_performed": False, "public_answer_route_enabled": False,
              "winner": None, "limitations": ["Schema constraints are system behavior, not intrinsic model fidelity.",
                                               "Separate sessions differ in load; latency remains descriptive.",
                                               "Schema-forced abstentions require separate reporting."]}
    save_artifact(root, "reports/llm/citation_budget600_comparison.json", result)
    return {key: result[key] for key in ("variant", "status", "summary", "activation_performed")}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Permit missing model calls in the separate 600-token follow-up")
    args = parser.parse_args()
    print(json.dumps(benchmark() if args.run else {"status": "plan_only", "changed_variable": "num_predict 200 to 600", "model_calls_performed": False}, indent=2))
