"""Audit retained observations and report claim precision without abstention credit."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import statistics

from ipindia_corpus.grounded_answers import _exact_quote
from ipindia_corpus.llm_experiment import ROOT, save_artifact, sha256, summarize, timestamp
from tools.freeze_llm_benchmark import load_frozen


def verified_artifact(path):
    path = Path(path)
    raw = path.read_bytes()
    if sha256(raw) != path.with_suffix(path.suffix + ".sha256").read_text(encoding="ascii").strip():
        raise ValueError("report_hash_mismatch")
    return json.loads(raw)


def verify_observation(root, row, case):
    relative = Path(row["immutable_observation_path"])
    path = (Path(root) / relative).resolve()
    parent = (Path(root) / "reports/llm/observations").resolve()
    if relative.is_absolute() or not path.is_relative_to(parent):
        raise ValueError("observation_path_outside_archive")
    raw = path.read_bytes()
    if sha256(raw) != row["immutable_observation_sha256"]:
        raise ValueError("immutable_observation_hash_mismatch")
    original = {k: v for k, v in row.items() if k not in {"immutable_observation_path", "immutable_observation_sha256"}}
    if json.loads(raw) != original:
        raise ValueError("report_observation_differs_from_archived_observation")
    if row["model_input_sha256"] != case["model_input_sha256"] or row["evidence_sha256"] != case["evidence_sha256"]:
        raise ValueError("observation_input_binding_mismatch")


def quality(rows, cases):
    """Emitted claims are the precision denominator; empty selections add none."""
    result = {"attempts": len(rows), "schema_valid_attempts": 0, "nonempty_schema_valid_responses": 0,
              "emitted_claims": 0, "valid_citation_claims": 0, "source_verified_quote_claims": 0,
              "document_id_used_as_chunk_id_claims": 0, "other_unknown_id_claims": 0,
              "literal_substring_quote_claims": 0, "expected_abstention_attempts": 0,
              "correct_abstention_attempts": 0, "positive_attempts": 0, "positive_case_passes": 0,
              "guard_only_attempts": 0}
    for row in rows:
        case = cases[row["case_id"]]
        score = row["score"]
        if score.get("guard_only") or not row.get("provider_called"):
            result["guard_only_attempts"] += 1
            continue
        if case["expected"]["policy"] == "empty_claims":
            result["expected_abstention_attempts"] += 1
            result["correct_abstention_attempts"] += score.get("correct_abstention") is True
        else:
            result["positive_attempts"] += 1
            result["positive_case_passes"] += score.get("passed") is True
        if score.get("schema_valid") is not True:
            continue
        result["schema_valid_attempts"] += 1
        claims = json.loads(row["response_json"]["message"]["content"])["claims"]
        result["nonempty_schema_valid_responses"] += bool(claims)
        eligible = {h["chunk_id"]: h for h in case["evidence"] if h.get("source") != "synthetic_fixture"}
        document_ids = {h["document_id"] for h in case["evidence"] if isinstance(h.get("document_id"), str)}
        for claim in claims:
            result["emitted_claims"] += 1
            hit = eligible.get(claim["chunk_id"])
            if hit is None:
                result["document_id_used_as_chunk_id_claims" if claim["chunk_id"] in document_ids else "other_unknown_id_claims"] += 1
                continue
            result["valid_citation_claims"] += 1
            result["literal_substring_quote_claims"] += claim["quote"] in hit["text"]
            result["source_verified_quote_claims"] += _exact_quote(hit["text"], claim["quote"]) is not None
    def ratio(numerator, denominator):
        return result[numerator] / result[denominator] if result[denominator] else None
    result.update(
        quote_precision=ratio("source_verified_quote_claims", "emitted_claims"),
        literal_quote_precision=ratio("literal_substring_quote_claims", "emitted_claims"),
        valid_citation_precision=ratio("valid_citation_claims", "emitted_claims"),
        correct_abstention_rate=ratio("correct_abstention_attempts", "expected_abstention_attempts"),
        positive_case_pass_rate=ratio("positive_case_passes", "positive_attempts"),
        precision_scope="Claims from schema-parseable responses, including incomplete responses; failures remain in attempt metrics. Empty claims earn no precision credit.",
        quote_matching="source_verified allows the existing whitespace normalization; literal_substring requires unchanged characters.",
    )
    return result


def cohort(root, model, cases):
    rows = model.get("observations", [])
    seen = set()
    for row in rows:
        key = (row["pass"], row["case_id"])
        if key in seen or row["case_id"] not in cases:
            raise ValueError("duplicate_or_unknown_case_observation")
        seen.add(key)
        verify_observation(root, row, cases[row["case_id"]])
        if row["model"] != model["model"] or row["model_digest"] != model["model_digest"]:
            raise ValueError("model_identity_mismatch")
        if row.get("profile") != model.get("profile"):
            raise ValueError("observation_profile_differs_from_report")
    resource_rows = [row.get("resource_summary", {}) for row in rows]
    gpu_peaks = [r["peak_gpu_memory_used_mb"] for r in resource_rows if r.get("peak_gpu_memory_used_mb") is not None]
    model_sizes = [row["cohort_after"]["reported_size_vram_bytes"] for row in rows
                   if row.get("cohort_after", {}).get("reported_size_vram_bytes") is not None]
    failures = Counter(error for row in rows for error in row["score"]["failures"])
    return {"model": model["model"], "model_digest": model.get("model_digest"), "status": model["status"],
            "profile": model.get("profile"), "summary": summarize(rows), "quality": quality(rows, cases),
            "by_review_kind": {kind: quality([row for row in rows if cases[row["case_id"]]["review_kind"] == kind], cases)
                               for kind in sorted({case["review_kind"] for case in cases.values()})},
            "by_category": {category: {"quality": quality([r for r in rows if r["category"] == category], cases),
                                       "summary": summarize([r for r in rows if r["category"] == category])}
                            for category in sorted({case["category"] for case in cases.values()})},
            "failures": dict(failures), "misses": [{"case_id": row["case_id"], "pass": row["pass"],
                                                     "status": row["status"], "failures": row["score"]["failures"]}
                                                    for row in rows if row["score"].get("passed") is not True],
            "peak_global_gpu_memory_used_mb": max(gpu_peaks) if gpu_peaks else None,
            "median_model_reported_size_vram_bytes": statistics.median(model_sizes) if model_sizes else None,
            "resource_observations_with_samples": sum(r.get("sample_count", 0) > 0 for r in resource_rows),
            "resource_observations_without_samples": sum(r.get("sample_count", 0) == 0 for r in resource_rows),
            "session_ranges": model.get("session_ranges"),
            "vram_scope": "Global NVIDIA peak includes other applications; /api/ps model placement is separate.",
            "all_retained_observations_verified": True}


def report(root=ROOT):
    root = Path(root)
    manifest, cases, system, schema = load_frozen(root)
    by_id = {case["id"]: case for case in cases}
    base = verified_artifact(root / "reports/llm/model_comparison.json")
    if base["benchmark_id"] != manifest["id"] or base["generation_id"] != manifest["generation_id"]:
        raise ValueError("comparison_benchmark_mismatch")
    models = [cohort(root, model, by_id) for model in base["models"]]
    result = {"created_at": timestamp(), "benchmark_id": manifest["id"], "generation_id": manifest["generation_id"],
              "main_comparison_sha256": sha256((root / "reports/llm/model_comparison.json").read_bytes()),
              "protected_state_unchanged": base["protected_state_unchanged"], "cases": len(cases),
              "passes": base["passes"], "models": models, "candidate": None, "budget600_candidate": None, "winner": None,
              "execution_mode": base.get("execution_mode", "single_comparison_process"),
              "comparison_limitations": base.get("limitations", []),
              "activation_performed": False, "public_answers_enabled": False,
              "limitations": ["Deterministic archived-target coverage is not qualified legal relevance review.",
                              "Translations and short-unit probes are exploratory; attacks/conflicts are synthetic.",
                              "Fixed-input model timings exclude retrieval, embeddings, archive validation and API queueing.",
                              "A source-valid quote can be irrelevant; precision and positive-case coverage must both be read."]}
    candidate_path = root / "reports/llm/citation_selection_comparison.json"
    if candidate_path.is_file():
        from ipindia_corpus.citation_selection import derive_case
        candidate = verified_artifact(candidate_path)
        if candidate["base_comparison_sha256"] != result["main_comparison_sha256"]:
            raise ValueError("candidate_uses_different_base_comparison")
        derived = {case["id"]: derive_case(case, system, schema)[0] for case in cases}
        result["candidate"] = cohort(root, candidate, derived)
        result["candidate"]["variant"] = candidate["variant"]
        result["candidate"]["forced_empty_case_ids"] = [b["case_id"] for b in candidate["case_bindings"] if b["forced_empty_selection"]]
        forced = set(result["candidate"]["forced_empty_case_ids"])
        result["candidate"]["non_forced_quality"] = quality(
            [row for row in candidate["observations"] if row["case_id"] not in forced], derived)
        result["candidate"]["schema_forced_quality"] = quality(
            [row for row in candidate["observations"] if row["case_id"] in forced], derived)
        result["candidate"]["forced_empty_abstention_caveat"] = "Empty selections in these cases are grammar constraints, not intrinsic model abstention evidence."
        budget_path = root / "reports/llm/citation_budget600_comparison.json"
        if budget_path.is_file():
            budget = verified_artifact(budget_path)
            if (budget["parent_enum_sha256"] != sha256(candidate_path.read_bytes())
                    or budget["base_comparison_sha256"] != result["main_comparison_sha256"]
                    or budget["case_bindings"] != candidate["case_bindings"]
                    or budget["model"] != candidate["model"] or budget["model_digest"] != candidate["model_digest"]
                    or budget["profile"] != {**candidate["profile"], "num_predict": 600}):
                raise ValueError("budget_followup_does_not_match_parent")
            extra = cohort(root, budget, derived)
            extra["variant"] = budget["variant"]
            extra["forced_empty_case_ids"] = sorted(forced)
            extra["non_forced_quality"] = quality([r for r in budget["observations"] if r["case_id"] not in forced], derived)
            extra["schema_forced_quality"] = quality([r for r in budget["observations"] if r["case_id"] in forced], derived)
            result["budget600_candidate"] = extra
    save_artifact(root, "reports/llm/quality_summary.json", result)
    return result


if __name__ == "__main__":
    result = report()
    print(json.dumps({"models": len(result["models"]), "candidate_measured": result["candidate"] is not None,
                      "winner": None, "protected_state_unchanged": result["protected_state_unchanged"]}, indent=2))
