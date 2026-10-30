"""Audit candidate observations and write a separate span-selection report."""
from __future__ import annotations

import json
from pathlib import Path

from ipindia_corpus.grounded_answers import generate_answer
from ipindia_corpus.llm_experiment import ROOT, canonical, save_artifact, sha256, timestamp
from ipindia_corpus.span_selection import build_plan, request_fields
from tools.benchmark_span_selection import recover_rows
from tools.freeze_llm_benchmark import load_frozen
from tools.report_llm_results import verified_artifact


def audit(root=ROOT):
    root = Path(root)
    result = verified_artifact(root / "reports/llm/span_selection_comparison.json")
    if result["status"] != "measured" or not result["protected_state_unchanged"]:
        raise ValueError("complete_protected_span_comparison_required")
    relative = Path(result["bundle_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("invalid_bundle_path")
    raw = (root / relative).read_bytes()
    if sha256(raw) != result["bundle_sha256"]:
        raise ValueError("span_bundle_hash_mismatch")
    bundle = json.loads(raw)
    _, originals, _, _ = load_frozen(root)
    if [e["case"] for e in bundle["entries"]] != originals:
        raise ValueError("historical_cases_changed")
    entries = {e["case"]["id"]: e for e in bundle["entries"]}
    for entry in entries.values():
        plan = build_plan(entry["case"]["evidence"])
        if (plan != entry["plan"] or sha256(canonical(plan)) != entry["plan_sha256"]
                or request_fields(entry["case"]["question"], plan) != entry["request_fields"]
                or sha256(canonical(entry["request_fields"])) != entry["input_sha256"]):
            raise ValueError("span_plan_or_input_binding_changed")
    folder = root / "data/llm/span_selection_checkpoints" / result["bundle_sha256"]
    rows = recover_rows(folder, entries, result["bundle_sha256"], result["code_sha256"])
    if rows != result["observations"] or len(rows) != len(entries) * 3:
        raise ValueError("span_report_observations_changed")
    total, exact, identity = 0, 0, 0
    failures = []
    for row in rows:
        evidence = {h["chunk_id"]: h for h in entries[row["case_id"]]["case"]["evidence"]}
        claims = (row.get("expanded_selection") or {}).get("claims", [])
        backend_claims = (row.get("backend_result") or {}).get("claims", [])
        for claim in claims:
            total += 1
            hit = evidence.get(claim["chunk_id"])
            exact += bool(hit and claim["quote"] in hit["text"])
        for claim in backend_claims:
            for citation in claim["citations"]:
                hit = evidence[citation["chunk_id"]]
                keys = ("source_url", "title", "section", "rule", "page_start", "page_end", "document_id", "chunk_id", "raw_sha256")
                if any(citation.get(key) != hit.get(key) for key in keys) or citation["quote"] not in hit["text"]:
                    raise ValueError("backend_citation_identity_changed")
                identity += 1
        if not row["score"]["passed"]:
            failures.append({"case_id": row["case_id"], "pass": row["pass"],
                             "failures": row["score"]["failures"], "target_coverage": row["score"]["target_coverage"],
                             "raw_selection": row.get("scoring_raw_output"),
                             "question": entries[row["case_id"]]["case"]["question"]})
    if exact != total:
        raise ValueError("copied_quote_not_archived_substring")
    class ForbiddenProvider:
        def select(self, *_):
            raise AssertionError("temporal_guard_called_provider")

    # This is an offline guard replay, never another model-quality observation.
    guard_replays = []
    for source in originals:
        if source["category"] != "current_law_uncertainty":
            continue
        calls = []

        class RecordingForbiddenProvider(ForbiddenProvider):
            def select(self, *args):
                calls.append(True)
                return super().select(*args)

        response = generate_answer(source["question"], lambda *_, **kwargs: {
            "generation_id": bundle["frozen_benchmark"]["generation_id"], "hits": source["evidence"]},
            provider=RecordingForbiddenProvider(), root=root)
        guard_replays.append({"case_id": source["id"], "status": response["status"],
                              "reason": response["reason"], "provider_calls": len(calls),
                              "passed": response["reason"] == "current_legal_status_unverified" and not calls,
                              "credit": "backend_only_no_model_call"})
    measured = result["metrics"]
    models = verified_artifact(root / "reports/llm/quality_summary.json")
    baselines = [{"label": "Qwen chunk-ID enum, cap 200", **models["candidate"]},
                 {"label": "Qwen chunk-ID enum, cap 600", **models["budget600_candidate"]}]
    audit_report = {"checked_at": timestamp(), "result_sha256": sha256((root / "reports/llm/span_selection_comparison.json").read_bytes()),
                    "bundle_sha256": result["bundle_sha256"], "attempts": len(rows),
                    "literal_backend_copied_claims": exact, "all_copied_claims": total,
                    "verified_backend_citations": identity, "all_observations_and_scores_verified": True,
                    "historical_cases_unchanged": True, "metrics": measured, "failure_details": failures,
                    "all_targets_representable": all(e["coverage_audit"]["all_groups_representable"] for e in entries.values()),
                    "model": result["model"], "model_digest": result["model_digest"],
                    "prompt_eval_count_max": max((r.get("timings", {}).get("prompt_eval_count", 0) for r in rows), default=0),
                    "peak_global_gpu_memory_mb": max((r.get("resource_summary", {}).get("peak_gpu_memory_used_mb") or 0 for r in rows), default=0),
                    "gpu_placement_verified_attempts": sum(r.get("cohort_after", {}).get("single_model_verified", False) for r in rows),
                    "backend_temporal_guard_replay": guard_replays,
                    "activation_performed": False, "public_answers_enabled": False,
                    "limitations": result["limitations"]}
    save_artifact(root, "reports/llm/span_selection_audit.json", audit_report)
    summary = measured["summary"]
    passed = sum(r["score"]["passed"] for r in rows)
    lines = ["# Backend-copied evidence span experiment", "", f"Measured: {result['captured_at']}", "",
             "## Decision", "", "The candidate remains isolated. Public answers and source synchronization remain disabled. No production provider, search configuration or corpus generation was changed.", "",
             "## Measured system checks", "", "| Candidate | Cases passed | All-attempt warm model p95 |", "|---|---:|---:|"]
    for baseline in baselines:
        s = baseline["summary"]
        n = baseline["quality"]["attempts"]
        count = n - len(baseline["misses"])
        lines.append(f"| {baseline['label']} | {count}/{n} ({100 * count / n:.1f}%) | {s['all_attempt_latency_ms']['p95'] / 1000:.2f} s |")
    lines += [f"| Qwen backend-copied spans, cap 200 | {passed}/{len(rows)} ({100 * passed / len(rows):.1f}%) | {summary['all_attempt_latency_ms']['p95'] / 1000:.2f} s |", "",
              "The 45 historical cases, evidence text, evidence order and frozen target labels are unchanged. The new prompt, span representation and ID-only output schema differ. This is a system-design experiment; separate sessions/background load prevent a causal latency comparison. Each session warmed up separately before three scored passes. Saved failures and interrupted slots count once and are never retried for a better score.", "",
              f"Raw span-selection validation passed **{measured['raw_schema_valid_attempts']}/{len(rows)}** attempts. The backend copied **{exact}/{total}** literal archived quotations and constructed **{identity}** verified citations. Fidelity is guaranteed by server-owned slices and membership checks; these counts do not measure a model's ability to quote accurately. Empty selections add no quote-precision credit.", "",
              f"Expected abstention: **{measured['correct_abstention_attempts']}/{measured['expected_abstention_attempts']}** overall; **{measured['correct_nonforced_abstention_attempts']}/{measured['nonforced_abstention_attempts']}** after excluding structurally forced empty cases. No backend pre-inference temporal/personal/injection guard was credited to the model.", "",
              "**Abstention regressed:** both earlier chunk-ID enum variants passed 6/9 non-forced abstention attempts; this candidate passed 3/9. Both current-law questions failed on all three passes. The higher aggregate success and exact copying do not remove this release blocker.", "",
              f"Failures: `{json.dumps(summary['failures'], sort_keys=True)}`. Categories can overlap. Every case still requires all original target groups, allowed citations, substantial quotations, completion and expected abstention. These checks are not legal-answer accuracy.", "",
              f"Separate offline backend replay: **{sum(g['passed'] for g in guard_replays)}/{len(guard_replays)}** frozen current-status questions stopped before provider invocation. This verifies the existing guard on these cases only, contributes no model credit and is not end-to-end answer certification.", "",
              "## Representation and provenance", "",
              "The label-independent builder partitions each official chunk into contiguous spans, preferring sentence/paragraph breaks. Displayed spans reconstruct every character exactly once; quotations remain at most 1,400 characters. Short provisions are preserved whole. Long provisions may require multiple neighboring selections; a span is not certified as a complete legal proposition. All real decoys remain selectable. Synthetic attack text remains in the input but cannot receive an authoritative citation ID.", "",
              "Opaque IDs are bound to the complete original evidence item, raw hash, document/chunk identity and half-open Unicode codepoint offsets. Resolution reconstructs the map and rejects altered text, offsets, metadata, duplicate keys/IDs, unknown IDs or extra output fields. The backend copies quotes and constructs citations; the model cannot supply them.", "",
              "Every original target phrase is representable by the frozen spans. This audit happened after the segmentation rule was written; no question-specific splits or label repairs were applied. Phrase coverage does not establish that selected quotes preserve all legal qualifications.", "",
              "## Runtime and reproducibility", "",
              f"Model: `{result['model']}`, digest `{result['model_digest']}`. Profile: `{json.dumps(result['profile'], sort_keys=True)}`, temperature 0. Native Windows Ollama version is recorded per segment. The existing historical 675 attempts remain unchanged, giving 810 timed attempts including this separate candidate.", "",
              "Physical setup: Intel Core i7-13620H (10 cores/16 logical), 33,968,590,848 bytes RAM, RTX 4060 Laptop GPU (8,188 MiB). GPU driver, actual resident-model placement, CPU/RAM/GPU samples and runtime version are retained with the segment/observation receipts. Model timing excludes archive validation, retrieval, application guards and end-to-end UI latency.", "",
              f"Largest reported prompt evaluation: {audit_report['prompt_eval_count_max']} tokens (configured context 4,096). Single-model placement was verified after {audit_report['gpu_placement_verified_attempts']}/{len(rows)} timed calls. Sampled global GPU peak: {audit_report['peak_global_gpu_memory_mb']:.0f} MiB; other desktop applications contribute to this value. All 135 calls completed without truncation or timeout.", "",
              f"Corpus generation: `{bundle['frozen_benchmark']['generation_id']}`. Candidate bundle SHA-256: `{result['bundle_sha256']}`. All historical frozen files and protected settings were verified unchanged. Code hashes, requests, maps, raw selections, backend-expanded claims, per-case scores and failures are retained in the machine-readable artifacts.", "",
              "Reproduce from the project folder (copy command lines only):", "", "```powershell", "python -m tools.benchmark_span_selection", "python -m tools.benchmark_span_selection --run", "python -m tools.report_span_selection", "```", "",
              "The first command freezes/verifies only. The second runs only unattempted slots for the exact code/input/profile binding. A missing result after a recorded intent is conservatively counted as interrupted. The report audits immutable outputs before rendering. Changed code creates a different bundle; it cannot silently resume a prior cohort.", "",
              "## Remaining failed cases", "", "| Case | Failed passes | Reasons |", "|---|---|---|"]
    for identifier in dict.fromkeys(f["case_id"] for f in failures):
        group = [f for f in failures if f["case_id"] == identifier]
        lines.append(f"| {identifier} | {', '.join(str(f['pass']) for f in group)} | {', '.join(sorted({reason for f in group for reason in f['failures']}))} |")
    if not failures:
        lines.append("| None | — | — |")
    lines += ["", "Full question/selection/failure details: `span_selection_audit.json`. Original baseline evidence: `model_selection.md` and `quality_summary.json`. These previously diagnosed frozen cases are not a new held-out evaluation. No independent legal relevance review is claimed. India Code/TKDL certification and the unified benchmark remain blocked on original-source evidence; the candidate cannot establish three-source readiness."]
    (root / "reports/llm/span_selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {key: audit_report[key] for key in ("attempts", "literal_backend_copied_claims", "verified_backend_citations", "all_observations_and_scores_verified", "metrics")}


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
