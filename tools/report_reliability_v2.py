"""Render measured Milestone 2 outcomes without changing frozen labels."""
from collections import Counter
import json

from ipindia_corpus.llm_experiment import canonical, sha256, timestamp
from tools.reliability_v2 import ROOT, DATA, DEST, write


def read(name, default=None):
    path = ROOT / DEST / name
    if not path.exists():
        return default
    raw = path.read_bytes()
    receipt = path.with_suffix(path.suffix + ".sha256")
    if not receipt.exists() or receipt.read_text().strip() != sha256(raw):
        raise ValueError("report_hash_mismatch:" + name)
    return json.loads(raw)


def table(rows):
    result = ["| Experiment | Mode / set | Attempts | Full-case | Fully correct answerable | p95 ms |",
              "|---|---|---:|---:|---:|---:|"]
    for r in rows:
        s = r["summary"]
        latency = s["warm_complete_ms"]["p95"]
        shown = f"{latency:.0f}" if isinstance(latency, (int, float)) else "unavailable"
        result.append(f"| {r['variant']} | {r['mode']} / {r['split']} | {s['attempts']} | {s['system_case_success']:.1%} | {s['answerable_coverage']:.1%} | {shown} |")
    return result


def main():
    reports = []
    for path in sorted((ROOT / DEST).glob("*-*passes.json")):
        item = read(path.name)
        checksum = item["candidate"]["candidate_sha256"]
        folder = ROOT / DATA / "observations" / checksum / f"{item['mode']}-{item['split']}-{item['passes']}passes"
        for row in item["rows"]:
            slot = f"{row['pass']}-{row['case_id']}"
            raw = canonical(row)
            expected = folder / "rows" / slot / (sha256(raw) + ".json")
            if not expected.exists() or expected.read_bytes() != raw:
                raise ValueError("unbound_observation")
        reports.append(item)
    attribution = read("failure_attribution.json")
    from tools.reliability_v2 import cases
    from tools.report_llm_results import verified_artifact
    review_cases = {c["id"]: c for group in cases().values() for c in group}
    review_queue = {}
    for split in ("historical", "development", "heldout"):
        retained = verified_artifact(ROOT / f"reports/reliability/http-{split}-3passes.json")
        for row in retained["rows"]:
            if "DISALLOWED_CLAIM" not in row["system_score"]["failures"]:
                continue
            c = review_cases[row["case_id"]]
            allowed = set(c["expected"].get("allowed_claim_chunk_ids", []))
            inventory = {h["chunk_id"]: h for h in row["result"].get("retrieval_evidence", [])}
            for claim in row["result"].get("claims", []):
                for citation in claim["citations"]:
                    if citation["chunk_id"] in allowed:
                        continue
                    key = (c["id"], citation["chunk_id"], citation["quote"])
                    review_queue[key] = {"case_id": c["id"], "question": c["question"],
                                         "original_required_targets": c["expected"]["target_groups"],
                                         "original_allowed_ids": sorted(allowed), "extra_citation": citation,
                                         "archived_passage": inventory.get(citation["chunk_id"]),
                                         "reviewer": None, "decision": "pending", "necessary_qualification": None,
                                         "disagreements": [], "historical_score_changed": False}
    write(ROOT, DEST + "/annotation_disagreements.json", {"independent_review_pending": True,
          "unique_question_citation_quotes": len(review_queue), "items": list(review_queue.values())})
    depth = read("retrieval_depth.json")
    languages = read("multilingual_retrieval.json")
    verification = {name: read(name + "_checks.json", {"pass": False, "status": "unexecuted"}) for name in ("offline", "postgres", "security")}
    final = read("final_state.json", {})
    selection = read("selection_decision.json", {})
    metrics = [{k: r[k] for k in ("variant", "mode", "split", "passes", "candidate", "summary")} for r in reports]
    write(ROOT, DEST + "/ablation_results.json", {"at": timestamp(), "experiments": metrics, "activation_performed": False})
    before_after = []
    citation_count, citation_failures, unassessable_citations = 0, [], []
    for r in reports:
        if r["passes"] != 3:
            continue
        old_split = "heldout" if r["split"] == "heldout_regression" else r["split"]
        old = verified_artifact(ROOT / f"reports/reliability/{r['mode']}-{old_split}-3passes.json")
        before_after.append({"mode": r["mode"], "set": r["split"], "before": old["summary"], "after": r["summary"],
                             "comparability": "Same frozen cases and pinned model; different runs/hardware load, not a paired causal latency estimate."})
        for row in r["rows"]:
            audit = row["result"].get("evaluation_citation_audit", {})
            if not audit:
                unassessable_citations.append({"case_id": row["case_id"], "pass": row["pass"], "mode": r["mode"], "set": r["split"]})
            citation_count += audit.get("verified_citations", 0)
            if audit.get("passed") is False:
                citation_failures.append({"case_id": row["case_id"], "pass": row["pass"], "mode": r["mode"], "set": r["split"]})
    write(ROOT, DEST + "/before_after.json", {"cohorts": before_after, "metrics_not_relaxed": True})
    comparison = []
    if before_after:
        comparison = ["", "## Same-case historical comparison", "", "| Mode / set | Before full-case | After | Before answerable | After | Before p95 ms | After |",
                      "|---|---:|---:|---:|---:|---:|---:|"]
        for r in before_after:
            b, a = r["before"], r["after"]
            latency = a["warm_complete_ms"]["p95"]
            shown = f"{latency:.0f}" if isinstance(latency, (int, float)) else "unavailable"
            comparison.append(f"| {r['mode']} / {r['set']} | {b['system_case_success']:.1%} | {a['system_case_success']:.1%} | {b['answerable_coverage']:.1%} | {a['answerable_coverage']:.1%} | {b['warm_complete_ms']['p95']:.0f} | {shown} |")
        comparison += ["", "Same cases; different run conditions. No causal speedup is inferred. Exploratory one-pass timing included postresponse evaluator auditing; final three-pass timing excludes it."]
    lines = ["# Milestone 2 measured experiments", "", "Each report binds code, corpus, model and observations. Earlier frozen scores remain unchanged.", ""] + table(reports)
    lines += ["", "One-pass development ablations are exploratory. Three-pass results measure repeatability, not additional independent questions.",
              "The new 60-case package is sealed but unreviewed and reuses earlier source targets. It is ineligible for independent certification.",
              "Each row retains its own source/configuration hash. Rejected exploratory variants were measured before the final evaluator timing and identifier-validation hardening; their final-code variants were not re-benchmarked or adopted.",
              "Context token statistics come from model telemetry. Resource sampling is bounded and can omit long-cohort tails; hardware snapshots include other desktop applications."]
    write(ROOT, DEST + "/performance_comparison.md", "\n".join(lines + comparison) + "\n")
    if depth:
        target_ids = {c["id"]: {cid for g in c["expected"].get("target_groups", []) for cid in g["required_chunk_ids"]}
                      for group in cases().values() for c in group}
        mrr = {}
        for k in (5, 8, 10, 15):
            reciprocal = []
            for row in depth["rows"]:
                if row["depth"] != k:
                    continue
                first = next((n for n, cid in enumerate(row["hit_ids"], 1) if cid in target_ids[row["case_id"]]), None)
                reciprocal.append(1 / first if first else 0)
            mrr[str(k)] = sum(reciprocal) / len(reciprocal)
        write(ROOT, DEST + "/retrieval_mrr.json", {"frozen_target_mrr_at_depth": mrr,
              "origin": "offline derived from retained retrieval replay; no additional requests or tuning",
              "scope": "first annotated target ID; does not establish all-group coverage or independent semantic relevance",
              "ndcg": "not reported: retrieved background has unresolved relevance labels; no graded judgments"})
        lines = ["# Retrieval depth", "", "15 answerable development questions, unchanged ranker v2; retrieval-only diagnostics.", "",
                 "| K | Target-group recall | Complete questions | Packed recall | Target MRR | p95 ms |", "|---|---:|---:|---:|---:|---:|"]
        for k, s in depth["summary"].items():
            lines.append(f"| {k} | {s['target_group_recall']:.1%} | {s['complete_questions']} | {s['packed_group_recall']:.1%} | {mrr[k]:.3f} | {s['p95_ms']:.0f} |")
        lines += ["", "Smallest depth with maximal target coverage: " + str(depth["best_depth"]),
                  "These are new diagnostic replays, not historical ranks. Target MRR is derived offline from retained hit IDs and is not all-group recall. Unreviewed background/graded labels do not support an independently reviewed nDCG. Token counts require real model telemetry."]
        write(ROOT, DEST + "/retrieval_depth_comparison.md", "\n".join(lines) + "\n")
    if languages:
        lines = ["# Multilingual retrieval diagnostic", "", "Twelve paired queries reuse development targets; no independent multilingual certification.", "",
                 "| Language | Original target-group recall | Normalized |", "|---|---:|---:|"]
        for language in ("English", "Hindi", "romanized", "mixed"):
            values = []
            for variant in ("original", "normalized"):
                subset = [r for r in languages["rows"] if r["language"] == language and r["variant"] == variant]
                values.append(sum(r["target_group_recall"] for r in subset) / len(subset))
            lines.append(f"| {language} | {values[0]:.1%} | {values[1]:.1%} |")
        lines += ["", "Normalization changes both lexical and semantic query input in this experiment. It is not an embedding-model upgrade.",
                  "An aggregate improvement does not excuse a regression on an individual language query."]
        write(ROOT, DEST + "/multilingual_retrieval.md", "\n".join(lines) + "\n")
    for variant, name, description in (
            ("multiquery", "multiquery_experiment.md", "Original query plus at most three Qwen expansions; reject identifier/scope changes and retain fallback observations."),
            ("coverage", "evidence_selection.md", "Whole-unit packing reserves explicit provision anchors; final output must cover requested anchors. IDs and quotations remain backend-controlled.")):
        selected = [r for r in reports if r["variant"] == variant]
        write(ROOT, DEST + "/" + name, "\n".join(["# " + variant, "", description, ""] + (table(selected) if selected else ["Unexecuted; no improvement claimed."])) + "\n")
    selection_path = ROOT / DEST / "evidence_selection.md"
    write(ROOT, DEST + "/evidence_selection.md", selection_path.read_text(encoding="utf-8") + """

## Remaining limitations

The guard checks parsed root section/rule identifiers. It does not prove semantic
coverage of every subclause or implicit/coordinated reference, such as 'rules N
drawings and M models'. The retained multi-provision misses include this parser
limitation. No final result was tuned or rescored after observing it.
Ambiguous legacy metadata can also cause conservative abstention. Published
metadata equality is not independent verification of a legal anchor's meaning.
These limitations and the failed quality gates prohibit activation.
""")
    write(ROOT, DEST + "/abstention_policy.md", """# Abstention and errors

Current-law uncertainty and personal advice retain the existing pre-model policy.
Incomplete requested provisions abstain with `incomplete_requested_provisions`.
Evidence insufficiency is separate from `timeout`, `provider_unavailable`,
`invalid_model_selection`, `archive_integrity_failure` and `retrieval_integrity_failure`.
Runtime/integrity errors use status `unavailable`; no partial legal answer is emitted.
The baseline path remains unchanged for controlled comparison. These finite tests
do not establish universal multilingual safety or immunity to prompt injection.
""")
    write(ROOT, DEST + "/security_regression.md", "# Security verification\n\n" + "\n".join(f"- {k}: {'passed' if v.get('pass') else v.get('status', 'failed')} (see {k}_checks.json)." for k, v in verification.items()) + "\n\nPublic answers and live synchronization remain disabled. UI code was unchanged; no new visual certification is claimed.\n")
    write(ROOT, DEST + "/next_milestone_plan.md", """# Remaining source and release work

1. Obtain independent relevance review and seal a genuinely unexposed source-bound
evaluation. Reused query targets are not an independent source-family test.
2. Resolve permitted access reviews before capturing India Code or public TKDL.
Certify originals, section/page fidelity, timeout rejection and provenance before admission.
3. Freeze a unified benchmark with at least ten reviewed passage targets per source.
4. Resume candidate-only synchronization only when authorized; validate two incremental
cycles, separate schedules, explicit promotion and rollback. No live fetch was enabled here.
5. Expand beyond copied extracts only after answer quality and citation gates pass.
Independent legal review, public security/load testing and deployment remain outstanding.

Stored offline certification evidence records eight staged India Code files and
51 TKDL files, with zero admitted/certified for both; five user PDFs also lack
verified official capture receipts. This is inspection of the existing report,
not a new live-access check. Candidate synchronization is configured but its
worker remains stopped; the release scheduler stays disabled and access reviews
remain unresolved. See reports/corpus_certification.md and candidate_sync_status.json.
""")
    failures = Counter(f for r in reports for row in r["rows"] for f in row["system_score"]["failures"])
    gates = {"independent_review_complete": False,
             "new_unbiased_quality_evaluation": False,
             "offline": verification["offline"]["pass"], "postgres": verification["postgres"]["pass"],
             "security": verification["security"]["pass"],
             "three_source_certified": False, "production_preserved": final.get("production_state_unchanged", False)}
    final_cohorts = [r for r in reports if r["passes"] == 3]
    raw_model = []
    interventions = []
    for r in final_cohorts:
        eligible = [row["raw_selection_score"] for row in r["rows"] if (row.get("raw_selection_score") or {}).get("provider_called")
                    and row["raw_selection_score"].get("passed") is not None]
        if r["mode"] == "controlled":
            raw_model.append({"set": r["split"], "provider_called_attempts": len(eligible),
                              "raw_selection_pass_rate": sum(s["passed"] for s in eligible) / len(eligible) if eligible else None,
                              "backend_guards_excluded": sum(not (row.get("raw_selection_score") or {}).get("provider_called", False) for row in r["rows"]),
                              "scope": "first span-selection pass; not the verifier or integrated system"})
        interventions.append({"mode": r["mode"], "set": r["split"],
                              "blocked_before_model": r["summary"]["backend_blocked_before_model"],
                              "reasons": dict(Counter(row["result"].get("reason") for row in r["rows"] if row["result"].get("reason")))})
    write(ROOT, DEST + "/model_and_backend_outcomes.json", {"raw_model_controlled": raw_model, "backend": interventions,
          "backend_guards_never_count_as_model_success": True, "http_raw_model_score": "not computed against controlled evidence; real retrieval differs"})
    gates.update({"all_six_regression_cohorts_complete": len(final_cohorts) == 6,
                  "strict_regression_quality": len(final_cohorts) == 6 and all(r["summary"]["quality_gates_passed"] for r in final_cohorts),
                  "citation_integrity": bool(final_cohorts) and not citation_failures and not unassessable_citations,
                  "warm_http_p95": any(r["mode"] == "http" for r in final_cohorts) and all(
                      isinstance(r["summary"]["warm_complete_ms"]["p95"], (int, float))
                      and r["summary"]["attempts_with_latency"] == r["summary"]["attempts"]
                      and r["summary"]["warm_complete_ms"]["p95"] <= 5000 for r in final_cohorts if r["mode"] == "http"),
                  "retrieval": read("retrieval_checks.json", {}).get("pass", False)})
    release = {"at": timestamp(), "scope": "isolated IP India engineering experiments", "gates": gates,
               "certified": False, "activated": False, "public_answers_enabled": False,
               "independent_review_status": "pending; no reviewer approval fabricated", "quality_thresholds": {"full_case": .95, "fully_correct_answerable": .95},
               "experiments": metrics, "failures": dict(failures), "historical_attribution": attribution,
               "verified_emitted_citations": citation_count, "citation_audit_failures": citation_failures,
               "unassessable_citation_attempts": unassessable_citations,
               "selection_decision": selection, "before_after": before_after,
               "limitations": ["Existing heldout is regression only", "Development experiments share targets", "Repeated passes not independent samples",
                               "First cold startup failure retained", "No current-law or three-source certification"]}
    write(ROOT, DEST + "/candidate_release_report.json", release)
    recommendation = ["# Milestone 2 recommendation", "", "Keep the working production ranker and public answer block. No candidate is certified or activated.", ""] + table(reports)
    recommendation += ["", "## Release blockers", "", "Independent source-bound relevance review and a genuinely new sealed evaluation remain outstanding.",
                       "The 95% quality thresholds remain in force. See candidate_release_report.json for measured cohorts and verification.",
                       "", "## Reproduction", "", "```powershell", "python -m tools.reliability_v2 snapshot", "python -m tools.reliability_v2 audit",
                       "python -m tools.reliability_v2 seal", "docker compose exec -T api python -m tools.reliability_v2_runtime prepare",
                       "python -m tools.reliability_v2_runtime depth", "python -m tools.reliability_v2_runtime language",
                       "python -m tools.reliability_v2_matrix --variants baseline depth8 coverage normalized multiquery --mode http --split development --passes 1",
                       "python -m tools.reliability_v2_checks offline", "docker compose exec -T api python -m tools.reliability_v2_checks postgres",
                       "python -m tools.report_reliability_v2", "```", "",
                       "Start the isolated search preview on loopback port 8002 before measurements. The matrix owns its private 8001 child processes.",
                       "Existing observation intents are never retried. Do not rerun completed experiments to select a favorable score."]
    recommendation += ["", "## Experiment decisions", "", "Keep depth five and the explicit completeness guard isolated. It has not demonstrated the required quality improvement.",
                       "Reject depth eight as an answer default despite its retrieval-coverage gain. Depth ten/fifteen add no development target coverage.",
                       "Reject general normalization as a default: it helps several romanized/mixed queries but worsens Hindi and introduces abstentions.",
                       "Reject multi-query as a default: it worsens strict development quality and exceeds the latency target. No combined default is justified.",
                       "", f"Final-cohort citation audits: {citation_count} verified; {len(citation_failures)} failed attempts. These are source-integrity checks, not legal-relevance approval."]
    write(ROOT, DEST + "/final_recommendation.md", "\n".join(recommendation + comparison) + "\n")
    counts = attribution["counts"]
    findings = ["# AyurIP Reliability Milestone 2", "", "Engineering experiments completed; answer certification and activation failed. All work used the existing archive.", "",
                "## What the experiments established", "",
                "1. The controlled-to-HTTP gap includes extra real citations outside frozen labels, missing retrieved targets, selection omissions and verifier rejection. It is not explained solely by model capability.",
                f"2. Historical retrieval misses: {counts['RETRIEVAL_TARGET_ABSENT_FROM_RETURNED_HITS']['unique_questions']} unique questions / {counts['RETRIEVAL_TARGET_ABSENT_FROM_RETURNED_HITS']['failed_attempts']} repeated failed attempts. Rank replays separately distinguish fused-pool absence and ranking below five.",
                f"3. Observed historical packing losses: {counts.get('PACKING_LOSS', {}).get('failed_attempts', 0)}. Missing historical stages are not fabricated.",
                f"4. Selection omissions: {counts['EVIDENCE_SELECTION_OMISSION']['unique_questions']} questions / {counts['EVIDENCE_SELECTION_OMISSION']['failed_attempts']} attempts; verifier omissions: {counts['SUBSET_VERIFICATION_OMISSION']['unique_questions']} / {counts['SUBSET_VERIFICATION_OMISSION']['failed_attempts']}. Categories overlap.",
                f"5. Extra citations need independent relevance review on {counts['EXTRA_CITATION_REQUIRES_RELEVANCE_REVIEW']['unique_questions']} questions / {counts['EXTRA_CITATION_REQUIRES_RELEVANCE_REVIEW']['failed_attempts']} attempts. This does not mean every extra is legally appropriate or every label is incorrect.",
                "6. Depth eight improved development retrieval coverage from 86.7% to 93.3%, but lowered strict answerable success from 66.7% to 46.7%. Ten/fifteen added no target coverage.",
                "7. Multi-query worsened strict development answerable success to 60% and p95 to 7.25 seconds. It was rejected as a default.",
                "8. Normalization helped romanized/mixed diagnostics but worsened Hindi, including one previously successful query. It did not improve strict development success; no multilingual certification is claimed.",
                "9. Coverage-aware packing/final checks improved the controlled historical cohort, but did not reduce the five extra-citation development failures or improve real HTTP answerable success. Implicit/coordinated references remain a parser limitation.",
                "10. Inappropriate abstentions were not eliminated. Completeness checks can abstain earlier on missing/ambiguous anchors. All designated safety-critical cases abstained correctly; backend guards receive no model credit.",
                f"11. Unchanged search: 96.67% Recall@5 / {read('retrieval_checks.json', {}).get('warm_p95_ms', 0):.0f} ms warm p95. Pooled final HTTP p95: {final.get('pooled_http_p95_ms', 0):.0f} ms. Startup and answerable/model-invoking latency are reported separately.",
                "12. Offline, disposable PostgreSQL and idle boundary checks passed. Earlier concurrent-load probes returned 429 and remain retained separately; finite local checks do not establish public security or injection immunity.",
                "13. Keep the production ranker/service. Retain diagnostics, immutable observations and the candidate completeness guard for further development; do not activate the answer candidate.",
                "14. Reject deeper retrieval, broad normalization and multi-query as current defaults. No combination is justified by these development results.",
                "15. Remaining blockers: independent relevance review/new evaluation, failed 95% integrated quality gates, parser/selection gaps, India Code/TKDL certification and subsequent public-release requirements.",
                "", "## Integrity and verification", "",
                f"{citation_count} final emitted citations verified against archived text, raw hash, document/title, official URL and published anchors; {len(citation_failures)} failed audits and {len(unassessable_citations)} unassessable attempts. These checks do not establish current law or semantic completeness.",
                "250 offline tests passed; two database tests passed separately; Ruff passed. The active generation, original database volume and protected baseline are preserved. Both previews are stopped, public answers stay blocked and synchronization remains paused.",
                "The new 60-case seed package is sealed but unreviewed, reuses earlier source targets and is not an independent final estimate. All existing labels and reports remain intact.",
                "", "Details: final_recommendation.md, candidate_release_report.json, runtime.json, model_and_backend_outcomes.json and final_state.json."]
    write(ROOT, DEST + "/milestone.md", "\n".join(findings + comparison) + "\n")
    evidence = {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()) for p in (ROOT / DEST).iterdir()
                if p.is_file() and p.suffix in (".json", ".jsonl", ".md") and p.name != "evidence_index.json"}
    write(ROOT, DEST + "/evidence_index.json", {"reports_sha256": evidence,
          "runtime_sha256": evidence.get(DEST + "/runtime.json"), "citation_scope": "final 630-attempt frozen candidate",
          "archive_and_model_binding": read("candidate.json"), "historical_benchmarks_modified": False})
    return {"reports": len(reports), "certified": False, "gates": gates}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
