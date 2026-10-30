"""Evidence-bound Milestone 3 release report. Missing review never becomes a pass."""
from collections import Counter
import json

from ipindia_corpus.llm_experiment import ROOT, percentile, timestamp
from .common import DEST, protected, read, sha, write
from tools.reliability_v2 import cases


def run():
    review = read(ROOT / DEST / "review_status.json")
    source = read(ROOT / DEST / "source_certification_candidates.json")
    experiment = {v: read(ROOT / DEST / f"experiment-{v}.json") for v in "ABC"}
    initial = read(ROOT / DEST / "initial-dependency-failure.json")
    offline = read(ROOT / DEST / "offline_checks.json")
    postgres = read(ROOT / DEST / "postgres_checks.json")
    retrieval = read(ROOT / DEST / "retrieval_checks.json")
    sync = read(ROOT / DEST / "sync_offline_validation.json")
    protection = protected()
    operational = read(ROOT / DEST / "final_state.json")
    development = {c["id"]: c for c in cases()["development"]}
    citations = []
    audited, failures, unavailable = 0, 0, 0
    comparison = {}
    for variant, report in experiment.items():
        summary = report["summary"]
        rows = report["rows"]
        for row in rows:
            audit = row["result"].get("evaluation_citation_audit")
            if audit is None:
                unavailable += 1
            elif not audit["passed"]:
                failures += 1
            else:
                audited += audit["verified_citations"]
            citations.extend(ci for cl in row["result"].get("claims", []) for ci in cl.get("citations", []))
        raw_called = [r["raw_selection_score"] for r in rows if r.get("raw_selection_score") and r["raw_selection_score"].get("provider_called")]
        invocation_times = [r["http_or_pipeline_ms"] for r in rows if r.get("model_called")]
        answerable_times = [r["http_or_pipeline_ms"] for r in rows if development[r["case_id"]]["expected"]["policy"] == "quote_targets"]
        comparison[variant] = {"candidate_sha256": report["candidate"]["candidate_sha256"],
                               "summary": summary, "raw_model_selection": {"calls": len(raw_called),
                                   "strict_pass_rate_on_called_requests": sum(bool(s.get("passed")) for s in raw_called) / len(raw_called) if raw_called else None,
                                   "guard_credit": False},
                               "model_invoking_http_p95_ms": percentile(invocation_times, .95),
                               "answerable_http_ms": {"p50": percentile(answerable_times, .5), "p95": percentile(answerable_times, .95)},
                               "backend_reasons": dict(Counter(r["result"].get("reason") or "answered" for r in rows)),
                               "citation_support_correctness": None, "unsupported_citation_rate": None,
                               "semantic_redundancy": None, "reason_semantic_metrics_unavailable": "no_completed_human_relevance_review"}
    verified = failures == unavailable == 0
    beneficial = {v: experiment[v]["summary"]["answerable_coverage"] > experiment["A"]["summary"]["answerable_coverage"]
                  and experiment[v]["summary"]["system_case_success"] >= experiment["A"]["summary"]["system_case_success"] for v in "BC"}
    quality = any(r["summary"]["quality_gates_passed"] for r in experiment.values())
    gates = {"activation_allowed": False, "quality_threshold": .95,
             "development_candidate_quality_gate": quality,
             "independent_relevance_review_completed": False,
             "independent_holdout_evaluated": False,
             "citation_provenance": {"passed": verified, "verified": audited, "failed": failures, "unassessable_attempts": unavailable},
             "quotation_fidelity_on_evaluated_outputs": verified, "search_no_regression": retrieval["passed"],
             "warm_answer_p95_within_5000ms": {v: r["summary"]["warm_complete_ms"]["p95"] <= 5000 for v, r in experiment.items()},
             "safety_critical_abstention": {v: r["summary"]["safety_critical_abstention"] == 1 for v, r in experiment.items()},
             "offline_and_lint": offline["passed"], "postgres_indexing_failure_rollback": postgres["passed"],
             "historical_preservation": protection["passed"], "source_certification": False,
             "production_service_and_volume_preserved": operational["passed"],
             "public_security_certification": False, "three_source_ready": False,
             "source_certification_details": source["sources"],
             "initial_failed_attempts_retained": initial["retained_attempts"],
             "reason": "Independent review/holdout and three-source certification incomplete; development measurements cannot authorize activation."}
    write("selection_comparison.json", {"cohort": "development_only_three_passes", "variants": comparison,
                                       "independently_reviewed_support": False, "independent_variable_benefit": beneficial,
                                       "D_combined": {"executed": False, "reason": "independent_correction_benefit_not_established" if not all(beneficial.values()) else "requires_separate_frozen_protocol"},
                                       "retained_initial_dependency_cohort": initial})
    write("candidate_release_gates.json", gates)
    performance = {"historical_search": {"recall_at_5": 29 / 30, "warm_p95_ms": 538},
                   "current_search": {k: retrieval[k] for k in ("recall_at_5", "warm_p95_ms", "generation_id", "ranking_version")},
                   "historical_pooled_http_p95_ms": 3521.9381999922916,
                   "development_http": {v: r["summary"] for v, r in experiment.items()},
                   "initial_preview_initialization_failure": initial,
                   "startup": {v: read(ROOT / DEST / f"startup-{v}.json") for v in "ABC"},
                   "runtime": read(ROOT / DEST / "runtime.json"),
                   "hardware_receipt_origin": "current runtime receipt plus per-cohort resource observations",
                   "causal_speedup_claim": False, "startup_included_in_warm_latency": False}
    write("performance_comparison.json", performance)
    parser_observed = [r for r in experiment["B"]["rows"] if r["result"].get("reason") == "incomplete_requested_provisions"]
    parser = {"new_structural_tests": 19, "offline_suite_passed": offline["tests"]["exit_code"] == 0,
              "parser_candidate": "tools/milestone3/references.py", "source_anchors_changed": False,
              "development_A_required_coverage": experiment["A"]["summary"]["required_evidence_coverage"],
              "development_B_required_coverage": experiment["B"]["summary"]["required_evidence_coverage"],
              "incomplete_guard_attempts": [{"case_id": r["case_id"], "pass": r["pass"],
                                            "before": r["result"].get("missing_before_selection"), "after": r["result"].get("missing_after_selection")} for r in parser_observed],
              "syntactic_reference_recall_is_legal_coverage": False}
    replay = read(ROOT / DEST / "parser_retained_replay.json")
    parser["new_offline_replay"] = {"attempts": len(replay["rows"]),
                                    "confirmed_implicit_case": [r for r in replay["rows"] if r["case_id"] == "reliability-positive-30"],
                                    "historical_scores_changed": False, "model_calls": 0}
    write("parser_regression_results.json", parser)
    write("sync_readiness.md", SYNC_READINESS + f"\nExecuted event fixture scenarios: {sync['executed']}; passed={sync['passed']}.\n")
    write("sync_blockers.md", """# Synchronization blockers

All source access reviews remain unresolved. Fresh checks stay deferred and the
worker remains stopped. No unattended source traffic or automatic promotion is
enabled. Capture receipts and fidelity checks still block India Code/TKDL admission.
Candidate publication is implemented; isolated candidate indexing/embedding and
review/promotion orchestration is not a single fully automated worker flow.
Two successful permitted incremental live cycles remain unexecuted per source.
""")
    lines = ["# AyurIP Milestone 3", "", "Engineering work completed; release remains blocked. `activation_allowed=false`.", "",
             f"Review handoff: {review['queued_question_count']} disputed questions, {review['queued_question_citation_quotes']} citation/quote pairs including {review['frozen_label_mismatch_citation_quotes']} frozen-label mismatches.",
             "Real human reviews completed: **0**. Necessary / acceptable / redundant / irrelevant / unsupported counts are **unavailable**, not zero findings.",
             "Stage attribution inspects 1,260 retained attempts without changing historical labels or scores. The original 31-question/93-attempt disagreement count remains preserved; the union adds one question from Milestone 2.",
             "Frozen-target retrieval/selection/verifier losses are observable technical failures. Whether an alternative citation legally satisfies a question remains unresolved until relevance review.", "",
             "## Development experiments", "", "30 unique questions per variant; three repeated attempts per question. Repeats are not independent samples.", "",
             "| Variant | Full-case | Strict answerable | Required coverage | HTTP p95 ms | Model-invoking p95 ms |",
             "|---|---:|---:|---:|---:|---:|"]
    for v, r in experiment.items():
        s = r["summary"]
        p = comparison[v]["model_invoking_http_p95_ms"]
        lines.append(f"| {v} | {s['system_case_success']:.1%} | {s['answerable_coverage']:.1%} | {s['required_evidence_coverage']:.1%} | {s['warm_complete_ms']['p95']:.0f} | {p:.0f} |")
    lines += ["", "A: protected M2 baseline. B: parser/completeness correction. C: evidence grouping. Retrieval stays ranker v2 / top five; no new model, normalization or multi-query was introduced.",
              "The first 270 timed attempts used an unready isolated search dependency. They remain immutable and are reported separately as operational failures (40% full-case; 0% answerable), not ranker/model results. A repaired admission harness was frozen before a separate cohort; no failed slot was overwritten.",
              "No combined candidate was run. Semantic redundancy/support cannot be scored automatically from provenance or frozen-label membership.", "",
              "Parser repair improves detection of the omitted Rule 16 in a retained regression replay; it does not improve measured development target coverage (86.7% for all three variants). Grouping does not improve strict correctness, so it is rejected as a default.", "",
              "## Verification and release decision", "",
              f"Search: {retrieval['recall_at_5']:.2%} Recall@5; {retrieval['warm_p95_ms']:.0f} ms warm p95 (historical 538 ms).",
              f"Emitted citations: {audited} verified, {failures} failed audits, {unavailable} unassessable attempts. Exact backend copies do not establish legal relevance.",
              f"Offline/lint: {offline['passed']}; disposable PostgreSQL/rollback: {postgres['passed']}; eight offline sync scenarios: {sync['passed']}; protected {protection['checked_files']} historical files: {protection['passed']}.",
              "Local security regressions are covered by finite tests, not public security certification. Main UI/API were unchanged; no UI tests were required for the private tools.",
              "India Code: 8 staged / 0 certified. TKDL: 51 staged / 0 certified; 43 timeout markers and 8 sparse/navigation flags require review/recapture. No live source access was attempted.",
              "Synchronization remains access-blocked and stopped; local candidate/ledger fixture checks pass. Candidate indexing and promotion remain separate explicit steps.",
              "The 60 new proposals use 30 new primary provision families with no old target-family overlap. They are AI-authored, unreviewed and unevaluated. No reviewed gold file was created.", "",
              "## What is justified next", "",
              "Keep the working search generation and ranker. Adopt the offline review workflow, immutable diagnostics and dependency-admission check as engineering tools. Keep parser/grouping behavior experimental pending measured benefit and human review.",
              "Reject deeper retrieval, multi-query, broad normalization and larger-model defaults without new specific evidence. Do not combine changes whose independent benefit is absent.",
              "Next release blockers: independent citation adjudication and source-bound benchmark review; 95% integrated and answerable quality gates; full three-source provenance/fidelity; permitted incremental cycles; public security/deployment review before exposure.", "",
              "Reproduce preparation: `python -m tools.milestone3.review export`, `python -m tools.milestone3.attribution`, `python -m tools.milestone3.preparation`.",
              "Development measurements: frozen `tools.milestone3.experiments matrix --passes 3` against the isolated preview. Existing intents prevent retries; preserve receipts before starting any distinct future experiment.",
              "Checks: `python -m tools.milestone3.checks offline`, `sync`, `retrieval`; run `postgres` inside the existing API container with its configured database connection."]
    write("executive_summary.md", "\n".join(lines) + "\n")
    write("next_milestone_recommendation.md", """# Next milestone

1. Human relevance/proposition review of the 41 disputed quotation pairs, using the
   complete 32-question packets. Adjudicate conflicts and preserve separate labels.
2. Human authoring/review and sealing of a genuinely independent source-family
   benchmark. Current AI drafts and old exposed sets cannot certify relevance.
3. Prioritize confirmed missing-provision selection errors using retained stages.
   Keep current top-five retrieval and Qwen model; do not tune inspected holdouts.
4. Resume source recovery only after access review and fresh-capture authorization.
   Certify India Code/public TKDL originals before unified admission.
5. Exercise isolated candidate indexing, change-only embedding, explicit promotion
   and rollback after two permitted incremental cycles. Keep live workers stopped.

The public answer gate and all 95% quality gates remain unchanged.
""")
    evidence = {p.relative_to(ROOT / DEST).as_posix(): sha(p.read_bytes()) for p in sorted((ROOT / DEST).rglob("*"))
                if p.is_file() and p.name != "evidence_index.json" and not p.name.endswith((".sha256", ".log"))}
    write("evidence_index.json", {"at": timestamp(), "files": evidence, "activation_allowed": False})
    return gates


SYNC_READINESS = """# Candidate synchronization readiness

Inspected: candidate_sync.py, sync_runner.py, sync.py, generations.py, indexer.py.
The worker is candidate-only and is stopped. Recorded access evidence permits
zero unattended sources. Daily checks and weekly discovery have independent due
times and retry deadlines; file locks prevent overlapping capture/maintenance.

Raw bytes are content-addressed and prior versions retained; conditional requests
use ETag/Last-Modified. The ledger distinguishes new bytes, unchanged content,
changed content, aliases and explicit source-version metadata. A version label
does not establish that an instrument is currently in force.

Failed/partial source snapshots do not become accepted rows. Corrupt/unparseable
responses are quarantined with history; failed indexing keeps the previous active
generation. Candidate-worker publication queues a hash-bound immutable generation
without changing filesystem production pointers or PostgreSQL active state.

The indexer can reuse embedding-cache entries by chunk hash/model revision, but
the worker does NOT automatically run isolated candidate indexing/quality gates.
That orchestration and explicit promotion remain a separate operational step.
No end-to-end live incremental operational claim is made from fixture success.

Event fixtures execute NEW_DOCUMENT, UNCHANGED, CONTENT_CHANGED, URL_MOVED,
VERSION_REPLACED (ledger NEW_VERSION), FETCH_FAILED, PARSE_FAILED, PARTIAL_CAPTURE.
Existing suites also cover retry checkpoints, overlapping locks, independent
schedules, disallowed redirects, incomplete discovery and publication failure.
"""


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=True))
