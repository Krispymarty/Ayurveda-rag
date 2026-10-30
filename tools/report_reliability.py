"""Audit measured reliability cohorts; report gates without activating answers."""
from collections import Counter
import json
from pathlib import Path

from ipindia_corpus.llm_experiment import ROOT, canonical, percentile, save_artifact, sha256, timestamp
from tools.benchmark_reliability import binding, system_score, summarize
from tools.freeze_llm_benchmark import load_frozen
from tools.freeze_reliability_benchmark import load
from tools.report_llm_results import verified_artifact


def report(root=ROOT):
    root = Path(root)
    receipt = verified_artifact(root / "reports/reliability/candidate.json")
    checksum = receipt["candidate_sha256"]
    candidate = {k: v for k, v in receipt.items() if k != "candidate_sha256"}
    if candidate != binding(root) or sha256(canonical(candidate)) != checksum:
        raise ValueError("candidate_source_changed")
    cohorts, all_by_mode, cases_by_mode = [], {}, {}
    for mode in ("controlled", "http"):
        combined, combined_cases = [], []
        for split in ("historical", "development", "heldout"):
            if split == "historical":
                _, cases, _, _ = load_frozen(root)
            else:
                _, cases = load(root, split, candidate_receipt=candidate)
            measured = verified_artifact(root / f"reports/reliability/{mode}-{split}-3passes.json")
            if measured["candidate_sha256"] != checksum or len(measured["rows"]) != len(cases) * 3:
                raise ValueError("cohort_identity_or_count_mismatch")
            case_map = {c["id"]: c for c in cases}
            keys = set()
            for row in measured["rows"]:
                key = (row["pass"], row["case_id"])
                if key in keys or row["case_id"] not in case_map:
                    raise ValueError("duplicate_or_unknown_attempt")
                keys.add(key)
                if row["system_score"] != system_score(case_map[row["case_id"]], row["result"]):
                    raise ValueError("score_mismatch")
                path = root / "data/reliability/observations" / checksum / f"{mode}-{split}-3passes" / "rows" / f"{row['pass']}-{row['case_id']}" / (sha256(canonical(row)) + ".json")
                if path.read_bytes() != canonical(row):
                    raise ValueError("immutable_observation_mismatch")
            stage_names = {name for r in measured["rows"] for name in r["result"].get("timings_ms", {})}
            stats = {name: percentile([r["result"]["timings_ms"][name] for r in measured["rows"]
                                      if name in r["result"].get("timings_ms", {})], .95) for name in stage_names}
            raw_assessed = [r["raw_selection_score"] for r in measured["rows"] if (r.get("raw_selection_score") or {}).get("provider_called")]
            positive_rows = [r for r in measured["rows"] if case_map[r["case_id"]]["expected"]["policy"] == "quote_targets"]
            cohort = {"mode": mode, "split": split, "summary": summarize(measured["rows"], cases),
                      "mean_answerable_target_group_coverage": sum(r["system_score"]["target_coverage"] for r in positive_rows) / len(positive_rows),
                      "answerable_coverage_definition": "Fraction of answerable attempts passing every frozen check; target-group recall is reported separately",
                      "stage_p95_ms": stats, "raw_selection_assessable": len(raw_assessed),
                      "raw_selection_passes": sum(r.get("passed") is True for r in raw_assessed),
                      "guard_success_not_credited_to_model": True,
                      "model_calls_attempted": sum(r["result"].get("model_calls_attempted", 0) for r in measured["rows"]),
                      "source_fixture_scope": measured["source_fixture_scope"]}
            cohorts.append(cohort)
            combined.extend(measured["rows"])
            combined_cases.extend(cases)
        all_by_mode[mode], cases_by_mode[mode] = combined, combined_cases
    retrieval = verified_artifact(root / "reports/reliability/readiness_retrieval.json")
    startup = verified_artifact(root / "reports/reliability/readiness_startup.json")
    offline = json.loads((root / "reports/verification/offline.json").read_bytes())
    postgres = json.loads((root / "reports/verification/postgres.json").read_bytes())
    promotion = verified_artifact(root / "reports/reliability/readiness_promotion.json")
    security = verified_artifact(root / "reports/reliability/security_boundaries.json")
    combined = {mode: summarize(rows, cases_by_mode[mode]) for mode, rows in all_by_mode.items()}
    from tools.audit_reliability_misses import audit
    citation_audit = audit(root, all_by_mode, cases_by_mode)
    bad = Counter(reason for rows in all_by_mode.values() for row in rows for reason in row["system_score"]["failures"])
    unavailable = sum(row["result"].get("status") == "unavailable" for rows in all_by_mode.values() for row in rows)
    citation_gate = not unavailable and not any(bad[x] for x in ("FABRICATED_CITATION", "FABRICATED_QUOTE", "ALTERED_SOURCE_ID", "UNSAFE_OR_UNVERIFIED_SELECTION"))
    gates = {"search_recall_no_regression": retrieval["no_historical_recall_regression"],
             "warm_search_p95": retrieval["warm_p95_ms"] <= 2000,
             "startup_within_60s": startup["ready"]["readiness"]["initialization_ms"] <= 60000,
             "cold_requests_fail_promptly": startup["observed_initializing_503"] and all(p["ms"] < 1000 for p in startup["concurrent_probes"]),
             "citation_integrity": citation_gate,
             "local_endpoint_boundaries": security["pass"],
             "offline": offline["pass"], "disposable_postgres": postgres["pass"],
             "all_answer_cohorts_quality": all(c["summary"]["quality_gates_passed"] for c in cohorts),
             "warm_complete_answer_p95": combined["http"]["warm_complete_ms"]["p95"] <= 5000}
    result = {"at": timestamp(), "candidate_sha256": checksum, "gates": gates,
              "configuration_sha256": sha256(canonical(candidate["configuration"])),
              "answer_candidate_passed_all_gates": all(gates.values()), "cohorts": cohorts,
              "combined": combined, "search": {k: v for k, v in retrieval.items() if k != "rows"},
              "startup": startup["ready"], "failure_counts": dict(bad),
              "citation_and_miss_audit": citation_audit,
              "unavailable_attempts_not_assumed_citation_valid": unavailable,
              "activation_performed": False, "public_answers_enabled": False,
              "activation_scope": "Answer candidate only; search readiness was promoted independently",
              "search_readiness_promoted": promotion["readiness_only_promoted"],
              "fresh_synchronization_performed": False, "all_legal_statuses": "unknown",
              "limitations": ["New questions reuse existing archived passages; not independent legal review or a new held-out corpus.",
                              "Held-out targets were frozen before implementation; only development results informed changes.",
                              "Controlled evidence and real retrieval are evaluated separately.",
                              "Backend request policy and copied text are system safeguards, not intrinsic model abilities.",
                              "Local hardware measurements are not public deployment certification.",
                              "Resource sampling is bounded to 300 observations per cohort segment; it may not cover the tail of long runs."]}
    save_artifact(root, "reports/reliability/release_report.json", result)
    lines = ["# AyurIP reliability milestone", "", f"Measured: {result['at']}", "",
             "## Release decision", "",
             "Public answers and fresh synchronization remain disabled. Search readiness was promoted independently after its own gates passed. The answer candidate remains isolated because its quality gates failed.", "",
             f"All answer gates passed: **{result['answer_candidate_passed_all_gates']}**. Candidate `{checksum}`.", "",
             "## Results", "", "| Mode | Set | Attempts | Full-case success | Fully correct answerable cases | Safety abstention | Warm complete p95 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for c in cohorts:
        s = c["summary"]
        lines.append(f"| {c['mode']} | {c['split']} | {s['attempts']} | {s['system_case_success']:.1%} | {s['answerable_coverage']:.1%} | {s['safety_critical_abstention']:.1%} | {s['warm_complete_ms']['p95']:.0f} ms |")
    lines += ["", "Controlled rows use frozen evidence, including explicitly synthetic attack/conflict fixtures. HTTP rows use actual unfiltered top-five hybrid retrieval; those fixtures are not inserted into the live corpus. Complete HTTP timing includes request policy, retrieval, archive checks, both model passes and response serialization. Warmup is excluded. Each cohort has three timed passes. Failed attempts occupy their original slots; interrupted requests are not retried.", "",
              "Fully correct answerable cases must pass all frozen checks, including allowed-citation precision and complete target coverage. This stricter measure is distinct from target-group recall; mean target-group coverage is also recorded per cohort in the JSON. Extra same-provision claims can fail the strict case score even when the required target was found. No score or gate is relaxed for those cases.", "",
              "The held-out set contains 15 answerable and 15 abstention cases, as does development. Thirty additional answerable queries reuse prior source-bound targets; this is a held-out query split, not independent legal annotation or new source coverage. Original 45 cases and historical reports remain intact. The strict patent-term target still excludes the separate revocation passage; labels were not relaxed.", "",
              f"All **{citation_audit['exact_archived_quote_and_identity_citations']}/{citation_audit['emitted_citations']}** emitted citations were rechecked against published text, original raw hash, official URL, document identity, title and anchors. This establishes source membership, not semantic completeness or current law. `miss_audit.json` separates missing retrieval targets, packing loss, selection omissions and extra same-provision citations. Same-anchor extras require annotation review; they are not automatically approved, rejected as legally irrelevant or rescored.", "",
              f"Search: **{retrieval['recall_at_5']:.2%} Recall@5**, **{retrieval['warm_p95_ms']:.0f} ms warm p95**, unchanged reviewed 30-query set over three passes. Startup: **{startup['ready']['readiness']['initialization_ms'] / 1000:.2f} s**; `/live` is independent and `/health`/`/search` fail promptly until embedding warmup and archive validation finish.", "",
              "## Gates", "", "| Gate | Passed |", "|---|---|"]
    lines += [f"| {key} | {value} |" for key, value in gates.items()]
    lines += ["", "## Remaining misses", ""]
    for c in cohorts:
        misses = c["summary"]["misses"]
        if misses:
            lines.append(f"### {c['mode']} / {c['split']}")
            lines.append("")
            for identifier in dict.fromkeys(m["case_id"] for m in misses):
                group = [m for m in misses if m["case_id"] == identifier]
                reasons = sorted({r for m in group for r in m["failures"]})
                lines.append(f"- `{identifier}`: passes {', '.join(str(m['pass']) for m in group)}; {', '.join(reasons)}.")
            lines.append("")
    lines += ["## Reproduction and boundaries", "",
              "Run `python -m tools.freeze_reliability_benchmark` to verify the existing frozen bundle, then `python -m tools.benchmark_reliability --freeze-candidate` only when intentionally freezing a new candidate. Existing held-out results must not be tuned away without a new evaluation protocol. Start the private service explicitly with `python -m ipindia_corpus.private_preview --serve --evaluation`. It binds only `127.0.0.1:8001`; evaluation disables only that process's rate limiter. The main API keeps its controls.", "",
              "For each mode (`controlled`, `http`) and set (`historical`, `development`, `heldout`), run `python -m tools.benchmark_reliability --run --frozen --mode MODE --split SET --passes 3`. The candidate manifest binds code, prompts, model digest, configuration, historical pointers and new benchmark hashes. Run `python -m tools.report_reliability` to audit retained observations and render this report.", "",
              "Pinned runtime: Qwen3.5 4B, digest `d8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04`, local Ollama, context 4096, output cap 200, temperature zero, thinking disabled, residency 30 minutes. Physical setup: i7-13620H, 10 cores/16 logical, 33,968,590,848 bytes RAM, RTX 4060 Laptop GPU 8,188 MiB. Resource samples and runtime version are retained per cohort. Embedding model/revision and active generation/ranker are unchanged.", "",
              "Resource sampling is bounded to 300 observations per cohort segment; long-run tails may not have resource counters. Every timed request still retains its measured duration. Global GPU counters include other desktop applications. Backend guard outcomes and model-selection scores remain separate.", "",
              "Current hardware/software and Docker resources: `runtime.json`. Local endpoint checks: `security_boundaries.json`. API-only promotion and preserved volume evidence: `readiness_promotion.json`. The first policy candidate and the slower full-metadata candidate remain archived in `candidate_v1_interrupted.json` and `candidate_v2_retained.json`; failed/interrupted attempts were not erased or reused as successes. Additional development-only experiments remain under their distinct candidate hashes. No held-out result was used to tune the final candidate.", "",
              "India Code/TKDL originals, certification, unified passage labels and permitted live incremental cycles remain outstanding. The local UI and public answer gate are not a public legal-advice release. All legal statuses remain unknown."]
    (root / "reports/reliability/milestone.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"candidate_sha256": checksum, "gates": gates, "combined": combined}


if __name__ == "__main__":
    print(json.dumps(report(), indent=2))
