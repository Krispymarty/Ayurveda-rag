"""Render measured local-model evidence as a readable selection report."""
from __future__ import annotations

import json
from pathlib import Path

from ipindia_corpus.llm_experiment import ROOT
from tools.report_llm_results import verified_artifact


def percent(value):
    return "unavailable" if value is None else f"{value * 100:.1f}%"


def number(value):
    return "unavailable" if value is None else f"{value:,.2f}"


def render(root=ROOT):
    root = Path(root)
    report = verified_artifact(root / "reports/llm/quality_summary.json")
    main = verified_artifact(root / "reports/llm/model_comparison.json")
    models = report["models"]
    lines = ["# Local LLM evaluation and selection", "", "Updated 17 October 2026, Asia/Kolkata.", "",
             "## Decision", "",
             "**No production activation has been performed.** Keep the retained evidence-search service and existing answer configuration. Public `/answer` remains disabled; source synchronization remains paused. The quality failures below require further work.", "",
             f"The base comparison contains **{sum(m['quality']['attempts'] for m in models)} timed attempts**: 45 fixed cases, three passes per model. Every saved failure is counted. No failure was retried for a better score. The final report uses verified resumed segments; other interrupted diagnostic runs remain separately archived.", "",
             "## Raw model comparison at the common 200-token cap", "",
             "| Model | Cases passed | Schema-valid attempts | Literal quote + ID precision | Correct abstention | Invalid-ID claims |",
             "|---|---:|---:|---:|---:|---:|"]
    def quality_row(model, label):
        q = model["quality"]
        abstention = model.get("non_forced_quality", q)
        passed = q["attempts"] - len(model["misses"])
        bad_ids = q["emitted_claims"] - q["valid_citation_claims"]
        return (f"| {label} | {passed}/{q['attempts']} ({percent(model['summary']['raw_model_case_pass_rate'])})"
                f" | {q['schema_valid_attempts']}/{q['attempts']}"
                f" | {q['literal_substring_quote_claims']}/{q['emitted_claims']} ({percent(q['literal_quote_precision'])})"
                f" | {abstention['correct_abstention_attempts']}/{abstention['expected_abstention_attempts']}"
                f" | {bad_ids} |")
    for model in models:
        lines.append(quality_row(model, model["model"]))
    lines += ["", "Case success requires schema/completion, correct citation identity, a substantial exact quotation, all frozen target groups, allowed relevance and expected abstention. This is not a legal-answer accuracy score. Literal quote precision divides source-matching quotations with the correct chunk ID by emitted claims in schema-parseable responses. Empty claims earn no quote-precision credit. Malformed and truncated responses remain failed attempts; a truncated response with valid JSON still fails.", "",
              "All three raw models fail the zero-invalid-citation gate. Many Ministral and Qwen errors are real document IDs substituted for full chunk IDs, rather than invented URLs. Those IDs still cannot resolve to an authorized passage. Qwen had the strongest full-case score under this initial common profile and was chosen for the general schema follow-up. Other models have not been compared at the larger follow-up budget; no universal model ranking is claimed. Full per-attempt misses and failure classifications are in `quality_summary.json`.", "",
              "## Runtime measurements", "",
              "| Model | Warm p50, all attempts (ms) | Warm p95, all attempts (ms) | Completed-only p95 (ms) | Median tokens/s, completed | Model VRAM (MiB) | Global GPU peak (MiB) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for model in models:
        s = model["summary"]
        vram = model["median_model_reported_size_vram_bytes"]
        lines.append(f"| {model['model']} | {number(s['all_attempt_latency_ms']['p50'])}"
                     f" | {number(s['all_attempt_latency_ms']['p95'])} | {number(s['warm_latency_ms']['p95'])}"
                     f" | {number(s['median_output_tokens_per_second'])} | {number(vram / 1048576 if vram else None)}"
                     f" | {number(model['peak_global_gpu_memory_used_mb'])} |")
    lines += ["", "All-attempt wall latency includes incomplete/failed calls. No timed request expired in the base comparison. Fixed-input model timings exclude retrieval, archive validation, embedding initialization and API queueing. Global GPU usage includes other desktop applications; model residency is recorded separately through `/api/ps`. Sparse or absent samples remain marked unavailable, and short calls do not establish a precise GPU utilization average.", "",
              "**The cohorts span sessions and dates.** Each resumed segment received an unscored warmup. Background load and OS/runtime caches differ across sessions, so these latency values describe the retained sessions and do not prove a causal speed ranking or an end-to-end service SLO.", "",
              "The existing Ministral profile measured first-cold HTTP time 52.92 seconds, including 19.83 seconds loading and 32.13 seconds prompt processing. Three identical-input warm calls had median 984.81 ms and p95 1,128.95 ms, about 85 tokens/s. The later combined 200-token/30-minute profile was also about 984 ms warm; its smaller cold time cannot be attributed to those settings because startup/file caches were not reset. The buffered streaming probe measured content TTFT 114.71 ms. No unverified legal content was streamed publicly.", "",
              "## General schema constraint", ""]
    candidate = report.get("candidate")
    budget = report.get("budget600_candidate")
    if candidate is None:
        lines += ["The chunk-ID enum experiment is pending. No improvement is claimed yet."]
    else:
        lines += [f"Measured variant: `{candidate['variant']}` on `{candidate['model']}`. Question, evidence text/order and system instructions match the frozen case; only the output schema gains every eligible official input ID, including decoys. Derived schema and input hashes are recorded. Expected passage labels are never used to construct the enum.", "",
                  "| Variant | System cases passed | Schema-valid attempts | Literal quote + ID precision | Abstention excluding forced cases | Invalid-ID claims |",
                  "|---|---:|---:|---:|---:|---:|", quality_row(candidate, "Chunk-ID enum"), "",
                  "This measures a grammar-constrained system. Zero invalid enum IDs is not evidence of intrinsic model citation reliability. The two cases with no eligible official citations are structurally forced empty and receive no credit as learned abstention. Valid IDs and exact text can still select irrelevant material or omit a required provision.", "",
                  f"Candidate all-attempt warm p95: **{number(candidate['summary']['all_attempt_latency_ms']['p95'])} ms**. Failures: `{json.dumps(candidate['failures'], sort_keys=True)}`. The candidate remains experimental and has not changed the active answer provider."]
    if budget is not None:
        lines += ["", "### One-variable output-budget follow-up", "",
                  "The same pinned model, messages, evidence ordering, enum schema, context, temperature, thinking option and residency were reused. Only `num_predict` changed from 200 to 600. The parent 200-token results remain intact.", "",
                  "| Variant | System cases passed | Schema-valid attempts | Literal quote + ID precision | Abstention excluding forced cases | Invalid-ID claims |",
                  "|---|---:|---:|---:|---:|---:|", quality_row(budget, "Chunk-ID enum, cap 600"), "",
                  f"All-attempt warm p95: **{number(budget['summary']['all_attempt_latency_ms']['p95'])} ms**. Failures: `{json.dumps(budget['failures'], sort_keys=True)}`. Full misses and scope-separated metrics are in `quality_summary.json`. These remain experimental system results; no model or schema was activated."]
    lines += ["", "## Remaining repeated failures", "",
              "The 600-token follow-up repeated five first-pass misses across all three passes: `archived-05` returned four altered quotations; `archived-08` and `archived-23` added quotations outside the frozen allowed-target set; `archived-13` returned a 1,995-character quotation exceeding the 1,600-character bound; and `current-status-02` selected historical text instead of expected Hindi temporal abstention. These are diagnostic descriptions, not a new legal relevance judgment. See `failure_analysis.json` for exact raw-observation references.", "",
              "## Review scope and safety", "",
              "Thirty cases inherit archived-source factual passage targets; six Hindi/mixed-language translations and one short-unit probe are exploratory. Eight attack/current-status/insufficiency/conflict cases are synthetic policy fixtures. Their results are reported separately. No qualified independent legal relevance review or current-law certification is claimed. Some inherited coverage phrases are broad; a later stronger benchmark needs a new version, preserving this frozen set.", "",
              "All 155 real evidence occurrences, covering 93 unique chunks, matched published text and citation metadata; all 307 source archive hashes and generation hashes validated. Synthetic evidence has no authoritative URL or raw hash and cannot become a valid citation. The preserved 44-character legal unit remains eligible. All legal statuses remain `unknown`.", "",
              "The model receives no tools or settings-writing capability. Exact quotations and identities are validated by the shared backend helper; official citations are constructed by the backend. Raw-model failures remain failures even when a backend guard would reject the request. Source-host allowlists, schedules, protected files and publication pointers are unchanged.", "",
              "### Independent backend guard fix after measurement", "",
              "The Hindi temporal miss exposed an English-only backend guard. A subsequent conservative question-only guard adds Hindi and romanized temporal cues and is tested to abstain before provider calls when legal status is unknown. Archived quotation requests remain eligible, and Devanagari word boundaries prevent matching unrelated longer words. This lexical guard is not a complete multilingual intent classifier.", "",
              "The exact measured provider and experiment sources were archived by hash before this fix. An AST comparison proves that every other provider statement, including model requests and quote validation, is unchanged; frozen SYSTEM/schema/case hashes also match. Historical model scores remain untouched and receive no credit for the new backend guard. See `measured_code_archive.json` and `backend_guard_fix.json`.", "",
              "## Runtime and exact model identities", "",
              "Native Windows Ollama 0.40.5; i7-13620H, 10 physical/16 logical cores, 33,968,590,848 bytes RAM; RTX 4060 Laptop GPU, 8,188 MiB VRAM, driver 617.42. Every measured model reported fully GPU-resident weights in its retained placement records. The Docker API uses `host.docker.internal:11434`; developer measurements use `127.0.0.1:11434`. Native Ollama does not require Docker CUDA access.", ""]
    licenses = {"ministral-3:3b": "Apache 2.0", "qwen3.5:4b": "Apache 2.0", "gemma3:4b": "Gemma Terms of Use"}
    for model in main["models"]:
        lines.append(f"- `{model['model']}`: `{model['model_digest']}`; Q4_K_M GGUF; {licenses[model['model']]}. Receipt: `{model['model_receipt_path']}`.")
    lines += ["", "Software versions and power connection are recorded in `software_environment.json`; runtime, exact licenses, raw timings and resource samples have separate receipts. Tags are mutable; this experiment binds the observed full digests. No remote model code or pickle checkpoint was loaded.", "",
              "## Follow-up and preserved release gates", "",
              "The measured 200-token cap causes substantial truncation, especially for multiple quotations. The separate 600-token follow-up isolates this limit. A further useful experiment is backend-defined short evidence spans with enumerated span IDs and backend-copied original text. This can reduce output size and prevent invented quote characters by construction; relevance, completeness, legal-status abstention and citation provenance still require validation. Span selection has not been benchmarked here.", "",
              "Larger models, additional quantizations and throughput/concurrency tuning were not justified before solving the measured structural/correctness failures. Reliable VRAM headroom for a larger model has not been established on this desktop workload. No extra large model was downloaded. Single-request correctness takes priority over throughput.", "",
              "The active retrieval generation remains `20261014T204940358650Z`, ranker v2; the prior generation and Ministral configuration remain available. The active prototype configuration retains its original 600-token cap on Ministral. Both Qwen profiles are separate experiments. India Code and TKDL admission, fresh source access/captures, a qualified relevance review and answer release gates remain unresolved. No public model selection, source synchronization or answer activation was performed.", "",
              "## Reproduce and inspect", "",
              "See [experiment instructions](README.md), [audited quality metrics](quality_summary.json), [main comparison](model_comparison.json), [baseline analysis](baseline_analysis.md), [frozen source verification](frozen_evidence_verification.json), and [dataset review](dataset_review.md). Every reported case is traceable to hash-addressed raw observations. Candidate receipts and checkpoint markers preserve derived inputs and every timed failure."]
    final_path = root / "reports/llm/final_state.json"
    if final_path.is_file():
        state = verified_artifact(final_path)
        lines += ["", "## Final local service verification", "",
                  f"Docker Engine {state['docker_engine']}, {state['docker_cpus']} VM CPUs, {state['docker_memory_bytes']:,} bytes VM memory. Native Ollama GPU measurements are separate from these Docker resources. Volume `{state['database_volume']['name']}` retains its original creation identity `{state['database_volume']['created_at']}`.", "",
                  "The restored API returns HTTP 200 for health and the expected HTTP 503 `three_source_not_certified` for public answers. Five functional search hits matched their published text, document identity, raw hash, official URL and anchors. The post-start functional search took 24.73 seconds with cold initialization; it is not a new warm latency benchmark. The first health call also performs generation/archive validation, so a short cold readiness timeout alone does not establish a forwarding defect.", "",
                  "The full offline verification passed 203 tests with two PostgreSQL tests skipped in the host run; both passed separately in a disposable database. Ruff passed. Source workers remain stopped and scheduling remains disabled. See [final state](final_state.json), [search/citation verification](final_service_verification.json), [backend guard proof](backend_guard_fix.json), [offline checks](../verification/offline.json), and [database checks](../verification/postgres.json)."]
    target = root / "reports/llm/model_selection.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


if __name__ == "__main__":
    print(render())
