"""Summarize the measured candidate and its immutable annotation review item."""
import json
from pathlib import Path

from ipindia_corpus.indexer import generation

ROOT = Path(__file__).resolve().parents[1]


def main():
    def read(name):
        return json.loads((ROOT / "reports" / name).read_text(encoding="utf-8"))
    original = read("retrieval_preview_original.json")
    expanded = read("retrieval_preview_expanded.json")
    passages = read("passage_validation_candidate.json")
    preview = read("retrieval_preview.json")
    gates = read("retrieval_activation_gates.json")
    _, docs, _ = generation(ROOT, generation_id=original["generation_id"])
    doc = next(d for d in docs if "1_58_1_design_act_1_" in d["source_url"])
    review = {"status": "review_required", "original_labels_unchanged": True,
              "affected_queries": ["ipindia-10", "ipindia-28"], "source_url": doc["source_url"],
              "raw_sha256": doc["sha256"], "raw_path": doc["raw_path"],
              "frozen_page": 2, "source_page": 3, "provision": "section 5(2)",
              "evidence": "The registration application provision containing 'prescribed fee' is on physical PDF page 3.",
              "next_action": "Create a separately reviewed evaluation version for factual page corrections; preserve both frozen sets and rerun before activation."}
    (ROOT / "reports/annotation_review.json").write_text(json.dumps(review, indent=2), encoding="utf-8")
    text = f"""# AyurIP offline retrieval and workspace status

Work now runs from **E:\\Ayurveda Rag**. The original India Code Markdown, TKDL Markdown, and regulatory PDFs are preserved and await provenance review before retrieval admission. The implemented service, archives, history, tests, reports, and pinned model cache were transferred. Migration checked 1,964 immutable files without mismatches; Linux verified all 50 model-cache files. E:\\SIH is retained as a backup. Docker retains project identity `sih` and volume `sih_pgdata`, with the API mounted from the new folder.

## Active service and candidate

The live selection remains `{preview['baseline_generation']}` with ranker **v1**, 307 IP India documents and 7,755 chunks. The repaired **candidate**, `{original['generation_id']}`, uses ranker v2 and contains {passages['documents']} documents and {passages['chunks']} chunks, including {passages['short_units_retained']} short units. All 82 PDFs match their archived extracted text page by page; there are zero text or page errors. Source text includes printed headers omitted by the old extraction, so 38 legacy PDF text streams differ from baseline while remaining lossless against the originals.

Printed section/rule context is retained across page continuations. Contents entries stay unanchored; uniquely aligned margin headings are attached with page/bounding-box provenance. Document identity/year is separated from passage intent. Retrieval fuses body, original-query vectors, printed headings, and explicit provisions; duplicate chunks are suppressed within a document while separately cited versions remain eligible.

## Measured retrieval

| Frozen evaluation | Queries / observations | Recall@5 | Warm p95 |
|---|---:|---:|---:|
| Original IP India subset | {original['queries']} / {original['observations']} | {original['recall_at_5']:.0%} | {original['p95_ms']:.0f} ms |
| Expanded IP India set | {expanded['queries']} / {expanded['observations']} | {expanded['recall_at_5']:.1%} | {expanded['p95_ms']:.0f} ms |

Both measurements use unfiltered HTTP search, warmup, and three passes. The seven prior successes are preserved. The two broad Trade Marks misses are now retrieved in the top five. All remaining misses are the frozen Designs Act page target, shared by original query 10 and expanded query 28. Those labels specify page 2, while the fee provision is on PDF page 3. Neither frozen set was changed; unresolved targets count as misses and prevent certification.

The initial preview exposed repeated document/chunk joins. A supporting `(generation_id, document_id)` index and refreshed PostgreSQL statistics reduced the recorded SQL execution from 6,493 ms to 30 ms. Body vectors are computed during indexing rather than per request; CPU inference uses four threads. The final candidate reused all existing preview embeddings and created zero new vectors. The pinned model is `{original['model']}`, revision `{original['model_revision']}`. Hardware: Intel Core i7-13620H, approximately 32 GB physical RAM; Docker 16 CPUs and approximately 15.4 GiB memory. Detailed software versions are included in evaluation JSON.

## Verification and release decision

57 offline tests passed, with two PostgreSQL tests skipped locally and then passed in a disposable database. Ruff passed. Database checks cover within-document duplicates, separately cited versions, explicit filters, indexing failure, inactive builds, generation/ranker rollback, citation tampering, and retention of the production selection. Hostile source text remains data. API results are checked against published metadata/text, archived hashes, official URLs, and document identity.

Candidate activation is **blocked** by factual annotation review; numeric recall and latency gates pass. The last searchable generation remains available. The preview server stopped after evaluation, while its isolated database/cache is retained for review and reuse. Activation and rollback commands require the recorded gate evidence and generation/ranker pair.

Fresh synchronization is deferred, the scheduler is disabled, and every legal status remains `unknown`. India Code/TKDL coverage and the complete three-source release are not certified. There is no generated legal-answer feature or new public interface.

Evidence: [migration](workspace_migration.json), [model cache](model_cache_migration.json), [passages](passage_validation_candidate.json), [original evaluation](retrieval_preview_original.json), [expanded evaluation](retrieval_preview_expanded.json), [ranking diagnostics](ranking_candidate.json), [annotation review](annotation_review.json), [activation gates](retrieval_activation_gates.json), [offline tests](verification/offline.json), [PostgreSQL tests](verification/postgres.json).
"""
    (ROOT / "reports/offline_retrieval_status.md").write_text(text, encoding="utf-8")
    print(json.dumps({"report": str(ROOT / "reports/offline_retrieval_status.md"),
                      "numeric_gates_pass": gates["checks"]["original_recall_and_latency"] and gates["checks"]["expanded_recall_and_latency"],
                      "activation_pass": gates["pass"]}))


if __name__ == "__main__":
    main()
