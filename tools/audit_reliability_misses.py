"""Factual miss diagnostics; never changes labels or adjudicates legal relevance."""
from collections import Counter

from ipindia_corpus.indexer import generation
from ipindia_corpus.llm_experiment import save_artifact, timestamp
from ipindia_corpus.llm_scoring import normalized


def audit(root, by_mode, cases_by_mode):
    gid, documents, chunks = generation(root, generation_id="20261014T204940358650Z")
    by_chunk = {c["chunk_id"]: c for c in chunks}
    by_doc = {d["document_id"]: d for d in documents}
    details, counts = [], Counter()
    total, exact = 0, 0
    for mode, rows in by_mode.items():
        cases = {c["id"]: c for c in cases_by_mode[mode]}
        for row in rows:
            case = cases[row["case_id"]]
            claims = [c for claim in row["result"].get("claims", []) for c in claim.get("citations", [])]
            for citation in claims:
                total += 1
                chunk = by_chunk[citation["chunk_id"]]
                doc = by_doc[chunk["document_id"]]
                if (citation["quote"] not in chunk["text"]
                        or citation.get("title") != (chunk.get("document_title") or doc["title"])
                        or any(citation.get(k) != chunk.get(k) for k in ("document_id", "raw_sha256", "source_url", "section", "rule", "page_start", "page_end"))):
                    raise ValueError("emitted_quote_or_citation_identity_mismatch")
                exact += 1
            if row["system_score"]["passed"]:
                continue
            categories, extras = [], []
            allowed = set(case["expected"]["allowed_claim_chunk_ids"])
            targets = [by_chunk[i] for i in allowed if i in by_chunk]
            for citation in claims:
                if citation["chunk_id"] in allowed:
                    continue
                chunk = by_chunk[citation["chunk_id"]]
                same_unit = any(chunk["document_id"] == t["document_id"]
                                and any(chunk.get(k) and str(chunk.get(k)).casefold() == str(t.get(k)).casefold()
                                        for k in ("section", "rule")) for t in targets)
                kind = "same_provision_extra_requires_annotation_review" if same_unit else "other_provision_or_version_requires_review"
                categories.append(kind)
                extras.append({"kind": kind, "chunk_id": citation["chunk_id"], "raw_sha256": citation["raw_sha256"],
                               "source_url": citation["source_url"], "section": citation.get("section"), "rule": citation.get("rule"),
                               "quote_excerpt": citation["quote"][:240]})
            retrieved = {h["chunk_id"] for h in row["result"].get("retrieval_evidence", [])}
            omitted = set(row["result"].get("context_omitted_chunk_ids", []))
            for group in case["expected"].get("target_groups", []):
                recalled = any(c["chunk_id"] in group["required_chunk_ids"]
                               and normalized(group["required_contains"]) in normalized(c["quote"]) for c in claims)
                if recalled:
                    continue
                ids = set(group["required_chunk_ids"])
                if not ids & retrieved:
                    kind = "retrieval_target_absent" if row["result"].get("retrieval_evidence") is not None else "policy_or_dependency_stopped_before_retrieval"
                elif not ids & (retrieved - omitted):
                    kind = "context_packing_loss"
                elif any(c["chunk_id"] in ids for c in claims):
                    kind = "excerpt_does_not_cover_frozen_target_phrase"
                else:
                    kind = "selection_or_verification_omission"
                categories.append(kind)
            counts.update(set(categories))
            details.append({"mode": mode, "case_id": row["case_id"], "pass": row["pass"], "question": case["question"],
                            "strict_failures": row["system_score"]["failures"], "categories": sorted(set(categories)),
                            "extra_claims": extras, "target_coverage": row["system_score"]["target_coverage"],
                            "reason": row["result"].get("reason")})
    report = {"at": timestamp(), "generation_id": gid, "emitted_citations": total,
              "exact_archived_quote_and_identity_citations": exact, "all_raw_generation_hashes_reverified": True,
              "counts_by_failed_attempt": dict(counts), "details": details,
              "reviewer": "Codex automated archived-source identity/anchor checks",
              "legal_relevance_review": "unresolved; same anchor is not proof of necessity or relevance",
              "historical_and_new_frozen_labels_changed": False, "diagnostics_change_release_scores": False}
    save_artifact(root, "reports/reliability/miss_audit.json", report)
    return {key: report[key] for key in ("emitted_citations", "exact_archived_quote_and_identity_citations", "counts_by_failed_attempt")}
