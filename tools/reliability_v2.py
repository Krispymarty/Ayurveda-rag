"""Immutable Milestone 2 audit, annotation preparation and experiment receipts.

Run from the project root. No production promotion or source fetching exists here.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path

from ipindia_corpus.llm_experiment import ROOT, canonical, sha256, timestamp
from ipindia_corpus.paths import immutable_write
from ipindia_corpus.span_selection import build_plan, resolve_selection
from tools.benchmark_reliability import binding
from tools.freeze_llm_benchmark import load_frozen
from tools.freeze_reliability_benchmark import load
from tools.report_llm_results import verified_artifact

DEST = "reports/reliability_v2"
DATA = "data/reliability_v2"
GENERATION = "20261014T204940358650Z"


def write(root, relative, value):
    raw = value.encode("utf-8") if isinstance(value, str) else canonical(value)
    checksum = sha256(raw)
    path = Path(root) / relative
    immutable_write(Path(root) / DATA / "artifacts" / checksum, raw)
    # A named report is a pointer; its complete previous bytes remain immutable.
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)
    path.with_suffix(path.suffix + ".sha256").write_text(checksum + "\n", encoding="ascii")
    return checksum


def cases(root=ROOT):
    _, historical, _, _ = load_frozen(root)
    receipt = binding(root)
    _, development = load(root, "development")
    _, former_holdout = load(root, "heldout", candidate_receipt=receipt)
    return {"historical": historical, "development": development, "heldout_regression": former_holdout}


def snapshot(root=ROOT):
    root = Path(root)
    paths = []
    for folder, pattern in (("ipindia_corpus", "*.py"), ("tools", "*.py"), ("tests", "*.py"), ("config", "*.json"), ("web", "*")):
        paths.extend(p for p in (root / folder).glob(pattern) if p.is_file())
    paths.extend(root / name for name in ("compose.yaml", "Dockerfile", "requirements.txt", "requirements-service.txt") if (root / name).exists())
    hashes = {p.relative_to(root).as_posix(): sha256(p.read_bytes()) for p in sorted(paths)}
    payload = {"at": timestamp(), "generation_id": GENERATION,
               "historical_candidate": verified_artifact(root / "reports/reliability/candidate.json")["candidate_sha256"],
               "git": {"available": False, "reason": "repository_has_no_git_metadata", "replacement": "content_hash_source_snapshot"},
               "binding": binding(root), "sources": hashes}
    checksum = sha256(canonical(payload))
    folder = root / DATA / "snapshots" / checksum
    immutable_write(folder / "manifest.json", canonical(payload))
    for name in hashes:
        immutable_write(folder / "sources" / name, (root / name).read_bytes())
    write(root, DEST + "/baseline_snapshot.json", {"snapshot_sha256": checksum, **payload})
    return checksum


def coverage(groups, hits):
    return [any(h["chunk_id"] in g["required_chunk_ids"] and
                " ".join(g["required_contains"].casefold().split()) in " ".join(h["text"].casefold().split())
                for h in hits) for g in groups]


def audit(root=ROOT):
    root = Path(root)
    all_cases = cases(root)
    traces, tags = [], defaultdict(list)
    for mode in ("controlled", "http"):
        for split, items in all_cases.items():
            name = "heldout" if split == "heldout_regression" else split
            report = verified_artifact(root / f"reports/reliability/{mode}-{name}-3passes.json")
            by_id = {c["id"]: c for c in items}
            for row in report["rows"]:
                c, result = by_id[row["case_id"]], row["result"]
                hits = result.get("retrieval_evidence", [])
                omitted = set(result.get("context_omitted_chunk_ids", []))
                packed = [{**h, "provenance": "verified_immutable_archive"} if h.get("source") != "synthetic_fixture" else h
                          for h in hits if h["chunk_id"] not in omitted]
                stages = {"retrieved": coverage(c["expected"].get("target_groups", []), hits),
                          "packed": coverage(c["expected"].get("target_groups", []), packed)}
                selections = {}
                for observation in result.get("model_observations", []):
                    try:
                        resolved = resolve_selection(observation["raw"], packed, build_plan(packed))
                        chosen = resolved["expanded_selection"]["claims"]
                        selections[observation["stage"]] = chosen
                    except (ValueError, KeyError, TypeError):
                        selections[observation["stage"]] = None
                for key in ("selection", "verification"):
                    if selections.get(key) is not None:
                        stages[key] = coverage(c["expected"].get("target_groups", []),
                                               [{"chunk_id": p["chunk_id"], "text": p["quote"]} for p in selections[key]])
                classification = []
                if not row["system_score"]["passed"]:
                    if stages["retrieved"] and not all(stages["retrieved"]):
                        classification.append("RETRIEVAL_TARGET_ABSENT_FROM_RETURNED_HITS")
                    if any(a and not b for a, b in zip(stages["retrieved"], stages["packed"])):
                        classification.append("PACKING_LOSS")
                    if stages["packed"] and all(stages["packed"]) and row["system_score"].get("target_coverage", 0) < 1:
                        classification.append("SELECTION_OR_VERIFICATION_OMISSION")
                        if stages.get("selection") and not all(stages["selection"]):
                            classification.append("EVIDENCE_SELECTION_OMISSION")
                        elif stages.get("selection") and all(stages["selection"]) and not all(stages.get("verification", [])):
                            classification.append("SUBSET_VERIFICATION_OMISSION")
                    if "DISALLOWED_CLAIM" in row["system_score"]["failures"]:
                        classification.append("EXTRA_CITATION_REQUIRES_RELEVANCE_REVIEW")
                    if "INAPPROPRIATE_ABSTENTION" in row["system_score"]["failures"]:
                        classification.append("INAPPROPRIATE_ABSTENTION")
                key = (mode, split, c["id"])
                for tag in classification:
                    tags[tag].append(key)
                traces.append({"origin": "retained_historical_observation", "mode": mode, "split": split,
                               "case_id": c["id"], "pass": row["pass"], "question": c["question"],
                               "candidate_sha256": row["candidate_sha256"], "stage_coverage": stages,
                               "retrieval": [{k: h.get(k) for k in ("chunk_id", "document_id", "section", "rule", "page_start", "score")} for h in hits],
                               "historical_component_ranks": None, "rank_availability": "not_recorded; replay required",
                               "packed_chunk_ids": [h["chunk_id"] for h in packed], "resolved_model_stages": selections,
                               "backend_claims": result.get("claims", []), "reason": result.get("reason"),
                               "strict_score": row["system_score"], "categories": classification})
    counts = {tag: {"failed_attempts": len(keys), "unique_questions": len({k[2] for k in keys}),
                    "unique_mode_split_questions": len(set(keys))} for tag, keys in tags.items()}
    payload = {"at": timestamp(), "counts": counts, "categories_overlap": True,
               "historical_results_changed": False, "causality": "stage observations; ranking versus pool absence needs replay",
               "former_heldout_use": "regression only; already inspected", "attempts": len(traces)}
    write(root, DEST + "/failure_attribution.json", payload)
    write(root, DEST + "/pipeline_trace.jsonl", "".join(canonical(t).decode() + "\n" for t in traces))
    lines = ["# Root-cause audit", "", "Retained observations only. Missing historical component ranks remain unavailable.",
             "Counts overlap; repeated attempts are not independent questions.", "",
             "| Category | Failed attempts | Unique questions |", "|---|---:|---:|"]
    for tag, n in sorted(counts.items(), key=lambda x: -x[1]["failed_attempts"]):
        lines.append(f"| {tag} | {n['failed_attempts']} | {n['unique_questions']} |")
    lines += ["", "Extra real citations are not automatically hallucinations. Source fidelity does not establish relevance.",
              "The original labels, scores and narrow archived-20 patent-term target are unchanged."]
    write(root, DEST + "/root_cause_analysis.md", "\n".join(lines) + "\n")
    write(root, DEST + "/prioritized_failure_summary.md", "\n".join(lines[:4] + ["", "Priority: review extra citations; replay missing retrieval targets; distinguish selection from verifier omissions."]) + "\n")
    return payload


def seal_review_package(root=ROOT):
    root = Path(root)
    source = cases(root)
    positives = [c for split in ("development", "heldout_regression") for c in source[split] if c["expected"]["policy"] == "quote_targets"]
    negatives = [c for split in ("development", "heldout_regression") for c in source[split] if c["expected"]["policy"] == "empty_claims"]
    proposed = []
    for number, original in enumerate(positives + negatives, 1):
        item = deepcopy(original)
        item.update(id=f"v2-review-{number:02}", parent_case_id=original["id"],
                    question="For this archived-source research request, " + original["question"][0].lower() + original["question"][1:],
                    annotation_status="awaiting_independent_review", independent_annotation=False,
                    limitations="Reworded query; reused source targets. Not an independent source-family holdout.")
        proposed.append(item)
    raw = b"".join(canonical(c) + b"\n" for c in proposed)
    folder = root / DATA / "sealed_review_v1"
    immutable_write(folder / "cases.jsonl", raw)
    manifest = {"version": "v2-sealed-review-v1", "cases_sha256": sha256(raw), "cases": 60,
                "answerable": 30, "abstention": 30, "generation": GENERATION,
                "independently_reviewed": False, "certification_allowed": False,
                "review_requirement": "Independent reviewer must replace/review wording and source-bound targets, record disagreements, then seal a new reviewed version before candidate exposure."}
    immutable_write(folder / "manifest.json", canonical(manifest))
    review = []
    for c in proposed:
        review.append({"id": c["id"], "question": c["question"], "parent_case_id": c["parent_case_id"],
                       "required_passages": c["expected"].get("target_groups", []),
                       "allowed_claim_chunk_ids": c["expected"].get("allowed_claim_chunk_ids", []),
                       "necessary_qualifications": c.get("necessary_qualifications", "Preserve source qualifications and unknown legal status."),
                       "excluded_background": c.get("disallowed_background_citations", "Unrequested related provisions require explicit relevance review."),
                       "source_evidence": c["evidence"], "reviewer": None, "decision": "pending", "disagreements": []})
    write(root, DEST + "/annotation_review_package.json", {"independent_review_pending": True, "cases": review, "historical_scores_unchanged": True})
    write(root, DEST + "/sealed_evaluation_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("snapshot", "audit", "seal"))
    args = parser.parse_args()
    result = {"snapshot": snapshot, "audit": audit, "seal": seal_review_package}[args.action]()
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
