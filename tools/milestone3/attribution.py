"""Retained-stage attribution; rank replays are never historical observations."""
from collections import defaultdict
import json

from ipindia_corpus.llm_experiment import ROOT, canonical
from tools.reliability_v2 import cases, coverage
from .common import observations, read, write
from .references import parse, roots
from .review import resolved_stages


def run(root=ROOT):
    all_cases = {c["id"]: c for items in cases(root).values() for c in items}
    replays = read(root / "reports/reliability_v2/rank_replays.json")
    rows, counters, packing = [], defaultdict(list), []
    for row in observations(root):
        case = all_cases[row["case_id"]]
        groups = case["expected"].get("target_groups", [])
        result = row["result"]
        stages = resolved_stages(result)
        covered = {stage: coverage(groups, [{**h, "text": h.get("text", h.get("quote", ""))} for h in value]) if value is not None else None for stage, value in stages.items()}
        covered["backend"] = coverage(groups, [{"chunk_id": ci["chunk_id"], "text": ci["quote"]}
            for claim in result.get("claims", []) for ci in claim.get("citations", [])])
        tags, earliest = [], None
        if not row["system_score"]["passed"]:
            if groups and not all(covered["retrieved"]):
                earliest = "returned_top_k"  # pool/rank unavailable in historical capture
                tags.append("INSUFFICIENT_EVIDENCE")
            if any(a and not b for a, b in zip(covered["retrieved"], covered["packed"])):
                tags.append("CONTEXT_PACKING_LOSS")
                earliest = earliest or "packing"
            if covered.get("selection") is not None and any(a and not b for a, b in zip(covered["packed"], covered["selection"])):
                tags.append("MODEL_SELECTION_OMISSION")
                earliest = earliest or "selection"
            if covered.get("selection") is not None and covered.get("verification") is not None and any(a and not b for a, b in zip(covered["selection"], covered["verification"])):
                tags.append("BACKEND_VERIFIER_REJECTION")
                earliest = earliest or "subset_verifier"
            if "DISALLOWED_CLAIM" in row["system_score"].get("failures", []):
                tags.append("ANNOTATION_DISAGREEMENT")
                # It is not known whether these are true over-selection errors.
                earliest = earliest or "relevance_adjudication_pending"
            if "INAPPROPRIATE_ABSTENTION" in row["system_score"].get("failures", []):
                tags.append("INAPPROPRIATE_ABSTENTION")
            from ipindia_corpus.reliability_candidate import requested_provisions
            old, new = requested_provisions(case["question"]), roots(case["question"])
            if old != new:
                tags.append("IMPLICIT_REFERENCE_PARSE_FAILURE" if any(r["origin"] == "implicit_coordinated" for r in parse(case["question"])["references"]) else "COORDINATED_REFERENCE_PARSE_FAILURE")
            if any(r.get("anchor", {}).get("rule") and any(h.get("section") == r["anchor"]["rule"] and not h.get("rule") for h in stages["retrieved"]) for r in groups):
                tags.append("AMBIGUOUS_PROVISION_ANCHOR")
        for tag in set(tags):
            counters[(row["milestone"], tag)].append(row)
        record = {k: row[k] for k in ("milestone", "mode", "split", "case_id", "pass", "candidate_sha256", "report", "report_sha256", "row_sha256")}
        record.update(origin="retained_observation", stage_coverage=covered, categories=sorted(set(tags)),
                      earliest_observed_loss=earliest, historical_component_ranks=None,
                      strict_score=row["system_score"], backend_reason=result.get("reason"),
                      semantic_adjudication="pending" if "ANNOTATION_DISAGREEMENT" in tags else "not_established_by_stage_audit")
        rows.append(record)
        if groups:
            packing.append({"milestone": row["milestone"], "case_id": row["case_id"], "mode": row["mode"], "pass": row["pass"],
                            "retrieved": covered["retrieved"], "packed": covered["packed"]})
    counts = [{"milestone": milestone, "category": tag, "attempts": len(items),
               "unique_questions": len({r["case_id"] for r in items}),
               "unique_mode_split_questions": len({(r["mode"], r["split"], r["case_id"]) for r in items})}
              for (milestone, tag), items in sorted(counters.items())]
    write("pipeline_trace_v3.jsonl", "".join(canonical(row).decode() + "\n" for row in rows), root)
    result = {"counts": counts, "retained_attempts": len(rows), "categories_overlap": True,
              "independent_questions": len(all_cases), "confirmed_semantic_overselection": None,
              "historical_scores_changed": False, "historical_ranks_available": False,
              "attribution_scope": "observable loss of frozen target text; alternative legal adequacy needs human review",
              "unobserved_categories": ["MODEL_OVERSELECTION", "RETRIEVAL_POOL_ABSENCE", "RANK_BELOW_CUTOFF"],
              "new_rank_replay_diagnostics": replays}
    write("failure_attribution_v3.json", result, root)
    oracle = {"A_actual_http": {"status": "retained_observations", "rows": [r for r in rows if r["mode"] == "http"]},
              "B_independent_gold": {"status": "blocked", "reason": "no_independently_reviewed_gold_passages",
                                     "model_calls": 0, "developer_label_controlled_observations_available": True,
                                     "developer_controlled_is_independent_gold": False},
              "C_pool_oracle": {"origin": "new_milestone2_rank_replay", "historical_ranks": False,
                                 "target_group_coverage": [{"case_id": row["case_id"],
                                    "groups": [{"pool_present": any(a["ranks"] and a["ranks"].get("fused_rank") is not None for a in g["alternatives"]),
                                                "top5_present": any(a["ranks"] and a["ranks"].get("fused_rank") is not None and a["ranks"]["fused_rank"] <= 5 for a in g["alternatives"])} for g in row["target_groups"]]} for row in replays["rows"]]},
              "D_packing_oracle": {"origin": "retained_observations", "rows": packing,
                                    "loss_attempts": sum(any(a and not b for a, b in zip(r["retrieved"], r["packed"])) for r in packing)},
              "labels_enter_normal_inference": False}
    write("oracle_diagnostics.json", oracle, root)
    lines = ["# Failure attribution v3", "", "Historical scores remain unchanged. Categories overlap; repeated attempts are not independent questions.",
             "Earliest observed evidence loss is distinguished from an exclusive causal root cause.",
             "Extra-citation failures remain unresolved relevance disagreements; none is automatically called legally wrong.",
             "Historical candidate-pool/rank details were not retained. The separate replay is diagnostic, never backfilled.", "",
             "| Milestone | Category | Unique questions | Attempts |", "|---|---|---:|---:|"]
    lines += [f"| {r['milestone']} | {r['category']} | {r['unique_questions']} | {r['attempts']} |" for r in counts]
    write("failure_summary_v3.md", "\n".join(lines) + "\n", root)
    return {k: v for k, v in result.items() if k != "new_rank_replay_diagnostics"}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=True))
