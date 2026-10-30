"""New rank diagnostics for historical misses; never backfill historical ranks."""
from collections import Counter

from tools.reliability_v2 import ROOT, DEST, cases, write
from tools.reliability_v2_runtime import session
from tools.report_llm_results import verified_artifact
from ipindia_corpus.llm_experiment import timestamp


def main():
    selected = {}
    for split, items in cases().items():
        old_name = "heldout" if split == "heldout_regression" else split
        report = verified_artifact(ROOT / f"reports/reliability/http-{old_name}-3passes.json")
        failed = {r["case_id"] for r in report["rows"] if not r["system_score"]["passed"]}
        for c in items:
            if c["id"] in failed:
                selected[c["id"]] = c
    client, rows = session(), []
    for c in selected.values():
        response = client.get("http://127.0.0.1:8002/_candidate_ranks", params={"q": c["question"]}, timeout=(3, 30))
        response.raise_for_status()
        ranks = {r["chunk_id"]: r for r in response.json()["rows"]}
        groups = []
        for group in c["expected"].get("target_groups", []):
            alternatives = []
            for cid in group["required_chunk_ids"]:
                row = ranks.get(cid)
                category = "INELIGIBLE_OR_ARCHIVED_ID_ABSENT" if row is None else "ABSENT_FROM_FUSED_POOL" if row["fused_rank"] is None else "BELOW_TOP_FIVE" if row["fused_rank"] > 5 else "RETURNABLE_IN_TOP_FIVE"
                alternatives.append({"chunk_id": cid, "category": category, "ranks": row})
            groups.append({"required_contains": group["required_contains"], "alternatives": alternatives})
        rows.append({"origin": "new_diagnostic_replay", "historical_ranks_recovered": False, "at": timestamp(),
                     "case_id": c["id"], "question": c["question"], "target_groups": groups})
        print(c["id"], flush=True)
    counts = Counter(a["category"] for r in rows for g in r["target_groups"] for a in g["alternatives"])
    payload = {"origin": "new_diagnostic_replay", "rows": rows, "alternative_target_counts": dict(counts),
               "historical_scores_changed": False, "interpretation": "Alternative IDs overlap; these are diagnostic eligibility and ranks, not extra independently scored cases."}
    write(ROOT, DEST + "/rank_replays.json", payload)
    return payload


if __name__ == "__main__":
    main()
