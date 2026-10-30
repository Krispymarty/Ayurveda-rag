"""Gate candidate retrieval activation and retain an atomic rollback pair."""
import argparse
import hashlib
import json
from datetime import datetime, timezone

from .indexer import _connect, generation, index, model_revision
from .locking import file_lock
from .sync_runner import ROOT


def gates():
    selection_path = ROOT / 'config/benchmark_selection.json'
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else None
    if selection:
        from .benchmark_review import verify_bundle
        bundle = ROOT / selection['bundle']
        reviewed = verify_bundle(ROOT, bundle)
        if hashlib.sha256((bundle / 'manifest.json').read_bytes()).hexdigest() != selection['manifest_sha256']:
            raise ValueError('benchmark_selection_manifest_mismatch')
        names = ('reviewed_retrieval_original.json', 'reviewed_retrieval_expanded.json',
                 'passage_validation_candidate.json', 'reviewed_source_validation.json',
                 'verification/offline.json', 'verification/postgres.json')
        label_paths = [ROOT / reviewed['sets'][name]['path'] for name in ('original', 'expanded')]
    else:
        names = ('retrieval_preview_original.json', 'retrieval_preview_expanded.json',
                 'passage_validation_candidate.json', 'frozen_source_validation.json',
                 'verification/offline.json', 'verification/postgres.json')
        label_paths = [ROOT / 'tests/retrieval_queries.json', ROOT / 'tests/ipindia_retrieval_expanded.json']
    reports = {name: json.loads((ROOT / "reports" / name).read_text()) for name in (
        names)}
    original, expanded = reports[names[0]], reports[names[1]]
    baseline_successes = {f"ipindia-{i:02}" for i in range(1, 8)}
    rank_hash = hashlib.sha256((ROOT / "ipindia_corpus/retrieval.py").read_bytes()).hexdigest()
    checks = {
        "original_recall_and_latency": original["recall_at_5"] >= .90 and original["p95_ms"] <= 2000 and original["queries"] == 10 and original["observations"] == 30,
        "original_successes_preserved": not any(m["id"] in baseline_successes for m in original["misses"]),
        "expanded_recall_and_latency": expanded["recall_at_5"] >= .90 and expanded["p95_ms"] <= 2000 and expanded["queries"] >= 30 and expanded["observations"] >= 90,
        "annotations_resolve": not original["label_issues"] and not expanded["label_issues"],
        "same_generation": original["generation_id"] == expanded["generation_id"] == reports["passage_validation_candidate.json"]["generation_id"],
        "frozen_labels": original["label_sha256"] == hashlib.sha256(label_paths[0].read_bytes()).hexdigest()
          and expanded["label_sha256"] == hashlib.sha256(label_paths[1].read_bytes()).hexdigest(),
        "evaluated_model_and_ranker": original["model_revision"] == expanded["model_revision"] == model_revision()
          and original.get("ranking_revision") == expanded.get("ranking_revision") == rank_hash,
        "passage_integrity": reports["passage_validation_candidate.json"]["pass"],
        "source_annotations_valid": reports[names[3]]["pass"],
        "offline_verification": reports["verification/offline.json"]["pass"],
        "postgres_verification": reports["verification/postgres.json"]["pass"],
    }
    result = {"pass": all(checks.values()), "checks": checks, "candidate_generation": original["generation_id"],
              "benchmark_bundle": selection['bundle'] if selection else None,
              "checked_at": datetime.now(timezone.utc).isoformat(), "ranking_revision": rank_hash,
              "report_hashes": {name: hashlib.sha256((ROOT / "reports" / name).read_bytes()).hexdigest() for name in reports}}
    (ROOT / "reports/retrieval_activation_gates.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def activate():
    with file_lock(ROOT / "data/locks/retrieval-release.lock"):
        proof = gates()
        if not proof["pass"]:
            raise ValueError("retrieval_activation_gates_not_passed")
        identifier = proof["candidate_generation"]
        generation(ROOT, generation_id=identifier)
        connection = _connect()
        try:
            previous = connection.execute("SELECT value FROM app_state WHERE key='active_generation'").fetchone()[0]
            profile = connection.execute("SELECT value FROM app_state WHERE key='ranking_version'").fetchone()
            previous_ranker = profile[0] if profile else "v1"
            # Index all rows/embeddings without changing the live selection.
            index(ROOT, generation_id=identifier, publish_hint=False, record_events=False, activate=False)
            current_proof = gates()
            if not current_proof["pass"] or current_proof["report_hashes"] != proof["report_hashes"] or current_proof["ranking_revision"] != proof["ranking_revision"]:
                raise ValueError("retrieval_evidence_changed")
            rollback = json.dumps({"generation_id": previous, "ranking_version": previous_ranker})
            with connection.transaction():
                selected = connection.execute("SELECT value FROM app_state WHERE key='active_generation' FOR UPDATE").fetchone()[0]
                if selected != previous:
                    raise ValueError("active_generation_changed_before_activation")
                connection.execute("INSERT INTO app_state VALUES ('retrieval_rollback',%s) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (rollback,))
                connection.execute("INSERT INTO app_state VALUES ('ranking_version','v2') ON CONFLICT(key) DO UPDATE SET value=excluded.value")
                connection.execute("UPDATE app_state SET value=%s WHERE key='active_generation'", (identifier,))
            receipt = {**proof, "previous_generation": previous, "previous_ranker": previous_ranker}
            directory = ROOT / "data/retrieval_releases"
            directory.mkdir(exist_ok=True)
            (directory / f"{identifier}_activation.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
            return receipt
        finally:
            connection.close()


def rollback():
    with file_lock(ROOT / "data/locks/retrieval-release.lock"):
        connection = _connect()
        try:
            row = connection.execute("SELECT value FROM app_state WHERE key='retrieval_rollback'").fetchone()
            if not row:
                raise ValueError("no_recorded_retrieval_rollback")
            previous = json.loads(row[0])
            _, documents, chunks = generation(ROOT, generation_id=previous["generation_id"])
            if previous["ranking_version"] not in ("v1", "v2"):
                raise ValueError("invalid_rollback_ranker")
            counts = connection.execute("SELECT (SELECT count(*) FROM corpus_documents WHERE generation_id=%s),"
                                        "(SELECT count(*) FROM corpus_chunks c JOIN embedding_cache e ON c.content_hash=e.content_hash AND e.model_revision=%s WHERE c.generation_id=%s)",
                                        (previous["generation_id"], model_revision(ROOT), previous["generation_id"])).fetchone()
            if counts != (len(documents), len(chunks)):
                raise ValueError("rollback_index_incomplete")
            with connection.transaction():
                connection.execute("UPDATE app_state SET value=%s WHERE key='active_generation'", (previous["generation_id"],))
                connection.execute("UPDATE app_state SET value=%s WHERE key='ranking_version'", (previous["ranking_version"],))
            return previous
        finally:
            connection.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "activate", "rollback"))
    args = parser.parse_args()
    result = gates() if args.action == "check" else activate() if args.action == "activate" else rollback()
    print(json.dumps(result, indent=2))
    if args.action == "check" and not result["pass"]:
        raise SystemExit(2)
