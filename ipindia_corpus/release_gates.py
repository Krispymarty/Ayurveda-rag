"""Fail-closed activation of unattended three-source synchronization."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from .indexer import _connect, generation
from .sync import Ledger, SOURCES
from .sync_runner import ROOT, atomic_json, policy
from .passage_evaluation import validate_labels
from .locking import file_lock
from .paths import archive_path, immutable_write


def check(root: Path = ROOT) -> dict:
    failures, active = [], None
    try:
        connection = _connect()
        try:
            row = connection.execute("SELECT value FROM app_state WHERE key='active_generation'").fetchone()
            active = row[0] if row else None
        finally:
            connection.close()
        if not active:
            raise ValueError("no_active_index")
        _, documents, chunks = generation(root, generation_id=active)
        coverage = {source: sum(doc["source"] == source for doc in documents) for source in SOURCES}
        for source, count in coverage.items():
            if not count:
                failures.append(f"{source}: no searchable capture")
    except Exception as exc:
        failures.append(f"index: {type(exc).__name__}: {exc}")
        documents, chunks = [], []
    ledger = Ledger(root / "data" / "manifests" / "manifest.sqlite", root / "data" / "raw")
    try:
        for source in SOURCES:
            try:
                policy(root, source)
            except (ValueError, OSError, KeyError) as exc:
                failures.append(str(exc))
            discovery = ledger.db.execute("SELECT min(run_id) FROM sync_runs WHERE source_id=? AND status='success' AND discovery=1", (source,)).fetchone()[0]
            cycles = ledger.db.execute("SELECT count(*) FROM sync_runs WHERE source_id=? AND status='success' AND discovery=0 AND run_id>?", (source, discovery or 2**63 - 1)).fetchone()[0]
            if not discovery or cycles < 2:
                failures.append(f"{source}: discovery and two incremental cycles required")
            if source == "indiacode":
                related = ledger.db.execute("SELECT status FROM pipeline_events WHERE source_id=? AND stage='discovery' ORDER BY event_id DESC LIMIT 1", (source,)).fetchone()
                if not related or related[0] != "success":
                    failures.append("indiacode: linked subordinate legislation requires verification")
    finally:
        ledger.close()
    try:
        path = root / "tests" / "retrieval_queries.json"
        labels = json.loads(path.read_text())
        validate_labels(labels, documents, chunks)
        report = json.loads((root / "reports" / "retrieval_evaluation.json").read_text())
        if not report.get("pass") or report.get("partial") or report["generation_id"] != active or report["label_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
            failures.append("current frozen retrieval evaluation has not passed")
        if report.get("recall_at_5", 0) < .9 or report.get("p95_ms", float("inf")) > 2000 or report.get("observations", 0) < 90:
            failures.append("retrieval accuracy or latency gate failed")
    except (OSError, ValueError, KeyError) as exc:
        failures.append(f"evaluation: {exc}")
    for name in ("verification/offline.json", "verification/postgres.json", "passage_validation.json"):
        try:
            proof = json.loads((root / "reports" / name).read_text(encoding="utf-8"))
            if not proof.get("pass"):
                failures.append(f"verification failed: {name}")
            proof_generation = proof.get("generation_id") or proof.get("active_generation_id")
            if proof_generation and proof_generation != active:
                failures.append(f"verification is for another generation: {name}")
        except (OSError, ValueError) as exc:
            failures.append(f"verification unavailable: {name}: {exc}")
    result = {"pass": not failures, "active_generation_id": active, "failures": failures, "checked_at": datetime.now(timezone.utc).isoformat()}
    atomic_json(root / "reports" / "release_gates.json", result)
    return result


def enable_scheduler(root: Path = ROOT):
    with file_lock(root / "data" / "locks" / "maintenance.lock"):
        result = check(root)
        if not result["pass"]:
            raise ValueError("release_gates_failed: " + "; ".join(result["failures"]))
        gate_hash = hashlib.sha256((root / "reports" / "release_gates.json").read_bytes()).hexdigest()
        gate_path = root / "data" / "raw" / "releases" / f"{gate_hash}.json"
        gate_path.parent.mkdir(parents=True, exist_ok=True)
        immutable_write(gate_path, (root / "reports" / "release_gates.json").read_bytes())
        atomic_json(root / "config" / "release.json", {"scheduler_enabled": True, "validated_generation_id": result["active_generation_id"], "gate_sha256": gate_hash, "gate_path": gate_path.relative_to(root).as_posix(), "activated_at": result["checked_at"]})
        ledger = Ledger(root / "data" / "manifests" / "manifest.sqlite", root / "data" / "raw")
        try:
            with ledger.db:
                ledger.db.execute("UPDATE source_registry SET enabled=1,sync_frequency='daily_check_weekly_discovery'")
        finally:
            ledger.close()
        return result


def verify_activation(root: Path = ROOT):
    config = json.loads((root / "config" / "release.json").read_text())
    if not config.get("scheduler_enabled"):
        raise ValueError("scheduler_disabled_until_three_source_release_passes")
    path = config.get("gate_path", "")
    if not path.startswith("data/raw/releases/"):
        raise ValueError("scheduler_activation_evidence_missing")
    raw = archive_path(root, path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != config.get("gate_sha256") or not json.loads(raw).get("pass"):
        raise ValueError("scheduler_activation_evidence_invalid")
    return config


if __name__ == "__main__":
    report = check()
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["pass"] else 2)
