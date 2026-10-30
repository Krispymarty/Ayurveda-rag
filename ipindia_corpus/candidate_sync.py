"""Policy-gated candidate synchronization; never indexes or activates production."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .locking import file_lock
from .sync import SOURCES
from .sync_runner import ROOT, atomic_json, policy, process_source, read_json


def _time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("candidate schedule timestamps must include timezone")
    return parsed


def _configuration(root: Path) -> dict:
    config = read_json(root / "config" / "synchronization.json")
    if config.get("mode") != "candidate_only":
        raise ValueError("candidate_only mode is required")
    for field in ("poll_seconds", "check_interval_seconds", "discovery_interval_seconds",
                  "retry_interval_seconds"):
        if type(config.get(field)) is not int or not 60 <= config[field] <= 31_536_000:
            raise ValueError(f"invalid candidate scheduler {field}")
    return config


def _state(root: Path) -> dict:
    path = root / "data" / "manifests" / "candidate_schedule.json"
    return read_json(path) if path.exists() else {"schema_version": 1, "sources": {}}


def _pending(root: Path) -> tuple[list[dict], list[dict]]:
    pending, failures = [], []
    for path in sorted((root / "data" / "candidates").glob("*.json")):
        try:
            receipt = read_json(path)
            identifier = receipt["generation_id"]
            if not identifier or any(char not in "0123456789TZ" for char in identifier):
                raise ValueError("invalid candidate generation ID")
            manifest = root / "data" / "generations" / identifier / "manifest.json"
            if hashlib.sha256(manifest.read_bytes()).hexdigest() != receipt["manifest_sha256"]:
                raise ValueError("candidate manifest hash mismatch")
            pending.append(receipt)
        except (OSError, ValueError, KeyError) as exc:
            failures.append({"receipt": path.name, "reason": str(exc)})
    return pending, failures


def status(root: Path = ROOT, now: datetime | None = None) -> dict:
    """Read candidate scheduler readiness without fetching or changing state."""
    root = Path(root)
    now = now or datetime.now(timezone.utc)
    config, state = _configuration(root), _state(root)
    policies = read_json(root / "config" / "source_policies.json")
    sources = []
    for source in SOURCES:
        item = state["sources"].get(source, {})
        enabled = bool(config.get("enabled") and config.get("sources", {}).get(source, {}).get("enabled"))
        review_status, reason = "verified", None
        try:
            policy(root, source)
        except (OSError, ValueError, KeyError) as exc:
            review_status, reason = "blocked", str(exc)
        checkpoint = root / "data" / "manifests" / "checkpoints" / f"{source}.json"
        check_due = not item.get("next_check") or _time(item["next_check"]) <= now
        discovery_due = not item.get("next_discovery") or _time(item["next_discovery"]) <= now
        retry_due = not item.get("next_attempt") or _time(item["next_attempt"]) <= now
        sources.append({"source": source, "enabled": enabled,
                        "eligible": enabled and review_status == "verified",
                        "access_review": review_status, "blocked_reason": reason,
                        "review_conclusion": policies.get(source, {}).get("conclusion"),
                        "review_notes": policies.get(source, {}).get("notes"),
                        "next_check": item.get("next_check"),
                        "next_discovery": item.get("next_discovery"),
                        "next_attempt": item.get("next_attempt"),
                        "check_due": check_due, "discovery_due": discovery_due,
                        "resume_checkpoint": checkpoint.exists(),
                        "due": retry_due and (check_due or discovery_due or checkpoint.exists()),
                        "last_result": item.get("last_result"),
                        "last_successful_capture": item.get("last_successful_capture"),
                        "last_candidate": item.get("last_candidate")})
    pending, failures = _pending(root)
    eligible = sum(item["eligible"] for item in sources)
    return {"schema_version": 1, "mode": "candidate_only", "enabled": bool(config.get("enabled")),
            "checked_at": now.isoformat(), "production_mutation": False,
            "state": "disabled" if not config.get("enabled") else "ready" if eligible else "access_blocked",
            "eligible_sources": eligible, "sources": sources,
            "pending_candidates": pending, "queue_failures": failures}


def run_once(root: Path = ROOT, now: datetime | None = None, *, clients: dict | None = None) -> dict:
    """Run each permitted due source once under a process lock and queue candidates."""
    root = Path(root)
    now = now or datetime.now(timezone.utc)
    with file_lock(root / "data" / "locks" / "candidate-worker.lock"):
        config, state = _configuration(root), _state(root)
        snapshot, results = status(root, now), []
        for source_status in snapshot["sources"]:
            if not source_status["eligible"] or not source_status["due"]:
                continue
            source = source_status["source"]
            item = state["sources"].setdefault(source, {})
            checkpoint = root / "data" / "manifests" / "checkpoints" / f"{source}.json"
            try:
                discover = source_status["discovery_due"]
                if checkpoint.exists():
                    checkpoint_state = read_json(checkpoint)
                    discover = checkpoint_state["discover"]
                    retry_after = checkpoint_state.get("retry_not_before")
                    if retry_after and _time(retry_after) > now:
                        item["next_attempt"] = retry_after
                        continue
                result = process_source(source, root=root, discover=discover,
                                        client=(clients or {}).get(source))
            except Exception as exc:
                result = {"source": source, "status": "failed", "reason": str(exc)}
            item["last_result"] = result
            if result["status"] in {"success", "pipeline_failed"}:
                item["last_successful_capture"] = now.isoformat()
            if result["status"] == "success":
                item["last_candidate"] = result["candidate"]["generation_id"]
                item["next_check"] = (now + timedelta(seconds=config["check_interval_seconds"])).isoformat()
                if discover:
                    item["next_discovery"] = (now + timedelta(seconds=config["discovery_interval_seconds"])).isoformat()
                item["next_attempt"] = None
            else:
                item["next_attempt"] = (now + timedelta(seconds=config["retry_interval_seconds"])).isoformat()
            results.append(result)
            atomic_json(root / "data" / "manifests" / "candidate_schedule.json", state)
        # Persist deferred Retry-After deadlines even when no fetch was attempted.
        atomic_json(root / "data" / "manifests" / "candidate_schedule.json", state)
        report = status(root, now)
        report["results"] = results
        atomic_json(root / "reports" / "candidate_sync_status.json", report)
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    commands.add_parser("run-once")
    schedule = commands.add_parser("schedule")
    schedule.add_argument("--poll-seconds", type=int)
    args = parser.parse_args()
    if args.command == "status":
        print(json.dumps(status(args.root), indent=2))
        return 0
    while True:
        report = run_once(args.root)
        print(json.dumps(report), flush=True)
        if args.command == "run-once":
            return 2 if any(result["status"] != "success" for result in report["results"]) else 0
        interval = args.poll_seconds or _configuration(args.root)["poll_seconds"]
        time.sleep(max(60, interval))


if __name__ == "__main__":
    raise SystemExit(main())
