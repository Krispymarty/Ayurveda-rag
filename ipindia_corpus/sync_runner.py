"""Curated-source synchronization with policy gates and resumable checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from contextlib import closing
from datetime import datetime, timezone, timedelta
from pathlib import Path

from .connectors import SourceClient, discover_links, official_url
from .pipeline import Fetch, extract
from .paths import archive_path, immutable_write
from .sync import Ledger, SOURCES
from .locking import file_lock
from .access_review import verify_review

ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def catalog(root: Path = ROOT) -> dict:
    return read_json(root / "config" / "curated_sources.json")


def policy(root: Path, source: str) -> dict:
    item = read_json(root / "config" / "source_policies.json")[source]
    if not all(item.get(key) for key in ("approved", "reviewer", "reviewed_at", "policy_url")):
        raise ValueError(f"{source}: recorded policy review required")
    verify_review(root, source, item)
    return item


def record_review(root: Path, source: str, reviewer: str, policy_url: str, notes: str, permitted_prefixes: list[str]) -> None:
    if source not in SOURCES or not reviewer.strip() or not policy_url.startswith("https://"):
        raise ValueError("source, reviewer and HTTPS policy URL required")
    path = root / "config" / "source_policies.json"
    data = read_json(path)
    data[source] = {"approved": True, "reviewer": reviewer.strip(),
                    "reviewed_at": datetime.now(timezone.utc).isoformat(),
                    "policy_url": policy_url, "notes": notes, "permitted_prefixes": permitted_prefixes,
                    "evidence": read_json(root / "reports" / "access" / f"{source}.json")["evidence"]}
    verify_review(root, source, data[source])
    atomic_json(path, data)


def seed_urls(root: Path, source: str) -> list[str]:
    config = catalog(root)[source]
    urls = list(config["seeds"])
    if source == "ipindia" and config.get("archive_existing"):
        path = root / "data" / "documents" / "documents.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                doc = json.loads(line)
                if doc.get("source") == source and doc.get("document_type") in {
                    "Act", "Rules", "Guidelines", "Guidance", "Manual", "Notification", "Regulations"
                }:
                    urls.append(doc["source_url"])
    if source == "tkdl" and config.get("archive_existing_candidates"):
        path = root / "data" / "documents" / "tkdl_candidates.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                url = json.loads(line)["source_url"]
                if "/langdefault/Ayurveda/" in url:
                    urls.append(url)
    result = []
    database = root / "data" / "manifests" / "manifest.sqlite"
    if database.exists():
        import sqlite3
        with closing(sqlite3.connect(database)) as connection:
            if connection.execute("SELECT 1 FROM sqlite_master WHERE name='accepted_documents'").fetchone():
                for row in connection.execute("SELECT record_json FROM accepted_documents WHERE source_id=?", (source,)):
                    urls.append(json.loads(row[0])["canonical_url"])
    for url in urls:
        official = official_url(source, url)
        if official not in result:
            result.append(official)
    return result


def _prior(ledger: Ledger, url: str):
    return ledger.db.execute("SELECT m.* FROM document_urls u JOIN document_manifest m USING(canonical_id) WHERE u.url=?", (url,)).fetchone()


def _quarantine(root: Path, source: str, body: bytes, content_type: str) -> None:
    suffix = ".pdf" if content_type == "application/pdf" else ".html"
    path = root / "data" / "raw" / "quarantine" / source / f"{hashlib.sha256(body).hexdigest()}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    immutable_write(path, body)


def _is_discovery(source: str, url: str) -> bool:
    path = url.lower().split("?", 1)[0]
    return ((source == "ipindia" and ("/resource/" in path or "-resources-" in path))
            or (source == "indiacode" and ("/handle/" in path or ("/server/api" in path and not path.endswith("/content")))))


def sync_source(source: str, root: Path = ROOT, *, discover: bool = False,
                client: SourceClient | None = None) -> dict:
    with file_lock(Path(root) / "data" / "locks" / "maintenance.lock"):
        with file_lock(Path(root) / "data" / "locks" / f"{source}.lock"):
            from .generations import accepted_rows
            accepted_rows(Path(root))
            return _sync_source(source, root, discover=discover, client=client)


def _sync_source(source: str, root: Path = ROOT, *, discover: bool = False,
                 client: SourceClient | None = None) -> dict:
    review = policy(root, source)
    root = Path(root)
    config = catalog(root)[source]
    client = client or SourceClient(source, root / "data" / "raw", permitted_prefixes=review["permitted_prefixes"], access_review=review)
    if isinstance(client, SourceClient):
        client.access_review = review
    checkpoint = root / "data" / "manifests" / "checkpoints" / f"{source}.json"
    state = read_json(checkpoint) if checkpoint.exists() else {"pending": seed_urls(root, source), "completed": [], "discover": discover}
    state.setdefault("links", [])
    if state.get("retry_not_before") and datetime.fromisoformat(state["retry_not_before"]) > datetime.now(timezone.utc):
        raise ValueError("source_retry_deferred_until:" + state["retry_not_before"])
    if state["discover"] != discover:
        raise ValueError("resume checkpoint with the same discovery mode first")
    ledger = Ledger(root / "data" / "manifests" / "manifest.sqlite", root / "data" / "raw")
    ledger.recover_abandoned(source)
    run_id = ledger.start(source, discovery=discover)
    failures = []
    discovered = set(state["pending"]) | set(state["completed"])
    try:
        if source == "indiacode" and config.get("api_root") and not state.get("api_discovery_done") and (
                discover or not ledger.db.execute("SELECT 1 FROM accepted_documents WHERE source_id=?", (source,)).fetchone()):
            from .india_code import discover as discover_india_code
            found, metadata, metadata_urls = discover_india_code(client, ledger, run_id, config)
            state["document_metadata"] = metadata
            state["links"] = found
            state["completed"] = list(dict.fromkeys(state["completed"] + metadata_urls))
            state["pending"] = list(dict.fromkeys([url for url in state["pending"] if url not in metadata_urls] + found))
            state["api_discovery_done"] = True
            atomic_json(checkpoint, state)
        while state["pending"]:
            url = state["pending"][0]
            prior = _prior(ledger, url)
            try:
                response = client.get(url, etag=prior["etag"] if prior else None,
                                      last_modified=prior["last_modified"] if prior else None)
                if response.status_code == 304:
                    ledger.checked(run_id, url, etag=response.etag, last_modified=response.last_modified)
                    body = archive_path(root, prior["raw_path"]).read_bytes() if state["discover"] and url in config["seeds"] else None
                    if body is not None and hashlib.sha256(body).hexdigest() != prior["sha256"]:
                        raise ValueError("archived discovery bytes changed")
                else:
                    body = response.body
                    if body is None:
                        raise ValueError("empty response")
                    if not _is_discovery(source, url):
                        probe = Fetch(response.url, body, response.content_type,
                                      datetime.now(timezone.utc).isoformat(), root / "data" / "raw")
                        try:
                            _, extracted, _ = extract(probe)
                            if len(extracted.strip()) < 80:
                                raise ValueError("too_little_extractable_text")
                        except (ValueError, OSError, ImportError) as exc:
                            _quarantine(root, source, body, response.content_type)
                            raise ValueError(f"manual_review_required: parse_failed: {exc}") from exc
                    ledger.observe(run_id, response.url, body, content_type=response.content_type,
                                   etag=response.etag, last_modified=response.last_modified,
                                   metadata={**(json.loads(prior["metadata_json"]) if prior else {}),
                                             **state.get("document_metadata", {}).get(url, {}),
                                             "evidence_role": config["evidence_role"],
                                             "discovery_only": _is_discovery(source, url)},
                                   canonical_id_hint=prior["canonical_id"] if prior else None,
                                   capture=getattr(response, "capture_receipt", None))
                    if response.url != url and _prior(ledger, url) is None:
                        canonical = _prior(ledger, response.url)["canonical_id"]
                        with ledger.db:
                            ledger.db.execute("INSERT INTO document_urls(url,canonical_id) VALUES (?,?)", (url, canonical))
                            ledger._change(run_id, canonical, url, "URL_MOVED",
                                           detail=f"same-source redirect to {response.url}")
                cached_html = bool(prior and archive_path(root, prior["raw_path"]).suffix == ".html")
                if state["discover"] and body and url in config["seeds"] and (response.content_type.startswith("text/html") or cached_html):
                    for found in discover_links(source, response.url, body, config["max_discovered"]):
                        if found not in state["links"]:
                            state["links"].append(found)
                        if found not in discovered:
                            if len(discovered) >= config["max_discovered"]:
                                raise ValueError("incomplete_discovery_limit")
                            state["pending"].append(found)
                            discovered.add(found)
            except (ValueError, OSError) as exc:
                ledger.failed(run_id, url, str(exc))
                failures.append({"url": url, "reason": str(exc)})
                deferred = re.search(r"retry_after_deferred:(\d+)", str(exc))
                if deferred:
                    state["retry_not_before"] = (datetime.now(timezone.utc) + timedelta(seconds=int(deferred.group(1)))).isoformat()
                    atomic_json(checkpoint, state)
                    break
            state["completed"].append(state["pending"].pop(0))
            atomic_json(checkpoint, state)
        discovery_path = root / "data" / "manifests" / "discovery" / f"{source}.json"
        discovered_state = None
        retry_discovery = False
        if discover and not failures:
            previous = set(read_json(discovery_path)["links"]) if discovery_path.exists() else set()
            current = set(state["links"])
            if previous and len(previous - current) > max(5, int(len(previous) * .2)):
                reason = "manual_review_required: discovery_link_set_shrank_abruptly"
                ledger.failed(run_id, config["seeds"][0], reason)
                failures.append({"url": config["seeds"][0], "reason": reason})
                retry_discovery = True
            else:
                for missing in sorted(previous - current):
                    ledger.removed_from_source(run_id, missing, verified_discovery=True)
                discovered_state = {"source": source, "checked_at": datetime.now(timezone.utc).isoformat(),
                                    "links": sorted(current)}
        result = ledger.finish(run_id, "partial" if failures else "success", root / "reports" / "sync",
                               accepted_urls=state["completed"], discovery=discover)
        ledger.event(source, "capture", result["status"])
        result["failures"] = failures
        if discovered_state is not None:
            atomic_json(discovery_path, discovered_state)
        if failures:
            state["pending"] = config["seeds"] if retry_discovery else list(dict.fromkeys([item["url"] for item in failures] + state["pending"]))
            if retry_discovery:
                state["completed"], state["links"] = [], []
            atomic_json(checkpoint, state)
        else:
            checkpoint.unlink(missing_ok=True)
        return result
    except BaseException as exc:
        if ledger.db.execute("SELECT status FROM sync_runs WHERE run_id=?", (run_id,)).fetchone()[0] == "running":
            ledger.finish(run_id, "failed", root / "reports" / "sync")
        ledger.event(source, "capture", "failed", str(exc))
        raise
    finally:
        ledger.close()


def process_source(source: str, *, discover=False, root: Path = ROOT,
                   client: SourceClient | None = None) -> dict:
    """Capture and queue a candidate; production activation is always explicit."""
    root = Path(root)
    result = sync_source(source, root, discover=discover, client=client)
    if result["status"] != "success":
        return result
    ledger = Ledger(root / "data" / "manifests" / "manifest.sqlite", root / "data" / "raw")
    stage = "candidate_publication"
    try:
        with file_lock(root / "data" / "locks" / "maintenance.lock"):
            from .generations import publish
            from .indexer import generation
            result["published"] = publish(root, lock_held=True, repair_pdf=True,
                                          update_pointer=False)
            identifier = result["published"]["generation_id"]
            _, documents, _ = generation(root, generation_id=identifier)
            if not any(doc["source"] == source for doc in documents):
                raise ValueError("candidate_missing_captured_source")
            manifest_path = root / "data" / "generations" / identifier / "manifest.json"
            receipt = {"schema_version": 1, "generation_id": identifier,
                       "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                       "source": source, "capture_run_id": result["run_id"],
                       "queued_at": datetime.now(timezone.utc).isoformat(),
                       "state": "pending_review", "mode": "candidate_only",
                       "production_activation": False}
            receipt_path = root / "data" / "candidates" / f"{identifier}.json"
            immutable_write(receipt_path, json.dumps(receipt, indent=2).encode("utf-8"))
            result["candidate"] = receipt
            ledger.event(source, stage, "success", generation_id=identifier,
                         detail="pending review; production index and pointers unchanged")
    except Exception as exc:
        ledger.event(source, stage, "failed", str(exc))
        with ledger.db:
            ledger.db.execute("UPDATE source_registry SET next_check=? WHERE source_id=?", ((datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(), source))
        result["pipeline_failure"] = {"stage": stage, "reason": str(exc)}
        result["status"] = "pipeline_failed"
    finally:
        ledger.close()
    return result


def due_mode(row, now):
    def due(value):
        return not value or datetime.fromisoformat(value) <= now
    return True if due(row["next_discovery"]) else False if due(row["next_check"]) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    review = commands.add_parser("record-review")
    review.add_argument("--source", choices=sorted(SOURCES), required=True)
    review.add_argument("--reviewer", required=True)
    review.add_argument("--policy-url", required=True)
    review.add_argument("--notes", default="")
    review.add_argument("--permitted-prefix", action="append", required=True)
    sync = commands.add_parser("sync")
    sync.add_argument("--source", choices=sorted(SOURCES), required=True)
    sync.add_argument("--discover", action="store_true")
    schedule = commands.add_parser("schedule")
    schedule.add_argument("--poll-seconds", type=int, default=3600)
    commands.add_parser("enable-scheduler")
    args = parser.parse_args()
    if args.command == "record-review":
        record_review(ROOT, args.source, args.reviewer, args.policy_url, args.notes, args.permitted_prefix)
        return 0
    if args.command == "enable-scheduler":
        from .release_gates import enable_scheduler
        print(json.dumps(enable_scheduler(ROOT), indent=2))
        return 0
    if args.command == "sync":
        result = process_source(args.source, discover=args.discover)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "success" else 2
    from .release_gates import verify_activation
    verify_activation(ROOT)
    while True:
        now = datetime.now(timezone.utc)
        for source in SOURCES:
            try:
                item = read_json(ROOT / "config" / "source_policies.json")[source]
                if not item.get("approved"):
                    continue
                ledger = Ledger(ROOT / "data" / "manifests" / "manifest.sqlite", ROOT / "data" / "raw")
                row = ledger.db.execute("SELECT * FROM source_registry WHERE source_id=?", (source,)).fetchone()
                discovery = due_mode(row, now)
                ledger.close()
                if not row["enabled"]:
                    continue
                checkpoint = ROOT / "data" / "manifests" / "checkpoints" / f"{source}.json"
                if discovery is not None or checkpoint.exists():
                    discovery = read_json(checkpoint)["discover"] if checkpoint.exists() else discovery
                    result = process_source(source, discover=discovery)
                    print(json.dumps(result), flush=True)
            except Exception as exc:
                print(json.dumps({"source": source, "error": str(exc)}), flush=True)
        time.sleep(max(60, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
