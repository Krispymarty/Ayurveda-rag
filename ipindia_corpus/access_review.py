"""Capture access evidence without approving an unavailable or ambiguous policy."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from .sync import validate_url
from .paths import archive_path, immutable_write

CONTACT = "https://github.com/Krispymarty/Ip-Shakti--Rag"
HOMES = {"ipindia": "https://ipindia.gov.in/", "indiacode": "https://indiacode.gov.in/",
         "tkdl": "https://www.tkdl.res.in/tkdl/langdefault/common/Home.asp?GL=Eng"}


def capture(root: Path, source: str, url: str, kind: str, session=None) -> dict:
    validate_url(source, url)
    if session is None:
        session = requests.Session()
        session.trust_env = False
    stamp = datetime.now(timezone.utc).isoformat()
    record = {"url": url, "requested_url": url, "kind": kind, "at": stamp,
              "retrieved_at": stamp, "source": source, "redirect_chain": [],
              "purpose": "access_evidence_only", "capture_permission_granted": False}
    current = url
    try:
        for _ in range(5):
            validate_url(source, current)
            response = session.get(current, timeout=(5, 12), allow_redirects=False, stream=True,
                                   headers={"User-Agent": f"AyurIP-AccessReview/1.0 ({CONTACT})"})
            if response.status_code in (301, 302, 303, 307, 308):
                target = urljoin(current, response.headers.get("Location", ""))
                response.close()
                validate_url(source, target)
                record["redirect_chain"].append({"from_url": current, "to_url": target, "status": response.status_code})
                current = target
                continue
            pieces, size = [], 0
            try:
                for piece in response.iter_content(65536):
                    size += len(piece)
                    if size > 2_000_000:
                        raise ValueError("access_evidence_too_large")
                    pieces.append(piece)
            finally:
                response.close()
            body = b"".join(pieces)
            digest = hashlib.sha256(body).hexdigest()
            path = root / "data" / "raw" / "access" / source / f"{digest}.bin"
            path.parent.mkdir(parents=True, exist_ok=True)
            immutable_write(path, body)
            record.update(final_url=current, status=response.status_code, sha256=digest,
                          raw_path=path.relative_to(root).as_posix(), content_type=response.headers.get("Content-Type", ""),
                          byte_length=len(body), etag=response.headers.get("ETag"), last_modified=response.headers.get("Last-Modified"))
            break
        else:
            raise ValueError("too_many_redirects")
    except (requests.RequestException, ValueError, OSError) as exc:
        record["error"] = type(exc).__name__ + ": " + str(exc)[:300]
    receipt = json.dumps(record, sort_keys=True).encode()
    path = root / "data" / "raw" / "access" / source / (hashlib.sha256(receipt).hexdigest() + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    immutable_write(path, receipt)
    record["receipt_path"] = path.relative_to(root).as_posix()
    return record


def collect(root: Path, source: str, extra_policy_urls=()) -> dict:
    homepage = HOMES[source]
    base = urlsplit(homepage)
    records = []
    robots = capture(root, source, f"https://{base.netloc}/robots.txt", "robots")
    records.append(robots)
    time.sleep(1.5)
    home = capture(root, source, homepage, "homepage")
    records.append(home)
    candidates = set(extra_policy_urls)
    if home.get("status") == 200:
        soup = BeautifulSoup(archive_path(root, home["raw_path"]).read_bytes(), "html.parser")
        for link in soup.find_all("a", href=True):
            label = (link.get_text(" ", strip=True) + " " + link["href"]).lower()
            if re.search(r"\b(?:copyright\s+(?:and\s+ipr\s+)?policy|website\s+polic(?:y|ies)|terms\s+(?:of\s+use|and\s+conditions)|privacy\s+policy|access\s+policy|disclaimer)\b|/(?:website|copyright|privacy|access)[-_]polic|/terms[-_]|/disclaimer", label):
                url = urljoin(home.get("final_url", homepage), link["href"])
                try:
                    validate_url(source, url)
                    candidates.add(url)
                except ValueError:
                    continue
    for url in sorted(candidates)[:8]:
        time.sleep(1.5)
        records.append(capture(root, source, url, "policy"))
    robots_ok = robots.get("status") in (200, 404)
    if robots.get("status") == 200:
        raw = archive_path(root, robots["raw_path"]).read_bytes()
        robots_ok = bool(raw) and b"<html" not in raw[:500].lower()
    policy_ok = any(item["kind"] == "policy" and item.get("status") == 200 for item in records)
    report = {"source": source, "reviewer": "Codex access evidence collection", "approved": False,
              "reviewed_at": datetime.now(timezone.utc).isoformat(), "contact": CONTACT,
              "conclusion": "review_needed" if robots_ok and policy_ok else "blocked_access_evidence",
              "evidence": records}
    path = root / "reports" / "access" / f"{source}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def verify_review(root: Path, source: str, review: dict) -> None:
    if not review.get("permitted_prefixes") or not review.get("evidence"):
        raise ValueError("archived access evidence and permitted paths required")
    robots, policies = [], []
    for record in review["evidence"]:
        receipt_path = archive_path(root, record["receipt_path"])
        receipt_bytes = receipt_path.read_bytes()
        if re.fullmatch(r"[0-9a-f]{64}\.json", receipt_path.name) and receipt_path.stem != hashlib.sha256(receipt_bytes).hexdigest():
            raise ValueError("access evidence receipt hash mismatch")
        receipt = json.loads(receipt_bytes)
        # Compare the whole archived observation, including final authority and
        # redirect history. Older receipts remain valid with their old schema.
        if not isinstance(receipt, dict) or receipt != {key: value for key, value in record.items() if key != "receipt_path"}:
            raise ValueError("access evidence receipt mismatch")
        validate_url(source, record.get("final_url", record["url"]))
        if "raw_path" not in record:
            if record["kind"] in ("robots", "policy"):
                raise ValueError("required_access_evidence_unavailable")
            continue
        body = archive_path(root, record["raw_path"]).read_bytes()
        if hashlib.sha256(body).hexdigest() != record["sha256"]:
            raise ValueError("access evidence hash mismatch")
        if record["kind"] == "robots" and record.get("status") in (200, 404):
            if record["status"] == 200 and (not body or b"<html" in body[:500].lower()):
                raise ValueError("robots response is not verified plain text")
            parser = RobotFileParser()
            parser.parse(body.decode("utf-8-sig", "replace").splitlines() if record["status"] == 200 else [])
            host = urlsplit(record.get("final_url", record["url"])).netloc
            for prefix in review["permitted_prefixes"]:
                if not prefix.startswith("/") or not parser.can_fetch(f"AyurIP-Research/1.0 ({CONTACT})", f"https://{host}{prefix}"):
                    raise ValueError("permitted path conflicts with robots")
            robots.append(record)
        if record["kind"] == "policy" and record.get("status") == 200 and record["url"] == review.get("policy_url"):
            policies.append(record)
    if not robots or not policies or not review.get("notes", "").strip():
        raise ValueError("verified robots, policy and review conclusion required")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=HOMES, required=True)
    parser.add_argument("--policy-url", action="append", default=[])
    args = parser.parse_args()
    print(json.dumps(collect(Path(__file__).resolve().parents[1], args.source, args.policy_url), indent=2))
