"""Bounded policy-evidence refresh and staged-source reconciliation; no admission."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup

from ipindia_corpus.access_review import HOMES, capture
from ipindia_corpus.corpus_certification import audit
from ipindia_corpus.paths import archive_path, immutable_write
from ipindia_corpus.sync import validate_url
from ipindia_corpus.sync_runner import policy, seed_urls

ROOT = Path(__file__).resolve().parents[1]
POLICY_LINK = re.compile(r"copyright\s+(?:and\s+ipr\s+)?policy|website\s+polic|terms\s+(?:of\s+use|and\s+conditions)|privacy\s+policy|access\s+policy|disclaimer|/(?:website|copyright|privacy|access)[-_]polic|/terms[-_]|/disclaimer", re.I)


def refresh_access(root, source):
    """Capture at most robots+home+four discovered policy pages, never corpus pages."""
    home = HOMES[source]
    base = urlsplit(home)
    records = []
    for url, kind in ((f"https://{base.netloc}/robots.txt", "robots"), (home, "homepage")):
        if records:
            time.sleep(1.5)
        records.append(capture(root, source, url, kind))
    candidates = set()
    homepage = records[1]
    if homepage.get("status") == 200 and homepage.get("raw_path"):
        soup = BeautifulSoup(archive_path(root, homepage["raw_path"]).read_bytes(), "html.parser")
        for link in soup.find_all("a", href=True):
            if not POLICY_LINK.search(link.get_text(" ", strip=True) + " " + link["href"]):
                continue
            url = urljoin(homepage.get("final_url", home), link["href"])
            try:
                validate_url(source, url)
            except ValueError:
                continue
            candidates.add(url)
    for url in sorted(candidates)[:4]:
        time.sleep(1.5)
        records.append(capture(root, source, url, "policy"))
    return {"source": source, "checked_at": datetime.now(timezone.utc).isoformat(),
            "reviewer": "Codex bounded access-evidence collection", "approved": False,
            "conclusion": "pending_access_review" if any(r["kind"] == "policy" and r.get("status") == 200 for r in records) else "blocked_access_evidence",
            "evidence": records, "policy_candidates_discovered": len(candidates),
            "max_requests_without_redirects": 6, "corpus_requests": 0,
            "note": "Evidence collection is not source access approval. Existing policy configuration is unchanged."}


def reconcile(root):
    root = Path(root)
    observed = audit(root)
    catalog = json.loads((root / "config/curated_sources.json").read_text(encoding="utf-8"))
    policies_raw = (root / "config/source_policies.json").read_bytes()
    reviews = {}
    for source in ("indiacode", "tkdl"):
        try:
            policy(root, source)
            reviews[source] = {"permitted": True, "blocker": None}
        except (ValueError, OSError, KeyError) as exc:
            reviews[source] = {"permitted": False, "blocker": str(exc)}
    documents = []
    for document in observed["documents"]:
        path = root / document["path"]
        title = path.read_text(encoding="utf-8", errors="replace").splitlines()[0].lstrip("# ")
        if document["source"] == "indiacode":
            scope = any(title.rstrip(".").casefold() == item["title"].casefold() for item in catalog["indiacode"]["instruments"])
        else:
            scope = bool(re.search(r"/langdefault/(?:common|Ayurveda)/", document["source_url"] or "", re.I))
        document.update(title=title, curated_scope_candidate=scope,
                        recapture_action="recapture_only_after_access_review" if scope else "exclude_from_current_curated_recovery",
                        can_recapture_now=scope and reviews[document["source"]]["permitted"],
                        expected_receipt_sidecar=document["path"] + ".receipt.json",
                        missing_evidence=["authentic_original_html_pdf", "original_to_passage_fidelity", "verified_citation_anchors"] +
                                         ([] if document["original_receipt"]["verified"] else ["valid_original_capture_receipt"]))
        documents.append(document)
    return {"schema_version": 1, "at": datetime.now(timezone.utc).isoformat(),
            "mode": "candidate_readiness_only", "source_policies_sha256": hashlib.sha256(policies_raw).hexdigest(),
            "sources": observed["sources"], "access_reviews": reviews,
            "documents": documents, "inventory_sha256": observed["inventory_sha256"],
            "indiacode_curated_instruments": catalog["indiacode"]["instruments"],
            "tkdl_public_seed_candidates": seed_urls(root, "tkdl"),
            "newly_certified_documents": 0, "candidate_generations_created": [],
            "production_activation": False, "three_source_ready": False,
            "note": "Fresh captures cannot retroactively authenticate staged transformed files; fidelity remains unverified."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--refresh-access", action="store_true")
    args = parser.parse_args(argv)
    result = reconcile(args.root)
    if args.refresh_access:
        result["fresh_access_evidence"] = [refresh_access(args.root, source) for source in ("indiacode", "tkdl")]
    payload = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
    path = args.root / "data/next_milestone" / ("source_readiness_" + hashlib.sha256(payload).hexdigest()[:16] + ".json")
    immutable_write(path, payload)
    print(json.dumps({"artifact": str(path), "sources": result["sources"], "newly_certified_documents": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
