"""Disabled-source metadata investigation. Never acquires document originals."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.parse import unquote, urlsplit

from bs4 import BeautifulSoup
import requests

from .access_review import HOMES, capture
from .connectors import official_url
from .paths import archive_path, immutable_write
from .sync_runner import ROOT

TOPIC = re.compile(r"biological|biodiversity|benefit.sharing|\bABS\b|amendment|(?:\b|_)act(?:\b|_)|rules|regulation|guideline|procedure|copyright|privacy|disclaimer|terms|policy", re.I)
PRIVATE = re.compile(r"login|signin|authenticate|captcha|password", re.I)
PUBLISHERS = {"nba": "National Biodiversity Authority", "gazette": "Gazette of India / Department of Publication"}
FAMILIES = ("Biological Diversity Act", "Biological Diversity amendments", "Biological Diversity rules", "ABS regulations and procedural guidance")


def metadata_destination(url, *, allow_document=False):
    """One decoded path/query contract for discovery and every metadata request."""
    parsed = urlsplit(url)
    destination = parsed.path + "?" + parsed.query
    try:
        decoded = unquote(destination, errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("metadata_only_invalid_encoding") from exc
    if re.search(r"%(?:[0-9a-f]{2})", decoded, re.I) or re.search(r"%(?![0-9a-f]{2})", destination, re.I):
        raise ValueError("metadata_only_ambiguous_nested_encoding")
    if PRIVATE.search(decoded) or any(ord(char) < 32 for char in decoded):
        raise ValueError("metadata_only_private_url_rejected")
    document = bool(re.search(r"\.pdf(?:$|[?&#=/])", decoded, re.I))
    if document and not allow_document:
        raise ValueError("metadata_only_document_url_rejected")
    return document


def listing_links(source, base, raw):
    result, seen = [], set()
    for anchor in BeautifulSoup(raw, "html.parser").find_all("a", href=True):
        label = anchor.get_text(" ", strip=True)
        row = anchor.find_parent("tr")
        context = row.get_text(" ", strip=True)[:800] if row else label
        try:
            url = official_url(source, anchor["href"], base)
            document = metadata_destination(url, allow_document=True)
        except ValueError:
            continue
        if url in seen or not TOPIC.search(label + " " + context + " " + url):
            continue
        seen.add(url)
        result.append({"title_as_listed": label or None, "context_as_listed": context or None, "url": url, "listing_url": base,
                       "publisher": PUBLISHERS[source], "source": source,
                       "kind": "document_link" if document else "listing_or_policy",
                       "downloaded": False, "legal_status": "unknown", "identity_verified": False})
    return result[:50]


def manual_queue(sources):
    patterns = (r"(?:\b|_)act(?:\b|_)", r"amend", r"rules", r"benefit|\babs\b|regulation|procedure|collaborative|biological resources|form.?10|form.?iii")
    rows = []
    for family, pattern in zip(FAMILIES, patterns):
        candidates, seen = [], set()
        for source in sources.values():
            for item in source["metadata_links"]:
                text = (item["title_as_listed"] or "") + " " + (item.get("context_as_listed") or "") + " " + item["url"]
                is_acts = item["listing_url"].rstrip("/").endswith("/acts-and-rules/acts")
                is_rules = item["listing_url"].rstrip("/").endswith("/acts-and-rules/rules")
                if (item["url"] in seen or "state-biodiversity-rules" in item["listing_url"]
                        or re.search(r"recruitment|vacanc|tender|career", text, re.I)):
                    continue
                if family == FAMILIES[0]:
                    keep = is_acts and "bdact-disclamer" in item["url"]
                elif family == FAMILIES[1]:
                    keep = is_acts and item["kind"] == "document_link" and re.search(pattern, text, re.I)
                elif family == FAMILIES[2]:
                    keep = is_rules and item["kind"] == "document_link" and re.search(pattern, text, re.I)
                else:
                    keep = ((item["kind"] == "document_link" and re.search(pattern, text, re.I))
                            or item["url"].rstrip("/").endswith("/regulations"))
                if not keep:
                    continue
                seen.add(item["url"])
                candidates.append({**item, "candidate_role": "download_link_requires_original_review" if item["kind"] == "document_link" else "reference_listing_not_verified_original"})
        rows.append({"family": family, "state": "not_acquired", "classification": "listing_text_candidate_requires_identity_review",
                     "official_listing_candidates": candidates,
                     "required_before_admission": ["permitted access review", "authentic complete original", "retrieval receipt", "identity and anchors", "source-to-passage fidelity"]})
    return rows


def persist(root, output):
    output["manual_acquisition"] = manual_queue(output["sources"])
    raw = json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
    path = Path(root) / "data/next_stage" / ("source_investigation_" + hashlib.sha256(raw).hexdigest()[:16] + ".json")
    immutable_write(path, raw)
    return path, output


def reparse_metadata(root, observation):
    """Enrich captured listing rows offline, retaining the original request history."""
    raw = Path(observation).read_bytes()
    output = json.loads(raw)
    output["derived_from_sha256"] = hashlib.sha256(raw).hexdigest()
    output["metadata_reparsed_offline"] = True
    for source, entry in output["sources"].items():
        links = []
        for receipt in entry["evidence"]:
            if receipt.get("status") != 200 or receipt["kind"] == "robots":
                continue
            body = archive_path(root, receipt["raw_path"]).read_bytes()
            if hashlib.sha256(body).hexdigest() != receipt["sha256"]:
                raise ValueError("metadata_original_changed")
            links.extend(listing_links(source, receipt.get("final_url", receipt["url"]), body))
        entry["metadata_links"] = links
    return persist(root, output)


class MetadataOnlySession(requests.Session):
    """Refuse PDF destinations and bodies, including a redirect to an original."""
    def __init__(self):
        super().__init__()
        self.trust_env = False

    def get(self, url, **kwargs):
        metadata_destination(url)
        # All redirect destinations return through this guard before network I/O.
        kwargs.update(allow_redirects=False, stream=True)
        response = super().get(url, **kwargs)
        if "application/pdf" in response.headers.get("Content-Type", "").lower():
            response.close()
            raise ValueError("metadata_only_pdf_response_rejected")
        original_iter = response.iter_content

        def metadata_chunks(size):
            for block in original_iter(size):
                if block.startswith(b"%PDF"):
                    response.close()
                    raise ValueError("metadata_only_pdf_body_rejected")
                yield block
        response.iter_content = metadata_chunks
        return response


def investigate(root=ROOT):
    root = Path(root)
    output = {"at": datetime.now(timezone.utc).isoformat(), "mode": "bounded_metadata_only",
              "originals_acquired": 0, "source_approvals_changed": False,
              "candidate_generations_created": [], "activation_allowed": False, "sources": {}}
    for source in ("nba", "gazette"):
        session = MetadataOnlySession()
        home = HOMES[source]
        records, links = [], []
        pending = [(f"https://{urlsplit(home).netloc}/robots.txt", "robots"), (home, "homepage")]
        seen = set()
        try:
            while pending and len(records) < 6:
                url, kind = pending.pop(0)
                if url in seen:
                    continue
                seen.add(url)
                if records:
                    time.sleep(1.5)
                receipt = capture(root, source, url, kind, session)
                records.append(receipt)
                if receipt.get("status") != 200 or not receipt.get("raw_path") or kind == "robots":
                    continue
                discovered = listing_links(source, receipt.get("final_url", url), archive_path(root, receipt["raw_path"]).read_bytes())
                links.extend(item for item in discovered if item not in links)
                # Two metadata listing/policy links beyond robots+home suffice.
                # Document links are catalogued only, never fetched.
                if kind == "homepage":
                    pages = [item for item in discovered if item["kind"] != "document_link" and item["url"] not in seen]
                    policy_pages = [item for item in pages if re.search(r"copyright|privacy|disclaimer|terms|policy", item["url"], re.I)]
                    legal_pages = [item for item in pages if item not in policy_pages]
                    pending.extend((item["url"], "listing_or_policy") for item in policy_pages[:2] + legal_pages[:2])
        finally:
            session.close()
        output["sources"][source] = {"publisher": PUBLISHERS[source], "approved": False,
                                     "evidence": records, "metadata_links": links,
                                     "blocker": "No verified access review permits original acquisition; listing metadata is not original provenance."}
    return persist(root, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--investigate-metadata", required=True, action="store_true")
    parser.parse_args()
    path, result = investigate()
    print(json.dumps({"artifact": str(path), "originals_acquired": result["originals_acquired"],
                      "responses": {name: [(r["url"], r.get("status"), r.get("error")) for r in value["evidence"]]
                                    for name, value in result["sources"].items()}}, indent=2))
