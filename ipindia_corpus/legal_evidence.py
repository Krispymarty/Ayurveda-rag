"""Evidence-bound historical legal relationships; never infer current force."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from .corpus_certification import official_public_url
from .paths import archive_path, immutable_write

RELATIONS = {"amends": r"\bamend(?:s|ment|ing|ed)?\b", "supersedes": r"\bsupersed(?:e|es|ing|ed)\b", "repeals": r"\brepeal(?:s|ed|ing)?\b"}


def _normalize(value: str) -> str:
    return " ".join(value.split())


def validate_annotation(root: Path | str, annotation: dict) -> dict:
    """Validate a reviewed assertion against exact archived text and source identity.

    Textual verification does not establish legal effect or a consolidated current version.
    Unresolved, contradictory or incomplete temporal evidence leaves status unknown.
    """
    root = Path(root).resolve()
    relation = annotation.get("relation")
    if relation not in RELATIONS:
        raise ValueError("unsupported legal relationship")
    if not annotation.get("reviewer") or not annotation.get("review_note"):
        raise ValueError("explicit review identity and note required")
    evidence = annotation.get("evidence", {})
    if not evidence.get("document_id") or not evidence.get("provision") or not evidence.get("quote"):
        raise ValueError("document identity, quote and provision required")
    if not official_public_url(evidence.get("source", ""), evidence.get("official_url", "")):
        raise ValueError("official evidence URL required")
    raw = archive_path(root, evidence["raw_path"])
    body = raw.read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    if digest != evidence.get("sha256"):
        raise ValueError("archived evidence hash mismatch")
    manifest = root / "documents.jsonl"
    if not manifest.exists():
        raise ValueError("document manifest unavailable")
    document = next((json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
                     if line.strip() and json.loads(line).get("document_id") == evidence["document_id"]), None)
    if not document or digest != document.get("sha256", document.get("content_hash")):
        raise ValueError("evidence document does not match archived identity")
    urls = {document.get("source_url"), document.get("canonical_url"), document.get("download_url"), *document.get("alternate_urls", [])}
    if evidence["official_url"] not in urls:
        raise ValueError("citation does not belong to evidence document")
    if document.get("source", "ipindia") != evidence["source"]:
        raise ValueError("source does not match evidence document")
    page = evidence.get("page")
    if body.startswith(b"%PDF"):
        import fitz
        with fitz.open(stream=body, filetype="pdf") as pdf:
            if type(page) is not int or not 1 <= page <= len(pdf):
                raise ValueError("exact PDF page required")
            extracted = pdf[page - 1].get_text()
    else:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(body, "html.parser")
        for node in soup.select("script, style, nav, form"):
            node.decompose()
        anchor = evidence.get("anchor")
        node = soup.find(id=anchor) if isinstance(anchor, str) and anchor else None
        if node is None:
            raise ValueError("HTML evidence requires an existing provision anchor")
        extracted = node.get_text(" ", strip=True)
        if page is not None:
            raise ValueError("HTML evidence cannot invent PDF page")
    quote = _normalize(evidence["quote"])
    target = _normalize(annotation.get("target_title", ""))
    if not quote or quote not in _normalize(extracted):
        raise ValueError("quote absent from cited original page")
    if not target or target.casefold() not in quote.casefold() or not re.search(RELATIONS[relation], quote, re.I):
        raise ValueError("quote must explicitly identify relationship and target")
    if re.search(r"\b(?:not|never|no|without)\b", quote, re.I):
        raise ValueError("negated or ambiguous relationship requires further review")
    statement_scope = annotation.get("statement_scope", "archived_statement_only")
    if statement_scope not in {"archived_statement_only", "operative_wording", "draft_proposal", "publication_uncertain"}:
        raise ValueError("unsupported archived statement scope")
    proposal = bool(re.search(r"\bdraft rules\b|\bproposes? to\b", quote, re.I))
    if proposal != (statement_scope == "draft_proposal"):
        raise ValueError("proposed amendment must be explicitly classified as draft_proposal")
    for qualifier in annotation.get("qualifying_evidence", []):
        if not isinstance(qualifier, dict) or qualifier.get("page") != page or not qualifier.get("quote"):
            raise ValueError("qualifier must cite the same verified original page")
        if _normalize(qualifier["quote"]) not in _normalize(extracted):
            raise ValueError("qualifying quote absent from cited original page")
    if statement_scope == "publication_uncertain" and not annotation.get("qualifying_evidence"):
        raise ValueError("publication uncertainty requires bound qualifying evidence")
    return {**annotation, "status": "unknown", "verified_archived_statement": True,
            "temporal_status": "unverified", "review_scope": "archived_text_only",
            "current_force_established": False, "target_version_id": None,
            "relationship_effective_from": None, "relationship_effective_to": None,
            "statement_scope": statement_scope, "relationship_effect_established": False}


def assess_document_status(annotations: list[dict], *, as_of: str | None = None) -> dict:
    """Expose evidence and conflicts without promoting historical text into current law."""
    subjects = {a.get("evidence", {}).get("document_id") for a in annotations}
    if len(subjects - {None}) > 1:
        raise ValueError("status assessment requires a single document")
    asserted = {a.get("asserted_status") for a in annotations if a.get("asserted_status")}
    return {"status": "unknown", "as_of": as_of, "current_force_established": False,
            "temporal_uncertainty": True, "conflicting_assertions": len(asserted) > 1,
            "evidence_count": len(annotations),
            "review_items": ["Qualified review of commencement, amendment, repeal and consolidation evidence is required."]}


def build_report(root: Path | str) -> dict:
    root = Path(root).resolve()
    documents_path = root / "documents.jsonl"
    documents = [json.loads(line) for line in documents_path.read_text(encoding="utf-8").splitlines() if line.strip()] if documents_path.exists() else []
    annotations_path = root / "config/legal_annotations.json"
    configuration = json.loads(annotations_path.read_text(encoding="utf-8")) if annotations_path.exists() else {"schema_version": 1, "annotations": []}
    if configuration.get("schema_version") != 1 or not isinstance(configuration.get("annotations"), list):
        raise ValueError("legal annotation configuration invalid")
    verified, proposals = [], []
    unresolved = []
    annotated_documents, identifiers = set(), set()
    for annotation in configuration["annotations"]:
        identifier = annotation.get("annotation_id") if isinstance(annotation, dict) else None
        if not identifier or identifier in identifiers:
            raise ValueError("unique explicit legal annotation IDs required")
        identifiers.add(identifier)
        document_id = annotation.get("evidence", {}).get("document_id")
        annotated_documents.add(document_id)
        try:
            checked = validate_annotation(root, annotation)
            (proposals if checked["statement_scope"] == "draft_proposal" else verified).append(checked)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            unresolved.append({"annotation_id": identifier, "document_id": document_id, "status": "unknown", "reason": str(exc)})
    for document in documents:
        if "amendment" not in document.get("title", "").lower():
            continue
        if document["document_id"] not in annotated_documents:
            unresolved.append({"document_id": document["document_id"], "title": document["title"],
                               "status": "unknown", "reason": "Title is a discovery candidate; original operative provision requires review."})
    return {"schema_version": 2, "audited_at": datetime.now(timezone.utc).isoformat(), "status_policy": "unknown_by_default",
            "annotation_config_sha256": hashlib.sha256(annotations_path.read_bytes()).hexdigest() if annotations_path.exists() else None,
            "verified_archived_relationships": verified, "verified_draft_proposals": proposals, "unresolved_candidates": unresolved,
            "current_in_force_documents": 0, "qualified_legal_review": False,
            "limitations": ["Verified means the cited words occur on the archived page, not a legal opinion about operative effect.",
                            "Relationships never establish current status or a complete consolidation history."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    report = build_report(args.root)
    if args.write_report:
        content = json.dumps(report, indent=2).encode("utf-8")
        immutable_write(args.root / "reports/legal_evidence_history" / f"{hashlib.sha256(content).hexdigest()}.json", content)
        (args.root / "reports/legal_evidence.json").write_bytes(content)
    print(json.dumps({"verified_relationships": len(report["verified_archived_relationships"]),
                      "verified_draft_proposals": len(report["verified_draft_proposals"]),
                      "unresolved_candidates": len(report["unresolved_candidates"]), "current_in_force_documents": 0}))


if __name__ == "__main__":
    main()
