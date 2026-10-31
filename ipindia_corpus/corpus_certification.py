"""Offline, conservative admission audit of staged source material.

No network requests, source-policy changes, or searchable publication occur here.
Hashes identify observed bytes; they do not establish official provenance.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

from .paths import immutable_write
from .sync import public_tkdl_destination

OFFICIAL_HOSTS = {
    "indiacode": {"indiacode.gov.in", "www.indiacode.gov.in", "indiacode.nic.in", "www.indiacode.nic.in", "upload.indiacode.nic.in"},
    "tkdl": {"tkdl.res.in", "www.tkdl.res.in"},
    "ipindia": {"ipindia.gov.in", "www.ipindia.gov.in", "ipindia.nic.in", "www.ipindia.nic.in"},
    "nba": {"nbaindia.nic.in", "www.nbaindia.nic.in"},
    "gazette": {"egazette.gov.in", "www.egazette.gov.in"},
}
SOURCE_URL = re.compile(r"(?:_Source:|# Source:)\s*(https?://[^\s<>]+)")
SHORT_NAV = {"image.md", "image1.md", "gallery-viewer.md", "gallery-viewer-aloe.md"}


def official_public_url(source: str, url: str) -> bool:
    """Reject ambiguous authorities and restricted TKDL destinations without fetching."""
    try:
        parsed = urlsplit(url)
        return bool(
            parsed.scheme == "https" and not parsed.username and not parsed.password
            and parsed.port in (None, 443) and parsed.hostname in OFFICIAL_HOSTS.get(source, set())
            and (source != "tkdl" or public_tkdl_destination(url))
            and "\\" not in url
        )
    except ValueError:
        return False


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _receipt(root: Path, path: Path, source: str, url: str) -> dict:
    """A sidecar can establish archived-original identity, never fidelity by assertion."""
    receipt_path = path.with_name(path.name + ".receipt.json")
    result = {"available": receipt_path.exists(), "verified": False, "capture_timestamp": None}
    if not receipt_path.exists():
        return result
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        original = (root / receipt["raw_path"]).resolve()
        if not original.is_relative_to((root / "data/raw").resolve()):
            raise ValueError("original outside immutable archive")
        stamp = datetime.fromisoformat(receipt["retrieved_at"].replace("Z", "+00:00"))
        if stamp.tzinfo is None or not official_public_url(source, receipt["url"]) or receipt["url"] != url:
            raise ValueError("invalid source identity or timestamp")
        if _hash(original.read_bytes()) != receipt["sha256"]:
            raise ValueError("original hash mismatch")
        result.update(verified=True, capture_timestamp=receipt["retrieved_at"],
                      original_path=original.relative_to(root).as_posix(), original_sha256=receipt["sha256"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result["error"] = str(exc)
    return result


def _record(root: Path, path: Path, source: str, metadata: dict | None = None) -> dict:
    data = path.read_bytes()
    text = data.decode("utf-8", errors="replace")
    match = SOURCE_URL.search(text)
    url = (metadata or {}).get("url") or (match.group(1).rstrip("_") if match else "")
    flags = []
    if not official_public_url(source, url):
        flags.append("missing_or_disallowed_official_url")
    if "\ufffd" in text or any(token in text for token in ("Ã", "Â", "Ľ", "˘")):
        flags.append("encoding_review")
    if re.search(r"session\s*(?:expired|timeout)", text, re.I):
        # Many archived pages append a session-timeout label to otherwise long
        # substantive content. Flag the marker without misreporting every page
        # as a captured error response.
        flags.append("session_timeout_marker")
        if len(text.strip()) < 600:
            flags.append("error_or_session_page")
    if re.search(r"access denied|403 forbidden|internal server error", text, re.I):
        flags.append("error_or_session_page")
    if re.search(r"ignore (?:all |previous |your )*instructions|system\s*prompt|<script\b", text, re.I):
        flags.append("untrusted_instruction_or_active_markup")
    if re.search(r"\[truncated\]|\.\.\.\s*truncated|content truncated", text, re.I):
        flags.append("explicit_truncation")
    if path.name in SHORT_NAV or len(text.strip()) < 200:
        flags.append("navigation_or_sparse_content")
    expected = re.search(r"_Total sections:\s*(\d+)_", text)
    sections = re.findall(r"^## Section\s+(\d+[A-Z]*)\s*[—–:-]", text, re.M)
    if expected and int(expected.group(1)) != len(sections):
        flags.append("section_count_mismatch")
    receipt = _receipt(root, path, source, url)
    reasons = ["original_to_passage_fidelity_unverified"]
    if not receipt["verified"]:
        reasons.append("original_capture_receipt_missing_or_invalid")
    if not official_public_url(source, url):
        reasons.append("official_public_identity_unverified")
    if any(flag in flags for flag in ("error_or_session_page", "explicit_truncation", "navigation_or_sparse_content")):
        reasons.append("content_quality_review_required")
    return {"path": path.relative_to(root).as_posix(), "sha256": _hash(data), "bytes": len(data),
            "source": source, "source_url": url or None, "official_url_syntax": official_public_url(source, url),
            "capture_timestamp": receipt["capture_timestamp"], "original_receipt": receipt,
            "document_status": "unknown", "evidence_role": "institutional_background" if source == "tkdl" else "legal_instrument_unverified",
            "observed_sections": len(sections), "declared_sections": int(expected.group(1)) if expected else None,
            "flags": flags, "certified": False, "search_admission": "blocked", "reasons": reasons}


def audit(root: Path | str) -> dict:
    """Inspect staged files in root, preserving every input and reporting non-admission."""
    root = Path(root).resolve()
    documents = []
    for path in sorted((root / "section_output/output").glob("*.md")):
        documents.append(_record(root, path, "indiacode"))
    staged = root / "tkdl_corpus_final/tkdl_corpus"
    metadata = {}
    metadata_rows = 0
    metadata_files = []
    metadata_issues = []
    for path in sorted(staged.rglob("metadata.json")):
        raw = path.read_bytes()
        metadata_files.append({"path": path.relative_to(root).as_posix(), "sha256": _hash(raw), "bytes": len(raw)})
        try:
            rows = json.loads(raw.decode("utf-8"))
            if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                raise ValueError("metadata must be a list of records")
        except (UnicodeDecodeError, ValueError) as exc:
            metadata_issues.append({"path": path.relative_to(root).as_posix(), "error": str(exc)})
            continue
        metadata_rows += len(rows)
        for row in rows:
            metadata[row.get("file", "").replace("\\", "/").lower()] = row
    for path in sorted(staged.rglob("*.md")):
        documents.append(_record(root, path, "tkdl", metadata.get(path.relative_to(staged).as_posix().lower())))
    duplicate_groups = {}
    for document in documents:
        duplicate_groups.setdefault(document["sha256"], []).append(document["path"])
    duplicates = [paths for paths in duplicate_groups.values() if len(paths) > 1]
    for document in documents:
        if len(duplicate_groups[document["sha256"]]) > 1:
            document["flags"].append("byte_identical_duplicate")
    policies_path = root / "config/source_policies.json"
    policies = json.loads(policies_path.read_text(encoding="utf-8")) if policies_path.exists() else {}
    sources = {}
    for source in ("indiacode", "tkdl"):
        entries = [d for d in documents if d["source"] == source]
        sources[source] = {"staged_documents": len(entries), "certified_documents": 0,
            "provenance_verified_documents": sum(d["original_receipt"]["verified"] for d in entries),
            "official_url_syntax_documents": sum(d["official_url_syntax"] for d in entries),
            "status": "blocked", "search_admission": "blocked", "access_approved": policies.get(source, {}).get("approved", False),
            "flags": dict(Counter(flag for d in entries for flag in d["flags"])),
            "actions": ["Preserve original official HTML/PDF and capture receipt when authorized and accessible.",
                        "Validate extraction against originals and bind section/page anchors before admission."]}
    originals = []
    for path in sorted(root.glob("*.pdf")):
        if path.name == "ayurveda-ipr-mvp-architecture.pdf":
            continue
        raw = path.read_bytes()
        record = {"path": path.name, "sha256": _hash(raw), "bytes": len(raw), "certified": False,
                  "document_status": "unknown", "search_admission": "blocked", "reasons": ["official_capture_provenance_unverified"]}
        try:
            import fitz
            with fitz.open(stream=raw, filetype="pdf") as pdf:
                record["pages"] = len(pdf)
                record["readable"] = not pdf.is_encrypted
        except Exception as exc:
            record["readable"] = False
            record["parse_error"] = type(exc).__name__
        originals.append(record)
    inventory = [{"path": d["path"], "sha256": d["sha256"], "bytes": d["bytes"]} for d in documents + originals] + metadata_files
    inventory_sha256 = _hash(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode())
    return {"schema_version": 1, "audited_at": datetime.now(timezone.utc).isoformat(),
            "mode": "offline_staged_audit", "inventory_sha256": inventory_sha256, "sources": sources,
            "tkdl_metadata_rows": metadata_rows, "metadata_files": metadata_files, "metadata_issues": metadata_issues,
            "documents": documents, "user_supplied_pdf_originals": originals,
            "duplicate_groups": duplicates, "three_source_ready": False,
            "limitations": ["Official host syntax is not proof of origin or current legal status.",
                            "Transformed Markdown cannot establish original-byte or page fidelity.",
                            "A successful PDF parse is not official provenance or evidence that a law is in force."]}


def write_report(root: Path | str, report: dict) -> None:
    root = Path(root).resolve()
    content = json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8")
    immutable_write(root / "reports/certification_history" / f"{_hash(content)}.json", content)
    (root / "reports/corpus_certification.json").write_bytes(content)
    lines = ["# Offline staged corpus certification", "", "No fresh capture or publication was performed.", "",
             "| Source | Staged files | Certified | Admission |", "|---|---:|---:|---|"]
    for name, entry in report["sources"].items():
        lines.append(f"| {name} | {entry['staged_documents']} | {entry['certified_documents']} | blocked |")
    lines += ["", f"Inventory SHA-256: `{report['inventory_sha256']}`.", "",
              f"Also retained {len(report['user_supplied_pdf_originals'])} user-supplied PDFs; none has verified official capture provenance.", "",
              "Original HTML/PDF capture receipts and original-to-passage checks are missing. Official URL strings alone cannot certify these transformed files.", "",
              "TKDL is institutional background, not evidence of a binding legal rule. All legal statuses remain unknown. Per-file hashes, quality flags, section counts and remediation appear in the JSON report."]
    (root / "reports/corpus_certification.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    report = audit(args.root)
    if args.write_report:
        write_report(args.root, report)
    print(json.dumps({key: report[key] for key in ("sources", "inventory_sha256", "three_source_ready")}, indent=2))


if __name__ == "__main__":
    main()
