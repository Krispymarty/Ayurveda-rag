"""Read-only source audit. Writes reports, never changes source datasets."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPORTS = ROOT / "reports"
CORPUS = ROOT / "corpus"
TKDL_ZIP = ROOT / "tkdl_corpus_final.zip"

REQUIRED_DOCUMENT = (
    "document_id", "title", "authority", "jurisdiction", "source_url",
    "document_type", "legal_domain", "language", "retrieved_at", "content_hash",
)
VERSION_FIELDS = (
    "publication_date", "effective_from", "effective_to", "version",
    "notification_number", "amends", "amended_by", "supersedes", "superseded_by", "status",
)
INJECTION = re.compile(r"ignore (?:all |any )?(?:previous|prior) instructions|system prompt|you are chatgpt|(?:^|\n)\s*(?:SYSTEM|assistant)\s*:\s*(?:ignore|override|follow|disregard)\b", re.I)
ACTIVE = re.compile(r"<script\b|<iframe\b|javascript:|\bon(?:click|load|error|submit|mouseover|focus)\s*=", re.I)
ENCODING = re.compile(r"\ufffd|â€™|â€œ|â€\x9d|â€“|Ã[\x80-\xbf]|\x00|[Ľ˘]")
QUESTIONS = {
    "traditional_knowledge_patentability": re.compile(r"traditional knowledge|section\s*3\s*\(?p\)?", re.I),
    "biological_resource_rules": re.compile(r"biological (?:material|resource)|biodiversity|genetic resource", re.I),
    "legal_provision_source": re.compile(r"section\s*3|patents act|patent rules", re.I),
}


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalized_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def write_csv(name: str, fieldnames: list[str], records: list[dict]) -> None:
    with (REPORTS / name).open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def audit_tkdl() -> tuple[dict, list[dict], dict[str, list[dict]]]:
    metadata_gaps: list[dict] = []
    version_gaps: list[dict] = []
    encoding_issues: list[dict] = []
    security_flags: list[dict] = []
    citation: list[dict] = []
    duplicate: list[dict] = []
    file_records = []
    with ZipFile(TKDL_ZIP) as archive:
        names = set(archive.namelist())
        lookup = {name.casefold(): name for name in names}
        metadata = []
        for name in sorted(n for n in names if n.endswith("/metadata.json")):
            metadata.extend(json.loads(archive.read(name).decode("utf-8-sig")))
        for index, item in enumerate(metadata):
            url = item.get("url", "")
            key = f"tkdl_{index:03d}"
            expected_path = "tkdl_corpus/" + item.get("file", "").replace("\\", "/")
            actual_path = expected_path if expected_path in names else lookup.get(expected_path.casefold())
            if actual_path != expected_path:
                metadata_gaps.append({"source": "tkdl", "document_id": key, "field": "file", "issue": "case_mismatch" if actual_path else "missing_file", "detail": expected_path})
            content = archive.read(actual_path).decode("utf-8-sig", errors="replace") if actual_path else ""
            source_header = re.search(r"^# Source:\s*(\S+)", content)
            if source_header and source_header.group(1) != url:
                metadata_gaps.append({"source": "tkdl", "document_id": key, "field": "source_url", "issue": "header_mismatch", "detail": source_header.group(1)})
            if urlsplit(url).hostname != "tkdl.res.in":
                metadata_gaps.append({"source": "tkdl", "document_id": key, "field": "source_domain", "issue": "unofficial_or_unknown", "detail": url})
            for field in REQUIRED_DOCUMENT:
                present = field == "title" and bool(item.get("title")) or field == "source_url" and bool(url)
                if not present:
                    metadata_gaps.append({"source": "tkdl", "document_id": key, "field": field, "issue": "missing", "detail": ""})
            version_gaps.append({"source": "tkdl", "document_id": key, "missing_fields": ",".join(VERSION_FIELDS), "metadata_review_required": True})
            if ENCODING.search(content):
                for match in list(ENCODING.finditer(content))[:5]:
                    encoding_issues.append({"source": "tkdl", "document_id": key, "location": actual_path or expected_path, "pattern": repr(match.group()), "example": content[max(0, match.start()-45):match.end()+45].replace("\n", " ")})
            for label, pattern in (("prompt_like_text", INJECTION), ("active_markup", ACTIVE)):
                for match in list(pattern.finditer(content))[:5]:
                    security_flags.append({"source": "tkdl", "document_id": key, "flag": label, "location": actual_path or expected_path, "example": content[max(0, match.start()-45):match.end()+45].replace("\n", " ")})
            citation.append({"source": "tkdl", "document_id": key, "chunk_id": "", "ready": False, "reason": "no_structured_chunks_or_retrieval_timestamp", "actual_url": url, "citation_url": source_header.group(1) if source_header else ""})
            file_records.append({"document_id": key, "url": url, "title": item.get("title", ""), "path": actual_path or "", "content": content, "sha256": sha256(content.encode("utf-8"))})

    by_hash: dict[str, list[dict]] = defaultdict(list)
    by_title: dict[str, list[dict]] = defaultdict(list)
    for item in file_records:
        by_hash[item["sha256"]].append(item)
        by_title[normalized_title(item["title"])].append(item)
    for matches in by_hash.values():
        if len(matches) > 1:
            duplicate.append({"source": "tkdl", "type": "identical_text", "document_ids": ";".join(x["document_id"] for x in matches), "urls": ";".join(x["url"] for x in matches), "detail": matches[0]["sha256"]})
    for matches in by_title.values():
        if len(matches) > 1 and len({x["sha256"] for x in matches}) > 1:
            duplicate.append({"source": "tkdl", "type": "same_normalized_title_different_content", "document_ids": ";".join(x["document_id"] for x in matches), "urls": ";".join(x["url"] for x in matches), "detail": "possible versions or page variants; review"})
    retrieval = {name: sum(bool(pattern.search(item["content"])) for item in file_records) for name, pattern in QUESTIONS.items()}
    summary = {
        "source": "tkdl", "documents": len(file_records), "chunks": 0,
        "source_domain_valid": sum(urlsplit(x["url"]).hostname == "tkdl.res.in" for x in file_records),
        "metadata_completeness_percent": round(100 * sum(2 for _ in file_records) / (len(REQUIRED_DOCUMENT) * len(file_records)), 1) if file_records else 0,
        "citation_ready_percent": 0.0, "encoding_issue_instances": len(encoding_issues),
        "duplicate_groups": len(duplicate), "security_flag_instances": len(security_flags),
        "fidelity_sample_score": None, "fidelity_limit": "No original HTML archived; transformed Markdown cannot be verified against source bytes offline.",
        "retrieval_term_hit_documents": retrieval,
        "verdict": "REQUIRES REPROCESSING",
    }
    return summary, file_records, {"metadata_gaps": metadata_gaps, "version_gaps": version_gaps, "encoding_issues": encoding_issues, "security_flags": security_flags, "citation": citation, "duplicate": duplicate}


def audit_ipindia() -> tuple[dict, list[dict], dict[str, list[dict]]]:
    documents = rows(CORPUS / "documents.jsonl")
    chunks = rows(CORPUS / "chunks.jsonl")
    chunks_by_doc: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_doc[chunk.get("document_id", "")].append(chunk)
    metadata_gaps: list[dict] = []
    version_gaps: list[dict] = []
    encoding_issues: list[dict] = []
    security_flags: list[dict] = []
    citation: list[dict] = []
    duplicate: list[dict] = []
    chunk_quality: list[dict] = []
    valid_domain = 0
    raw_hash_matches = 0
    version_missing = Counter()
    sample_candidates: dict[tuple[str, str], tuple[dict, list[dict]]] = {}
    for document in documents:
        doc_id = document.get("document_id", "")
        actual_url = document.get("download_url") or document.get("source_url", "")
        if urlsplit(actual_url).hostname in {"ipindia.gov.in", "www.ipindia.gov.in"}:
            valid_domain += 1
        else:
            metadata_gaps.append({"source": "ipindia", "document_id": doc_id, "field": "source_domain", "issue": "unofficial_or_unknown", "detail": actual_url})
        raw_path = Path(document.get("raw_file_path", ""))
        if raw_path.is_file() and raw_path.resolve().is_relative_to((CORPUS / "cache").resolve()):
            if sha256(raw_path.read_bytes()) == document.get("content_hash"):
                raw_hash_matches += 1
            else:
                metadata_gaps.append({"source": "ipindia", "document_id": doc_id, "field": "content_hash", "issue": "raw_hash_mismatch", "detail": str(raw_path)})
        else:
            metadata_gaps.append({"source": "ipindia", "document_id": doc_id, "field": "raw_file_path", "issue": "missing_or_outside_cache", "detail": str(raw_path)})
        for field in REQUIRED_DOCUMENT:
            mapped = "ip_domain" if field == "legal_domain" else "content_hash" if field == "content_hash" else field
            if mapped == "source_url":
                value = actual_url
            else:
                value = document.get(mapped)
            if value in (None, "", []):
                metadata_gaps.append({"source": "ipindia", "document_id": doc_id, "field": field, "issue": "missing", "detail": ""})
        missing_version = [field for field in VERSION_FIELDS if document.get(field) in (None, "", [])]
        version_missing.update(missing_version)
        version_gaps.append({"source": "ipindia", "document_id": doc_id, "missing_fields": ",".join(missing_version), "metadata_review_required": bool(missing_version)})
        local_chunks = chunks_by_doc[doc_id]
        sample_key = (document.get("ip_domain", ""), document.get("document_type", ""))
        if local_chunks and sample_key not in sample_candidates:
            sample_candidates[sample_key] = (document, local_chunks)
        lengths = [len(c.get("text", "")) for c in local_chunks]
        chunk_quality.append({
            "source": "ipindia", "document_id": doc_id, "chunks": len(lengths),
            "min_chars": min(lengths) if lengths else 0, "mean_chars": round(statistics.mean(lengths), 1) if lengths else 0,
            "max_chars": max(lengths) if lengths else 0, "empty": sum(x == 0 for x in lengths),
            "short_under_80": sum(x < 80 for x in lengths), "long_over_6000": sum(x > 6000 for x in lengths),
            "section_or_rule_labeled": sum(bool(c.get("section") or c.get("rule")) for c in local_chunks),
        })
        for chunk in local_chunks:
            chunk_id = chunk.get("chunk_id", "")
            text = chunk.get("text", "")
            source_url = chunk.get("source_url", "")
            ready = bool(chunk_id and chunk.get("document_title") and chunk.get("authority") and chunk.get("jurisdiction") and source_url == actual_url and text and chunk.get("document_id") == doc_id)
            reasons = []
            if source_url != actual_url:
                reasons.append("citation_url_is_parent_or_alias")
            if not chunk.get("section") and not chunk.get("rule") and document.get("document_type") in {"Act", "Rules"}:
                reasons.append("legal_unit_missing")
            if len(text) < 80:
                reasons.append("very_short")
            citation.append({"source": "ipindia", "document_id": doc_id, "chunk_id": chunk_id, "ready": ready, "reason": ";".join(reasons), "actual_url": actual_url, "citation_url": source_url})
            for match in list(ENCODING.finditer(text))[:3]:
                encoding_issues.append({"source": "ipindia", "document_id": doc_id, "location": chunk_id, "pattern": repr(match.group()), "example": text[max(0, match.start()-45):match.end()+45].replace("\n", " ")})
            for label, pattern in (("prompt_like_text", INJECTION), ("active_markup", ACTIVE)):
                for match in list(pattern.finditer(text))[:2]:
                    security_flags.append({"source": "ipindia", "document_id": doc_id, "flag": label, "location": chunk_id, "example": text[max(0, match.start()-45):match.end()+45].replace("\n", " ")})

    by_hash: dict[str, list[dict]] = defaultdict(list)
    by_title: dict[str, list[dict]] = defaultdict(list)
    for document in documents:
        by_hash[document.get("content_hash", "")].append(document)
        by_title[normalized_title(document.get("title", ""))].append(document)
    for matches in by_hash.values():
        if len(matches) > 1:
            duplicate.append({"source": "ipindia", "type": "identical_bytes_in_export", "document_ids": ";".join(x["document_id"] for x in matches), "urls": ";".join(x.get("download_url") or x.get("source_url", "") for x in matches), "detail": matches[0].get("content_hash", "")})
    for title, matches in by_title.items():
        if title and len(matches) > 1 and len({x.get("content_hash") for x in matches}) > 1:
            duplicate.append({"source": "ipindia", "type": "same_normalized_title_different_content", "document_ids": ";".join(x["document_id"] for x in matches), "urls": ";".join(x.get("download_url") or x.get("source_url", "") for x in matches), "detail": "possible distinct sections or legal versions; do not deduplicate automatically"})
    chunk_hashes = Counter(sha256(normalized_text(c.get("text", "")).encode()) for c in chunks if c.get("text", "").strip())
    for value, count in chunk_hashes.items():
        if count > 1:
            duplicate.append({"source": "ipindia", "type": "duplicate_chunk_text", "document_ids": "", "urls": "", "detail": f"sha256={value}; count={count}"})

    fidelity = []
    try:
        from ipindia_corpus.pipeline import Fetch, extract
        for document, local_chunks in list(sample_candidates.values())[:16]:
            raw = Path(document["raw_file_path"])
            meta = json.loads(raw.with_suffix(".json").read_text(encoding="utf-8"))
            fetch = Fetch(document.get("download_url") or document["source_url"], raw.read_bytes(), meta["content_type"], meta["retrieved_at"], raw)
            _, extracted, _ = extract(fetch)
            reference_lines = [line for line in extracted.splitlines() if not re.fullmatch(r"\s*(?:\[Page \d+\]|\d{1,4})\s*", line)]
            reference = normalized_text("\n".join(reference_lines)).casefold()
            checked = local_chunks[: min(30, len(local_chunks))]
            matches = sum(normalized_text(chunk["text"]).casefold() in reference for chunk in checked)
            fidelity.append({"document_id": document["document_id"], "checked": len(checked), "matched": matches, "score": round(matches / len(checked), 3) if checked else None})
    except (OSError, ValueError, ImportError) as exc:
        fidelity.append({"error": str(exc)})

    completeness = 1 - len([x for x in metadata_gaps if x["issue"] == "missing"]) / (len(REQUIRED_DOCUMENT) * len(documents)) if documents else 0
    citation_ready = sum(x["ready"] for x in citation)
    lengths = [len(c.get("text", "")) for c in chunks]
    retrieval = {name: sum(bool(pattern.search(c.get("text", ""))) for c in chunks) for name, pattern in QUESTIONS.items()}
    summary = {
        "source": "ipindia", "documents": len(documents), "chunks": len(chunks),
        "source_domain_valid": valid_domain, "raw_hash_matches": raw_hash_matches,
        "metadata_completeness_percent": round(100 * completeness, 1),
        "citation_ready_percent": round(100 * citation_ready / len(citation), 1) if citation else 0,
        "citation_url_mismatches": sum(x["actual_url"] != x["citation_url"] for x in citation),
        "encoding_issue_instances": len(encoding_issues), "duplicate_groups": len(duplicate),
        "duplicate_chunk_groups": sum(x["type"] == "duplicate_chunk_text" for x in duplicate),
        "security_flag_instances": len(security_flags),
        "fidelity_sample_score": round(sum(x.get("matched", 0) for x in fidelity) / sum(x.get("checked", 0) for x in fidelity), 3) if sum(x.get("checked", 0) for x in fidelity) else None,
        "fidelity_samples": fidelity,
        "chunk_lengths": {"min": min(lengths) if lengths else 0, "mean": round(statistics.mean(lengths), 1) if lengths else 0, "max": max(lengths) if lengths else 0, "under_80": sum(x < 80 for x in lengths), "over_6000": sum(x > 6000 for x in lengths)},
        "version_field_missing_counts": dict(version_missing),
        "retrieval_term_hit_chunks": retrieval,
        "verdict": "REQUIRES REPROCESSING" if citation_ready < len(citation) else "PASS WITH FIXES",
    }
    return summary, documents, {"metadata_gaps": metadata_gaps, "version_gaps": version_gaps, "encoding_issues": encoding_issues, "security_flags": security_flags, "citation": citation, "duplicate": duplicate, "chunk_quality": chunk_quality}


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    tkdl, _, tk_rows = audit_tkdl()
    ipindia, _, ip_rows = audit_ipindia()
    combined = {"audited_at": "2026-10-14", "datasets": {"tkdl": tkdl, "ipindia": ipindia}, "indiacode": {"present": False, "reason": "No India Code dataset found in workspace"}}
    (REPORTS / "dataset_validation.json").write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
    specs = [
        ("duplicate_report.csv", ["source", "type", "document_ids", "urls", "detail"], "duplicate"),
        ("encoding_issues.csv", ["source", "document_id", "location", "pattern", "example"], "encoding_issues"),
        ("metadata_gaps.csv", ["source", "document_id", "field", "issue", "detail"], "metadata_gaps"),
        ("versioning_gaps.csv", ["source", "document_id", "missing_fields", "metadata_review_required"], "version_gaps"),
        ("citation_readiness.csv", ["source", "document_id", "chunk_id", "ready", "reason", "actual_url", "citation_url"], "citation"),
        ("security_content_flags.csv", ["source", "document_id", "flag", "location", "example"], "security_flags"),
    ]
    for name, fields, key in specs:
        write_csv(name, fields, tk_rows.get(key, []) + ip_rows.get(key, []))
    write_csv("chunk_quality.csv", ["source", "document_id", "chunks", "min_chars", "mean_chars", "max_chars", "empty", "short_under_80", "long_over_6000", "section_or_rule_labeled"], [{"source": "tkdl", "document_id": "ALL", "chunks": 0, "min_chars": 0, "mean_chars": 0, "max_chars": 0, "empty": 0, "short_under_80": 0, "long_over_6000": 0, "section_or_rule_labeled": 0}] + ip_rows["chunk_quality"])
    print(json.dumps({"tkdl": {k: tkdl[k] for k in ("documents", "chunks", "verdict")}, "ipindia": {k: ipindia[k] for k in ("documents", "chunks", "citation_ready_percent", "verdict")}}, indent=2))


if __name__ == "__main__":
    main()
