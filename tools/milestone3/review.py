"""Offline citation packets and human-supplied, append-only judgments.

Export: python -m tools.milestone3.review export
Import: python -m tools.milestone3.review import --file path/to/decisions.jsonl
Status: python -m tools.milestone3.review status
Priority status/import: append --packet data/next_milestone/review/six-case-priority
No command creates a human judgment or silently changes benchmark labels.
"""
import argparse
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from functools import lru_cache
import html
import json
from pathlib import Path
import re

from bs4 import BeautifulSoup

from ipindia_corpus.grounded_answers import _unique_object
from ipindia_corpus.indexer import generation
from ipindia_corpus.llm_experiment import ROOT, canonical, timestamp
from ipindia_corpus.locking import file_lock
from ipindia_corpus.paths import archive_path, immutable_write
from tools.reliability_v2 import cases
from .common import DATA, DEST, GENERATION, observations, read, sha, write

JUDGMENTS = {"REQUIRED", "ACCEPTABLE_ALTERNATIVE", "SUPPORTED_BUT_REDUNDANT", "IRRELEVANT",
             "UNSUPPORTED_BY_SOURCE", "INCORRECT_LEGAL_INTERPRETATION", "INSUFFICIENT_INFORMATION",
             "NEEDS_QUALIFIED_LEGAL_REVIEW"}
DIMENSIONS = {"passage_relevance", "proposition_support"}
PRIORITY_CASES = ("traditional_ashwagandha_extract", "new_formulation", "novel_process",
                  "abs_request", "hindi_intake", "unclear_description")
PRIORITY_SOURCE = Path("reports/product_mvp/relevance_review/run-dc7ecb60")
UNRESOLVED = {"INSUFFICIENT_INFORMATION", "NEEDS_QUALIFIED_LEGAL_REVIEW"}


def _pretty(value):
    # The frozen Passport packets use these exact bytes, not canonical().
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _json(raw):
    return json.loads(raw, object_pairs_hook=_unique_object)


def _packet_payloads(directory, manifest, expected):
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(expected):
        raise ValueError("priority_packet_files_mismatch")
    payloads = {}
    for name, digest in files.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("priority_packet_path_escape")
        raw = path.read_bytes()
        if sha(raw) != digest:
            raise ValueError("priority_packet_hash_mismatch")
        payloads[name] = raw
    return payloads


def _priority_context(root, packet):
    """Adapt the sealed six-case preparation packet; never regenerate it."""
    root = Path(root).resolve()
    directory = (root / packet).resolve()
    if not directory.is_relative_to(root):
        raise ValueError("priority_packet_outside_workspace")
    manifest_raw = (directory / "manifest.json").read_bytes()
    manifest = _json(manifest_raw)
    packet_id = manifest.get("packet_id")
    if (manifest.get("schema_version") != "passport-priority-review-v1"
            or manifest.get("ordered_case_ids") != list(PRIORITY_CASES)
            or manifest.get("case_count") != 6 or manifest.get("review_unit_count") != 31
            or manifest.get("human_reviews_completed") != 0
            or packet_id != sha(_pretty({k: v for k, v in manifest.items() if k != "packet_id"}))):
        raise ValueError("priority_packet_manifest_mismatch")
    case_names = {f"cases/{qid}.json" for qid in PRIORITY_CASES}
    payloads = _packet_payloads(directory, manifest, case_names | {"review_queue.jsonl", "instructions.md"})
    source = (root / PRIORITY_SOURCE).resolve()
    if not source.is_relative_to(root):
        raise ValueError("priority_source_path_escape")
    source_raw = (source / "manifest.json").read_bytes()
    if sha(source_raw) != manifest.get("source_packet_manifest_sha256"):
        raise ValueError("priority_original_packet_manifest_mismatch")
    original = _json(source_raw)
    original_payloads = _packet_payloads(source, original, {"selected_run.json", "review_queue.jsonl", "instructions.md"})
    if original["files"] != manifest.get("source_packet_payload_sha256"):
        raise ValueError("priority_original_packet_binding_mismatch")
    report_raw = (root / "reports/product_mvp/verified_demo_cases.json").read_bytes()
    if sha(report_raw) != manifest.get("source_report_sha256") or sha(report_raw) != original.get("input_report_sha256"):
        raise ValueError("priority_source_report_hash_mismatch")
    runs = [run for run in _json(report_raw)["runs"] if run.get("run_id") == manifest.get("selected_run_id")]
    if (len(runs) != 1 or runs[0].get("synthetic") is not True or original_payloads["selected_run.json"] != _pretty(runs[0])
            or original.get("selected_run_id") != manifest.get("selected_run_id")
            or runs[0].get("scenario_file_sha256") != manifest.get("scenario_file_sha256")):
        raise ValueError("priority_source_run_binding_mismatch")
    originals = [_json(line) for line in original_payloads["review_queue.jsonl"].splitlines() if line.strip()]
    source_rows = {row["review_id"]: row for row in originals}
    if len(source_rows) != len(originals) or len(originals) != 50:
        raise ValueError("priority_original_queue_mismatch")
    rows = [_json(line) for line in payloads["review_queue.jsonl"].splitlines() if line.strip()]
    ids = [row.get("review_id") for row in rows]
    if len(rows) != 31 or len(set(ids)) != 31 or ids != manifest.get("preserved_review_ids"):
        raise ValueError("priority_review_ids_mismatch")
    cases = {case["case_id"]: case for case in runs[0]["case_results"]}
    contexts = {}
    for qid in PRIORITY_CASES:
        case = _json(payloads[f"cases/{qid}.json"])
        source_case = cases.get(qid, {})
        response = source_case.get("response", {})
        if (case.get("case_id") != qid or case.get("run_id") != manifest["selected_run_id"]
                or case.get("intake") != source_case.get("intake") or case.get("issues") != response.get("issues")
                or case.get("selected_quotations_and_citations") != response.get("verified_evidence")
                or case.get("generation_id") != response.get("generation_id")
                or case.get("unanswered_questions") != response.get("unresolved_questions")
                or case.get("historical_model_presented_context", {}).get("available") is not False
                or case.get("review_ids") != [row["review_id"] for row in rows if row["case_id"] == qid]):
            raise ValueError("priority_case_binding_mismatch")
        contexts[qid] = case
    queue, texts = {}, {}
    for row in rows:
        rid = row["review_id"]
        if (row != source_rows.get(rid) or rid != sha(_pretty({k: v for k, v in row.items() if k != "review_id"}))
                or row.get("review_status") != "pending" or any(row.get(k) is not None for k in ("decision", "reviewer", "rationale"))
                or row.get("case_id") not in contexts or row.get("packet_id") != original.get("packet_id")):
            raise ValueError("priority_original_review_identity_mismatch")
        evidence = row.get("evidence")
        if not isinstance(evidence, dict):
            raise ValueError("priority_citation_evidence_missing")
        context = contexts[row["case_id"]]
        matches = [c for c in context["archive_contexts"] if c.get("evidence_id") == evidence["evidence_id"]]
        if len(matches) != 1:
            raise ValueError("priority_citation_context_mismatch")
        chunk = matches[0]["archived_chunk"]
        if (chunk.get("chunk_id") != evidence["chunk_id"] or chunk.get("document_id") != evidence["document_id"]
                or chunk.get("raw_sha256") != evidence["raw_sha256"] or chunk.get("source_url") != evidence["official_url"]
                or evidence["exact_quote"] not in chunk["text"]):
            raise ValueError("priority_citation_binding_mismatch")
        passages = [chunk, *matches[0].get("adjacent_chunks", [])]
        for passage in passages:
            if (passage.get("document_id") != evidence["document_id"] or passage.get("raw_sha256") != evidence["raw_sha256"]
                    or sha(passage["text"].encode("utf-8")) != passage.get("content_hash")):
                raise ValueError("priority_displayed_passage_mismatch")
        texts[rid] = [p["text"] for p in passages]
        queue[rid] = {"question_id": row["case_id"], "citation_id": rid, "source_hash": evidence["raw_sha256"],
                      "packet_sha256": manifest["files"][f"cases/{row['case_id']}.json"]}
    return {"queue": queue, "texts": texts, "packet_id": packet_id, "manifest_sha256": sha(manifest_raw),
            "directory": root / "data/next_stage/reviews" / packet_id,
            "distinct_case_citations": len({(r["case_id"], r["evidence"]["document_id"], r["evidence"]["chunk_id"], r["evidence"]["exact_quote"]) for r in rows})}


def norm(text):
    return " ".join(text.split())


class Sources:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.gid, documents, chunks = generation(root, generation_id=GENERATION)
        self.docs = {d["document_id"]: d for d in documents}
        self.chunks = {c["chunk_id"]: c for c in chunks}

    @lru_cache(maxsize=512)
    def original(self, document_id, page):
        doc = self.docs[document_id]
        path = archive_path(self.root, doc["raw_path"])
        raw = path.read_bytes()
        if sha(raw) != doc["sha256"]:
            raise ValueError("source_mutated")
        if raw.startswith(b"%PDF"):
            import fitz
            with fitz.open(stream=raw, filetype="pdf") as pdf:
                if not page or not 1 <= page <= len(pdf):
                    return {"text": None, "reason": "pdf_page_unavailable", "raw_path": doc["raw_path"]}
                text = pdf[page - 1].get_text()
                blocks = [{"bbox": list(b[:4]), "text": b[4]} for b in pdf[page - 1].get_text("blocks")]
            return {"text": norm(text), "extraction": "PyMuPDF_original_page", "blocks": blocks,
                    "raw_path": doc["raw_path"], "page": page}
        soup = BeautifulSoup(raw, "html.parser")
        for node in soup(["script", "style"]):
            node.decompose()
        return {"text": norm(soup.get_text(" ", strip=True)), "raw_path": doc["raw_path"],
                "extraction": "raw_HTML_text_for_review_only_not_browser_visibility"}

    def passage(self, cid):
        chunk = self.chunks[cid]
        doc = self.docs[chunk["document_id"]]
        original = self.original(chunk["document_id"], chunk.get("page_start"))
        text, target = original.get("text"), norm(chunk["text"])
        at = text.find(target) if text else -1
        method = "whitespace_normalized_exact" if at >= 0 else "not_exactly_located"
        if at < 0 and text:
            at = text.find(target[:90])
            if at >= 0:
                method = "prefix_only_not_full_fidelity_proof"
        excerpt = text[max(0, at - 350):at + len(target) + 350] if at >= 0 else text
        return {"chunk": chunk, "document": doc, "original_source_excerpt": excerpt,
                "original_source": {k: v for k, v in original.items() if k != "text"},
                "location_method": method, "original_excerpt_offsets": "whitespace_normalized_source_text",
                "source_hash": doc["sha256"], "legal_relevance_review": "pending"}


def resolved_stages(result):
    from ipindia_corpus.span_selection import build_plan, resolve_selection
    hits = result.get("retrieval_evidence", [])
    omitted = set(result.get("context_omitted_chunk_ids", []))
    packed_ids = result.get("packed_chunk_ids")
    packed = [{**h, "provenance": "verified_immutable_archive"} if h.get("source") != "synthetic_fixture" else h
              for h in hits if h["chunk_id"] not in omitted and (packed_ids is None or h["chunk_id"] in packed_ids)]
    stages = {"retrieved": hits, "packed": packed}
    plan = build_plan(packed)
    for observation in result.get("model_observations", []):
        if observation["stage"] not in {"selection", "verification"}:
            continue
        try:
            stages[observation["stage"]] = resolve_selection(observation["raw"], packed, plan)["expanded_selection"]["claims"]
        except (ValueError, TypeError, KeyError):
            stages[observation["stage"]] = None
    return stages


def render(packet):
    """No executable source HTML, external resources, inline script or raw links."""
    def e(value):
        return html.escape(str(value), quote=True)
    expected = packet["expected_passages"]
    actual = packet["actual_passages"]
    expected_texts = [p["chunk"]["text"] for p in expected]

    def overlap(text):
        intervals = []
        for target in expected_texts:
            intervals += [(b.a, b.a + b.size) for b in SequenceMatcher(None, text, target, autojunk=False).get_matching_blocks() if b.size >= 24]
        merged = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        result, offset = [], 0
        for start, end in merged:
            result += [e(text[offset:start]), "<mark>" + e(text[start:end]) + "</mark>"]
            offset = end
        result.append(e(text[offset:]))
        return "".join(result)

    def column(items):
        parts = []
        for item in items:
            c = item["chunk"]
            quotes = sorted({ci["quote"] for attempt in packet.get("attempts", []) for cl in attempt.get("citations", [])
                             for ci in cl.get("citations", []) if ci["chunk_id"] == c["chunk_id"]})
            emitted = "".join("<pre>" + overlap(q) + "</pre>" for q in quotes)
            parts.append(f'<article><h3>{e(c["chunk_id"])}</h3><p>{e(c["document_title"])}</p>'
                         f'<p>{e(c["source_url"])}</p><p>SHA: {e(item["source_hash"])}</p>'
                         f'<p>section {e(c.get("section"))}; rule {e(c.get("rule"))}; page {e(c.get("page_start"))}</p>'
                         f'<pre>{overlap(c["text"])}</pre><h4>Emitted quotations</h4>{emitted or "<p>None in these disputed attempts.</p>"}'
                         f'<details><summary>Original capture excerpt ({e(item["location_method"])})</summary>'
                         f'<pre>{e(item["original_source_excerpt"])}</pre></details></article>')
        return "".join(parts) or "<p>No passage recorded.</p>"
    comparisons = "".join(f'<li>{e(json.dumps(d, ensure_ascii=False))}</li>' for d in packet["differences"])
    missing = "".join(f'<li>{e(a["milestone"])} / {e(a["mode"])} / attempt {e(a["pass"])}: '
                      f'{e(json.dumps(a.get("missing_expected_groups", []), ensure_ascii=False))}</li>' for a in packet.get("attempts", []) if a.get("missing_expected_groups"))
    return ('<!doctype html><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" '
            'content="default-src \'none\'; style-src \'unsafe-inline\'; base-uri \'none\'; form-action \'none\'">'
            '<title>AyurIP citation review</title><style>body{font:16px system-ui;margin:2rem;max-width:1500px}'
            '.columns{display:grid;grid-template-columns:1fr 1fr;gap:2rem}pre{white-space:pre-wrap;overflow-wrap:anywhere}'
            'article{border:1px solid #aaa;padding:1rem;margin:1rem 0}mark{background:#ffe6a0}'
            '@media(max-width:800px){.columns{display:block}}</style>'
            f'<h1>{e(packet["question_id"])}</h1><p>{e(packet["question"])}</p>'
            '<p>Pending human review. Matching source bytes does not establish relevance or current law.</p>'
            '<p>Yellow marks literal text overlaps of at least 24 characters with frozen expected passages. These are text comparisons, not relevance judgments.</p>'
            f'<h2>Differences requiring review</h2><ul>{comparisons}</ul><h2>Missing frozen target groups</h2><ul>{missing or "<li>None recorded for these attempts.</li>"}</ul><div class="columns">'
            f'<section><h2>Frozen expected passages</h2>{column(expected)}</section>'
            f'<section><h2>Actually cited passages</h2>{column(actual)}</section></div>')


def export(root=ROOT):
    root = Path(root)
    source = Sources(root)
    all_cases = {c["id"]: (split, c) for split, items in cases(root).items() for c in items}
    by_question = defaultdict(list)
    for row in observations(root):
        if "DISALLOWED_CLAIM" in row["system_score"].get("failures", []):
            by_question[row["case_id"]].append(row)
    queue = []
    for qid, rows in sorted(by_question.items()):
        split, case = all_cases[qid]
        groups = case["expected"].get("target_groups", [])
        expected_ids = sorted({cid for g in groups for cid in g["required_chunk_ids"]})
        actual_ids = sorted({ci["chunk_id"] for row in rows for claim in row["result"].get("claims", [])
                             for ci in claim.get("citations", []) if ci["chunk_id"] in source.chunks})
        differences = []
        for cid in actual_ids:
            actual = source.chunks[cid]
            differences.append({"chunk_id": cid, "frozen_allowed": cid in case["expected"].get("allowed_claim_chunk_ids", []),
                                "exact_target_alternative": cid in expected_ids,
                                "text_identical_expected": [x for x in expected_ids if source.chunks[x]["text"] == actual["text"]],
                                "same_document_expected": [x for x in expected_ids if source.chunks[x]["document_id"] == actual["document_id"]],
                                "same_root_expected": [x for x in expected_ids if any(actual.get(k) and actual.get(k) == source.chunks[x].get(k) for k in ("section", "rule"))],
                                "actual_anchor": {k: actual.get(k) for k in ("section", "rule", "page_start", "page_end")},
                                "expected_anchors": {x: {k: source.chunks[x].get(k) for k in ("section", "rule", "page_start", "page_end")} for x in expected_ids},
                                "relevance_judgment": None})
        attempts = []
        for row in rows:
            result = row["result"]
            stages = resolved_stages(result)
            attempts.append({k: row.get(k) for k in ("milestone", "mode", "split", "pass", "at", "candidate_sha256", "report", "report_sha256", "row_sha256", "system_score")}
                            | {"retrieved": stages["retrieved"], "packed_chunk_ids": [h["chunk_id"] for h in stages["packed"]],
                               "selected": stages.get("selection"), "verified_subset": stages.get("verification"),
                               "citations": result.get("claims", []), "model_output": result.get("model_observations", []),
                               "backend_validation": {k: result.get(k) for k in ("status", "reason", "evaluation_citation_audit")},
                               "missing_expected_groups": [g for g in groups if not any(ci["chunk_id"] in g["required_chunk_ids"] and norm(g["required_contains"]).casefold() in norm(ci["quote"]).casefold()
                                for cl in result.get("claims", []) for ci in cl.get("citations", []))]})
        packet = {"schema_version": 1, "question_id": qid, "question": case["question"],
                  "language": case.get("language", "mixed" if re.search(r"[\u0900-\u097f]", case["question"]) else "en"),
                  "split": split, "generation_id": GENERATION, "required_groups": groups,
                  "annotation_status": "pending_human_review", "historical_labels_modified": False,
                  "synthetic_attack": "synthetic" in json.dumps(case).casefold(),
                  "expected_passages": [source.passage(cid) for cid in expected_ids],
                  "actual_passages": [source.passage(cid) for cid in actual_ids], "differences": differences,
                  "retrieved_passages": [source.passage(cid) for cid in sorted({h["chunk_id"] for row in rows for h in row["result"].get("retrieval_evidence", []) if h["chunk_id"] in source.chunks})],
                  "attempts": attempts}
        checksum = write(f"citation_review_packets/{qid}.json", packet, root)
        write(f"citation_review_packets/{qid}.html", render(packet), root)
        citations = sorted({(ci["chunk_id"], ci["quote"]) for row in rows for cl in row["result"].get("claims", []) for ci in cl.get("citations", [])})
        for cid, quote in citations:
            hit = source.chunks.get(cid)
            if not hit:
                continue
            quote_id = sha(canonical({"question_id": qid, "chunk_id": cid, "quote": quote}))
            queue.append({"annotation_version": "milestone3-pending-v1", "question_id": qid,
                          "citation_id": quote_id, "chunk_id": cid, "source_hash": hit["raw_sha256"],
                          "quote": quote, "packet_sha256": checksum,
                          "frozen_label_mismatch": cid not in case["expected"].get("allowed_claim_chunk_ids", []),
                          "packet": f"citation_review_packets/{qid}.json", "dimensions": sorted(DIMENSIONS),
                          "reviewer": None, "review_timestamp": None, "judgment": None,
                          "supporting_quote": None, "rationale": None, "adjudication_status": "pending"})
    write("evidence_review_queue.jsonl", "".join(canonical(q).decode() + "\n" for q in queue), root)
    write("review_instructions.md", INSTRUCTIONS, root)
    return status(root)


def validate(decision, queued):
    required = {"annotation_version", "question_id", "citation_id", "source_hash", "packet_sha256", "reviewer",
                "review_timestamp", "judgment", "supporting_quote", "rationale", "adjudication_status", "dimension", "human_attestation"}
    if not required <= set(decision):
        raise ValueError("review_fields_missing")
    if decision["human_attestation"] is not True or not isinstance(decision["reviewer"], str) or not decision["reviewer"].strip():
        raise ValueError("human_review_attestation_required")
    for key in ("annotation_version", "supporting_quote", "rationale"):
        if not isinstance(decision[key], str) or not decision[key].strip():
            raise ValueError("review_text_required")
    if decision["dimension"] not in DIMENSIONS or decision["judgment"] not in JUDGMENTS:
        raise ValueError("invalid_review_judgment")
    if decision["adjudication_status"] not in {"submitted", "adjudicated"}:
        raise ValueError("invalid_adjudication_status")
    time = datetime.fromisoformat(decision["review_timestamp"].replace("Z", "+00:00"))
    if time.tzinfo is None:
        raise ValueError("review_timestamp_timezone_required")
    if time > datetime.now(time.tzinfo):
        raise ValueError("review_timestamp_in_future")
    if any(decision[key] != queued[key] for key in ("question_id", "citation_id", "source_hash", "packet_sha256")):
        raise ValueError("review_evidence_binding_mismatch")
    # A reviewer must quote displayed original/published text. Substantive
    # support is a human judgment; substring validation does not establish it.
    if decision["judgment"] == "INCORRECT_LEGAL_INTERPRETATION" and decision.get("qualified_legal_reviewer") is not True:
        raise ValueError("qualified_legal_review_required")
    if decision["adjudication_status"] == "adjudicated" and not decision.get("resolves"):
        raise ValueError("adjudication_must_resolve_record_hashes")


def records(root=ROOT, *, directory=None):
    result = []
    base = Path(root) / DATA if directory is None else Path(directory)
    for path in sorted((base / "judgments").glob("*.json")):
        raw = path.read_bytes()
        if sha(raw) != path.stem:
            raise ValueError("review_record_mutated")
        result.append((path.stem, json.loads(raw)))
    return result


def _validate_adjudication(decision, existing):
    if decision["adjudication_status"] != "adjudicated":
        return
    refs = decision["resolves"]
    if not isinstance(refs, list) or any(not isinstance(key, str) for key in refs):
        raise ValueError("adjudication_records_mismatch")
    resolved = [existing.get(key) for key in refs]
    if len(set(refs)) < 2 or any(not r or any(r[k] != decision[k] for k in ("question_id", "citation_id", "dimension")) for r in resolved):
        raise ValueError("adjudication_records_mismatch")
    if any(r["reviewer"].strip().casefold() == decision["reviewer"].strip().casefold() for r in resolved):
        raise ValueError("independent_adjudicator_required")


def _validate_priority(decision, context):
    if not isinstance(decision, dict) or not isinstance(decision.get("citation_id"), str):
        raise ValueError("invalid_review_record")
    queued = context["queue"].get(decision["citation_id"])
    if queued is None:
        raise ValueError("unknown_review_citation")
    if (decision.get("priority_packet_id") != context["packet_id"]
            or decision.get("priority_manifest_sha256") != context["manifest_sha256"]):
        raise ValueError("priority_decision_binding_mismatch")
    try:
        validate(decision, queued)
    except (AttributeError, TypeError) as exc:
        raise ValueError("invalid_review_field_types") from exc
    if decision["dimension"] == "proposition_support" and decision["judgment"] not in UNRESOLVED:
        raise ValueError("proposition_support_not_assessable_missing_proposition")
    if not any(decision["supporting_quote"] in text for text in context["texts"][decision["citation_id"]]):
        raise ValueError("supporting_quote_not_in_reviewed_passage")


def _review_aggregate(decisions):
    groups = defaultdict(list)
    for key, decision in decisions:
        groups[(decision["citation_id"], decision["dimension"])].append((key, decision))
    conflicts, completed, counts = [], set(), defaultdict(int)
    for (cid, dimension), items in groups.items():
        resolved = {key for _, d in items if d["adjudication_status"] == "adjudicated" for key in d["resolves"]}
        active = [(key, d) for key, d in items if key not in resolved]
        judgments = {d["judgment"] for _, d in active}
        if len(judgments) != 1:
            conflicts.append({"citation_id": cid, "dimension": dimension, "records": [key for key, _ in active]})
        elif judgments.isdisjoint(UNRESOLVED):
            completed.add((cid, dimension))
            counts[dimension + ":" + next(iter(judgments))] += 1
    return conflicts, completed, dict(counts)


def _priority_status(context, root):
    decisions = records(root, directory=context["directory"])
    existing = dict(decisions)
    for _, decision in decisions:
        _validate_priority(decision, context)
        _validate_adjudication(decision, existing)
    conflicts, completed, counts = _review_aggregate(decisions)
    return {"schema_version": "passport-priority-human-review-status-v1", "at": timestamp(),
        "priority_packet_id": context["packet_id"], "priority_manifest_sha256": context["manifest_sha256"],
        "question_count": len({q["question_id"] for q in context["queue"].values()}), "review_unit_count": len(context["queue"]),
        "distinct_case_citations": context["distinct_case_citations"], "review_ids": list(context["queue"]),
        "review_identity_fields": list(context["queue"].values()), "human_submitted_records": len(decisions),
        "passage_relevance_completed_units": sum((rid, "passage_relevance") in completed for rid in context["queue"]),
        "fully_reviewed_units_both_dimensions": 0, "judgment_counts_by_dimension": counts, "conflicts": conflicts,
        "dimension_availability": {"passage_relevance": "assessable", "proposition_support": "not_assessable_missing_proposition"},
        "historical_labels_modified": False, "certification_allowed": False,
        "limitation": "Human attestations do not authenticate identity or expertise. Missing historical propositions cannot be reconstructed from research topics or quotations."}


def _import_priority(path, root, packet):
    context = _priority_context(root, packet)
    directory = context["directory"]
    if not directory.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("priority_review_store_path_escape")
    with file_lock(directory / "review-import.lock"):
        context = _priority_context(root, packet)
        raw_batch = Path(path).read_bytes()
        decisions = [_json(line) for line in raw_batch.splitlines() if line.strip()]
        if not decisions or len(decisions) > 5000:
            raise ValueError("invalid_review_batch_size")
        existing = dict(records(root, directory=directory))
        # Check stored attestations/bindings as well as the whole incoming batch.
        _priority_status(context, root)
        for decision in decisions:
            _validate_priority(decision, context)
            _validate_adjudication(decision, existing)
        # Recheck sealed evidence before persisting; never update the packet.
        if _priority_context(root, packet)["manifest_sha256"] != context["manifest_sha256"]:
            raise ValueError("priority_packet_changed_during_import")
        immutable_write(directory / "review_batches" / (sha(raw_batch) + ".jsonl"), raw_batch)
        for decision in decisions:
            raw = canonical(decision)
            immutable_write(directory / "judgments" / (sha(raw) + ".json"), raw)
        return _priority_status(context, root)


def import_decisions(path, root=ROOT, *, packet=None):
    if packet is not None:
        return _import_priority(path, Path(root), packet)
    root = Path(root)
    with file_lock(root / DATA / "review-import.lock"):
        queue = {q["citation_id"]: q for q in (json.loads(line) for line in (root / DEST / "evidence_review_queue.jsonl").read_text(encoding="utf-8").splitlines())}
        decisions = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        if not decisions or len(decisions) > 5000:
            raise ValueError("invalid_review_batch_size")
        existing = dict(records(root))
        for decision in decisions:
            queued = queue.get(decision.get("citation_id"))
            if not queued:
                raise ValueError("unknown_review_citation")
            validate(decision, queued)
            packet = read(root / DEST / queued["packet"])
            if sha(canonical(packet)) != queued["packet_sha256"]:
                raise ValueError("review_packet_mutated")
            texts = [p["chunk"]["text"] for p in packet["actual_passages"] + packet["expected_passages"]]
            texts += [p["original_source_excerpt"] or "" for p in packet["actual_passages"] + packet["expected_passages"]]
            if not any(decision["supporting_quote"] in text for text in texts):
                raise ValueError("supporting_quote_not_in_packet")
            _validate_adjudication(decision, existing)
        # Validate the whole batch before persisting any decision.
        raw = Path(path).read_bytes()
        immutable_write(root / DATA / "review_batches" / (sha(raw) + ".jsonl"), raw)
        for decision in decisions:
            raw = canonical(decision)
            immutable_write(root / DATA / "judgments" / (sha(raw) + ".json"), raw)
        return status(root)


def status(root=ROOT, *, packet=None):
    if packet is not None:
        context = _priority_context(root, packet)
        if not context["directory"].resolve().is_relative_to(Path(root).resolve()):
            raise ValueError("priority_review_store_path_escape")
        return _priority_status(context, root)
    root = Path(root)
    queue = [json.loads(line) for line in (root / DEST / "evidence_review_queue.jsonl").read_text(encoding="utf-8").splitlines() if line]
    decisions = records(root)
    conflicts, completed, counts = _review_aggregate(decisions)
    full = sum(all((q["citation_id"], dim) in completed for dim in DIMENSIONS) for q in queue)
    result = {"at": timestamp(), "queued_question_count": len({q["question_id"] for q in queue}),
              "queued_question_citation_quotes": len(queue), "human_submitted_records": len(decisions),
              "frozen_label_mismatch_citation_quotes": sum(q.get("frozen_label_mismatch", False) for q in queue),
              "human_reviewed_citations_both_dimensions": full, "judgment_counts_by_dimension": dict(counts),
              "unresolved_citations": len(queue) - full, "conflicts": conflicts,
              "historical_labels_modified": False, "certification_allowed": False,
              "limitation": "Imports preserve human attestations; the tool cannot authenticate reviewer identity or certify legal expertise."}
    write("review_status.json", result, root)
    lines = ["# Annotation disagreements", "", f"{len(queue)} citation/quote pairs across {result['queued_question_count']} questions await two-dimensional review.",
             f"Completed human reviews: {full}. Unresolved conflicts: {len(conflicts)}.", "",
             "A frozen-label mismatch remains an unresolved relevance question until human review. Source fidelity does not resolve it.",
             "No historical score or label is changed. archived-20 retains its narrow patent-term target.",
             "Any future reviewed annotation version is separate from historical benchmarks."]
    write("annotation_disagreement_report.md", "\n".join(lines) + "\n", root)
    return result


INSTRUCTIONS = """# Human citation review handoff

Open citation_review_packets/<question_id>.html locally; the adjacent JSON holds
all retained attempts, raw model responses, backend results and source metadata.
Left: frozen expected passages. Right: actually cited passages. Inspect the raw
capture at the recorded workspace-relative path and hash, not just rendered text.
Page blocks preserve available original PDF bounding boxes; absent anchors stay absent.

For EACH citation assess BOTH passage_relevance and proposition_support separately.
Use REQUIRED, ACCEPTABLE_ALTERNATIVE, SUPPORTED_BUT_REDUNDANT, IRRELEVANT,
UNSUPPORTED_BY_SOURCE, INCORRECT_LEGAL_INTERPRETATION, INSUFFICIENT_INFORMATION,
or NEEDS_QUALIFIED_LEGAL_REVIEW. Record qualifications and disagreements in rationale.
Same section, identical wording, and authentic source bytes alone do not establish
that a quotation answers this question. Preserve the narrow archived-20 label.

Copy queue identity fields into a new JSONL file; provide annotation_version,
question_id, citation_id, source_hash, packet_sha256, reviewer (real identity),
review_timestamp (ISO timestamp with timezone), dimension, judgment,
supporting_quote (exact packet text), rationale, human_attestation=true,
adjudication_status=submitted. Never submit AI decisions as human judgments.
Legal-interpretation findings require qualified_legal_reviewer=true from a
qualified human. This flag is an attestation, not automatic credential verification.

Import with: python -m tools.milestone3.review import --file <decisions.jsonl>
Status with: python -m tools.milestone3.review status
Conflicting judgments remain unresolved. A different human adjudicator submits a
new record with adjudication_status=adjudicated and resolves=[record SHA, ...].
The original records remain immutable. No latest-write-wins adjudication exists.

These records form a separate content-addressed annotation version; they never
overwrite frozen labels. Full independent benchmark review is an additional gate.
No reviewed_gold_v3.jsonl is created until genuine benchmark review is completed.
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("export", "import", "status"))
    parser.add_argument("--file", type=Path)
    parser.add_argument("--packet", type=Path, help="Sealed six-case Passport packet; import/status only")
    args = parser.parse_args(argv)
    if args.command == "import" and not args.file:
        parser.error("--file required")
    if args.packet and args.command == "export":
        parser.error("--packet cannot regenerate a sealed packet")
    result = export() if args.command == "export" else import_decisions(args.file, packet=args.packet) if args.command == "import" else status(packet=args.packet)
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
