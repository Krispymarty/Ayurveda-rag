"""Manual, serial functional observations for the private passport.

Run from the repository with: python -m tools.passport_demo
This is an opt-in live run, not a benchmark or a legal correctness evaluation.
Only loopback HTTP is allowed. No retries, services, or external fetches occur.
The sole report keeps previous runs, failures, exact input snapshots and raw
responses. Importing this module does not run the demo or write anything.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from http.cookiejar import CookieJar
import json
import os
from pathlib import Path
import re
import time
import uuid
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import (
    HTTPRedirectHandler, HTTPCookieProcessor, ProxyHandler, Request, build_opener,
)

ROOT = Path(__file__).resolve().parents[1]
HARNESS_VERSION = "passport-demo-harness-v2"
SCENARIOS = ROOT / "tests" / "passport_scenarios.json"
REPORT = ROOT / "reports" / "product_mvp" / "verified_demo_cases.json"
STATUSES = {"ready", "partial", "blocked", "unavailable", "needs_clarification"}
NOVELTIES = {
    "new_formulation", "new_process", "modified_formulation",
    "delivery_mechanism", "unsure",
}
INTAKE_FIELDS = {
    "description", "novelty", "goal", "jurisdiction", "ingredients",
    "traditional_use", "public_disclosure", "additional_context",
    "acknowledged_nonconfidential",
}
EVIDENCE_FIELDS = {
    "evidence_id", "document_title", "section", "rule", "page_start", "page_end",
    "exact_quote", "official_url", "document_status", "document_id",
    "chunk_id", "raw_sha256",
}
NOTICE = (
    "Synthetic, legally unreviewed functional observations only. Citation "
    "integrity checks do not establish legal relevance, current law, novelty, "
    "patentability, medical safety, or compliance. No benchmark optimization."
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_scenarios(path=SCENARIOS):
    raw = Path(path).read_bytes()
    catalog = json.loads(raw.decode("utf-8-sig"))
    cases = catalog["cases"]
    if (catalog.get("synthetic") is not True
            or catalog.get("legal_relevance_review") != "unreviewed"
            or not isinstance(cases, list) or len(cases) != 12):
        raise ValueError("expected exactly 12 synthetic, legally unreviewed cases")
    ids = set()
    for case in cases:
        identifier = case.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError("scenario IDs must be unique nonempty strings")
        ids.add(identifier)
        if (case.get("synthetic") is not True
                or case.get("legal_relevance_review") != "unreviewed"):
            raise ValueError("each scenario must be synthetic and legally unreviewed")
        intake = case["intake"]
        if set(intake) != INTAKE_FIELDS:
            raise ValueError(f"{identifier}: unexpected or missing intake fields")
        for field, minimum, maximum in (
            ("description", 20, 1600), ("ingredients", 0, 400),
            ("additional_context", 0, 400),
        ):
            value = intake[field]
            if not isinstance(value, str) or not minimum <= len(value) <= maximum:
                raise ValueError(f"{identifier}: invalid {field} length")
        if (intake["novelty"] not in NOVELTIES
                or intake["goal"] != "patent_research"
                or intake["jurisdiction"] != "India"
                or intake["acknowledged_nonconfidential"] is not True
                or intake["traditional_use"] not in {"yes", "no", "unknown"}
                or intake["public_disclosure"] not in {"yes", "no", "unknown"}):
            raise ValueError(f"{identifier}: invalid intake schema")
        expectations = case["structural_expectations"]
        if any(type(expectations.get(key)) is not bool for key in (
            "requires_verified_quote", "requires_block_or_unresolved",
        )):
            raise ValueError("structural expectations must be explicit booleans")
        if expectations.get("legal_truth_label") is not None:
            raise ValueError("legal truth labels are not permitted")
    return catalog, hashlib.sha256(raw).hexdigest()


def validate_origin(value):
    parsed = urlsplit(value)
    if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1"
            or parsed.username or parsed.password or parsed.path not in {"", "/"}
            or parsed.query or parsed.fragment or not parsed.port):
        raise ValueError("base URL must be http://127.0.0.1:<port>")
    return f"http://127.0.0.1:{parsed.port}"


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class RequestFailure(Exception):
    def __init__(self, kind, message, **details):
        super().__init__(message)
        self.details = {"kind": kind, "message": message, **details}


class LocalClient:
    def __init__(self, origin, timeout):
        self.origin = origin
        self.timeout = timeout
        # Ignore environment proxies; redirects cannot escape loopback.
        self.opener = build_opener(
            ProxyHandler({}), HTTPCookieProcessor(CookieJar()), NoRedirect(),
        )

    def request(self, route, payload=None, token=None):
        headers = {"Accept": "application/json", "Origin": self.origin}
        data = None
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers.update({"Content-Type": "application/json",
                            "X-Passport-Token": token})
        request = Request(self.origin + route, data=data, headers=headers)
        started = time.monotonic()
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                raw = response.read(2_000_001)
                status = response.status
        except HTTPError as exc:
            body = exc.read(2_000_000).decode("utf-8", errors="replace")
            raise RequestFailure("http_error", str(exc), http_status=exc.code,
                                 response_text=body) from exc
        except (URLError, OSError, TimeoutError) as exc:
            raise RequestFailure("transport_error", str(exc)) from exc
        if len(raw) > 2_000_000:
            raise RequestFailure("response_too_large", "response exceeded 2 MB")
        text = raw.decode("utf-8", errors="replace")
        try:
            body = json.loads(text)
        except (ValueError, UnicodeError) as exc:
            raise RequestFailure("invalid_json", str(exc), http_status=status,
                                 response_text=text) from exc
        if not isinstance(body, dict):
            raise RequestFailure("invalid_response", "expected JSON object",
                                 http_status=status, response_body=body)
        return body, status, round((time.monotonic() - started) * 1000, 2)


class CitationVerifier:
    def __init__(self, root):
        self.root = Path(root)
        self.generations = {}
        self.generation_errors = {}

    def generation(self, identifier):
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("missing response generation_id")
        if identifier in self.generation_errors:
            raise ValueError(self.generation_errors[identifier])
        if identifier not in self.generations:
            # Verifies generation manifest, document/chunk hashes and raw archive
            # bytes. Loading uses the returned ID, never a hardcoded/current ID.
            from ipindia_corpus.indexer import generation
            try:
                loaded, docs, chunks = generation(
                    self.root, generation_id=identifier,
                )
                if loaded != identifier:
                    raise ValueError("response generation ID mismatch")
                self.generations[identifier] = (
                    {d["document_id"]: d for d in docs},
                    {c["chunk_id"]: c for c in chunks},
                )
            except Exception as exc:
                self.generation_errors[identifier] = (
                    f"generation verification unavailable: {type(exc).__name__}: {exc}"
                )
                raise ValueError(self.generation_errors[identifier]) from exc
        return self.generations[identifier]

    def evidence(self, item, documents, chunks):
        errors = []
        if not EVIDENCE_FIELDS.issubset(item):
            errors.append("missing_required_evidence_fields")
        chunk = chunks.get(item.get("chunk_id"))
        document = documents.get(item.get("document_id"))
        if not chunk or not document:
            return errors + ["unknown_chunk_or_document_in_response_generation"]
        if (chunk["document_id"] != document["document_id"]
                or document.get("source") != "ipindia"
                or chunk.get("source") != "ipindia"):
            errors.append("document_identity_or_source_mismatch")
        quote = item.get("exact_quote")
        if not isinstance(quote, str) or not quote or quote not in chunk["text"]:
            errors.append("quote_not_exact_generation_chunk_substring")
        comparisons = {
            "document_title": chunk.get("document_title"),
            "document_id": chunk.get("document_id"),
            "raw_sha256": chunk.get("raw_sha256"),
            "official_url": chunk.get("source_url"),
            **{field: chunk.get(field) for field in (
                "section", "rule", "page_start", "page_end",
            )},
        }
        for field, expected in comparisons.items():
            if item.get(field) != expected:
                errors.append(f"evidence_{field}_mismatch")
        if (item.get("raw_sha256") != document.get("sha256")
                or item.get("official_url") != document.get("source_url")):
            errors.append("evidence_document_provenance_mismatch")
        if item.get("document_status") != "unknown":
            errors.append("document_legal_status_not_unknown")
        if not official_ipindia_url(item.get("official_url")):
            errors.append("nonofficial_or_nonhttps_evidence_url")
        # indexer.generation has verified every raw archive hash; record the
        # exact integrity scope, not an assertion that a legal claim is true.
        return errors


def official_ipindia_url(value):
    if not isinstance(value, str):
        return False
    parsed = urlsplit(value)
    if (parsed.scheme != "https" or parsed.username or parsed.password
            or parsed.port not in {None, 443}):
        return False
    from ipindia_corpus.connectors import official_url
    try:
        official_url("ipindia", value)
    except ValueError:
        return False
    return True


def model_observation(body):
    model = body.get("model")
    if not isinstance(model, dict):
        return {"actual_model_run": False, "reason": "model_telemetry_absent"}
    from ipindia_corpus.reliable_answers import MODEL, DIGEST
    completed = model.get("calls_completed")
    observed = (
        model.get("name") == MODEL and model.get("digest") == DIGEST
        and type(completed) is int and completed > 0
    )
    return {
        "actual_model_run": observed,
        "basis": "server_reported_completed_calls_and_pinned_identity",
        "model": model,
        "reason": "completed_pinned_model_call_observed" if observed
                  else "no_completed_pinned_model_call_observed",
    }


def validate_result(body, case, verifier):
    errors = []
    if body.get("status") not in STATUSES:
        errors.append("invalid_status")
    if body.get("overall_status") != "informational_evidence_only":
        errors.append("overall_status_not_informational_evidence_only")
    if body.get("legal_relevance_review") != "unreviewed":
        errors.append("legal_relevance_review_not_unreviewed")
    evidence = body.get("verified_evidence")
    issues = body.get("issues")
    unresolved = body.get("unresolved_questions")
    unsupported = body.get("unsupported_domains")
    for key, value in (
        ("verified_evidence", evidence), ("issues", issues),
        ("unresolved_questions", unresolved), ("unsupported_domains", unsupported),
    ):
        if not isinstance(value, list):
            errors.append(f"{key}_not_list")
    evidence = evidence if isinstance(evidence, list) else []
    issues = issues if isinstance(issues, list) else []
    unresolved = unresolved if isinstance(unresolved, list) else []
    unsupported = unsupported if isinstance(unsupported, list) else []
    if not unsupported or any(
        not isinstance(item, dict) or not item.get("domain")
        or not item.get("limitation") for item in unsupported
    ):
        errors.append("unsupported_domain_catalog_missing_or_incomplete")
    if not isinstance(body.get("timings_ms"), dict):
        errors.append("timings_ms_missing")
    try:
        documents, chunks = verifier.generation(body.get("generation_id"))
    except Exception as exc:
        errors.append(str(exc))
        documents, chunks = {}, {}
    verified_ids = set()
    evidence_checks = []
    for item in evidence:
        if not isinstance(item, dict):
            errors.append("evidence_entry_not_object")
            continue
        identifier = item.get("evidence_id")
        item_errors = verifier.evidence(item, documents, chunks)
        if not isinstance(identifier, str) or not identifier or identifier in verified_ids:
            item_errors.append("missing_or_duplicate_evidence_id")
        if isinstance(identifier, str) and identifier:
            verified_ids.add(identifier)
        evidence_checks.append({
            "evidence_id": identifier, "errors": item_errors,
            "integrity_verified": not item_errors,
            "scope": "exact_generation_chunk_substring_and_verified_raw_source_hash",
        })
        errors.extend(item_errors)
    for issue in issues:
        if not isinstance(issue, dict):
            errors.append("issue_not_object")
            continue
        if issue.get("evidence_status") not in {
            "source_verified", "not_found", "selection_rejected", "dependency_unavailable",
        }:
            errors.append("issue_evidence_status_missing_or_invalid")
        ids = issue.get("evidence_ids")
        if not isinstance(ids, list) or any(
            not isinstance(i, str) or i not in verified_ids for i in ids
        ):
            errors.append("issue_evidence_reference_missing_or_unknown")
        if issue.get("evidence_status") == "source_verified" and not ids:
            errors.append("source_verified_issue_has_no_evidence")
        if issue.get("legal_relevance_review", "unreviewed") != "unreviewed":
            errors.append("issue_legal_relevance_not_unreviewed")
    # Legal-status assertions remain constrained throughout the response.
    # URL authority checks apply to verified evidence only. User-provided text
    # mirrored in an invention summary or diagnostic is still untrusted intake.
    def inspect(value, evidence_urls=False):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"document_status", "legal_status"} and child != "unknown":
                    errors.append("non_unknown_legal_status")
                inspect(child, evidence_urls)
        elif isinstance(value, list):
            for child in value:
                inspect(child, evidence_urls)
        elif evidence_urls and isinstance(value, str):
            for url in re.findall(r"https?://[^\s<>\"']+", value):
                if not official_ipindia_url(url.rstrip(".,;)")):
                    errors.append("untrusted_or_nonofficial_evidence_url")
    inspect(body)
    inspect(evidence, evidence_urls=True)
    expectations = case["structural_expectations"]
    if expectations["requires_verified_quote"] and not evidence:
        errors.append("research_case_has_no_verified_quote")
    if (expectations["requires_block_or_unresolved"]
            and body.get("status") not in {"blocked", "unavailable", "needs_clarification"}
            and not unresolved):
        errors.append("sensitive_or_current_law_request_neither_blocked_nor_unresolved")
    observation = model_observation(body)
    if expectations["requires_verified_quote"] and not observation["actual_model_run"]:
        errors.append("research_case_has_no_completed_pinned_model_observation")
    if body.get("status") == "unavailable":
        outcome = "dependency_or_model_unavailable"
    elif errors:
        outcome = "validation_failed"
    elif evidence:
        outcome = "observed_with_verified_evidence"
    else:
        outcome = "safe_abstention_observed"
    return {
        "outcome": outcome, "structural_and_integrity_checks_passed": not errors,
        "errors": sorted(set(errors)), "evidence_checks": evidence_checks,
        "model_observation": observation,
        "legal_relevance_review": "unreviewed",
    }


def persist(report):
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    # Write the one authorized report file only; no sidecars or alternate reports.
    with REPORT.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def run_demo(origin, timeout):
    catalog, input_sha256 = load_scenarios()
    if REPORT.exists():
        report = json.loads(REPORT.read_text(encoding="utf-8-sig"))
        if not isinstance(report.get("runs"), list):
            raise ValueError("existing report has no runs array; refusing to erase it")
    else:
        report = {"schema_version": "passport-live-demo-v1", "notice": NOTICE, "runs": []}
    run = {
        "run_id": str(uuid.uuid4()), "started_at": utc_now(),
        "harness_version": HARNESS_VERSION,
        "status": "running", "origin": origin, "serial": True, "retries": 0,
        "synthetic": True, "legal_relevance_review": "unreviewed", "notice": NOTICE,
        "scenario_file": str(SCENARIOS), "scenario_file_sha256": input_sha256,
        "preserved_input_set": catalog, "case_results": [],
    }
    report["runs"].append(run)
    # Freeze exact inputs before the very first live request.
    persist(report)
    client = LocalClient(origin, timeout)
    verifier = CitationVerifier(ROOT)
    session_error = None
    try:
        session, status, elapsed = client.request("/api/session")
        token = session.get("csrf_token")
        if not isinstance(token, str) or not token:
            raise RequestFailure("session_error", "session has no CSRF token")
        run["session"] = {"http_status": status, "elapsed_ms": elapsed}
    except RequestFailure as exc:
        session_error = exc.details
        token = None
        run["session"] = {"error": session_error}
    try:
        for case in catalog["cases"]:
            row = {"case_id": case["id"], "started_at": utc_now(),
                   "intake": case["intake"], "legal_relevance_review": "unreviewed",
                   "outcome": "in_progress", "attempted": session_error is None}
            run["case_results"].append(row)
            persist(report)
            if session_error:
                row.update(outcome="session_unavailable", attempted=False,
                           error=session_error)
            else:
                row["attempted"] = True
                started = time.monotonic()
                try:
                    body, status, elapsed = client.request(
                        "/api/passport", case["intake"], token,
                    )
                    row.update(response=body, http_status=status, elapsed_ms=elapsed)
                    persist(report)
                    try:
                        row["validation"] = validate_result(body, case, verifier)
                        row["outcome"] = row["validation"]["outcome"]
                    except Exception as exc:
                        row.update(outcome="validation_error",
                                   error={"kind": type(exc).__name__, "message": str(exc)})
                except RequestFailure as exc:
                    row.update(outcome=exc.details["kind"], error=exc.details,
                               elapsed_ms=round((time.monotonic() - started) * 1000, 2))
            row["finished_at"] = utc_now()
            persist(report)
            print(f"{case['id']}: {row['outcome']}", flush=True)
        run["status"] = "completed"
    except KeyboardInterrupt:
        run["status"] = "interrupted"
        if run["case_results"] and run["case_results"][-1]["outcome"] == "in_progress":
            run["case_results"][-1].update(
                outcome="interrupted_before_response_recorded", finished_at=utc_now(),
            )
        run["not_attempted_case_ids"] = [
            case["id"] for case in catalog["cases"][len(run["case_results"]):]
        ]
    finally:
        run["finished_at"] = utc_now()
        run["outcome_counts"] = {
            value: sum(row["outcome"] == value for row in run["case_results"])
            for value in sorted({row["outcome"] for row in run["case_results"]})
        }
        persist(report)
    print(f"Saved observations to {REPORT}", flush=True)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8003")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="per-request timeout in seconds; no retries")
    args = parser.parse_args()
    if not 0 < args.timeout <= 900:
        parser.error("--timeout must be greater than 0 and at most 900 seconds")
    origin = validate_origin(args.base_url)
    run = run_demo(origin, args.timeout)
    return 0 if (
        run["status"] == "completed" and len(run["case_results"]) == 12
        and all(row.get("validation", {}).get("structural_and_integrity_checks_passed")
                for row in run["case_results"])
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())

