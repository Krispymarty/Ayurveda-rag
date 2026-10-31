"""Opt-in frozen-input ASGI experiment; never installed in a running service.

The only alternative reuses /api/show metadata within one HTTP request. Every
model call and every existing archive check remains real. Historical observations
are not rewritten. Uses two explicit synthetic scenarios, never ordinary inputs.
"""
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

from fastapi.testclient import TestClient

from ipindia_corpus import reliable_answers as extracts
from ipindia_corpus.llm_experiment import ExperimentFailure, OllamaExperimentClient
from ipindia_corpus.passport.api import create_app
from ipindia_corpus.passport.service import (
    Intake, LocalSearch, PassportEngine, PublishedVerifier, parse_plan,
)
from ipindia_corpus.paths import immutable_write
from tools.passport_demo import CitationVerifier, load_scenarios, validate_result

ROOT = Path(__file__).resolve().parents[1]
CASES = ("traditional_ashwagandha_extract", "novel_process")


class ScopedMetadataClient:
    """Metadata only: no inference output cache, no lifetime beyond one request."""
    def __init__(self, client):
        self.client = client
        self.cached = None

    def receipt(self, model):
        if self.cached is None:
            value = self.client.receipt(model)
            if value.get("model_digest") != extracts.DIGEST or not value.get("thinking_capable"):
                raise ExperimentFailure("model_digest_changed")
            self.cached = deepcopy(value)
        else:
            if model != self.cached["model"]:
                raise ExperimentFailure("model_digest_changed")
            # Retain an exact /api/tags identity check before EVERY later chat.
            self.client.verify_identity(self.cached)
        return deepcopy(self.cached)

    def request(self, *args, **kwargs):
        return self.client.request(*args, **kwargs)


class BoundProvider:
    def __init__(self, client, *, reuse_show=False, expected=None, intake=None, configuration=None):
        self.client = client
        self.provider = extracts.SpanProvider(ScopedMetadataClient(client) if reuse_show else client)
        self.expected = deepcopy(expected)
        self.intake = intake
        self.records = []
        self.binding = extracts.canonical(extracts.configuration() if configuration is None else configuration)

    def chat(self, fields):
        if extracts.canonical(extracts.configuration()) != self.binding:
            raise ExperimentFailure("frozen_configuration_changed")
        index = len(self.records)
        record = {"fields": deepcopy(fields), "request_sha256": hashlib.sha256(extracts.canonical(fields)).hexdigest(),
                  "completed": False, "guard_passed": False}
        self.records.append(record)
        started = time.monotonic()
        try:
            if self.expected is not None and (
                    index >= len(self.expected)
                    or extracts.canonical(fields) != extracts.canonical(self.expected[index]["fields"])):
                raise ExperimentFailure("frozen_call_fields_changed")
            record["guard_passed"] = True
            raw, timing = self.provider.chat(fields)
            record.update(raw=raw, timing=timing, completed=True)
            if index == 0 and self.expected is not None and self.intake is not None:
                # Validate both objects, then compare their original opaque IDs.
                # Display expansion truncates user context and cannot be an ID.
                parse_plan(raw, self.intake)
                parse_plan(self.expected[0]["raw"], self.intake)
                if json.loads(raw) != json.loads(self.expected[0]["raw"]):
                    raise ExperimentFailure("frozen_planner_changed")
            return raw, timing
        except Exception as exc:
            record["error"] = str(exc) if isinstance(exc, ExperimentFailure) else type(exc).__name__
            raise
        finally:
            record["wall_ms"] = (time.monotonic() - started) * 1000


class TimedClient(OllamaExperimentClient):
    def __init__(self):
        super().__init__(timeout=45)
        self.observations = []

    def request(self, method, path, payload=None, **kwargs):
        record = {"method": method, "path": path, "completed": False}
        self.observations.append(record)
        started = time.monotonic()
        try:
            result = super().request(method, path, payload, **kwargs)
            record["completed"] = True
            return result
        finally:
            record["wall_ms"] = (time.monotonic() - started) * 1000


class RecordedSearch:
    def __init__(self, live, frozen=None):
        self.live = live
        self.responses = {} if frozen is None else deepcopy(frozen)
        self.frozen = frozen is not None

    def status(self):
        # Live generation binding checks still occur before/after every passport.
        return self.live.status()

    def __call__(self, question, top_k=5):
        if top_k != 5:
            raise ValueError("fixed_issue_search_depth")
        if self.frozen:
            if question not in self.responses:
                raise ValueError("frozen_search_question_changed")
        else:
            self.responses[question] = self.live(question, top_k=5)
        return deepcopy(self.responses[question])


def stable_result(value):
    return {key: value.get(key) for key in (
        "status", "reason", "generation_id", "issues", "verified_evidence", "unresolved_questions",
        "unsupported_domains", "recommended_verification_steps", "request_policy_version",
    )}


def http_attempt(case, *, archive, frozen=None, reuse_show=False):
    """Complete existing private ASGI route/boundary, excluding a TCP network hop."""
    intake = Intake(**case["intake"])
    model = TimedClient()
    provider = BoundProvider(model, reuse_show=reuse_show,
                             expected=None if frozen is None else frozen["calls"], intake=intake,
                             configuration=None if frozen is None else frozen["configuration"])
    search = RecordedSearch(LocalSearch(), None if frozen is None else frozen["search"])
    engine = PassportEngine(search, archive, provider=provider, prefetch=False)
    app = create_app(engine=engine, enabled=True)

    async def transport(scope, receive, send):
        # Same synthetic loopback peer adapter as the project's API test suite.
        # Host/Origin/cookie/token/body/concurrency checks still run unchanged.
        await app({**scope, "client": ("127.0.0.1", 50001)}, receive, send)

    started = time.monotonic()
    try:
        with TestClient(transport, base_url="http://127.0.0.1:8003") as client:
            session = client.get("/api/session")
            session.raise_for_status()
            started = time.monotonic()
            response = client.post("/api/passport", json=case["intake"], headers={
                "Origin": "http://127.0.0.1:8003", "X-Passport-Token": session.json()["csrf_token"],
            })
            elapsed = (time.monotonic() - started) * 1000
    except BaseException as exc:
        exc.experiment_partial = {"calls": provider.records, "search": search.responses,
            "provider_http_requests": model.observations,
            "asgi_http_ms": (time.monotonic() - started) * 1000}
        raise
    body = response.json()
    physical_calls = [x for x in model.observations if x["path"] == "/api/chat"]
    row = {"case_id": case["id"], "variant": "scoped_show_metadata" if reuse_show else "baseline",
           "asgi_http_status": response.status_code, "asgi_http_ms": elapsed,
           "server_timing": response.headers.get("Server-Timing"), "response": body,
           "physical_model_calls_attempted": len(physical_calls),
           "physical_model_calls_completed": sum(x["completed"] for x in physical_calls),
           "provider_http_requests": model.observations, "calls": provider.records,
           "search": search.responses, "configuration": extracts.configuration(),
           "scope": "existing private ASGI HTTP path; simulated loopback peer/TCP excluded; live hybrid search frozen during pairs"}
    if frozen is not None:
        row["frozen_sequence_complete"] = (len(provider.records) == len(frozen["calls"])
            and all(c.get("guard_passed") and c.get("completed") and not c.get("error") for c in provider.records))
        row["same_backend_outcome_as_capture"] = stable_result(body) == stable_result(frozen["response"])
    return row


def run(output):
    directory = Path(output).resolve()
    allowed = (ROOT / "data/next_stage/nba_fidelity_20261101/passport").resolve()
    if not directory.is_relative_to(allowed) or directory.exists():
        raise ValueError("new_excluded_experiment_directory_required")
    directory.mkdir(parents=True)
    catalog, scenario_hash = load_scenarios()
    selected = [next(c for c in catalog["cases"] if c["id"] == key) for key in CASES]
    live = LocalSearch()
    gid = live.status()["generation_id"]
    if gid != "20261014T204940358650Z":
        raise ValueError("protected_generation_changed")
    archive = PublishedVerifier(ROOT, gid)
    verifier = CitationVerifier(ROOT)
    report = {"at": datetime.now(timezone.utc).isoformat(), "synthetic": True,
              "legal_relevance_review": "unreviewed", "configuration": extracts.configuration(),
              "scenario_sha256": scenario_hash, "cases": list(CASES), "generation": gid,
              "captures": [], "attempts": [], "promotion_allowed": False,
              "notice": "Frozen inputs/real outputs; no source/answer activation; ASGI timing excludes TCP and live hybrid retrieval during pairs. No retries."}

    def save():
        (directory / "observations.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    save()
    try:
        for case in selected:
            captured = {"case_id": case["id"], "status": "in_progress"}
            report["captures"].append(captured)
            save()
            try:
                captured.update(http_attempt(case, archive=archive))
                save()
                captured["validation"] = validate_result(captured["response"], case, verifier)
                captured["status"] = "completed"
            except BaseException as exc:
                captured.update(getattr(exc, "experiment_partial", {}))
                captured.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                                error_type=type(exc).__name__, error=str(exc))
                save()
                raise
            save()
            if (captured["asgi_http_status"] != 200 or not captured["calls"]
                    or not all(c.get("completed") for c in captured["calls"])
                    or not captured["validation"]["structural_and_integrity_checks_passed"]):
                raise ValueError("capture_incomplete_no_experiment")
        freeze = {"configuration": report["configuration"], "cases": report["captures"]}
        freeze_bytes = json.dumps(freeze, sort_keys=True, ensure_ascii=False).encode("utf-8")
        immutable_write(directory / "frozen_inputs.json", freeze_bytes)
        report["frozen_inputs_sha256"] = hashlib.sha256(freeze_bytes).hexdigest()
        save()
        for n in range(1, 4):
            for case, captured in zip(selected, report["captures"]):
                # Interleave A/B order. Every interrupted/failed attempt is kept.
                for reuse in ((False, True) if n % 2 else (True, False)):
                    marker = {"case_id": case["id"], "pair": n, "reuse_show": reuse, "status": "in_progress"}
                    report["attempts"].append(marker)
                    save()
                    try:
                        if hashlib.sha256((directory / "frozen_inputs.json").read_bytes()).hexdigest() != report["frozen_inputs_sha256"]:
                            raise ValueError("frozen_bundle_changed")
                        row = http_attempt(case, archive=archive, frozen=captured, reuse_show=reuse)
                        marker.update(row)
                        save()
                        row["validation"] = validate_result(row["response"], case, verifier)
                        marker.update(row, status="completed")
                    except Exception as exc:
                        marker.update(getattr(exc, "experiment_partial", {}))
                        marker.update(status="failed", error_type=type(exc).__name__, error=str(exc))
                    except BaseException as exc:
                        marker.update(getattr(exc, "experiment_partial", {}))
                        marker.update(status="interrupted", error_type=type(exc).__name__)
                        save()
                        raise
                    save()
                    print(f"pair {n} {case['id']} reuse_show={reuse}: {marker['status']} {marker.get('asgi_http_ms')}", flush=True)
        report["status"] = "completed"
    except BaseException as exc:
        report.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        for variant in ("baseline", "scoped_show_metadata"):
            rows = [r for r in report["attempts"] if r.get("variant") == variant]
            if rows:
                report[variant] = {"attempts": len(rows), "asgi_http_median_ms": statistics.median(r["asgi_http_ms"] for r in rows),
                                  "strictly_equivalent": sum(r.get("frozen_sequence_complete", False) and r.get("same_backend_outcome_as_capture", False) for r in rows)}
        report["promotion_allowed"] = False
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        save()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--live-local-model", action="store_true", required=True)
    arguments = parser.parse_args()
    run(arguments.output)
