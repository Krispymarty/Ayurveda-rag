"""Local evidence search and gated, quotation-only answer API."""

from __future__ import annotations

import sqlite3
import re
import json
import hashlib
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from psycopg import Error as DatabaseError
from pydantic import BaseModel, ConfigDict, Field, StrictBool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import Literal

from .indexer import DIMENSIONS, MODEL_ID, _connect, model_revision, generation
from .paths import archive_path
from . import retrieval, retrieval_legacy
from .sync_runner import ROOT
from .grounded_answers import generate_answer, load_config, NOTICE
from .http_security import configure_security
from .readiness import Readiness
from .cloud_config import load_deployment_config
from . import demo_answers

DEPLOYMENT = load_deployment_config()


@asynccontextmanager
async def lifespan(app):
    readiness.start()
    yield
    readiness.close()

app = FastAPI(title="AyurIP Evidence Search", version="0.1.0", lifespan=lifespan,
              docs_url=None if DEPLOYMENT.public_research else "/docs",
              redoc_url=None if DEPLOYMENT.public_research else "/redoc",
              openapi_url=None if DEPLOYMENT.public_research else "/openapi.json")
configure_security(app)
app.mount("/assets", StaticFiles(directory=ROOT / "web", check_dir=False), name="assets")
app.mount("/Images", StaticFiles(directory=ROOT / "web/images", check_dir=False), name="images_cap")
app.mount("/images", StaticFiles(directory=ROOT / "web/images", check_dir=False), name="images_low")
logger = logging.getLogger(__name__)
RANKING_REVISIONS = {"v1": hashlib.sha256(Path(retrieval_legacy.__file__).read_bytes()).hexdigest(),
                     "v2": hashlib.sha256(Path(retrieval.__file__).read_bytes()).hexdigest()}


class SearchHit(BaseModel):
    chunk_id: str
    document_id: str
    title: str | None
    source: str
    legal_domain: str
    evidence_role: str
    source_url: str
    retrieved_at: str
    authority: str
    section: str | None
    rule: str | None
    page_start: int | None
    page_end: int | None
    status: str
    raw_sha256: str
    text: str
    score: float
    short_unit: bool = False


class SearchResponse(BaseModel):
    generation_id: str
    query: str
    hits: list[SearchHit]
    notice: str
    ranking_version: str = "v1"
    ranking_revision: str = ""


class AnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    question: str = Field(min_length=2, max_length=500)
    source: Literal["ipindia", "indiacode", "tkdl"] | None = None
    domain: str | None = Field(default=None, max_length=80)
    top_k: int = Field(default=5, ge=1, le=10, strict=True)
    acknowledged_nonconfidential: StrictBool = False
    acknowledged_external_processing: StrictBool = False


@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
def home_endpoint():
    path = ROOT / "web/index.html"
    if not path.is_file():
        raise HTTPException(status_code=503, detail="interface_unavailable")
    return FileResponse(path, media_type="text/html", headers={"Cache-Control": "no-store"})


@app.get("/research", include_in_schema=False)
@app.get("/research.html", include_in_schema=False)
def research_endpoint():
    path = ROOT / "web/research.html"
    if not path.is_file():
        raise HTTPException(status_code=503, detail="interface_unavailable")
    return FileResponse(path, media_type="text/html", headers={"Cache-Control": "no-store"})


@app.get("/styles.css", include_in_schema=False)
def styles_endpoint():
    path = ROOT / "web/styles.css"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="not_found")
    return FileResponse(path, media_type="text/css")


@app.get("/app.js", include_in_schema=False)
def app_js_endpoint():
    path = ROOT / "web/app.js"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="not_found")
    return FileResponse(path, media_type="application/javascript")


def _certification_summary():
    try:
        report = json.loads((ROOT / "reports/corpus_certification.json").read_text(encoding="utf-8"))
        return {source: {**{key: report["sources"].get(source, {}).get(key) for key in (
            "staged_documents", "certified_documents", "provenance_verified_documents", "status", "search_admission")},
            "certified_generation_id": report["sources"].get(source, {}).get("certified_generation_id", report.get("generation_id"))}
                for source in ("indiacode", "tkdl")}
    except (OSError, ValueError, KeyError, TypeError):
        return {source: {"status": "unverified", "certified_documents": 0} for source in ("indiacode", "tkdl")}


def _active_source_counts():
    connection = _connect()
    try:
        row = connection.execute("SELECT value FROM app_state WHERE key='active_generation'").fetchone()
        if not row:
            return None, {}
        generation_id = row[0]
        counts = dict(connection.execute("SELECT metadata->>'source',count(*) FROM corpus_documents WHERE generation_id=%s GROUP BY metadata->>'source'", (generation_id,)).fetchall())
        return generation_id, {source: int(counts.get(source, 0)) for source in ("ipindia", "indiacode", "tkdl")}
    finally:
        connection.close()


def _answer_capabilities():
    certification = _certification_summary()
    try:
        generation_id, counts = _active_source_counts()
    except Exception:
        logger.exception("Active corpus capabilities unavailable")
        generation_id, counts = None, {}
    certified = bool(generation_id and all(counts.get(source, 0) > 0 for source in ("ipindia", "indiacode", "tkdl"))
                     and all(certification[source].get("status") == "certified"
                             and certification[source].get("search_admission") == "accepted"
                             and certification[source].get("certified_generation_id") == generation_id
                             and type(certification[source].get("certified_documents")) is int
                             and certification[source]["certified_documents"] >= counts.get(source, 0)
                             for source in ("indiacode", "tkdl")))
    try:
        config = load_config(ROOT)
        configured = bool(config.get("enabled"))
        provider = {"provider": config["provider"], "model": config["model"], "model_digest": config["model_digest"], "digest_verified": False}
    except (OSError, ValueError, TypeError, KeyError):
        configured, provider = False, {}
    if DEPLOYMENT.public_research:
        configured, provider = False, {}
    if demo_answers.enabled() and not DEPLOYMENT.public_research:
        from .inference import configured_identity
        try:
            identity = configured_identity()
        except Exception:
            identity = {}
        return {"active_generation_id": generation_id, "index_available": bool(generation_id),
                "active_source_document_counts": counts, "three_source_certified": certified,
                "certification": certification, "provider_configured": bool(identity), "provider": identity,
                "answer_generation_enabled": bool(generation_id and counts.get("ipindia", 0) and identity),
                "answer_generation_reason": None if generation_id and counts.get("ipindia", 0) and identity else "demo_dependencies_unavailable",
                "answer_mode": "verified_extracts", "demo": True, "scope": "ipindia_only",
                "public_release_ready": False, "deployment": DEPLOYMENT.mode,
                "private_passport_enabled": False, "notice": demo_answers.DEMO_NOTICE}
    return {"active_generation_id": generation_id, "index_available": bool(generation_id),
            "active_source_document_counts": counts, "three_source_certified": certified,
            "certification": certification, "provider_configured": configured, "provider": provider,
            "answer_generation_enabled": bool(certified and configured), "answer_mode": "verified_extracts",
            "answer_generation_reason": None if certified and configured else "three_source_not_certified" if not certified else "answer_generation_disabled",
            "public_release_ready": False, "deployment": DEPLOYMENT.mode,
            "private_passport_enabled": False, "notice": NOTICE}


@app.get("/capabilities")
def capabilities_endpoint():
    return _answer_capabilities()


def _verify_answer_hit(hit, generation_id):
    expected = _published_chunks(generation_id).get(hit["chunk_id"])
    if expected is None:
        raise ValueError("published_chunk_missing")
    for field in ("document_id", "source_url", "raw_sha256", "section", "rule", "page_start", "page_end"):
        if hit.get(field) != expected.get(field):
            raise ValueError("answer_citation_mismatch")
    expected_title = expected.get("document_title") or expected.get("title")
    if hit.get("title") != expected_title:
        raise ValueError("answer_title_mismatch")
    verify_hit(expected, hit["text"], _documents(generation_id), root=ROOT, expected_chunk=expected)


@app.post("/answer")
def answer_endpoint(request: AnswerRequest, http_request: Request = None):
    if demo_answers.enabled() and not DEPLOYMENT.public_research:
        if http_request is None:
            return JSONResponse(status_code=403, content=demo_answers.response("http_request_required"))
        origin = http_request.headers.get("origin")
        if origin and origin != f"http://{http_request.headers.get('host')}":
            return JSONResponse(status_code=403, content=demo_answers.response("same_origin_required"))
        if not request.acknowledged_nonconfidential or not request.acknowledged_external_processing:
            return JSONResponse(status_code=403, content=demo_answers.response("external_processing_consent_required"))
        if request.source not in (None, "ipindia"):
            return demo_answers.response("source_not_available_in_demo")
        try:
            snapshot = _ensure_ready()
            gid = snapshot["binding"][0]
        except Exception:
            return JSONResponse(status_code=503, content=demo_answers.response("search_not_ready", status="unavailable"))

        def bound():
            if _ensure_ready()["binding"][0] != gid:
                raise ValueError("active_generation_changed")

        def demo_search(question, top_k=5):
            response = search(question, top_k=5, source="ipindia", domain=request.domain)
            if response["generation_id"] != gid or any(hit.get("source") != "ipindia" for hit in response["hits"]):
                raise ValueError("demo_source_or_generation_changed")
            return response

        result = demo_answers.answer(request.question, demo_search, _verify_answer_hit, bound)
        return JSONResponse(status_code=503 if result["status"] == "unavailable" else 200, content=result)
    if DEPLOYMENT.public_research:
        return JSONResponse(status_code=503, content={"status": "unavailable", "answer_text": "", "claims": [],
            "reason": "three_source_not_certified", "notice": NOTICE,
            "generation_id": None, "mode": "verified_extracts", "provider": {}})
    capabilities = _answer_capabilities()
    if not capabilities["answer_generation_enabled"]:
        return JSONResponse(status_code=503, content={"status": "unavailable", "answer_text": "", "claims": [],
            "reason": capabilities["answer_generation_reason"], "notice": NOTICE,
            "generation_id": capabilities["active_generation_id"], "mode": "verified_extracts", "provider": capabilities["provider"]})
    # Recheck the exact generation returned by search. A concurrent switch must
    # not let a previously gated request consume a different incomplete index.
    def verified_search(question, **kwargs):
        response = search(question, **kwargs)
        if response.get("generation_id") != capabilities["active_generation_id"]:
            raise ValueError("active_generation_changed")
        return response
    result = generate_answer(request.question, verified_search, verify_hit_fn=_verify_answer_hit,
                             source=request.source, domain=request.domain, top_k=request.top_k, root=ROOT)
    return JSONResponse(status_code=503 if result["status"] == "unavailable" else 200, content=result)

def _load_model(record):
    started = time.monotonic()
    import torch
    # Small batches regress badly with the container's large default thread pool.
    # This is local CPU inference; keep threads bounded and record it in evals.
    torch.set_num_threads(4)
    from sentence_transformers import SentenceTransformer
    record("embedding_imports", time.monotonic() - started)
    started = time.monotonic()
    model = SentenceTransformer(MODEL_ID, revision=model_revision())
    if model.get_sentence_embedding_dimension() != DIMENSIONS:
        raise RuntimeError("embedding dimension mismatch")
    record("embedding_load", time.monotonic() - started)
    started = time.monotonic()
    model.encode(["query: AyurIP startup readiness"], normalize_embeddings=True)
    record("first_inference", time.monotonic() - started)
    return model


def _binding():
    connection = _connect()
    try:
        state = dict(connection.execute("SELECT key,value FROM app_state WHERE key IN ('active_generation','ranking_version')").fetchall())
        gid = state.get("active_generation")
        if not gid:
            raise ValueError("no_active_index")
        version = state.get("ranking_version", "v1")
        row = connection.execute("SELECT documents,chunks,model_revision,manifest_sha256 FROM index_generations WHERE generation_id=%s", (gid,)).fetchone()
        if not row or row[2] != model_revision() or version not in RANKING_REVISIONS:
            raise ValueError("index_binding_invalid")
        manifest_hash = hashlib.sha256((ROOT / "data/generations" / gid / "manifest.json").read_bytes()).hexdigest()
        if row[3] != manifest_hash:
            raise ValueError("index_manifest_changed")
        return (gid, manifest_hash, row[2], version, RANKING_REVISIONS[version]), row[:2]
    finally:
        connection.close()


def _initialize(record):
    started = time.monotonic()
    binding, counts = _binding()
    connection = _connect()
    try:
        actual = connection.execute("SELECT (SELECT count(*) FROM corpus_documents WHERE generation_id=%s),(SELECT count(*) FROM corpus_chunks c JOIN embedding_cache e ON c.content_hash=e.content_hash AND e.model_revision=%s WHERE c.generation_id=%s)", (binding[0], binding[2], binding[0])).fetchone()
        if actual != counts:
            raise ValueError("active_index_incomplete")
    finally:
        connection.close()
    record("database", time.monotonic() - started)
    started = time.monotonic()
    _, documents, chunks = generation(ROOT, generation_id=binding[0])
    record("archive_validation", time.monotonic() - started)
    model = _load_model(record)
    if _binding()[0] != binding:
        raise ValueError("active_binding_changed_during_initialization")
    return {"binding": binding, "model": model,
            "documents": {d["document_id"]: d for d in documents},
            "chunks": {c["chunk_id"]: c for c in chunks}}


readiness = Readiness(_initialize, deadline=60)


def _ensure_ready():
    snapshot = readiness.require()
    if _binding()[0] != snapshot["binding"]:
        readiness.invalidate()
        raise ValueError("active_binding_changed")
    return snapshot


def _model():
    return readiness.require()["model"]


def _documents(generation_id):
    snapshot = readiness.require()
    if snapshot["binding"][0] != generation_id or snapshot["binding"][2] != model_revision():
        raise ValueError("active_binding_changed")
    return snapshot["documents"]


def _published_chunks(generation_id):
    _documents(generation_id)
    return readiness.require()["chunks"]


def verify_hit(metadata, text, documents, root=ROOT, *, expected_chunk=None):
    if expected_chunk is not None and metadata != expected_chunk:
        raise ValueError("published_chunk_mismatch")
    doc = documents.get(metadata["document_id"])
    if not doc or metadata["source_url"] != doc["source_url"] or metadata["raw_sha256"] != doc["sha256"]:
        raise ValueError("citation_mismatch")
    if hashlib.sha256(text.encode()).hexdigest() != metadata["content_hash"]:
        raise ValueError("indexed_text_hash_mismatch")
    if hashlib.sha256(archive_path(root, doc["raw_path"]).read_bytes()).hexdigest() != doc["sha256"]:
        raise ValueError("archive_hash_mismatch")


def search(query: str, top_k: int = 5, source: str | None = None,
           domain: str | None = None) -> dict:
    revision = model_revision()
    vector = _model().encode(["query: " + query], normalize_embeddings=True)[0]
    embedding = "[" + ",".join(str(float(value)) for value in vector) + "]"
    provision = re.search(r"\b(section|rule)\s+(\d+[a-z]?)\b", query, re.I)
    section = provision.group(2).lower() if provision and provision.group(1).lower() == "section" else None
    rule = provision.group(2).lower() if provision and provision.group(1).lower() == "rule" else None
    connection = _connect()
    try:
        # Read the release pair in one PostgreSQL statement so an activation
        # between statements cannot mix an older generation with a newer ranker.
        selection = dict(connection.execute("SELECT key,value FROM app_state WHERE key IN ('active_generation','ranking_version')").fetchall())
        active = (selection['active_generation'],) if selection.get('active_generation') else None
        if not active:
            raise ValueError("no indexed corpus generation")
        indexed_revision = connection.execute("SELECT model_revision,manifest_sha256 FROM index_generations WHERE generation_id=%s", (active[0],)).fetchone()
        if not indexed_revision or indexed_revision[0] != revision:
            raise ValueError("active_model_revision_mismatch")
        _documents(active[0])
        if hashlib.sha256((ROOT / "data" / "generations" / active[0] / "manifest.json").read_bytes()).hexdigest() != indexed_revision[1]:
            raise ValueError("active_manifest_hash_mismatch")
        ranking_version = selection.get('ranking_version', 'v1')
        if ranking_version not in ("v1", "v2"):
            raise ValueError("unsupported_ranking_version")
        ranker = retrieval if ranking_version == "v2" else retrieval_legacy
        rows = connection.execute(ranker.SEARCH_SQL, {"query": query, "source": source, "domain": domain,
                                               "embedding": embedding, "revision": revision, "limit": top_k,
                                               "section": section, "rule": rule, "generation_id": active[0], **ranker.query_parameters(query)}).fetchall()
    finally:
        connection.close()
    hits = []
    documents = _documents(active[0])
    published_chunks = _published_chunks(active[0])
    for chunk_id, text, metadata, score in rows:
        expected = published_chunks.get(chunk_id)
        if expected is None:
            raise ValueError("published_chunk_missing")
        verify_hit(metadata, text, documents, expected_chunk=expected)
        hits.append({"chunk_id": chunk_id, "document_id": metadata["document_id"],
                     "title": metadata.get("document_title") or metadata.get("title"),
                     "source": metadata["source"], "legal_domain": metadata["legal_domain"],
                     "evidence_role": metadata["evidence_role"], "source_url": metadata["source_url"],
                     "retrieved_at": metadata["retrieved_at"], "authority": metadata["authority"],
                     "section": metadata.get("section"), "rule": metadata.get("rule"),
                     "page_start": metadata.get("page_start"), "page_end": metadata.get("page_end"),
                     "status": "unknown", "raw_sha256": metadata["raw_sha256"],
                     "text": text, "score": float(score), "short_unit": metadata.get("short_unit", False)})
    return {"generation_id": active[0], "ranking_version": ranking_version,
            "ranking_revision": RANKING_REVISIONS[ranking_version], "query": query, "hits": hits,
            "notice": "Evidence search only. Legal status is unverified; results do not establish current law."}


@app.get("/search", response_model=SearchResponse)
def search_endpoint(q: str = Query(min_length=2, max_length=500),
                    top_k: int = Query(default=5, ge=1, le=20),
                    source: str | None = Query(default=None, pattern="^(ipindia|indiacode|tkdl)$"),
                    domain: str | None = Query(default=None, max_length=80)):
    try:
        _ensure_ready()
        return search(q, top_k, source, domain)
    except Exception as exc:
        logger.exception("Search unavailable")
        reason = readiness.status().get("reason") or "search_unavailable"
        raise HTTPException(status_code=503, detail=reason) from exc


@app.get("/sources")
def sources_endpoint():
    if DEPLOYMENT.public_research:
        # Public status describes only the validated searchable snapshot. Local
        # capture paths, failed URLs/exception details and reviewer data stay local.
        try:
            snapshot = _ensure_ready()
            sources = []
            for source, domain in (("ipindia", "ipindia.gov.in"), ("indiacode", "indiacode.gov.in"), ("tkdl", "tkdl.res.in")):
                documents = [doc for doc in snapshot["documents"].values() if doc.get("source") == source]
                captures = [doc["retrieved_at"] for doc in documents if isinstance(doc.get("retrieved_at"), str)]
                sources.append({"source_id": source, "domain": domain,
                                "searchable_document_count": len(documents),
                                "last_successful_capture": max(captures) if captures else None,
                                "synchronization_enabled": False,
                                "legal_status": "unknown",
                                "search_admission": "accepted" if documents else "not_admitted"})
            return {"active_generation_id": snapshot["binding"][0], "index_available": True,
                    "sources": sources, "staged_certification": _certification_summary()}
        except (ValueError, OSError, DatabaseError) as exc:
            logger.exception("Public source status unavailable")
            raise HTTPException(status_code=503, detail="sources_unavailable") from exc
    path = ROOT / "data" / "manifests" / "manifest.sqlite"
    if not path.exists():
        raise HTTPException(status_code=503, detail="manifest unavailable")
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        policies = json.loads((ROOT / "config" / "source_policies.json").read_text(encoding="utf-8"))
        searchable = {}
        active_generation = None
        try:
            postgres = _connect()
            try:
                row = postgres.execute("SELECT value FROM app_state WHERE key='active_generation'").fetchone()
                active_generation = row[0] if row else None
                if active_generation:
                    searchable = dict(postgres.execute("SELECT metadata->>'source',count(*) FROM corpus_documents WHERE generation_id=%s GROUP BY metadata->>'source'", (active_generation,)).fetchall())
            finally:
                postgres.close()
        except (DatabaseError, ValueError, OSError):
            logger.exception("Source index status unavailable")
        rows = connection.execute("SELECT s.source_id,s.domain,s.enabled,s.sync_frequency,s.last_successful_sync,s.next_sync,s.next_check,s.next_discovery,s.last_successful_publication,"
                                  "(SELECT count(*) FROM document_manifest m WHERE m.source_id=s.source_id) archived_record_count "
                                  "FROM source_registry s ORDER BY s.source_id").fetchall()
        result = []
        for row in rows:
            failures = connection.execute("SELECT c.url,c.detail FROM sync_changes c JOIN sync_runs r USING(run_id) "
                                          "WHERE r.source_id=? AND c.change_type='FAILED' ORDER BY c.change_id DESC LIMIT 10",
                                          (row["source_id"],)).fetchall()
            review = policies.get(row["source_id"], {})
            events = connection.execute("SELECT stage,status,at,generation_id,detail FROM pipeline_events WHERE source_id=? AND status IN ('failed','partial','review_required') ORDER BY event_id DESC LIMIT 5", (row["source_id"],)).fetchall()
            report_path = ROOT / "reports" / "access" / f"{row['source_id']}.json"
            access = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
            result.append({**dict(row), "policy_approved": bool(review.get("approved")),
                           "policy_reviewed_at": review.get("reviewed_at"),
                           "access_review_status": "approved" if review.get("approved") else access.get("conclusion", "not_reviewed"),
                           "searchable_document_count": searchable.get(row["source_id"], 0),
                           "recent_review_items": [dict(item) for item in failures],
                           "pipeline_failures": [dict(item) for item in events]})
        return {"active_generation_id": active_generation, "index_available": active_generation is not None,
                "sources": result, "staged_certification": _certification_summary()}
    except (sqlite3.Error, OSError, ValueError, KeyError) as exc:
        logger.exception("Source registry unavailable")
        raise HTTPException(status_code=503, detail="sources_unavailable") from exc
    finally:
        connection.close()


@app.get("/health")
def health_endpoint():
    try:
        snapshot = _ensure_ready()
        return {"status": "ok", "generation_id": snapshot["binding"][0], "readiness": readiness.status()}
    except (ValueError, OSError, DatabaseError):
        if readiness.status()["state"] == "ready":
            logger.exception("Index readiness unavailable")
        return JSONResponse(status_code=503, content={"status": "unavailable", "detail": readiness.status().get("reason") or "index_unavailable", "readiness": readiness.status()})


@app.get("/live")
def live_endpoint():
    return {"status": "alive"}
