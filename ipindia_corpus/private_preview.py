"""Explicit loopback-only evaluation server; no public answer gate changes."""
import argparse
import hashlib
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import requests

from .corpus_certification import official_public_url
from .http_security import configure_security
from .indexer import generation
from .paths import archive_path
from .reliable_answers import SpanProvider, answer, configuration
from .search_api import AnswerRequest


class ArchiveVerifier:
    def __init__(self, root):
        self.root = Path(root)
        self.gid, docs, chunks = generation(self.root, generation_id="20261014T204940358650Z")
        self.docs = {d["document_id"]: d for d in docs}
        self.chunks = {c["chunk_id"]: c for c in chunks}

    def __call__(self, hit, gid):
        if gid != self.gid:
            raise ValueError("preview_generation_changed")
        expected = self.chunks.get(hit.get("chunk_id"))
        if not expected:
            raise ValueError("unpublished_chunk")
        for key in ("document_id", "source", "source_url", "raw_sha256", "text", "section", "rule", "page_start", "page_end"):
            if hit.get(key) != expected.get(key):
                raise ValueError("citation_mismatch")
        doc = self.docs[hit["document_id"]]
        if hit.get("title") != (expected.get("document_title") or doc["title"]):
            raise ValueError("title_mismatch")
        if not official_public_url(hit["source"], hit["source_url"]) or hit.get("status") != "unknown":
            raise ValueError("unverified_source_or_status")
        if hashlib.sha256(hit["text"].encode()).hexdigest() != expected["content_hash"]:
            raise ValueError("text_changed")
        if hashlib.sha256(archive_path(self.root, doc["raw_path"]).read_bytes()).hexdigest() != doc["sha256"]:
            raise ValueError("archive_changed")


def create_app(root, *, evaluation=False, search_port=8000):
    if search_port not in (8000, 8002):
        raise ValueError("local_search_port_required")
    base_url = f"http://127.0.0.1:{search_port}"
    if evaluation:
        # Explicit private benchmark process only; the main API is unaffected.
        os.environ["AYURIP_RATE_LIMIT_ENABLED"] = "0"
    app = FastAPI(title="AyurIP private reliability evaluation", docs_url=None, redoc_url=None)
    configure_security(app)
    verifier = ArchiveVerifier(root)
    session = requests.Session()
    session.trust_env = False
    provider = SpanProvider()

    def search(question, top_k=5):
        response = session.get(base_url + "/search", params={"q": question, "top_k": top_k}, timeout=(5, 10))
        response.raise_for_status()
        return response.json()

    @app.get("/health")
    def health():
        try:
            response = session.get(base_url + "/health", timeout=(2, 3))
            provider.client.verify_identity({"model": configuration()["model"], "model_digest": configuration()["model_digest"]})
            return JSONResponse(status_code=response.status_code, content={"search": response.json(), "candidate": configuration(), "evaluation_mode": evaluation, "search_port": search_port})
        except Exception:
            return JSONResponse(status_code=503, content={"reason": "preview_dependency_unavailable"})

    @app.post("/answer")
    def respond(request: AnswerRequest):
        if request.source is not None or request.domain is not None or request.top_k != 5:
            return JSONResponse(status_code=400, content={"reason": "preview_uses_unfiltered_top_five"})
        result = answer(request.question, search, verifier, provider=provider)
        return JSONResponse(status_code=503 if result["status"] == "unavailable" else 200, content=result)

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", required=True)
    parser.add_argument("--evaluation", action="store_true", help="Disable only this process's rate limiter for serial measurements")
    parser.add_argument("--search-port", type=int, choices=(8000, 8002), default=8000)
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(create_app(Path(__file__).resolve().parents[1], evaluation=args.evaluation, search_port=args.search_port), host="127.0.0.1", port=8001, proxy_headers=False)
