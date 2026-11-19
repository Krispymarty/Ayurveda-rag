"""Versioned E5 query router for the existing Docker Space."""
import asyncio
from functools import lru_cache
import hashlib
import hmac
import json
import logging
import os
from pathlib import Path
import re
import threading

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

router = APIRouter(prefix="/embed/e5/v1")
MODEL_ID = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
DIMENSION = 384
IDENTITY = {"model_id": MODEL_ID, "revision": REVISION, "dimension": DIMENSION}
MODEL_GATE = threading.Lock()
logger = logging.getLogger(__name__)


def authenticate(request):
    key = os.environ.get("AYURIP_E5_API_KEY", "")
    if not re.fullmatch(r"[0-9a-f]{64}", key):
        raise HTTPException(status_code=503, detail="embedding_unavailable")
    values = request.headers.getlist("x-ayurip-embedding-key")
    if len(values) != 1 or not values[0].isascii() or not hmac.compare_digest(values[0], key):
        raise HTTPException(status_code=401, detail="authentication_required")


def verify_snapshot(root, lock_path):
    lock = json.loads(lock_path.read_bytes())
    if (lock["model_id"], lock["revision"]) != (MODEL_ID, REVISION):
        raise ValueError("model_lock_mismatch")
    for name, digest in lock["files"].items():
        relative = Path(name)
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root) or not path.is_file():
            raise ValueError("model_file_missing")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError("model_file_mismatch")


@lru_cache(maxsize=1)
def load_encoder():
    root = Path(os.environ.get("AYURIP_E5_SNAPSHOT", "/opt/ayurip-e5")).resolve()
    verify_snapshot(root, Path(__file__).with_name("embedding-files.lock.json"))
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(root), local_files_only=True, trust_remote_code=False, device="cpu")
    if model.get_sentence_embedding_dimension() != DIMENSION:
        raise ValueError("model_dimension_mismatch")
    model.encode(["query: AyurIP startup readiness"], normalize_embeddings=True)
    return model


def model_operation(text=None):
    if not MODEL_GATE.acquire(timeout=0.3):
        raise HTTPException(status_code=503, detail="embedding_busy")
    try:
        model = load_encoder()
        if text is None:
            return {"status": "ready", **IDENTITY}
        values = model.encode(["query: " + text], normalize_embeddings=True)[0].tolist()
        return {"embedding": values, **IDENTITY}
    except Exception:
        logger.error("AyurIP E5 dependency failed")
        raise HTTPException(status_code=503, detail="embedding_unavailable") from None
    finally:
        MODEL_GATE.release()


@router.get("/ready")
def ready(request: Request):
    authenticate(request)
    return model_operation()


@router.post("/query")
async def query(request: Request):
    authenticate(request)
    if request.headers.get("content-type", "").split(";", 1)[0].lower() != "application/json":
        raise HTTPException(status_code=415, detail="json_required")
    body = bytearray()
    try:
        async with asyncio.timeout(2):
            async for block in request.stream():
                if len(body) + len(block) > 8192:
                    raise HTTPException(status_code=413, detail="body_too_large")
                body.extend(block)
        value = json.loads(body)
    except (ValueError, RecursionError, TimeoutError):
        raise HTTPException(status_code=422, detail="invalid_query") from None
    text = value.get("text") if isinstance(value, dict) and set(value) == {"text"} else None
    if not isinstance(text, str) or not 2 <= len(text) <= 500 or not text.strip() or any(
        0xD800 <= ord(char) <= 0xDFFF for char in text
    ):
        raise HTTPException(status_code=422, detail="invalid_query")
    return await run_in_threadpool(model_operation, text)
