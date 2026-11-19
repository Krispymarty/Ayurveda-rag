"""A release-bound client for the authenticated E5 query service."""
import json
import math
import os
import re
import threading
import time

import requests

from .cloud_config import _origin
from .indexer import DIMENSIONS, MODEL_ID

DEFAULT_ORIGIN = "https://yashmit-legal-rag.hf.space"
ROUTE = "/embed/e5/v1"
MAX_RESPONSE = 16384


class EmbeddingUnavailable(ValueError):
    """The configured embedding dependency cannot currently serve requests."""


class EmbeddingBindingError(EmbeddingUnavailable):
    """The dependency returned data incompatible with this release."""


class RemoteEmbeddings:
    def __init__(self, revision, *, environ=None, session_factory=requests.Session):
        env = os.environ if environ is None else environ
        self.origin = _origin(env.get("AYURIP_E5_ORIGIN", DEFAULT_ORIGIN))
        self.key = env.get("AYURIP_E5_API_KEY", "")
        if not re.fullmatch(r"[0-9a-f]{64}", self.key):
            raise EmbeddingBindingError("embedding_credentials_invalid")
        self.revision = revision
        self.verify = env.get("REQUESTS_CA_BUNDLE") or True
        self.session_factory = session_factory
        self.sessions = threading.local()
        self.lock = threading.Lock()
        self.next_probe = 0.0
        self.failure = None
        self.binding_failed = False

    def _session(self):
        session = getattr(self.sessions, "session", None)
        if session is None:
            session = self.session_factory()
            session.trust_env = False
            self.sessions.session = session
        return session

    def _request(self, method, endpoint, **kwargs):
        response_body = bytearray()
        try:
            with self._session().request(
                method, self.origin + ROUTE + endpoint, headers={"X-AyurIP-Embedding-Key": self.key},
                timeout=(0.5, 1.15), allow_redirects=False, stream=True, verify=self.verify, **kwargs,
            ) as response:
                if response.status_code != 200:
                    raise EmbeddingUnavailable("embedding_unavailable")
                if response.headers.get("Content-Type", "").split(";", 1)[0].strip() != "application/json":
                    raise EmbeddingBindingError("embedding_binding_invalid")
                started = time.monotonic()
                for block in response.iter_content(chunk_size=4096):
                    if len(response_body) + len(block) > MAX_RESPONSE:
                        raise EmbeddingBindingError("embedding_binding_invalid")
                    if time.monotonic() - started > 1.25:
                        raise EmbeddingUnavailable("embedding_unavailable")
                    response_body.extend(block)
            value = json.loads(response_body)
        except requests.RequestException:
            raise EmbeddingUnavailable("embedding_unavailable") from None
        except (ValueError, RecursionError) as exc:
            if isinstance(exc, EmbeddingUnavailable):
                raise
            raise EmbeddingBindingError("embedding_binding_invalid") from None
        if (not isinstance(value, dict) or value.get("model_id") != MODEL_ID
                or value.get("revision") != self.revision or value.get("dimension") != DIMENSIONS):
            raise EmbeddingBindingError("embedding_binding_invalid")
        return value

    def _record_failure(self, exc):
        if isinstance(exc, EmbeddingBindingError):
            self.binding_failed = True
        self.failure = type(exc)
        self.next_probe = time.monotonic() + 3

    def assert_ready(self):
        if not self.lock.acquire(blocking=False):
            raise EmbeddingUnavailable("embedding_unavailable")
        try:
            if self.binding_failed:
                raise EmbeddingBindingError("embedding_binding_invalid")
            if time.monotonic() < self.next_probe:
                if self.failure:
                    raise EmbeddingUnavailable("embedding_unavailable")
                return
            try:
                if self._request("GET", "/ready").get("status") != "ready":
                    raise EmbeddingUnavailable("embedding_unavailable")
            except EmbeddingUnavailable as exc:
                self._record_failure(exc)
                raise
            self.failure = None
            self.next_probe = time.monotonic() + 10
        finally:
            self.lock.release()

    def query(self, query):
        self.assert_ready()
        try:
            vector = self._request("POST", "/query", json={"text": query}).get("embedding")
            if (not isinstance(vector, list) or len(vector) != DIMENSIONS
                    or any(type(v) not in (int, float) or not -1 <= v <= 1 for v in vector)
                    or not math.isclose(math.fsum(v * v for v in vector), 1.0, abs_tol=1e-4)):
                raise EmbeddingBindingError("embedding_binding_invalid")
        except EmbeddingUnavailable as exc:
            with self.lock:
                self._record_failure(exc)
            raise
        with self.lock:
            if self.binding_failed:
                raise EmbeddingBindingError("embedding_binding_invalid")
        return vector
