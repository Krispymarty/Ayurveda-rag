"""Anonymous hosted extracts admission. The database holds counters only."""
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import hmac
import os
import re
import secrets
import time

import psycopg
from psycopg.conninfo import conninfo_to_dict

from .request_metrics import stage
from .groq_candidate import DEFAULT_MODEL as GROQ_MODEL
from .inference import DEFAULT_MODEL as GEMINI_MODEL, make_provider, provider_identity, read_setting

COOKIE = "__Host-ayurip-research"
COOKIE_AGE = 86400
BROWSER_LIMIT = 5
DAILY_LIMIT = 30
# This connection owns the advisory lock for the complete answer attempt.
ANSWER_LOCK = 648500306940


class AdmissionError(Exception):
    def __init__(self, reason, status, retry_after=None):
        super().__init__(reason)
        self.status, self.retry_after = status, retry_after


@lru_cache(maxsize=1)
def settings():
    secret = os.environ.get("AYURIP_ANSWER_COOKIE_KEY", "")
    dsn = os.environ.get("AYURIP_ANSWER_QUOTA_DSN", "")
    try:
        info = conninfo_to_dict(dsn)
        if (not re.fullmatch(r"[0-9a-f]{64}", secret)
                or info.get("sslmode") != "verify-full"
                or not all(info.get(k) for k in ("host", "dbname", "user"))):
            return None
    except (ValueError, psycopg.Error):
        return None
    return bytes.fromhex(secret), dsn


def providers():
    if read_setting("AYURIP_LLM_PROVIDER", "groq") != "groq":
        raise ValueError("hosted_primary_must_be_groq")
    if read_setting("AYURIP_FALLBACK_PROVIDER", "gemini") != "gemini":
        raise ValueError("hosted_fallback_must_be_gemini")
    primary = make_provider("groq")
    try:
        fallback = make_provider("gemini")
        if (provider_identity(primary)["model"], provider_identity(fallback)["model"]) != (GROQ_MODEL, GEMINI_MODEL):
            fallback.session.close()
            raise ValueError("hosted_model_binding_changed")
    except Exception:
        primary.session.close()
        raise
    return primary, fallback


def _signature(value, key):
    return hmac.digest(key, value.encode("ascii"), "sha256").hex()


def issue_cookie(key, now=None):
    issued = int(time.time() if now is None else now)
    value = f"{secrets.token_hex(16)}.{issued}"
    return f"{value}.{_signature(value, key)}"


def browser_id(cookie, key, now=None):
    if not cookie or not re.fullmatch(r"[0-9a-f]{32}\.[0-9]{1,12}\.[0-9a-f]{64}", cookie):
        return None
    identifier, issued, signature = cookie.split(".")
    current = int(time.time() if now is None else now)
    if not 0 <= current - int(issued) < COOKIE_AGE:
        return None
    if not hmac.compare_digest(signature, _signature(f"{identifier}.{issued}", key)):
        return None
    return hashlib.sha256(identifier.encode("ascii")).hexdigest()


@contextmanager
def reserve(dsn, browser):
    try:
        with stage("quota_connect"):
            connection = psycopg.connect(dsn, autocommit=True, connect_timeout=3)
    except psycopg.Error:
        raise AdmissionError("answer_quota_unavailable", 503) from None
    with connection:
        try:
            with stage("quota_admission"):
                _claim(connection, browser)
        except psycopg.Error:
            raise AdmissionError("answer_quota_unavailable", 503) from None
        yield


def _claim(connection, browser):
    connection.execute("SET statement_timeout = '3s'")
    if not connection.execute("SELECT pg_try_advisory_lock(%s)", (ANSWER_LOCK,)).fetchone()[0]:
        raise AdmissionError("service_busy", 429, 1)
    with connection.transaction():
        now = connection.execute("SELECT floor(extract(epoch FROM now()))::bigint").fetchone()[0]
        day, hour = now // 86400 * 86400, now // 3600 * 3600
        rows = ((86400, day, "site", DAILY_LIMIT), (3600, hour, browser, BROWSER_LIMIT))
        for period, start, subject, cap in rows:
            inserted = connection.execute(
                """INSERT INTO answer_quota.usage AS current (period, started, subject)
                VALUES (%s, %s, %s)
                ON CONFLICT (period, started, subject) DO UPDATE SET used = current.used + 1
                WHERE current.used < %s RETURNING used""",
                (period, start, subject, cap),
            )
            if inserted.fetchone() is None:
                # Both rows roll back when either limit is exhausted.
                raise AdmissionError("answer_daily_limit" if period == 86400 else "answer_browser_limit",
                                     429, start + period - now)
        connection.execute("DELETE FROM answer_quota.usage WHERE started < %s", (day - 86400,))
