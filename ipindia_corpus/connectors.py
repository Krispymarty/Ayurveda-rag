"""Policy-gated, bounded fetch and discovery for the three public sources."""

from __future__ import annotations

import hashlib
import os
import re
import time
import math
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

from .sync import SOURCES, capture_receipt, validate_url
from .paths import immutable_write

USER_AGENT = "AyurIP-Research/1.0 (public legal corpus; contact: operator-configured)"
BLOCKED = re.compile(r"captcha|verify you are human|access denied|cloudflare challenge", re.I)
DOCUMENT_LINK = {
    "ipindia": re.compile(r"/(?:acts/|storage/uploads/|resource/.*(?:acts|rules|guidelines|manuals))|\.pdf$", re.I),
    "indiacode": re.compile(r"/(?:bitstream/|showfile\b)|[?&]type=(?:rule|actfile)\b", re.I),
    "tkdl": re.compile(r"/tkdl/langdefault/(?:common/(?:Home|Abouttkdl|GenInfo|BioPiracy|Outcome|Milestone)[^/]*|Ayurveda/Ayu_(?:Principles|Drugs|Glossary|FAQ|Cosmetics))\.(?:asp|htm)$", re.I),
}


def official_url(source: str, value: str, base: str | None = None) -> str:
    url = urljoin(base, value) if base else value
    parsed = urlsplit(url)
    url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))
    validate_url(source, url)
    return url


@dataclass(frozen=True)
class Snapshot:
    url: str
    body: bytes | None
    content_type: str
    etag: str | None
    last_modified: str | None
    status_code: int
    capture_receipt: dict | None = None


class SourceClient:
    def __init__(self, source: str, raw_root: Path, *, session: requests.Session | None = None,
                 delay: float = 1.5, max_bytes: int = 20_000_000, permitted_prefixes=None, access_review=None):
        if source not in SOURCES:
            raise ValueError("unknown source")
        contact = os.environ.get("AYURIP_CONTACT", "").strip()
        if session is None and not contact:
            raise ValueError("AYURIP_CONTACT is required for live requests")
        self.source = source
        self.raw_root = Path(raw_root)
        self.session = session or requests.Session()
        self.user_agent = f"AyurIP-Research/1.0 ({contact})" if contact else USER_AGENT
        self.session.headers.update({"User-Agent": self.user_agent, "Accept": "text/html,application/pdf,*/*;q=0.5"})
        self.delay = max(1.0, delay)
        self.max_bytes = max_bytes
        self.last_request = 0.0
        self.robots: dict[str, RobotFileParser] = {}
        self.permitted_prefixes = permitted_prefixes
        self.access_review = access_review
        self.redirect_chain = []

    def _once(self, url: str, headers: dict[str, str] | None = None):
        wait = self.delay - (time.monotonic() - self.last_request)
        if wait > 0:
            time.sleep(wait)
        self.last_request = time.monotonic()
        return self.session.get(url, headers=headers, timeout=(10, 30), allow_redirects=False, stream=True)

    def _request(self, url: str, headers: dict[str, str] | None = None, *, check_robots: bool = True):
        current = official_url(self.source, url)
        redirects = []
        for _ in range(5):
            if check_robots and self.permitted_prefixes is not None and not any(
                    urlsplit(current).path.startswith(prefix) for prefix in self.permitted_prefixes):
                raise ValueError("path_not_approved")
            if check_robots and not self.allowed(current):
                raise ValueError("robots_disallowed")
            for attempt in range(4):
                try:
                    response = self._once(current, headers)
                except requests.RequestException as exc:
                    if attempt == 3:
                        raise ValueError("network_error") from exc
                    time.sleep(2 ** attempt)
                    continue
                if response.status_code in (429, 500, 502, 503, 504):
                    retry = response.headers.get("Retry-After", "")
                    response.close()
                    if attempt == 3:
                        raise ValueError(f"http_{response.status_code}")
                    try:
                        requested = int(retry)
                    except ValueError:
                        try:
                            requested = math.ceil((parsedate_to_datetime(retry) - datetime.now(timezone.utc)).total_seconds())
                        except (ValueError, TypeError, OverflowError):
                            requested = 0
                    delay = max(2 ** attempt, requested)
                    if delay > 60:
                        raise ValueError(f"retry_after_deferred:{delay}")
                    if not retry:
                        delay = 2 ** attempt
                    time.sleep(delay)
                    continue
                break
            if response.status_code in (301, 302, 303, 307, 308):
                target = response.headers.get("Location")
                response.close()
                if not target:
                    raise ValueError("redirect_without_location")
                target = official_url(self.source, target, current)
                redirects.append({"from_url": current, "to_url": target, "status": response.status_code})
                current = target
                continue
            if response.status_code in (401, 403):
                response.close()
                raise ValueError("manual_review_required: access_restricted")
            if response.status_code >= 400 and response.status_code != 404:
                response.close()
                raise ValueError(f"http_{response.status_code}")
            self.redirect_chain = redirects
            return current, response
        raise ValueError("too_many_redirects")

    def _body(self, response, limit: int) -> bytes:
        try:
            length = int(response.headers.get("Content-Length") or 0)
        except ValueError as exc:
            response.close()
            raise ValueError("invalid_content_length") from exc
        if length > limit:
            response.close()
            raise ValueError("manual_review_required: oversized_resource")
        pieces, size = [], 0
        try:
            for piece in response.iter_content(64 * 1024):
                size += len(piece)
                if size > limit:
                    raise ValueError("manual_review_required: oversized_resource")
                pieces.append(piece)
        except requests.RequestException as exc:
            raise ValueError("network_error_during_body") from exc
        finally:
            response.close()
        return b"".join(pieces)

    def allowed(self, url: str) -> bool:
        host = urlsplit(url).hostname or ""
        if host not in self.robots:
            robots_url = official_url(self.source, f"https://{host}/robots.txt")
            _, response = self._request(robots_url, check_robots=False)
            if response.status_code == 404:
                response.close()
                value = ""
            else:
                raw = self._body(response, 200_000)
                if not raw or b"<html" in raw[:500].lower():
                    raise ValueError("manual_review_required: robots_unverified")
                value = raw.decode("utf-8-sig", errors="replace")
                digest = hashlib.sha256(raw).hexdigest()
                path = self.raw_root / "robots" / self.source / f"{digest}.txt"
                path.parent.mkdir(parents=True, exist_ok=True)
                immutable_write(path, raw)
            parser = RobotFileParser()
            parser.parse(value.splitlines())
            self.robots[host] = parser
            crawl_delay = parser.crawl_delay(self.user_agent)
            if crawl_delay is not None:
                self.delay = max(self.delay, float(crawl_delay))
        return self.robots[host].can_fetch(self.user_agent, url)

    def get(self, url: str, *, etag: str | None = None, last_modified: str | None = None) -> Snapshot:
        headers = {}
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified
        current, response = self._request(url, headers)
        content_type = response.headers.get("Content-Type", "").split(";")[0].lower()
        new_etag = response.headers.get("ETag") or etag
        new_modified = response.headers.get("Last-Modified") or last_modified
        if response.status_code == 304:
            response.close()
            return Snapshot(current, None, content_type, new_etag, new_modified, 304)
        if response.status_code == 404:
            response.close()
            raise ValueError("http_404")
        body = self._body(response, self.max_bytes)
        if not body or BLOCKED.search(body[:5000].decode("utf-8", errors="ignore")):
            raise ValueError("manual_review_required: empty_or_challenge")
        accepted_types = {"text/html", "application/xhtml+xml", "application/pdf"}
        if self.source == "indiacode" and urlsplit(current).path.startswith(("/server/api", "/assets/config.json")):
            accepted_types |= {"application/json", "application/hal+json"}
        if content_type not in accepted_types:
            if body.startswith(b"%PDF-"):
                content_type = "application/pdf"
            elif re.search(br"<\s*(?:!doctype\s+html|html\b)", body[:512], re.I):
                content_type = "text/html"
        if content_type not in accepted_types:
            raise ValueError("manual_review_required: unsupported_content_type")
        if content_type in {"text/html", "application/xhtml+xml"}:
            soup = BeautifulSoup(body, "html.parser")
            title = soup.title.get_text(" ", strip=True) if soup.title else ""
            visible = soup.get_text(" ", strip=True)
            if re.search(r"^(?:error|access denied|session (?:expired|timeout)|page not found|internal server error)\b", title, re.I) or (
                    len(visible) < 300 and re.search(r"session\s*(?:expired|timeout)|page not found|internal server error", visible, re.I)):
                raise ValueError("manual_review_required: error_or_session_page")
        receipt = capture_receipt(self.raw_root, self.source, requested_url=official_url(self.source, url),
                                  final_url=current, body=body, content_type=content_type,
                                  redirects=self.redirect_chain, etag=new_etag,
                                  last_modified=new_modified, access_review=self.access_review)
        return Snapshot(current, body, content_type, new_etag, new_modified, 200, receipt)

    def get_json(self, url: str) -> tuple[dict, Snapshot]:
        import json
        if self.source != "indiacode":
            raise ValueError("JSON discovery is restricted to India Code")
        snapshot = self.get(url)
        if snapshot.body is None or snapshot.content_type not in {"application/json", "application/hal+json"}:
            raise ValueError("expected public JSON metadata")
        try:
            data = json.loads(snapshot.body)
        except (ValueError, UnicodeError) as exc:
            raise ValueError("invalid_json_metadata") from exc
        if not isinstance(data, dict):
            raise ValueError("metadata_must_be_object")
        return data, snapshot


def discover_links(source: str, base: str, body: bytes, limit: int) -> list[str]:
    """Only follow in-scope public document links from a configured seed page."""
    if source not in DOCUMENT_LINK:
        raise ValueError("unknown source")
    soup = BeautifulSoup(body, "html.parser")
    result: list[str] = []
    for anchor in soup.find_all("a", href=True):
        try:
            url = official_url(source, anchor["href"], base)
        except ValueError:
            continue
        parsed = urlsplit(url)
        if source == "tkdl" and ("login" in parsed.path.lower() or "search" in parsed.path.lower()):
            continue
        if DOCUMENT_LINK[source].search(parsed.path) or (source == "indiacode" and DOCUMENT_LINK[source].search(parsed.path + "?" + parsed.query)):
            if url not in result:
                result.append(url)
        if len(result) >= limit:
            break
    return result
