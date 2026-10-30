from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import io
import json
import re
import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

BASE = "https://ipindia.gov.in/"
HOSTS = {"ipindia.gov.in", "www.ipindia.gov.in"}
USER_AGENT = "AyurvedaIPLegalCorpus/0.1 (+public legal research; respectful crawler)"
SEEDS = [
    BASE,
    "https://ipindia.gov.in/pages/home/acts-rules-ipr",
    "https://ipindia.gov.in/resource/patents-resources-act",
    "https://ipindia.gov.in/resource/patents-resources-rules",
    "https://ipindia.gov.in/resource/patents-resources-manuals",
    "https://ipindia.gov.in/resource/patents-resources-guidelines",
    "https://ipindia.gov.in/acts/patent-act-1970/section-3",
    "https://ipindia.gov.in/trade-marks-resources-act",
    "https://ipindia.gov.in/trade-marks-resources-rules",
    "https://ipindia.gov.in/geographical-indications-resources-act",
    "https://ipindia.gov.in/geographical-indications-resources-rules",
    "https://ipindia.gov.in/designs-resources-act",
    "https://ipindia.gov.in/designs-resources-rules",
]
TOPIC_PATTERNS = {
    "AYUSH": r"\bayush\b|\bayurved\w*\b|\bsiddha\b|\bunani\b",
    "Traditional Knowledge": r"traditional knowledge|\btkdl\b|traditional medicinal knowledge",
    "Biological Resources": r"biological (?:material|resource)|genetic resource|medicinal plant|\bherbal\b",
    "Patent Section 3": r"section\s*3\s*\(?[cdeijp]\)?|\b3\s*\([cdeijp]\)",
    "Access and Benefit Sharing": r"access and benefit sharing|\bABS\b|biodiversity act",
}
CONTROLLED_TOPIC_PATTERNS = {
    "ayurveda": r"\bayurved\w*\b",
    "ayush": r"\bayush\b",
    "traditional_knowledge": r"traditional knowledge",
    "tkdl": r"\btkdl\b|traditional knowledge digital library",
    "biological_material": r"biological material",
    "biological_resource": r"biological resource",
    "medicinal_plant": r"medicinal plants?",
    "herbal": r"\bherbal\b",
    "patentability": r"\bpatentability\b|patentable",
    "prior_art": r"prior art",
    "novelty": r"\bnovelty\b|\bnovel\b",
    "inventive_step": r"inventive step",
    "section_3c": r"\b(?:section\s*)?3\s*\(c\)",
    "section_3d": r"\b(?:section\s*)?3\s*\(d\)",
    "section_3e": r"\b(?:section\s*)?3\s*\(e\)",
    "section_3i": r"\b(?:section\s*)?3\s*\(i\)",
    "section_3j": r"\b(?:section\s*)?3\s*\(j\)",
    "section_3p": r"\b(?:section\s*)?3\s*\(p\)",
    "mere_admixture": r"mere admixture",
    "natural_substance": r"natural substances?",
    "treatment_method": r"method of treatment|treatment method",
    "trademark": r"trade[ -]?marks?",
    "gi": r"geographical indications?",
    "design": r"\bdesigns?\b",
}
POSITIVE = re.compile(
    r"patent|ayush|ayurved|traditional|tkdl|trade[ -]?mark|geographical|"
    r"\bGI\b|design|manual|guideline|\brules?\b|\bacts?\b|notification|"
    r"amendment|\bforms?\b|filing|classification|search|\bfaq\b|circular|public.notice",
    re.I,
)
NEGATIVE = re.compile(
    r"tender|recruit|vacanc|staff|roster|seniority|cause.list|hearing|"
    r"application.status|e-?filing|login|register.of|journal|newsletter|"
    r"news|event|gallery|contact|grievance|dashboard|annual.report|"
    r"subscription|agent.exam|facilitator|certificate.issued|"
    r"patents?.granted|patentgranted|grant.list|individual.applications?",
    re.I,
)
FILE_TYPES = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".csv"}
DOC_TYPES = [
    ("Amendment", r"amendment|amends?"),
    ("Act", r"\bact\b|\/acts\/"),
    ("Rules", r"\brules?\b"),
    ("Manual", r"manual"),
    ("Guidelines", r"guideline"),
    ("Form", r"\bforms?\b"),
    ("Notification", r"notification|circular|public.notice"),
    ("FAQ", r"\bfaq\b"),
    ("Procedure", r"procedure|filing|process|classification|search"),
]
SECTION = re.compile(r"^\s*(?:SECTION\s+|Section\s+|S\.\s*)(\d+[A-Z]?)\s*[.\-:]?\s*(.*)$")
RULE = re.compile(r"^\s*(?:RULE\s+|Rule\s+)(\d+[A-Z]?)\s*[.\-:]?\s*(.*)$")
NUMBERED_RULE = re.compile(r"^\s*(\d+[A-Z]?)\.\s+([A-Z][^\n]{3,120})$")
CHAPTER = re.compile(r"^\s*(?:CHAPTER\s+)([IVXLCDM\d]+)\b\s*(.*)$", re.I)
CLAUSE = re.compile(r"^\s*\(([a-z])\)\s+(.+)$")
SUBSECTION = re.compile(r"^\s*\((\d+)\)\s+(.+)$")
BLOCKED_MARKERS = re.compile(r"captcha|verify you are human|access denied|bot detection|cloudflare challenge", re.I)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(url: str, base: str = BASE) -> str | None:
    parsed = urlsplit(urljoin(base, url))
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or host not in HOSTS or parsed.username or parsed.password or parsed.port:
        return None
    query = urlencode([(k, v) for k, v in parse_qsl(parsed.query) if not k.lower().startswith(("utm_", "fbclid", "gclid"))])
    return urlunsplit(("https", host, parsed.path or "/", query, ""))


def score(url: str, label: str = "") -> int:
    candidate = f"{urlsplit(url).path} {label}"
    if NEGATIVE.search(candidate):
        return -1
    matches = POSITIVE.findall(candidate)
    value = len(matches) + (3 if re.search(r"act|rule|manual|guideline|ayush|traditional", candidate, re.I) else 0)
    if re.search(r"ayush|ayurved|traditional knowledge|\btkdl\b", candidate, re.I):
        value += 12
    elif re.search(r"biological (?:material|resource)|medicinal plant", candidate, re.I):
        value += 7
    if Path(urlsplit(url).path).suffix.lower() in FILE_TYPES:
        value += 3
    if re.search(r"\bdraft\b|inviting comments", candidate, re.I):
        value -= 5
    if re.search(r"/acts/patent-act-1970/section-(?!3(?:\D|$))", url, re.I):
        value -= 2
    return value


def domains(value: str) -> list[str]:
    s = value.lower()
    result = []
    for name, pattern in [
        ("Patent", r"patent|inpass|section\s*3"),
        ("Trademark", r"trade[ -]?mark|\btmr\b"),
        ("GI", r"geographical indication|\bgi\b"),
        ("Design", r"\bdesigns?\b"),
        ("Traditional Knowledge", r"traditional knowledge|\btkdl\b"),
        ("AYUSH", r"ayush|ayurved|siddha|unani"),
    ]:
        if re.search(pattern, s):
            result.append(name)
    return result or ["General IP"]


def classify_domains(title: str, url: str, source_url: str, content: str = "") -> tuple[str, list[str]]:
    title_domains = domains(title)
    url_domains = domains(url)
    source_domains = domains(source_url)
    content_domains = domains(content[:2000]) if content else ["General IP"]
    for preferred in ("Trademark", "GI", "Design", "Patent"):
        if preferred in title_domains:
            primary = preferred
            break
    else:
        primary = next((d for values in (url_domains, source_domains, title_domains, content_domains) for d in values if d != "General IP"), "General IP")
    related = []
    for value in title_domains + url_domains + source_domains + content_domains:
        if value not in (primary, "General IP") and value not in related:
            related.append(value)
    return primary, related


def doc_type(value: str) -> str:
    for name, pattern in DOC_TYPES:
        if re.search(pattern, value, re.I):
            return name
    return "Guidance"


def topics(value: str) -> list[str]:
    return [name for name, pattern in TOPIC_PATTERNS.items() if re.search(pattern, value, re.I)]


def controlled_topics(content: str) -> list[str]:
    """Apply controlled tags only to extracted content, never URL or filename."""
    return [name for name, pattern in CONTROLLED_TOPIC_PATTERNS.items() if re.search(pattern, content, re.I)]


def log_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


@dataclass
class Fetch:
    url: str
    data: bytes
    content_type: str
    retrieved_at: str
    path: Path


@dataclass
class ResponseSnapshot:
    url: str
    data: bytes | None
    content_type: str
    checked_at: str
    etag: str | None
    last_modified: str | None
    status_code: int


class Client:
    def __init__(self, out: Path, delay: float = 1.5, max_bytes: int = 20_000_000):
        self.out, self.delay, self.max_bytes = out, max(1.0, delay), max_bytes
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html,application/pdf,application/xml,text/xml,*/*;q=0.5"})
        self.last_request = 0.0
        self.robots: dict[str, RobotFileParser] = {}
        self.robots_text: dict[str, str] = {}
        self.robots_state: dict[str, str] = {}

    def _request(self, url: str, check_robots: bool = True, headers: dict[str, str] | None = None) -> requests.Response:
        current = url
        for _ in range(5):
            if canonical(current) is None:
                raise ValueError("external_redirect")
            if check_robots and not self.allowed(current):
                raise ValueError("robots_disallowed")
            for attempt in range(4):
                wait = self.delay - (time.monotonic() - self.last_request)
                if wait > 0:
                    time.sleep(wait)
                self.last_request = time.monotonic()
                try:
                    response = self.session.get(current, timeout=(10, 30), allow_redirects=False, stream=True, headers=headers)
                except requests.RequestException as exc:
                    if attempt == 3:
                        raise ValueError(f"network_error: {exc.__class__.__name__}") from exc
                    time.sleep(2 ** attempt)
                    continue
                if response.status_code in (429, 500, 502, 503, 504):
                    response.close()
                    if attempt == 3:
                        raise ValueError(f"http_{response.status_code}")
                    retry = response.headers.get("Retry-After", "")
                    try:
                        seconds = max(2 ** attempt, min(60, int(retry)))
                    except ValueError:
                        seconds = 2 ** attempt
                    time.sleep(seconds)
                    continue
                break
            if response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get("Location")
                response.close()
                if not location:
                    raise ValueError("redirect_without_location")
                current = urljoin(current, location)
                continue
            if response.status_code in (401, 403):
                response.close()
                raise ValueError("manual_review_required: access_restricted")
            if response.status_code >= 400:
                response.close()
                raise ValueError(f"http_{response.status_code}")
            return response
        raise ValueError("too_many_redirects")

    def load_robots(self, host: str) -> RobotFileParser:
        if host in self.robots:
            return self.robots[host]
        url = f"https://{host}/robots.txt"
        try:
            response = self._request(url, check_robots=False)
            pieces, size = [], 0
            try:
                for piece in response.iter_content(16 * 1024):
                    size += len(piece)
                    if size > 200_000:
                        raise ValueError("robots_too_large")
                    pieces.append(piece)
            finally:
                response.close()
            raw = b"".join(pieces)
            if not raw or b"<html" in raw[:500].lower():
                raise ValueError("robots_not_plain_text")
            value = raw.decode("utf-8-sig", errors="replace")
            archive = self.out / "robots" / f"{digest(raw)}.txt"
            archive.parent.mkdir(parents=True, exist_ok=True)
            if archive.exists() and digest(archive.read_bytes()) != digest(raw):
                raise ValueError("robots_archive_hash_mismatch")
            if not archive.exists():
                archive.write_bytes(raw)
            parser = RobotFileParser()
            parser.parse(value.splitlines())
            self.robots[host] = parser
            self.robots_text[host] = value
            self.robots_state[host] = f"retrieved and parsed; sha256={digest(raw)}"
            crawl_delay = parser.crawl_delay(USER_AGENT)
            if crawl_delay is not None:
                self.delay = max(self.delay, float(crawl_delay))
            return parser
        except ValueError as exc:
            if str(exc) == "http_404":
                parser = RobotFileParser()
                parser.parse([])
                self.robots[host] = parser
                self.robots_text[host] = ""
                self.robots_state[host] = "not published (HTTP 404)"
                return parser
            raise ValueError(f"manual_review_required: robots_unverified: {exc}") from exc

    def allowed(self, url: str) -> bool:
        host = urlsplit(url).hostname or ""
        return self.load_robots(host).can_fetch(USER_AGENT, url)

    def fetch(self, url: str) -> Fetch:
        url = canonical(url) or ""
        if not url:
            raise ValueError("external_url")
        if not self.allowed(url):
            raise ValueError("robots_disallowed")
        key = digest(url.encode())
        meta = self.out / "cache" / f"{key}.json"
        raw = self.out / "cache" / f"{key}.bin"
        if raw.exists() and meta.exists():
            cached = json.loads(meta.read_text(encoding="utf-8"))
            return Fetch(url, raw.read_bytes(), cached["content_type"], cached["retrieved_at"], raw)
        response = self._request(url)
        content_type = response.headers.get("Content-Type", "").split(";")[0].lower()
        length = int(response.headers.get("Content-Length") or 0)
        if length > self.max_bytes:
            response.close()
            raise ValueError("manual_review_required: oversized_resource")
        chunks, size = [], 0
        try:
            for chunk in response.iter_content(64 * 1024):
                size += len(chunk)
                if size > self.max_bytes:
                    raise ValueError("manual_review_required: oversized_resource")
                chunks.append(chunk)
        finally:
            response.close()
        data = b"".join(chunks)
        if BLOCKED_MARKERS.search(data[:5000].decode("utf-8", errors="ignore")):
            raise ValueError("manual_review_required: challenge_detected")
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(data)
        timestamp = utcnow()
        meta.write_text(json.dumps({"url": url, "content_type": content_type, "retrieved_at": timestamp, "sha256": digest(data)}), encoding="utf-8")
        return Fetch(url, data, content_type, timestamp, raw)

    def revalidate(self, url: str, etag: str | None = None, last_modified: str | None = None) -> ResponseSnapshot:
        """Live conditional check; never overwrites the existing URL cache."""
        url = canonical(url) or ""
        if not url:
            raise ValueError("external_url")
        headers = {}
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified
        response = self._request(url, headers=headers)
        checked_at = utcnow()
        content_type = response.headers.get("Content-Type", "").split(";")[0].lower()
        response_etag = response.headers.get("ETag") or etag
        response_last_modified = response.headers.get("Last-Modified") or last_modified
        if response.status_code == 304:
            response.close()
            return ResponseSnapshot(url, None, content_type, checked_at, response_etag, response_last_modified, 304)
        size = 0
        pieces = []
        try:
            for piece in response.iter_content(64 * 1024):
                size += len(piece)
                if size > self.max_bytes:
                    raise ValueError("manual_review_required: oversized_resource")
                pieces.append(piece)
        finally:
            response.close()
        data = b"".join(pieces)
        if BLOCKED_MARKERS.search(data[:5000].decode("utf-8", errors="ignore")):
            raise ValueError("manual_review_required: challenge_detected")
        return ResponseSnapshot(url, data, content_type, checked_at, response_etag, response_last_modified, response.status_code)


def html_content(data: bytes) -> tuple[str, str, list[tuple[str, str]]]:
    soup = BeautifulSoup(data, "html.parser")
    title = (soup.title.get_text(" ", strip=True) if soup.title else "").strip()
    links = []
    for anchor in soup.find_all("a", href=True):
        label = anchor.get_text(" ", strip=True)
        row = anchor.find_parent("tr")
        if row and (not label or re.match(r"^(?:Format:|\(?\d+(?:\.\d+)?\s*(?:KB|MB)\)?)", label, re.I)):
            cells = row.find_all(["td", "th"], recursive=False)
            if len(cells) >= 2:
                label = cells[1].get_text(" ", strip=True)
        links.append((anchor.get("href", ""), re.sub(r"\s+", " ", label).strip()))
    for tag in soup.select("script,style,nav,footer,header,form,aside,button,noscript,iframe,object,embed,template,[hidden],[aria-hidden='true'],[style*='display:none'],[style*='display: none'],[style*='visibility:hidden'],[style*='visibility: hidden']"):
        tag.decompose()
    legal = soup.select_one(".actsAreaBox")
    if legal:
        chapter = legal.select_one("h2 span")
        heading = legal.select_one(".chapterHeadinBox")
        body = legal.select_one(".section-body") or legal.select_one(".chapterDetailsAreaBox")
        lines = []
        if chapter:
            lines.append(re.sub(r"\s+", " ", chapter.get_text(" ", strip=True)))
        if heading:
            lines.append(re.sub(r"\s+", " ", heading.get_text(" ", strip=True)))
        if body:
            for tag in body.find_all(["p", "li", "tr"], recursive=True):
                if tag.find_parent(["p", "li", "tr"]):
                    continue
                value = re.sub(r"\s+", " ", tag.get_text(" ", strip=True))
                if value:
                    lines.append(value)
        if lines:
            return title, "\n".join(lines), links
    root = soup.select_one("main, article, [role=main], .main-content, .content-area") or soup.body or soup
    heading = root.find("h1")
    if heading:
        heading_text = heading.get_text(" ", strip=True)
        if heading_text and heading_text.lower() not in ("home", "page title") and (not title or "IP India" in title):
            title = heading_text
    lines = []
    for tag in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "tr", "pre"]):
        if tag.find_parent(["p", "li", "tr", "pre"]):
            continue
        value = tag.get_text(" ", strip=True)
        if value and (not lines or value != lines[-1]):
            lines.append(value)
    return title, "\n".join(lines), links


def extract(fetch: Fetch) -> tuple[str, str, list[tuple[str, str]]]:
    suffix = Path(urlsplit(fetch.url).path).suffix.lower()
    kind = fetch.content_type
    if suffix == ".pdf" or kind == "application/pdf":
        import fitz
        try:
            document = fitz.open(stream=fetch.data, filetype="pdf")
        except Exception as exc:
            raise ValueError("manual_review_required: invalid_pdf") from exc
        page_texts = [page.get_text(sort=True) for page in document]
        if sum(len(page.strip()) for page in page_texts) < 100:
            raise ValueError("manual_review_required: pdf_has_no_extractable_text")
        text = "\n".join(f"[Page {i + 1}]\n{page}" for i, page in enumerate(page_texts))
        return document.metadata.get("title") or Path(urlsplit(fetch.url).path).stem, text, []
    if kind in ("text/html", "application/xhtml+xml") or suffix in (".htm", ".html") or (not suffix and kind.startswith("text/")):
        return html_content(fetch.data)
    if suffix == ".docx":
        from docx import Document
        document = Document(io.BytesIO(fetch.data))
        lines = [p.text for p in document.paragraphs if p.text.strip()]
        for table in document.tables:
            lines.extend(" | ".join(cell.text for cell in row.cells) for row in table.rows)
        return Path(urlsplit(fetch.url).path).stem, "\n".join(lines), []
    if suffix == ".doc":
        antiword = shutil.which("antiword")
        if not antiword:
            raise ValueError("manual_review_required: antiword_unavailable_for_doc")
        with tempfile.TemporaryDirectory(dir=fetch.path.parent) as directory:
            path = Path(directory) / "document.doc"
            path.write_bytes(fetch.data)
            result = subprocess.run([antiword, str(path)], capture_output=True, timeout=60, check=False)
            if result.returncode:
                raise ValueError("manual_review_required: doc_extraction_failed")
            return Path(urlsplit(fetch.url).path).stem, result.stdout.decode("utf-8", errors="replace"), []
    if suffix == ".xlsx":
        from openpyxl import load_workbook
        workbook = load_workbook(io.BytesIO(fetch.data), read_only=True, data_only=True)
        rows = []
        for sheet in workbook:
            rows.append(f"# {sheet.title}")
            for cells in sheet.iter_rows(values_only=True):
                value = " | ".join(str(cell) if cell is not None else "" for cell in cells).strip(" |")
                if value:
                    rows.append(value)
        workbook.close()
        return Path(urlsplit(fetch.url).path).stem, "\n".join(rows), []
    if suffix == ".csv":
        value = fetch.data.decode("utf-8-sig", errors="replace")
        rows = [" | ".join(row) for row in csv.reader(io.StringIO(value))]
        return Path(urlsplit(fetch.url).path).stem, "\n".join(rows), []
    if suffix == ".xls":
        import xlrd
        workbook = xlrd.open_workbook(file_contents=fetch.data)
        lines = []
        for sheet in workbook.sheets():
            lines.append(f"# {sheet.name}")
            for row in range(sheet.nrows):
                cells = " | ".join(str(value) for value in sheet.row_values(row)).strip(" |")
                if cells:
                    lines.append(cells)
        return Path(urlsplit(fetch.url).path).stem, "\n".join(lines), []
    raise ValueError("manual_review_required: unsupported_format")


def structure_chunks(text: str, document: dict) -> list[dict]:
    """Group by legal headings; preserve sections/rules and Section 3 clauses."""
    state = {"chapter": "", "section": "", "subsection": "", "rule": "", "heading": ""}
    chunks: list[dict] = []
    lines: list[tuple[str, int | None]] = []
    current_page: int | None = None

    def flush() -> None:
        if not lines:
            return
        parts = []
        current = []
        current_size = 0
        for line, page in lines:
            if current and current_size + len(line) > 6000:
                parts.append(current)
                current, current_size = [], 0
            current.append((line, page))
            current_size += len(line)
        if current:
            parts.append(current)
        lines.clear()
        for part in parts:
            body = "\n".join(line for line, _ in part)
            pages = [page for _, page in part if page is not None]
            ordinal = len(chunks)
            chunk_id = f"{document['document_id']}_{ordinal:05d}_{digest(body.encode())[:8]}"
            chunks.append({
                "chunk_id": chunk_id, "document_id": document["document_id"],
                "document_title": document["title"], "title": document["title"], "ip_domain": document["ip_domain"],
                "legal_domain": document["ip_domain"], "source": document.get("source", "ipindia"), "status": document.get("status", "unknown"),
                "jurisdiction": "India", "authority": document["authority"],
                "document_type": document["document_type"], **state,
                "page_start": min(pages) if pages else None, "page_end": max(pages) if pages else None,
                "effective_from": document["effective_from"], "effective_to": document["effective_to"],
                "source_url": document["source_url"], "official_source": True,
                "content_hash": digest(body.encode()), "topics": controlled_topics(body), "text": body,
            })

    text = re.sub(r"(?<!\w)(\([a-z]\))\s+", r"\n\1 ", text) if document["document_type"] == "Act" else text
    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        page_marker = re.fullmatch(r"\[Page (\d+)\]", line)
        if page_marker:
            flush()
            current_page = int(page_marker.group(1))
            continue
        if not line or re.fullmatch(r"\d{1,4}", line):
            continue
        chapter = CHAPTER.match(line)
        section = SECTION.match(line)
        rule = RULE.match(line)
        clause = CLAUSE.match(line)
        subsection = SUBSECTION.match(line)
        if chapter:
            flush()
            state.update(chapter=chapter.group(1), section="", subsection="", rule="", heading=chapter.group(2))
            continue
        elif section:
            flush()
            state.update(section=section.group(1), subsection="", rule="", heading=section.group(2))
        elif rule:
            flush()
            state.update(rule=rule.group(1), section="", subsection="", heading=rule.group(2))
        elif document["document_type"] == "Rules" and NUMBERED_RULE.match(line):
            numbered_rule = NUMBERED_RULE.match(line)
            flush()
            state.update(rule=numbered_rule.group(1), section="", subsection="", heading=numbered_rule.group(2))
        elif document.get("source") == "indiacode" and document["document_type"] == "Act" and re.match(r"^\d+[A-Za-z]?\.\s+.+[—–]", line):
            numbered = re.match(r"^(\d+[A-Za-z]?)\.\s+(.+?)[—–]", line)
            flush()
            state.update(section=numbered.group(1), rule="", subsection="", heading=numbered.group(2))
        elif clause and state["section"] == "3":
            flush()
            state["subsection"] = clause.group(1)
        elif subsection and (state["section"] or state["rule"]):
            flush()
            state["subsection"] = subsection.group(1)
        elif re.match(r"^(?:Provided(?: further)? that|Explanation\b)", line, re.I) and (state["section"] or state["rule"]):
            flush()
            state["heading"] = "Proviso" if line.lower().startswith("provided") else "Explanation"
        elif re.match(r"^\d+(?:\.\d+){0,3}\s+[A-Z][A-Za-z].{2,100}$", line) and document["document_type"] in ("Manual", "Guidelines", "Guidance"):
            flush()
            state.update(heading=line, subsection="")
        elif line.startswith("# ") or (document["document_type"] in ("Manual", "Guidelines", "Guidance") and len(line) < 130 and line.isupper() and len(line.split()) >= 2):
            flush()
            state.update(heading=line.lstrip("# "), subsection="")
        lines.append((line, current_page))
    flush()
    return chunks


def build_document(fetch: Fetch, title: str, content: str, source_url: str) -> dict:
    title = normalized_document_title(title, fetch.url)
    identifier = digest(fetch.url.encode())[:20]
    primary_domain, related_domains = classify_domains(title, fetch.url, source_url, content)
    security_flags = []
    if fetch.content_type in ("text/html", "application/xhtml+xml"):
        raw_prefix = fetch.data.decode("utf-8", errors="replace")
        if re.search(r"<\s*(?:script|iframe|object|embed)\b|javascript:|\bon(?:click|load|error|submit)\s*=", raw_prefix, re.I):
            security_flags.append("active_html_removed_from_extracted_text")
        if re.search(r"\bhidden\b|aria-hidden\s*=|display\s*:\s*none|visibility\s*:\s*hidden", raw_prefix, re.I):
            security_flags.append("hidden_html_removed_from_extracted_text")
    return {
        "document_id": identifier, "canonical_id": identifier, "source": "ipindia", "source_domain": urlsplit(fetch.url).hostname,
        "title": title or Path(urlsplit(fetch.url).path).name,
        "authority": "IP India / CGPDTM", "jurisdiction": "India",
        "ip_domain": primary_domain, "legal_domain": primary_domain, "related_ip_domains": related_domains,
        "document_type": doc_type(title) if doc_type(title) != "Guidance" else doc_type(fetch.url),
        "publication_date": None, "effective_from": None, "effective_to": None,
        "notification_number": None, "version": None, "status": "unknown",
        "amends": [], "amended_by": [], "supersedes": [], "superseded_by": [],
        "source_url": fetch.url, "canonical_url": fetch.url, "download_url": fetch.url,
        "discovered_from_url": source_url, "alternate_urls": [],
        "retrieved_at": fetch.retrieved_at, "last_checked_at": fetch.retrieved_at,
        "etag": None, "last_modified": None, "official_source": True,
        "content_hash": digest(fetch.data), "sha256": digest(fetch.data), "language": language(content), "topics": controlled_topics(content),
        "raw_file_path": str(fetch.path.resolve()), "raw_path": str(fetch.path.resolve()),
        "metadata_review_required": True, "security_flags": security_flags,
        "contains_active_content": "active_html_removed_from_extracted_text" in security_flags,
    }


def language(content: str) -> str:
    sample = content[:20_000]
    devanagari = sum("\u0900" <= ch <= "\u097f" for ch in sample)
    latin = sum("a" <= ch.lower() <= "z" for ch in sample)
    if devanagari > 100 and latin > 100:
        return "mul"
    if devanagari > 100:
        return "hi"
    if latin > 100:
        return "en"
    return "und"


def is_index_page(url: str) -> bool:
    path = urlsplit(url).path.lower().rstrip("/")
    return (
        path.startswith(("/resource/", "/resource-links/"))
        or bool(re.search(r"-resources-(?:act|acts|rules|manuals?|guidelines?)$", path))
        or bool(re.search(r"/publications/(?:acts?|rules|manuals?|guidelines?)$", path))
        or bool(re.search(r"/pages/gi/learn/(?:acts?|rules|manuals?|guidelines?)$", path))
        or path in ("/pages/home/acts-rules-ipr", "/acts/patent-act-1970", "/sicldr-resources-act-rules")
    )


def best_title(extracted: str, linked: str, url: str) -> str:
    def useful(value: str) -> bool:
        return bool(value and len(value) > 8 and not re.match(r"^(?:\(?\d+(?:\.\d+)?\s*(?:KB|MB)|Format:|IP India -|Microsoft Word -|Book Rajpatra)", value, re.I))

    if useful(linked):
        return re.sub(r"\s+", " ", linked).strip()
    if useful(extracted):
        return re.sub(r"\s+", " ", extracted).strip()
    return re.sub(r"[_-]+", " ", Path(urlsplit(url).path).stem).strip() or extracted


def normalized_document_title(title: str, url: str) -> str:
    path = urlsplit(url).path.lower()
    if re.search(r"/acts/patent-act-1970/(?:section-)?\d+[a-z]?$", path):
        return "The Patents Act, 1970"
    if re.search(r"/acts/patent-rules-2003/rule-\d+[a-z]?$", path):
        return "The Patents Rules, 2003"
    return re.sub(r"\s+", " ", title).strip()


def select_export_records(entries: list[tuple[dict, list[dict]]]) -> list[tuple[dict, list[dict]]]:
    groups: dict[str, list[tuple[dict, list[dict]]]] = {}
    for document, chunks in entries:
        if chunks:
            groups.setdefault(document["content_hash"], []).append((document, chunks))

    def rank(entry: tuple[dict, list[dict]]) -> tuple[int, int]:
        document = entry[0]
        url = document["download_url"] or document["source_url"]
        basename = Path(urlsplit(url).path).stem
        descriptive = 0 if re.fullmatch(r"[0-9a-f-]{20,}", basename, re.I) else 1
        return descriptive, len(document["title"])

    selected = []
    for duplicates in groups.values():
        winner = max(duplicates, key=rank)
        document = dict(winner[0])
        aliases = sorted({d["download_url"] or d["source_url"] for d, _ in duplicates} - {document["download_url"] or document["source_url"]})
        if aliases:
            document["alternate_download_urls"] = aliases
        selected.append((document, winner[1]))
    selected.sort(key=lambda item: item[0]["document_id"])
    return selected


class Pipeline:
    def __init__(self, out: Path, delay: float, max_bytes: int):
        self.out = out
        self.client = Client(out, delay, max_bytes)
        self.audit = out / "audit.jsonl"
        self.external = out / "external_official_links.jsonl"
        self.seen: set[str] = set()
        self.queue: list[tuple[int, int, int, str, str, str]] = []
        self.serial = 0
        self.external_seen: set[str] = set()
        if self.external.exists():
            for line in self.external.read_text(encoding="utf-8").splitlines():
                try:
                    self.external_seen.add(json.loads(line)["external_url"])
                except (ValueError, KeyError):
                    continue

    def audit_event(self, url: str, status: str, reason: str = "") -> None:
        log_jsonl(self.audit, {"at": utcnow(), "url": url, "status": status, "reason": reason})

    def preflight(self, terms_reviewed: bool) -> bool:
        try:
            robots = self.client.load_robots("ipindia.gov.in")
            self.audit_event(BASE + "robots.txt", "verified", self.client.robots_state["ipindia.gov.in"])
            if not robots.can_fetch(USER_AGENT, BASE):
                raise ValueError("robots_disallowed")
            home = self.client.fetch(BASE)
            title, body, links = html_content(home.data)
            if not title or len(body) < 100:
                raise ValueError("homepage_inaccessible")
            self.audit_event(BASE, "verified", "public homepage accessible")
            policy_links = [urljoin(BASE, href) for href, label in links if re.search(r"terms(?:.and.conditions)?|website.policy|privacy.policy", f"{href} {label}", re.I)]
            if not policy_links:
                self.audit_event(BASE, "manual_review_required", "No terms or policy link discovered automatically; operator review needed")
            for link in policy_links[:5]:
                normalized = canonical(link)
                if normalized:
                    try:
                        self.client.fetch(normalized)
                        self.audit_event(normalized, "verified", "policy page accessible; review its terms")
                    except ValueError as exc:
                        self.audit_event(normalized, "manual_review_required", str(exc))
            if not terms_reviewed:
                self.audit_event(BASE, "manual_review_required", "Review website terms/policy and rerun with --terms-reviewed")
                return False
            self.audit_event(BASE, "verified", "--terms-reviewed flag supplied; operator remains responsible for policy interpretation")
            return True
        except ValueError as exc:
            self.audit_event(BASE, "manual_review_required", str(exc))
            return False

    def enqueue(self, url: str, base: str, label: str, depth: int) -> None:
        absolute = urljoin(base, url)
        normalized = canonical(absolute)
        if normalized is None:
            if urlsplit(absolute).scheme == "https" and urlsplit(absolute).hostname and urlsplit(absolute).hostname.endswith(".gov.in") and absolute not in self.external_seen:
                self.external_seen.add(absolute)
                log_jsonl(self.external, {"source_url": base, "external_url": absolute, "label": label})
            return
        if normalized in self.seen or score(normalized, label) < 0:
            return
        priority = score(normalized, label)
        if priority == 0 and depth > 1:
            return
        self.serial += 1
        heapq.heappush(self.queue, (-priority, depth, self.serial, normalized, base, label))

    def discover_sitemaps(self) -> None:
        robots = self.client.robots_text.get("ipindia.gov.in", "")
        urls = re.findall(r"(?im)^sitemap:\s*(\S+)", robots) or [BASE + "sitemap.xml"]
        visited: set[str] = set()
        while urls and len(visited) < 20:
            url = urls.pop(0)
            if url in visited or canonical(url) is None:
                continue
            visited.add(url)
            try:
                fetch = self.client.fetch(url)
                root = ET.fromstring(fetch.data)
                locs = [node.text.strip() for node in root.iter() if node.tag.endswith("loc") and node.text]
                if root.tag.endswith("sitemapindex"):
                    urls.extend(locs[:100])
                else:
                    for loc in locs[:5000]:
                        self.enqueue(loc, BASE, "sitemap", 1)
                self.audit_event(url, "verified", f"sitemap entries: {len(locs)}")
            except (ValueError, ET.ParseError) as exc:
                self.audit_event(url, "manual_review_required", f"sitemap_unavailable: {exc}")

    def run(self, max_pages: int = 100, max_depth: int = 3) -> None:
        for seed in SEEDS:
            self.enqueue(seed, BASE, "legal resource", 0)
        self.discover_sitemaps()
        processed = 0
        while self.queue and processed < max_pages:
            _, depth, _, url, source, label = heapq.heappop(self.queue)
            if url in self.seen or depth > max_depth:
                continue
            self.seen.add(url)
            processed += 1
            try:
                fetch = self.client.fetch(url)
                title, content, links = extract(fetch)
                if len(content.strip()) < 80:
                    raise ValueError("manual_review_required: too_little_extractable_text")
                if Path(urlsplit(url).path).suffix.lower() in FILE_TYPES:
                    title = best_title(title, label, url)
                if is_index_page(url):
                    self.audit_event(url, "discovery_only", "resource index; links retained for discovery")
                elif score(url, f"{title} {label}") > 0 and not NEGATIVE.search(f"{url} {title} {label}"):
                    document = build_document(fetch, title, content, source)
                    chunks = structure_chunks(content, document)
                    if not chunks:
                        raise ValueError("manual_review_required: no_usable_chunks")
                    target = self.out / "documents" / document["document_id"]
                    target.mkdir(parents=True, exist_ok=True)
                    (target / "metadata.json").write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
                    (target / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
                    self.audit_event(url, "ingested", f"{len(chunks)} chunks")
                else:
                    self.audit_event(url, "skipped", "outside focused legal corpus")
                if depth < max_depth:
                    for href, label in links:
                        self.enqueue(href, url, label, depth + 1)
            except (ValueError, OSError, ImportError) as exc:
                reason = str(exc)
                self.audit_event(url, "manual_review_required" if "manual_review_required" in reason or "robots" in reason else "error", reason)
        self.export()

    def rebuild(self) -> None:
        """Re-extract existing cached documents without network access."""
        docs_root = (self.out / "documents").resolve()
        cache_root = (self.out / "cache").resolve()
        for metadata_path in sorted(docs_root.glob("*/metadata.json")):
            prior = json.loads(metadata_path.read_text(encoding="utf-8"))
            url = prior.get("download_url") or prior.get("source_url", "")
            url = canonical(url)
            doc_id = prior.get("document_id", "")
            target = metadata_path.parent.resolve()
            if not url or not re.fullmatch(r"[0-9a-f]{20}", doc_id) or target.parent != docs_root or target.name != doc_id:
                self.audit_event(str(metadata_path), "manual_review_required", "unsafe_or_invalid_stored_document")
                continue
            key = digest(url.encode())
            raw = cache_root / f"{key}.bin"
            cache_meta = cache_root / f"{key}.json"
            if not raw.exists() or not cache_meta.exists():
                self.audit_event(url, "manual_review_required", "cached_source_missing")
                continue
            cache_info = json.loads(cache_meta.read_text(encoding="utf-8"))
            data = raw.read_bytes()
            if digest(data) != cache_info.get("sha256"):
                self.audit_event(url, "manual_review_required", "cached_hash_mismatch")
                continue
            if NEGATIVE.search(f"{url} {prior.get('title', '')}") or is_index_page(url):
                self.audit_event(url, "skipped", "outside focused legal corpus; archived bytes retained")
                continue
            fetch = Fetch(url, data, cache_info["content_type"], cache_info["retrieved_at"], raw)
            try:
                extracted_title, content, _ = extract(fetch)
                if len(content.strip()) < 80:
                    raise ValueError("too_little_extractable_text")
                title = best_title(extracted_title, prior.get("title", ""), url) if Path(urlsplit(url).path).suffix.lower() in FILE_TYPES else extracted_title
                document = build_document(fetch, title, content, prior["source_url"])
                chunks = structure_chunks(content, document)
                if not chunks:
                    raise ValueError("manual_review_required: no_usable_chunks")
                metadata_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
                (target / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
                self.audit_event(url, "rebuilt", f"{len(chunks)} chunks")
            except (ValueError, OSError, ImportError) as exc:
                self.audit_event(url, "manual_review_required", f"rebuild_failed: {exc}")
        if self.external.exists():
            unique = {}
            for line in self.external.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                unique.setdefault(record["external_url"], record)
            with self.external.open("w", encoding="utf-8") as stream:
                for record in unique.values():
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.export()

    def export(self) -> None:
        entries = []
        for path in sorted((self.out / "documents").glob("*/metadata.json")):
            document = json.loads(path.read_text(encoding="utf-8"))
            chunks_path = path.parent / "chunks.json"
            if not chunks_path.exists():
                continue
            chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
            url = document["download_url"] or document["source_url"]
            if NEGATIVE.search(f"{url} {document.get('title', '')}") or is_index_page(url):
                continue
            if document["source_url"] != url:
                document.setdefault("discovered_from_url", document["source_url"])
                document["source_url"] = url
            document.setdefault("canonical_url", url)
            document["title"] = normalized_document_title(document["title"], url)
            primary_domain, related_domains = classify_domains(document["title"], url, document.get("discovered_from_url") or document["source_url"])
            document["ip_domain"] = primary_domain
            document["related_ip_domains"] = related_domains
            for chunk in chunks:
                chunk["ip_domain"] = primary_domain
                chunk["document_title"] = document["title"]
                chunk["source_url"] = url
            entries.append((document, chunks))
        selected = select_export_records(entries)
        with (self.out / "documents.jsonl").open("w", encoding="utf-8") as docs_file, (self.out / "chunks.jsonl").open("w", encoding="utf-8") as chunks_file:
            for document, chunks in selected:
                docs_file.write(json.dumps(document, ensure_ascii=False) + "\n")
                for chunk in chunks:
                    chunks_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        print(f"Exported {len(selected)} unique documents and {sum(len(chunks) for _, chunks in selected)} chunks")
        if self.audit.exists():
            latest = {}
            for line in self.audit.read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                latest[event["url"]] = event
            with (self.out / "review_queue.jsonl").open("w", encoding="utf-8") as queue:
                for event in latest.values():
                    if event["status"] == "manual_review_required" and "sitemap_unavailable: http_404" not in event["reason"]:
                        queue.write(json.dumps(event, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Conservative IP India legal corpus ingestion")
    parser.add_argument("command", choices=["preflight", "crawl", "rebuild", "export"])
    parser.add_argument("--out", type=Path, default=Path("corpus"))
    parser.add_argument("--terms-reviewed", action="store_true", help="Record that an operator reviewed site terms/policy")
    parser.add_argument("--delay", type=float, default=1.5)
    parser.add_argument("--max-pages", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-bytes", type=int, default=20_000_000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pipeline = Pipeline(args.out, args.delay, args.max_bytes)
    if args.command == "export":
        pipeline.export()
        return 0
    if args.command == "rebuild":
        pipeline.rebuild()
        return 0
    passed = pipeline.preflight(args.terms_reviewed)
    if args.command == "preflight":
        return 0 if passed else 2
    if not passed:
        print("Preflight stopped: inspect audit.jsonl for manual_review_required")
        return 2
    pipeline.run(args.max_pages, args.max_depth)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
