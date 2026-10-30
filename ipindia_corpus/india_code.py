"""Bounded public HAL discovery with explicit central-instrument identities."""
from __future__ import annotations
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from .connectors import official_url


def _values(metadata, predicate):
    return [str(row.get("value", "")) for key, rows in metadata.items() if predicate(key.lower())
            for row in rows if isinstance(row, dict)]


def _normal(value):
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).removeprefix("the ").strip()


def verify_identity(item: dict, expected: dict) -> dict:
    metadata = item.get("metadata", {})
    titles = _values(metadata, lambda key: key == "dc.title")
    if not any(_normal(title) == _normal(expected["title"]) for title in titles):
        raise ValueError("instrument_title_mismatch")
    numbers = _values(metadata, lambda key: "actnumber" in re.sub(r"[^a-z]", "", key) or "actno" in re.sub(r"[^a-z]", "", key))
    if not any(re.search(rf"\b{expected['act_number']}\b", number) for number in numbers):
        raise ValueError("instrument_act_number_unverified")
    dates = _values(metadata, lambda key: key == "dc.date.issued" or "enactment" in key or "actyear" in key)
    if not any(str(expected["enactment_year"]) in date for date in dates):
        raise ValueError("instrument_enactment_year_unverified")
    jurisdictions = _values(metadata, lambda key: any(word in key for word in ("jurisdiction", "acttype", "actid", "act.id")))
    if not any(re.search(r"\bcentral\b|AC_CEN", value, re.I) for value in jurisdictions):
        raise ValueError("central_jurisdiction_unverified")
    return {"title": expected["title"], "act_number": expected["act_number"],
            "enactment_year": expected["enactment_year"], "jurisdiction": "Central"}


def _items(value):
    if isinstance(value, dict):
        if value.get("type") == "item" and isinstance(value.get("metadata"), dict):
            yield value
        else:
            for child in value.values():
                yield from _items(child)
    elif isinstance(value, list):
        for child in value:
            yield from _items(child)


def _collection(value, kind):
    if isinstance(value, dict):
        if value.get("type") == kind:
            yield value
        else:
            for child in value.get("_embedded", {}).values():
                yield from _collection(child, kind)
    elif isinstance(value, list):
        for child in value:
            yield from _collection(child, kind)


def _link(value, name):
    entry = value.get("_links", {}).get(name)
    if not isinstance(entry, dict) or not entry.get("href"):
        raise ValueError(f"missing_public_metadata_link:{name}")
    return entry["href"].split("{", 1)[0]


def discover(client, ledger, run_id: int, config: dict) -> tuple[list[str], dict, list[str]]:
    count, metadata_urls = 0, []
    urls, document_metadata = [], {}
    subordinate_review = []

    def read(url):
        nonlocal count
        url = official_url("indiacode", url)
        if not re.match(r"^/server/api(?:$|/(?:discover|core/(?:items|bundles|bitstreams))(?:/|$))", urlsplit(url).path):
            raise ValueError("unapproved_metadata_relation")
        count += 1
        if count > config["max_discovered"]:
            raise ValueError("incomplete_api_discovery_limit")
        data, snapshot = client.get_json(url)
        ledger.observe(run_id, snapshot.url, snapshot.body, content_type=snapshot.content_type,
                       etag=snapshot.etag, last_modified=snapshot.last_modified,
                       metadata={"discovery_only": True})
        metadata_urls.append(snapshot.url)
        return data

    def collection(url, kind):
        seen = set()
        while url:
            if url in seen:
                raise ValueError("api_pagination_cycle")
            seen.add(url)
            data = read(url)
            yield from _collection(data, kind)
            page = data.get("page", {})
            next_link = data.get("_links", {}).get("next")
            if page.get("totalPages", 1) > page.get("number", 0) + 1 and not next_link:
                raise ValueError("incomplete_api_pagination")
            url = next_link.get("href") if isinstance(next_link, dict) else None

    def documents_for(item, title, identity, *, subordinate=False):
        record_url = official_url("indiacode", f"https://indiacode.gov.in/items/{item['id']}")
        found = 0
        for bundle in collection(_link(item, "bundles"), "bundle"):
            if bundle.get("name", "").upper() != "ORIGINAL":
                continue
            for bitstream in collection(_link(bundle, "bitstreams"), "bitstream"):
                if not bitstream.get("name", "").lower().endswith(".pdf"):
                    continue
                url = official_url("indiacode", _link(bitstream, "content"))
                urls.append(url)
                found += 1
                document_metadata[url] = {"title": title, "official_record_url": record_url,
                                          "parent_identity" if subordinate else "verified_identity": identity,
                                          "discovery_only": False, "bitstream_name": bitstream.get("name"),
                                          "metadata_item_url": official_url("indiacode", _link(item, "self"))}
        return found

    api = read(config["api_root"])
    discovery = read(_link(api, "discover"))
    search_base = _link(discovery, "search")
    for expected in config["instruments"]:
        parts = urlsplit(search_base)
        parameters = dict(parse_qsl(parts.query))
        parameters.update(query=expected["title"], size="20", page="0")
        next_url = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(parameters), ""))
        matches, seen = {}, set()
        while next_url:
            if next_url in seen:
                raise ValueError("api_pagination_cycle")
            seen.add(next_url)
            response = read(next_url)
            for candidate in _items(response):
                titles = _values(candidate["metadata"], lambda key: key == "dc.title")
                if any(_normal(title) == _normal(expected["title"]) for title in titles):
                    identity = verify_identity(candidate, expected)
                    matches[candidate["id"]] = (candidate, identity)
            pagination = response.get("_embedded", {}).get("searchResult", response)
            next_link = pagination.get("_links", {}).get("next")
            page = pagination.get("page", {})
            if page.get("totalPages", 1) > page.get("number", 0) + 1 and not next_link:
                raise ValueError("incomplete_api_pagination")
            next_url = next_link.get("href") if isinstance(next_link, dict) else None
        if len(matches) != 1:
            raise ValueError(f"central_instrument_missing_or_ambiguous:{expected['title']}")
        item, identity = next(iter(matches.values()))
        item = read(_link(item, "self"))
        verify_identity(item, expected)
        if not documents_for(item, identity["title"], identity):
            raise ValueError(f"instrument_has_no_public_pdf:{identity['title']}")
        # Subordinate legislation is followed only from explicit public metadata relations.
        related_count = 0
        for relation in ("rules", "regulations", "notifications"):
            if relation not in item.get("_links", {}):
                continue
            for subordinate in collection(_link(item, relation), "item"):
                title = _values(subordinate["metadata"], lambda key: key == "dc.title")
                if not title or not re.search(config.get("subordinate_relevance", r"rules|regulations"), title[0], re.I):
                    continue
                subordinate = read(_link(subordinate, "self"))
                related_count += documents_for(subordinate, title[0], identity, subordinate=True)
        if not related_count:
            subordinate_review.append(identity["title"])
    ledger.event("indiacode", "discovery", "review_required" if subordinate_review else "success",
                 "Related legislation requires verification for: " + "; ".join(subordinate_review) if subordinate_review else "Linked subordinate legislation captured")
    return list(dict.fromkeys(urls)), document_metadata, list(dict.fromkeys(metadata_urls))
