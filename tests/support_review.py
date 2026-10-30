"""Explicit synthetic access evidence for offline connector tests."""
import json
from ipindia_corpus.access_review import capture
from ipindia_corpus.sync import SOURCES


class Response:
    status_code = 200
    headers = {"Content-Type": "text/plain"}

    def __init__(self, body):
        self.body = body

    def iter_content(self, _):
        yield self.body

    def close(self):
        pass


class Session:
    def get(self, url, **kwargs):
        return Response(b"User-agent: *\nDisallow:\n" if url.endswith("robots.txt") else b"Synthetic fixture: public retrieval is permitted.")


def approve_fixture(root, source):
    base = "https://" + SOURCES[source]
    evidence = [capture(root, source, base + "/robots.txt", "robots", Session()),
                capture(root, source, base + "/policy", "policy", Session())]
    path = root / "config" / "source_policies.json"
    policies = json.loads(path.read_text()) if path.exists() else {}
    policies[source] = {"approved": True, "reviewer": "synthetic offline fixture", "reviewed_at": "2026-01-01T00:00:00Z",
                        "policy_url": base + "/policy", "notes": "Synthetic test permission, never live authorization",
                        "permitted_prefixes": ["/"], "evidence": evidence}
    path.write_text(json.dumps(policies), encoding="utf-8")
    return policies[source]
