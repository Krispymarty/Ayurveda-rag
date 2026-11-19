import contextlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ipindia_corpus.http_security import SecurityMiddleware
from ipindia_corpus.remote_embeddings import EmbeddingBindingError, EmbeddingUnavailable, RemoteEmbeddings
from tools.prepare_vercel import prepare

ROOT = Path(__file__).resolve().parents[1]
KEY = "e" * 64
REV = json.loads((ROOT / "data/model.lock.json").read_bytes())["revision"]
IDENTITY = {"model_id": "intfloat/multilingual-e5-small", "revision": REV, "dimension": 384}
VECTOR = [1.0] + [0.0] * 383


def response(value, status=200):
    raw = json.dumps(value).encode()
    return contextlib.nullcontext(Mock(status_code=status, headers={"Content-Type": "application/json"},
                                       iter_content=lambda **_: iter([raw])))


class RemoteClientTests(unittest.TestCase):
    def setup_client(self, *responses):
        session = Mock()
        session.request.side_effect = responses
        return RemoteEmbeddings(REV, environ={"AYURIP_E5_API_KEY": KEY}, session_factory=lambda: session), session

    def test_verified_identity_normalized_vector_and_safe_transport(self):
        client, session = self.setup_client(response({**IDENTITY, "status": "ready"}),
                                            response({**IDENTITY, "embedding": VECTOR}))
        self.assertEqual(client.query("धारा 3"), VECTOR)
        sent = session.request.call_args
        self.assertEqual(sent.args, ("POST", "https://yashmit-legal-rag.hf.space/embed/e5/v1/query"))
        self.assertEqual(sent.kwargs["json"], {"text": "धारा 3"})
        self.assertFalse(sent.kwargs["allow_redirects"])
        self.assertTrue(sent.kwargs["verify"])
        self.assertFalse(session.trust_env)

    def test_wrong_revision_and_malformed_vector_latch_the_dependency(self):
        for patch_value in ({"revision": "unexpected"}, {"embedding": [False] * 384},
                            {"embedding": [0] * 384}, {"embedding": [float("nan")] * 384}):
            with self.subTest(patch_value=list(patch_value)):
                client, session = self.setup_client(response({**IDENTITY, "status": "ready"}),
                    response({**IDENTITY, "embedding": VECTOR, **patch_value}))
                with self.assertRaises(EmbeddingBindingError):
                    client.query("Patents Act")
                with self.assertRaises(EmbeddingBindingError):
                    client.assert_ready()
                self.assertEqual(session.request.call_count, 2)

    def test_outage_recovers_without_loading_a_local_model(self):
        client, session = self.setup_client(response({}, 503), response({**IDENTITY, "status": "ready"}),
                                            response({**IDENTITY, "embedding": VECTOR}))
        with patch("ipindia_corpus.remote_embeddings.time.monotonic", return_value=1):
            with self.assertRaises(EmbeddingUnavailable):
                client.assert_ready()
            with self.assertRaises(EmbeddingUnavailable):
                client.assert_ready()
        self.assertEqual(session.request.call_count, 1)
        with patch("ipindia_corpus.remote_embeddings.time.monotonic", return_value=5):
            self.assertEqual(client.query("Patents Act"), VECTOR)


class HostedRoutingTests(unittest.TestCase):
    def test_origin_secret_is_required_except_for_probe_routes(self):
        app = FastAPI()

        @app.get("/search")
        @app.get("/health")
        def probe():
            return {"ok": True}

        app.add_middleware(SecurityMiddleware)
        env = {"AYURIP_DEPLOYMENT_MODE": "public_research", "AYURIP_PUBLIC_ORIGIN": "https://demo.vercel.app",
               "AYURIP_ALLOWED_HOSTS": "backend.onrender.com", "AYURIP_ORIGIN_PROXY_SECRET": KEY}
        with patch.dict("os.environ", env, clear=True), TestClient(app, base_url="https://backend.onrender.com") as client:
            self.assertEqual(client.get("/health").status_code, 200)
            self.assertEqual(client.get("/search").status_code, 403)
            result = client.get("/search", headers={"X-Origin-Secret": KEY})
            self.assertEqual((result.status_code, result.headers["Cache-Control"]), (200, "no-store"))
            self.assertEqual(client.get("/search", headers=[("X-Origin-Secret", KEY), ("X-Origin-Secret", KEY)]).status_code, 403)
            self.assertEqual(client.get("/search", headers={"X-Origin-Secret": KEY, "Origin": "https://evil.invalid"}).status_code, 403)

    def test_vercel_package_contains_only_static_assets_and_proxy_secret_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            out = prepare(Path(temp) / "site", "https://backend.onrender.com")
            config = json.loads((out / "vercel.json").read_bytes())
            proxy = config["routes"][1]
            self.assertEqual(proxy["dest"], "https://backend.onrender.com/$1")
            self.assertEqual(proxy["transforms"][0]["args"], "$ORIGIN_SECRET")
            self.assertNotIn("passport", proxy["src"])
            self.assertTrue((out / "assets" / "app.js").is_file())
            self.assertFalse((out / ".env").exists())
            research = (out / "research.html").read_text(encoding="utf-8")
            self.assertIn('id="hosted-embedding-notice" class="service-notice" data-en', research)


class SpaceRouterTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("space_e5", ROOT / "deploy/hf_space/ayurip_e5.py")
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        app = FastAPI()
        app.include_router(self.module.router)
        self.client = TestClient(app)

    def test_auth_and_bounded_inputs_precede_model(self):
        route = "/embed/e5/v1/query"
        with patch.dict("os.environ", {"AYURIP_E5_API_KEY": KEY}), patch.object(self.module, "load_encoder") as load:
            self.assertEqual(self.client.post(route, json={"text": "Patents Act"}).status_code, 401)
            headers = {"X-AyurIP-Embedding-Key": KEY}
            self.assertEqual(self.client.post(route, headers=headers, json={"text": "a"*501}).status_code, 422)
            self.assertEqual(self.client.post(route, headers=headers, json={"text": "a"*9000}).status_code, 413)
            load.assert_not_called()
            statuses = Mock()
            statuses.tolist.return_value = VECTOR
            model = Mock()
            model.encode.return_value = [statuses]
            load.return_value = model
            reply = self.client.post(route, headers=headers, json={"text": "धारा 3"})
            self.assertEqual(reply.json(), {**IDENTITY, "embedding": VECTOR})
            model.encode.assert_called_once_with(["query: धारा 3"], normalize_embeddings=True)
