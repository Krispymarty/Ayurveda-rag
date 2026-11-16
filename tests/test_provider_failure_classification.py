"""Transport availability must remain distinct from rejected/malformed data."""
import unittest
import requests
from ipindia_corpus.inference import GeminiProvider
from ipindia_corpus.llm_experiment import ExperimentFailure
from test_gemini_provider import KEY, MODEL, FIELDS, Response, Session


class BadJSON(Response):
    def iter_content(self, chunk_size=1024):
        yield b'not json'


class FailureClassificationTests(unittest.TestCase):
    def test_rejected_http_and_bad_json_are_not_transient_unavailability(self):
        for response, reason in [
            (Response({}, 400), 'provider_request_rejected'),
            (Response({}, 402), 'provider_request_rejected'),
            (Response({}, 404), 'provider_model_unavailable'),
            (Response({}, 408), 'timeout'),
            (Response({}, 503), 'provider_unavailable'),
            (BadJSON({}), 'provider_invalid_response'),
            (requests.exceptions.SSLError(KEY), 'provider_tls_failed'),
            (requests.ConnectionError(KEY), 'provider_unavailable'),
        ]:
            with self.subTest(reason=reason):
                session = Session(response)
                provider = GeminiProvider(api_key=KEY, model=MODEL, session=session)
                with self.assertRaisesRegex(ExperimentFailure, '^' + reason + '$'):
                    provider.chat(FIELDS)
                self.assertEqual(len(session.calls), 1)


if __name__ == '__main__':
    unittest.main()
