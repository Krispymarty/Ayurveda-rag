"""Offline transport deadline checks; no model calls or source changes."""
import json
import unittest
from unittest.mock import patch

import requests

from ipindia_corpus.llm_experiment import ExperimentFailure, OllamaExperimentClient


class Clock:
    def __init__(self, now=100.0):
        self.now = now

    def __call__(self):
        return self.now


class Response:
    status_code = 200

    def __init__(self, body, clock, *, advance=0, stream=False, late_eof=False):
        self.body, self.clock = body, clock
        self.advance, self.stream, self.late_eof = advance, stream, late_eof
        self.closed = False

    def iter_content(self, chunk_size):
        self.clock.now += self.advance
        yield json.dumps(self.body).encode()
        if self.late_eof:
            self.clock.now += 100

    def iter_lines(self):
        self.clock.now += self.advance
        yield json.dumps(self.body).encode()
        if self.late_eof:
            self.clock.now += 100

    def close(self):
        self.closed = True


class Session:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []
        self.trust_env = True

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class OllamaDeadlineTests(unittest.TestCase):
    def test_constructor_rejects_invalid_absolute_deadlines(self):
        for value in (True, False, 'soon', float('nan'), float('inf'), -float('inf'), -1):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, '^invalid_absolute_deadline$'):
                OllamaExperimentClient(absolute_deadline=value)

    def test_expired_deadline_rejects_before_network(self):
        clock, session = Clock(), Session([])
        with patch('ipindia_corpus.llm_experiment.time.monotonic', clock):
            client = OllamaExperimentClient(session=session, absolute_deadline=100)
            with self.assertRaisesRegex(ExperimentFailure, '^timeout$'):
                client.request('POST', '/api/chat', {})
        self.assertEqual(session.calls, [])

    def test_receipt_and_chat_share_remaining_deadline(self):
        clock = Clock()
        session = Session([
            Response({'models': [{'name': 'qwen3.5:4b', 'digest': 'a' * 64}]}, clock, advance=2),
            Response({'capabilities': ['completion', 'thinking']}, clock, advance=3),
            Response({'model': 'qwen3.5:4b', 'done': True, 'message': {'content': '{}'}}, clock, advance=1),
        ])
        with patch('ipindia_corpus.llm_experiment.time.monotonic', clock):
            client = OllamaExperimentClient(timeout=45, session=session, absolute_deadline=110)
            receipt = client.receipt('qwen3.5:4b')
            result = client.request('POST', '/api/chat', {})
        self.assertTrue(receipt['thinking_capable'])
        self.assertEqual(result['response']['message']['content'], '{}')
        self.assertEqual([call[2]['timeout'] for call in session.calls], [(10, 10), (8, 8), (5, 5)])
        self.assertTrue(all(not call[2]['allow_redirects'] for call in session.calls))
        self.assertFalse(session.trust_env)

    def test_expiry_during_receipt_never_starts_chat(self):
        clock = Clock()
        responses = [Response({'models': [{'name': 'qwen3.5:4b', 'digest': 'a' * 64}]}, clock, advance=2),
                     Response({'capabilities': ['completion']}, clock, advance=4)]
        session = Session(responses)
        with patch('ipindia_corpus.llm_experiment.time.monotonic', clock):
            client = OllamaExperimentClient(session=session, absolute_deadline=105)
            with self.assertRaisesRegex(ExperimentFailure, '^timeout$'):
                client.receipt('qwen3.5:4b')
        self.assertEqual(len(session.calls), 2)
        self.assertTrue(all(response.closed for response in responses))

    def test_late_body_and_late_eof_discard_output_in_both_modes(self):
        for streaming in (False, True):
            for late_eof in (False, True):
                clock = Clock()
                response = Response({'message': {'content': 'private response'}, 'done': True}, clock,
                                    advance=1 if late_eof else 6, late_eof=late_eof)
                session = Session([response])
                with self.subTest(streaming=streaming, late_eof=late_eof):
                    with patch('ipindia_corpus.llm_experiment.time.monotonic', clock):
                        client = OllamaExperimentClient(session=session, absolute_deadline=105)
                        with self.assertRaisesRegex(ExperimentFailure, '^timeout$'):
                            client.request('POST', '/api/chat', {}, streaming=streaming)
                    self.assertTrue(response.closed)

    def test_default_timeout_is_unchanged_and_errors_have_stable_reasons(self):
        clock = Clock()
        session = Session([Response({}, clock)])
        with patch('ipindia_corpus.llm_experiment.time.monotonic', clock):
            client = OllamaExperimentClient(timeout=45, session=session)
            client.request('GET', '/api/tags')
        self.assertEqual(session.calls[0][2]['timeout'], (10, 45))
        for exc, reason in [(requests.Timeout('sensitive transport information'), 'timeout'),
                            (requests.ConnectionError('sensitive transport information'), 'invalid_or_unavailable_response')]:
            session = Session([exc])
            with patch('ipindia_corpus.llm_experiment.time.monotonic', clock):
                client = OllamaExperimentClient(session=session, absolute_deadline=105)
                with self.assertRaisesRegex(ExperimentFailure, '^' + reason + '$') as caught:
                    client.request('POST', '/api/chat', {})
                self.assertNotIn('sensitive', str(caught.exception))


if __name__ == '__main__':
    unittest.main()
