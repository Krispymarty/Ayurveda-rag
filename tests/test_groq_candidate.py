"""Isolated candidate transport checks; no real credentials or inference."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import requests

from ipindia_corpus.groq_candidate import GroqCandidate, ENDPOINT, file_settings, read_key
from ipindia_corpus.llm_experiment import ExperimentFailure

KEY = 'synthetic-groq-key-not-a-credential'
MODEL = 'openai/gpt-oss-120b'
FIELDS = {'messages': [{'role': 'system', 'content': 'Select opaque IDs only.'},
                       {'role': 'user', 'content': 'Untrusted fixture text.'}],
          'format': {'type': 'object', 'additionalProperties': False, 'required': ['span_ids'],
                     'properties': {'span_ids': {'type': 'array', 'maxItems': 4, 'uniqueItems': True,
                                               'items': {'type': 'string', 'enum': ['S001']}}}}}


def generated(content='{"span_ids":["S001"]}', **overrides):
    return {'model': MODEL, 'choices': [{'index': 0, 'finish_reason': 'stop',
            'message': {'role': 'assistant', 'content': content}}],
            'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120,
                      'completion_tokens_details': {'reasoning_tokens': 8}}, **overrides}


class Response:
    def __init__(self, value=None, *, status=200, raw=None):
        self.value, self.status_code, self.raw, self.closed = value, status, raw, False
    def iter_content(self, chunk_size=1024):
        yield self.raw if self.raw is not None else json.dumps(self.value).encode()
    def close(self):
        self.closed = True


class Session:
    def __init__(self, *responses):
        self.responses, self.calls, self.trust_env = list(responses), [], True
    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class GroqCandidateTests(unittest.TestCase):
    def provider(self, *responses, **kwargs):
        session = Session(*(responses or [Response(generated())]))
        return GroqCandidate(api_key=KEY, model=MODEL, session=session, **kwargs), session

    def test_fixed_endpoint_header_only_key_schema_and_limits(self):
        provider, session = self.provider()
        before = deepcopy(FIELDS)
        raw, telemetry = provider.chat(FIELDS)
        self.assertEqual(json.loads(raw), {'span_ids': ['S001']})
        self.assertEqual(FIELDS, before)
        method, url, args = session.calls[0]
        self.assertEqual((method, url), ('POST', ENDPOINT))
        self.assertEqual(args['headers']['Authorization'], 'Bearer ' + KEY)
        self.assertNotIn(KEY, url + json.dumps(args['json']) + str(provider.metadata) + str(telemetry))
        self.assertFalse(session.trust_env)
        self.assertFalse(args['allow_redirects'])
        self.assertTrue(args['verify'])
        payload = args['json']
        self.assertEqual(payload['max_completion_tokens'], 200)
        self.assertEqual(payload['reasoning_effort'], 'low')
        self.assertFalse(payload['include_reasoning'])
        self.assertEqual(payload['temperature'], 0)
        self.assertFalse(payload['stream'])
        expected = deepcopy(FIELDS['format'])
        expected['properties']['span_ids'].pop('uniqueItems')
        self.assertEqual(payload['response_format']['json_schema']['schema'], expected)
        self.assertTrue(FIELDS['format']['properties']['span_ids']['uniqueItems'])
        self.assertTrue(payload['response_format']['json_schema']['strict'])
        self.assertNotIn('tools', payload)
        self.assertIsNone(provider.metadata['model_digest'])
        self.assertEqual(provider.metadata['model_version'], MODEL)

    def test_nested_wire_uniqueness_is_removed_without_weakening_local_schema(self):
        from ipindia_corpus.groq_candidate import _wire_schema
        schema = {'type': 'array', 'uniqueItems': True, 'maxItems': 4,
                  'items': {'type': 'object', 'properties': {'nested': {
                      'type': 'array', 'uniqueItems': True, 'items': {'type': 'string'}}}}}
        before = deepcopy(schema)
        wire = _wire_schema(schema)
        self.assertNotIn('uniqueItems', wire)
        self.assertNotIn('uniqueItems', wire['items']['properties']['nested'])
        self.assertEqual(wire['maxItems'], 4)
        self.assertEqual(schema, before)

    def test_http_errors_sanitized_and_never_retried(self):
        for status, reason in [(302, 'provider_redirect_rejected'), (400, 'provider_request_rejected'),
                               (403, 'provider_auth_failed'), (404, 'provider_model_unavailable'),
                               (408, 'timeout'), (429, 'provider_rate_limited'), (503, 'provider_unavailable')]:
            with self.subTest(status=status):
                response = Response({'error': KEY}, status=status)
                p, session = self.provider(response)
                with self.assertRaisesRegex(ExperimentFailure, '^' + reason + '$'):
                    p.chat(FIELDS)
                self.assertEqual(len(session.calls), 1)
                self.assertTrue(response.closed)

    def test_tls_timeout_and_connection_errors(self):
        for exc, reason in [(requests.Timeout(KEY), 'timeout'),
                            (requests.exceptions.SSLError(KEY), 'provider_tls_failed'),
                            (requests.ConnectionError(KEY), 'provider_unavailable')]:
            p, _ = self.provider(exc)
            with self.assertRaisesRegex(ExperimentFailure, '^' + reason + '$'):
                p.chat(FIELDS)

    def test_malformed_duplicate_unknown_or_extra_ids_fail(self):
        for text in ['not JSON', '{"span_ids":["forged"]}', '{"span_ids":["S001","S001"]}',
                     '{"span_ids":[],"span_ids":["S001"]}', '{"span_ids":[],"advice":"approved"}']:
            with self.subTest(text=text):
                p, _ = self.provider(Response(generated(text)))
                with self.assertRaisesRegex(ExperimentFailure, '^provider_invalid_selection$'):
                    p.chat(FIELDS)

    def test_truncation_tools_reasoning_and_refusal_rejected(self):
        for changes, reason in [({'finish_reason': 'length'}, 'provider_incomplete_response'),
                ({'message': {'role': 'assistant', 'content': '{}', 'tool_calls': [{'name': 'execute'}]}}, 'provider_invalid_response'),
                ({'message': {'role': 'assistant', 'content': '{}', 'reasoning': 'hidden content'}}, 'provider_invalid_response'),
                ({'message': {'role': 'assistant', 'content': '{}', 'refusal': 'no'}}, 'provider_safety_blocked')]:
            value = generated()
            value['choices'][0].update(changes)
            p, _ = self.provider(Response(value))
            with self.assertRaisesRegex(ExperimentFailure, '^' + reason + '$'):
                p.chat(FIELDS)

    def test_model_mismatch_latches_without_second_network_call(self):
        p, session = self.provider(Response(generated(model='other/model')))
        for _ in range(2):
            with self.assertRaisesRegex(ExperimentFailure, '^provider_model_changed$'):
                p.chat(FIELDS)
        self.assertEqual(len(session.calls), 1)

    def test_failed_length_diagnostics_are_bounded_and_reset_per_call(self):
        value = generated(usage={'prompt_tokens': 120, 'completion_tokens': 200, 'total_tokens': 320,
                                'completion_tokens_details': {'reasoning_tokens': 185}})
        value['choices'][0]['finish_reason'] = 'length'
        p, _ = self.provider(Response(value), Response({'error': KEY}, status=503))
        with self.assertRaisesRegex(ExperimentFailure, '^provider_incomplete_response$'):
            p.chat(FIELDS)
        self.assertEqual(p.last_call_diagnostics, {'http_status': 200, 'returned_model_matches': True,
                         'prompt_tokens': 120, 'completion_tokens': 200, 'total_tokens': 320,
                         'reasoning_tokens': 185, 'finish_reason': 'length'})
        self.assertNotIn(KEY, str(p.last_call_diagnostics))
        with self.assertRaisesRegex(ExperimentFailure, '^provider_unavailable$'):
            p.chat(FIELDS)
        self.assertEqual(p.last_call_diagnostics, {'http_status': 503})
        with self.assertRaises(ExperimentFailure):
            p.chat({})
        self.assertEqual(p.last_call_diagnostics, {})

    def test_diagnostics_never_retain_arbitrary_server_strings_or_bad_usage(self):
        value = generated(usage={'prompt_tokens': KEY, 'completion_tokens': -1, 'total_tokens': 10_000_001,
                                'completion_tokens_details': {'reasoning_tokens': True}})
        value['choices'][0]['finish_reason'] = KEY
        p, _ = self.provider(Response(value))
        with self.assertRaisesRegex(ExperimentFailure, '^provider_incomplete_response$'):
            p.chat(FIELDS)
        self.assertEqual(p.last_call_diagnostics, {'http_status': 200, 'returned_model_matches': True})

    def test_usage_counts_and_conservative_preflight(self):
        for usage, reason in [({'prompt_tokens': 3801, 'completion_tokens': 1, 'total_tokens': 3802}, 'evidence_context_limit'),
                              ({'prompt_tokens': 1, 'completion_tokens': 201, 'total_tokens': 202}, 'evidence_context_limit'),
                              ({'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 9}, 'provider_invalid_response'),
                              ({'prompt_tokens': True, 'completion_tokens': 1, 'total_tokens': 2}, 'provider_invalid_response')]:
            p, _ = self.provider(Response(generated(usage=usage)))
            with self.assertRaisesRegex(ExperimentFailure, '^' + reason + '$'):
                p.chat(FIELDS)
        fields = deepcopy(FIELDS)
        fields['messages'][1]['content'] = 'x' * 4000
        p, session = self.provider()
        with self.assertRaisesRegex(ExperimentFailure, '^evidence_context_limit$'):
            p.chat(fields)
        self.assertFalse(session.calls)

    def test_counter_injection_margin_and_invalid_counts(self):
        p, session = self.provider(token_counter=lambda fields: 100)
        _, telemetry = p.chat(FIELDS)
        self.assertEqual(telemetry['input_preflight_count_with_margin'], 356)
        for count in [-1, True, 3.5, 3800]:
            p, session = self.provider(token_counter=lambda fields, value=count: value)
            with self.assertRaises(ExperimentFailure):
                p.chat(FIELDS)
            self.assertFalse(session.calls)

    def test_invalid_config_and_fields_never_call_network(self):
        for model in ['latest', 'https://evil.test/model']:
            with self.assertRaisesRegex(ExperimentFailure, '^groq_configuration_invalid$'):
                GroqCandidate(api_key=KEY, model=model)
        for key in ['', 'short', KEY + '\n']:
            with self.assertRaises(ExperimentFailure):
                GroqCandidate(api_key=key, model=MODEL)
        p, session = self.provider()
        with self.assertRaises(ExperimentFailure):
            p.chat({**FIELDS, 'tools': ['unapproved']})
        self.assertFalse(session.calls)

    def test_env_reader_is_allowlisted_no_expansion_no_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'.env').write_text('GROQ_API_KEY="' + KEY + '" # comment\nOTHER_SECRET=ignored\nAYURIP_GROQ_MODEL=openai/gpt-oss-120b\n', encoding='utf-8')
            before = dict(os.environ)
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(read_key(root), KEY)
                self.assertEqual(set(file_settings(root)), {'GROQ_API_KEY', 'AYURIP_GROQ_MODEL'})
                self.assertFalse(os.environ)
            self.assertEqual(dict(os.environ), before)
            (root/'.env').write_text('GROQ_API_KEY=' + KEY + '\nGROQ_API_KEY=conflicting-different-key\n', encoding='utf-8')
            with self.assertRaisesRegex(ExperimentFailure, '^groq_configuration_invalid$'):
                file_settings(root)

    def test_expired_and_late_deadline_close_and_reject(self):
        p, session = self.provider()
        with patch('ipindia_corpus.groq_candidate.time.monotonic', return_value=10):
            with self.assertRaisesRegex(ExperimentFailure, '^timeout$'):
                p.chat_with_deadline(FIELDS, 9)
        self.assertFalse(session.calls)
        response = Response(generated())
        p, session = self.provider(response)
        with patch('ipindia_corpus.groq_candidate.time.monotonic', side_effect=[10, 12]):
            with self.assertRaisesRegex(ExperimentFailure, '^timeout$'):
                p.chat_with_deadline(FIELDS, 11)
        self.assertTrue(response.closed)
        self.assertLessEqual(session.calls[0][2]['timeout'][1], 1)

    def test_oversized_or_invalid_http_body_rejected(self):
        for raw, reason in [(b'x'*256001, 'provider_response_too_large'),
                            (b'not json', 'provider_invalid_response')]:
            p, _ = self.provider(Response(raw=raw))
            with self.assertRaisesRegex(ExperimentFailure, '^' + reason + '$'):
                p.chat(FIELDS)


if __name__ == '__main__':
    unittest.main()
