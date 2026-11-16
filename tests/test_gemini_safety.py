"""Backend consent, version drift and budgets; synthetic fixtures only."""
import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic import ValidationError

from ipindia_corpus.llm_experiment import ExperimentFailure
from ipindia_corpus.passport.api import create_app
from ipindia_corpus.passport.readiness import EngineReadiness
from ipindia_corpus.passport.service import CountingProvider, Intake, PassportEngine
from test_passport import Search, intake
from test_reliable_answers import Provider


class Remote(Provider):
    failed_binding = False
    metadata = {'provider': 'gemini', 'model': 'gemini-3.5-flash-lite', 'model_version': 'gemini-3.5-flash-lite',
                'model_digest': None, 'external_processing': True}


def remote(outputs=None):
    return Remote(outputs or [json.dumps({'issues': [{'topic_id': 'traditional_knowledge', 'context_id': 'I001'}],
                                          'needs_clarification': False}), '{"span_ids":["S001"]}', '{"span_ids":["S001"]}'])


class GeminiSafetyTests(unittest.TestCase):
    def test_no_remote_consent_means_zero_calls_and_true_preserves_fidelity(self):
        p = remote()
        e = PassportEngine(Search(), lambda *args: None, provider=p)
        result = e.build(Intake(**intake()))
        self.assertEqual(result['reason'], 'external_processing_consent_required')
        self.assertFalse(p.calls)
        self.assertFalse(result['verified_evidence'])
        result = e.build(Intake(**intake(acknowledged_external_processing=True)))
        self.assertEqual(result['status'], 'ready')
        self.assertEqual(len(p.calls), 3)
        self.assertEqual(result['verified_evidence'][0]['exact_quote'], e.search.evidence[0]['text'])
        self.assertIsNone(result['model']['digest'])
        self.assertEqual(result['model']['provider'], 'gemini')
        with self.assertRaises(ValidationError):
            Intake(**intake(acknowledged_external_processing='true'))

    def test_late_provider_result_is_failure_not_completion(self):
        clock = [0.0]
        p = Provider(['{"span_ids":[]}'])
        original = p.chat
        def late(fields):
            clock[0] = 91.0
            return original(fields)
        p.chat = late
        with patch('ipindia_corpus.passport.service.time.monotonic', side_effect=lambda: clock[0]):
            counted = CountingProvider(p)
            with self.assertRaisesRegex(ExperimentFailure, '^timeout$'):
                counted.chat({'messages': []})
        self.assertEqual(counted.completed, 0)
        self.assertEqual(counted.observations[0]['failure_reason'], 'timeout')
        self.assertFalse(counted.observations[0]['completed'])

    def test_later_issue_model_drift_invalidates_whole_passport_and_readiness(self):
        plan = json.dumps({'issues': [{'topic_id': 'traditional_knowledge', 'context_id': 'I001'},
                                     {'topic_id': 'novelty_inventive_step', 'context_id': 'I001'}],
                           'needs_clarification': False})
        p = remote([plan, '{"span_ids":["S001"]}', '{"span_ids":["S001"]}'])
        original = p.chat
        def drift(fields):
            if len(p.calls) == 3:
                p.failed_binding = True
                raise ExperimentFailure('provider_model_changed')
            return original(fields)
        p.chat = drift
        e = PassportEngine(Search(), lambda *args: None, provider=p)
        result = e.build(Intake(**intake(acknowledged_external_processing=True)))
        self.assertEqual(result['status'], 'unavailable')
        self.assertEqual(result['reason'], 'provider_model_changed')
        self.assertFalse(result['verified_evidence'])
        self.assertTrue(all(not issue['evidence_ids'] for issue in result['issues']))
        self.assertFalse(e.ready())
        snapshot = EngineReadiness(engine=e).snapshot()
        self.assertFalse(snapshot['ready'])
        self.assertEqual(snapshot['reason'], 'provider_model_changed')

    def test_planner_failure_reasons_remain_actionable(self):
        for reason in ('provider_auth_failed', 'provider_rate_limited', 'provider_model_unavailable', 'timeout'):
            with self.subTest(reason=reason):
                p = remote()
                p.chat = lambda fields, reason=reason: (_ for _ in ()).throw(ExperimentFailure(reason))
                result = PassportEngine(Search(), lambda *args: None, provider=p).build(
                    Intake(**intake(acknowledged_external_processing=True)))
                self.assertEqual(result['reason'], reason)
                self.assertEqual(result['model']['calls_completed'], 0)
                self.assertEqual(result['performance']['model_requests'][0]['failure_reason'], reason)

    def test_isolated_port_retains_host_origin_and_csrf_checks(self):
        from test_passport import engine
        app = create_app(engine=engine(), enabled=True, port=18003)
        async def loopback(scope, receive, send):
            await app({**scope, 'client': ('127.0.0.1', 1234)}, receive, send)
        origin = 'http://127.0.0.1:18003'
        with TestClient(loopback, base_url=origin) as client:
            session = client.get('/api/session', headers={'Origin': origin})
            self.assertEqual(session.status_code, 200)
            self.assertEqual(client.get('/api/session', headers={'Origin': 'http://127.0.0.1:8003'}).status_code, 403)
            self.assertEqual(client.get('/api/session', headers={'Host': '127.0.0.1:8003'}).status_code, 400)
            self.assertEqual(client.post('/api/passport', json=intake(), headers={'Origin': origin}).status_code, 403)
            self.assertEqual(client.post('/api/passport', json=intake(), headers={'Origin': origin,
                             'X-Passport-Token': 'wrong-session'}).status_code, 403)
        for port in (True, 8000, 443):
            with self.assertRaises(ValueError):
                create_app(enabled=True, port=port)


if __name__ == '__main__':
    unittest.main()
