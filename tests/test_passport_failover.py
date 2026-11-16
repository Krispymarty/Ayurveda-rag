"""Synthetic forced failures; no API credentials or legal relevance labels."""
import json
import unittest

from ipindia_corpus.llm_experiment import ExperimentFailure
from ipindia_corpus.passport.service import Intake, PassportEngine
from test_gemini_safety import Remote
from test_passport import Search, intake
from test_reliable_answers import Provider

PLAN = json.dumps({'issues': [{'topic_id': 'traditional_knowledge', 'context_id': 'I001'}], 'needs_clarification': False})
SELECT = '{"span_ids":["S001"]}'


class BrokenRemote(Remote):
    def __init__(self, *, reason='provider_unavailable', before=0):
        super().__init__([PLAN, SELECT, SELECT])
        self.reason, self.before = reason, before

    def chat(self, fields):
        if len(self.calls) >= self.before:
            self.calls.append(fields)
            raise ExperimentFailure(self.reason)
        return super().chat(fields)


class PassportFailoverTests(unittest.TestCase):
    def build(self, primary, local=None, *, verify=None, search=None, **changes):
        local = local or Provider([PLAN, SELECT, SELECT])
        engine = PassportEngine(search or Search(), verify or (lambda *args: None),
                                provider=primary, fallback_provider=local)
        result = engine.build(Intake(**intake(acknowledged_external_processing=True, **changes)))
        return engine, local, result

    def test_primary_success_never_calls_fallback(self):
        _, local, result = self.build(Remote([PLAN, SELECT, SELECT]))
        self.assertEqual(result['status'], 'ready')
        self.assertFalse(local.calls)
        self.assertFalse(result['routing']['fallback_used'])
        self.assertEqual(result['model']['provider'], 'gemini')

    def test_each_transient_failure_starts_fresh_local_pipeline(self):
        for reason in ('timeout', 'provider_rate_limited', 'provider_unavailable'):
            with self.subTest(reason=reason):
                _, local, result = self.build(BrokenRemote(reason=reason))
                self.assertEqual(result['status'], 'ready')
                self.assertEqual(len(local.calls), 3)
                self.assertEqual(result['model']['provider'], 'ollama')
                self.assertTrue(result['routing']['fallback_used'])
                self.assertEqual(result['routing']['calls_attempted_total'], 4)
                self.assertTrue(result['routing']['external_processing_attempted'])

    def test_failed_verifier_discards_primary_and_restarts_planning(self):
        _, local, result = self.build(BrokenRemote(before=2))
        self.assertEqual(result['status'], 'ready')
        self.assertEqual(local.calls[0]['format']['required'], ['issues', 'needs_clarification'])
        self.assertEqual(result['routing']['calls_attempted_total'], 6)
        self.assertTrue(result['routing']['attempts'][0]['discarded'])

    def test_no_fallback_for_policy_safety_integrity_or_quality_failures(self):
        for reason in ('provider_auth_failed', 'provider_request_rejected', 'provider_redirect_rejected',
                       'provider_safety_blocked', 'provider_invalid_response', 'provider_invalid_selection',
                       'provider_model_changed', 'evidence_context_limit'):
            with self.subTest(reason=reason):
                _, local, result = self.build(BrokenRemote(reason=reason))
                self.assertFalse(local.calls)
                self.assertFalse(result['routing']['fallback_used'])
        _, local, result = self.build(Remote([PLAN, '{"span_ids":[]}']))
        self.assertFalse(local.calls)
        self.assertEqual(result['status'], 'partial')

    def test_consent_and_current_status_block_before_either_model(self):
        primary, local = BrokenRemote(), Provider([PLAN, SELECT, SELECT])
        engine = PassportEngine(Search(), lambda *args: None, provider=primary, fallback_provider=local)
        for changes in ({'acknowledged_external_processing': False},
                        {'description': 'Is this patent currently in force today?'}):
            payload = intake(acknowledged_external_processing=True)
            payload.update(changes)
            self.assertEqual(engine.build(Intake(**payload))['status'], 'blocked')
        self.assertFalse(primary.calls)
        self.assertFalse(local.calls)

    def test_corrupt_source_never_triggers_local_model(self):
        def broken(*args):
            raise ValueError('synthetic archive failure')
        _, local, result = self.build(Remote([PLAN, SELECT, SELECT]), verify=broken)
        self.assertFalse(local.calls)
        self.assertFalse(result['verified_evidence'])

    def test_both_unavailable_release_no_partial_evidence(self):
        local = BrokenRemote()
        # Explicitly label this injected fixture as local; real fallback checks digest.
        local.metadata = {'provider': 'ollama', 'model': 'qwen3.5:4b',
                          'model_digest': 'd8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04',
                          'model_version': None, 'external_processing': False}
        _, _, result = self.build(BrokenRemote(), local=local)
        self.assertEqual(result['status'], 'unavailable')
        self.assertFalse(result['verified_evidence'])

    def test_rate_limit_circuit_skips_primary_on_next_request(self):
        primary = BrokenRemote(reason='provider_rate_limited')
        local = Provider([PLAN, SELECT, SELECT, PLAN, SELECT, SELECT])
        engine, _, first = self.build(primary, local=local)
        second = engine.build(Intake(**intake(acknowledged_external_processing=True)))
        self.assertEqual(first['status'], 'ready')
        self.assertEqual(second['status'], 'ready')
        self.assertEqual(len(primary.calls), 1)
        self.assertFalse(second['routing']['external_processing_attempted'])
        self.assertEqual(second['routing']['trigger'], 'provider_circuit_open')

    def test_fallback_valid_abstention_is_preserved(self):
        _, _, result = self.build(BrokenRemote(), local=Provider([PLAN, '{"span_ids":[]}']))
        self.assertEqual(result['status'], 'partial')
        self.assertFalse(result['verified_evidence'])
        self.assertTrue(result['routing']['fallback_used'])

    def test_fresh_three_topic_fallback_fits_own_seven_call_limit(self):
        from ipindia_corpus.passport.service import TOPICS
        topics = list(TOPICS)[:3]
        plan = json.dumps({'issues': [{'topic_id': t, 'context_id': 'I001'} for t in topics], 'needs_clarification': False})
        local = Provider([plan, SELECT, SELECT, SELECT, SELECT, SELECT, SELECT])
        _, _, result = self.build(BrokenRemote(), local=local)
        self.assertEqual(result['routing']['calls_attempted_total'], 8)
        self.assertEqual(len(local.calls), 7)
        self.assertEqual(result['status'], 'ready')
        self.assertTrue(result['verified_evidence'])
        self.assertTrue(all(a['calls_attempted'] <= 7 for a in result['routing']['attempts']))

    def test_denied_model_call_clears_earlier_fallback_evidence(self):
        from unittest.mock import patch
        from ipindia_corpus.passport.failover import RequestBudget
        from ipindia_corpus.passport.service import TOPICS
        plan = json.dumps({'issues': [{'topic_id': t, 'context_id': 'I001'} for t in list(TOPICS)[:2]], 'needs_clarification': False})
        local = Provider([plan, SELECT, SELECT, SELECT, SELECT])
        original = RequestBudget.claim
        def deny(budget, deadline=None):
            if budget.attempted >= 3:
                raise ExperimentFailure('timeout')
            return original(budget, deadline)
        with patch.object(RequestBudget, 'claim', deny):
            _, _, result = self.build(BrokenRemote(), local=local)
        self.assertEqual(result['status'], 'unavailable')
        self.assertFalse(result['verified_evidence'])
        self.assertTrue(all(not i['evidence_ids'] for i in result['issues']))

    def test_generation_change_prevents_fallback(self):
        search = Search()
        primary = BrokenRemote()
        original = primary.chat
        def mutate(fields):
            search.gid = 'changed'
            return original(fields)
        primary.chat = mutate
        _, local, result = self.build(primary, search=search)
        self.assertFalse(local.calls)
        self.assertFalse(result['verified_evidence'])

    def test_no_primary_quotes_survive_a_later_topic_failure(self):
        from test_span_selection import hit
        primary = BrokenRemote(before=3)
        primary.outputs = iter([json.dumps({'issues': [
            {'topic_id': 'traditional_knowledge', 'context_id': 'I001'},
            {'topic_id': 'novelty_inventive_step', 'context_id': 'I001'}], 'needs_clarification': False}), SELECT, SELECT])
        local = Provider([PLAN, '{"span_ids":["S002"]}', '{"span_ids":["S002"]}'])
        _, _, result = self.build(primary, local=local, search=Search(evidence=[hit(), hit('local-only')]))
        self.assertEqual(result['status'], 'ready')
        self.assertEqual(result['routing']['calls_attempted_total'], 7)
        self.assertEqual([h['chunk_id'] for h in result['verified_evidence']], ['local-only'])

    def test_late_final_archive_check_cannot_release_after_shared_deadline(self):
        clock, checks = [1000.0], []
        def verify(*args):
            checks.append(True)
            if len(checks) == 3:
                clock[0] += 91
        local = Provider([PLAN, SELECT, SELECT])
        engine = PassportEngine(Search(), verify, provider=Remote([PLAN, SELECT, SELECT]),
                                fallback_provider=local, clock=lambda: clock[0])
        result = engine.build(Intake(**intake(acknowledged_external_processing=True)))
        self.assertEqual(result['status'], 'unavailable')
        self.assertEqual(result['reason'], 'timeout')
        self.assertFalse(result['verified_evidence'])
        self.assertFalse(local.calls)

    def test_primary_subdeadline_includes_final_checks_before_fallback(self):
        clock, checks = [1000.0], []
        def verify(*args):
            checks.append(True)
            if len(checks) == 3:
                clock[0] += 31
        local = Provider([PLAN, SELECT, SELECT])
        engine = PassportEngine(Search(), verify, provider=Remote([PLAN, SELECT, SELECT]),
                                fallback_provider=local, clock=lambda: clock[0])
        result = engine.build(Intake(**intake(acknowledged_external_processing=True)))
        self.assertEqual(result['status'], 'ready')
        self.assertEqual(result['model']['provider'], 'ollama')
        self.assertEqual(result['routing']['attempts'][0]['reason'], 'timeout')

    def test_late_archive_failure_keeps_its_cause_and_never_falls_back(self):
        clock, checks = [1000.0], []
        def verify(*args):
            checks.append(True)
            if len(checks) == 3:
                clock[0] += 31
                raise ValueError('synthetic late corrupt original')
        local = Provider([PLAN, SELECT, SELECT])
        engine = PassportEngine(Search(), verify, provider=Remote([PLAN, SELECT, SELECT]),
                                fallback_provider=local, clock=lambda: clock[0])
        result = engine.build(Intake(**intake(acknowledged_external_processing=True)))
        self.assertEqual(result['reason'], 'archive_verification_failed')
        self.assertEqual(result['status'], 'unavailable')
        self.assertFalse(result['verified_evidence'])
        self.assertFalse(local.calls)

    def test_generation_change_during_last_verification_rejects_result(self):
        search, checks = Search(), []
        def verify(*args):
            checks.append(True)
            if len(checks) == 3:
                search.gid = 'changed-during-final-check'
        _, local, result = self.build(Remote([PLAN, SELECT, SELECT]), search=search, verify=verify)
        self.assertEqual(result['status'], 'unavailable')
        self.assertEqual(result['reason'], 'active_generation_changed')
        self.assertFalse(result['verified_evidence'])
        self.assertFalse(local.calls)


if __name__ == '__main__':
    unittest.main()
