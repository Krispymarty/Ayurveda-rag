"""Offline limits and circuit controls; no provider calls or legal judgments."""
from concurrent.futures import ThreadPoolExecutor
import unittest

from ipindia_corpus.llm_experiment import ExperimentFailure
from ipindia_corpus.passport.failover import Circuit, FALLBACK_REASONS, RequestBudget


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class FailoverControlTests(unittest.TestCase):
    def test_fallback_budget_cannot_extend_parent_deadline(self):
        clock = Clock()
        parent = RequestBudget(clock=clock)
        clock.now = 130
        child = RequestBudget(clock=clock, deadline=parent.deadline)
        self.assertEqual(child.deadline, 190)
        self.assertEqual(child.claim(), 190)
        clock.now = 190
        with self.assertRaisesRegex(ExperimentFailure, '^timeout$'):
            child.claim()
        for deadline in [True, float('nan'), '190']:
            with self.assertRaises(ValueError):
                RequestBudget(clock=clock, deadline=deadline)

    def test_reasons_are_only_temporary_availability_failures(self):
        self.assertEqual(FALLBACK_REASONS,
                         frozenset({'timeout', 'provider_rate_limited', 'provider_unavailable'}))

    def test_budget_is_shared_and_never_extends_the_absolute_deadline(self):
        clock = Clock()
        budget = RequestBudget(clock=clock)
        self.assertEqual(budget.started, 100)
        self.assertEqual(budget.deadline, 190)
        self.assertEqual(budget.claim(), 190)
        self.assertEqual(budget.claim(deadline=130), 130)
        self.assertEqual(budget.claim(deadline=999), 190)
        self.assertEqual(budget.attempted, 3)
        clock.now = 190
        with self.assertRaisesRegex(ExperimentFailure, '^timeout$'):
            budget.claim()
        self.assertEqual(budget.attempted, 3)

    def test_exhausted_attempt_budget_and_expired_override_do_not_consume_calls(self):
        clock = Clock()
        budget = RequestBudget(clock=clock, max_calls=2)
        with self.assertRaisesRegex(ExperimentFailure, '^timeout$'):
            budget.claim(deadline=100)
        self.assertEqual(budget.attempted, 0)
        budget.claim()
        budget.claim()
        with self.assertRaisesRegex(ExperimentFailure, '^timeout$'):
            budget.claim()
        self.assertEqual(budget.attempted, 2)

    def test_concurrent_claims_cannot_exceed_seven_calls(self):
        budget = RequestBudget(clock=Clock())

        def claim(_):
            try:
                budget.claim()
                return True
            except ExperimentFailure:
                return False

        with ThreadPoolExecutor(max_workers=16) as pool:
            completed = list(pool.map(claim, range(100)))
        self.assertEqual(sum(completed), 7)
        self.assertEqual(budget.attempted, 7)

    def test_three_failures_open_then_cooldown_and_success_reset(self):
        clock = Clock()
        circuit = Circuit(clock=clock)
        self.assertFalse(circuit.skip_primary())
        for _ in range(2):
            circuit.failure('provider_unavailable')
            self.assertFalse(circuit.skip_primary())
        circuit.failure('timeout')
        self.assertTrue(circuit.skip_primary())
        self.assertEqual(circuit.snapshot(),
                         {'state': 'open', 'retry_after': 30, 'failure_count': 3})
        clock.now = 130
        self.assertFalse(circuit.skip_primary())
        self.assertEqual(circuit.snapshot()['state'], 'half_open')
        circuit.failure('timeout')
        self.assertTrue(circuit.skip_primary())
        circuit.success()
        self.assertEqual(circuit.snapshot(),
                         {'state': 'closed', 'retry_after': 0, 'failure_count': 0})

    def test_rate_limit_opens_immediately_and_later_failure_does_not_shorten_it(self):
        clock = Clock()
        circuit = Circuit(clock=clock, threshold=1)
        circuit.failure('provider_rate_limited')
        self.assertEqual(circuit.snapshot()['retry_after'], 60)
        clock.now = 110
        circuit.failure('timeout')
        self.assertEqual(circuit.snapshot()['retry_after'], 50)

    def test_policy_integrity_and_configuration_failures_never_open_circuit(self):
        circuit = Circuit(clock=Clock(), threshold=1)
        for reason in ('provider_auth_failed', 'provider_safety_blocked',
                       'provider_invalid_selection', 'provider_model_changed',
                       'archive_verification_failed', 'insufficient_relevant_evidence',
                       'gemini_key_missing', 'provider_redirect_rejected'):
            circuit.failure(reason)
        self.assertFalse(circuit.skip_primary())
        self.assertEqual(circuit.snapshot()['failure_count'], 0)

    def test_concurrent_circuit_updates_are_not_lost(self):
        circuit = Circuit(clock=Clock())
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(circuit.failure, ['provider_unavailable'] * 32))
        self.assertEqual(circuit.snapshot()['failure_count'], 32)
        self.assertTrue(circuit.skip_primary())

    def test_invalid_limits_cannot_expand_or_disable_existing_request_limits(self):
        for kwargs in ({'timeout': 91}, {'timeout': 0}, {'timeout': True},
                       {'timeout': float('nan')}, {'max_calls': 8}, {'max_calls': 0},
                       {'max_calls': True}, {'clock': None}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RequestBudget(**kwargs)
        for kwargs in ({'threshold': 0}, {'threshold': True}, {'cooldown': -1},
                       {'cooldown': float('inf')}, {'rate_limit_cooldown': False}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Circuit(**kwargs)


if __name__ == '__main__':
    unittest.main()
