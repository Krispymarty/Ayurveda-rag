"""Small request limits and availability circuit for the private Passport.

These controls never inspect evidence or model text. A provider's policy,
integrity or structured-output rejection is not an availability failure.
"""
from __future__ import annotations

import math
import threading
import time

from ..llm_experiment import ExperimentFailure


FALLBACK_REASONS = frozenset({'timeout', 'provider_rate_limited', 'provider_unavailable'})


def _positive_number(value):
    return (type(value) in (int, float) and math.isfinite(value) and value > 0)


def _now(clock):
    value = clock()
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('invalid_failover_clock')
    return value


class RequestBudget:
    """Seven calls per attempt, bounded by an optional original request deadline."""

    def __init__(self, *, clock=time.monotonic, timeout=90, max_calls=7, deadline=None):
        if (not callable(clock) or not _positive_number(timeout) or timeout > 90
                or type(max_calls) is not int or not 1 <= max_calls <= 7):
            raise ValueError('invalid_request_budget')
        self.clock, self.max_calls = clock, max_calls
        self.started = _now(clock)
        self.deadline = self.started + timeout
        if deadline is not None:
            if type(deadline) not in (int, float) or not math.isfinite(deadline):
                raise ValueError('invalid_request_budget')
            self.deadline = min(self.deadline, deadline)
        self.attempted = 0
        self._lock = threading.Lock()

    def claim(self, deadline=None):
        """Consume a call only while both the shared and optional shorter limit fit."""
        with self._lock:
            if deadline is not None and (type(deadline) not in (int, float)
                                         or not math.isfinite(deadline)):
                raise ExperimentFailure('timeout')
            bounded = self.deadline if deadline is None else min(self.deadline, deadline)
            if self.attempted >= self.max_calls or _now(self.clock) >= bounded:
                raise ExperimentFailure('timeout')
            self.attempted += 1
            return bounded


class Circuit:
    """Pause primary calls after repeated availability failures or a rate limit.

    After the cooldown, a primary attempt is allowed. Success closes the circuit;
    another eligible failure reopens it. No provider or user payload is retained.
    """

    def __init__(self, *, clock=time.monotonic, threshold=3, cooldown=30,
                 rate_limit_cooldown=60):
        if (not callable(clock) or type(threshold) is not int or threshold < 1
                or not _positive_number(cooldown)
                or not _positive_number(rate_limit_cooldown)):
            raise ValueError('invalid_failover_circuit')
        self.clock, self.threshold = clock, threshold
        self.cooldown, self.rate_limit_cooldown = cooldown, rate_limit_cooldown
        self._failure_count = 0
        self._open_until = None
        self._lock = threading.Lock()

    def skip_primary(self):
        with self._lock:
            return self._open_until is not None and _now(self.clock) < self._open_until

    def failure(self, reason):
        """Count only an allowlisted failure; return whether it was eligible."""
        if not isinstance(reason, str) or reason not in FALLBACK_REASONS:
            return False
        with self._lock:
            self._failure_count += 1
            if reason == 'provider_rate_limited' or self._failure_count >= self.threshold:
                duration = (self.rate_limit_cooldown if reason == 'provider_rate_limited'
                            else self.cooldown)
                until = _now(self.clock) + duration
                self._open_until = (until if self._open_until is None
                                    else max(self._open_until, until))
        return True

    def success(self):
        with self._lock:
            self._failure_count, self._open_until = 0, None

    def snapshot(self):
        with self._lock:
            remaining = (0 if self._open_until is None
                         else max(0, self._open_until - _now(self.clock)))
            state = ('closed' if self._open_until is None else
                     'open' if remaining > 0 else 'half_open')
            return {'state': state, 'retry_after': math.ceil(remaining),
                    'failure_count': self._failure_count}
