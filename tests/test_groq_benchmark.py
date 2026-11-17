"""Offline orchestration fixtures; no API calls, token downloads or model loads."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ipindia_corpus.inference import make_provider
from ipindia_corpus.llm_experiment import ExperimentFailure
from ipindia_corpus.reliable_answers import selection_fields
from ipindia_corpus.span_selection import build_plan
from test_evidence_quality import entry
from tools import benchmark_evidence_quality as benchmark


class Clock:
    def __init__(self):
        self.now = 100.0
    def monotonic(self):
        return self.now
    def sleep(self, seconds):
        self.now += seconds


class Candidate:
    def __init__(self, provider, clock, *, rate_limit_after_warmup=False):
        self.provider, self.clock = provider, clock
        self.calls = []
        self.rate_limit_after_warmup = rate_limit_after_warmup
        self.last_call_diagnostics = {}
        self.profile = {'num_ctx': 4096, 'num_predict': 200, 'temperature': 0,
                        'reasoning_effort': 'low', 'include_reasoning': False,
                        'grammar_normalization': 'drop_uniqueItems_backend_enforced'}

    @property
    def metadata(self):
        return {'provider': self.provider,
                'model': 'openai/gpt-oss-120b' if self.provider == 'groq' else 'qwen3.5:4b',
                'model_version': 'openai/gpt-oss-120b' if self.provider == 'groq' else None,
                'model_digest': None if self.provider == 'groq' else 'd8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04',
                'external_processing': self.provider == 'groq'}

    def chat(self, fields):
        self.calls.append(deepcopy(fields))
        self.last_call_diagnostics = {}
        if self.rate_limit_after_warmup and len(self.calls) == 2:
            self.clock.now += 0.125
            self.last_call_diagnostics = {'http_status': 429}
            raise ExperimentFailure('provider_rate_limited')
        self.clock.now += 0.01
        allowed = fields['format']['properties']['span_ids']['items']['enum']
        return json.dumps({'span_ids': allowed}), {'fixture': True}


def prepared_fixture():
    items = []
    for identifier in ('question-one', 'question-two'):
        item = deepcopy(entry())
        item['case']['id'] = identifier
        item['plan'] = build_plan(item['case']['evidence'])
        item['fields'] = selection_fields(item['case']['question'], item['plan'])
        item['input_sha256'] = benchmark.digest(item['fields'])
        items.append(item)
    return {'generation_id': 'synthetic-generation', 'entries': items,
            'bundle_sha256': 'b' * 64, 'independent_legal_accuracy_score': None}


class GroqBenchmarkTests(unittest.TestCase):
    def test_rate_limit_stops_groq_only_and_failure_timing_excludes_pacing(self):
        clock = Clock()
        groq = Candidate('groq', clock, rate_limit_after_warmup=True)
        qwen = Candidate('ollama', clock)
        prepared = prepared_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # The isolated runner must bind every named code file before calls.
            # These tiny copies exercise that check without touching real sources.
            for name in ('tools/benchmark_evidence_quality.py', 'ipindia_corpus/reliable_answers.py',
                         'ipindia_corpus/span_selection.py', 'ipindia_corpus/llm_scoring.py',
                         'ipindia_corpus/inference.py', 'ipindia_corpus/groq_candidate.py'):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('# controlled synthetic code-binding fixture\n', encoding='utf-8')
            bundle = root / 'bundle.json'
            bundle.write_text(json.dumps({'synthetic': True}), encoding='utf-8')
            output = root / 'data' / 'run'
            with patch.object(benchmark, 'prepare', return_value=prepared) as prepare, \
                    patch.object(benchmark, 'evaluation_provider', side_effect=lambda name: {'groq': groq, 'ollama': qwen}[name]), \
                    patch('tools.benchmark_evidence_quality.time.monotonic', side_effect=clock.monotonic), \
                    patch('tools.benchmark_evidence_quality.time.time', side_effect=clock.monotonic), \
                    patch('tools.benchmark_evidence_quality.time.sleep', side_effect=clock.sleep), \
                    patch('importlib.metadata.version', return_value='synthetic-version'):
                benchmark.run(bundle, output, providers=('groq', 'ollama'), warmup=True, root=root)
            summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(prepare.call_count, 2)
            self.assertEqual(len(groq.calls), 2)  # warmup + first question, no retry
            self.assertEqual(len(qwen.calls), 5)  # warmup + selection/verifier twice
            self.assertEqual(summary['stopped_providers'], ['groq'])
            self.assertEqual(summary['metrics']['groq']['attempted_questions'], 1)
            self.assertEqual(summary['metrics']['groq']['unexecuted_questions'], 1)
            self.assertEqual(summary['metrics']['groq']['completed_questions'], 0)
            self.assertEqual(summary['metrics']['ollama']['completed_questions'], 2)
            self.assertEqual(summary['metrics']['ollama']['final_strict_success_completed'], 1)
            self.assertFalse(summary['activation_performed'])
            failure = json.loads((output / '001-groq' / 'selection.call_failure.json').read_text(encoding='utf-8'))
            row = json.loads((output / '001-groq' / 'result.json').read_text(encoding='utf-8'))
            self.assertEqual(failure['failure'], 'provider_rate_limited')
            self.assertEqual(failure['diagnostics'], {'http_status': 429})
            self.assertAlmostEqual(failure['elapsed_ms'], 125, places=5)
            self.assertGreater(row['elapsed_ms'], 19_000)  # pacing belongs to row wall time
            self.assertLess(failure['elapsed_ms'], row['elapsed_ms'])
            self.assertFalse((output / '002-groq').exists())

    def test_candidate_configuration_never_inherits_qwen_identity_or_keepalive(self):
        candidate = Candidate('groq', Clock())
        with patch('importlib.metadata.version', return_value='synthetic-version'):
            config = benchmark.evaluation_configuration(candidate)
        self.assertEqual(config['provider'], 'groq')
        self.assertEqual(config['model'], 'openai/gpt-oss-120b')
        self.assertIsNone(config['model_digest'])
        self.assertIsNone(config['keep_alive'])
        self.assertEqual(config['profile'], candidate.profile)
        self.assertNotIn('qwen3.5:4b', json.dumps(config))

    def test_unattempted_provider_has_no_fabricated_quality_score(self):
        summary = benchmark.summarize([], 2, providers=['groq'])['groq']
        self.assertEqual(summary['attempted_questions'], 0)
        self.assertEqual(summary['unexecuted_questions'], 2)
        self.assertIsNone(summary['final_strict_success_completed'])
        self.assertIsNone(summary['final_strict_success_all_attempted'])
        self.assertIsNone(summary['independent_legal_accuracy_score'])

    def test_production_factory_still_rejects_groq(self):
        with patch('ipindia_corpus.inference.GeminiProvider') as remote:
            with self.assertRaisesRegex(ExperimentFailure, '^gemini_configuration_invalid$'):
                make_provider('groq')
        remote.assert_not_called()


if __name__ == '__main__':
    unittest.main()
