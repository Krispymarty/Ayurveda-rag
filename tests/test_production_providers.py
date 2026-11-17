"""Production provider identity, context acceptance and private key isolation."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from ipindia_corpus.inference import configured_identity, make_provider
from ipindia_corpus.reliable_answers import configuration
from ipindia_corpus.groq_candidate import GroqCandidate
from ipindia_corpus.llm_experiment import ExperimentFailure
from test_groq_candidate import FIELDS, KEY, Response, Session, generated


class ProductionProviderTests(unittest.TestCase):
    def test_default_factory_uses_groq_and_explicit_gemini_is_preserved(self):
        with patch('ipindia_corpus.inference.read_setting', side_effect=lambda name, default: default), \
                patch('ipindia_corpus.groq_candidate.GroqCandidate') as groq:
            self.assertIs(make_provider(), groq.return_value)
        groq.assert_called_once_with(usage_checked_context=True)
        with patch('ipindia_corpus.inference.GeminiProvider') as gemini:
            self.assertIs(make_provider('gemini'), gemini.return_value)

    def test_default_configured_identity_has_no_invented_digest(self):
        with patch('ipindia_corpus.inference.read_setting', side_effect=lambda name, default: default):
            identity = configured_identity()
        self.assertEqual(identity['provider'], 'groq')
        self.assertEqual(identity['model'], 'qwen/qwen3.8-27b')
        self.assertIsNone(identity['model_digest'])
        self.assertTrue(identity['external_processing'])

    def test_groq_profile_survives_counting_wrapper(self):
        p = GroqCandidate(api_key=KEY, usage_checked_context=True)
        class Wrapper:
            provider = p
            metadata = p.metadata
        result = configuration(Wrapper())
        self.assertEqual(result['model'], 'qwen/qwen3.8-27b')
        self.assertIsNone(result['model_digest'])
        self.assertIsNone(result['keep_alive'])
        self.assertEqual(result['profile']['num_predict'], 200)
        self.assertEqual(result['profile']['context_enforcement'], 'bounded_utf8_bytes_then_returned_usage')

    def test_runtime_accepts_larger_bytes_only_with_valid_returned_usage(self):
        model = 'qwen/qwen3.8-27b'
        fields = deepcopy(FIELDS)
        fields['messages'][1]['content'] = 'Archived legal text. ' * 250
        good = GroqCandidate(api_key=KEY, model=model, usage_checked_context=True,
                             session=Session(Response(generated(model=model))))
        good.chat(fields)
        bad = GroqCandidate(api_key=KEY, model=model, usage_checked_context=True,
            session=Session(Response(generated(model=model, usage={
                'prompt_tokens': 3900, 'completion_tokens': 20, 'total_tokens': 3920}))))
        with self.assertRaisesRegex(ExperimentFailure, '^evidence_context_limit$'):
            bad.chat(fields)
        fields['messages'][1]['content'] = 'x' * 48_001
        with self.assertRaisesRegex(ExperimentFailure, '^evidence_context_limit$'):
            good.chat(fields)


if __name__ == '__main__':
    unittest.main()
