"""Synthetic structural checks, never legal relevance certification."""
from copy import deepcopy
import json
import unittest

from ipindia_corpus.span_selection import build_plan
from tools.benchmark_evidence_quality import failure_category, qualification_coverage, score_output, summarize


def hit(identifier, text):
    return {'chunk_id': identifier, 'document_id': 'doc', 'text': text,
            'source': 'ipindia', 'source_url': 'https://ipindia.gov.in/fixture.pdf',
            'raw_sha256': 'a' * 64, 'provenance': 'verified_immutable_archive',
            'evidence_role': 'legal_source', 'status': 'unknown'}


def entry():
    text = 'Synthetic requirement preserves this qualification unless the stated exception applies.'
    evidence = [hit('one', text), hit('two', 'A distinct synthetic provision supplies the second necessary requirement.')]
    case = {'id': 'fixture', 'question': 'Quote both synthetic requirements.', 'evidence': evidence,
            'expected': {'policy': 'quote_targets', 'allowed_claim_chunk_ids': ['one', 'two'],
                         'target_groups': [{'required_chunk_ids': ['one'], 'required_contains': 'Synthetic requirement'},
                                           {'required_chunk_ids': ['two'], 'required_contains': 'second necessary requirement'}]}}
    start = text.index('unless')
    return {'case': case, 'qualifications': [{'chunk_id': 'one', 'start': start, 'end': len(text),
                                             'quote': text[start:], 'kind': 'exception'}]}


class EvidenceQualityTests(unittest.TestCase):
    def test_complete_requires_both_groups_but_exception_remains_separate(self):
        item = entry()
        plan = build_plan(item['case']['evidence'])
        partial = score_output(item, '{"span_ids":["S001"]}', plan)
        self.assertEqual(partial['score']['target_coverage'], 0.5)
        self.assertFalse(partial['score']['passed'])
        self.assertTrue(partial['all_qualifications_retained'])
        full = score_output(item, '{"span_ids":["S001","S002"]}', plan)
        self.assertTrue(full['score']['passed'])
        self.assertTrue(full['backend_accepted'])
        self.assertEqual(full['expanded_selection']['claims'][0]['quote'], item['case']['evidence'][0]['text'])

    def test_missing_exception_not_masked_by_valid_citation(self):
        item = entry()
        plan = build_plan(item['case']['evidence'])
        result = score_output(item, '{"span_ids":["S002"]}', plan)
        self.assertTrue(result['score']['source_valid'])
        self.assertFalse(result['all_qualifications_retained'])

    def test_unicode_offsets_and_cross_span_exception_require_complete_union(self):
        text = 'परीक्षण 🌿 ' * 200
        evidence = [hit('one', text)]
        plan = build_plan(evidence)
        spans = list(plan['spans'])
        first = plan['spans'][spans[0]]
        start, end = first['end'] - 5, first['end'] + 8
        qualification = [{'chunk_id': 'one', 'start': start, 'end': end, 'quote': text[start:end]}]
        self.assertFalse(qualification_coverage(qualification, evidence, plan, spans[:1])[0]['retained'])
        self.assertTrue(qualification_coverage(qualification, evidence, plan, spans[:2])[0]['retained'])
        qualification[0]['quote'] = 'altered'
        with self.assertRaisesRegex(ValueError, 'qualification_annotation_mismatch'):
            qualification_coverage(qualification, evidence, plan, spans)

    def test_unknown_ids_and_verifier_additions_fail_without_quote_credit(self):
        item = entry()
        plan = build_plan(item['case']['evidence'])
        for raw, subset in [('{"span_ids":["fake"]}', None),
                            ('{"span_ids":["S001","S002"]}', ['S001'])]:
            result = score_output(item, raw, plan, selected_subset=subset)
            self.assertFalse(result['valid_selection'])
            self.assertFalse(result['score']['passed'])
            self.assertFalse(result['qualification_results'][0]['retained'])

    def test_annotation_changes_cannot_change_model_fields(self):
        from ipindia_corpus.reliable_answers import selection_fields
        item = entry()
        before = selection_fields(item['case']['question'], build_plan(item['case']['evidence']))
        item['qualifications'] = []
        item['case']['expected']['target_groups'].reverse()
        self.assertEqual(before, selection_fields(item['case']['question'], build_plan(item['case']['evidence'])))

    def test_failure_denominators_and_legal_accuracy_not_fabricated(self):
        item = entry()
        result = score_output(item, '{"span_ids":["S001","S002"]}', build_plan(item['case']['evidence']))
        completed = {'provider': 'gemini', 'case_id': 'one', 'policy': 'quote_targets', 'completed': True,
                     'qualification_count': 1, 'initial': result, 'final': result}
        failed = {'provider': 'gemini', 'case_id': 'two', 'policy': 'quote_targets', 'completed': False,
                  'qualification_count': 1, 'failure': 'provider_rate_limited'}
        metrics = summarize([completed, failed], 3)['gemini']
        self.assertEqual(metrics['completion_rate'], 0.5)
        self.assertEqual(metrics['availability_excluding_transport_failures'], 0.5)
        self.assertEqual(metrics['final_strict_success_completed'], 1)
        self.assertEqual(metrics['final_strict_success_all_attempted'], 0.5)
        self.assertEqual(metrics['unexecuted_questions'], 1)
        self.assertIsNone(metrics['independent_legal_accuracy_score'])

    def test_verifier_failure_preserves_initial_denominator_and_structured_failure_category(self):
        item = entry()
        initial = score_output(item, '{"span_ids":["S001","S002"]}', build_plan(item['case']['evidence']))
        rows = [{'provider': 'gemini', 'case_id': 'one', 'policy': 'quote_targets', 'completed': False,
                 'qualification_count': 1, 'initial': initial, 'failure': 'provider_invalid_selection'}]
        metrics = summarize(rows, 1)['gemini']
        self.assertEqual(metrics['initial_outputs_observed'], 1)
        self.assertEqual(metrics['initial_strict_success_observed'], 1)
        self.assertEqual(metrics['completion_rate'], 0)
        self.assertEqual(metrics['provider_unavailable_questions'], 0)
        self.assertEqual(metrics['model_or_structured_output_failure_questions'], 1)
        self.assertEqual(failure_category('provider_incomplete_response'), 'model_or_structured_output_failure')

    def test_empty_abstention_is_not_counted_as_citation_integrity_evidence(self):
        item = entry()
        item['case']['expected'] = {'policy': 'empty_claims', 'target_groups': []}
        result = score_output(item, '{"span_ids":[]}', build_plan(item['case']['evidence']))
        rows = [{'provider': 'gemini', 'case_id': 'one', 'policy': 'empty_claims', 'completed': True,
                 'qualification_count': 0, 'initial': result, 'final': result}]
        metrics = summarize(rows, 1)['gemini']
        self.assertEqual(metrics['empty_final_outputs'], 1)
        self.assertEqual(metrics['nonempty_final_outputs'], 0)
        self.assertEqual(metrics['copied_claim_count'], 0)
        self.assertIsNone(metrics['quotation_and_citation_integrity'])

    def test_model_abstention_independent_of_backend(self):
        item = deepcopy(entry())
        item['case']['expected'] = {'policy': 'empty_claims', 'target_groups': []}
        plan = build_plan(item['case']['evidence'])
        result = score_output(item, json.dumps({'span_ids': []}), plan)
        self.assertTrue(result['score']['correct_abstention'])
        self.assertFalse(result['backend_accepted'])


if __name__ == '__main__':
    unittest.main()
