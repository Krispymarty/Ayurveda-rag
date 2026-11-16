"""Opt-in paired archived-evidence evaluation; never activates a configuration.

Labels are engineering annotations, not independent legal review. Source-bound
questions and evidence may be sent to Gemini only when --run is explicit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from ipindia_corpus.citation_selection import canonical
from ipindia_corpus.indexer import generation
from ipindia_corpus.inference import make_provider, provider_identity, PROVIDER_FAILURES
from ipindia_corpus.llm_scoring import score_case
from ipindia_corpus.reliable_answers import configuration, selection_fields, verifier_fields
from ipindia_corpus.span_selection import build_plan, resolve_selection
from tools.freeze_llm_benchmark import _evidence
from tools.benchmark_span_selection import coverage_audit

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def failure_category(reason):
    if reason in {'provider_invalid_selection', 'provider_invalid_response',
                  'provider_incomplete_response', 'provider_response_too_large',
                  'provider_safety_blocked'}:
        return 'model_or_structured_output_failure'
    if reason in {'timeout', 'provider_rate_limited', 'provider_auth_failed',
                  'provider_model_unavailable', 'provider_unavailable'}:
        return 'provider_unavailable'
    if reason in {'provider_model_changed', 'model_digest_changed'}:
        return 'model_binding_failure'
    if reason in {'evidence_context_limit', 'gemini_key_missing', 'gemini_configuration_invalid'}:
        return 'context_or_configuration_failure'
    return 'evaluation_or_unclassified_failure'


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def qualification_coverage(qualifications, evidence, plan, selected_ids):
    """Require every Unicode codepoint of each annotated qualification interval."""
    by_chunk = {hit['chunk_id']: hit for hit in evidence}
    results = []
    for item in qualifications:
        chunk = by_chunk[item['chunk_id']]
        start, end = item['start'], item['end']
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(chunk['text'])
                or chunk['text'][start:end] != item['quote']):
            raise ValueError('qualification_annotation_mismatch')
        intervals = sorted((plan['spans'][key]['start'], plan['spans'][key]['end'])
                           for key in selected_ids
                           if plan['spans'][key]['chunk_id'] == item['chunk_id'])
        reached = start
        for lower, upper in intervals:
            if lower > reached:
                break
            reached = max(reached, upper)
        results.append({**item, 'retained': reached >= end})
    return results


def score_output(entry, raw, plan, *, selected_subset=None):
    case = entry['case']
    try:
        resolved = resolve_selection(raw, case['evidence'], plan)
        identifiers = resolved['selection']['span_ids']
        if selected_subset is not None and not set(identifiers).issubset(selected_subset):
            raise ValueError('verifier_added_unselected_id')
    except ValueError as exc:
        return {'valid_selection': False, 'error': str(exc),
                'score': score_case(case, None),
                'qualification_results': [{**item, 'retained': False} for item in entry.get('qualifications', [])],
                'all_qualifications_retained': False, 'backend_accepted': False}
    qualifications = qualification_coverage(entry.get('qualifications', []), case['evidence'], plan, identifiers)
    return {'valid_selection': True, **resolved,
            'score': score_case(case, canonical(resolved['expanded_selection']).decode('utf-8')),
            'qualification_results': qualifications,
            'all_qualifications_retained': all(item['retained'] for item in qualifications),
            'backend_accepted': resolved['backend_result']['status'] == 'answered'}


def prepare(bundle, *, root=ROOT):
    """Bind every real hit to a fully verified archived generation before calls."""
    generation_id = bundle.get('generation_id', '20261014T204940358650Z')
    actual, documents, chunks = generation(root, generation_id=generation_id)
    by_chunk = {chunk['chunk_id']: chunk for chunk in chunks}
    by_doc = {document['document_id']: document for document in documents}
    entries, ids = [], set()
    for item in bundle['entries']:
        case = item['case']
        if case['id'] in ids:
            raise ValueError('duplicate_case_id')
        ids.add(case['id'])
        for hit in case['evidence']:
            if hit.get('source') == 'synthetic_fixture':
                if hit.get('evidence_role') != 'synthetic_adversarial':
                    raise ValueError('unlabelled_synthetic_evidence')
                continue
            chunk = by_chunk.get(hit['chunk_id'])
            if chunk is None or hit != _evidence(chunk, by_doc[chunk['document_id']]):
                raise ValueError('case_evidence_differs_from_generation')
        plan = build_plan(case['evidence'])
        fields = selection_fields(case['question'], plan)
        if 'plan' in item and item['plan'] != plan:
            raise ValueError('supplied_plan_changed')
        if 'fields' in item and item['fields'] != fields:
            raise ValueError('supplied_fields_changed')
        qualification_coverage(item.get('qualifications', []), case['evidence'], plan, [])
        entries.append({**item, 'plan': plan, 'fields': fields,
                        'input_sha256': digest(fields), 'plan_sha256': digest(plan),
                        'candidate_coverage_audit': coverage_audit(case, plan),
                        'packing_applied': False,
                        'evidence_utf8_bytes': sum(len(hit['text'].encode('utf-8')) for hit in case['evidence'])})
    return {'generation_id': actual, 'entries': entries,
            'bundle_sha256': digest(bundle), 'independent_legal_accuracy_score': None}


def summarize(rows, planned):
    result = {}
    for provider in sorted({row['provider'] for row in rows}):
        cohort = [row for row in rows if row['provider'] == provider]
        completed = [row for row in cohort if row['completed']]
        initial_observed = [row['initial'] for row in cohort if 'initial' in row]
        unavailable = [row for row in cohort if failure_category(row.get('failure')) == 'provider_unavailable']
        structured_failures = [row for row in cohort if failure_category(row.get('failure')) == 'model_or_structured_output_failure']
        answerable = [row for row in completed if row['policy'] == 'quote_targets']
        abstention = [row for row in completed if row['policy'] == 'empty_claims']
        qualification_cases = [row for row in answerable if row['qualification_count']]
        final = [row['final'] for row in completed]
        nonempty_final = [row for row in final if row.get('expanded_selection', {}).get('claims')]
        claim_count = sum(len(row['expanded_selection']['claims']) for row in nonempty_final)
        def rate(values):
            return sum(values) / len(values) if values else None
        result[provider] = {
            'planned_questions': planned, 'attempted_questions': len(cohort),
            'completed_questions': len(completed), 'unexecuted_questions': planned - len(cohort),
            'completion_rate': len(completed) / len(cohort) if cohort else None,
            'availability': len(completed) / len(cohort) if cohort else None,
            'availability_legacy_definition': 'completed_pipeline_questions_divided_by_attempts; includes structured-output failures',
            'provider_unavailable_questions': len(unavailable),
            'model_or_structured_output_failure_questions': len(structured_failures),
            'availability_excluding_transport_failures': (len(cohort) - len(unavailable)) / len(cohort) if cohort else None,
            'initial_outputs_observed': len(initial_observed),
            'initial_strict_success_observed': rate([row['score']['passed'] for row in initial_observed]),
            'initial_strict_success_completed': rate([row['initial']['score']['passed'] for row in completed]),
            'initial_strict_success_completed_count': len(completed),
            'final_strict_success_completed': rate([row['score']['passed'] for row in final]),
            'final_strict_success_all_attempted': sum(row['score']['passed'] for row in final) / len(cohort),
            'answerable_count_completed': len(answerable),
            'answerable_complete_target_coverage': rate([row['final']['score']['target_coverage'] == 1 for row in answerable]),
            'answerable_mean_target_group_recall': rate([row['final']['score']['target_coverage'] for row in answerable]),
            'answerable_strict_correctness': rate([row['final']['score']['passed'] for row in answerable]),
            'nonempty_final_outputs': len(nonempty_final),
            'empty_final_outputs': sum(row.get('valid_selection') and not row['expanded_selection']['claims'] for row in final),
            'invalid_final_outputs': sum(not row.get('valid_selection') for row in final),
            'copied_claim_count': claim_count,
            'quotation_and_citation_integrity': rate([row['score']['source_valid'] is True and row['valid_selection'] for row in nonempty_final]),
            'abstention_count_completed': len(abstention),
            'correct_abstention': rate([row['final']['score']['correct_abstention'] for row in abstention]),
            'qualification_cases_completed': len(qualification_cases),
            'all_qualifications_retained': rate([row['final']['all_qualifications_retained'] for row in qualification_cases]),
            'qualification_units_retained': rate([item['retained'] for row in qualification_cases for item in row['final']['qualification_results']]),
            'independent_legal_accuracy_score': None,
            'legal_review_status': 'not_independently_human_reviewed',
            'failures': [{'case_id': row['case_id'], 'reason': row.get('failure') or row['final']['score']['failures']}
                         for row in cohort if not row['completed'] or not row['final']['score']['passed']],
        }
    return result


def run(bundle_path, output, *, providers=('ollama', 'gemini'), interval=15.0, root=ROOT):
    if interval < 15:
        raise ValueError('gemini_interval_must_be_at_least_15_seconds')
    if len(set(providers)) != len(providers):
        raise ValueError('duplicate_provider')
    output = Path(output).resolve()
    if not output.is_relative_to((root / 'data').resolve()) or output.exists():
        raise ValueError('output_must_be_new_under_data')
    bundle = json.loads(Path(bundle_path).read_text(encoding='utf-8'))
    prepared = prepare(bundle, root=root)
    output.mkdir(parents=True)
    write_new(output / 'prepared.json', prepared)
    code_files = ('tools/benchmark_evidence_quality.py', 'ipindia_corpus/reliable_answers.py',
                  'ipindia_corpus/span_selection.py', 'ipindia_corpus/llm_scoring.py',
                  'ipindia_corpus/inference.py')
    code_hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in code_files}
    write_new(output / 'code_hashes.json', code_hashes)
    clients = {name: make_provider(name) for name in providers}
    write_new(output / 'configuration.json', {name: configuration(client) for name, client in clients.items()})
    rows, stopped, last_gemini = [], set(), None

    def call(client, name, fields, path):
        nonlocal last_gemini
        if name == 'gemini' and last_gemini is not None:
            remaining = interval - (time.monotonic() - last_gemini)
            if remaining > 0:
                time.sleep(remaining)
        write_new(path.with_suffix('.started.json'), {'input_sha256': digest(fields), 'started_unix': time.time()})
        if name == 'gemini':
            last_gemini = time.monotonic()
        started = time.monotonic()
        raw, telemetry = client.chat(fields)
        write_new(path.with_suffix('.raw.json'), {'raw': raw, 'telemetry': telemetry,
                                                'elapsed_ms': (time.monotonic() - started) * 1000,
                                                'model': provider_identity(client)})
        return raw

    for ordinal, entry in enumerate(prepared['entries'], 1):
        for name, client in clients.items():
            if name in stopped:
                continue
            case, plan = entry['case'], entry['plan']
            folder = output / f'{ordinal:03d}-{name}'
            folder.mkdir()
            started = time.monotonic()
            row = {'provider': name, 'case_id': case['id'], 'policy': case['expected']['policy'],
                   'qualification_count': len(entry.get('qualifications', [])),
                   'input_sha256': entry['input_sha256'], 'completed': False, 'stage': 'selection'}
            try:
                raw = call(client, name, entry['fields'], folder / 'selection')
                initial = score_output(entry, raw, plan)
                row['initial'] = initial
                if not initial['valid_selection']:
                    row['final'] = initial
                else:
                    selected = initial['selection']['span_ids']
                    if selected:
                        row['stage'] = 'verification'
                        raw = call(client, name, verifier_fields(case['question'], plan, selected), folder / 'verifier')
                        row['final'] = score_output(entry, raw, plan, selected_subset=selected)
                    else:
                        row['final'] = initial
                row['completed'] = True
                row['outcome_category'] = 'quality_pass' if row['final']['score']['passed'] else 'quality_failure'
            except Exception as exc:
                reason = str(exc)
                row['failure'] = reason if reason in PROVIDER_FAILURES or reason in {
                    'model_digest_changed', 'timeout', 'provider_incomplete_response'} else 'provider_or_evaluation_failure'
                row['exception_type'] = type(exc).__name__
                row['outcome_category'] = failure_category(row['failure'])
                if name == 'gemini' and row['failure'] == 'provider_rate_limited':
                    stopped.add(name)
            row['elapsed_ms'] = (time.monotonic() - started) * 1000
            row['model'] = provider_identity(client)
            write_new(folder / 'result.json', row)
            rows.append(row)
            print(json.dumps({'provider': name, 'case_id': case['id'], 'completed': row['completed'],
                              'passed': row.get('final', {}).get('score', {}).get('passed'),
                              'failure': row.get('failure')}, ensure_ascii=True), flush=True)
    # Revalidate archived originals and bindings after all model calls.
    prepare(bundle, root=root)
    if any(hashlib.sha256((root / name).read_bytes()).hexdigest() != value for name, value in code_hashes.items()):
        raise ValueError('evaluation_code_changed_during_run')
    write_new(output / 'summary.json', {'metrics': summarize(rows, len(prepared['entries'])),
              'rows': rows, 'stopped_providers': sorted(stopped), 'activation_performed': False,
              'source_revalidated_after_calls': True, 'bundle_sha256': prepared['bundle_sha256']})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--providers', nargs='+', choices=('ollama', 'gemini'), default=['ollama', 'gemini'])
    parser.add_argument('--gemini-interval', type=float, default=15)
    parser.add_argument('--run', action='store_true', help='Authorize actual model calls on these archived research inputs')
    args = parser.parse_args()
    if not args.run:
        parser.error('--run is required for model calls')
    run(args.bundle, args.out, providers=args.providers, interval=args.gemini_interval)


if __name__ == '__main__':
    main()
