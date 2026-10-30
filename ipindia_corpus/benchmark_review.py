"""Version factual benchmark corrections without editing historical frozen labels."""
import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

from .indexer import generation
from .passage_evaluation import matches
from .paths import archive_path, immutable_write
from .sync_runner import ROOT


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_bundle(root, bundle):
    root, bundle = Path(root).resolve(), Path(bundle).resolve()
    if not bundle.is_relative_to(root / 'data/benchmarks'):
        raise ValueError('benchmark_bundle_outside_archive')
    manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
    if set(manifest.get('sets', {})) != {'original', 'expanded'}:
        raise ValueError('benchmark_set_membership_invalid')
    expected_history = {'tests/retrieval_queries.json', 'tests/ipindia_retrieval_expanded.json'}
    history = manifest.get('historical_inputs', [])
    reviewed_files = manifest.get('reviewed_files', [])
    if len(history) != 2 or {item['path'] for item in history} != expected_history:
        raise ValueError('benchmark_history_membership_invalid')
    expected_reviewed = [manifest['sets'][name] for name in ('original', 'expanded')]
    if reviewed_files != expected_reviewed:
        raise ValueError('benchmark_review_membership_invalid')
    for name, count in (('original', 10), ('expanded', 30)):
        item = manifest['sets'][name]
        path = bundle / (name + '.json')
        if item['path'] != path.relative_to(root).as_posix():
            raise ValueError('benchmark_review_path_invalid')
        if path.with_suffix('.sha256').read_text().strip() != item['sha256']:
            raise ValueError('benchmark_review_receipt_mismatch')
        dataset = json.loads(path.read_text(encoding='utf-8'))
        if (not dataset.get('frozen') or len(dataset.get('queries', [])) != count
                or dataset.get('frozen_generation_id') != manifest['generation_id']):
            raise ValueError('benchmark_review_dataset_invalid')
    for kind in ('historical_inputs', 'reviewed_files'):
        for item in manifest[kind]:
            path = (root / item['path']).resolve()
            if not path.is_relative_to(root) or digest(path) != item['sha256']:
                raise ValueError('benchmark_history_or_review_hash_mismatch')
    return manifest


def create_reviewed(root=ROOT, *, generation_id, version='ipindia-reviewed-20261015-v1', reviewer='Codex archived-source factual review'):
    root = Path(root).resolve()
    if not reviewer.strip() or not version or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in version):
        raise ValueError('valid_version_and_reviewer_required')
    bundle = root / 'data/benchmarks' / version
    if (bundle / 'manifest.json').exists():
        return verify_bundle(root, bundle)
    historical = root / 'tests/retrieval_queries.json'
    expanded_path = root / 'tests/ipindia_retrieval_expanded.json'
    expanded = json.loads(expanded_path.read_text(encoding='utf-8'))
    if expanded['original_label_sha256'] != digest(historical) or expanded_path.with_suffix('.sha256').read_text().strip() != digest(expanded_path):
        raise ValueError('historical_labels_changed')
    original_dataset = json.loads(historical.read_text(encoding='utf-8'))
    original_queries = [q for q in original_dataset['queries'] if q['source'] == 'ipindia' and q.get('reviewed')]
    if expanded['queries'][:10] != original_queries or len(original_queries) != 10 or len(expanded['queries']) != 30:
        raise ValueError('historical_query_alignment_failed')
    _, documents, chunks = generation(root, generation_id=generation_id)
    corrections = []
    outputs = {}
    now = datetime.now(timezone(timedelta(hours=5, minutes=30))).isoformat()
    for name, queries, scope in (('original', original_queries, 'ipindia_reviewed_subset'),
                                 ('expanded', expanded['queries'], 'ipindia_only')):
        reviewed = copy.deepcopy(queries)
        correction_id = 'ipindia-10' if name == 'original' else 'ipindia-28'
        ids = {'ipindia-10', correction_id}
        for item in reviewed:
            if item['id'] not in ids:
                continue
            for target in item['targets']:
                if target.get('page') != 2 or target.get('contains') != 'prescribed fee':
                    raise ValueError('unexpected_historical_annotation')
                doc = next((d for d in documents if d['source_url'] in target['urls'] and d['sha256'] in target['raw_sha256']), None)
                if not doc:
                    raise ValueError('review_source_missing')
                import fitz
                with fitz.open(archive_path(root, doc['raw_path'])) as pdf:
                    before = ' '.join(pdf[1].get_text(sort=True).split())
                    after = ' '.join(pdf[2].get_text(sort=True).split())
                if 'prescribed fee' in before.casefold() or 'prescribed fee' not in after.casefold():
                    raise ValueError('physical_page_correction_not_verified')
                start = after.casefold().index('prescribed fee')
                quote = after[max(0, start - 230):start + 50]
                corrections.append({'set': name, 'query_id': item['id'], 'field': 'page', 'before': 2, 'after': 3,
                                    'source_url': doc['source_url'], 'raw_sha256': doc['sha256'],
                                    'evidence_quote': quote, 'reviewer': reviewer, 'review_kind': 'factual_physical_page',
                                    'legal_status_conclusion': 'unknown'})
                target['page'] = 3
                if not any(matches(chunk, target) for chunk in chunks):
                    raise ValueError('corrected_passage_not_in_candidate')
        for item in reviewed:
            for target in item['targets']:
                if not any(matches(chunk, target) for chunk in chunks):
                    raise ValueError('reviewed_target_unresolved:' + item['id'])
        dataset = {'schema_version': 3, 'version': version, 'scope': scope, 'frozen': True,
                   'reviewed_at': now, 'reviewer': reviewer, 'frozen_generation_id': generation_id,
                   'review_kind': 'archived_source_factual_anchors_not_legal_status', 'queries': reviewed}
        path = bundle / (name + '.json')
        immutable_write(path, json.dumps(dataset, indent=2, ensure_ascii=False).encode())
        immutable_write(path.with_suffix('.sha256'), digest(path).encode('ascii'))
        outputs[name] = {'path': path.relative_to(root).as_posix(), 'sha256': digest(path)}
    manifest = {'schema_version': 1, 'version': version, 'generation_id': generation_id,
                'created_at': now, 'reviewer': reviewer, 'historical_inputs': [
                    {'path': p.relative_to(root).as_posix(), 'sha256': digest(p)} for p in (historical, expanded_path)],
                'reviewed_files': list(outputs.values()), 'sets': outputs, 'corrections': corrections,
                'historical_labels_unchanged': True, 'legal_review': False}
    immutable_write(bundle / 'manifest.json', json.dumps(manifest, indent=2, ensure_ascii=False).encode())
    return verify_bundle(root, bundle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation-id', required=True)
    parser.add_argument('--version', default='ipindia-reviewed-20261015-v1')
    parser.add_argument('--reviewer', default='Codex archived-source factual review')
    args = parser.parse_args()
    result = create_reviewed(generation_id=args.generation_id, version=args.version, reviewer=args.reviewer)
    print(json.dumps({'version': result['version'], 'corrections': len(result['corrections']), 'historical_labels_unchanged': True}, indent=2))
