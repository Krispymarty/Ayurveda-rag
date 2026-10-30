"""Probe only the loopback development service; this is not a public penetration test."""
import json
from datetime import datetime, timezone

import requests

from ipindia_corpus.sync_runner import ROOT


def main():
    client = requests.Session()
    client.trust_env = False
    base = 'http://127.0.0.1:8000'
    checks = {}
    page = client.get(base + '/', timeout=30)
    checks['local_ui'] = page.status_code == 200 and 'The evidence desk' in page.text
    checks['strict_csp'] = "default-src 'self'" in page.headers.get('Content-Security-Policy', '')
    checks['anti_framing'] = page.headers.get('X-Frame-Options') == 'DENY'
    checks['nosniff'] = page.headers.get('X-Content-Type-Options') == 'nosniff'
    health = client.get(base + '/health', timeout=45)
    checks['active_index_healthy'] = health.status_code == 200 and health.json().get('status') == 'ok'
    capabilities = client.get(base + '/capabilities', timeout=15).json()
    answer = client.post(base + '/answer', json={'question': 'What does the Patents Act say?'}, timeout=15)
    checks['uncertified_answers_blocked'] = (not capabilities['answer_generation_enabled']
        and answer.status_code == 503 and answer.json().get('reason') == 'three_source_not_certified')
    checks['untrusted_host_rejected'] = client.get(base + '/', headers={'Host': 'attacker.invalid'}, timeout=10).status_code == 400
    checks['bounded_search'] = client.get(base + '/search', params={'q': 'test', 'top_k': 21}, timeout=10).status_code == 422
    checks['bounded_answer'] = client.post(base + '/answer', json={'question': 'x' * 501}, timeout=10).status_code == 422
    checks['extra_answer_fields_rejected'] = client.post(base + '/answer', json={'question': 'test', 'endpoint': 'https://attacker.invalid'}, timeout=10).status_code == 422
    checks['asset_traversal_rejected'] = client.get(base + '/assets/%2e%2e/.env', timeout=10).status_code == 404
    foreign = client.get(base + '/', headers={'Origin': 'https://attacker.invalid'}, timeout=10)
    checks['no_permissive_cors'] = 'Access-Control-Allow-Origin' not in foreign.headers
    # Invalid short queries never load a model. Distinct spoofed headers must not
    # evade the peer-IP bucket because Uvicorn runs with --no-proxy-headers.
    statuses = [client.get(base + '/search', params={'q': 'x'},
        headers={'X-Forwarded-For': f'198.51.100.{number}'}, timeout=10).status_code for number in range(1, 66)]
    checks['spoofed_forwarding_cannot_evade_rate_limit'] = 429 in statuses and set(statuses) <= {422, 429}
    report = {'pass': all(checks.values()), 'checked_at': datetime.now(timezone.utc).isoformat(),
              'scope': 'local loopback HTTP smoke checks', 'checks': checks,
              'generation_id': health.json().get('generation_id'),
              'rate_probe_requests': len(statuses), 'rate_limited_responses': statuses.count(429),
              'public_release_certified': False,
              'limitations': ['No internet deployment, distributed abuse, TLS, authentication or sustained public load was tested.',
                              'This probe temporarily consumes the local search rate bucket.']}
    (ROOT / 'reports/http_security_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0 if report['pass'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
