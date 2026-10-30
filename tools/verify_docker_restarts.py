"""Verify ordinary Desktop restart recovery without touching runtime sockets."""
import argparse
from datetime import datetime, timezone
import json
import subprocess
import time

from ipindia_corpus.sync_runner import ROOT


def run(arguments, timeout=135):
    result = subprocess.run(arguments, cwd=ROOT, capture_output=True, text=True,
        timeout=timeout, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if result.returncode:
        raise RuntimeError(f'{arguments[:3]} failed: {(result.stdout + result.stderr)[-1200:]}')
    return result.stdout.strip()


def verify(cycles=3):
    names = set(filter(None, run(['docker', 'ps', '--format', '{{.Names}}']).splitlines()))
    if names - {'sih-db-1', 'sih-api-1', 'sih-candidate-worker-1'}:
        raise RuntimeError('Unrelated running Docker workloads require review before restart tests.')
    report = {'at': datetime.now(timezone.utc).isoformat(), 'method': 'normal Desktop stop/start',
              'manual_socket_cleanup': False, 'factory_reset': False, 'cycles': [], 'passed': False,
              'windows_reboot_or_sleep_resume_tested': False}
    target = ROOT / 'reports/docker_restart_verification.json'
    expected_volume = None
    try:
        for number in range(1, cycles + 1):
            started = time.monotonic()
            run(['docker', 'desktop', 'stop', '--timeout', '60'], timeout=75)
            stopped = time.monotonic()
            run(['docker', 'desktop', 'start', '--timeout', '120'])
            ready = time.monotonic()
            engine = run(['docker', 'info', '--format', '{{.ServerVersion}}'], timeout=20)
            volume = json.loads(run(['docker', 'volume', 'inspect', 'sih_pgdata']))[0]
            identity = {key: volume[key] for key in ('Name', 'Driver', 'CreatedAt')}
            if expected_volume is not None and identity != expected_volume:
                raise ValueError('volume_identity_changed')
            expected_volume = identity
            observation = {'cycle': number, 'stop_seconds': round(stopped-started, 2),
                'start_seconds': round(ready-stopped, 2), 'engine': engine,
                'volume': identity, 'passed': True}
            report['cycles'].append(observation)
            target.write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps(observation), flush=True)
        report['passed'] = len(report['cycles']) == cycles
        return report
    finally:
        target.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-restarts', action='store_true')
    args = parser.parse_args()
    if not args.run_restarts:
        parser.error('--run-restarts required')
    print(json.dumps(verify(), indent=2))
