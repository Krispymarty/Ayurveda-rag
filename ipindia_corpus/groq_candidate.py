"""Isolated Groq evaluation adapter; never installed in the production factory.

The candidate preserves the existing messages, schema and 200-token completion
cap (including hidden reasoning). Its byte preflight is deliberately conservative
and is not presented as an exact Groq tokenizer. Returned usage is checked again.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import time

import requests

from .grounded_answers import _unique_object
from .inference import PROVIDER_FAILURES, _validate
from .llm_experiment import ExperimentFailure

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = 'https://api.groq.com/openai/v1/chat/completions'
DEFAULT_MODEL = 'openai/gpt-oss-120b'
MODELS = frozenset({'openai/gpt-oss-120b', 'openai/gpt-oss-20b'})
INPUT_LIMIT, OUTPUT_LIMIT, CONTEXT_LIMIT = 3800, 200, 4096
MAX_RESPONSE = 256_000
CANDIDATE_FAILURES = PROVIDER_FAILURES | {'groq_key_missing', 'groq_configuration_invalid'}


def file_settings(root=ROOT):
    """Read only documented candidate settings, with no expansion or mutation."""
    path = Path(root) / '.env'
    if not path.is_file():
        return {}
    found = {}
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        match = re.fullmatch(r'\s*(?:export\s+)?(GROQ_API_KEY|AYURIP_GROQ_MODEL)\s*=\s*(.*?)\s*', line)
        if not match:
            continue
        name, value = match.groups()
        if value.startswith(('"', "'")):
            end = value.find(value[0], 1)
            if end < 1 or (value[end + 1:].strip() and not value[end + 1:].strip().startswith('#')):
                raise ExperimentFailure('groq_configuration_invalid')
            value = value[1:end]
        else:
            value = re.split(r'\s+#', value, maxsplit=1)[0].strip()
        if name in found and found[name] != value:
            raise ExperimentFailure('groq_configuration_invalid')
        found[name] = value
    return found


def read_key(root=ROOT):
    if 'GROQ_API_KEY' in os.environ:
        return os.environ['GROQ_API_KEY'].strip()
    return file_settings(root).get('GROQ_API_KEY', '')


def read_model(root=ROOT):
    if 'AYURIP_GROQ_MODEL' in os.environ:
        return os.environ['AYURIP_GROQ_MODEL']
    return file_settings(root).get('AYURIP_GROQ_MODEL', DEFAULT_MODEL)


def _unique_arrays(value, schema):
    if schema.get('type') == 'array':
        if schema.get('uniqueItems') and len({json.dumps(x, sort_keys=True) for x in value}) != len(value):
            raise ExperimentFailure('provider_invalid_selection')
        for item in value:
            _unique_arrays(item, schema['items'])
    elif schema.get('type') == 'object':
        for name, item in value.items():
            _unique_arrays(item, schema['properties'][name])


def _wire_schema(schema):
    """Groq rejects uniqueItems; preserve its original constraint in the backend."""
    if isinstance(schema, dict):
        return {name: _wire_schema(value) for name, value in schema.items() if name != 'uniqueItems'}
    if isinstance(schema, list):
        return [_wire_schema(value) for value in schema]
    return schema


class GroqCandidate:
    def __init__(self, *, api_key=None, model=None, session=None, timeout=45, root=ROOT,
                 token_counter=None):
        self.model = read_model(root) if model is None else model
        if self.model not in MODELS or type(timeout) is not int or not 1 <= timeout <= 45:
            raise ExperimentFailure('groq_configuration_invalid')
        self._key = read_key(root) if api_key is None else api_key
        if not self._key:
            raise ExperimentFailure('groq_key_missing')
        if (not isinstance(self._key, str) or not 16 <= len(self._key) <= 512
                or any(not 33 <= ord(char) <= 126 for char in self._key)):
            raise ExperimentFailure('groq_configuration_invalid')
        self.session = session or requests.Session()
        self.session.trust_env = False
        self.timeout, self.model_version, self.failed_binding = timeout, None, False
        self.last_call_diagnostics = {}
        if token_counter is not None and not callable(token_counter):
            raise ExperimentFailure('groq_configuration_invalid')
        self.token_counter = token_counter

    @property
    def metadata(self):
        return {'provider': 'groq', 'model': self.model, 'model_version': self.model_version,
                'model_digest': None, 'external_processing': True}

    @property
    def profile(self):
        return {'num_ctx': CONTEXT_LIMIT, 'num_predict': OUTPUT_LIMIT, 'temperature': 0,
                'reasoning_effort': 'low', 'include_reasoning': False,
                'context_enforcement': ('trusted_counter_plus_256_then_returned_usage' if self.token_counter
                                        else 'conservative_utf8_bytes_then_returned_usage'),
                'grammar_normalization': 'drop_uniqueItems_backend_enforced',
                'strict_json_schema': True}

    def chat(self, fields):
        return self.chat_with_deadline(fields, None)

    def chat_with_deadline(self, fields, deadline):
        self.last_call_diagnostics = {}
        try:
            return self._chat(fields, deadline)
        except (ValueError, TypeError, KeyError, AttributeError, RecursionError):
            raise ExperimentFailure('provider_invalid_response') from None

    def _chat(self, fields, supplied_deadline):
        if self.failed_binding:
            raise ExperimentFailure('provider_model_changed')
        if (not isinstance(fields, dict) or set(fields) != {'messages', 'format'}
                or not isinstance(fields['messages'], list) or len(fields['messages']) != 2
                or not isinstance(fields['format'], dict)):
            raise ExperimentFailure('groq_configuration_invalid')
        if any(not isinstance(m, dict) or set(m) != {'role', 'content'}
               or not isinstance(m['content'], str) for m in fields['messages']):
            raise ExperimentFailure('groq_configuration_invalid')
        if [m['role'] for m in fields['messages']] != ['system', 'user']:
            raise ExperimentFailure('groq_configuration_invalid')
        input_bytes = len(json.dumps(fields, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
        preflight_count = input_bytes
        if self.token_counter is not None:
            try:
                preflight_count = self.token_counter(fields)
            except Exception:
                raise ExperimentFailure('groq_configuration_invalid') from None
            if type(preflight_count) is not int or preflight_count < 0:
                raise ExperimentFailure('groq_configuration_invalid')
            preflight_count += 256
        if preflight_count > INPUT_LIMIT:
            raise ExperimentFailure('evidence_context_limit')
        started = time.monotonic()
        deadline = started + self.timeout
        if supplied_deadline is not None:
            if type(supplied_deadline) not in (int, float) or not math.isfinite(supplied_deadline):
                raise ExperimentFailure('timeout')
            deadline = min(deadline, supplied_deadline)
        remaining = deadline - started
        if remaining <= 0:
            raise ExperimentFailure('timeout')
        payload = {'model': self.model, 'messages': deepcopy(fields['messages']),
                   'max_completion_tokens': OUTPUT_LIMIT, 'temperature': 0,
                   'reasoning_effort': 'low', 'include_reasoning': False,
                   'stream': False, 'n': 1,
                   'response_format': {'type': 'json_schema', 'json_schema': {
                       'name': 'ayurip_evidence_selection', 'strict': True,
                       'schema': _wire_schema(fields['format'])}}}
        response = None
        try:
            response = self.session.request('POST', ENDPOINT,
                headers={'Authorization': 'Bearer ' + self._key, 'Accept': 'application/json'},
                json=payload, stream=True, allow_redirects=False, verify=True,
                timeout=(min(5, remaining), remaining))
            status = response.status_code
            if type(status) is int and 100 <= status <= 599:
                self.last_call_diagnostics['http_status'] = status
            if status != 200:
                reason = ('provider_redirect_rejected' if 300 <= status < 400 else
                          'provider_auth_failed' if status in (401, 403) else
                          'provider_rate_limited' if status == 429 else
                          'provider_model_unavailable' if status == 404 else
                          'timeout' if status == 408 else
                          'provider_unavailable' if 500 <= status < 600 else 'provider_request_rejected')
                raise ExperimentFailure(reason)
            raw = bytearray()
            for block in response.iter_content(chunk_size=1024):
                raw.extend(block)
                if time.monotonic() >= deadline:
                    raise ExperimentFailure('timeout')
                if len(raw) > MAX_RESPONSE:
                    raise ExperimentFailure('provider_response_too_large')
            if time.monotonic() >= deadline:
                raise ExperimentFailure('timeout')
            value = json.loads(raw, object_pairs_hook=_unique_object)
        except requests.Timeout:
            raise ExperimentFailure('timeout') from None
        except requests.exceptions.SSLError:
            raise ExperimentFailure('provider_tls_failed') from None
        except requests.RequestException:
            raise ExperimentFailure('provider_unavailable') from None
        except (ValueError, TypeError, RecursionError):
            raise ExperimentFailure('provider_invalid_response') from None
        finally:
            if response is not None:
                response.close()
        if not isinstance(value, dict) or value.get('error'):
            raise ExperimentFailure('provider_invalid_response')
        self.last_call_diagnostics['returned_model_matches'] = value.get('model') == self.model
        # Failed/truncated calls retain bounded numeric evidence only. Diagnostic
        # capture never changes acceptance and never stores server text or keys.
        diagnostic_usage = value.get('usage')
        if isinstance(diagnostic_usage, dict):
            for name in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                amount = diagnostic_usage.get(name)
                if type(amount) is int and 0 <= amount <= 10_000_000:
                    self.last_call_diagnostics[name] = amount
            detail = diagnostic_usage.get('completion_tokens_details')
            if isinstance(detail, dict):
                amount = detail.get('reasoning_tokens')
                if type(amount) is int and 0 <= amount <= 10_000_000:
                    self.last_call_diagnostics['reasoning_tokens'] = amount
        if value.get('model') != self.model:
            self.failed_binding = True
            raise ExperimentFailure('provider_model_changed')
        choices = value.get('choices')
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise ExperimentFailure('provider_incomplete_response')
        choice = choices[0]
        if choice.get('finish_reason') in {'stop', 'length', 'tool_calls', 'function_call', 'content_filter'}:
            self.last_call_diagnostics['finish_reason'] = choice['finish_reason']
        if choice.get('finish_reason') != 'stop':
            raise ExperimentFailure('provider_incomplete_response')
        message = choice.get('message')
        allowed = {'role', 'content', 'tool_calls', 'function_call', 'reasoning', 'refusal'}
        if not isinstance(message, dict) or not set(message).issubset(allowed):
            raise ExperimentFailure('provider_invalid_response')
        if message.get('refusal'):
            raise ExperimentFailure('provider_safety_blocked')
        if (message.get('role') != 'assistant' or not isinstance(message.get('content'), str)
                or len(message['content'].encode('utf-8')) > 6000 or message.get('reasoning')
                or message.get('tool_calls') or message.get('function_call')):
            raise ExperimentFailure('provider_invalid_response')
        usage = value.get('usage')
        if not isinstance(usage, dict):
            raise ExperimentFailure('provider_invalid_response')
        prompt, completion, total = (usage.get(name) for name in ('prompt_tokens', 'completion_tokens', 'total_tokens'))
        if (any(type(n) is not int or n < 0 for n in (prompt, completion, total))
                or prompt + completion != total):
            raise ExperimentFailure('provider_invalid_response')
        if prompt > INPUT_LIMIT or completion > OUTPUT_LIMIT or total > CONTEXT_LIMIT:
            raise ExperimentFailure('evidence_context_limit')
        details = usage.get('completion_tokens_details', {})
        if not isinstance(details, dict):
            raise ExperimentFailure('provider_invalid_response')
        reasoning = details.get('reasoning_tokens', 0)
        if type(reasoning) is not int or not 0 <= reasoning <= completion:
            raise ExperimentFailure('provider_invalid_response')
        try:
            decoded = json.loads(message['content'], object_pairs_hook=_unique_object)
            _validate(decoded, fields['format'])
            _unique_arrays(decoded, fields['format'])
        except (ValueError, TypeError, KeyError, RecursionError):
            raise ExperimentFailure('provider_invalid_selection') from None
        self.model_version = value['model']
        return message['content'], {'provider_wall_ms': (time.monotonic() - started) * 1000,
            'http_requests': 1, 'prompt_tokens': prompt, 'output_tokens': completion,
            'reasoning_tokens': reasoning, 'input_preflight_utf8_bytes': input_bytes,
            'input_preflight_count_with_margin': preflight_count}
