"""Bounded Gemini evidence selection; credentials and source data stay separate.

The application supplies the same messages/schema to either provider. Gemini has
no locally verifiable weights digest: its returned modelVersion is recorded and
latched for the life of this instance instead. Passport owns request-level
fallback; individual provider calls never silently switch models.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import time

import requests

from .grounded_answers import _unique_object
from .llm_experiment import ExperimentFailure

ROOT = Path(__file__).resolve().parents[1]
API = 'https://generativelanguage.googleapis.com'
DEFAULT_MODEL = 'gemini-3.5-flash-lite'
MODELS = {'gemini-3.5-flash-lite', 'gemini-3.6-flash', 'gemini-3.8-flash',
          'gemini-2.5-flash', 'gemini-2.5-flash-lite'}
MAX_RESPONSE = 256_000
CONTEXT_TOKENS, INPUT_TOKENS, OUTPUT_TOKENS = 4096, 3800, 200
PROVIDER_FAILURES = frozenset({'timeout', 'model_digest_changed', 'provider_incomplete_response',
    'evidence_context_limit', 'provider_model_changed', 'provider_rate_limited', 'provider_auth_failed',
    'provider_model_unavailable', 'provider_safety_blocked', 'provider_invalid_selection',
    'provider_invalid_response', 'provider_unavailable', 'provider_response_too_large',
    'provider_request_rejected', 'provider_redirect_rejected', 'provider_tls_failed',
    'gemini_key_missing', 'gemini_configuration_invalid', 'groq_key_missing', 'groq_configuration_invalid'})


def _file_settings(root=ROOT):
    """Read only documented settings; never expand values or mutate the env."""
    path = Path(root) / '.env'
    if not path.is_file():
        return {}
    keys = {}
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        match = re.fullmatch(r'\s*(?:export\s+)?(GEMINI_API_KEY|GOOGLE_API_KEY|GROQ_API_KEY|AYURIP_LLM_PROVIDER|AYURIP_GEMINI_MODEL|AYURIP_GROQ_MODEL|AYURIP_FALLBACK_PROVIDER|AYURIP_QWEN_FALLBACK)\s*=\s*(.*?)\s*', line)
        if match:
            name, value = match.groups()
            if value.startswith(('\"', "'")):
                end = value.find(value[0], 1)
                if end < 1 or (value[end + 1:].strip() and not value[end + 1:].strip().startswith('#')):
                    raise ExperimentFailure('gemini_configuration_invalid')
                value = value[1:end]
            else:
                value = re.split(r'\s+#', value, maxsplit=1)[0].strip()
            if name in keys and keys[name] != value:
                raise ExperimentFailure('gemini_configuration_invalid')
            keys[name] = value
    return keys


def read_api_key(root=ROOT):
    """Keys never enter metadata, request bodies, URLs or environment mutation."""
    for name in ('GEMINI_API_KEY', 'GOOGLE_API_KEY'):
        if os.environ.get(name, '').strip():
            return os.environ[name].strip()
    keys = _file_settings(root)
    return keys.get('GEMINI_API_KEY') or keys.get('GOOGLE_API_KEY', '')


def read_setting(name, default):
    if name in os.environ:
        return os.environ[name]
    return _file_settings().get(name, default)


def configured_identity():
    kind = read_setting('AYURIP_LLM_PROVIDER', 'groq')
    if kind == 'groq':
        from .groq_candidate import DEFAULT_MODEL as GROQ_MODEL, MODELS as GROQ_MODELS
        model = read_setting('AYURIP_GROQ_MODEL', GROQ_MODEL)
        if model not in GROQ_MODELS:
            raise ExperimentFailure('groq_configuration_invalid')
        return {'provider': 'groq', 'model': model, 'model_version': None,
                'model_digest': None, 'external_processing': True}
    if kind == 'ollama':
        from .reliable_answers import MODEL, DIGEST
        return {'provider': 'ollama', 'model': MODEL, 'model_digest': DIGEST,
                'model_version': None, 'external_processing': False}
    model = read_setting('AYURIP_GEMINI_MODEL', DEFAULT_MODEL)
    if kind != 'gemini' or model not in MODELS:
        raise ExperimentFailure('gemini_configuration_invalid')
    return {'provider': 'gemini', 'model': model, 'model_version': None,
            'model_digest': None, 'external_processing': True}


def provider_identity(provider):
    metadata = getattr(provider, 'metadata', None)
    if isinstance(metadata, dict):
        return {key: metadata.get(key) for key in
                ('provider', 'model', 'model_version', 'model_digest', 'external_processing')}
    # Legacy injected fixtures and historical explicitly local experiments.
    from .reliable_answers import MODEL, DIGEST
    return {'provider': 'ollama', 'model': MODEL, 'model_digest': DIGEST,
            'model_version': None, 'external_processing': False}


def make_provider(kind=None):
    kind = kind or read_setting('AYURIP_LLM_PROVIDER', 'groq')
    if kind == 'groq':
        from .groq_candidate import GroqCandidate
        return GroqCandidate(usage_checked_context=True)
    if kind == 'ollama':
        from .reliable_answers import SpanProvider
        return SpanProvider()
    if kind != 'gemini':
        raise ExperimentFailure('gemini_configuration_invalid')
    return GeminiProvider()


def provider_binding_valid(provider):
    seen = set()
    for _ in range(5):
        if id(provider) in seen or getattr(provider, 'failed_binding', False):
            return False
        seen.add(id(provider))
        child = getattr(provider, 'provider', None)
        if child is None:
            return True
        provider = child
    return False


def _validate(value, schema):
    """Validate the small schema subset emitted by the existing selectors."""
    kind = schema.get('type')
    valid = {'object': isinstance(value, dict), 'array': isinstance(value, list),
             'string': isinstance(value, str), 'boolean': type(value) is bool}.get(kind, False)
    if not valid or ('enum' in schema and value not in schema['enum']):
        raise ExperimentFailure('provider_invalid_selection')
    if kind == 'object':
        props = schema.get('properties', {})
        if (not set(schema.get('required', ())).issubset(value)
                or (schema.get('additionalProperties') is False and not set(value).issubset(props))):
            raise ExperimentFailure('provider_invalid_selection')
        for key, item in value.items():
            if key not in props:
                raise ExperimentFailure('provider_invalid_selection')
            _validate(item, props[key])
    elif kind == 'array':
        if not schema.get('minItems', 0) <= len(value) <= schema.get('maxItems', 100):
            raise ExperimentFailure('provider_invalid_selection')
        for item in value:
            _validate(item, schema['items'])


class GeminiProvider:
    def __init__(self, *, api_key=None, model=None, session=None, timeout=45, root=ROOT):
        self.model = model or read_setting('AYURIP_GEMINI_MODEL', DEFAULT_MODEL)
        if self.model not in MODELS or type(timeout) is not int or not 1 <= timeout <= 45:
            raise ExperimentFailure('gemini_configuration_invalid')
        self._key = read_api_key(root) if api_key is None else api_key
        if not self._key:
            raise ExperimentFailure('gemini_key_missing')
        # API keys are opaque; validate header safety, not an assumed key alphabet.
        if (not isinstance(self._key, str) or not 16 <= len(self._key) <= 512
                or any(not 33 <= ord(char) <= 126 for char in self._key)):
            raise ExperimentFailure('gemini_configuration_invalid')
        self.timeout, self.model_version = timeout, None
        self.session = session or requests.Session()
        self.session.trust_env = False
        self.failed_binding = False

    @property
    def metadata(self):
        return {'provider': 'gemini', 'model': self.model, 'model_version': self.model_version,
                'model_digest': None, 'external_processing': True}

    def _request(self, operation, payload, deadline):
        if operation not in {'countTokens', 'generateContent'}:
            raise ExperimentFailure('gemini_configuration_invalid')
        started, response = time.monotonic(), None
        remaining = deadline - started
        if remaining <= 0:
            raise ExperimentFailure('timeout')
        try:
            response = self.session.request('POST', f'{API}/v1beta/models/{self.model}:{operation}',
                headers={'x-goog-api-key': self._key, 'Accept': 'application/json'}, json=payload,
                stream=True, allow_redirects=False, verify=True, timeout=(min(5, remaining), remaining))
            status = response.status_code
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
                if time.monotonic() > deadline:
                    raise ExperimentFailure('timeout')
                if len(raw) > MAX_RESPONSE:
                    raise ExperimentFailure('provider_response_too_large')
            value = json.loads(raw, object_pairs_hook=_unique_object)
            if not isinstance(value, dict) or value.get('error'):
                raise ExperimentFailure('provider_invalid_response')
            return value, (time.monotonic() - started) * 1000
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

    def chat(self, fields):
        return self.chat_with_deadline(fields, None)

    def chat_with_deadline(self, fields, deadline):
        try:
            return self._chat(fields, deadline)
        except (ValueError, TypeError, KeyError, AttributeError, RecursionError):
            raise ExperimentFailure('provider_invalid_response') from None

    def _chat(self, fields, request_deadline=None):
        started = time.monotonic()
        if self.failed_binding:
            raise ExperimentFailure('provider_model_changed')
        if (not isinstance(fields, dict) or set(fields) != {'messages', 'format'}
                or not isinstance(fields['messages'], list) or len(fields['messages']) != 2
                or not isinstance(fields['format'], dict)):
            raise ExperimentFailure('gemini_configuration_invalid')
        system, user = fields['messages']
        if any(not isinstance(message, dict) or set(message) != {'role', 'content'}
               or not isinstance(message['content'], str) for message in (system, user)):
            raise ExperimentFailure('gemini_configuration_invalid')
        if system['role'] != 'system' or user['role'] != 'user':
            raise ExperimentFailure('gemini_configuration_invalid')
        if len(json.dumps(fields, ensure_ascii=False).encode()) > 48_000:
            raise ExperimentFailure('evidence_context_limit')
        config = {'maxOutputTokens': OUTPUT_TOKENS, 'candidateCount': 1,
                  'responseMimeType': 'application/json', 'responseJsonSchema': deepcopy(fields['format'])}
        if self.model.startswith('gemini-2.5-'):
            config.update(temperature=0, thinkingConfig={'thinkingBudget': 0})
        else:
            # Current 3.5+ models no longer accept custom sampling parameters.
            config['thinkingConfig'] = {'thinkingLevel': 'MINIMAL', 'includeThoughts': False}
        payload = {'systemInstruction': {'parts': [{'text': system['content']}]},
                   'contents': [{'role': 'user', 'parts': [{'text': user['content']}]}],
                   'generationConfig': config}
        deadline = min(started + self.timeout, request_deadline) if request_deadline is not None else started + self.timeout
        counted, count_ms = self._request('countTokens',
            {'generateContentRequest': {'model': 'models/' + self.model, **payload}}, deadline)
        count = counted.get('totalTokens')
        if type(count) is not int or count < 0:
            raise ExperimentFailure('provider_invalid_response')
        if count > INPUT_TOKENS:
            raise ExperimentFailure('evidence_context_limit')
        response, generation_ms = self._request('generateContent', payload, deadline)
        if response.get('promptFeedback', {}).get('blockReason'):
            raise ExperimentFailure('provider_safety_blocked')
        version = response.get('modelVersion')
        if (not isinstance(version, str) or len(version) > 160 or
                (version != self.model and not version.startswith(self.model + '-'))):
            self.failed_binding = True
            raise ExperimentFailure('provider_model_changed')
        if self.model_version is not None and version != self.model_version:
            self.failed_binding = True
            raise ExperimentFailure('provider_model_changed')
        candidates = response.get('candidates')
        if not isinstance(candidates, list) or len(candidates) != 1:
            raise ExperimentFailure('provider_incomplete_response')
        candidate = candidates[0]
        if not isinstance(candidate, dict) or candidate.get('finishReason') != 'STOP':
            raise ExperimentFailure('provider_incomplete_response')
        content = candidate.get('content', {})
        parts = content.get('parts') if isinstance(content, dict) else None
        if content.get('role') != 'model' or not isinstance(parts, list) or len(parts) != 1:
            raise ExperimentFailure('provider_invalid_response')
        part = parts[0]
        if (not isinstance(part, dict) or not set(part).issubset({'text', 'thoughtSignature'})
                or not isinstance(part.get('text'), str) or len(part['text'].encode()) > 6000):
            raise ExperimentFailure('provider_invalid_response')
        usage = response.get('usageMetadata', {})
        prompt, output, thoughts = (usage.get('promptTokenCount'), usage.get('candidatesTokenCount'), usage.get('thoughtsTokenCount', 0))
        if any(type(n) is not int or n < 0 for n in (prompt, output, thoughts)):
            raise ExperimentFailure('provider_invalid_response')
        if prompt > INPUT_TOKENS or output + thoughts > OUTPUT_TOKENS:
            raise ExperimentFailure('evidence_context_limit')
        try:
            decoded = json.loads(part['text'], object_pairs_hook=_unique_object)
            _validate(decoded, fields['format'])
        except (ValueError, TypeError, KeyError, RecursionError):
            raise ExperimentFailure('provider_invalid_selection') from None
        self.model_version = version
        return part['text'], {'provider_wall_ms': (time.monotonic() - started) * 1000,
            'token_count_http_ms': count_ms, 'generation_http_ms': generation_ms,
            'http_requests': 2, 'prompt_tokens': prompt, 'output_tokens': output,
            'thought_tokens': thoughts, 'preflight_tokens': count,
            'response_sha256_present': bool(hashlib.sha256(part['text'].encode()).hexdigest())}
