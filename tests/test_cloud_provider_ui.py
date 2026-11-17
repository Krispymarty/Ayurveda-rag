"""Provider-consent presentation checks; synthetic browser fixtures only."""
from pathlib import Path
import os
import unittest

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / 'ipindia_corpus/passport/ui'


class CloudDisclosureTests(unittest.TestCase):
    def test_initial_html_discloses_both_possible_cloud_processors(self):
        html = (UI / 'index.html').read_text(encoding='utf-8')
        notice = html.split('id="external-processing-notice"', 1)[1].split('</div>', 1)[0]
        self.assertIn('Groq', notice)
        self.assertIn('Google Gemini', notice)
        self.assertIn('fallback processing', html)
        self.assertIn('acknowledged_external_processing', html)

    def test_provider_text_uses_safe_dynamic_identity_and_no_local_fallback_claim(self):
        script = (UI / 'passport.js').read_text(encoding='utf-8')
        self.assertIn('session.fallback.provider', script)
        self.assertIn('providerName(data.routing.primary)', script)
        self.assertIn('providerName(model)', script)
        self.assertNotIn('This result used the local Qwen fallback.', script)
        self.assertNotIn('.innerHTML', script)


@unittest.skipUnless(os.environ.get('AYURIP_RUN_BROWSER_TESTS') == '1', 'Opt-in installed-browser test')
class CloudDisclosureBrowserTests(unittest.TestCase):
    def test_cloud_fallback_consent_actual_identity_and_legacy_local_mode(self):
        from playwright.sync_api import sync_playwright
        groq = {'provider': 'groq', 'model': 'qwen/qwen3.8-27b', 'external_processing': True}
        gemini = {'provider': 'gemini', 'model': 'gemini-3.5-flash-lite', 'external_processing': True}
        local = {'provider': 'ollama', 'model': 'qwen3.5:4b', 'external_processing': False}
        phase = {'primary': groq, 'fallback': gemini}
        sent, errors = [], []
        def serve(route):
            path = route.request.url.split('127.0.0.1:8003', 1)[1]
            if path == '/api/session':
                identity = phase['primary']
                return route.fulfill(json={'csrf_token': 'synthetic-token', 'ready': True,
                    'provider': identity, 'model': identity['model'], 'external_processing': identity['external_processing'],
                    'fallback': {'configured': phase['fallback'] is not None, 'provider': phase['fallback']}})
            if path == '/api/passport':
                sent.append(route.request.post_data_json)
                final = phase['fallback'] or phase['primary']
                return route.fulfill(json={'status': 'needs_clarification', 'generation_id': 'synthetic-generation',
                    'model': {**final, 'name': final['model']},
                    'routing': {'fallback_used': phase['fallback'] is not None,
                                'primary': phase['primary'], 'external_processing_attempted': phase['primary']['external_processing']}})
            files = {'/': (UI/'index.html', 'text/html'), '/assets/passport.js': (UI/'passport.js', 'text/javascript'),
                     '/assets/passport.css': (UI/'passport.css', 'text/css'), '/assets/base.css': (ROOT/'web/styles.css', 'text/css')}
            if path in files:
                file, mime = files[path]
                return route.fulfill(body=file.read_bytes(), content_type=mime+'; charset=utf-8')
            return route.fulfill(status=404)
        with sync_playwright() as p:
            executable = os.environ.get('AYURIP_BROWSER_EXECUTABLE')
            if not executable and os.name == 'nt':
                executable = 'C:/Program Files/Google/Chrome/Application/chrome.exe'
            browser = p.chromium.launch(headless=True, **({'executable_path': executable} if executable else {}))
            try:
                context = browser.new_context(viewport={'width': 390, 'height': 844})
                context.route('**/*', serve)
                page = context.new_page()
                page.on('pageerror', lambda e: errors.append(str(e)))
                for primary, fallback in [(groq, gemini), (gemini, local), (local, None)]:
                    phase.update(primary=primary, fallback=fallback)
                    page.goto('http://127.0.0.1:8003/')
                    page.wait_for_function("!document.getElementById('submit-passport').disabled")
                    notice = page.locator('#external-processing-notice')
                    if primary['provider'] == 'groq':
                        self.assertIn('Groq and Google Gemini', notice.inner_text())
                    elif primary['provider'] == 'gemini':
                        self.assertIn('Google Gemini', notice.inner_text())
                        self.assertNotIn('Groq', notice.inner_text())
                    else:
                        self.assertTrue(notice.is_hidden())
                    page.locator('#example-fill').click()
                    page.locator('#acknowledgment').check()
                    before = len(sent)
                    if primary['external_processing']:
                        page.locator('#submit-passport').click()
                        self.assertEqual(len(sent), before)
                        page.locator('#external-acknowledgment').check()
                    page.locator('#submit-passport').click()
                    page.locator('#passport-results').wait_for(state='visible')
                    self.assertEqual(sent[-1]['acknowledged_external_processing'], primary['external_processing'])
                    metadata = page.locator('.result-meta').inner_text()
                    self.assertIn('Research model: '+(fallback or primary)['model'], metadata)
                    if primary['provider'] == 'groq':
                        self.assertIn('Google Gemini fallback', metadata)
                        self.assertNotIn('local Qwen', metadata)
                        self.assertIn('not processed exclusively on this computer', metadata)
                    self.assertTrue(page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'))
                self.assertEqual(errors, [])
            finally:
                browser.close()


if __name__ == '__main__':
    unittest.main()
