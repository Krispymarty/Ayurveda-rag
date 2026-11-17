import unittest
from unittest.mock import patch
from ipindia_corpus.request_policy import classify


class DemoLanguageTests(unittest.TestCase):
    def test_statutory_terms_are_not_personal_advice(self):
        for q in [
            'What does section 3(e) say about a mere admixture?',
            'What does archived section 3(e) say about a mere product-mixture?',
            'From archived section 3, quote any passage about a mere idea.',
            'धारा 3(i) में चिकित्सीय उपचार की प्रक्रियाओं के बारे में क्या कहा गया है?',
            'धारा 10 के अनुसार पूर्ण पेटेंट विनिर्देश में कौन-सी जानकारी शामिल होनी चाहिए?',
            'धारा 3 में इलाज की प्रक्रिया के बारे में क्या प्रावधान है?']:
            with self.subTest(question=q):
                self.assertTrue(classify(q)['allowed'])

    def test_real_advice_is_not_unblocked_by_adding_a_section(self):
        for q in ['mere patent ke liye kya karun?', 'mere product ko patent mil sakta hai?',
                  'धारा 3 का हवाला देते हुए बुखार के उपचार की सलाह दें।',
                  'धारा 3 के साथ बच्चे के बुखार के उपचार की सलाह दें।', 'धारा 3 के अनुसार मेरे लिए कौन सा उपचार सही है?',
                  'मुझे इलाज बताएं', 'कौन सा उपचार लेना चाहिए?', 'धारा 3 के अनुसार कौन सा उपचार लेना चाहिए?',
                  'What dose should my child take under section 3?', 'Can I patent my mixture?',
                  'मेरे बच्चे की खुराक बताएं', 'Quote current law on treatment']:
            with self.subTest(question=q):
                self.assertFalse(classify(q)['allowed'])

    def test_hindi_query_mapping_retains_references_and_english_is_unchanged(self):
        from ipindia_corpus.demo_queries import retrieval_query
        q = 'संग्रहीत पेटेंट अधिनियम की धारा ३(p) में पारंपरिक ज्ञान के बारे में क्या कहा गया है?'
        mapped = retrieval_query(q)
        for fragment in ['Patents Act', 'section 3(p)', 'traditional knowledge']:
            self.assertIn(fragment, mapped)
        english = 'Quote sections 48 and 53 of the Patents Act'
        self.assertEqual(retrieval_query(english), english)
        self.assertEqual(retrieval_query('धारावाहिक कहानी'), 'धारावाहिक कहानी')

    def test_original_question_and_source_bytes_are_not_rewritten_for_models(self):
        from ipindia_corpus import demo_answers
        from test_groq_failover import Primary
        from test_span_selection import hit
        q = 'धारा ३(p) में पारंपरिक ज्ञान के बारे में क्या कहा गया है?'
        h = hit()
        primary = Primary(['{"span_ids":["S001"]}', '{"span_ids":["S001"]}'])
        seen = []
        with patch.object(demo_answers, 'read_setting', return_value='none'):
            demo_answers.answer(q, lambda query, **kw: seen.append(query) or {'generation_id':'g','hits':[h]},
                                lambda *args: None, lambda: None, primary=primary)
        self.assertIn('section 3(p)', seen[0])
        self.assertIn(q, primary.calls[0]['messages'][1]['content'])
        self.assertIn(h['text'], primary.calls[0]['messages'][1]['content'])
