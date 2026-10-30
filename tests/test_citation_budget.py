import unittest

from tools.benchmark_citation_budget import budget_profile


class BudgetTests(unittest.TestCase):
    def test_only_output_limit_changes_and_parent_remains_immutable(self):
        original = {"num_predict": 200, "num_ctx": 4096, "keep_alive": "30m", "think": False}
        result = budget_profile(original)
        self.assertEqual(result, {**original, "num_predict": 600})
        self.assertEqual(original["num_predict"], 200)
        self.assertEqual([key for key in result if result[key] != original[key]], ["num_predict"])

    def test_noncomparable_parent_settings_are_rejected(self):
        original = {"num_predict": 200, "num_ctx": 4096, "keep_alive": "30m"}
        for change in ({"num_predict": 400}, {"num_ctx": 8192}, {"think": True}, {"extra_option": True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                budget_profile({**original, **change})
