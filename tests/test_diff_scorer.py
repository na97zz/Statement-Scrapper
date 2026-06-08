import unittest

from src.hawkometer.diff_scorer import HawkometerDiffScorer


class TestDiffScorer(unittest.TestCase):
    def test_added_hawkish_phrase_creates_hawkish_impulse(self):
        scorer = HawkometerDiffScorer("data/hawkometer.json")
        result = scorer.compare(
            "Inflation has eased.",
            "Inflation has eased but inflation remains elevated.",
        )
        self.assertGreater(result["diff_score"], 0)
        self.assertEqual(result["added_hawkish_phrases"][0]["phrase"], "inflation remains elevated")
        self.assertIn("hawkish", result["explanation_summary"])

    def test_removed_hawkish_phrase_creates_dovish_impulse(self):
        scorer = HawkometerDiffScorer("data/hawkometer.json")
        result = scorer.compare(
            "Inflation remains elevated.",
            "Inflation has eased.",
        )
        self.assertLess(result["diff_score"], 0)
        self.assertEqual(result["removed_hawkish_phrases"][0]["phrase"], "inflation remains elevated")

    def test_added_dovish_phrase_creates_dovish_impulse(self):
        scorer = HawkometerDiffScorer("data/hawkometer.json")
        result = scorer.compare(
            "Inflation remains elevated.",
            "Inflation remains elevated. Disinflation continues.",
        )
        self.assertLess(result["diff_score"], 0)
        self.assertEqual(result["added_dovish_phrases"][0]["phrase"], "disinflation continues")

    def test_removed_dovish_phrase_creates_hawkish_impulse(self):
        scorer = HawkometerDiffScorer("data/hawkometer.json")
        result = scorer.compare(
            "Disinflation continues.",
            "Inflation is above target.",
        )
        self.assertGreater(result["diff_score"], 0)
        self.assertEqual(result["removed_dovish_phrases"][0]["phrase"], "disinflation continues")

    def test_final_adjusted_score_is_clipped(self):
        scorer = HawkometerDiffScorer("data/hawkometer.json")
        result = scorer.compare("", " ".join(["further tightening may be required"] * 20))
        self.assertEqual(result["final_adjusted_score"], 10)


if __name__ == "__main__":
    unittest.main()
