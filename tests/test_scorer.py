import math
import unittest

from src.hawkometer.scorer import HawkometerScorer


class TestScorer(unittest.TestCase):
    def test_scores_hawkish_and_dovish_phrases(self):
        scorer = HawkometerScorer("data/hawkometer.json")
        result = scorer.score(
            "Inflation remains elevated. Inflation remains elevated. Disinflation continues.",
            central_bank="FED",
        )
        expected_hawkish = 3.0 * (1 + math.log(2))
        expected_dovish = 2.0 * (1 + math.log(1))
        self.assertAlmostEqual(result["hawkish_score"], expected_hawkish)
        self.assertAlmostEqual(result["dovish_score"], expected_dovish)
        self.assertAlmostEqual(result["raw_score"], expected_hawkish - expected_dovish)
        self.assertEqual(result["classification"], "Hawkish")
        self.assertEqual(result["lean"], "hawkish")
        self.assertGreater(result["confidence"], 0)
        self.assertEqual(result["score"], result["final_score"])
        self.assertEqual(len(result["matched_phrases"]), 2)

    def test_applies_central_bank_filter(self):
        scorer = HawkometerScorer("data/hawkometer.json")
        fed = scorer.score("The committee remains highly attentive to inflation risks.", central_bank="FED")
        boe = scorer.score("The committee remains highly attentive to inflation risks.", central_bank="BOE")
        self.assertGreater(fed["final_score"], 0)
        self.assertEqual(boe["final_score"], 0)

    def test_clips_score(self):
        scorer = HawkometerScorer("data/hawkometer.json")
        text = " ".join(["further tightening may be required"] * 200)
        result = scorer.score(text)
        self.assertEqual(result["final_score"], 10)

    def test_neutral_phrase_is_reported_but_does_not_score(self):
        scorer = HawkometerScorer("data/hawkometer.json")
        result = scorer.score("Policy remains data dependent.")
        self.assertEqual(result["final_score"], 0)
        self.assertEqual(result["classification"], "Neutral")
        self.assertEqual(result["lean"], "none")
        self.assertEqual(result["matched_phrases"][0]["direction"], "neutral")


if __name__ == "__main__":
    unittest.main()
