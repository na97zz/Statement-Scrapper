import unittest

from src.hawkometer.classifier import classify_score, confidence_score


class TestClassifier(unittest.TestCase):
    def test_classification_buckets(self):
        cases = [
            (8, "Strongly Hawkish", "hawkish"),
            (10, "Strongly Hawkish", "hawkish"),
            (3, "Hawkish", "hawkish"),
            (7.99, "Hawkish", "hawkish"),
            (0.5, "Neutral-Hawkish", "none"),
            (2.99, "Neutral-Hawkish", "hawkish"),
            (0, "Neutral", "none"),
            (-0.5, "Neutral", "none"),
            (-0.51, "Neutral-Dovish", "dovish"),
            (-3, "Dovish", "dovish"),
            (-7.99, "Dovish", "dovish"),
            (-8, "Strongly Dovish", "dovish"),
            (-10, "Strongly Dovish", "dovish"),
        ]
        for score, classification, lean in cases:
            with self.subTest(score=score):
                result = classify_score(score)
                self.assertEqual(result["classification"], classification)
                self.assertEqual(result["lean"], lean)

    def test_confidence_increases_with_more_evidence_and_diversity(self):
        weak = [
            {"direction": "hawkish", "theme": "inflation_too_high", "weight": 1, "count": 1},
        ]
        strong = [
            {"direction": "hawkish", "theme": "inflation_too_high", "weight": 3, "count": 2},
            {"direction": "hawkish", "theme": "wage_inflation", "weight": 2, "count": 2},
            {"direction": "hawkish", "theme": "restrictive_stance", "weight": 2, "count": 2},
            {"direction": "hawkish", "theme": "upside_inflation_risks", "weight": 2, "count": 2},
        ]
        self.assertGreater(confidence_score(6, strong), confidence_score(1, weak))

    def test_conflicting_phrases_reduce_confidence(self):
        aligned = [
            {"direction": "hawkish", "theme": "inflation_too_high", "weight": 3, "count": 2},
            {"direction": "hawkish", "theme": "restrictive_stance", "weight": 2, "count": 2},
        ]
        conflicted = [
            {"direction": "hawkish", "theme": "inflation_too_high", "weight": 3, "count": 2},
            {"direction": "dovish", "theme": "disinflation", "weight": -2, "count": 2},
        ]
        self.assertLess(confidence_score(1, conflicted), confidence_score(5, aligned))

    def test_no_matches_have_zero_confidence(self):
        self.assertEqual(confidence_score(0, []), 0.0)


if __name__ == "__main__":
    unittest.main()
