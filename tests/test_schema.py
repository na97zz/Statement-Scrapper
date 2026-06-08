import json
import unittest
from pathlib import Path

from src.hawkometer.schema import HawkometerConfig
from src.hawkometer.validator import validate_config


class TestSchema(unittest.TestCase):
    def test_default_config_is_valid(self):
        config = json.loads(Path("data/hawkometer.json").read_text(encoding="utf-8"))
        self.assertEqual(validate_config(config), [])

    def test_config_dataclass_loads_phrase_entries(self):
        config = json.loads(Path("data/hawkometer.json").read_text(encoding="utf-8"))
        parsed = HawkometerConfig.from_dict(config)
        self.assertGreater(len(parsed.phrase_library), 0)
        self.assertEqual(parsed.phrase_library[0].direction, "hawkish")

    def test_invalid_weight_fails_validation(self):
        config = json.loads(Path("data/hawkometer.json").read_text(encoding="utf-8"))
        config["phrase_library"][0]["weight"] = 4
        errors = validate_config(config)
        self.assertTrue(any("weight outside" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
