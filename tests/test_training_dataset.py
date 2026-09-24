from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.prepare_dataset import load_examples, prepare  # noqa: E402


class TrainingDatasetTests(unittest.TestCase):
    def test_synthetic_examples_require_explicit_opt_in(self) -> None:
        source = ROOT / "training" / "data" / "examples.jsonl"
        with self.assertRaises(ValueError):
            load_examples(source)
        values = load_examples(source, allow_synthetic=True)
        self.assertEqual(len(values), 3)

    def test_split_is_by_date_and_never_empty(self) -> None:
        source = ROOT / "training" / "data" / "examples.jsonl"
        with tempfile.TemporaryDirectory() as directory:
            train, validation = prepare(
                source,
                Path(directory),
                allow_synthetic=True,
                validation_ratio=0.34,
            )
            train_dates = {
                json.loads(line)["date"]
                for line in train.read_text(encoding="utf-8").splitlines()
            }
            validation_dates = {
                json.loads(line)["date"]
                for line in validation.read_text(encoding="utf-8").splitlines()
            }
            self.assertTrue(train_dates)
            self.assertTrue(validation_dates)
            self.assertFalse(train_dates.intersection(validation_dates))


if __name__ == "__main__":
    unittest.main()
