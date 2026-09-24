from __future__ import annotations

import unittest

from bulletin.models import CommodityInsight


class BulletinModelTests(unittest.TestCase):
    def test_normalizes_empty_and_oversized_editorial_fields(self) -> None:
        insight = CommodityInsight.from_dict(
            {
                "symbol": "QBS",
                "classification": "Neutral",
                "analysis_lines": ["", "A" * 300],
                "factors": ["", "B" * 300, "Factor válido"],
            }
        )
        self.assertTrue(all(insight.analysis_lines))
        self.assertTrue(all(insight.factors))
        self.assertLessEqual(max(map(len, insight.analysis_lines)), 180)
        self.assertLessEqual(max(map(len, insight.factors)), 150)


if __name__ == "__main__":
    unittest.main()
