from __future__ import annotations

import unittest

import pandas as pd

from technical.excel_loader import TechnicalDataError, normalize_checklist


def frames(days: int = 50) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2026-01-01", periods=days, freq="D")
    summary_rows = []
    detail_rows = []
    for index, date in enumerate(dates):
        close = 100.0 + index
        summary_rows.append(
            {
                "Commodity": "AAA",
                "Fecha": date,
                "Senal": "Largo" if index == days - 1 else "Mixto",
                "Condiciones_Largo": "prueba",
                "Condiciones_Corto": "prueba",
            }
        )
        values = {
            "EMA": f"EMA50={close - 2:.2f}, EMA200={close - 5:.2f}, Close={close:.2f}",
            "Cruce de Medias": "Sin cruce",
            "MACD": "0.50/0.25",
            "RSI": "58.50",
            "ATR": "1.5>1.4",
            "Volumen": "10.0%",
            "Gap": "Sin gaps",
        }
        for indicator, value in values.items():
            detail_rows.append(
                {
                    "Commodity": "AAA",
                    "Fecha": date,
                    "Indicador": indicator,
                    "Valor": value,
                    "Condicion Evaluada": "prueba",
                    "Resultado": "Sin senal",
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


class TechnicalExcelTests(unittest.TestCase):
    def test_normalizes_checklist_and_calculates_historical_metrics(self) -> None:
        summary, detail = frames()
        dataset = normalize_checklist(
            summary,
            detail,
            source_file="test.xlsx",
            source_sha256="abc",
        )
        latest = dataset.records[-1]
        self.assertEqual(len(dataset.records), 50)
        self.assertEqual(latest.trend, "Alcista")
        self.assertAlmostEqual(latest.variation_pct or 0, 100 / 148, places=5)
        self.assertAlmostEqual(latest.ma20 or 0, 139.5)
        self.assertAlmostEqual(latest.ma50 or 0, 124.5)
        self.assertIsNone(latest.support)
        self.assertIsNone(latest.resistance)

    def test_rejects_duplicate_business_key(self) -> None:
        summary, detail = frames(2)
        summary = pd.concat([summary, summary.iloc[[0]]], ignore_index=True)
        with self.assertRaises(TechnicalDataError):
            normalize_checklist(
                summary,
                detail,
                source_file="test.xlsx",
                source_sha256="abc",
            )


if __name__ == "__main__":
    unittest.main()
