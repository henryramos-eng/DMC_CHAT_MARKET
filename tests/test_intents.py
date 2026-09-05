from __future__ import annotations

import unittest

from rag.intents import Intent, classify_intent


class IntentTests(unittest.TestCase):
    def test_basic_intents(self) -> None:
        self.assertIs(classify_intent("Hola"), Intent.GREETING)
        self.assertIs(classify_intent("Gracias"), Intent.THANKS)
        self.assertIs(classify_intent("Hasta luego"), Intent.GOODBYE)

    def test_report_and_obvious_out_of_scope(self) -> None:
        self.assertIs(
            classify_intent("¿Qué se menciona sobre China?"),
            Intent.REPORT_QUERY,
        )
        self.assertIs(
            classify_intent("¿Quién ganó el Mundial de fútbol?"),
            Intent.OUT_OF_SCOPE,
        )


if __name__ == "__main__":
    unittest.main()
