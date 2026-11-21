"""The two front ends behind ``tiny_voca.text``: eSpeak (default) and dizione ("own").

The dizione tests are skipped when the package or its built data is missing.
"""

import unittest

from tiny_voca import text
from tiny_voca.text import SYMBOLS, SYMBOL_TO_ID

# Fingerprint of the eSpeak front end (version 3): the tokens of every v3
# annotation row and of the checkpoints trained on them.
ESPEAK_FINGERPRINT = "a6f1cf322f2958270371f255f18ba9c41d12830490e46df64762a493878be338"

try:
    import dizione
    from dizione.engine import default_directory
    HAVE_DIZIONE = (default_directory() / "lexicon.tsv").exists()
except ImportError:
    dizione, HAVE_DIZIONE = None, False


class FrontEndSwitchTests(unittest.TestCase):
    def tearDown(self):
        text.set_frontend("espeak")

    def test_espeak_stays_the_default_and_keeps_its_fingerprint(self):
        text.set_frontend("espeak")
        self.assertEqual(text.frontend(), "espeak")
        self.assertEqual(text.frontend_fingerprint(), ESPEAK_FINGERPRINT)
        self.assertEqual(text.frontend_identity()["frontend_version"], 3)
        with self.assertRaises(ValueError):
            text.set_frontend("festival")

    @unittest.skipUnless(dizione, "needs the dizione package")
    def test_dizione_has_its_own_identity_and_the_same_inventory(self):
        self.assertEqual(tuple(dizione.symbols.SYMBOLS), SYMBOLS)
        text.set_frontend("own")
        identity = text.frontend_identity()
        self.assertEqual((identity["frontend_version"], identity["phonemizer"], identity["symbols"]), (dizione.VERSION, "dizione", list(SYMBOLS)))
        self.assertNotEqual(text.frontend_fingerprint(), ESPEAK_FINGERPRINT)

    @unittest.skipUnless(HAVE_DIZIONE, "needs dizione and its built data (DIZIONE_DATA)")
    def test_phonemize_uses_dizione_when_selected(self):
        text.set_frontend("own")
        result = text.phonemize("Hello, it costs $3.50.")
        expected = dizione.phonemize("Hello, it costs $3.50.")
        self.assertEqual((result.symbols, result.stress), (expected.symbols, expected.stress))
        self.assertEqual(result.ids, tuple(SYMBOL_TO_ID[s] for s in result.symbols))
        self.assertEqual(result.unknown, ())
        self.assertEqual(text.frontend_runtime(), {"phonemizer": dizione.identity()["fingerprint"]})


if __name__ == "__main__":
    unittest.main()
