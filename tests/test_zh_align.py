"""Tests for the MFA TextGrid adapter (no MFA install required)."""

import unittest

from tiny_voca import zh_text
from tiny_voca.mfa_align import Interval, canonical_label, durations, parse_intervals, phone_positions

ZH = zh_text.PHONE_SET

TEXTGRID = '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1.2
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 1.2
        intervals: size = 6
        intervals [1]:
            xmin = 0
            xmax = 0.1
            text = ""
        intervals [2]:
            xmin = 0.1
            xmax = 0.2
            text = "n"
        intervals [3]:
            xmin = 0.2
            xmax = 0.5
            text = "i3"
        intervals [4]:
            xmin = 0.5
            xmax = 0.7
            text = "sil"
        intervals [5]:
            xmin = 0.7
            xmax = 0.8
            text = "h"
        intervals [6]:
            xmin = 0.8
            xmax = 1.2
            text = "ao3"
'''


class ParseTest(unittest.TestCase):
    def test_intervals_are_read_in_order(self):
        intervals = parse_intervals(TEXTGRID)
        self.assertEqual(len(intervals), 6)
        self.assertEqual(intervals[1], Interval(0.1, 0.2, "n"))
        self.assertEqual(intervals[-1].label, "ao3")

    def test_missing_tier_is_reported(self):
        with self.assertRaises(ValueError):
            parse_intervals(TEXTGRID, tier="words")


class LabelTest(unittest.TestCase):
    def test_silence_markers_are_not_phones(self):
        for label in ("", "sil", "sp", "spn"):
            self.assertIsNone(canonical_label(label, ZH))

    def test_inventory_label_passes_through(self):
        self.assertEqual(canonical_label("ong1", ZH), "ong1")
        self.assertEqual(canonical_label("zh", ZH), "zh")

    def test_umlaut_is_folded(self):
        # A final is its own phone; whole syllables are not tokens.
        self.assertEqual(canonical_label("ü4", ZH), "v4")
        self.assertEqual(canonical_label("ü", ZH), None)

    def test_unknown_label_is_none(self):
        self.assertIsNone(canonical_label("qq9", ZH))

    def test_phone_positions_skips_punctuation(self):
        self.assertEqual(phone_positions(("<bos>", "n", "i3", ",", "h", "ao3", "<eos>"), ZH), [1, 2, 4, 5])


class DurationsTest(unittest.TestCase):
    SYMBOLS = ("<bos>", "n", "i3", "h", "ao3", "<eos>")

    def run_counts(self, symbols=None, **kwargs):
        intervals = parse_intervals(TEXTGRID)
        return durations(intervals, symbols or self.SYMBOLS, frames=112, hop=256,
                         sample_rate=24000, phone_set=ZH, **kwargs)

    def test_counts_sum_to_frames(self):
        counts = self.run_counts()
        self.assertEqual(sum(counts), 112)
        self.assertEqual(len(counts), len(self.SYMBOLS))

    def test_leading_silence_goes_to_bos(self):
        self.assertGreaterEqual(self.run_counts()[0], 1)

    def test_every_phone_has_a_frame(self):
        counts = self.run_counts()
        for index in phone_positions(self.SYMBOLS, ZH):
            self.assertGreaterEqual(counts[index], 1)

    def test_interior_gap_goes_to_punctuation(self):
        symbols = ("<bos>", "n", "i3", ",", "h", "ao3", "<eos>")
        counts = self.run_counts(symbols)
        self.assertGreaterEqual(counts[3], 7)
        self.assertEqual(sum(counts), 112)

    def test_short_gap_stays_with_previous_phone(self):
        symbols = ("<bos>", "n", "i3", ",", "h", "ao3", "<eos>")
        counts = self.run_counts(symbols, min_pause_frames=100)
        self.assertEqual(counts[3], 0)

    def test_count_mismatch_is_reported(self):
        with self.assertRaises(ValueError):
            self.run_counts(("<bos>", "n", "<eos>"))


if __name__ == "__main__":
    unittest.main()
