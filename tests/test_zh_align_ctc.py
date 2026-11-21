"""Tests for the character-level aligner (no model download required)."""

import unittest

from tiny_voca.zh_align_ctc import durations_from_chars, initial_fraction, split_syllable


class SplitTest(unittest.TestCase):
    def test_zero_initial_owns_the_whole_span(self):
        self.assertEqual(split_syllable(10, 30, ""), (0, 20))

    def test_single_frame_span(self):
        self.assertEqual(split_syllable(5, 6, "h"), (1, 0))

    def test_voiced_initial_uses_the_class_fraction(self):
        head, tail = split_syllable(0, 20, "m")
        self.assertEqual(head + tail, 20)
        self.assertGreaterEqual(head, 1)
        self.assertGreaterEqual(tail, 1)
        self.assertAlmostEqual(head / 20, initial_fraction("m"), delta=0.06)

    def test_voiceless_initial_breaks_where_voicing_starts(self):
        # 40 frames, voicing turns on at frame 12.
        voiced = [frame >= 12 for frame in range(40)]
        head, tail = split_syllable(0, 40, "h", voiced)
        self.assertEqual(head, 12)
        self.assertEqual(tail, 28)

    def test_voicing_already_on_at_the_edge_falls_back(self):
        voiced = [True] * 40
        head, tail = split_syllable(0, 40, "h", voiced)
        self.assertEqual(head + tail, 40)
        self.assertGreaterEqual(tail, 1)

    def test_fricative_takes_more_than_a_stop(self):
        self.assertGreater(initial_fraction("s"), initial_fraction("p"))


class DurationsTest(unittest.TestCase):
    SYMBOLS = ("<bos>", "n", "i3", "h", "ao3", "<eos>")
    OWNERS = (-1, 0, 0, 1, 1, -1)

    def test_counts_cover_every_frame(self):
        counts = durations_from_chars(self.SYMBOLS, self.OWNERS, [(5, 30), (35, 80)],
                                      [False] * 100, 100)
        self.assertEqual(sum(counts), 100)
        self.assertEqual(len(counts), len(self.SYMBOLS))

    def test_every_token_of_a_syllable_gets_frames(self):
        counts = durations_from_chars(self.SYMBOLS, self.OWNERS, [(5, 30), (35, 80)],
                                      [False] * 100, 100)
        for index in (1, 2, 3, 4):
            self.assertGreaterEqual(counts[index], 1)

    def test_leading_and_trailing_go_to_the_boundary_tokens(self):
        counts = durations_from_chars(self.SYMBOLS, self.OWNERS, [(5, 30), (35, 80)],
                                      [False] * 100, 100)
        self.assertEqual(counts[0], 5)
        self.assertEqual(counts[-1], 20)

    def test_long_gap_goes_to_punctuation(self):
        symbols = ("<bos>", "n", "i3", ",", "h", "ao3", "<eos>")
        owners = (-1, 0, 0, -1, 1, 1, -1)
        counts = durations_from_chars(symbols, owners, [(2, 20), (40, 70)], [False] * 80, 80)
        self.assertGreaterEqual(counts[3], 7)
        self.assertEqual(sum(counts), 80)

    def test_short_gap_stays_with_the_previous_token(self):
        symbols = ("<bos>", "n", "i3", ",", "h", "ao3", "<eos>")
        owners = (-1, 0, 0, -1, 1, 1, -1)
        counts = durations_from_chars(symbols, owners, [(2, 20), (23, 70)], [False] * 80, 80,
                                      min_pause_frames=50)
        self.assertEqual(counts[3], 0)

    def test_span_count_mismatch_is_reported(self):
        with self.assertRaises(ValueError):
            durations_from_chars(self.SYMBOLS, self.OWNERS, [(5, 30)], [False] * 100, 100)

    def test_empty_span_is_reported(self):
        with self.assertRaises(ValueError):
            durations_from_chars(self.SYMBOLS, self.OWNERS, [(10, 10), (35, 80)], [False] * 100, 100)


if __name__ == "__main__":
    unittest.main()
