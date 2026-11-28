"""Unit tests for the Mandarin front end.

These run without audio, without torch and without pypinyin being installed in
a particular version: everything under test is pure text handling.
"""

import unittest

from tiny_voca import zh_text
from tiny_voca.zh_text import (PHONE_SET, SYMBOLS, SYMBOL_TO_ID, Syllable, apply_sandhi,
                               normalize_text, phonemize, split_pinyin, tokenize)


class InventoryTest(unittest.TestCase):
    def test_symbols_are_unique(self):
        self.assertEqual(len(SYMBOLS), len(set(SYMBOLS)))

    def test_specials_come_first(self):
        self.assertEqual(SYMBOLS[:5], ("<pad>", "<bos>", "<eos>", "<sp>", "<unk>"))

    def test_every_final_has_five_tones(self):
        for final in zh_text.FINALS:
            for tone in (1, 2, 3, 4, 5):
                self.assertIn(final + str(tone), SYMBOL_TO_ID)

    def test_initials_are_phones_but_not_vowels(self):
        for initial in zh_text.INITIALS:
            self.assertIn(initial, PHONE_SET)
            self.assertNotIn(initial, zh_text.VOWEL_SET)

    def test_phone_ids_cover_inventory(self):
        self.assertEqual(len(zh_text.PHONE_IDS), len(zh_text.PHONES))


class SplitPinyinTest(unittest.TestCase):
    def test_plain_syllable(self):
        self.assertEqual(split_pinyin("zhong1"), ("zh", "ong", 1))

    def test_aspirated_affricate(self):
        self.assertEqual(split_pinyin("chi2"), ("ch", "i", 2))

    def test_umlaut_is_written_v(self):
        self.assertEqual(split_pinyin("lv4"), ("l", "v", 4))
        self.assertEqual(split_pinyin("nv3"), ("n", "v", 3))

    def test_umlaut_after_j_q_x(self):
        self.assertEqual(split_pinyin("ju4"), ("j", "v", 4))
        self.assertEqual(split_pinyin("que4"), ("q", "ve", 4))
        self.assertEqual(split_pinyin("xuan1"), ("x", "van", 1))
        self.assertEqual(split_pinyin("jun1"), ("j", "vn", 1))

    def test_y_and_w_are_zero_initial(self):
        self.assertEqual(split_pinyin("yi1"), ("", "i", 1))
        self.assertEqual(split_pinyin("you3"), ("", "iu", 3))
        self.assertEqual(split_pinyin("yu2"), ("", "v", 2))
        self.assertEqual(split_pinyin("wu3"), ("", "u", 3))
        self.assertEqual(split_pinyin("wo3"), ("", "uo", 3))
        self.assertEqual(split_pinyin("wang2"), ("", "uang", 2))

    def test_unknown_final_is_rejected(self):
        self.assertIsNone(split_pinyin("hm5"))
        self.assertIsNone(split_pinyin("zzz1"))


class SandhiTest(unittest.TestCase):
    @staticmethod
    def make(*items):
        return [Syllable("", final, tone, char) for final, tone, char in items]

    def test_third_tone_pair(self):
        result = apply_sandhi(self.make(("i", 3, "你"), ("ao", 3, "好")))
        self.assertEqual([s.tone for s in result], [2, 3])

    def test_three_third_tones(self):
        result = apply_sandhi(self.make(("uo", 3, "我"), ("en", 3, "很"), ("ao", 3, "好")))
        self.assertEqual([s.tone for s in result], [2, 2, 3])

    def test_bu_before_fourth(self):
        result = apply_sandhi(self.make(("u", 4, "不"), ("ui", 4, "对")))
        self.assertEqual([s.tone for s in result], [2, 4])

    def test_bu_before_other_tones_is_unchanged(self):
        result = apply_sandhi(self.make(("u", 4, "不"), ("ing", 2, "行")))
        self.assertEqual([s.tone for s in result], [4, 2])

    def test_yi_before_fourth_becomes_second(self):
        result = apply_sandhi(self.make(("i", 1, "一"), ("e", 4, "个")))
        self.assertEqual([s.tone for s in result], [2, 4])

    def test_yi_before_first_becomes_fourth(self):
        result = apply_sandhi(self.make(("i", 1, "一"), ("ian", 1, "天")))
        self.assertEqual([s.tone for s in result], [4, 1])

    def test_other_yi1_is_untouched(self):
        result = apply_sandhi(self.make(("i", 1, "衣"), ("ang", 4, "裳")))
        self.assertEqual([s.tone for s in result], [1, 4])

    def test_yi_after_ordinal_marker_keeps_first_tone(self):
        # 第一名: 一 is an ordinal, not a counting word before 名.
        result = apply_sandhi(self.make(("i", 4, "第"), ("i", 1, "一"), ("ing", 2, "名")))
        self.assertEqual([s.tone for s in result], [4, 1, 2])

    def test_yi_at_the_end_of_a_number_keeps_first_tone(self):
        # 十一个: 一 is the units digit of eleven.
        result = apply_sandhi(self.make(("i", 2, "十"), ("i", 1, "一"), ("e", 4, "个")))
        self.assertEqual([s.tone for s in result], [2, 1, 4])

    def test_yi_in_a_date_keeps_first_tone(self):
        # 一月 and 一日 are names of a month and a day.
        result = apply_sandhi(self.make(("i", 1, "一"), ("ve", 4, "月")))
        self.assertEqual([s.tone for s in result], [1, 4])

    def test_yi_already_changed_upstream_is_restored_in_numbers(self):
        # pypinyin may hand over 一 with a changed tone; inside a number it goes back to the first tone.
        result = apply_sandhi(self.make(("i", 4, "第"), ("i", 2, "一"), ("i", 4, "次")))
        self.assertEqual([s.tone for s in result], [4, 1, 4])

    def test_yi_tones_in_whole_sentences(self):
        from tiny_voca.zh_text import syllables_of
        def tones(text):
            return [s.tone for s in syllables_of(text) if s.char == "一"]
        self.assertEqual(tones("第一名"), [1])
        self.assertEqual(tones("十一个人"), [1])
        self.assertEqual(tones("一月一日"), [1, 1])
        self.assertEqual(tones("二零二一年"), [1])
        self.assertEqual(tones("一个"), [2])
        self.assertEqual(tones("一天"), [4])
        self.assertEqual(tones("一千一百"), [4, 4])
        self.assertEqual(tones("一百一十"), [4, 1])


class NormalizeTest(unittest.TestCase):
    def test_chinese_punctuation_folds_into_inventory(self):
        self.assertEqual(normalize_text("你好，世界。"), "你好,世界.")

    def test_three_dots_become_ellipsis(self):
        self.assertEqual(normalize_text("等等..."), "等等…")

    def test_year_is_read_digit_by_digit(self):
        self.assertEqual(normalize_text("2026年"), "二零二六年")

    def test_teens_drop_the_leading_one(self):
        self.assertEqual(normalize_text("10"), "十")
        self.assertEqual(normalize_text("15"), "十五")

    def test_percentage(self):
        self.assertEqual(normalize_text("50%"), "百分之五十")

    def test_decimal(self):
        self.assertEqual(normalize_text("3.14"), "三点一四")

    def test_long_digit_runs_are_read_digit_by_digit(self):
        # Phone numbers must not be read as a cardinal number.
        self.assertEqual(normalize_text("13812345678"), "一三八一二三四五六七八")


class PhonemizeTest(unittest.TestCase):
    def test_tokens_are_bos_phones_eos(self):
        self.assertEqual(" ".join(phonemize("你好").symbols), "<bos> n i2 h ao3 <eos>")

    def test_tone_lives_in_the_final_token(self):
        symbols = phonemize("绿色").symbols
        self.assertIn("v4", symbols)
        self.assertIn("s", symbols)
        self.assertIn("e4", symbols)

    def test_punctuation_is_its_own_token(self):
        symbols = phonemize("好，走").symbols
        self.assertIn(",", symbols)

    def test_repeated_punctuation_collapses(self):
        symbols = phonemize("好！！").symbols
        self.assertEqual(list(symbols).count("!"), 1)

    def test_ids_may_be_used_as_tokens(self):
        result = phonemize("中文")
        self.assertEqual(len(result.ids), len(result.symbols))
        self.assertLess(max(result.ids), len(SYMBOLS))

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            phonemize("   ")

    def test_text_with_no_syllable_raises(self):
        with self.assertRaises(zh_text.NothingToPronounce):
            phonemize("……")

    def test_digits_alone_are_read_aloud(self):
        # "1234" normalizes to 一千二百三十四, which is pronounceable.
        self.assertIn("q", phonemize("1234").symbols)


class OwnersTest(unittest.TestCase):
    """The aligner hands frames to tokens through these owners."""

    def test_owners_mark_the_syllable_of_every_token(self):
        from tiny_voca.zh_text import phonemize_with_owners, syllables_of

        result, owners = phonemize_with_owners("你好，走")
        self.assertEqual(len(owners), len(result.symbols))
        self.assertEqual(owners, (-1, 0, 0, 1, 1, -1, 2, 2, -1))
        self.assertEqual(max(owners) + 1, len(syllables_of("你好，走")))

    def test_punctuation_and_boundaries_have_no_owner(self):
        from tiny_voca.zh_text import phonemize_with_owners

        result, owners = phonemize_with_owners("好。")
        self.assertEqual(owners[0], -1)
        self.assertEqual(owners[-1], -1)
        self.assertEqual(owners[result.symbols.index(".")], -1)


class TokenizeTest(unittest.TestCase):
    def test_manual_pieces(self):
        result = tokenize([("syllable", Syllable("m", "a", 1, "妈")),
                           ("punct", ","),
                           ("syllable", Syllable("", "i", 4, "意"))])
        self.assertEqual(result.symbols, ("<bos>", "m", "a1", ",", "i4", "<eos>"))

    def test_unknown_final_becomes_unk_and_is_reported(self):
        result = tokenize([("syllable", Syllable("", "zz", 3, "?")), ("punct", ".")])
        self.assertIn("<unk>", result.symbols)
        self.assertEqual(result.unknown, ("zz3",))


if __name__ == "__main__":
    unittest.main()
