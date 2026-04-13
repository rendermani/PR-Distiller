"""
Unit tests for pipeline/deduplication.py — DeduplicationFilter.

Tests verify:
- First occurrence of a string is not a duplicate.
- Exact repeat is detected as duplicate.
- reset() clears all seen state.
- Whitespace-normalized equivalents are treated as duplicates.
- Empty string is handled without error.

Run with:
    cd /home/mani/Projects/PR-Analysis/backend
    ./venv/bin/python -m pytest tests/test_deduplication.py -v --tb=short
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.deduplication import DeduplicationFilter


class TestDeduplicationFilterFirstOccurrence(unittest.TestCase):
    def setUp(self):
        self.dedup = DeduplicationFilter()

    def test_first_occurrence_is_not_duplicate(self):
        self.assertFalse(self.dedup.is_duplicate("use async/await instead of promises"))

    def test_first_occurrence_empty_string_is_not_duplicate(self):
        self.assertFalse(self.dedup.is_duplicate(""))

    def test_two_distinct_strings_are_both_not_duplicate_on_first_call(self):
        self.assertFalse(self.dedup.is_duplicate("string one"))
        self.assertFalse(self.dedup.is_duplicate("string two"))


class TestDeduplicationFilterExactDuplicate(unittest.TestCase):
    def setUp(self):
        self.dedup = DeduplicationFilter()

    def test_exact_repeat_is_duplicate(self):
        text = "Always use a linter before committing code."
        self.dedup.is_duplicate(text)
        self.assertTrue(self.dedup.is_duplicate(text))

    def test_empty_string_repeat_is_duplicate(self):
        self.dedup.is_duplicate("")
        self.assertTrue(self.dedup.is_duplicate(""))

    def test_third_occurrence_is_still_duplicate(self):
        text = "No magic numbers."
        self.dedup.is_duplicate(text)
        self.dedup.is_duplicate(text)
        self.assertTrue(self.dedup.is_duplicate(text))


class TestDeduplicationFilterNormalization(unittest.TestCase):
    """The filter normalizes whitespace and case before hashing."""

    def setUp(self):
        self.dedup = DeduplicationFilter()

    def test_extra_whitespace_treated_as_same_content(self):
        self.dedup.is_duplicate("use  async/await")
        self.assertTrue(self.dedup.is_duplicate("use async/await"))

    def test_different_case_treated_as_same_content(self):
        self.dedup.is_duplicate("Use Async/Await")
        self.assertTrue(self.dedup.is_duplicate("use async/await"))

    def test_leading_trailing_whitespace_treated_as_same_content(self):
        self.dedup.is_duplicate("  trim this  ")
        self.assertTrue(self.dedup.is_duplicate("trim this"))


class TestDeduplicationFilterReset(unittest.TestCase):
    def setUp(self):
        self.dedup = DeduplicationFilter()

    def test_reset_clears_seen_hashes(self):
        text = "avoid bare except clauses"
        self.dedup.is_duplicate(text)
        self.dedup.reset()
        # After reset the same string must be a first occurrence again.
        self.assertFalse(self.dedup.is_duplicate(text))

    def test_reset_on_empty_filter_does_not_raise(self):
        # reset() on a fresh filter must be a no-op.
        self.dedup.reset()
        self.assertFalse(self.dedup.is_duplicate("anything"))


if __name__ == "__main__":
    unittest.main()
