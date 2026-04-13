"""
Unit tests for pipeline/security_redactor.py — SecurityRedactor.

Presidio loads spaCy models at construction time; that is an expected
one-time cost. All tests use a single shared instance via setUpClass.

Tests verify:
- Email addresses are redacted.
- AWS-style 20-char uppercase key IDs (AKIA pattern) are redacted.
- Non-PII plain text is returned unchanged.
- Empty string input returns empty string.
- Text with no PII is returned verbatim.

Phone-number detection is intentionally NOT asserted because Presidio's
phone-number recogniser is language/model dependent and can produce
false negatives for synthetic test numbers. Testing that boundary would
couple the tests to a specific Presidio model version.

Run with:
    cd /home/mani/Projects/PR-Analysis/backend
    ./venv/bin/python -m pytest tests/test_security_redactor.py -v --tb=short
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.security_redactor import SecurityRedactor

# A 20-character all-caps-and-digits string that matches the AWS key pattern.
FAKE_AWS_KEY = "AKIAIOSFODNN7EXAMPLE"


class TestSecurityRedactorSetup(unittest.TestCase):
    """Verify the redactor initialises without error."""

    @classmethod
    def setUpClass(cls):
        cls.redactor = SecurityRedactor()

    def test_instance_created_without_error(self):
        self.assertIsNotNone(self.redactor)


class TestRedactEmptyAndNoPII(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.redactor = SecurityRedactor()

    def test_empty_string_returns_empty_string(self):
        self.assertEqual(self.redactor.redact_text(""), "")

    def test_plain_text_without_pii_is_returned_unchanged(self):
        plain = "Always write unit tests before production code."
        result = self.redactor.redact_text(plain)
        self.assertEqual(result, plain)

    def test_code_snippet_without_pii_is_returned_unchanged(self):
        code = "def calculate_total(price: float, tax: float) -> float:\n    return price + tax"
        result = self.redactor.redact_text(code)
        self.assertEqual(result, code)


class TestRedactEmailAddresses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.redactor = SecurityRedactor()

    def test_email_is_removed_from_text(self):
        text = "Contact the author at alice@example.com for details."
        result = self.redactor.redact_text(text)
        self.assertNotIn("alice@example.com", result)

    def test_text_surrounding_email_is_preserved(self):
        text = "Contact the author at alice@example.com for details."
        result = self.redactor.redact_text(text)
        self.assertIn("Contact the author at", result)
        self.assertIn("for details.", result)


class TestRedactAwsKeys(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.redactor = SecurityRedactor()

    def test_aws_key_pattern_is_redacted(self):
        text = f"Set env var AWS_ACCESS_KEY_ID={FAKE_AWS_KEY} in your shell."
        result = self.redactor.redact_text(text)
        self.assertNotIn(FAKE_AWS_KEY, result)
        self.assertIn("[REDACTED_AWS_KEY]", result)

    def test_aws_key_preceded_by_non_uppercase_is_redacted(self):
        # The pattern uses a negative lookbehind for [A-Z0-9] so a key
        # immediately following a lowercase character must still be caught.
        text = f"key={FAKE_AWS_KEY}"
        result = self.redactor.redact_text(text)
        self.assertNotIn(FAKE_AWS_KEY, result)

    def test_shorter_uppercase_string_is_not_redacted(self):
        # Only exactly-20-char all-caps sequences trigger the AWS pattern.
        short_key = "AKIAIOSFODNN7EXAM"  # 17 chars — too short.
        text = f"This is a normal word: {short_key}"
        result = self.redactor.redact_text(text)
        self.assertNotIn("[REDACTED_AWS_KEY]", result)


if __name__ == "__main__":
    unittest.main()
