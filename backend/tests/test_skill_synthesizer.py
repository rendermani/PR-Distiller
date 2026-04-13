"""
Unit tests for SkillSynthesizer.generate_cursorrules, generate_claude_md,
and export() dispatch for those two new format types.

Run with:
  cd /home/mani/Projects/PR-Analysis/backend
  ./venv/bin/python -m pytest pipeline/test_skill_synthesizer.py -v
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from pipeline.skill_synthesizer import SkillSynthesizer

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

REPO = "acme/myapp"

FULL_RULE = (
    "Rule: No direct DB calls in handlers"
    " - Context: Handlers must delegate to service layer. Enforce: Use service objects only."
)
BARE_RULE = "Always write unit tests."


# ---------------------------------------------------------------------------
# generate_cursorrules
# ---------------------------------------------------------------------------

class TestGenerateCursorRules:
    def test_returns_string(self):
        result = SkillSynthesizer.generate_cursorrules([FULL_RULE], REPO)
        assert isinstance(result, str)

    def test_contains_repo_header(self):
        result = SkillSynthesizer.generate_cursorrules([FULL_RULE], REPO)
        assert REPO in result

    def test_full_rule_title_appears(self):
        result = SkillSynthesizer.generate_cursorrules([FULL_RULE], REPO)
        assert "No direct DB calls in handlers" in result

    def test_enforcement_text_appears(self):
        result = SkillSynthesizer.generate_cursorrules([FULL_RULE], REPO)
        assert "Use service objects only" in result

    def test_bare_rule_appears_verbatim(self):
        result = SkillSynthesizer.generate_cursorrules([BARE_RULE], REPO)
        assert BARE_RULE in result

    def test_multiple_rules_all_present(self):
        rules = [FULL_RULE, BARE_RULE]
        result = SkillSynthesizer.generate_cursorrules(rules, REPO)
        assert "No direct DB calls in handlers" in result
        assert BARE_RULE in result

    def test_each_rule_is_on_separate_line(self):
        rules = [FULL_RULE, BARE_RULE]
        result = SkillSynthesizer.generate_cursorrules(rules, REPO)
        lines = result.splitlines()
        # At least two non-header lines must exist
        non_header_lines = [l for l in lines if l.strip() and not l.startswith("#")]
        assert len(non_header_lines) >= 2

    def test_empty_rules_returns_empty_section(self):
        """Calling with zero rules should still return a structurally valid string."""
        result = SkillSynthesizer.generate_cursorrules([], REPO)
        assert isinstance(result, str)
        assert REPO in result


# ---------------------------------------------------------------------------
# generate_claude_md
# ---------------------------------------------------------------------------

class TestGenerateClaudeMd:
    def test_returns_string(self):
        result = SkillSynthesizer.generate_claude_md([FULL_RULE], REPO)
        assert isinstance(result, str)

    def test_starts_with_h1_heading(self):
        result = SkillSynthesizer.generate_claude_md([FULL_RULE], REPO)
        assert result.startswith("#")

    def test_contains_repo_in_heading(self):
        result = SkillSynthesizer.generate_claude_md([FULL_RULE], REPO)
        assert REPO in result

    def test_rule_title_is_h2_section(self):
        result = SkillSynthesizer.generate_claude_md([FULL_RULE], REPO)
        assert "## " in result

    def test_context_text_appears(self):
        result = SkillSynthesizer.generate_claude_md([FULL_RULE], REPO)
        assert "Handlers must delegate to service layer" in result

    def test_enforcement_text_appears(self):
        result = SkillSynthesizer.generate_claude_md([FULL_RULE], REPO)
        assert "Use service objects only" in result

    def test_bare_rule_appears(self):
        result = SkillSynthesizer.generate_claude_md([BARE_RULE], REPO)
        assert BARE_RULE in result

    def test_multiple_rules_produce_multiple_sections(self):
        rules = [FULL_RULE, BARE_RULE]
        result = SkillSynthesizer.generate_claude_md(rules, REPO)
        section_count = result.count("## ")
        assert section_count >= 2

    def test_empty_rules_returns_valid_document(self):
        result = SkillSynthesizer.generate_claude_md([], REPO)
        assert isinstance(result, str)
        assert REPO in result


# ---------------------------------------------------------------------------
# export() dispatch
# ---------------------------------------------------------------------------

class TestExportDispatch:
    def test_cursorrules_format_dispatches_to_cursorrules(self):
        result = SkillSynthesizer.export([FULL_RULE], REPO, "cursorrules")
        # Should not be the "unknown format" fallback
        assert "Unknown format" not in result
        assert REPO in result

    def test_claude_md_format_dispatches_to_claude_md(self):
        result = SkillSynthesizer.export([FULL_RULE], REPO, "claude_md")
        assert "Unknown format" not in result
        assert REPO in result

    def test_export_with_empty_rules_returns_no_rules_message(self):
        result = SkillSynthesizer.export([], REPO, "cursorrules")
        assert "No active rules" in result

    def test_export_unknown_format_returns_error_message(self):
        result = SkillSynthesizer.export([FULL_RULE], REPO, "nonexistent_format")
        assert "Unknown format" in result
