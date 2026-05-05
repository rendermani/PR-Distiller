"""Tests that store_rule produces unique ChromaDB ids even when the LLM emits
the same kebab-case slug for two different rules in the same repo.

Regression test for the rule_id collision bug introduced when the extractor
prompt switched from UUIDs to descriptive slugs.
"""
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _make_db():
    from db.lightrag_manager import LightRAGManager
    db = LightRAGManager.__new__(LightRAGManager)
    db.collection = MagicMock()
    return db


def _rule(slug, repo, title, description, enforcement):
    return {
        "rule_id": slug,
        "metadata": {
            "status": "active",
            "repo": repo,
            "occurrence_count": 1,
            "confidence": 0.9,
        },
        "content": {
            "title": title,
            "description": description,
            "enforcement_prompt": enforcement,
        },
    }


def _stored_id(db) -> str:
    return db.collection.add.call_args.kwargs["ids"][0]


class TestRuleIdCollision(unittest.TestCase):
    def test_two_rules_with_same_slug_get_distinct_chroma_ids(self):
        """Same slug + same repo + different content must yield different ChromaDB ids."""
        db = _make_db()
        db.store_rule(_rule(
            "use-parameterized-sql-queries",
            "acme/app",
            "Use parameterized SQL queries",
            "Concatenating user input is a SQLi risk",
            "Always use placeholders",
        ))
        first_id = _stored_id(db)

        db.collection.add.reset_mock()
        db.store_rule(_rule(
            "use-parameterized-sql-queries",
            "acme/app",
            "Use parameterized SQL queries",
            "Different context, different rule",
            "Use the ORM's bind parameters",
        ))
        second_id = _stored_id(db)

        self.assertNotEqual(first_id, second_id)

    def test_chroma_id_is_deterministic_for_same_content(self):
        """Identical content + slug + repo should produce the same ChromaDB id (idempotent)."""
        db = _make_db()
        rule = _rule(
            "use-context-managers",
            "acme/app",
            "Use context managers for files",
            "Avoid leaking file descriptors",
            "Use `with open(...) as f:`",
        )
        db.store_rule(rule)
        first_id = _stored_id(db)

        db.collection.add.reset_mock()
        db.store_rule(rule)
        second_id = _stored_id(db)

        self.assertEqual(first_id, second_id)

    def test_chroma_id_includes_repo_namespace(self):
        """Same slug in different repos must yield different ids."""
        db = _make_db()
        db.store_rule(_rule(
            "use-parameterized-sql-queries", "acme/app",
            "T", "D", "E",
        ))
        first_id = _stored_id(db)

        db.collection.add.reset_mock()
        db.store_rule(_rule(
            "use-parameterized-sql-queries", "other-org/other-app",
            "T", "D", "E",
        ))
        second_id = _stored_id(db)

        self.assertNotEqual(first_id, second_id)


if __name__ == "__main__":
    unittest.main()
