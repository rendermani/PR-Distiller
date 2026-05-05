"""Tests for LightRAGManager.delete_repo_rules."""
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


class TestDeleteRepoRules(unittest.TestCase):
    def test_returns_zero_and_skips_delete_when_repo_has_no_rules(self):
        db = _make_db()
        db.collection.get.return_value = {"ids": []}
        removed = db.delete_repo_rules("acme/empty")
        self.assertEqual(removed, 0)
        db.collection.delete.assert_not_called()

    def test_returns_count_and_deletes_all_ids_for_repo(self):
        db = _make_db()
        db.collection.get.return_value = {"ids": ["r1", "r2", "r3"]}
        removed = db.delete_repo_rules("acme/app")
        self.assertEqual(removed, 3)
        db.collection.delete.assert_called_once_with(ids=["r1", "r2", "r3"])

    def test_filters_by_repo_in_chroma_get(self):
        db = _make_db()
        db.collection.get.return_value = {"ids": ["r1"]}
        db.delete_repo_rules("acme/app")
        db.collection.get.assert_called_once_with(where={"repo": "acme/app"})


if __name__ == "__main__":
    unittest.main()
