"""
Tests for GitHub issue #8: Upgrade embedding model from all-MiniLM-L6-v2.

Covers:
- LightRAGManager selects SentenceTransformerEmbeddingFunction when EMBEDDING_MODEL is set
- LightRAGManager falls back to DefaultEmbeddingFunction when EMBEDDING_MODEL is unset
  (backward-compat, explicitly approved by user per issue spec)
- ConfigManager stores and retrieves embedding_model field
- ConfigManager default config includes embedding_model
- reindex_all_rules() re-embeds all documents in enterprise_rejections using the
  current embedding function
- POST /api/admin/reindex endpoint calls reindex_all_rules() and returns success status
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_EMBEDDING_ENV_VAR = "EMBEDDING_MODEL"
_DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
_FALLBACK_MODEL_SENTINEL = "DefaultEmbeddingFunction"


# ---------------------------------------------------------------------------
# LightRAGManager — embedding function selection
# ---------------------------------------------------------------------------

class TestEmbeddingFunctionSelection(unittest.TestCase):
    """LightRAGManager.__init__ must choose the embedding function from EMBEDDING_MODEL."""

    def _make_manager_with_env(self, env_vars: dict):
        """Instantiate LightRAGManager with mocked ChromaDB and a controlled env."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch.dict(os.environ, env_vars, clear=False), \
             patch("chromadb.PersistentClient", return_value=mock_client), \
             patch("chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction") as mock_st, \
             patch("chromadb.utils.embedding_functions.DefaultEmbeddingFunction") as mock_default:

            from importlib import reload
            import db.lightrag_manager as lrm
            reload(lrm)

            manager = lrm.LightRAGManager()
            return manager, mock_st, mock_default

    def test_uses_sentence_transformer_when_embedding_model_env_set(self):
        """When EMBEDDING_MODEL is set, SentenceTransformerEmbeddingFunction is used."""
        manager, mock_st, mock_default = self._make_manager_with_env(
            {_EMBEDDING_ENV_VAR: _DEFAULT_MODEL}
        )
        mock_st.assert_called_once_with(model_name=_DEFAULT_MODEL)
        mock_default.assert_not_called()

    def test_uses_default_when_embedding_model_env_absent(self):
        """When EMBEDDING_MODEL is absent, DefaultEmbeddingFunction is used (backward compat)."""
        env_without_model = {k: v for k, v in os.environ.items() if k != _EMBEDDING_ENV_VAR}
        with patch.dict(os.environ, {}, clear=True):
            # Remove EMBEDDING_MODEL entirely
            os.environ.pop(_EMBEDDING_ENV_VAR, None)
            manager, mock_st, mock_default = self._make_manager_with_env({})

        mock_default.assert_called_once()
        mock_st.assert_not_called()

    def test_embed_fn_attribute_is_set_on_manager(self):
        """manager.embed_fn must be the resolved embedding function instance."""
        mock_embed_instance = MagicMock()
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = MagicMock()

        with patch.dict(os.environ, {_EMBEDDING_ENV_VAR: _DEFAULT_MODEL}, clear=False), \
             patch("chromadb.PersistentClient", return_value=mock_client), \
             patch(
                 "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction",
                 return_value=mock_embed_instance,
             ):
            from importlib import reload
            import db.lightrag_manager as lrm
            reload(lrm)
            manager = lrm.LightRAGManager()

        self.assertIs(manager.embed_fn, mock_embed_instance)

    def test_custom_model_name_is_passed_through(self):
        """Any model name set in EMBEDDING_MODEL is forwarded to SentenceTransformer."""
        custom_model = "sentence-transformers/all-mpnet-base-v2"
        manager, mock_st, _ = self._make_manager_with_env(
            {_EMBEDDING_ENV_VAR: custom_model}
        )
        mock_st.assert_called_once_with(model_name=custom_model)


# ---------------------------------------------------------------------------
# LightRAGManager — reindex_all_rules
# ---------------------------------------------------------------------------

class TestReindexAllRules(unittest.TestCase):
    """reindex_all_rules() must re-add all existing documents with the current embed_fn."""

    def _make_db_with_documents(self, documents: list[str], ids: list[str], metadatas: list[dict]):
        """Return a LightRAGManager with a mocked collection pre-populated with data."""
        from db.lightrag_manager import LightRAGManager

        db = LightRAGManager.__new__(LightRAGManager)
        db.embed_fn = MagicMock()
        db.collection = MagicMock()
        db.history_collection = MagicMock()

        db.collection.get.return_value = {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
        }
        return db

    def test_reindex_deletes_and_readds_each_document(self):
        """reindex_all_rules must delete then re-add every rule to apply the new embed_fn."""
        from db.lightrag_manager import LightRAGManager

        documents = ["Rule: No bare except", "Rule: Use type hints"]
        ids = ["id1", "id2"]
        metadatas = [
            {"rule_id": "id1", "repo": "owner/repo", "status": "active", "occurrence_count": 1, "path_patterns": ""},
            {"rule_id": "id2", "repo": "owner/repo", "status": "active", "occurrence_count": 2, "path_patterns": ""},
        ]

        db = self._make_db_with_documents(documents, ids, metadatas)
        result = LightRAGManager.reindex_all_rules(db)

        # All IDs deleted first
        db.collection.delete.assert_called_once_with(ids=ids)

        # Re-added in a single batch
        db.collection.add.assert_called_once()
        add_kwargs = db.collection.add.call_args.kwargs
        self.assertEqual(add_kwargs["ids"], ids)
        self.assertEqual(add_kwargs["documents"], documents)
        self.assertEqual(add_kwargs["metadatas"], metadatas)

    def test_reindex_returns_count_of_reindexed_rules(self):
        """reindex_all_rules must return the number of rules re-embedded."""
        from db.lightrag_manager import LightRAGManager

        documents = ["doc1", "doc2", "doc3"]
        ids = ["a", "b", "c"]
        metadatas = [{"rule_id": x, "repo": "r", "status": "active", "occurrence_count": 1, "path_patterns": ""} for x in ids]

        db = self._make_db_with_documents(documents, ids, metadatas)
        count = LightRAGManager.reindex_all_rules(db)

        self.assertEqual(count, 3)

    def test_reindex_on_empty_collection_returns_zero(self):
        """reindex_all_rules on an empty collection must return 0 and not call delete/add."""
        from db.lightrag_manager import LightRAGManager

        db = self._make_db_with_documents([], [], [])
        count = LightRAGManager.reindex_all_rules(db)

        self.assertEqual(count, 0)
        db.collection.delete.assert_not_called()
        db.collection.add.assert_not_called()

    def test_reindex_preserves_metadata_integrity(self):
        """Metadata must be passed back unchanged — no fields dropped or added."""
        from db.lightrag_manager import LightRAGManager

        original_meta = {
            "rule_id": "xyz",
            "repo": "owner/repo",
            "status": "active",
            "occurrence_count": 5,
            "path_patterns": "**/auth/*.py",
            "confidence": 0.9,
            "category": "security",
        }
        db = self._make_db_with_documents(["doc"], ["xyz"], [original_meta])
        LightRAGManager.reindex_all_rules(db)

        add_kwargs = db.collection.add.call_args.kwargs
        stored_meta = add_kwargs["metadatas"][0]
        self.assertEqual(stored_meta, original_meta)


# ---------------------------------------------------------------------------
# ConfigManager — embedding_model field
# ---------------------------------------------------------------------------

class TestConfigManagerEmbeddingModel(unittest.TestCase):
    """ConfigManager must persist and retrieve the embedding_model setting."""

    def _make_config_manager(self, tmp_dir: str):
        """Return a ConfigManager that reads/writes to a temporary directory."""
        import tempfile
        from pipeline.config_manager import ConfigManager

        manager = ConfigManager.__new__(ConfigManager)
        manager.base_dir = tmp_dir
        manager.config_path = os.path.join(tmp_dir, "config.json")
        manager.key_path = os.path.join(tmp_dir, ".secret_key")
        from cryptography.fernet import Fernet
        manager.cipher = Fernet(manager._resolve_key())
        manager._ensure_default_config()
        return manager

    def setUp(self):
        import tempfile
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_default_config_includes_embedding_model(self):
        """A freshly initialized config must include the embedding_model key."""
        manager = self._make_config_manager(self.tmp_dir)
        config = manager.load_config()
        self.assertIn("embedding_model", config)

    def test_default_embedding_model_is_bge_base(self):
        """The default embedding model must be BAAI/bge-base-en-v1.5."""
        manager = self._make_config_manager(self.tmp_dir)
        config = manager.load_config()
        self.assertEqual(config["embedding_model"], "BAAI/bge-base-en-v1.5")

    def test_save_and_load_custom_embedding_model(self):
        """Saving a custom embedding_model persists and is retrievable."""
        manager = self._make_config_manager(self.tmp_dir)
        manager.save_config({"embedding_model": "sentence-transformers/all-mpnet-base-v2"})
        config = manager.load_config()
        self.assertEqual(config["embedding_model"], "sentence-transformers/all-mpnet-base-v2")

    def test_save_config_does_not_lose_embedding_model_on_unrelated_update(self):
        """Saving only llm_model must not wipe the stored embedding_model."""
        manager = self._make_config_manager(self.tmp_dir)
        manager.save_config({"embedding_model": "BAAI/bge-large-en-v1.5"})
        manager.save_config({"llm_model": "openai/gpt-4o"})
        config = manager.load_config()
        self.assertEqual(config["embedding_model"], "BAAI/bge-large-en-v1.5")


# ---------------------------------------------------------------------------
# API — POST /api/admin/reindex endpoint
# ---------------------------------------------------------------------------

class TestReindexEndpoint(unittest.TestCase):
    """POST /api/admin/reindex must call db.reindex_all_rules() and return a count."""

    def _make_api_client(self, mock_db_reindex_return: int):
        mock_db = MagicMock()
        mock_db.reindex_all_rules.return_value = mock_db_reindex_return
        mock_conf = MagicMock()
        mock_orch = MagicMock()

        with patch("db.lightrag_manager.LightRAGManager", return_value=mock_db), \
             patch("pipeline.config_manager.ConfigManager", return_value=mock_conf), \
             patch("pipeline.job_orchestrator.JobOrchestrator", return_value=mock_orch):
            from importlib import reload
            import api as api_module
            reload(api_module)

        from fastapi.testclient import TestClient
        with patch.object(api_module, "db", mock_db):
            client = TestClient(api_module.app)
            return client, mock_db, api_module

    def test_reindex_endpoint_returns_200(self):
        """POST /api/admin/reindex must return HTTP 200."""
        client, _, _ = self._make_api_client(mock_db_reindex_return=5)
        response = client.post("/api/admin/reindex")
        self.assertEqual(response.status_code, 200)

    def test_reindex_endpoint_returns_reindexed_count(self):
        """Response body must include the count of re-embedded rules."""
        client, _, _ = self._make_api_client(mock_db_reindex_return=12)
        response = client.post("/api/admin/reindex")
        data = response.json()
        self.assertEqual(data["reindexed_count"], 12)

    def test_reindex_endpoint_calls_reindex_all_rules(self):
        """The endpoint must delegate to db.reindex_all_rules()."""
        client, mock_db, api_module = self._make_api_client(mock_db_reindex_return=3)
        with patch.object(api_module, "db", mock_db):
            client.post("/api/admin/reindex")
        mock_db.reindex_all_rules.assert_called_once()

    def test_reindex_endpoint_returns_status_success(self):
        """Response body must include status: 'success'."""
        client, _, _ = self._make_api_client(mock_db_reindex_return=0)
        response = client.post("/api/admin/reindex")
        self.assertEqual(response.json()["status"], "success")


if __name__ == "__main__":
    unittest.main()
