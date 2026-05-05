"""Tests for backend.auth.ensure_api_token resolution order and persistence."""
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestEnsureApiToken(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_env_var_wins_when_set(self):
        from auth import ensure_api_token
        with patch("settings.API_AUTH_TOKEN", "operator-supplied"), \
             patch("settings.DATA_DIR", self.tmp_dir):
            self.assertEqual(ensure_api_token(), "operator-supplied")
        # Did not write a token file.
        self.assertFalse(os.path.exists(os.path.join(self.tmp_dir, ".api_token")))

    def test_reads_existing_token_file_when_env_unset(self):
        from auth import ensure_api_token
        path = os.path.join(self.tmp_dir, ".api_token")
        with open(path, "w") as f:
            f.write("persisted-token-abc\n")
        with patch("settings.API_AUTH_TOKEN", ""), patch("settings.DATA_DIR", self.tmp_dir):
            self.assertEqual(ensure_api_token(), "persisted-token-abc")

    def test_generates_and_persists_when_neither_env_nor_file(self):
        from auth import ensure_api_token
        with patch("settings.API_AUTH_TOKEN", ""), patch("settings.DATA_DIR", self.tmp_dir):
            t = ensure_api_token()
        self.assertEqual(len(t), 64)  # 32 bytes hex
        self.assertTrue(t.isalnum())
        path = os.path.join(self.tmp_dir, ".api_token")
        self.assertTrue(os.path.exists(path))
        with open(path) as f:
            self.assertEqual(f.read().strip(), t)
        # File must be mode 0600 so other users on the host cannot read it.
        mode = os.stat(path).st_mode & 0o777
        self.assertEqual(mode, 0o600)

    def test_subsequent_calls_return_same_token(self):
        from auth import ensure_api_token
        with patch("settings.API_AUTH_TOKEN", ""), patch("settings.DATA_DIR", self.tmp_dir):
            first = ensure_api_token()
            second = ensure_api_token()
        self.assertEqual(first, second)

    def test_token_file_disappears_after_check_falls_through_to_generation(self):
        """If the file vanishes between the existence check and the read,
        ensure_api_token must produce a valid token rather than crashing
        (i.e. EAFP, not LBYL)."""
        from auth import ensure_api_token

        # Lie: claim the file exists so the LBYL path is taken; then the
        # subsequent read will hit FileNotFoundError because the file is
        # actually absent.
        with patch("settings.API_AUTH_TOKEN", ""), \
             patch("settings.DATA_DIR", self.tmp_dir), \
             patch("os.path.exists", return_value=True):
            token = ensure_api_token()

        self.assertEqual(len(token), 64)
        # The generated token must have been persisted.
        with open(os.path.join(self.tmp_dir, ".api_token")) as f:
            self.assertEqual(f.read().strip(), token)


if __name__ == "__main__":
    unittest.main()
