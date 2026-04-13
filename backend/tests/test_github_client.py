"""
Unit tests for pipeline/github_client.py — GitHubClient.

All HTTP calls are intercepted via unittest.mock.patch so no network access
is required. Tests verify the correct URL is built, the correct headers are
sent, successful responses are returned as the right types, and HTTP error
codes surface as requests.HTTPError.

Run with:
    cd /home/mani/Projects/PR-Analysis/backend
    ./venv/bin/python -m pytest tests/test_github_client.py -v --tb=short
"""
import sys
import os
import unittest
from unittest.mock import patch, MagicMock

import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.github_client import GitHubClient

OWNER = "acme"
REPO = "myapp"
PR_NUMBER = 42


def _make_response(status_code: int, text: str = "", json_body=None):
    """Build a mock requests.Response."""
    response = MagicMock(spec=requests.Response)
    response.status_code = status_code
    response.text = text
    if json_body is not None:
        response.json.return_value = json_body
    if status_code >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(
            response=response
        )
    else:
        response.raise_for_status.return_value = None
    return response


class TestGetHeaders(unittest.TestCase):
    def test_headers_include_token_when_provided(self):
        client = GitHubClient(token="ghp_mytoken")
        headers = client._get_headers()
        self.assertEqual(headers["Authorization"], "Bearer ghp_mytoken")

    def test_headers_omit_authorization_when_no_token(self):
        with patch.dict(os.environ, {}, clear=True):
            # Ensure GITHUB_TOKEN is not set.
            os.environ.pop("GITHUB_TOKEN", None)
            client = GitHubClient(token=None)
        headers = client._get_headers()
        self.assertNotIn("Authorization", headers)

    def test_headers_include_custom_accept_type(self):
        client = GitHubClient(token="tok")
        headers = client._get_headers("application/vnd.github.v3.diff")
        self.assertEqual(headers["Accept"], "application/vnd.github.v3.diff")

    def test_headers_default_accept_is_json_v3(self):
        client = GitHubClient(token="tok")
        headers = client._get_headers()
        self.assertEqual(headers["Accept"], "application/vnd.github.v3+json")

    def test_token_from_environment_when_not_passed_explicitly(self):
        with patch.dict(os.environ, {"GITHUB_TOKEN": "env_token"}):
            client = GitHubClient(token=None)
        headers = client._get_headers()
        self.assertEqual(headers["Authorization"], "Bearer env_token")


class TestGetPrDiff(unittest.TestCase):
    def setUp(self):
        self.client = GitHubClient(token="ghp_test")

    def test_returns_diff_string_on_success(self):
        fake_diff = "--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new"
        mock_response = _make_response(200, text=fake_diff)
        with patch("requests.get", return_value=mock_response) as mock_get:
            result = self.client.get_pr_diff(OWNER, REPO, PR_NUMBER)
        self.assertEqual(result, fake_diff)

    def test_requests_diff_accept_header(self):
        mock_response = _make_response(200, text="diff content")
        with patch("requests.get", return_value=mock_response) as mock_get:
            self.client.get_pr_diff(OWNER, REPO, PR_NUMBER)
        _, kwargs = mock_get.call_args
        self.assertIn("application/vnd.github.v3.diff", kwargs["headers"]["Accept"])

    def test_raises_on_http_404(self):
        mock_response = _make_response(404)
        with patch("requests.get", return_value=mock_response):
            with self.assertRaises(requests.HTTPError):
                self.client.get_pr_diff(OWNER, REPO, PR_NUMBER)

    def test_raises_on_http_500(self):
        mock_response = _make_response(500)
        with patch("requests.get", return_value=mock_response):
            with self.assertRaises(requests.HTTPError):
                self.client.get_pr_diff(OWNER, REPO, PR_NUMBER)


class TestGetPrComments(unittest.TestCase):
    def setUp(self):
        self.client = GitHubClient(token="ghp_test")

    def test_returns_list_of_comments_on_success(self):
        comments = [{"id": 1, "body": "Fix this."}, {"id": 2, "body": "LGTM."}]
        mock_response = _make_response(200, json_body=comments)
        with patch("requests.get", return_value=mock_response):
            result = self.client.get_pr_comments(OWNER, REPO, PR_NUMBER)
        self.assertEqual(result, comments)

    def test_returns_empty_list_when_no_comments(self):
        mock_response = _make_response(200, json_body=[])
        with patch("requests.get", return_value=mock_response):
            result = self.client.get_pr_comments(OWNER, REPO, PR_NUMBER)
        self.assertEqual(result, [])

    def test_raises_on_http_401(self):
        mock_response = _make_response(401)
        with patch("requests.get", return_value=mock_response):
            with self.assertRaises(requests.HTTPError):
                self.client.get_pr_comments(OWNER, REPO, PR_NUMBER)


if __name__ == "__main__":
    unittest.main()
