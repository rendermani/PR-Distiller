"""
Unit tests for crawl_closed_issues() in deep_crawler.py.

All GitHub HTTP calls are intercepted via unittest.mock.patch so no
network access is required. Tests follow the red-green-refactor cycle:
each test was written before the implementation and run to confirm it
fails with ImportError / AttributeError before any production code was added.
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock, call

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.deep_crawler import crawl_closed_issues


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_issue(
    number: int,
    body: str = "This is a bug — avoid this pattern",
    labels: list = None,
    state: str = "closed",
) -> dict:
    """Build a minimal closed-issue dict matching the GitHub API shape."""
    return {
        "number": number,
        "state": state,
        "body": body,
        "labels": [{"name": l} for l in (labels or ["bug"])],
        "title": f"Issue #{number}",
    }


def _make_comment(body: str, login: str = "alice") -> dict:
    """Build a minimal issue-comment dict."""
    return {
        "id": 9000 + hash(body) % 1000,
        "body": body,
        "user": {"login": login},
    }


def _ok_response(json_payload) -> MagicMock:
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = json_payload
    r.headers = {}
    return r


def _empty_response() -> MagicMock:
    return _ok_response([])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCrawlClosedIssuesReturnShape(unittest.TestCase):
    """crawl_closed_issues always returns (list, dict)."""

    @patch("scripts.deep_crawler.time.sleep")
    @patch("scripts.deep_crawler.requests.get")
    def test_returns_dataset_and_cursors_tuple(self, mock_get, _sleep):
        mock_get.return_value = _empty_response()

        result = crawl_closed_issues(["owner/repo"])

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        dataset, cursors = result
        self.assertIsInstance(dataset, list)
        self.assertIsInstance(cursors, dict)

    @patch("scripts.deep_crawler.time.sleep")
    @patch("scripts.deep_crawler.requests.get")
    def test_empty_repo_list_returns_empty_dataset(self, mock_get, _sleep):
        dataset, cursors = crawl_closed_issues([])

        self.assertEqual(dataset, [])
        mock_get.assert_not_called()

    @patch("scripts.deep_crawler.time.sleep")
    @patch("scripts.deep_crawler.requests.get")
    def test_dataset_entries_are_two_tuples(self, mock_get, _sleep):
        long_comment = "avoid this approach because it causes a race condition in production"
        issue = _make_issue(1, labels=["bug"])
        comment = _make_comment(long_comment)

        # Actual call order: issues p1 → comments for issue #1 → issues p2 (terminates)
        mock_get.side_effect = [
            _ok_response([issue]),    # issues page 1
            _ok_response([comment]),  # comments for issue #1
            _empty_response(),        # issues page 2 (terminates loop)
        ]

        dataset, _ = crawl_closed_issues(["owner/repo"])

        self.assertTrue(len(dataset) >= 1)
        for entry in dataset:
            self.assertIsInstance(entry, tuple)
            self.assertEqual(len(entry), 2)


class TestCrawlClosedIssuesLabelFiltering(unittest.TestCase):
    """Only issues with qualifying labels are processed."""

    @patch("scripts.deep_crawler.time.sleep")
    @patch("scripts.deep_crawler.requests.get")
    def test_issue_with_bug_label_is_included(self, mock_get, _sleep):
        long_comment = "avoid this approach — it causes a security risk in production code"
        issue = _make_issue(10, labels=["bug"])
        comment = _make_comment(long_comment)

        mock_get.side_effect = [
            _ok_response([issue]),
            _ok_response([comment]),
            _empty_response(),
        ]

        dataset, _ = crawl_closed_issues(["owner/repo"])

        self.assertTrue(len(dataset) >= 1)

    @patch("scripts.deep_crawler.time.sleep")
    @patch("scripts.deep_crawler.requests.get")
    def test_issue_with_wontfix_label_is_included(self, mock_get, _sleep):
        long_comment = "avoid this approach — it causes a security risk in production code"
        issue = _make_issue(11, labels=["wontfix"])
        comment = _make_comment(long_comment)

        mock_get.side_effect = [
            _ok_response([issue]),
            _ok_response([comment]),
            _empty_response(),
        ]

        dataset, _ = crawl_closed_issues(["owner/repo"])

        self.assertTrue(len(dataset) >= 1)

    @patch("scripts.deep_crawler.time.sleep")
    @patch("scripts.deep_crawler.requests.get")
    def test_issue_with_unrelated_label_is_excluded(self, mock_get, _sleep):
        # Body must not contain any keyword — otherwise the issue qualifies by keyword path.
        neutral_body = "Updated the README with new usage examples."
        issue = _make_issue(12, body=neutral_body, labels=["documentation"])
        # Comments endpoint should never be called for this issue.
        # Call order: issues p1 (issue skipped) → issues p2 (empty, terminates)
        mock_get.side_effect = [
            _ok_response([issue]),
            _empty_response(),
        ]

        dataset, _ = crawl_closed_issues(["owner/repo"])

        self.assertEqual(dataset, [])
        for c in mock_get.call_args_list:
            self.assertNotIn("/comments", str(c))

    @patch("scripts.deep_crawler.time.sleep")
    @patch("scripts.deep_crawler.requests.get")
    def test_issue_with_no_labels_but_keyword_in_body_is_included(self, mock_get, _sleep):
        """Issues without qualifying labels but containing keywords still qualify."""
        keyword_body = "avoid using this approach because it's a known anti-pattern causing bugs"
        issue = _make_issue(13, body=keyword_body, labels=["enhancement"])
        long_comment = "avoid this approach — it causes a security risk in production code"
        comment = _make_comment(long_comment)

        mock_get.side_effect = [
            _ok_response([issue]),
            _ok_response([comment]),
            _empty_response(),
        ]

        dataset, _ = crawl_closed_issues(["owner/repo"])

        self.assertTrue(len(dataset) >= 1)


class TestCrawlClosedIssuesKeywordFiltering(unittest.TestCase):
    """Individual issue comments are filtered by keyword list."""

    @patch("scripts.deep_crawler.time.sleep")
    @patch("scripts.deep_crawler.requests.get")
    def test_comment_without_keyword_is_excluded(self, mock_get, _sleep):
        issue = _make_issue(20, labels=["bug"])
        comment = _make_comment("This is a great feature, thanks for the contribution!")

        mock_get.side_effect = [
            _ok_response([issue]),
            _ok_response([comment]),
            _empty_response(),
        ]

        dataset, _ = crawl_closed_issues(["owner/repo"])

        self.assertEqual(dataset, [])

    @patch("scripts.deep_crawler.time.sleep")
    @patch("scripts.deep_crawler.requests.get")
    def test_comment_too_short_is_excluded(self, mock_get, _sleep):
        issue = _make_issue(21, labels=["bug"])
        comment = _make_comment("avoid this")  # < 40 chars

        mock_get.side_effect = [
            _ok_response([issue]),
            _ok_response([comment]),
            _empty_response(),
        ]

        dataset, _ = crawl_closed_issues(["owner/repo"])

        self.assertEqual(dataset, [])

    @patch("scripts.deep_crawler.time.sleep")
    @patch("scripts.deep_crawler.requests.get")
    def test_comment_with_avoid_keyword_is_included(self, mock_get, _sleep):
        issue = _make_issue(22, labels=["bug"])
        long_comment = "avoid using global state here because it causes race conditions in async code"
        comment = _make_comment(long_comment)

        mock_get.side_effect = [
            _ok_response([issue]),
            _ok_response([comment]),
            _empty_response(),
        ]

        dataset, _ = crawl_closed_issues(["owner/repo"])

        self.assertTrue(len(dataset) >= 1)

    @patch("scripts.deep_crawler.time.sleep")
    @patch("scripts.deep_crawler.requests.get")
    def test_comment_context_is_issue_body_truncated_to_2000_chars(self, mock_get, _sleep):
        long_body = "X" * 3000
        issue = _make_issue(23, body=long_body, labels=["bug"])
        long_comment = "avoid this approach — it causes a security risk in production code and should not be used"
        comment = _make_comment(long_comment)

        mock_get.side_effect = [
            _ok_response([issue]),
            _ok_response([comment]),
            _empty_response(),
        ]

        dataset, _ = crawl_closed_issues(["owner/repo"])

        self.assertEqual(len(dataset), 1)
        _comment_text, context = dataset[0]
        self.assertEqual(len(context), 2000)


class TestCrawlClosedIssuesBotFiltering(unittest.TestCase):
    """Bot comments must be silently discarded."""

    @patch("scripts.deep_crawler.time.sleep")
    @patch("scripts.deep_crawler.requests.get")
    def test_bot_comment_is_excluded(self, mock_get, _sleep):
        issue = _make_issue(30, labels=["bug"])
        long_comment = "avoid this approach — it causes a security risk in production code"
        bot_comment = _make_comment(long_comment, login="dependabot[bot]")

        mock_get.side_effect = [
            _ok_response([issue]),
            _ok_response([bot_comment]),
            _empty_response(),
        ]

        dataset, _ = crawl_closed_issues(["owner/repo"])

        self.assertEqual(dataset, [])

    @patch("scripts.deep_crawler.time.sleep")
    @patch("scripts.deep_crawler.requests.get")
    def test_generic_bot_suffix_is_excluded(self, mock_get, _sleep):
        issue = _make_issue(31, labels=["bug"])
        long_comment = "avoid this approach — it causes a security risk in production code"
        bot_comment = _make_comment(long_comment, login="some-custom[bot]")

        mock_get.side_effect = [
            _ok_response([issue]),
            _ok_response([bot_comment]),
            _empty_response(),
        ]

        dataset, _ = crawl_closed_issues(["owner/repo"])

        self.assertEqual(dataset, [])


class TestCrawlClosedIssuesCursorBehavior(unittest.TestCase):
    """Incremental cursor: issues at or below cursor are skipped; cursor advances."""

    @patch("scripts.deep_crawler.time.sleep")
    @patch("scripts.deep_crawler.requests.get")
    def test_cursor_key_uses_repo_string_with_issues_suffix(self, mock_get, _sleep):
        mock_get.return_value = _empty_response()

        _, cursors = crawl_closed_issues(["owner/repo"])

        self.assertIn("owner/repo_issues", cursors)

    @patch("scripts.deep_crawler.time.sleep")
    @patch("scripts.deep_crawler.requests.get")
    def test_cursor_advances_to_highest_issue_number_seen(self, mock_get, _sleep):
        issue_low = _make_issue(5, labels=["bug"])
        issue_high = _make_issue(42, labels=["bug"])
        long_comment = "avoid this approach — it causes a security risk in production code"
        comment = _make_comment(long_comment)

        # Call order: issues p1 → comments #5 → comments #42 → issues p2 (terminates)
        mock_get.side_effect = [
            _ok_response([issue_low, issue_high]),
            _ok_response([comment]),  # comments for issue 5
            _ok_response([comment]),  # comments for issue 42
            _empty_response(),        # issues page 2 terminates
        ]

        _, cursors = crawl_closed_issues(["owner/repo"])

        self.assertEqual(cursors["owner/repo_issues"], 42)

    @patch("scripts.deep_crawler.time.sleep")
    @patch("scripts.deep_crawler.requests.get")
    def test_issues_at_or_below_cursor_are_skipped(self, mock_get, _sleep):
        issue = _make_issue(10, labels=["bug"])
        long_comment = "avoid this approach — it causes a security risk in production code"
        comment = _make_comment(long_comment)

        # cursor already at 10 — this issue should be skipped entirely
        mock_get.side_effect = [
            _ok_response([issue]),
            _empty_response(),
            _ok_response([comment]),
        ]

        dataset, _ = crawl_closed_issues(
            ["owner/repo"], cursors={"owner/repo_issues": 10}
        )

        self.assertEqual(dataset, [])
        # Comments endpoint must NOT have been called since issue was skipped
        for c in mock_get.call_args_list:
            self.assertNotIn("/issues/10/comments", str(c))

    @patch("scripts.deep_crawler.time.sleep")
    @patch("scripts.deep_crawler.requests.get")
    def test_existing_cursors_are_preserved_for_other_repos(self, mock_get, _sleep):
        mock_get.return_value = _empty_response()

        existing = {"other/repo_issues": 99, "owner/repo_issues": 0}
        _, cursors = crawl_closed_issues(["owner/repo"], cursors=existing)

        self.assertEqual(cursors["other/repo_issues"], 99)

    @patch("scripts.deep_crawler.time.sleep")
    @patch("scripts.deep_crawler.requests.get")
    def test_none_cursors_treated_as_empty_dict(self, mock_get, _sleep):
        mock_get.return_value = _empty_response()

        # Must not raise; must return a valid cursor dict
        dataset, cursors = crawl_closed_issues(["owner/repo"], cursors=None)

        self.assertIsInstance(cursors, dict)
        self.assertIn("owner/repo_issues", cursors)


class TestCrawlClosedIssuesRateLimiting(unittest.TestCase):
    """Rate-limit responses (403/429) trigger sleep then retry."""

    @patch("scripts.deep_crawler.time.sleep")
    @patch("scripts.deep_crawler.time.time", return_value=1000.0)
    @patch("scripts.deep_crawler.requests.get")
    def test_rate_limit_response_causes_retry(self, mock_get, _mock_time, mock_sleep):
        rate_limited = MagicMock()
        rate_limited.status_code = 429
        rate_limited.headers = {"x-ratelimit-reset": "1065"}  # 65 s away

        mock_get.side_effect = [
            rate_limited,
            _empty_response(),  # success on retry
        ]

        crawl_closed_issues(["owner/repo"])

        # sleep must have been called at least once
        mock_sleep.assert_called()

    @patch("scripts.deep_crawler.time.sleep")
    @patch("scripts.deep_crawler.requests.get")
    def test_cancellation_during_rate_limit_returns_early(self, mock_get, _sleep):
        rate_limited = MagicMock()
        rate_limited.status_code = 403
        rate_limited.headers = {"x-ratelimit-reset": str(int(9e9))}  # far future

        mock_get.return_value = rate_limited
        cancelled_after = [0]

        def is_cancelled():
            cancelled_after[0] += 1
            return cancelled_after[0] > 1

        dataset, _ = crawl_closed_issues(["owner/repo"], is_cancelled=is_cancelled)

        self.assertEqual(dataset, [])


class TestCrawlClosedIssuesCancellation(unittest.TestCase):
    """is_cancelled() checked at the top of each page loop."""

    @patch("scripts.deep_crawler.time.sleep")
    @patch("scripts.deep_crawler.requests.get")
    def test_cancellation_before_first_request_returns_empty(self, mock_get, _sleep):
        dataset, cursors = crawl_closed_issues(
            ["owner/repo"], is_cancelled=lambda: True
        )

        self.assertEqual(dataset, [])
        mock_get.assert_not_called()


class TestCrawlClosedIssuesApiErrors(unittest.TestCase):
    """Non-200, non-rate-limit errors break the page loop gracefully."""

    @patch("scripts.deep_crawler.time.sleep")
    @patch("scripts.deep_crawler.requests.get")
    def test_404_response_breaks_loop_and_returns_empty(self, mock_get, _sleep):
        not_found = MagicMock()
        not_found.status_code = 404
        not_found.headers = {}

        mock_get.return_value = not_found

        dataset, cursors = crawl_closed_issues(["owner/repo"])

        self.assertEqual(dataset, [])
        self.assertIn("owner/repo_issues", cursors)

    @patch("scripts.deep_crawler.time.sleep")
    @patch("scripts.deep_crawler.requests.get")
    def test_comments_api_error_skips_that_issue_and_continues(self, mock_get, _sleep):
        """A bad comments response for one issue should not abort the whole crawl."""
        issue_a = _make_issue(40, labels=["bug"])
        issue_b = _make_issue(41, labels=["bug"])
        long_comment = "avoid this approach — it causes a security risk in production code"
        good_comment = _make_comment(long_comment)

        comments_error = MagicMock()
        comments_error.status_code = 500
        comments_error.headers = {}

        # Call order: issues p1 → comments #40 (error) → comments #41 (ok) → issues p2 (terminates)
        mock_get.side_effect = [
            _ok_response([issue_a, issue_b]),  # issues page 1
            comments_error,                    # comments for issue 40 fail
            _ok_response([good_comment]),      # comments for issue 41 succeed
            _empty_response(),                 # issues page 2 terminates
        ]

        dataset, _ = crawl_closed_issues(["owner/repo"])

        # issue_b's comment should still be harvested
        self.assertEqual(len(dataset), 1)


class TestCrawlClosedIssuesApiUrlConstruction(unittest.TestCase):
    """Verify the issues API is called with the correct parameters."""

    @patch("scripts.deep_crawler.time.sleep")
    @patch("scripts.deep_crawler.requests.get")
    def test_issues_url_contains_state_closed(self, mock_get, _sleep):
        mock_get.return_value = _empty_response()

        crawl_closed_issues(["myorg/myrepo"])

        first_url = mock_get.call_args_list[0][0][0]
        self.assertIn("state=closed", first_url)
        self.assertIn("myorg/myrepo", first_url)

    @patch("scripts.deep_crawler.time.sleep")
    @patch("scripts.deep_crawler.requests.get")
    def test_status_callback_receives_messages(self, mock_get, _sleep):
        mock_get.return_value = _empty_response()
        messages = []

        crawl_closed_issues(["owner/repo"], status_callback=messages.append)

        self.assertTrue(len(messages) >= 1)

    @patch("scripts.deep_crawler.time.sleep")
    @patch("scripts.deep_crawler.requests.get")
    def test_repo_without_slash_is_skipped_gracefully(self, mock_get, _sleep):
        dataset, cursors = crawl_closed_issues(["invalid-repo-no-slash"])

        self.assertEqual(dataset, [])
        mock_get.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
