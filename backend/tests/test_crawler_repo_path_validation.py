"""A repo entry without owner/ must fail loudly, not vanish.

All three crawlers did `if "/" in repo_string: ... else: continue`, silently
skipping any entry lacking a slash. A repo registered as "tdp-components"
instead of "tucowsinc/tdp-components" therefore produced a job that finished in
seconds with 0 comments crawled, 0 sent to the LLM, and no error anywhere —
indistinguishable from "this repo genuinely has no review comments".
"""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.deep_crawler import (
    InvalidRepoPathError,
    _split_repo_path,
    crawl_closed_issues,
    crawl_human_rejections,
    crawl_pr_reviews,
)


class TestSplitRepoPath(unittest.TestCase):
    def test_valid_path_splits(self):
        self.assertEqual(_split_repo_path("tucowsinc/tdp-components"),
                         ("tucowsinc", "tdp-components"))

    def test_bare_name_raises(self):
        with self.assertRaises(InvalidRepoPathError) as ctx:
            _split_repo_path("tdp-components")
        message = str(ctx.exception)
        self.assertIn("tdp-components", message)
        self.assertIn("owner/repo", message)

    def test_too_many_segments_raises(self):
        with self.assertRaises(InvalidRepoPathError):
            _split_repo_path("github.com/owner/repo")

    def test_empty_owner_raises(self):
        with self.assertRaises(InvalidRepoPathError):
            _split_repo_path("/repo")

    def test_empty_repo_raises(self):
        with self.assertRaises(InvalidRepoPathError):
            _split_repo_path("owner/")

    def test_whitespace_is_rejected_not_trimmed_into_validity(self):
        """A padded path is still a configuration error worth surfacing."""
        self.assertEqual(_split_repo_path("  owner/repo  "), ("owner", "repo"))


class TestCrawlersRejectBareRepoNames(unittest.TestCase):
    """Each crawler must raise rather than return an empty dataset."""

    @patch("scripts.deep_crawler.GitHubClient")
    def test_human_rejections_raises(self, _client):
        with self.assertRaises(InvalidRepoPathError):
            crawl_human_rejections(["tdp-components"])

    @patch("scripts.deep_crawler.GitHubClient")
    def test_pr_reviews_raises(self, _client):
        with self.assertRaises(InvalidRepoPathError):
            crawl_pr_reviews(["tdp-components"])

    @patch("scripts.deep_crawler.GitHubClient")
    def test_closed_issues_raises(self, _client):
        with self.assertRaises(InvalidRepoPathError):
            crawl_closed_issues(["tdp-components"])

    @patch("scripts.deep_crawler.GitHubClient")
    def test_error_names_the_offending_repo(self, _client):
        with self.assertRaises(InvalidRepoPathError) as ctx:
            crawl_human_rejections(["good/one", "bare-name"])
        self.assertIn("bare-name", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
