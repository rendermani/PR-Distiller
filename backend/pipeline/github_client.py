import os
import requests
from typing import Dict, Any, List

class GitHubClient:
    """
    Client to interact with GitHub API for retrieving PR threads, 
    comments, and Diffs for rule extraction.
    """
    def __init__(self, token: str = None):
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.base_url = "https://api.github.com"
        
    def _get_headers(self, accept_type: str = "application/vnd.github.v3+json") -> Dict[str, str]:
        headers = {"Accept": accept_type}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers
        
    def get_pr_diff(self, owner: str, repo: str, pr_number: int) -> str:
        """Fetches the raw diff of a Pull Request."""
        url = f"{self.base_url}/repos/{owner}/{repo}/pulls/{pr_number}"
        # Requesting diff format explicitly
        response = requests.get(url, headers=self._get_headers("application/vnd.github.v3.diff"))
        response.raise_for_status()
        return response.text
        
    def get_pr_comments(self, owner: str, repo: str, pr_number: int) -> List[Dict[str, Any]]:
        """Fetches review comments on a Pull Request."""
        url = f"{self.base_url}/repos/{owner}/{repo}/pulls/{pr_number}/comments"
        response = requests.get(url, headers=self._get_headers())
        response.raise_for_status()
        return response.json()
        
    def get_pr_status(self, owner: str, repo: str, pr_number: int) -> Dict[str, Any]:
        """Checks the state of the PR (merged, open, closed)."""
        url = f"{self.base_url}/repos/{owner}/{repo}/pulls/{pr_number}"
        response = requests.get(url, headers=self._get_headers())
        response.raise_for_status()
        return response.json()
