import os
import requests
from typing import Dict, Any, List


class GitHubAuthError(Exception):
    """Raised when GitHub rejects the configured token (401/403)."""


def _explain_github_error(response: requests.Response) -> str:
    """Turn a GitHub error response into a human-actionable message."""
    status = response.status_code
    try:
        body = response.json()
        msg = body.get("message", "")
    except Exception:
        msg = response.text[:200]
    if status == 401:
        return "GitHub returned 401 Unauthorized — the token is missing, invalid, or revoked."
    if status == 403:
        if "rate limit" in msg.lower():
            return f"GitHub rate limit hit: {msg}"
        return f"GitHub returned 403 Forbidden — the token lacks required scopes (need 'repo' for private repos). Detail: {msg}"
    if status == 404:
        return "GitHub returned 404 — repository not found or token can't see it."
    return f"GitHub API error {status}: {msg}"


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

    def _raise_for_status(self, response: requests.Response) -> None:
        if response.status_code in (401, 403):
            raise GitHubAuthError(_explain_github_error(response))
        response.raise_for_status()

    def validate_token(self) -> Dict[str, Any]:
        """Ping /user to verify the token is usable. Returns {valid, login, scopes, error}."""
        if not self.token:
            return {"valid": False, "error": "No token provided."}
        try:
            r = requests.get(f"{self.base_url}/user", headers=self._get_headers(), timeout=10)
        except requests.RequestException as e:
            return {"valid": False, "error": f"Could not reach GitHub: {e}"}
        if r.status_code == 200:
            return {
                "valid": True,
                "login": r.json().get("login"),
                "scopes": [s.strip() for s in r.headers.get("X-OAuth-Scopes", "").split(",") if s.strip()],
            }
        return {"valid": False, "error": _explain_github_error(r)}
        
    def get_pr_diff(self, owner: str, repo: str, pr_number: int) -> str:
        """Fetches the raw diff of a Pull Request."""
        url = f"{self.base_url}/repos/{owner}/{repo}/pulls/{pr_number}"
        response = requests.get(url, headers=self._get_headers("application/vnd.github.v3.diff"))
        self._raise_for_status(response)
        return response.text

    def get_pr_comments(self, owner: str, repo: str, pr_number: int) -> List[Dict[str, Any]]:
        """Fetches review comments on a Pull Request."""
        url = f"{self.base_url}/repos/{owner}/{repo}/pulls/{pr_number}/comments"
        response = requests.get(url, headers=self._get_headers())
        self._raise_for_status(response)
        return response.json()

    def get_pr_status(self, owner: str, repo: str, pr_number: int) -> Dict[str, Any]:
        """Checks the state of the PR (merged, open, closed)."""
        url = f"{self.base_url}/repos/{owner}/{repo}/pulls/{pr_number}"
        response = requests.get(url, headers=self._get_headers())
        self._raise_for_status(response)
        return response.json()
