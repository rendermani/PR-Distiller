import os
import sys
import json
import time
import requests
from datetime import datetime, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline.github_client import GitHubClient

# ---------------------------------------------------------------------------
# Module-level constants — single source of truth for filtering
# ---------------------------------------------------------------------------

# Keywords that indicate a comment contains an actionable coding lesson.
# Used across all three crawl functions.
FILTER_KEYWORDS = [
    "don't use", "do not use", "instead of", "deprecated", "hack", "anti-pattern",
    "please use", "not the best way", "change this to", "should not", "shouldn't",
    "avoid", "wrong approach", "bad practice", "not recommended", "security risk",
    "race condition", "memory leak", "n+1", "sql injection", "xss",
    "breaks when", "will fail", "bug", "regression", "missing check",
    "error handling", "edge case", "null check", "type safety",
]

# Comments that match FILTER_KEYWORDS but are docs/translation noise.
NOISE_INDICATORS = [
    "translation", "翻译", "翻譯", "tradução", "traducción", "번역",
    "typo", "spelling", "grammar", "wording", "phrasing",
    "glossary", "locale", "i18n", "l10n",
]

# Bot accounts whose comments are never useful training signal.
BOT_NAMES = frozenset([
    "dependabot[bot]", "github-actions[bot]", "vercel[bot]",
    "copilot", "renovate[bot]", "codecov[bot]",
])

# ---------------------------------------------------------------------------


def crawl_human_rejections(target_repos: list, months_back: int = 2, status_callback=None, is_cancelled=None, cursors: dict = None):
    """
    Crawls repositories for rejected patterns incrementally.
    Uses per-repo cursors (last_comment_id) to skip already-processed comments.
    Returns (dataset, updated_cursors).
    """
    client = GitHubClient()
    dataset = []
    if cursors is None:
        cursors = {}
    updated_cursors = dict(cursors)

    # Calculate exact dynamic timeframe
    time_ago = (datetime.utcnow() - timedelta(days=months_back*30)).isoformat() + "Z"

    for repo_string in target_repos:
        if "/" in repo_string:
            owner, repo = repo_string.split("/")
        else:
            continue

        last_seen_id = int(cursors.get(repo_string, 0))
        high_water_mark = last_seen_id

        print(f"\\n[*] Deep Crawling {owner}/{repo} (incremental, after comment #{last_seen_id})...")
        if status_callback:
            status_callback(f"Spidering {owner}/{repo} (Page 1)...")

        repo_data = []
        page = 1
        while True:
            if is_cancelled and is_cancelled():
                print(f"\\n[!] Spider interrupted by User Signal for {owner}/{repo}")
                break

            if status_callback and page > 1:
                status_callback(f"Spidering {owner}/{repo} (Page {page})...")

            # Sort ascending so we process oldest-first and can stop early on re-runs
            url = f"https://api.github.com/repos/{owner}/{repo}/pulls/comments?per_page=100&page={page}&since={time_ago}&sort=created&direction=asc"
            headers = {"Accept": "application/vnd.github.v3+json"}
            if client.token:
                headers["Authorization"] = f"Bearer {client.token}"

            r = requests.get(url, headers=headers)

            if r.status_code == 403 or r.status_code == 429:
                reset_time = int(r.headers.get("x-ratelimit-reset", time.time() + 60))
                sleep_dur = max(int(reset_time - time.time()) + 5, 60)
                msg = f"GitHub Rate Limit. Sleeping {sleep_dur}s..."
                print(f"\\n[!] {msg}")
                if status_callback:
                    status_callback(msg)

                for _ in range(sleep_dur):
                    if is_cancelled and is_cancelled():
                        print("\\n[!] Cancelled during rate limit lock.")
                        return [], cursors
                    time.sleep(1)

                continue

            if r.status_code != 200:
                print(f"API Error {r.status_code}. Breaking repo scrape.")
                break

            comments = r.json()
            if not comments:
                break

            for c in comments:
                comment_id = int(c.get("id", 0))

                # Skip already-processed comments
                if comment_id <= last_seen_id:
                    continue

                if comment_id > high_water_mark:
                    high_water_mark = comment_id

                user_login = (c.get("user") or {}).get("login", "").lower()

                if user_login in BOT_NAMES or "[bot]" in user_login:
                    continue

                body = c.get("body", "")
                body_lower = body.lower()
                diff_hunk = c.get("diff_hunk", "")

                # Must have actual code context and meaningful length
                if len(body) < 40 or not diff_hunk:
                    continue

                # Skip translation/docs noise
                if any(noise in body_lower for noise in NOISE_INDICATORS):
                    continue

                # Skip comments on .md / .po / .rst files (doc translations)
                path = c.get("path", "")
                if path.endswith((".md", ".po", ".pot", ".rst", ".txt")):
                    continue

                if any(kw in body_lower for kw in FILTER_KEYWORDS):
                    repo_data.append({
                        "id": str(comment_id),
                        "pr_url": c.get("pull_request_url"),
                        "diff_hunk": diff_hunk,
                        "reviewer_comment": body
                    })
                    print(f"    -> Harvested Human Insight from {user_login} [Page {page}]")

            time.sleep(3.5)
            page += 1

        updated_cursors[repo_string] = high_water_mark
        print(f"    [Complete] Yielded {len(repo_data)} new anti-patterns (cursor: {last_seen_id} -> {high_water_mark})")
        dataset.extend([(d["reviewer_comment"], d["diff_hunk"]) for d in repo_data])

    print(f"\\n[*] Incremental crawl returning {len(dataset)} new tuples.")
    return dataset, updated_cursors

def crawl_pr_reviews(target_repos: list, months_back: int = 2, status_callback=None, is_cancelled=None, cursors: dict = None):
    """
    Crawls top-level PR review bodies (REQUEST_CHANGES / COMMENT reviews).
    These often contain high-level architectural feedback not found in inline comments.
    Returns (dataset, updated_cursors) — same format as crawl_human_rejections.
    """
    client = GitHubClient()
    dataset = []
    if cursors is None:
        cursors = {}
    updated_cursors = dict(cursors)

    time_ago = (datetime.utcnow() - timedelta(days=months_back * 30)).isoformat() + "Z"

    for repo_string in target_repos:
        if "/" not in repo_string:
            continue
        owner, repo = repo_string.split("/")

        last_seen_pr = int(cursors.get(f"{repo_string}_reviews", 0))
        high_water_mark = last_seen_pr

        if status_callback:
            status_callback(f"Crawling PR reviews for {owner}/{repo}...")

        # Fetch recently updated closed/merged PRs
        page = 1
        review_data = []
        while True:
            if is_cancelled and is_cancelled():
                break

            url = (f"https://api.github.com/repos/{owner}/{repo}/pulls"
                   f"?state=closed&sort=updated&direction=desc&per_page=50&page={page}")
            headers = {"Accept": "application/vnd.github.v3+json"}
            if client.token:
                headers["Authorization"] = f"Bearer {client.token}"

            r = requests.get(url, headers=headers)
            if r.status_code in (403, 429):
                reset_time = int(r.headers.get("x-ratelimit-reset", time.time() + 60))
                sleep_dur = max(int(reset_time - time.time()) + 5, 60)
                if status_callback:
                    status_callback(f"Rate limited. Sleeping {sleep_dur}s...")
                for _ in range(sleep_dur):
                    if is_cancelled and is_cancelled():
                        return [], cursors
                    time.sleep(1)
                continue

            if r.status_code != 200:
                break

            pulls = r.json()
            if not pulls:
                break

            # Stop if we've gone past our time window
            oldest_updated = pulls[-1].get("updated_at", "")
            if oldest_updated < time_ago:
                pulls = [p for p in pulls if p.get("updated_at", "") >= time_ago]
                if not pulls:
                    break

            for pr in pulls:
                pr_number = pr.get("number", 0)
                if pr_number <= last_seen_pr:
                    continue
                if pr_number > high_water_mark:
                    high_water_mark = pr_number

                # Fetch reviews for this PR
                rev_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/reviews?per_page=100"
                rev_r = requests.get(rev_url, headers=headers)
                if rev_r.status_code != 200:
                    continue

                for review in rev_r.json():
                    body = review.get("body", "") or ""
                    user_login = (review.get("user") or {}).get("login", "").lower()
                    state = review.get("state", "")

                    if user_login in BOT_NAMES or "[bot]" in user_login:
                        continue
                    if len(body) < 40:
                        continue
                    # Only interested in reviews with substance
                    if state not in ("CHANGES_REQUESTED", "COMMENTED"):
                        continue

                    body_lower = body.lower()
                    if any(kw in body_lower for kw in FILTER_KEYWORDS):
                        # Use PR diff as context (first 2000 chars)
                        diff_context = pr.get("body", "") or ""
                        review_data.append({
                            "id": str(review.get("id", 0)),
                            "reviewer_comment": body,
                            "diff_hunk": diff_context[:2000]
                        })
                        print(f"    -> PR Review from {user_login} on PR#{pr_number}")

                time.sleep(1)  # Throttle per-PR review fetch

            time.sleep(3.5)
            page += 1

        updated_cursors[f"{repo_string}_reviews"] = high_water_mark
        print(f"    [Reviews Complete] Yielded {len(review_data)} review insights for {repo_string}")
        dataset.extend([(d["reviewer_comment"], d["diff_hunk"]) for d in review_data])

    return dataset, updated_cursors


QUALIFYING_ISSUE_LABELS = frozenset([
    "bug", "wontfix", "invalid", "won't fix", "not a bug", "duplicate",
])

# Alias so that issue-crawl helpers use the same single source of truth.
ISSUE_KEYWORDS = FILTER_KEYWORDS

ISSUE_CONTEXT_MAX_CHARS = 2000
ISSUE_COMMENT_MIN_LENGTH = 40


def _issue_qualifies_by_label(issue: dict) -> bool:
    """True if any of the issue's labels are in QUALIFYING_ISSUE_LABELS."""
    issue_labels = {lbl.get("name", "").lower() for lbl in (issue.get("labels") or [])}
    return bool(issue_labels & QUALIFYING_ISSUE_LABELS)


def _issue_qualifies_by_keyword(issue: dict) -> bool:
    """True if the issue body contains at least one architectural lesson keyword."""
    body = (issue.get("body") or "").lower()
    return any(kw in body for kw in ISSUE_KEYWORDS)


def _is_bot_comment(comment: dict) -> bool:
    login = (comment.get("user") or {}).get("login", "").lower()
    return login in BOT_NAMES or "[bot]" in login


def _comment_contains_lesson(body: str) -> bool:
    body_lower = body.lower()
    return any(kw in body_lower for kw in ISSUE_KEYWORDS)


def _handle_rate_limit(response, status_callback, is_cancelled):
    """
    Sleeps until the rate-limit window resets.
    Returns True if the caller should retry the request.
    Returns False if cancellation was requested during sleep.
    """
    reset_time = int(response.headers.get("x-ratelimit-reset", time.time() + 60))
    sleep_dur = max(int(reset_time - time.time()) + 5, 60)
    msg = f"GitHub Rate Limit. Sleeping {sleep_dur}s..."
    print(f"\n[!] {msg}")
    if status_callback:
        status_callback(msg)

    for _ in range(sleep_dur):
        if is_cancelled and is_cancelled():
            print("\n[!] Cancelled during rate limit sleep.")
            return False
        time.sleep(1)

    return True


def _fetch_issue_comments(owner: str, repo: str, issue_number: int, headers: dict) -> list:
    """
    Fetches all comments for a closed issue.
    Returns an empty list on any API error so the caller can skip gracefully.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments?per_page=100"
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        print(f"    [!] Comments fetch failed for issue #{issue_number}: HTTP {r.status_code}")
        return []
    return r.json()


def _extract_lessons_from_issue(issue: dict, issue_comments: list) -> list:
    """
    Returns (comment_body, issue_body_context) tuples for all comments
    that pass bot-filtering, minimum-length, and keyword checks.
    The issue body (truncated to ISSUE_CONTEXT_MAX_CHARS) serves as context,
    mirroring the role diff_hunk plays in PR-comment crawlers.
    """
    context = (issue.get("body") or "")[:ISSUE_CONTEXT_MAX_CHARS]
    lessons = []

    for comment in issue_comments:
        if _is_bot_comment(comment):
            continue

        body = comment.get("body") or ""
        if len(body) < ISSUE_COMMENT_MIN_LENGTH:
            continue

        if _comment_contains_lesson(body):
            user_login = (comment.get("user") or {}).get("login", "unknown")
            print(f"    -> Harvested Issue Lesson from {user_login} on issue #{issue.get('number')}")
            lessons.append((body, context))

    return lessons


def crawl_closed_issues(
    target_repos: list,
    months_back: int = 2,
    status_callback=None,
    is_cancelled=None,
    cursors: dict = None,
):
    """
    Crawls closed GitHub issues for architectural lessons.

    Issues qualify if they carry a label in QUALIFYING_ISSUE_LABELS OR if
    their body contains a keyword from ISSUE_KEYWORDS.  Each qualifying
    issue's comments are then individually filtered by keyword and length.

    Uses cursor key ``{repo_string}_issues`` (= highest issue number seen)
    for incremental re-runs — same contract as the PR crawlers.

    Returns (dataset, updated_cursors) where dataset is a list of
    (comment_body, issue_body_context) tuples.
    """
    client = GitHubClient()
    dataset = []
    if cursors is None:
        cursors = {}
    updated_cursors = dict(cursors)

    time_ago = (datetime.utcnow() - timedelta(days=months_back * 30)).isoformat() + "Z"

    for repo_string in target_repos:
        if "/" not in repo_string:
            continue

        owner, repo = repo_string.split("/")
        cursor_key = f"{repo_string}_issues"

        last_seen_issue = int(cursors.get(cursor_key, 0))
        high_water_mark = last_seen_issue

        if status_callback:
            status_callback(f"Crawling closed issues for {owner}/{repo}...")

        headers = {"Accept": "application/vnd.github.v3+json"}
        if client.token:
            headers["Authorization"] = f"Bearer {client.token}"

        repo_lessons = []
        page = 1

        while True:
            if is_cancelled and is_cancelled():
                print(f"\n[!] Closed-issue crawl interrupted for {owner}/{repo}")
                break

            if status_callback and page > 1:
                status_callback(f"Crawling closed issues for {owner}/{repo} (Page {page})...")

            url = (
                f"https://api.github.com/repos/{owner}/{repo}/issues"
                f"?state=closed&since={time_ago}&sort=updated&direction=asc"
                f"&per_page=100&page={page}"
            )
            r = requests.get(url, headers=headers)

            if r.status_code in (403, 429):
                should_retry = _handle_rate_limit(r, status_callback, is_cancelled)
                if not should_retry:
                    return [], cursors
                continue

            if r.status_code != 200:
                print(f"    [!] Issues API error {r.status_code} for {owner}/{repo}. Stopping.")
                break

            issues = r.json()
            if not issues:
                break

            for issue in issues:
                issue_number = issue.get("number", 0)

                if issue_number <= last_seen_issue:
                    continue

                if issue_number > high_water_mark:
                    high_water_mark = issue_number

                qualifies = (
                    _issue_qualifies_by_label(issue)
                    or _issue_qualifies_by_keyword(issue)
                )
                if not qualifies:
                    continue

                issue_comments = _fetch_issue_comments(owner, repo, issue_number, headers)
                lessons = _extract_lessons_from_issue(issue, issue_comments)
                repo_lessons.extend(lessons)

                time.sleep(1)  # Throttle per-issue comment fetch

            time.sleep(3.5)
            page += 1

        updated_cursors[cursor_key] = high_water_mark
        print(
            f"    [Issues Complete] Yielded {len(repo_lessons)} lessons"
            f" for {repo_string} (cursor: {last_seen_issue} -> {high_water_mark})"
        )
        dataset.extend(repo_lessons)

    print(f"\n[*] Closed-issue crawl returning {len(dataset)} new tuples.")
    return dataset, updated_cursors


if __name__ == "__main__":
    targets = [
        ("pallets", "flask"),
        ("tiangolo", "sqlmodel"),
        ("encode", "starlette"),
        ("langchain-ai", "langgraph")
    ]
    crawl_human_rejections(targets)
