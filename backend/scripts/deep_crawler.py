import os
import sys
import json
import time
import requests
from datetime import datetime, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline.github_client import GitHubClient

def crawl_human_rejections(targets: list):
    """
    Crawls medium-sized repositories for specific rejected patterns.
    Explicitly throttled (Spider Slow) and limits history precisely to the last 2 months.
    """
    client = GitHubClient()
    dataset = []
    
    # Calculate exactly 2 months ago (60 days) to restrict the scope
    two_months_ago = (datetime.utcnow() - timedelta(days=60)).isoformat() + "Z"
    
    keywords = ["don't use", "instead of", "deprecated", "hack", "anti-pattern", "please use", "not the best way", "change this to"]
    bot_names = ["dependabot[bot]", "github-actions[bot]", "vercel[bot]", "copilot", "renovate[bot]", "codecov[bot]"]
    
    for owner, repo in targets:
        print(f"\\n[*] Deep Crawling {owner}/{repo} for messy HUMAN developer anti-patterns...")
        repo_data = []
        # We parse until we hit the end of the 2-month window or the repo is exhausted.
        # We don't artificially limit `search_pages` anymore; relying strictly on the time bound.
        page = 1
        while True:
            # We use `since` to restrict historical depth effectively
            url = f"https://api.github.com/repos/{owner}/{repo}/pulls/comments?per_page=100&page={page}&since={two_months_ago}&sort=created&direction=desc"
            headers = {"Accept": "application/vnd.github.v3+json"}
            if client.token:
                headers["Authorization"] = f"Bearer {client.token}"
                
            r = requests.get(url, headers=headers)
            
            # SPIDER SLOW: Handle explicit GitHub primary rate-limiting and secondary abuse blocks cleanly
            if r.status_code == 403 or r.status_code == 429:
                reset_time = int(r.headers.get("x-ratelimit-reset", time.time() + 60))
                sleep_dur = max(int(reset_time - time.time()) + 5, 60)
                print(f"\\n[!] GitHub Rate Limit hit! Sleeping spider for {sleep_dur} seconds...")
                time.sleep(sleep_dur)
                continue # Retry the exact same URL payload
                
            if r.status_code != 200:
                print(f"API Error {r.status_code}. Breaking repo scrape.")
                break
                
            comments = r.json()
            if not comments:
                break # We reached the chronological bottom of the 2-month window
                
            for c in comments:
                user_login = c.get("user", {}).get("login", "").lower()
                
                # STRICT HUMAN FILTER: Reject bots entirely
                if user_login in bot_names or "[bot]" in user_login:
                    continue
                    
                body_lower = c.get("body", "").lower()
                if any(kw in body_lower for kw in keywords) and len(c.get("body", "")) > 40:
                    repo_data.append({
                        "id": str(c.get("id")),
                        "pr_url": c.get("pull_request_url"),
                        "diff_hunk": c.get("diff_hunk", ""),
                        "reviewer_comment": c.get("body", "")
                    })
                    print(f"    -> Harvested Human Insight from {user_login} [Page {page}]")
            
            # SPIDER SLOW: Enforce hard crawl delays to avoid triggering GitHub abuse heuristics
            time.sleep(3.5)
            page += 1

        print(f"    [Complete] Yielded {len(repo_data)} anti-patterns over the 2-month span.")
        dataset.append({
            "repo": f"{owner}/{repo}",
            "human_comments": repo_data
        })

    out = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/human_rejected_dataset.json'))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(dataset, f, indent=2)
        
    print(f"\\n[*] Harvested cross-repo Human reject patterns to {out}")

if __name__ == "__main__":
    targets = [
        ("pallets", "flask"),          # Medium Python Framework
        ("tiangolo", "sqlmodel"),      # Medium Database ORM
        ("encode", "starlette"),       # Deep architectural logic
        ("langchain-ai", "langgraph")  # Complex multi-step agent topologies
    ]
    crawl_human_rejections(targets)
