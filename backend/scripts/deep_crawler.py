import os
import sys
import json
import time
import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline.github_client import GitHubClient

def crawl_human_rejections(targets: list, search_pages: int = 2):
    """
    Crawls a list of repositories for specific rejected patterns mimicking the "Enterprise AI Mistake" dilemma.
    Explicitly filters OUT known bots (dependabot, github-actions, copilot).
    """
    client = GitHubClient()
    dataset = []
    
    keywords = ["don't use", "instead of", "deprecated", "hack", "anti-pattern", "please use", "not the best way", "change this to"]
    bot_names = ["dependabot[bot]", "github-actions[bot]", "vercel[bot]", "copilot", "renovate[bot]", "codecov[bot]"]
    
    for owner, repo in targets:
        print(f"\\n[*] Deep Crawling {owner}/{repo} for messy HUMAN developer anti-patterns...")
        repo_data = []
        
        for page in range(1, search_pages + 1):
            url = f"https://api.github.com/repos/{owner}/{repo}/pulls/comments?per_page=50&page={page}&sort=created&direction=desc"
            headers = {"Accept": "application/vnd.github.v3+json"}
            if client.token:
                headers["Authorization"] = f"Bearer {client.token}"
                
            r = requests.get(url, headers=headers)
            if r.status_code != 200:
                print(f"API Error {r.status_code}. Rate limit?")
                break
                
            comments = r.json()
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
                    print(f"    -> Harvested Human Insight from {user_login}")
                    
                # We just need a high-quality sample size per repo
                if len(repo_data) >= 5:
                    break
            
            if len(repo_data) >= 5:
                    break
            time.sleep(1) # Rate limit respect
            
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
        ("vercel", "next.js"),     # React framework, tricky UI patterns
        ("pydantic", "pydantic"),  # Deep python architecture rules
        ("huggingface", "transformers") # ML infrastructure rules
    ]
    crawl_human_rejections(targets, search_pages=3)
