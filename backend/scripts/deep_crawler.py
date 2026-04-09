import os
import sys
import json
import time
import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline.github_client import GitHubClient

def deep_crawl_rejections(owner: str, repo: str, search_pages: int = 5):
    """
    Crawls a repository for specific rejected patterns mimicking the "Enterprise AI Mistake" dilemma.
    We are looking for comments containing "use instead", "deprecated", "don't use", "temporary".
    """
    client = GitHubClient()
    dataset = []
    
    keywords = ["don't use", "do not use", "deprecated", "use instead", "temporary solution", "hack", "anti-pattern"]
    
    print(f"[*] Deep Crawling {owner}/{repo} for Rejected Copilot/Developer anti-patterns...")
    
    # We will search closed PRs for extensive comments
    for page in range(1, search_pages + 1):
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls?state=closed&per_page=30&page={page}&sort=updated&direction=desc"
        headers = {"Accept": "application/vnd.github.v3+json"}
        if client.token:
            headers["Authorization"] = f"Bearer {client.token}"
            
        r = requests.get(url, headers=headers)
        if r.status_code != 200:
            print(f"API Error {r.status_code}: Relaying...")
            break
            
        prs = r.json()
        for pr in prs:
            pr_num = pr["number"]
            try:
                comments = client.get_pr_comments(owner, repo, pr_num)
                
                # Filter for our gold-mine keywords
                golden_comments = []
                for c in comments:
                    body_lower = c.get("body", "").lower()
                    if any(kw in body_lower for kw in keywords):
                        golden_comments.append(c)
                        
                if golden_comments:
                    diff = client.get_pr_diff(owner, repo, pr_num)
                    dataset.append({
                        "pr": pr_num,
                        "title": pr["title"],
                        "golden_comments": golden_comments,
                        "diff_size": len(diff)
                    })
                    print(f" -> Found {len(golden_comments)} golden anti-pattern nodes in PR #{pr_num}")
            except Exception as e:
                pass
                
        time.sleep(1) # Respect rate limits slightly while searching deeply

    os.makedirs(os.path.join(os.path.dirname(__file__), '../data'), exist_ok=True)
    out = os.path.join(os.path.dirname(__file__), '../data/golden_rejected_dataset.json')
    with open(out, 'w') as f:
        json.dump(dataset, f, indent=2)
        
    print(f"\\n[*] Harvested {len(dataset)} high-quality enterprise-like reject patterns to {out}")

if __name__ == "__main__":
    # Use FastAPI as the target: high coding standards, lots of architectural rules
    deep_crawl_rejections("tiangolo", "fastapi", search_pages=3)
