import os
import sys
import json
import requests
from typing import List

# Adjust path to import pipeline modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline.github_client import GitHubClient

def fetch_tuning_data(owner: str, repo: str, count: int = 5):
    client = GitHubClient()
    print(f"Fetching recent {count} closed PRs from {owner}/{repo}...")
    
    # 1. Fetching a list of closed PRs
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls?state=closed&per_page={count}&sort=updated&direction=desc"
    headers = {"Accept": "application/vnd.github.v3+json"}
    if client.token:
        headers["Authorization"] = f"Bearer {client.token}"
        
    response = requests.get(url, headers=headers)
    
    if response.status_code == 403:
        print("API Rate Limit exceeded! Please set the GITHUB_TOKEN environment variable.")
        sys.exit(1)
        
    response.raise_for_status()
    prs = response.json()
    
    dataset = []
    
    for pr in prs:
        pr_number = pr["number"]
        print(f"Fetching details for PR #{pr_number}: {pr['title']}")
        
        try:
            # Fetch Comments
            comments = client.get_pr_comments(owner, repo, pr_number)
            if not comments:
                print(f"  -> Skipping PR #{pr_number} (No review comments found)")
                continue 
                
            # Fetch Diff
            diff = client.get_pr_diff(owner, repo, pr_number)
            
            dataset.append({
                "pr_number": pr_number,
                "title": pr["title"],
                "merge_commit_sha": pr.get("merge_commit_sha"),
                "comments": comments,
                "diff": diff
            })
            print(f"  -> Success: Collected {len(comments)} comment nodes.")
        except Exception as e:
            print(f"Failed fetching PR #{pr_number}. Error: {e}")
            
    # Save to disk
    os.makedirs(os.path.join(os.path.dirname(__file__), '../data'), exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__), '../data/test_prs.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)
        
    print(f"\\nSaved {len(dataset)} valid PR threads into {out_path} for system evaluation!")

if __name__ == "__main__":
    # Target repository selection. 
    # 'langchain' is highly active, producing extremely technical Python AI PRs, perfect for tuning
    TARGET_OWNER = "langchain-ai" 
    TARGET_REPO = "langchain"
    FETCH_COUNT = 15
    fetch_tuning_data(TARGET_OWNER, TARGET_REPO, count=FETCH_COUNT)
