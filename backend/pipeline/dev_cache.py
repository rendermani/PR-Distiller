import os
import json
import tempfile
from datetime import datetime

import settings

CACHE_DIR = settings.DEV_CACHE_DIR


def cache_path(repo: str) -> str:
    safe_name = repo.replace("/", "__")
    return os.path.join(CACHE_DIR, f"{safe_name}.json")


def save_crawl(repo: str, tuples: list):
    """Saves raw (comment, diff_hunk) tuples to disk."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = cache_path(repo)

    # Append to existing cache rather than overwriting (incremental crawls add data)
    existing = load_crawl(repo)
    seen = {(c, d) for c, d in existing}
    new_count = 0
    for comment, diff in tuples:
        if (comment, diff) not in seen:
            existing.append((comment, diff))
            seen.add((comment, diff))
            new_count += 1

    payload = {
        "repo": repo,
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "count": len(existing),
        "tuples": [[c, d] for c, d in existing]
    }
    # Write atomically: write to a temp file then rename so a crash mid-write
    # never leaves a truncated or corrupt cache file.
    dir_name = os.path.dirname(path)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=dir_name, suffix=".tmp", delete=False
    ) as tmp_file:
        json.dump(payload, tmp_file, indent=2, ensure_ascii=False)
        tmp_path = tmp_file.name

    os.replace(tmp_path, path)

    print(f"[Dev Cache] Saved {new_count} new tuples for {repo} (total: {len(existing)})")


def load_crawl(repo: str) -> list:
    """Loads cached (comment, diff_hunk) tuples from disk. Returns [] if no cache."""
    path = cache_path(repo)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            print(f"[Dev Cache] Corrupt cache for {repo}, ignoring: {exc}")
            return []
    return [(t[0], t[1]) for t in data.get("tuples", [])]


def has_cache(repo: str) -> bool:
    return os.path.exists(cache_path(repo))


def cache_info(repo: str) -> dict:
    path = cache_path(repo)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {"repo": repo, "count": data.get("count", 0), "updated_at": data.get("updated_at")}
