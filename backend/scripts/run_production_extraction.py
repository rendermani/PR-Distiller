import os
import sys
import json
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.security_redactor import SecurityRedactor
from pipeline.deduplication import DeduplicationFilter
from pipeline.ast_slicer import ASTSlicer
from db.lightrag_manager import LightRAGManager
from llm.intent_checker import IntentChecker
from llm.router import SmallLLMRouter
from llm.extractor import LargeLLMExtractor


def repo_slug_from_pr_url(pull_request_url: str) -> str:
    """Extracts 'owner/repo' from a GitHub API pull_request_url.

    Example input:
        https://api.github.com/repos/langchain-ai/langchain/pulls/36613
    Example output:
        langchain-ai/langchain
    Returns an empty string when the URL does not match the expected pattern.
    """
    prefix = "https://api.github.com/repos/"
    if not pull_request_url or not pull_request_url.startswith(prefix):
        return ""
    remainder = pull_request_url[len(prefix):]
    parts = remainder.split("/")
    if len(parts) < 2:
        return ""
    return f"{parts[0]}/{parts[1]}"


def process_comment(
    comment: dict,
    pr: dict,
    extractor: LargeLLMExtractor,
    redactor: SecurityRedactor,
    dedup: DeduplicationFilter,
    intent: IntentChecker,
    slicer: ASTSlicer,
) -> dict | None:
    """Applies the full filter pipeline to a single comment and calls extract_rule.

    Returns the extracted rule dict on success, or None when the comment is
    skipped (duplicate, non-code intent) or extraction fails.
    """
    body = comment.get("body", "")

    if dedup.is_duplicate(body):
        return None

    safe_text = redactor.redact_text(body)

    if intent.check_intent(safe_text) != "CODE":
        return None

    diff = pr.get("diff", "")
    target_line = comment.get("line") or comment.get("original_line") or 0
    sliced_ast = slicer.get_node_at_line(diff, target_line)

    pr_url = comment.get("pull_request_url", "")
    repo = repo_slug_from_pr_url(pr_url)

    return extractor.extract_rule(safe_text, sliced_ast, repo)


def main():
    print("🚀 Initializing Enterprise AGILE-RULE-EXTRACTOR Production Pipeline...")
    
    data_path = os.path.join(os.path.dirname(__file__), '../data/test_prs.json')
    with open(data_path, 'r') as f:
        prs = json.load(f)

    # Initialize Core Pipeline Filter Layers
    redactor = SecurityRedactor()
    dedup = DeduplicationFilter()
    slicer = ASTSlicer()
    intent = IntentChecker()
    
    # Initialize Core Persistent CodeRAG Components
    db_manager = LightRAGManager()
    
    # Initialize Intelligent AI Layers
    router = SmallLLMRouter(db_manager)
    extractor = LargeLLMExtractor(db_manager)

    print(f"[*] Sub-systems synced. Target Extractor Endpoint: {extractor.api_base}")
    print(f"[*] Processing {len(prs)} PRs...")
    
    success = 0
    start_cpu_time = time.time()
    
    for pr in prs:
        for comment in pr["comments"]:
            body = comment.get("body", "")

            # 1. Pipeline: Defensive Filtering (dedup + redaction + intent)
            if dedup.is_duplicate(body):
                continue
            safe_text = redactor.redact_text(body)
            if intent.check_intent(safe_text) != "CODE":
                continue

            # 2. Pipeline: AI Routing & CodeRAG cache pre-check
            route_info = router.categorize_comment(safe_text)
            route = route_info.get("route")

            if route == "exact_match_found":
                print(f"\\n✅ [CodeRAG Cache Hit]: Reusing Rule ID {route_info['cached_rule']}")
                success += 1
                continue
            elif route == "discard_noise":
                continue

            # 3. Pipeline: AST Context Assembly + Extraction
            diff = pr.get("diff", "")
            target_line = comment.get("line") or comment.get("original_line") or 0
            sliced_ast = slicer.get_node_at_line(diff, target_line)
            pr_url = comment.get("pull_request_url", "")
            repo = repo_slug_from_pr_url(pr_url)

            print(f"\\n⏳ [Extraction Pending] Target: {safe_text[:60]}... \\n -> Pushing to Heavy Architecture Endpoints...")
            rule = extractor.extract_rule(safe_text, sliced_ast, repo)

            if rule:
                print(f"\\n🎯 [Extraction Success] Rule Generated & Inserted into CodeRAG Context:\\n{json.dumps(rule, indent=2)}")
                success += 1
            else:
                print(f"❌ [Extraction Failed] Heavy Endpoint dropped the JSON Payload.")

    elapsed = time.time() - start_cpu_time
    print(f"\\n==========================\\n🏁 PRODUCTION RUN COMPLETE\\nProcessed Test Nodes in {elapsed:.2f}s | Successfully Generated & Vented Rules to CodeRAG Database: {success}\\n==========================")

if __name__ == "__main__":
    main()
