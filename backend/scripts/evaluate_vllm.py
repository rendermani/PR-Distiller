import os
import sys
import json
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.security_redactor import SecurityRedactor
from pipeline.deduplication import DeduplicationFilter
from pipeline.ast_slicer import ASTSlicer
from openai import OpenAI

# Connect to GX10 vLLM
client = OpenAI(
    base_url="http://gx10:8000/v1",
    api_key="vllm-dummy-key"
)

# You can adjust this to evaluate multiple models loaded or test iterations
MODEL_NAME = "google/gemma-4-9b-it"

def extract_with_llm(pr_comment: str, modified_ast_code: str):
    system_prompt = (
        "You are an automated extraction tool. Your ONLY task is to extract coding rules into JSON. "
        "Strictly return nothing but the JSON following this pattern: "
        '{"rule_id": "...", "metadata": {"status": "active", "labels": [], "confidence_score": 1.0}, "content": {"title": "...", "description": "...", "condition": "...", "enforcement_prompt": "...", "examples": {"bad": "...", "good": "..."}}}'
    )
    
    user_prompt = f"""
<REVIEW_DATA>
{pr_comment}
</REVIEW_DATA>

<CODE_DIFF>
{modified_ast_code}
</CODE_DIFF>
"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )
        return json.loads(response.choices[0].message.content.strip('```json\n'))
    except Exception as e:
        print(f"Extraction failed: {e}")
        return None

def main():
    data_path = os.path.join(os.path.dirname(__file__), '../data/test_prs.json')
    with open(data_path, 'r') as f:
        prs = json.load(f)

    redactor = SecurityRedactor()
    dedup = DeduplicationFilter()
    slicer = ASTSlicer()

    print(f"Starting actual evaluation against vLLM on GX10 using {MODEL_NAME}...")

    success = 0
    start_time = time.time()
    
    for pr in prs:
        for comment in pr["comments"]:
            body = comment.get("body", "")
            
            # Avoid API calls for duplicates
            if dedup.is_duplicate(body):
                continue
                
            safe_text = redactor.redact_text(body)
            diff = pr.get("diff", "")
            target_line = comment.get("line") or comment.get("original_line") or 0
            sliced_ast = slicer.get_node_at_line(diff, target_line)
            
            print(f"Processing comment: {safe_text[:50]}...")
            rule = extract_with_llm(safe_text, sliced_ast)
            if rule:
                success += 1
                print(f" -> Extracted Rule: {rule['content']['title']}")

    dur = time.time() - start_time
    print(f"\\nEvaluation finished! Successfully extracted {success} rules via LLM Inference in {dur:.2f}s.")

if __name__ == "__main__":
    main()
