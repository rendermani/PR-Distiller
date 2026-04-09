import os
import sys
import json
import asyncio
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from llm.extractor import LargeLLMExtractor
from db.lightrag_manager import LightRAGManager

async def run_massive_batch(dataset_path: str):
    print("🚀 Initializing Horizontal Batch Extraction Pipeline...")
    with open(dataset_path, 'r') as f:
        datasets = json.load(f)

    db_manager = LightRAGManager()
    extractor = LargeLLMExtractor(db_manager)
    
    # We strip down the AI gateway URL directly to hit our deployed local 32B/70B model or Gemini Pro endpoints
    print(f"[*] Targeting Remote Asynchronous Endpoint: {extractor.api_base}")
    
    payloads = []
    # Collapse the fetched JSON into an aggressive raw tuple payload array
    for repo_data in datasets:
        for c in repo_data.get("human_comments", []):
            raw_text = c.get("reviewer_comment", "")
            diff = c.get("diff_hunk", "")
            payloads.append((raw_text, diff))
            
    print(f"[*] Dispatching {len(payloads)} asynchronous CodeRAG tasks to the cluster concurrently...")
    
    start = time.time()
    
    # Execute massive throughput logic, pushing the DGX GPU pipeline to maximum concurrent capacity
    # Using litellm's acompletion internally
    results = await extractor.batch_extract(payloads)
    
    elapsed = time.time() - start
    print(f"\\n✅ Batch Pipeline Complete! Processed {len(payloads)} elements in {elapsed:.2f}s")
    print(f"   Successfully serialized {len(results)} rules straight into ChromaDB.")

if __name__ == "__main__":
    target = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/human_rejected_dataset.json'))
    if os.path.exists(target):
        asyncio.run(run_massive_batch(target))
    else:
        print(f"Target data doesn't exist yet: {target}")
