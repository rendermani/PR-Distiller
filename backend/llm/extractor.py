import os
import json
import asyncio
from litellm import completion, acompletion
from db.lightrag_manager import LightRAGManager
from pipeline.semantic_fuser import SemanticFuser

class LargeLLMExtractor:
    def __init__(self, db_manager: LightRAGManager):
        self.model = os.environ.get("EXTRACTOR_MODEL", "openai/Qwen/Qwen2.5-Coder-7B-Instruct")
        self.api_base = os.environ.get("EXTRACTOR_API_BASE", "http://gx10:8080/v1")
        self.api_key = os.environ.get("EXTRACTOR_API_KEY", "dgx-dummy-key")
        self.db = db_manager
        self.fuser = SemanticFuser(db_manager)
        
    def extract_rule(self, pr_comment: str, modified_ast_code: str) -> dict:
        system_prompt = (
            "You are an expert AI Architect extracting coding rules into strict JSON. "
            "Your goal is to parse reviewer feedback and isolate what the Copilot did WRONG. "
            "Output ONLY valid JSON following this schema: "
            "{'rule_id': 'uuid', 'metadata': {'status': 'active'}, 'content': {'title': '...', 'description': '...', 'enforcement_prompt': '...'}}"
        )
        
        user_prompt = f'''
        <REVIEW_DATA>
        {pr_comment}
        </REVIEW_DATA>
        
        <CODE_DIFF>
        {modified_ast_code}
        </CODE_DIFF>
        '''
        
        try:
            response = completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                api_base=self.api_base,
                api_key=self.api_key,
                temperature=0.1
            )
            raw_content = response.choices[0].message.content.strip()
            
            if raw_content.startswith("```json"):
                raw_content = raw_content.replace("```json", "").replace("```", "").strip()
                
            rule_dict = json.loads(raw_content)
            
            # Replaced naive storage with the new Dynamic Semantic Deduplication engine
            self.fuser.process_and_fuse(rule_dict)
            return rule_dict
            
        except Exception as e:
            print(f"[Extractor Error]: {e}")
            return None

    async def async_extract_rule(self, pr_comment: str, modified_ast_code: str) -> dict:
        system_prompt = (
            "You are an expert AI Architect extracting coding rules into strict JSON. "
            "Your goal is to parse reviewer feedback and isolate what the Copilot did WRONG. "
            "Output ONLY valid JSON following this schema: "
            "{'rule_id': 'uuid', 'metadata': {'status': 'active'}, 'content': {'title': '...', 'description': '...', 'enforcement_prompt': '...'}}"
        )
        
        user_prompt = f'''
        <REVIEW_DATA>
        {pr_comment}
        </REVIEW_DATA>
        
        <CODE_DIFF>
        {modified_ast_code}
        </CODE_DIFF>
        '''

        try:
            response = await acompletion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                api_base=self.api_base,
                api_key=self.api_key,
                temperature=0.1
            )
            raw_content = response.choices[0].message.content.strip()
            if raw_content.startswith("```json"):
                raw_content = raw_content.replace("```json", "").replace("```", "").strip()
                
            rule_dict = json.loads(raw_content)
            
            # Run organic deduplication merge
            self.fuser.process_and_fuse(rule_dict)
            return rule_dict
        except Exception as e:
            print(f"[Async Extractor Error]: {e}")
            return None

    async def batch_extract(self, payloads: list) -> list:
        tasks = [self.async_extract_rule(c, a) for (c, a) in payloads]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]
