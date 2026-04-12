import asyncio
import time
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from scripts.deep_crawler import crawl_human_rejections
from llm.extractor import LargeLLMExtractor
from db.lightrag_manager import LightRAGManager
from pipeline.config_manager import ConfigManager
from pipeline.deduplication import DeduplicationFilter
from pipeline.security_redactor import SecurityRedactor
from pipeline import dev_cache

class JobOrchestrator:
    """
    Manages long-running pipeline distillations asynchronously preventing API thread locks.
    Maintains active volatile state strings for frontend react metric rendering.
    """
    def __init__(self, db_instance: LightRAGManager):
        self.db = db_instance
        self.conf = ConfigManager()
        self.active_jobs = {}
        self.cancel_flags = {}
        
    def get_status(self, job_id: str):
        return self.active_jobs.get(job_id, {"status": "not_found"})

    def request_cancel(self, job_id: str):
        """Flips true halting the specific pipeline iteration safely"""
        if job_id in self.active_jobs:
            self.cancel_flags[job_id] = True
            return True
        return False

    async def _execute_distillation(self, job_id: str, payload: dict, config: dict):
        try:
            repo = payload.get("repo")
            months = payload.get("months", 2)
            threshold = payload.get("threshold", 0.45) # Overridden in Extract/Fuser classes
            
            # Setup Git Token overriding os.environ securely from config matrix
            token = config.get("github_token", "")
            if token:
                os.environ["GITHUB_TOKEN"] = token
                
            self.active_jobs[job_id]["status"] = f"Spidering {repo} ({months} months)"
            self.active_jobs[job_id]["progress"] = 10

            def set_spider_status(msg):
                if job_id in self.active_jobs:
                    self.active_jobs[job_id]["status"] = msg

            def check_cancel():
                return self.cancel_flags.get(job_id, False)

            # Load per-repo cursors from config
            current_conf = self.conf.load_config()
            cursors = current_conf.get("crawl_cursors", {})

            use_cache = payload.get("use_cache", False)

            if use_cache and dev_cache.has_cache(repo):
                info = dev_cache.cache_info(repo)
                self.active_jobs[job_id]["status"] = f"Using dev cache ({info['count']} tuples from {info['updated_at'][:10]})"
                pr_data = await asyncio.to_thread(dev_cache.load_crawl, repo)
            else:
                # 1. Incremental crawl — only fetches comments after last cursor
                pr_data, updated_cursors = await asyncio.to_thread(
                    crawl_human_rejections, [repo], months, set_spider_status, check_cancel, cursors
                )

                # Persist updated cursors immediately so even cancelled runs save progress
                current_conf["crawl_cursors"] = updated_cursors
                self.conf.save_config(current_conf)

                # Save to dev cache for future replay
                if pr_data:
                    await asyncio.to_thread(dev_cache.save_crawl, repo, pr_data)

            if check_cancel():
                self.active_jobs[job_id]["status"] = "Pipeline Aborted via User Interrupt"
                self.active_jobs[job_id]["progress"] = -1
                return

            # 2. Pre-process: hash dedup + PII redaction before any LLM call
            dedup = DeduplicationFilter()
            redactor = SecurityRedactor()
            cleaned_payloads = []
            for comment_body, diff_hunk in pr_data:
                if dedup.is_duplicate(comment_body):
                    continue
                safe_body = redactor.redact_text(comment_body)
                cleaned_payloads.append((safe_body, diff_hunk))

            self.active_jobs[job_id]["status"] = f"Batch Processing {len(cleaned_payloads)} rule payloads with {config.get('llm_model')}"
            self.active_jobs[job_id]["progress"] = 35

            # 3. Extract
            os.environ["EXTRACTOR_API_BASE"] = config.get("llm_api_base", "http://gx10:8080/v1")
            os.environ["EXTRACTOR_API_KEY"] = config.get("llm_api_key", "dummy-key")
            os.environ["EXTRACTOR_MODEL"] = config.get("llm_model", "openai/Qwen/Qwen2.5-Coder-7B-Instruct")
            extractor = LargeLLMExtractor(self.db)

            await extractor.batch_extract(cleaned_payloads, repo=repo)
            
            self.active_jobs[job_id]["status"] = f"Completed Pipeline Run successfully!"
            self.active_jobs[job_id]["progress"] = 100
            
        except Exception as e:
            self.active_jobs[job_id]["status"] = f"FAILED: {str(e)}"
            self.active_jobs[job_id]["progress"] = -1

    def trigger_job(self, payload: dict, current_config: dict) -> str:
        """Kicks off the asynchronous process detached from the current block"""
        job_id = f"job-{int(time.time())}"
        self.active_jobs[job_id] = {"status": "Initializing Engine...", "progress": 0}
        self.cancel_flags[job_id] = False
        
        # Fire and forget directly into the active FastAPI root thread reliably
        asyncio.create_task(self._execute_distillation(job_id, payload, current_config))
        return job_id
