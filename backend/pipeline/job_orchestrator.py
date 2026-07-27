import asyncio
import logging
import time
import os
import sys

from litellm import acompletion

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import settings

logger = logging.getLogger(__name__)
from scripts.deep_crawler import crawl_human_rejections, crawl_pr_reviews, crawl_closed_issues
from llm.extractor import LargeLLMExtractor
from db.lightrag_manager import LightRAGManager
from pipeline.config_manager import ConfigManager
from pipeline.deduplication import DeduplicationFilter
from pipeline.model_registry import resolve_model, ModelRegistryError
from pipeline import dev_cache, SecurityRedactor

class JobOrchestrator:
    """
    Manages long-running pipeline distillations asynchronously preventing API thread locks.
    Maintains active volatile state strings for frontend react metric rendering.
    """
    def __init__(self, db_instance: LightRAGManager, system_status=None):
        self.db = db_instance
        self.conf = ConfigManager()
        self.active_jobs = {}
        self.cancel_flags = {}
        self.system_status = system_status
        if system_status is not None:
            system_status.set_on_ready(self._drain_queue)
        
    def get_status(self, job_id: str):
        return self.active_jobs.get(job_id, {"status": "not_found"})

    def request_cancel(self, job_id: str) -> bool:
        """Cancel a job whether queued or already running.

        For queued jobs: removes from the system queue and marks the active_jobs
        entry so the caller can observe the cancellation.
        For running jobs: sets the cancel flag checked by _execute_distillation.
        """
        if self.system_status is not None and self.system_status.dequeue_job(job_id):
            if job_id in self.active_jobs:
                self.active_jobs[job_id]["status"] = "Cancelled (was queued)"
                self.active_jobs[job_id]["progress"] = -1
            return True
        if job_id in self.active_jobs:
            self.cancel_flags[job_id] = True
            return True
        return False

    async def _execute_distillation(self, job_id: str, payload: dict, config: dict):
        # Capture the prior GITHUB_TOKEN before entering the try block so the
        # finally clause can restore it even if the try body raises before setting it.
        _prior_github_token = os.environ.get("GITHUB_TOKEN", "")
        try:
            repo = payload.get("repo")
            months = payload.get("months", 2)
            threshold = payload.get("threshold", 0.45)  # Overridden in Extract/Fuser classes

            # Resolve GitHub token from config, falling back to the environment.
            # The token is passed to crawl functions via GitHubClient, which reads
            # GITHUB_TOKEN from the environment. We set it only for this task's
            # duration and restore the prior value in finally to avoid leaking
            # across concurrent jobs.
            token = config.get("github_token", "") or settings.GITHUB_TOKEN
            if token:
                os.environ["GITHUB_TOKEN"] = token
                
            self.active_jobs[job_id]["status"] = f"Spidering {repo} ({months} months)"
            self.active_jobs[job_id]["progress"] = 10

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
                # Each crawler phase gets its own status callback that also
                # updates the numeric progress within that phase's range so the
                # UI shows live page-by-page advancement instead of a frozen bar.

                # Phase 1a: inline PR comments — 10 → 18 %
                _phase_a_page = [0]
                def _status_a(msg: str) -> None:
                    if job_id not in self.active_jobs:
                        return
                    _phase_a_page[0] += 1
                    # Each page nudges forward by 1 % up to the phase ceiling.
                    pct = min(10 + _phase_a_page[0], 18)
                    self.active_jobs[job_id]["status"] = msg
                    self.active_jobs[job_id]["progress"] = pct

                pr_data, updated_cursors = await asyncio.to_thread(
                    crawl_human_rejections, [repo], months, _status_a, check_cancel, cursors
                )
                current_conf["crawl_cursors"] = updated_cursors
                self.conf.save_config(current_conf)

                if check_cancel():
                    self.active_jobs[job_id]["status"] = "Pipeline Aborted via User Interrupt"
                    self.active_jobs[job_id]["progress"] = -1
                    return

                # Phase 1b: top-level PR reviews — 18 → 27 %
                _phase_b_page = [0]
                self.active_jobs[job_id]["status"] = f"Crawling PR reviews for {repo}..."
                self.active_jobs[job_id]["progress"] = 18

                def _status_b(msg: str) -> None:
                    if job_id not in self.active_jobs:
                        return
                    _phase_b_page[0] += 1
                    pct = min(18 + _phase_b_page[0], 27)
                    self.active_jobs[job_id]["status"] = msg
                    self.active_jobs[job_id]["progress"] = pct

                review_data, updated_cursors = await asyncio.to_thread(
                    crawl_pr_reviews, [repo], months, _status_b, check_cancel, updated_cursors
                )
                current_conf["crawl_cursors"] = updated_cursors
                self.conf.save_config(current_conf)

                pr_data.extend(review_data)

                # Phase 1c: closed issues — 27 → 35 %
                _phase_c_page = [0]
                self.active_jobs[job_id]["status"] = f"Crawling closed issues for {repo}..."
                self.active_jobs[job_id]["progress"] = 27

                def _status_c(msg: str) -> None:
                    if job_id not in self.active_jobs:
                        return
                    _phase_c_page[0] += 1
                    pct = min(27 + _phase_c_page[0], 35)
                    self.active_jobs[job_id]["status"] = msg
                    self.active_jobs[job_id]["progress"] = pct

                issue_data, updated_cursors = await asyncio.to_thread(
                    crawl_closed_issues, [repo], months, _status_c, check_cancel, updated_cursors
                )
                current_conf["crawl_cursors"] = updated_cursors
                self.conf.save_config(current_conf)

                pr_data.extend(issue_data)

                # Save to dev cache for future replay
                if pr_data:
                    await asyncio.to_thread(dev_cache.save_crawl, repo, pr_data)

            if check_cancel():
                self.active_jobs[job_id]["status"] = "Pipeline Aborted via User Interrupt"
                self.active_jobs[job_id]["progress"] = -1
                return

            # 2. Pre-process: hash dedup + PII redaction before any LLM call.
            # SecurityRedactor requires presidio; if unavailable comments are passed
            # through unredacted (acceptable for local/dev deployments without NLP deps).
            dedup = DeduplicationFilter()
            redactor = SecurityRedactor() if SecurityRedactor is not None else None
            cleaned_payloads = []
            for comment_body, diff_hunk in pr_data:
                if dedup.is_duplicate(comment_body):
                    continue
                safe_body = redactor.redact_text(comment_body) if redactor is not None else comment_body
                cleaned_payloads.append((safe_body, diff_hunk))

            # 3. Resolve active models from the registry. Skip unknown/disabled ids.
            registry = config.get("llm_models", [])
            active_ids = config.get("llm_models_active", [])
            provider_keys = config.get("provider_api_keys", {})

            if not active_ids:
                raise RuntimeError("No models selected: llm_models_active is empty")

            per_model_results: dict[str, dict] = {}
            self.active_jobs[job_id]["per_model_results"] = per_model_results
            total_attempted = len(cleaned_payloads)

            # Dev-mode replay: wipe existing rules ONCE up front, not per-model,
            # because each pass would otherwise wipe its predecessor's output.
            if use_cache and dev_cache.has_cache(repo):
                removed = await asyncio.to_thread(self.db.delete_repo_rules, repo)
                print(f"[Dev Mode] Cleared {removed} prior rules for {repo}")

            # Per-pass progress: divide the 35→95% range across N models.
            n_models = len(active_ids)
            pct_per_model = max(1, (95 - 35) // max(n_models, 1))

            # Sequential, not parallel: a single Ollama GPU can only serve one
            # model at a time. Running passes concurrently just queues requests
            # in Ollama without gaining throughput, while making per-model
            # status text and progress reporting harder to track.
            for idx, model_id in enumerate(active_ids):
                if check_cancel():
                    self.active_jobs[job_id]["status"] = "Pipeline Aborted via User Interrupt"
                    self.active_jobs[job_id]["progress"] = -1
                    return

                entry = next((e for e in registry if e.get("id") == model_id), None)
                label = entry.get("label", model_id) if entry else model_id

                pass_pct_start = 35 + idx * pct_per_model
                pass_pct_end = 35 + (idx + 1) * pct_per_model

                self.active_jobs[job_id]["status"] = f"[{idx+1}/{n_models}] Preflighting {label}..."
                self.active_jobs[job_id]["progress"] = pass_pct_start

                try:
                    resolved = resolve_model(model_id, registry, provider_keys)
                except ModelRegistryError as exc:
                    per_model_results[model_id] = {
                        "label": label,
                        "attempted": total_attempted,
                        "extracted": 0,
                        "failed_reason": str(exc),
                    }
                    continue

                llm_model = resolved["model"]
                llm_api_base = resolved["api_base"]
                llm_api_key = resolved["api_key"]
                # LiteLLM's native ollama/ provider rejects the /v1 suffix on api_base.
                if llm_model.startswith("ollama/") and llm_api_base.endswith("/v1"):
                    llm_api_base = llm_api_base[:-3]

                try:
                    await acompletion(
                        model=llm_model,
                        messages=[{"role": "user", "content": "ping"}],
                        api_base=llm_api_base,
                        api_key=llm_api_key,
                        max_tokens=1, temperature=0,
                    )
                except Exception as exc:
                    per_model_results[model_id] = {
                        "label": label,
                        "attempted": total_attempted,
                        "extracted": 0,
                        "failed_reason": f"Preflight failed: {exc}",
                    }
                    continue

                extractor = LargeLLMExtractor(
                    self.db,
                    model=llm_model,
                    api_base=llm_api_base,
                    api_key=llm_api_key,
                    model_id=model_id,
                    model_label=label,
                )

                # Wrap the extractor's 35→95 internal range into this pass's slice.
                # Closure binds (label, pass_pct_start, pass_pct_end) by default args
                # so each pass's callback has its own values.
                def _progress(msg: str, pct_inner: int,
                              _label=label, _start=pass_pct_start, _end=pass_pct_end,
                              _idx=idx, _n=n_models):
                    # extractor reports 35→95; map to the pass's allocated slice.
                    local_pct = max(0, min(60, pct_inner - 35))
                    scaled = _start + (local_pct / 60.0) * (_end - _start)
                    if job_id in self.active_jobs:
                        self.active_jobs[job_id]["status"] = f"[{_idx+1}/{_n}] {_label}: {msg}"
                        self.active_jobs[job_id]["progress"] = int(scaled)

                try:
                    extracted = await extractor.batch_extract(
                        cleaned_payloads, repo=repo, progress_callback=_progress
                    )
                    per_model_results[model_id] = {
                        "label": label,
                        "attempted": total_attempted,
                        "extracted": len(extracted) if extracted else 0,
                    }
                except Exception as exc:
                    per_model_results[model_id] = {
                        "label": label,
                        "attempted": total_attempted,
                        "extracted": 0,
                        "failed_reason": str(exc),
                    }
                    continue

            # Final summary.
            total_extracted = sum(r.get("extracted", 0) for r in per_model_results.values())
            summary_parts = [
                f"{r['label']}: {r['extracted']}" + (" (FAILED)" if r.get("failed_reason") else "")
                for r in per_model_results.values()
            ]
            self.active_jobs[job_id]["status"] = (
                f"Completed: {total_extracted} extractions across {n_models} model(s) — "
                + ", ".join(summary_parts)
            )
            self.active_jobs[job_id]["progress"] = 100

        except Exception as e:
            logger.exception("Pipeline job %s failed", job_id)
            self.active_jobs[job_id]["status"] = f"FAILED: {str(e)}"
            self.active_jobs[job_id]["progress"] = -1
        finally:
            # Restore GITHUB_TOKEN to its prior value so concurrent jobs are not affected.
            if _prior_github_token:
                os.environ["GITHUB_TOKEN"] = _prior_github_token
            elif "GITHUB_TOKEN" in os.environ:
                del os.environ["GITHUB_TOKEN"]

    def trigger_job(self, payload: dict, current_config: dict) -> str:
        """Kick off a distillation job, or enqueue it if the embedding model is not ready.

        When system_status is None or reports ready, the job starts immediately.
        Otherwise it is added to the system queue and will be drained once the
        embedding model finishes loading (_drain_queue is registered as the
        on-ready callback in __init__).
        """
        job_id = f"job-{int(time.time())}"
        trigger_source = payload.get("trigger_source", "manual")
        self.cancel_flags[job_id] = False

        if self.system_status is None or self.system_status.is_ready():
            self.active_jobs[job_id] = {
                "status": "Initializing Engine...",
                "progress": 0,
                "trigger_source": trigger_source,
            }
            asyncio.create_task(self._execute_distillation(job_id, payload, current_config))
        else:
            self.active_jobs[job_id] = {
                "status": "Queued — waiting for embedding model",
                "progress": 0,
                "trigger_source": trigger_source,
            }
            self.system_status.enqueue_job(
                job_id, payload, current_config,
                reason="embedding model not ready",
            )

        return job_id

    def _drain_queue(self) -> None:
        """Start all queued jobs now that the embedding model is ready.

        Registered as the on-ready callback via system_status.set_on_ready().
        Called by SystemStatus._schedule_on_ready() on the FastAPI event loop
        via call_soon_threadsafe, so asyncio.create_task() is safe here.
        """
        if self.system_status is None:
            return
        for job_id, payload, config in self.system_status.drain_queue():
            if job_id not in self.active_jobs:
                continue
            self.active_jobs[job_id]["status"] = "Starting (was queued)"
            asyncio.create_task(self._execute_distillation(job_id, payload, config))
