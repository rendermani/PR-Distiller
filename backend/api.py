import hashlib
import hmac
import json as _json
import logging
import os
import sys
import time
import uvicorn
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from db.lightrag_manager import LightRAGManager
from pipeline.config_manager import ConfigManager
from pipeline.job_orchestrator import JobOrchestrator
from pipeline import dev_cache
import settings

logger = logging.getLogger(__name__)

app = FastAPI(title="PR-Distiller Knowledge API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to the persistent ChromaDB cluster safely
db = LightRAGManager()
conf_manager = ConfigManager()
orchestrator = JobOrchestrator(db)

# ---------------------------------------------------------------------------
# Webhook helpers
# ---------------------------------------------------------------------------

WEBHOOK_MIN_INTERVAL_SECONDS = settings.WEBHOOK_MIN_INTERVAL


def verify_github_signature(payload_body: bytes, signature: str, secret: str) -> bool:
    """Return True iff the HMAC-SHA256 signature matches the payload and secret."""
    expected = "sha256=" + hmac.new(secret.encode(), payload_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def is_merged_pull_request(payload: dict) -> bool:
    """Return True only for pull_request events that were closed via merge."""
    if "pull_request" not in payload:
        return False
    if payload.get("action") != "closed":
        return False
    return bool(payload["pull_request"].get("merged"))


class WebhookRateLimiter:
    """
    Enforces a minimum interval between webhook-triggered jobs per repository.

    Tracks trigger counts and last trigger timestamps for the stats endpoint.
    """

    def __init__(self, min_interval_seconds: int = WEBHOOK_MIN_INTERVAL_SECONDS):
        self._min_interval = min_interval_seconds
        self._last_trigger_times: dict[str, float] = {}
        self._trigger_counts: dict[str, int] = {}

    def is_allowed(self, repo: str) -> bool:
        """Return True if a trigger for *repo* is not within the minimum interval."""
        last = self._last_trigger_times.get(repo)
        if last is None:
            return True
        return (time.time() - last) >= self._min_interval

    def record_trigger(self, repo: str) -> None:
        """Mark a successful trigger for *repo*."""
        self._last_trigger_times[repo] = time.time()
        self._trigger_counts[repo] = self._trigger_counts.get(repo, 0) + 1

    def get_stats(self) -> dict:
        """Return per-repo trigger counts and last trigger timestamps."""
        return {
            repo: {
                "trigger_count": self._trigger_counts[repo],
                "last_trigger_time": self._last_trigger_times[repo],
            }
            for repo in self._trigger_counts
        }


# Module-level rate limiter — one instance shared across requests.
_webhook_rate_limiter = WebhookRateLimiter()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    code_diff: str
    top_k: int = 5
    categories: list[str] | None = None
    file_path: str | None = None

class JobRequest(BaseModel):
    repo: str
    months: int = 2
    threshold: float = 0.45
    use_cache: bool = False

def _redact_sensitive_fields(config: dict) -> dict:
    """Return a copy of config with sensitive secrets replaced by '***' or ''."""
    redacted = dict(config)
    for field in ("github_token", "llm_api_key"):
        redacted[field] = "***" if config.get(field) else ""
    return redacted


@app.get("/api/config")
def get_config():
    """Serves the Unified JSON configurations to the Next.js UI Settings panel."""
    return _redact_sensitive_fields(conf_manager.load_config())

@app.post("/api/config")
def update_config(payload: dict):
    """Mutates global architecture settings from UI slider payloads natively."""
    return conf_manager.save_config(payload)

@app.get("/api/cache/{repo:path}")
def get_cache_info(repo: str):
    """Returns dev cache status for a repo."""
    info = dev_cache.cache_info(repo)
    if not info:
        return {"cached": False}
    return {"cached": True, **info}

@app.post("/api/jobs/start")
async def start_pipeline(req: JobRequest):
    """Hits the explicit trigger allocating asynchronous DGX mapping routines."""
    current_config = conf_manager.load_config()
    job_id = orchestrator.trigger_job(req.model_dump(), current_config)
    return {"job_id": job_id, "status": "started"}

@app.post("/api/jobs/cancel/{job_id}")
async def cancel_pipeline(job_id: str):
    """Hits the strict interrupt routines to break threaded background nodes safely."""
    success = orchestrator.request_cancel(job_id)
    return {"status": "cancelled" if success else "not_found"}

@app.get("/api/jobs/status/{job_id}")
def check_pipeline_status(job_id: str):
    return orchestrator.get_status(job_id)

@app.get("/api/rules")
def get_all_rules(repo: str = None):
    """
    Returns all extracted rules isolated strictly for the Contextual Control HUD.
    """
    if repo:
        coll = db.collection.get(where={"repo": repo})
    else:
        coll = db.collection.get()
    
    # Check if empty
    if not coll.get("ids"):
        return {"rules": []}
        
    rules = []
    docs = coll["documents"]
    metadatas = coll["metadatas"]
    ids = coll["ids"]
    
    for i in range(len(ids)):
        rules.append({
            "id": ids[i],
            "document": docs[i],
            "metadata": metadatas[i] if metadatas and i < len(metadatas) else {}
        })
        
    # Sort DESC by triage occurrence_count
    rules.sort(key=lambda x: x["metadata"].get("occurrence_count", 0), reverse=True)
    
    return {"rules": rules}


@app.post("/api/mcp/query")
def mcp_contextual_query(request: QueryRequest):
    """
    Called by the Typescript MCP Server when Cursor asks for best practices on a given diff.
    """
    try:
        results = db.get_contextual_rules(
            request.code_diff,
            request.top_k,
            request.categories,
            file_path=request.file_path,
        )

        # Format cleanly for typescript consumption
        matched_rules = []
        if isinstance(results, dict) and results.get("ids") and len(results["ids"][0]) > 0:
            docs = results["documents"][0]
            metadatas = results["metadatas"][0] if results.get("metadatas") else []
            for j in range(len(docs)):
                matched_rules.append({
                    "document": docs[j],
                    "metadata": metadatas[j] if j < len(metadatas) else {}
                })
        
        return {"matched_rules": matched_rules}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/rules/{rule_id}/history")
def get_rule_history(rule_id: str):
    """Returns the merge lineage for a rule: all rules that were merged into it."""
    history = db.get_rule_history(rule_id)
    return {"history": history}


@app.post("/api/rules/{rule_id}/approve")
def approve_rule(rule_id: str):
    """
    Used by the dashboard to flip an intercepted rule from needs_review -> active.
    """
    db.approve_rule(rule_id)
    return {"status": "success", "rule_id": rule_id, "state": "approved"}

@app.post("/api/rules/{rule_id}/block")
def block_rule(rule_id: str):
    """Blocks a rule so it is excluded from exports and MCP queries."""
    db.block_rule(rule_id)
    return {"status": "success", "rule_id": rule_id, "state": "blocked"}

@app.delete("/api/rules/{rule_id}")
def reject_rule(rule_id: str):
    """Permanently wipes obsolete extractions entirely dropping them out of the Vector map."""
    db.collection.delete(ids=[rule_id])
    return {"status": "deleted"}

class AddRuleRequest(BaseModel):
    repo: str
    title: str
    description: str
    enforcement: str


class FeedbackRequest(BaseModel):
    action: Literal["applied", "dismissed"]


@app.post("/api/rules/{rule_id}/feedback")
def record_rule_feedback(rule_id: str, req: FeedbackRequest):
    """
    Records whether a rule suggestion was applied or dismissed by the IDE.
    Updates times_served, times_applied/times_dismissed counters.
    Auto-promotes 'needs_review' rules that reach >=5 serves with >70% acceptance.
    """
    try:
        db.record_feedback(rule_id, req.action)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"status": "recorded", "rule_id": rule_id, "action": req.action}


@app.get("/api/rules/{rule_id}/effectiveness")
def get_rule_effectiveness(rule_id: str):
    """Returns per-rule effectiveness stats: serves, applied, dismissed, acceptance_rate."""
    try:
        return db.get_rule_effectiveness(rule_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/api/stats/effectiveness")
def get_effectiveness_stats():
    """Returns aggregate effectiveness stats across all rules for the dashboard."""
    return db.get_all_effectiveness_stats()


@app.post("/api/rules")
def add_rule(req: AddRuleRequest):
    """Manually adds a new rule from the dashboard."""
    rule_id = db.add_rule(req.repo, req.title, req.description, req.enforcement)
    return {"status": "created", "rule_id": rule_id}

@app.post("/api/admin/reindex")
def reindex_embeddings():
    """Re-embed all rules using the current EMBEDDING_MODEL.

    Must be called after changing the EMBEDDING_MODEL environment variable to
    prevent dimension mismatches between old and new vectors.
    """
    reindexed_count = db.reindex_all_rules()
    return {"status": "success", "reindexed_count": reindexed_count}


@app.get("/api/export")
def export_rules(repo: str, type: str = "prompt", arch: str = "openai"):
    """Synthesizes isolated repository tuples securely returning explicit agent formats."""
    try:
        from pipeline.skill_synthesizer import SkillSynthesizer
        
        results = db.collection.get(where={"repo": repo})
        active_docs = []
        
        if results and results.get("documents"):
            for doc, meta in zip(results["documents"], results["metadatas"]):
                if meta and meta.get("status") == "active":
                    active_docs.append(doc)
                    
        export_content = SkillSynthesizer.export(active_docs, repo, type, arch)
        return {"content": export_content}
    except Exception as e:
        logger.exception("Export pipeline failed for repo=%s type=%s arch=%s", repo, type, arch)
        raise HTTPException(status_code=500, detail="Export pipeline failed. See server logs for details.")

# ---------------------------------------------------------------------------
# Webhook endpoints
# ---------------------------------------------------------------------------

@app.post("/api/webhooks/github")
async def receive_github_webhook(request: Request):
    """
    Receives GitHub webhook payloads and triggers incremental extraction for merged PRs.

    Validates X-Hub-Signature-256 when GITHUB_WEBHOOK_SECRET is set.
    Rate-limits to one trigger per repo per 60 seconds.
    """
    payload_body = await request.body()
    event_type = request.headers.get("X-GitHub-Event", "")

    webhook_secret = os.environ.get("GITHUB_WEBHOOK_SECRET", "")
    if webhook_secret:
        signature = request.headers.get("X-Hub-Signature-256", "")
        if not signature:
            raise HTTPException(status_code=401, detail="Missing X-Hub-Signature-256 header")
        if not verify_github_signature(payload_body, signature, webhook_secret):
            raise HTTPException(status_code=401, detail="Invalid webhook signature")

    if event_type != "pull_request":
        return {"status": "ignored", "reason": "not a pull_request event"}

    payload = _json.loads(payload_body)
    if not is_merged_pull_request(payload):
        return {"status": "ignored", "reason": "PR not merged"}

    repo = payload["repository"]["full_name"]

    if not _webhook_rate_limiter.is_allowed(repo):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limited: minimum {WEBHOOK_MIN_INTERVAL_SECONDS}s between triggers for {repo}",
        )

    _webhook_rate_limiter.record_trigger(repo)

    current_config = conf_manager.load_config()
    job_payload = {"repo": repo, "trigger_source": "webhook"}
    job_id = orchestrator.trigger_job(job_payload, current_config)

    return {"status": "triggered", "job_id": job_id, "repo": repo}


@app.get("/api/webhooks/stats")
def get_webhook_stats():
    """Returns webhook trigger counts and last trigger time per repo."""
    return {"repos": _webhook_rate_limiter.get_stats()}


if __name__ == "__main__":
    print(f"[*] Starting PR-Distiller Edge API on {settings.API_HOST}:{settings.API_PORT}...")
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
