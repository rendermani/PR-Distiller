import asyncio
import hashlib
import hmac
import json as _json
import logging
import os
import shutil
import sys
import threading
import time
import uvicorn
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from auth import ensure_api_token
from db.lightrag_manager import LightRAGManager
from db_proxy import LazyDbProxy
from embedding_loader import capture_hf_progress
from pipeline.config_manager import ConfigManager
from pipeline.job_orchestrator import JobOrchestrator
from pipeline import dev_cache
import settings
from system_status import SYSTEM_STATUS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding retry helpers
# ---------------------------------------------------------------------------

_HF_CACHE_DIR = os.environ.get("HF_HOME", "/cache/huggingface")


def _wipe_hf_cache() -> None:
    """Best-effort wipe of the HuggingFace cache directory."""
    if os.path.isdir(_HF_CACHE_DIR):
        shutil.rmtree(_HF_CACHE_DIR, ignore_errors=True)


_loader_thread_lock = threading.Lock()
_loader_thread: threading.Thread | None = None


def _runner() -> None:
    """Background work executed by the embedding-loader thread.

    Transitions SYSTEM_STATUS through downloading -> loading -> ready.
    On failure, transitions to error with the exception class and message.
    Promoted to module level so it is patchable in tests and so the Thread
    target reference remains valid across calls.
    """
    SYSTEM_STATUS.update(
        state="downloading",
        model_name=settings.EMBEDDING_MODEL,
        error=None,
    )
    try:
        with capture_hf_progress():
            real_db = LightRAGManager()
        SYSTEM_STATUS.update(state="loading")
        # Defensive guard: if a concurrent caller somehow already bound the
        # proxy (should not happen given the lock in
        # _start_background_embedding_load, but protects manual bind() calls
        # made from tests or future code paths), absorb the error rather than
        # transitioning to state=error despite the system being operational.
        try:
            db.bind(real_db)
        except RuntimeError:
            pass
        SYSTEM_STATUS.update(state="ready")
    except Exception as exc:
        SYSTEM_STATUS.update(state="error", error=f"{type(exc).__name__}: {exc}")
        logger.exception("Embedding model load failed")


def _start_background_embedding_load() -> None:
    """Spawn the bg thread that downloads + loads the embedding model.

    Idempotent: if a loader thread is still alive, this is a no-op.
    The thread updates SYSTEM_STATUS state through downloading -> loading -> ready.
    On failure it sets state="error" with the exception class+message.
    Returns immediately; the caller must not await or join the thread.
    """
    global _loader_thread
    with _loader_thread_lock:
        if _loader_thread is not None and _loader_thread.is_alive():
            return
        _loader_thread = threading.Thread(
            target=_runner, daemon=True, name="embedding-loader"
        )
        _loader_thread.start()


def _embedding_load_starter() -> None:
    """Retry entry point: spawns the background embedding-load thread."""
    _start_background_embedding_load()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: wire the event loop, resolve auth token, start loader."""
    SYSTEM_STATUS.attach_loop(asyncio.get_running_loop())
    settings.API_AUTH_TOKEN = ensure_api_token()
    _start_background_embedding_load()
    yield


app = FastAPI(title="PR-Distiller Knowledge API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_api_token(request: Request) -> None:
    """Optional Bearer-token gate for mutating endpoints.

    Behaviour:
    - When `settings.API_AUTH_TOKEN` is empty, the gate is a no-op (preserves
      the local-dev default of an open API).
    - When set, requests must carry `Authorization: Bearer <token>` matching it
      via constant-time comparison; otherwise the gate raises 401.

    Public-by-design endpoints (health checks, webhooks, MCP query) do not
    apply this dependency.
    """
    expected = settings.API_AUTH_TOKEN
    if not expected:
        return
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    presented = header[len("Bearer "):]
    if not hmac.compare_digest(presented, expected):
        raise HTTPException(status_code=401, detail="Invalid bearer token")

# db is a lazy proxy; the real LightRAGManager is loaded in the background thread.
db = LazyDbProxy()
conf_manager = ConfigManager()
orchestrator = JobOrchestrator(db, system_status=SYSTEM_STATUS)

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


if not settings.GITHUB_WEBHOOK_SECRET:
    logger.warning(
        "GITHUB_WEBHOOK_SECRET is not set. The /api/webhooks/github endpoint "
        "will accept unsigned payloads (dev mode). Set the env var to enforce "
        "HMAC verification before exposing this server."
    )


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


class ConfigUpdate(BaseModel):
    """Typed payload for POST /api/config.

    All fields optional so callers can send partial updates. Unknown top-level
    keys are rejected to prevent arbitrary writes into the persisted config.
    """
    model_config = {"extra": "forbid"}

    github_token: str | None = None
    huggingface_token: str | None = None
    github_webhook_secret: str | None = None
    llm_models: list[dict] | None = None
    llm_models_active: list[str] | None = None
    provider_api_keys: dict[str, str] | None = None
    embedding_model: str | None = None
    repos: dict | None = None
    provider_models: dict | None = None
    crawl_cursors: dict | None = None

def _redact_sensitive_fields(config: dict) -> dict:
    """Return a copy of config with sensitive secrets replaced by '***' or ''."""
    redacted = dict(config)
    for field in ("github_token", "github_webhook_secret", "huggingface_token"):
        redacted[field] = "***" if config.get(field) else ""
    provider_keys = config.get("provider_api_keys", {})
    redacted["provider_api_keys"] = {p: "***" if k else "" for p, k in provider_keys.items()}
    # Mask per-entry api_key_override values without dropping other fields.
    llm_models = config.get("llm_models", [])
    redacted["llm_models"] = [
        {**entry, "api_key_override": "***" if entry.get("api_key_override") else ""}
        for entry in llm_models
    ]
    return redacted


class TokenValidateRequest(BaseModel):
    token: str | None = None


@app.post("/api/github/validate")
def validate_github_token(req: TokenValidateRequest):
    """Validate a GitHub token against the GitHub API.

    If no token is provided in the request, falls back to the currently saved one.
    Never echoes the token back. Returns {valid, login?, scopes?, error?}.
    """
    from pipeline.github_client import GitHubClient

    token = (req.token or "").strip()
    if not token:
        token = conf_manager.load_config().get("github_token", "") or settings.GITHUB_TOKEN
    return GitHubClient(token=token or None).validate_token()


@app.get("/api/health")
def liveness_probe():
    """Cheap, auth-free, dependency-free liveness ping.

    Used by docker-compose healthcheck instead of `/api/rules`, which spins up
    ChromaDB on first request and can race the start_period during cold-start.
    """
    return {"status": "ok"}


@app.get("/api/system/status")
def get_system_status():
    """Auth-free snapshot of embedding state, download progress, and queued jobs.

    Intentionally has no ``Depends(require_api_token)`` — the frontend uses this
    for initial load and as a SSE fallback before a token is available.  Auth is
    applied per-route in this app; omitting the dependency here is sufficient.
    """
    return SYSTEM_STATUS.snapshot()


@app.get("/api/system/events")
async def system_events(request: Request):
    """SSE stream of SystemStatus mutations.

    Sends the current snapshot immediately on subscribe, then one event per
    mutation.  A keepalive comment every 15 s prevents proxy idle-timeouts.
    Auth-free, mirrors /api/system/status.
    """
    async def event_stream():
        q = SYSTEM_STATUS.subscribe()
        try:
            yield f"data: {_json.dumps(SYSTEM_STATUS.snapshot())}\n\n"
            while True:
                if await request.is_disconnected():
                    return
                try:
                    update = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield f"data: {_json.dumps(update)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            SYSTEM_STATUS.unsubscribe(q)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/system/embedding/retry", dependencies=[Depends(require_api_token)])
def retry_embedding(clean: bool = False):
    """Trigger a fresh embedding download. Optionally wipes the HF cache first."""
    if clean:
        _wipe_hf_cache()
    SYSTEM_STATUS.update(state="idle", error=None, bytes_downloaded=0, bytes_total=0)
    _embedding_load_starter()
    return {"status": "retry-started"}


@app.get("/api/health/llm")
def check_llm_health():
    """Reach the active LLM(s) to tell the UI whether inference will work.

    With multi-model support, this probes the api_base of the first active
    model. The UI shows a single banner; per-model status is implicit in the
    job results.
    Returns {reachable, api_base, provider_hint, error?}.
    """
    import requests as _requests

    cfg = conf_manager.load_config()
    active_ids = cfg.get("llm_models_active", [])
    if not active_ids:
        return {"reachable": False, "api_base": "", "error": "No active LLM models configured."}

    # Only the first active model is probed — the UI renders a single health banner.
    # Per-model status surfaces through job results, not the health endpoint.
    registry = cfg.get("llm_models", [])
    entry = next((e for e in registry if e.get("id") == active_ids[0]), None)
    if entry is None:
        return {
            "reachable": False, "api_base": "",
            "error": f"Active model id {active_ids[0]!r} not found in registry.",
        }

    api_base = (entry.get("api_base") or "").rstrip("/")
    if not api_base:
        return {"reachable": False, "api_base": "", "error": "Active model has no api_base configured."}

    # Heuristic: Ollama exposes /api/tags outside the OpenAI-compat prefix;
    # fall back to /models for OpenAI-compatible servers.
    probes = []
    if api_base.endswith("/v1"):
        probes.append(api_base[:-3] + "/api/tags")
    probes.append(api_base + "/models")

    # Use model-string prefix rather than port heuristic — robust to custom ports.
    hint = "ollama" if entry.get("model", "").startswith("ollama/") else "openai-compatible"
    for url in probes:
        try:
            r = _requests.get(url, timeout=3)
            if r.status_code < 500:
                return {"reachable": True, "api_base": api_base, "provider_hint": hint}
        except _requests.RequestException:
            continue
    return {
        "reachable": False,
        "api_base": api_base,
        "provider_hint": hint,
        "error": f"Could not reach {api_base}. Is the LLM server running?",
    }


@app.get("/api/config", dependencies=[Depends(require_api_token)])
def get_config(request: Request):
    """Serves the Unified JSON configurations to the Next.js UI Settings panel.

    Augments the redacted config with `env_overrides` and a computed
    `webhook_url` so the Vault page can render env-disabled inputs and
    the GitHub-webhook copy field.
    """
    payload = _redact_sensitive_fields(conf_manager.load_config())
    payload["env_overrides"] = conf_manager.env_overrides()

    public_base = settings.WEBHOOK_PUBLIC_URL or f"{request.url.scheme}://{request.headers.get('host', '')}"
    payload["webhook_url"] = f"{public_base.rstrip('/')}/api/webhooks/github"
    return payload

@app.post(
    "/api/config",
    dependencies=[Depends(require_api_token)],
    responses={400: {"description": "Malformed llm_models entry — see detail."}},
)
def update_config(payload: ConfigUpdate):
    """Updates the persisted configuration. All fields are optional; omitted
    fields are left unchanged.

    Drops the redaction sentinel '***' so the UI can round-trip GET→POST
    without overwriting real secrets with the placeholder. For per-entry
    `api_key_override` set to '***', the stored plaintext is restored from
    the current config before saving so masked round-trips don't clear keys.

    Each incoming llm_models entry is validated via the model_registry
    validator; malformed entries return HTTP 400 with the validator's error.
    """
    from pipeline.model_registry import validate_registry, ModelRegistryError

    data = payload.model_dump(exclude_unset=True)

    if data.get("github_token") == "***":
        data.pop("github_token")
    if data.get("huggingface_token") == "***":
        data.pop("huggingface_token")
    if data.get("github_webhook_secret") == "***":
        data.pop("github_webhook_secret")

    if "provider_api_keys" in data and isinstance(data["provider_api_keys"], dict):
        data["provider_api_keys"] = {
            p: k for p, k in data["provider_api_keys"].items() if k != "***"
        }

    if "llm_models" in data:
        try:
            # validate_registry also rejects duplicate ids, which previously
            # produced dead config: resolve_model takes the first match, so the
            # shadowed entry was unreachable while the UI still displayed it.
            validate_registry(data["llm_models"])
        except ModelRegistryError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # Preserve masked api_key_override values: when the UI re-POSTs a
        # masked entry, look up the currently stored plaintext by id so the
        # secret isn't dropped. Untouched entries keep their override.
        # NOTE: non-atomic read-then-write — concurrent POSTs can race here.
        # ConfigManager.save_config holds a file lock around the write, but
        # this read is outside it. A merge-on-write helper inside
        # ConfigManager is the correct long-term fix.
        current = conf_manager.load_config()
        # .get("id") because a stored entry may predate validation (hand-edited
        # config.json); indexing with e["id"] made one such entry raise KeyError
        # and turn any masked-secret POST into a 500.
        current_by_id = {
            e["id"]: e
            for e in current.get("llm_models", [])
            if isinstance(e, dict) and e.get("id")
        }
        for entry in data["llm_models"]:
            if entry.get("api_key_override") == "***":
                entry["api_key_override"] = current_by_id.get(entry["id"], {}).get("api_key_override", "")

    # Every active id must name a registry entry. Without this the request
    # succeeded and the failure resurfaced at job runtime as a per-model
    # preflight error, far from the change that caused it. Validate against the
    # incoming registry when one is supplied, else the stored one.
    if "llm_models_active" in data:
        if "llm_models" in data:
            known_ids = {e["id"] for e in data["llm_models"]}
        else:
            stored = conf_manager.load_config().get("llm_models", [])
            known_ids = {e["id"] for e in stored if isinstance(e, dict) and e.get("id")}
        unknown = [mid for mid in data["llm_models_active"] if mid not in known_ids]
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"llm_models_active names unknown model id(s): {', '.join(unknown)}. "
                    f"Known ids: {', '.join(sorted(known_ids)) or '(none)'}"
                ),
            )

    return conf_manager.save_config(data)

@app.get("/api/cache/{repo:path}")
def get_cache_info(repo: str):
    """Returns dev cache status for a repo."""
    info = dev_cache.cache_info(repo)
    if not info:
        return {"cached": False}
    return {"cached": True, **info}

@app.post("/api/jobs/start", dependencies=[Depends(require_api_token)])
async def start_pipeline(req: JobRequest):
    """Hits the explicit trigger allocating asynchronous DGX mapping routines."""
    current_config = conf_manager.load_config()
    job_id = orchestrator.trigger_job(req.model_dump(), current_config)
    return {"job_id": job_id, "status": "started"}

@app.post("/api/jobs/cancel/{job_id}", dependencies=[Depends(require_api_token)])
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


@app.post("/api/rules/{rule_id}/approve", dependencies=[Depends(require_api_token)])
def approve_rule(rule_id: str):
    """
    Used by the dashboard to flip an intercepted rule from needs_review -> active.
    """
    db.approve_rule(rule_id)
    return {"status": "success", "rule_id": rule_id, "state": "approved"}

@app.post("/api/rules/{rule_id}/block", dependencies=[Depends(require_api_token)])
def block_rule(rule_id: str):
    """Blocks a rule so it is excluded from exports and MCP queries."""
    db.block_rule(rule_id)
    return {"status": "success", "rule_id": rule_id, "state": "blocked"}

@app.delete("/api/rules/{rule_id}", dependencies=[Depends(require_api_token)])
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


@app.post("/api/rules/{rule_id}/feedback", dependencies=[Depends(require_api_token)])
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


@app.post("/api/rules", dependencies=[Depends(require_api_token)])
def add_rule(req: AddRuleRequest):
    """Manually adds a new rule from the dashboard."""
    rule_id = db.add_rule(req.repo, req.title, req.description, req.enforcement)
    return {"status": "created", "rule_id": rule_id}

@app.delete("/api/rules", dependencies=[Depends(require_api_token)])
def delete_rules_for_repo(repo: str):
    """Delete all rules for a repo. Useful for clean re-extraction runs."""
    removed = db.delete_repo_rules(repo)
    return {"status": "deleted", "repo": repo, "removed": removed}


@app.post("/api/admin/reindex", dependencies=[Depends(require_api_token)])
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
    # The lifespan hook resolves the auth token when uvicorn runs the ASGI app.
    # We still resolve it here so the print below reflects the actual token length
    # before uvicorn's startup sequence begins.
    settings.API_AUTH_TOKEN = ensure_api_token()
    print(
        f"[*] API auth token resolved (len={len(settings.API_AUTH_TOKEN)}). "
        f"Persisted at {settings.DATA_DIR}/.api_token (chmod 600)."
    )
    print(f"[*] Starting PR-Distiller Edge API on {settings.API_HOST}:{settings.API_PORT}...")
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
