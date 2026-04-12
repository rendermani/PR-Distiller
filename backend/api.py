import os
import sys
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from db.lightrag_manager import LightRAGManager
from pipeline.config_manager import ConfigManager
from pipeline.job_orchestrator import JobOrchestrator
from pipeline import dev_cache

app = FastAPI(title="PR-Distiller Knowledge API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to the persistent ChromaDB cluster safely
db = LightRAGManager()
conf_manager = ConfigManager()
orchestrator = JobOrchestrator(db)

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

@app.get("/api/config")
def get_config():
    """Serves the Unified JSON configurations to the Next.js UI Settings panel."""
    return conf_manager.load_config()

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
    job_id = orchestrator.trigger_job(req.dict(), current_config)
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

@app.post("/api/rules")
def add_rule(req: AddRuleRequest):
    """Manually adds a new rule from the dashboard."""
    rule_id = db.add_rule(req.repo, req.title, req.description, req.enforcement)
    return {"status": "created", "rule_id": rule_id}

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
        import traceback
        return {"content": f"PYTHON PIPELINE CRASH:\\n{str(e)}\\n\\n{traceback.format_exc()}"}

if __name__ == "__main__":
    print("[*] Starting PR-Distiller Edge API on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
