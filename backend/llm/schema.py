from pydantic import BaseModel
from typing import List, Optional, Literal

class SourceInfo(BaseModel):
    repository: str
    pr_number: int
    commit_hash: Optional[str] = None

class Metadata(BaseModel):
    status: Literal["active", "superseded", "archived", "needs_review"] = "active"
    labels: List[str]
    confidence_score: float

class Scoping(BaseModel):
    language: List[str]
    path_patterns: List[str]

class ContentExamples(BaseModel):
    bad: str
    good: str

class RuleContent(BaseModel):
    title: str
    description: str
    condition: str
    enforcement_prompt: str
    examples: ContentExamples

class MigrationContext(BaseModel):
    is_migrating: bool
    replaced_by_path: Optional[str] = None

class ExtractedRuleSchema(BaseModel):
    """
    Strict validation schema for LLM Extractions (Stage 3 & 4)
    Prevents Prompt Injections from corrupting the rule DB.
    """
    rule_id: str
    source: SourceInfo
    metadata: Metadata
    scoping: Scoping
    content: RuleContent
    migration_context: Optional[MigrationContext] = None
