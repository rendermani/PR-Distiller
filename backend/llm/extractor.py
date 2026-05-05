import os
import json
import asyncio
from litellm import completion, acompletion
from db.lightrag_manager import LightRAGManager
from pipeline.semantic_fuser import SemanticFuser
import settings


def _qwen_extra_body(model: str) -> dict:
    """Return Qwen3-specific completion kwargs, or {} for other providers.

    `enable_thinking=False` is recognised by Qwen3's chat template only;
    OpenAI/Anthropic/Gemini reject unknown extra_body keys with 4xx, so the
    flag must be gated on the model name.
    """
    if "qwen" in (model or "").lower():
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    return {}


class LargeLLMExtractor:
    CLASSIFY_PROMPT = (
        "You classify PR review comments. Respond with ONLY one word:\n"
        '- "EXTRACT" if the comment contains an actionable, generalizable coding rule '
        "(code quality, architecture, security, performance, correctness)\n"
        '- "SKIP" if it\'s a translation fix, typo, question, discussion, one-time fix, or not about code\n\n'
        "Respond with exactly one word: EXTRACT or SKIP"
    )

    def __init__(
        self,
        db_manager: LightRAGManager,
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
    ):
        # Per-instance config; env vars are only consulted when the caller
        # passes None, preserving the orchestrator's old contract while
        # eliminating cross-job leakage when explicit values are supplied.
        self.model = model or os.environ.get("EXTRACTOR_MODEL", settings.LLM_MODEL)
        self.api_base = api_base or os.environ.get("EXTRACTOR_API_BASE", settings.LLM_API_BASE)
        self.api_key = (
            api_key
            or os.environ.get("EXTRACTOR_API_KEY", settings.LLM_API_KEY)
            or "unused"
        )
        self.db = db_manager
        self.fuser = SemanticFuser(
            db_manager, model=self.model, api_base=self.api_base, api_key=self.api_key
        )
        
    # All valid category values for issue #7.
    VALID_CATEGORIES = frozenset({
        "security", "performance", "testing", "code-style", "architecture", "correctness"
    })

    # Confidence threshold above which a rule is auto-approved (issue #6).
    AUTO_APPROVE_CONFIDENCE_THRESHOLD = 0.8

    SYSTEM_PROMPT = (
        "You extract reusable coding rules from PR review feedback.\n\n"
        "ONLY extract rules that are:\n"
        "- About CODE QUALITY, ARCHITECTURE, SECURITY, PERFORMANCE, or CORRECTNESS\n"
        "- Generalizable — would apply to future code in this project, not just a one-off fix\n"
        "- Actionable — an AI code assistant could enforce this rule\n\n"
        "DO NOT extract rules about:\n"
        "- Translations, documentation wording, typos, or formatting\n"
        "- One-time version bumps, link fixes, or config value changes\n"
        "- Comments that are questions or discussions, not corrections\n\n"
        "If the review comment is NOT a generalizable coding rule, respond with exactly: {\"skip\": true}\n\n"
        "Otherwise output ONLY valid JSON:\n"
        "{\"rule_id\": \"descriptive-kebab-case-slug\", "
        "\"confidence\": 0.0, "
        "\"category\": \"security|performance|testing|code-style|architecture|correctness\", "
        "\"metadata\": {\"status\": \"active\"}, "
        "\"scoping\": {\"path_patterns\": [\"e.g., **/auth/*.py\"]}, "
        "\"content\": {\"title\": \"short imperative title\", "
        "\"description\": \"when and why this rule matters\", "
        "\"enforcement_prompt\": \"what an AI assistant must do or avoid\", "
        "\"code_examples\": {\"bad_code\": \"...\", \"good_code\": \"...\"}}}\n\n"
        "Fields:\n"
        "- rule_id: a unique, descriptive kebab-case slug summarizing the rule (e.g. \"use-parameterized-sql-queries\", \"avoid-mutable-default-args\"). NEVER use generic placeholders like \"uuid\" or \"unique_rule_id_1\".\n"
        "- confidence: float 0.0-1.0 — how confident you are this is a real, generalizable rule\n"
        "- category: exactly one of: security, performance, testing, code-style, architecture, correctness"
    )

    @staticmethod
    def _build_user_prompt(pr_comment: str, modified_ast_code: str) -> str:
        """Builds the user message sent to the LLM for rule extraction."""
        return (
            f"        <REVIEW_DATA>\n"
            f"        {pr_comment}\n"
            f"        </REVIEW_DATA>\n\n"
            f"        <CODE_DIFF>\n"
            f"        {modified_ast_code}\n"
            f"        </CODE_DIFF>\n"
            f"        "
        )

    @staticmethod
    def _parse_llm_response(raw_content: str) -> dict:
        """Strips optional markdown fences and parses the JSON response from the LLM."""
        if raw_content.startswith("```json"):
            raw_content = raw_content.replace("```json", "").replace("```", "").strip()
        return json.loads(raw_content)

    def _apply_metadata(self, rule_dict: dict, repo: str) -> dict:
        """
        Populates metadata from LLM fields: confidence, category, repo, status.
        High confidence (> AUTO_APPROVE_CONFIDENCE_THRESHOLD) auto-approves the rule.
        Returns the mutated rule_dict.
        """
        if "metadata" not in rule_dict:
            rule_dict["metadata"] = {}

        confidence = rule_dict.get("confidence")
        category = rule_dict.get("category")

        if confidence is not None:
            rule_dict["metadata"]["confidence"] = confidence

        if category is not None:
            rule_dict["metadata"]["category"] = category

        is_high_confidence = (
            confidence is not None
            and confidence > self.AUTO_APPROVE_CONFIDENCE_THRESHOLD
        )
        rule_dict["metadata"]["status"] = "active" if is_high_confidence else "needs_review"
        rule_dict["metadata"]["repo"] = repo

        return rule_dict

    def extract_rule(self, pr_comment: str, modified_ast_code: str, repo: str) -> dict:
        user_prompt = self._build_user_prompt(pr_comment, modified_ast_code)

        try:
            response = completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                api_base=self.api_base,
                api_key=self.api_key,
                temperature=0.1,
                **_qwen_extra_body(self.model),
            )
            raw_content = response.choices[0].message.content.strip()
            if '</think>' in raw_content:
                raw_content = raw_content.split('</think>')[-1].strip()
            rule_dict = self._parse_llm_response(raw_content)

            # LLM signaled this comment isn't a useful rule
            if rule_dict.get("skip"):
                print("[Extractor] Skipped non-rule comment")
                return None

            self._apply_metadata(rule_dict, repo)
            self.fuser.process_and_fuse(rule_dict)
            return rule_dict

        except Exception as e:
            print(f"[Extractor Error]: {e}")
            return None

    async def async_extract_rule(self, pr_comment: str, modified_ast_code: str, repo: str) -> dict:
        user_prompt = self._build_user_prompt(pr_comment, modified_ast_code)

        try:
            response = await acompletion(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                api_base=self.api_base,
                api_key=self.api_key,
                temperature=0.1,
                **_qwen_extra_body(self.model),
            )
            raw_content = response.choices[0].message.content.strip()
            if '</think>' in raw_content:
                raw_content = raw_content.split('</think>')[-1].strip()
            rule_dict = self._parse_llm_response(raw_content)

            if rule_dict.get("skip"):
                print("[Extractor] Skipped non-rule comment")
                return None

            self._apply_metadata(rule_dict, repo)
            self.fuser.process_and_fuse(rule_dict)
            return rule_dict
        except Exception as e:
            print(f"[Async Extractor Error]: {e}")
            return None

    async def _async_classify(self, comment: str) -> bool:
        """Fast binary classifier — returns True if worth extracting."""
        try:
            response = await acompletion(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.CLASSIFY_PROMPT},
                    {"role": "user", "content": f"Comment: {comment[:500]}"}
                ],
                api_base=self.api_base,
                api_key=self.api_key,
                temperature=0.0,
                max_tokens=64,
                **_qwen_extra_body(self.model),
            )
            raw = response.choices[0].message.content.strip()
            # Strip Qwen3 reasoning if the model ignored enable_thinking=False.
            if '</think>' in raw:
                raw = raw.split('</think>')[-1].strip()
            return 'EXTRACT' in raw.upper()
        except Exception:
            return True  # err on side of caution

    async def batch_extract(self, payloads: list, repo: str) -> list:
        # Pass 1: fast classify all comments in parallel
        classify_tasks = [self._async_classify(c) for (c, _) in payloads]
        classifications = await asyncio.gather(*classify_tasks)

        keepers = [(c, d) for (c, d), keep in zip(payloads, classifications) if keep]
        skipped = len(payloads) - len(keepers)
        print(f"[Two-Pass] Classified {len(payloads)} -> {len(keepers)} to extract, {skipped} skipped")

        # Pass 2: extract only the keepers
        extract_tasks = [self.async_extract_rule(c, a, repo) for (c, a) in keepers]
        results = await asyncio.gather(*extract_tasks)
        return [r for r in results if r is not None]
