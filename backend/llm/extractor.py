import json
import asyncio
from litellm import completion, acompletion
from db.lightrag_manager import LightRAGManager
from pipeline.semantic_fuser import SemanticFuser


def _qwen_extra_body(model: str) -> dict:
    """Return Qwen3-specific completion kwargs, or {} for other providers.

    `enable_thinking=False` is recognised by Qwen3's chat template only;
    OpenAI/Anthropic/Gemini reject unknown extra_body keys with 4xx, so the
    flag must be gated on the model name.
    """
    if "qwen" in (model or "").lower():
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    return {}


def _qwen_no_think(model: str) -> dict:
    """Extra body kwargs that disable Qwen3 thinking for fast classify calls.

    Ollama's native /api/chat honours a top-level `think: false` field which
    suppresses the reasoning chain entirely, cutting latency from ~15s to ~0.3s.
    LiteLLM passes extra_body keys through to the Ollama request body.
    Safe to pass to non-Qwen models — unknown keys are ignored.
    """
    if "qwen" in (model or "").lower():
        return {"extra_body": {"think": False}}
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
        model_id: str | None = None,
        model_label: str | None = None,
    ):
        # The orchestrator is the single source of truth for model config.
        # If model or api_base are None here, the LLM call will fail loudly —
        # that is intentional; the orchestrator must always pass explicit values.
        self.model = model
        self.api_base = api_base
        self.api_key = api_key or "unused"
        self.model_id = model_id
        self.model_label = model_label
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
        elif raw_content.startswith("```"):
            raw_content = raw_content.replace("```", "").strip()
        # Extract just the first JSON object — models sometimes append extra text.
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(raw_content.strip())
        return obj

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
                **_qwen_no_think(self.model),
            )
            msg = response.choices[0].message
            raw_content = (msg.content or "").strip()
            if not raw_content:
                raw_content = (getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or "").strip()
            if '</think>' in raw_content:
                raw_content = raw_content.split('</think>')[-1].strip()
            rule_dict = self._parse_llm_response(raw_content)

            # LLM signaled this comment isn't a useful rule
            if rule_dict.get("skip"):
                print("[Extractor] Skipped non-rule comment")
                return None

            self._apply_metadata(rule_dict, repo)
            if self.model_id:
                rule_dict["metadata"]["extracted_by_model"] = self.model_id
            if self.model_label:
                rule_dict["metadata"]["extracted_by_label"] = self.model_label
            rule_dict["metadata"].setdefault("merged_with_models", [])
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
                # Disable Qwen3 thinking on extract too — the JSON output
                # doesn't need a reasoning chain and thinking turns ~3s calls
                # into ~30s calls, multiplied across thousands of payloads.
                **_qwen_no_think(self.model),
            )
            msg = response.choices[0].message
            raw_content = (msg.content or "").strip()
            if not raw_content:
                raw_content = (getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or "").strip()
            if '</think>' in raw_content:
                raw_content = raw_content.split('</think>')[-1].strip()
            rule_dict = self._parse_llm_response(raw_content)

            if rule_dict.get("skip"):
                print("[Extractor] Skipped non-rule comment")
                return None

            self._apply_metadata(rule_dict, repo)
            if self.model_id:
                rule_dict["metadata"]["extracted_by_model"] = self.model_id
            if self.model_label:
                rule_dict["metadata"]["extracted_by_label"] = self.model_label
            rule_dict["metadata"].setdefault("merged_with_models", [])
            # process_and_fuse is synchronous (ChromaDB + optional LLM merge).
            # Run it in a thread so the event loop stays free to serve HTTP
            # requests and fire progress callbacks between extractions.
            await asyncio.to_thread(self.fuser.process_and_fuse, rule_dict)
            return rule_dict
        except Exception as e:
            print(f"[Async Extractor Error]: {e}")
            return None

    async def _async_classify(self, comment: str) -> bool:
        """Fast binary classifier — returns True if worth extracting.

        Uses think:false (Ollama native) to skip Qwen3's reasoning chain,
        cutting latency from ~15s to ~0.3s per call.
        """
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
                max_tokens=16,
                **_qwen_no_think(self.model),
            )
            msg = response.choices[0].message
            raw = (msg.content or "").strip()
            if not raw:
                raw = (getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or "").strip()
            if '</think>' in raw:
                raw = raw.split('</think>')[-1].strip()
            return 'EXTRACT' in raw.upper()
        except Exception:
            return True  # err on side of caution

    # Maximum concurrent LLM requests. Ollama on Apple Silicon with Metal can
    # handle several requests in flight; more than ~4 just queues without
    # gaining throughput. Classify uses /no_think so it is fast; extract is
    # slower but still benefits from a few slots of overlap.
    LLM_CONCURRENCY = 4

    async def batch_extract(self, payloads: list, repo: str, progress_callback=None) -> list:
        """Two-pass extract. progress_callback(status_str, pct_int) is called after
        each classify and extract completion so the UI stays live."""
        total = len(payloads)
        sem = asyncio.Semaphore(self.LLM_CONCURRENCY)

        async def _classify_one(comment: str) -> bool:
            async with sem:
                return await self._async_classify(comment)

        async def _extract_one(comment: str, diff: str) -> dict:
            async with sem:
                return await self.async_extract_rule(comment, diff, repo)

        # Pass 1: classify — bounded concurrency, report per-completion
        classify_tasks = [
            asyncio.ensure_future(_classify_one(c))
            for c, _ in payloads
        ]
        classifications = [None] * total
        done_count = 0
        for coro in asyncio.as_completed(classify_tasks):
            try:
                await coro
            except Exception:
                pass  # tolerate individual failures; see the done/result scan below
            # Map back: find the first task that is done and unrecorded
            for i, t in enumerate(classify_tasks):
                if t.done() and classifications[i] is None:
                    try:
                        classifications[i] = t.result()
                    except Exception:
                        classifications[i] = True
            done_count += 1
            if progress_callback:
                pct = 35 + int(done_count / total * 25)  # 35→60%
                progress_callback(f"Classifying {done_count}/{total} comments...", pct)

        # Resolve any remaining None slots
        for i, v in enumerate(classifications):
            if v is None:
                classifications[i] = True

        keepers = [(c, d) for (c, d), keep in zip(payloads, classifications) if keep]
        skipped = total - len(keepers)
        print(f"[Two-Pass] Classified {total} -> {len(keepers)} to extract, {skipped} skipped")
        if progress_callback:
            progress_callback(f"Classified {len(keepers)}/{total} for extraction ({skipped} skipped)", 60)

        # Pass 2: extract keepers — bounded concurrency, report per-completion
        extract_tasks = [
            asyncio.ensure_future(_extract_one(c, a))
            for c, a in keepers
        ]
        results = [None] * len(keepers)
        done_count = 0
        for coro in asyncio.as_completed(extract_tasks):
            try:
                await coro
            except Exception:
                pass
            for i, t in enumerate(extract_tasks):
                if t.done() and results[i] is None:
                    try:
                        results[i] = t.result()
                    except Exception:
                        results[i] = False
            done_count += 1
            if progress_callback:
                pct = 60 + int(done_count / max(len(keepers), 1) * 35)  # 60→95%
                progress_callback(f"Extracting rules {done_count}/{len(keepers)}...", pct)

        return [r for r in results if r and r is not False]
