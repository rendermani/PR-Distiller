---
name: Architecture_State
description: Aggregated enterprise constraints ensuring Copilot compliance for Architecture_State.
---

# Architecture State Guidelines

You must strictly adhere to the following architecture rules aggregated from previous PR rejections.
Failure to do so will result in enterprise architectural violations.

## Constraints & Anti-Patterns:
1. Rule: Strict Constraint for vercel/next.js - Context: Aggregated from core maintainer feedback to prevent recurring architecture violations.. Enforce: Yeah that works too, they were dropped from the codebase entirely. I meant removed from config and such. Idk if supported still implies deprecated, as in you could try to set it, like `experimental.useCache`\n2. Rule: Strict Constraint for pydantic/pydantic - Context: Aggregated from core maintainer feedback to prevent recurring architecture violations.. Enforce: To get this skipped in coverage, could you update the coverage config to be (instead of using `exclude_lines`):

```toml
[tool.coverage.report]
precision = 2
exclude_also = [
    'raise\sNotImplementedError',
    '@(typing\.)?overload',
    'class .*\bProtocol\):',
    '(typing\.)?.assert_never',
]
```\n