---
name: Database_Security
description: Aggregated enterprise constraints ensuring Copilot compliance for Database_Security.
---

# Database Security Guidelines

You must strictly adhere to the following architecture rules aggregated from previous PR rejections.
Failure to do so will result in enterprise architectural violations.

## Constraints & Anti-Patterns:
1. Rule: Strict Constraint for tiangolo/sqlmodel - Context: Aggregated from core maintainer feedback to prevent recurring architecture violations.. Enforce: `ty` complaints with:

> warning[invalid-legacy-positional-parameter]: Invalid use of the legacy convention for positional-only parameters
   --> sqlmodel\sql\_expression_select_gen.py:132:5
    |
130 | @overload
131 | def select(
132 |     entity_0: _TScalar_0,
    |     -------- Prior parameter here was positional-or-keyword
133 |     __ent1: _TCCA[_T1],
    |     ^^^^^^ Parameter name begins with `__` but will not be treated as positional-only
134 | ) -> Select[tuple[_TScalar_0, _T1]]: ...
    |
info: A parameter can only be positional-only if it precedes all positional-or-keyword parameters
info: rule `invalid-legacy-positional-parameter` is enabled by default

So basically we can't have `__var` if there is a scalar parameter in front of this one.

As a quick fix, I changed all double underscores to singles, but it feels a bit like a hack. We could also suppress the `ty` warning, but that also feels wrong...\n