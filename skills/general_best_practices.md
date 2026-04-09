---
name: General_Best_Practices
description: Aggregated enterprise constraints ensuring Copilot compliance for General_Best_Practices.
---

# General Best Practices Guidelines

You must strictly adhere to the following architecture rules aggregated from previous PR rejections.
Failure to do so will result in enterprise architectural violations.

## Constraints & Anti-Patterns:
1. Rule: Strict Constraint for pydantic/pydantic - Context: Aggregated from core maintainer feedback to prevent recurring architecture violations.. Enforce: _just doing field-by-field comparison with the custom comparators where needed_

To compare those that don't use a custom comparator, I also need to create another dictionary excluding those that do use custom logic, so I think it's hard to avoid the auxiliar dict creation.

In `attrs`, they have a kind of python macro (using `eval`) that builds the `__eq__`  when the data model is created, and they compare field-by-field using the `and` operator `self.f1 == other.f1 and ... and self.fn == other.fn`, which I believe is less optimum than comparing dicts\n2. Rule: Strict Constraint for pydantic/pydantic - Context: Aggregated from core maintainer feedback to prevent recurring architecture violations.. Enforce: Instead of this new test, please amend the test above to also test `model_validate_strings` after the `model_validate_json` assert.\n3. Rule: Strict Constraint for huggingface/transformers - Context: Aggregated from core maintainer feedback to prevent recurring architecture violations.. Enforce: Maybe a small helper function instead of repeating twice?\n4. Rule: Strict Constraint for vercel/next.js - Context: Aggregated from core maintainer feedback to prevent recurring architecture violations.. Enforce: Does this kind of description work to get the agent to read this entry? I wonder if we can straight up open with, `The middleware.js file convention is deprecated, see proxy.js`\n5. Rule: Strict Constraint for vercel/next.js - Context: Aggregated from core maintainer feedback to prevent recurring architecture violations.. Enforce: "are no longer necessary and have since been removed." @icyJoseph is this true? Removed? ppr is deprecated right\n6. Rule: Strict Constraint for huggingface/transformers - Context: Aggregated from core maintainer feedback to prevent recurring architecture violations.. Enforce: yeah the processor ensures height==width, this was for extra safety if processor was not used. But I removed it now, to allow hackers the freedom to do whatever they want.\n7. Rule: Strict Constraint for huggingface/transformers - Context: Aggregated from core maintainer feedback to prevent recurring architecture violations.. Enforce: yep, tbh i thought it was already a warning, instead of error in the past. Maybe smth changed recently\n8. Rule: Strict Constraint for vercel/next.js - Context: Aggregated from core maintainer feedback to prevent recurring architecture violations.. Enforce: Use `access_mut` instead of `access_read` so it can return that guard directly\n9. Rule: Strict Constraint for huggingface/transformers - Context: Aggregated from core maintainer feedback to prevent recurring architecture violations.. Enforce: very few lines are changed. I wish in future, modular can be used with git-like diffs so that the bulk of code is no longer needed to be copied.

comments added

```python
# This differs from Vivit by using bilinear mode instead of bicubic.
patch_pos_embed = nn.functional.interpolate(
    patch_pos_embed,
    size=(num_row_patches, num_col_patches),
    mode="bilinear",
    antialias=True,
)

# no cls token is added unlike Vivit
```\n10. Rule: Strict Constraint for huggingface/transformers - Context: Aggregated from core maintainer feedback to prevent recurring architecture violations.. Enforce: Sorry, just saw the original issue mentioning it's deprecated method.

Anyway, it turns out we need to set

torch.backends.cudnn.conv.fp32_precision = "ieee"

for our CI. It's not clear to me why 

torch.backends.fp32_precision = "ieee"

isn't enough.\n11. Rule: Strict Constraint for pydantic/pydantic - Context: Aggregated from core maintainer feedback to prevent recurring architecture violations.. Enforce: Yeah it is necessary because the sign bit of `NaN` may or may not be positive, so `is_sign_positive` alone would skip negative `NaN`s and result in, say, `1NaNj` instead of `1+NaNj` as all `NaN`s are serialized as `NaN`. Python's `complex` behavior here:
```
>>> complex(1, float("nan"))
(1+nanj)
>>> complex(1, float("-nan"))
(1+nanj)
>>> math.copysign(1, float("nan"))
1.0
>>> math.copysign(1, float("-nan"))
-1.0
```
It's usually better to use [`signum`](https://docs.rs/num-traits/0.2.19/num_traits/float/trait.Float.html#tymethod.signum) which will propagate `NaN` instead, but since `+` is needed for `NaN`s as well we would have to check for `NaN` either way.\n12. Rule: Strict Constraint for pydantic/pydantic - Context: Aggregated from core maintainer feedback to prevent recurring architecture violations.. Enforce: Please use a `TypeAdapter` in this test too, rather than the internal function. I also think we should probably use an import string like `'collections.defaultdict:get'` which a buggy implementation might actually resolve, rather than a truly nonexistent path.\n