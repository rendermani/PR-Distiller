# Multi-model benchmark — 2026-05-19

Source of the "~55% overlap / ~40% more unique rules" figures quoted in
`2026-05-19-multi-model-pipeline.md` (same directory), which motivated running
several models per job and fusing the results.

Note: this lives alongside the plan rather than in `docs/benchmarks/`, because
`.gitignore` reserves `benchmarks/` for local-only generated output.

The raw per-run JSON (`results/*.json`, ~0.36 MB each, containing every extracted
rule) is gitignored as generated output and was **not** retained. This file keeps
the headline numbers so the conclusions remain auditable without re-running a
~4.5-hour-per-model sweep.

## Setup

- Corpus: the same cached 1,241-comment crawl for every model, so differences
  are attributable to the model rather than the input.
- Pipeline: full two-pass extract (classify → extract) per model, rules wiped
  between runs.
- Harness: `scripts/bench_models.sh`.
- Host: Apple Silicon, Ollama on the host with Metal acceleration.

## Results

| Model | Rules extracted | Wall time |
|---|---:|---:|
| `ollama/qwen3:8b` | 307 | not recorded |
| `ollama/gemma4:e4b` | 300 | 15,744 s (~4h 22m) |
| `ollama/qwen2.5-coder:7b-instruct` | 299 | 16,379 s (~4h 33m) |
| `ollama/gpt-oss:20b` | 298 | 16,626 s (~4h 37m) |

## Conclusions

**Rule counts are nearly identical (298–307) but the rules themselves are not.**
Pairwise overlap was ~55%, so roughly 45% of each model's output was unique. That
is the entire basis for the multi-model design: running N models and letting the
semantic fuser deduplicate yields ~40% more unique rules than any single model,
at N× wall time.

**Model size did not predict quality here.** `gpt-oss:20b` (13.79 GB) produced
the fewest rules and took the longest; `gemma4:e4b` (9.61 GB) matched it in
count at slightly lower cost. On this corpus the smaller models were the better
Pareto choice, which is why `qwen3:8b` (5.23 GB) is the default.

**Wall time is the binding constraint, not accuracy.** ~4.5 hours per model means
a 3-model job is a ~13-hour run. Cancellation therefore has to work mid-pass —
see the `check_cancel` plumbing in `batch_extract`.

## Reproducing

Requires re-pulling the models (~33 GB for the four above; they were removed to
reclaim disk):

```bash
ollama pull qwen3:8b gemma4:e4b qwen2.5-coder:7b-instruct gpt-oss:20b
REPO=<owner/name> MONTHS=5 scripts/bench_models.sh
```

Output lands in `results/` (gitignored). Expect ~4.5 h per model.
