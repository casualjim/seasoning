## Code Review: Step 1: Evaluate delta and preservation conformance

### Verdict: APPROVE

### Summary
The Step 1 update completes the required STATUS tracking for delta/preservation review and records the main conformance risk I could substantiate from the current repo state: TP-004 widened into out-of-scope `.mise` repo-gate edits even though its approved edit surface was limited to Cargo/README/src-owned paths. The notes also correctly capture that the requested public semantic contract and preservation constraints are present in the assembled implementation, so I do not see a blocking gap in this step's completion.

### Issues Found
1. None.

### Pattern Violations
- None.

### Test Gaps
- None blocking for this STATUS-only verification step.

### Suggestions
- When Step 3 writes the canonical conformance report, make the preserved non-goals explicit alongside the current notes (for example: query expansion remains out of scope, arbitrary model-family support was not added, and local model loading is still restricted to the approved Hugging Face GGUF artifacts). That evidence is implied today but would make the final report easier to defend.
- `openspec/tasks/TP-005-verify-add-local-embedding-support/STATUS.md:89` may be worth wording a bit more narrowly in the final report: the scope-breach finding is well-supported, but whether the current `mise test` shape truly "reduces proof strength" should be tied to Step 2's repo-gate evidence rather than treated as already established in Step 1.
