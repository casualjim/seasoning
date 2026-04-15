## Plan Review: Step 1: Implement the approved capability slice

### Verdict: APPROVE

### Summary
The Step 1 plan is aligned with the approved task contract and stays focused on outcome-level implementation rather than over-specifying code mechanics. It covers the core required changes: the dialect/model-family contract correction, embedding and reranker behavior updates, required failure semantics, and preservation of the existing public entry points.

### Issues Found
1. **[Severity: minor]** — `STATUS.md` line 30 says "Cover the missing edge-case semantics uncovered during implementation review," but this is vague and does not name the concrete required edge cases from the prompt. This is not blocking because the parent item already commits to implementing the approved edge-case semantics, but replacing that sub-bullet with the actual required cases would make execution easier to track.

### Missing Items
- None.

### Suggestions
- Consider naming the concrete non-obvious Step 1 outcomes directly in the sub-bullets: feature-gated `LlamaCpp` construction behavior, unsupported dialect/family validation, query-only instruction application, and reranker score-count/index correspondence.
- If the worker expects Step 1 progress to be checkpointed in STATUS.md, a brief mention of the exact local GGUF model scope would make the plan easier to audit against the acceptance criteria.
