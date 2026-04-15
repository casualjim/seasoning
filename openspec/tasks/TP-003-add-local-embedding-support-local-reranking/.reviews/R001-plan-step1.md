## Plan Review: Step 1: Implement the approved capability slice

### Verdict: REVISE

### Summary
The task packet is well-scoped, but the current Step 1 plan in `STATUS.md` is still too abstract to show that the slice’s contract-critical behaviors will be covered. It mostly restates “implement the delta” without naming the specific outcome groups and preservation risks that make this capability correct.

### Issues Found
1. **[Severity: important]** — `openspec/tasks/TP-003-add-local-embedding-support-local-reranking/STATUS.md:26-28` is too generic for this slice. Step 1 needs to explicitly cover the three core outcome groups from the contract: the semantic embedding API change (role/title/query instruction), construction-time validation for unsupported dialect/family/feature combinations, and the local llama.cpp backend scope for the exact supported GGUF models. As written, the plan could claim completion while still missing contract-critical behaviors from `PROMPT.md:37-49`.
2. **[Severity: important]** — The plan does not call out the async/runtime preservation constraint from `PROMPT.md:107-110` and `design.md` Decision 6. This step is not just “add local support”; it must preserve the existing async public API while local llama.cpp work is internally blocking and stateful. Add an outcome-level item or note covering that compatibility requirement so it is treated as part of the implementation slice, not an incidental detail.
3. **[Severity: important]** — The plan lacks explicit proof-obligation linkage for the edge cases that are easiest to miss during implementation: query-only application of embedding instructions, Gemma’s `title: none | text: ...` document formatting, and local reranker score/index correspondence. Detailed test design belongs in Step 2, but Step 1 should still name these behaviors as implementation outcomes so they are not deferred accidentally.

### Missing Items
- Explicit Step 1 coverage for validating unsupported `Dialect`/`ModelFamily`/feature combinations at construction time.
- Explicit Step 1 coverage for the exact supported local model allowlist/scope.
- Explicit Step 1 coverage for preserving async caller-facing APIs while introducing blocking local execution.
- A short note tying Step 1 implementation to the required edge-case semantics that Step 2 must verify.

### Suggestions
- Keep the checklist outcome-level, but expand Step 1 with 2-4 concrete bullets or notes for: semantic formatting contract, local backend/allowlist and validation, and async-preserving local execution.
- If preflight discovered that any required public API surface lives outside the listed edit surface, record that as an amendment now rather than discovering the mismatch mid-implementation.
