## Plan Review: Step 3: Documentation and examples

### Verdict: APPROVE

### Summary
The Step 3 plan covers the required outcomes from the task packet: updating the mandated docs/examples and checking the adjacent spec docs if the implementation changed wording or proof-obligation evidence. For a documentation step, the current outcome-level granularity is appropriate and does not omit any blocking requirement from `PROMPT.md`.

### Issues Found
1. **[Severity: minor]** — No blocking issues found.

### Missing Items
- None.

### Suggestions
- When executing this step, make sure the examples visibly exercise the new semantic API surface (`ModelFamily`, `EmbeddingRole`, optional `title`, and query-side instruction/task text), since that is the main behavioral change callers need to learn.
- In the README or inline docs, explicitly call out local-feature enablement and the intentionally narrow supported GGUF model list so the feature-gated `LlamaCpp` path is discoverable and its scope is clear.
- If `proposal.md` and `design.md` do not need edits after review, note that decision in `STATUS.md` so the "check if affected" obligation is clearly closed.
