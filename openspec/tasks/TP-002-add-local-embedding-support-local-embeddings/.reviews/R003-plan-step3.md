## Plan Review: Step 3: Documentation and examples

### Verdict: APPROVE

### Summary
The Step 3 plan is sufficient at the right level of granularity for this task. As with the earlier approved Step 1 and Step 2 plans, the task packet already enumerates the required documentation surfaces and adjacent proof-obligation checks, so the current outcomes are specific enough without turning the step into a file-by-file checklist.

### Issues Found
None.

### Missing Items
None.

### Suggestions
- Make sure the updated examples explicitly demonstrate the new semantic embedding input shape (`EmbeddingRole`, optional title metadata, and query-side instruction/task text) rather than only renaming old types.
- In `README.md` and crate-level docs, call out the supported local GGUF model scope plus the feature-gated `local` / `cuda` / `metal` / `vulkan` usage so the new local capability is discoverable and bounded correctly.
- When reviewing the adjacent change docs, double-check that the existing change summary still clearly communicates the semantic bug fix and minor-version surface expansion after the final documentation wording lands.
