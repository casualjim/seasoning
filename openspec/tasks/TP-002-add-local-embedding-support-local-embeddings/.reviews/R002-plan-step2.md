## Plan Review: Step 2: Tests and verification work

### Verdict: APPROVE

### Summary
The Step 2 plan is appropriately outcome-focused for this task size: it commits to adding the required capability tests, includes adversarial coverage, and reserves targeted validation during implementation. Because the task packet already defines the exact required test scenarios, the current plan is sufficient without being over-specified.

### Issues Found
None.

### Missing Items
None.

### Suggestions
- Make sure the targeted validation covers both build modes that matter for this change: default features (to verify explicit `LlamaCpp` rejection) and `local`-enabled builds (to verify the local capability surface compiles and the supported paths are exercised as far as practical).
- Keep non-regression coverage for existing remote embedding/reranking flows in mind when updating tests, since the public semantic model changed even though remote support must remain intact.
- Prefer `mise` for repo-level gates, and if you use narrower direct `cargo test` commands while iterating, note that they are targeted deviations rather than replacements for Step 4.
