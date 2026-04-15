## Code Review: Step 3: Documentation and examples

### Verdict: APPROVE

### Summary
The Step 3 doc updates are aligned with the behavior added in the earlier implementation and test steps. README and crate-level docs now explicitly document the canonical `llama.cpp` config alias alongside the previously supported spellings, which addresses the user-facing gap introduced by the config parsing update without widening the API semantics.

### Issues Found
1. None.

### Pattern Violations
- None observed.

### Test Gaps
- None blocking for this documentation-focused step.

### Suggestions
- Consider updating `src/error.rs:38` so the invalid-dialect error text reflects the newly documented `llama.cpp` alias as well; this is a consistency improvement, not a blocker for Step 3.
