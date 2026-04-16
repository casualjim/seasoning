## Code Review: Step 1: Implement the approved capability slice

### Verdict: APPROVE

### Summary
The implementation addresses the contract-critical reranker correspondence issue by validating score counts for DeepInfra and local execution, and by rebuilding OpenAI-style results into strict input order with explicit failures for missing, duplicate, or out-of-range indices. I did not find a blocking defect in the changed code, and `cargo test --all` passes locally as a fallback validation path.

### Issues Found
1. None.

### Pattern Violations
- None observed in the changed code.

### Test Gaps
- `src/reranker.rs:245-278,336-362` adds important malformed-response handling, but there is not yet an adversarial test covering DeepInfra count mismatches or OpenAI duplicate/missing/out-of-range indices. That belongs in Step 2, not as a Step 1 blocker.
- `src/api.rs:18-27` adds serde aliases for `Dialect::LlamaCpp`; a small config/serde round-trip test would help lock in backward-compatible parsing for `llamacpp`, `llama-cpp`, and `llama_cpp`.

### Suggestions
- Add Step 2 tests that assert the new `Error::RerankScoreCountMismatch` and `Error::InvalidRerankScoreIndex` paths so future refactors do not silently reintroduce score/input misalignment.
- If repo-gate reporting is captured later, note that `mise test` was not runnable in this environment because the task wrapper expects `DEEPINFRA_API_KEY` templating; I used `cargo test --all` only as a validation fallback.
