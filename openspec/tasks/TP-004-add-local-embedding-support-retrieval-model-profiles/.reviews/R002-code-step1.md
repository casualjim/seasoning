## Code Review: Step 1: Implement the approved capability slice

### Verdict: APPROVE

### Summary
The Step 1 change is small but correct: `AppConfig` now accepts the additional `llama.cpp` dialect spelling during config-to-runtime conversion, which closes the config-side alias gap without changing the canonical public `Dialect` representation. I verified the diff and surrounding code, and `cargo test --all` passes locally; I had to use raw `cargo test` because `mise test` currently fails in this environment while rendering `env.DEEPINFRA_API_KEY` from `mise.toml`.

### Issues Found
1. None.

### Pattern Violations
- None.

### Test Gaps
- Add a focused unit test in Step 2 that exercises `AppConfig`/`Embedding::to_embedder_config` and `Reranker::to_reranker_config` with `dialect = "llama.cpp"` so this newly accepted alias stays covered explicitly.

### Suggestions
- If you want config error messaging to be more discoverable, consider documenting the accepted llama.cpp aliases alongside the invalid-dialect error text or config docs examples, but this is not blocking for correctness.
