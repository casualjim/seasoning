## Plan Review: Step 2: Tests and verification work

### Verdict: REVISE

### Summary
The Step 2 plan is close: it covers the required formatting tests, validation tests, and iterative test execution at a high level. The blocking gap is that the verification plan does not explicitly exercise the feature-gated `local` test path, so this step could be marked complete without ever running the local-only tests that prove the new backend and model allowlist behavior.

### Issues Found
1. **[Severity: important]** — The validation item in `STATUS.md:37` is too generic for a feature-gated capability. `local` is an opt-in Cargo feature (`Cargo.toml:13-18`), and the only repo test task is `cargo test --all` via `mise test` (`mise.toml:16-17`), which does not enable `--features local`. That means Step 2 currently has no explicit plan to run the `#[cfg(feature = "local")]` tests alongside the `#[cfg(not(feature = "local"))]` cases. Add an explicit outcome covering targeted validation in both modes: default features for the unsupported-without-`local` path, and a `local`-enabled test run for the local model/allowlist/backend coverage. If a direct Cargo command is needed because there is no `mise` wrapper for feature-enabled targeted runs, call out that exception explicitly.

### Missing Items
- Explicit targeted validation for both build modes: without `local` and with `local` enabled.

### Suggestions
- Step 1 was already hydrated to keep query-only instruction behavior, Gemma `title: none | text: ...`, and score/index correspondence visible as contract-critical outcomes; keep those cases explicit in the final test names/assertions even if some coverage already exists.
- If the formatting tests continue to live in `src/api.rs`, make sure the targeted test run scope includes that module too, not just `embedding` and `reranker`.
