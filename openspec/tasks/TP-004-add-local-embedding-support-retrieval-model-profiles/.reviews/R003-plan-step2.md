## Plan Review: Step 2: Tests and verification work

### Verdict: REVISE

### Summary
The Step 2 plan is appropriately outcome-oriented, and the prompt already defines the main required test buckets for formatting, validation, and reranking semantics. However, the current STATUS entry is too generic around verification of the feature-gated local path, which means the step could finish without validating an acceptance criterion that the default repo gates do not cover.

### Issues Found
1. **[Severity: important]** — `STATUS.md` lines 39-40 only commit to generic tests and targeted validation, but they do not explicitly include a `local`-feature validation path. In this repo, `mise test` maps to `cargo test --all`, which does not enable `--features local`, so the feature-gated llama.cpp embedder/reranker code and `#[cfg(feature = "local")]` tests can be skipped entirely unless the plan calls that out. Add an explicit Step 2 outcome covering targeted validation for the `local` feature (for example, compile/test execution with `--features local`, using a raw cargo invocation only if no `mise` task supports feature selection and documenting that deviation).

### Missing Items
- Explicit targeted validation for the `local` feature so the acceptance criteria for local embedding/reranking clients are actually exercised, not just the default non-local build.

### Suggestions
- Carry forward the Step 1 code-review test gap by adding focused coverage for config conversion accepting `dialect = "llama.cpp"` in both `Embedding::to_embedder_config` and `Reranker::to_reranker_config`.
- If you expand the Step 2 notes, mention both default-feature and `local`-feature permutations so it is clear how the `#[cfg(not(feature = "local"))]` and `#[cfg(feature = "local")]` branches will both be verified.
