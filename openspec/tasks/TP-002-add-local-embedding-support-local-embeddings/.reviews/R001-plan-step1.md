## Plan Review: Step 1: Implement the approved capability slice

### Verdict: APPROVE

### Summary
The Step 1 plan covers the required implementation outcomes at the right level of granularity: it captures the semantic API correction, family-aware formatting, local/remote backend validation, reranking behavior, and the feature-gated local surface. It also leaves room for the narrow shared-type plumbing already called out in the notes, which is appropriate given the current `src/api.rs`/`src/error.rs` coupling.

### Issues Found
1. **[Severity: minor]** — No blocking issues found. The hydrated Step 1 outcomes are aligned with the approved design/spec and are specific enough to guide implementation without over-prescribing file-by-file mechanics.

### Missing Items
- None.

### Suggestions
- As implementation starts, keep the Step 1 work item on the local capability surface explicitly tied to the spec’s Hugging Face `hf:` GGUF resolution and `cuda`/`metal`/`vulkan` feature passthrough requirements, so Cargo/local-runtime work does not drift into broader local model support.
- When updating the shared request/config surface, double-check compatibility plumbing for adjacent public types (`src/api.rs`, and any batching/service code that constructs `EmbeddingInput`) since the current notes already identify shared-type ripple effects.
