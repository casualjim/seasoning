# TP-002: add-local-embedding-support / local-embeddings — Status

**Current Step:** Step 4: Repo gates
**Status:** 🟡 In Progress
**Last Updated:** 2026-04-15
**Review Level:** 2
**Review Counter:** 3
**Iteration:** 1
**Size:** M

> **Hydration:** Checkboxes represent meaningful outcomes, not individual code changes. Expand them only when runtime discovery or review feedback requires it.

---

### Step 0: Preflight
**Status:** ✅ Complete

- [x] Read the contract references and confirm the task remains within the approved change contract
- [x] Confirm the current code snapshot still matches the approved edit surface before modifying files

---

### Step 1: Implement the approved capability slice
**Status:** ✅ Complete

- [x] Introduce the corrected retrieval contract across the public embedding/reranking configuration surface, including explicit dialect, model family, embedding role, query instruction/task text, and narrow shared-type plumbing required by the current layout
- [x] Implement model-family-aware embedding formatting and construction-time validation for supported remote and local embedding backends
- [x] Implement reranking construction/execution semantics for supported remote and local backends, including explicit unsupported-configuration failures and stable score correspondence
- [x] Wire the local feature/capability surface without broadening the crate beyond the approved GGUF model scope or breaking existing async entry points

---

### Step 2: Tests and verification work
**Status:** ✅ Complete

- [x] Add or update the required tests for this capability, including adversarial coverage
- [x] Run targeted validation while iterating on the implementation

---

### Step 3: Documentation and examples
**Status:** ✅ Complete

- [x] Update the required docs and examples for this capability slice
- [x] Review adjacent docs or examples listed in the proof obligations and update them if affected

---

### Step 4: Repo gates
**Status:** 🟨 In Progress

- [ ] Run the required repo gates and leave the repository in a fully passing state

---

## Reviews

| # | Type | Step | Verdict | File |
|---|------|------|---------|------|

---

## Discoveries

| Discovery | Disposition | Location |
|-----------|-------------|----------|

---

## Execution Log

| Timestamp | Action | Outcome |
|-----------|--------|---------|
| 2026-04-15 | Task staged | PROMPT.md and STATUS.md created |
| 2026-04-15 16:18 | Task started | Runtime V2 lane-runner execution |
| 2026-04-15 16:18 | Step 0 started | Preflight |
| 2026-04-15 16:18 | Step 1 hydrated | Expanded implementation outcomes around semantic retrieval API, local llama.cpp support, and validation semantics |
| 2026-04-15 16:54 | Step 1 complete | Retrieval semantics, local feature surface, and llama.cpp-backed embedding/reranking paths implemented |
| 2026-04-15 17:00 | Step 2 complete | Added semantic-formatting and validation tests; default and `local` builds validated |
| 2026-04-15 17:06 | Step 3 complete | Updated README, crate docs, and inline examples; adjacent proof-obligation docs re-verified |

---

## Blockers

*None*

---

## Notes

- 2026-04-15: Reviewed `openspec/tasks/CONTEXT.md`, `openspec/tasks/PHASE-IMPLEMENTATION.md`, and the approved proposal/design/spec for `add-local-embedding-support`; implementation remains within the contracted `local-embeddings` slice.
- 2026-04-15: Confirmed the code snapshot aligns with the task’s owned embedding/config/reranker modules; current public request and error types also flow through adjacent shared modules (`src/api.rs`, `src/error.rs`) that may need narrow companion edits to realize the approved public API delta.
- 2026-04-15: Introduced the retrieval semantic contract in `src/api.rs`, `src/config.rs`, `src/embedding.rs`, `src/reranker.rs`, `src/error.rs`, and `src/service.rs`, including explicit `Dialect`, `ModelFamily`, `EmbeddingRole`, query instruction/task handling, and configuration parsing.
- 2026-04-15: Implemented family-aware embedding formatting for Gemma and Qwen3 plus local/remote constructor branching, with explicit `local` feature validation and supported Hugging Face GGUF model checks.
- 2026-04-15: Implemented reranker validation and execution branching for remote/OpenAI, remote/DeepInfra, and local llama.cpp flows, preserving query-empty failures and one-score-per-document ordering.
- 2026-04-15: Added `local`, `cuda`, `metal`, and `vulkan` Cargo features plus `hf-hub` / `llama-cpp-2` integration in `src/local.rs`; validated default and `local`-feature library builds with `cargo test --lib` and `cargo test --lib --features local --no-run`.
- 2026-04-15: Added retrieval-formatting tests for Gemma and Qwen3, adversarial constructor tests for unsupported `LlamaCpp` usage and invalid reranker families, and instruction-formatting coverage for Qwen3 reranking.
- 2026-04-15: Targeted validation succeeded with `cargo test --lib` and `cargo test --lib --features local --no-run`, covering the default library test set and feature-gated local compilation.
- 2026-04-15: Updated `README.md`, crate-level docs in `src/lib.rs`, and inline module examples in `src/embedding.rs`, `src/reranker.rs`, and `src/config.rs`; `cargo test --doc` passed.
- 2026-04-15: Re-reviewed the adjacent `proposal.md` and `design.md` proof-obligation docs after implementation; the current wording still matches the shipped semantic and local-backend behavior, so no contract-doc edits were required.
| 2026-04-15 16:23 | Review R001 | plan Step 1: APPROVE |
| 2026-04-15 16:42 | Review R002 | plan Step 2: APPROVE |
| 2026-04-15 16:48 | Review R003 | plan Step 3: APPROVE |
