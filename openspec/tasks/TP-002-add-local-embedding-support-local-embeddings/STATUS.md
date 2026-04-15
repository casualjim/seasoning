# TP-002: add-local-embedding-support / local-embeddings — Status

**Current Step:** Step 1: Implement the approved capability slice
**Status:** 🟡 In Progress
**Last Updated:** 2026-04-15
**Review Level:** 2
**Review Counter:** 0
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
**Status:** 🟨 In Progress

- [ ] Introduce the corrected retrieval contract across the public embedding/reranking configuration surface, including explicit dialect, model family, embedding role, query instruction/task text, and narrow shared-type plumbing required by the current layout
- [ ] Implement model-family-aware embedding formatting and construction-time validation for supported remote and local embedding backends
- [ ] Implement reranking construction/execution semantics for supported remote and local backends, including explicit unsupported-configuration failures and stable score correspondence
- [ ] Wire the local feature/capability surface without broadening the crate beyond the approved GGUF model scope or breaking existing async entry points

---

### Step 2: Tests and verification work
**Status:** ⬜ Not Started

- [ ] Add or update the required tests for this capability, including adversarial coverage
- [ ] Run targeted validation while iterating on the implementation

---

### Step 3: Documentation and examples
**Status:** ⬜ Not Started

- [ ] Update the required docs and examples for this capability slice
- [ ] Review adjacent docs or examples listed in the proof obligations and update them if affected

---

### Step 4: Repo gates
**Status:** ⬜ Not Started

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

---

## Blockers

*None*

---

## Notes

- 2026-04-15: Reviewed `openspec/tasks/CONTEXT.md`, `openspec/tasks/PHASE-IMPLEMENTATION.md`, and the approved proposal/design/spec for `add-local-embedding-support`; implementation remains within the contracted `local-embeddings` slice.
- 2026-04-15: Confirmed the code snapshot aligns with the task’s owned embedding/config/reranker modules; current public request and error types also flow through adjacent shared modules (`src/api.rs`, `src/error.rs`) that may need narrow companion edits to realize the approved public API delta.

