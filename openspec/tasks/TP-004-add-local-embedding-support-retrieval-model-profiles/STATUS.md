# TP-004: add-local-embedding-support / retrieval-model-profiles — Status

**Current Step:** Step 2: Tests and verification work
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

- [x] Apply the requested delta for this capability inside the approved module ownership and edit surface
  - [x] Reconcile embedding-client construction and semantic formatting behavior with the approved dialect/model-family contract
  - [x] Reconcile reranker/local-runtime validation and score semantics with the approved dialect/model-family contract
- [x] Implement the approved behavioral semantics and exact failure or edge-case semantics
  - [x] Cover the missing edge-case semantics uncovered during implementation review
- [x] Preserve all stated constraints and do not alter interfaces beyond the approved delta
  - [x] Keep the remote client entry points and public semantic API stable while applying the approved fixes

---

### Step 2: Tests and verification work
**Status:** 🟨 In Progress

- [ ] Add or update the required tests for this capability, including adversarial coverage
  - [ ] Add focused config-conversion coverage for the canonical `llama.cpp` dialect alias
- [ ] Run targeted validation while iterating on the implementation
  - [ ] Validate the default-feature test path used by repo gates
  - [ ] Validate the feature-gated `local` path explicitly with a targeted `--features local` test run

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
| 2026-04-15 17:15 | Task started | Runtime V2 lane-runner execution |
| 2026-04-15 17:15 | Step 0 started | Preflight |
| 2026-04-15 17:19 | Step 0 completed | Contract references and edit surface confirmed |
| 2026-04-15 17:19 | Step 1 started | Capability implementation |
| 2026-04-15 17:28 | Step 1 completed | Added config-side llama.cpp alias handling without widening the public API |
| 2026-04-15 17:28 | Step 2 started | Tests and targeted validation |

---

## Blockers

*None*

---

## Notes

- Plan review R003: explicitly verify both default-feature and `local`-feature paths in Step 2 because `mise test` maps to `cargo test --all` and does not enable `--features local`.
- Plan review R003 suggestion: add focused config-conversion coverage for `dialect = "llama.cpp"` in both embedding and reranker config conversion paths.
| 2026-04-15 17:18 | Review R001 | plan Step 1: APPROVE |
| 2026-04-15 17:23 | Review R002 | code Step 1: APPROVE |
| 2026-04-15 17:25 | Review R003 | plan Step 2: REVISE |
