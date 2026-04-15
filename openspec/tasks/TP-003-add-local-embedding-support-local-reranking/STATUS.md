# TP-003: add-local-embedding-support / local-reranking — Status

**Current Step:** Step 1: Implement the approved capability slice
**Status:** 🟡 In Progress
**Last Updated:** 2026-04-15
**Review Level:** 2
**Review Counter:** 1
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

- [ ] Implement the semantic embedding contract with explicit role/title/query-instruction handling across the approved public API surface
- [ ] Enforce construction-time validation and exact edge-case semantics for dialect/family compatibility, local feature gating, and supported GGUF model scope
- [ ] Preserve async public APIs while routing local llama.cpp embedding and reranking through internal blocking execution with stable score/input correspondence

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
| 1 | plan | 1 | REVISE | .reviews/R001-plan-step1.md |

---

## Discoveries

| Discovery | Disposition | Location |
|-----------|-------------|----------|

---

## Execution Log

| Timestamp | Action | Outcome |
|-----------|--------|---------|
| 2026-04-15 | Task staged | PROMPT.md and STATUS.md created |
| 2026-04-15 16:55 | Task started | Runtime V2 lane-runner execution |
| 2026-04-15 16:55 | Step 0 started | Preflight |
| 2026-04-15 16:58 | Step 0 completed | Contract references and edit surface confirmed |
| 2026-04-15 16:58 | Step 1 started | Implementation |
| 2026-04-15 16:59 | Review R001 | plan Step 1: REVISE |

---

## Blockers

*None*

---

## Notes

- Step 1 plan hydrated after R001 to explicitly cover semantic embedding formatting, construction-time validation/model allowlists, and async-preserving local execution.
- Step 1 implementation must keep query-only instruction application, Gemma `title: none | text: ...` formatting, and local reranker score/index correspondence visible as contract-critical outcomes ahead of Step 2 verification.
