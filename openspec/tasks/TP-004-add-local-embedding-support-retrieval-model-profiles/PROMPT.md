# Task: TP-004 — add-local-embedding-support / retrieval-model-profiles

**Created:** 2026-04-15
**Change:** add-local-embedding-support
**Size:** M

## Review Level: 2 (Plan + Code)

**Assessment:** Capability implementation packet generated from the approved planner contract.
**Score:** 4/8 — Blast radius: 1, Pattern novelty: 1, Security: 0, Reversibility: 2

## Canonical Task Folder

```text
openspec/tasks/TP-004-add-local-embedding-support-retrieval-model-profiles/
├── PROMPT.md   ← This file (immutable above --- divider)
├── STATUS.md   ← Execution state (worker updates this)
├── .reviews/   ← Reviewer output (created by the orchestrator runtime)
└── .DONE       ← Created when complete
```

## Mission

Implement the approved capability `retrieval-model-profiles` for change `add-local-embedding-support`. This task realizes one approved contract slice from the planner proposal, design, and delta spec without inventing new behavior or broadening scope.

**Change summary:**
Add feature-gated local llama.cpp embedding and reranking support for the exact Hugging Face GGUF models used by QMD, and correct the crate’s retrieval semantics by making backend dialect, model family, embedding role, title metadata, and query-side instruction/task text explicit in the public API.

**Requested delta for this task:**
Introduce a corrected retrieval contract that separates backend dialect from model-family formatting, adds first-class embedding role and metadata, and adds a feature-gated local llama.cpp backend for the exact Gemma and Qwen3 GGUF models in scope.

**Capability requirements:**
- Embedding inputs model retrieval role
- Query-side embedding instructions
- Gemma retrieval formatting
- Qwen3 retrieval formatting
- Formatting is crate-owned behavior

**Behavioral semantics:**
- Embedding requests are semantic, not raw-string-only: the crate receives query/document role and formats the final text internally.
- `ModelFamily::Gemma` uses retrieval prefixes for queries and title/text formatting for documents.
- `ModelFamily::Qwen3` uses instruction-aware query formatting and plain or title-prefixed document formatting.
- `Dialect::LlamaCpp` executes embedding and reranking locally through llama.cpp using supported Hugging Face GGUF artifacts.
- Remote transports remain supported, but transport compatibility does not override model-family formatting rules.

**Failure / edge-case semantics:**
- Constructing a `LlamaCpp` client without the `local` feature fails with an explicit unsupported-configuration error.
- Unsupported dialect/family combinations fail at construction time with explicit validation errors.
- Local reranking returns exactly one score per input document and preserves input-to-score correspondence.
- Embedding instructions apply only to query inputs and are ignored for document inputs.
- Gemma document formatting uses `title: none | text: ...` when no title is supplied.

## Dependencies

- **None**

## Context to Read First

**Tier 2 (area context):**
- `openspec/tasks/CONTEXT.md`
- `openspec/tasks/PHASE-IMPLEMENTATION.md` — implementation-phase guidance for planner-compiled packets

**Tier 3 (load only if needed):**
- `openspec/changes/add-local-embedding-support/proposal.md` — approved change intent and scope boundaries
- `openspec/changes/add-local-embedding-support/design.md` — requested delta, preservation constraints, and proof obligations
- `openspec/changes/add-local-embedding-support/specs/retrieval-model-profiles/spec.md` — capability requirements and scenarios

## Environment

- **Workspace:** Project root
- **Services required:** None

## File Scope

- `src/lib.rs`
- `src/config.rs`
- `src/embedding.rs`
- `src/reranker.rs`
- `src/local.rs`
- `src/local/*.rs`

## Exact Edit Targets

Primary implementation work is owned by:

- `Cargo.toml`
- `README.md`
- `src/lib.rs`
- `src/config.rs`
- `src/embedding.rs`
- `src/reranker.rs`

New local runtime support may add new modules under one of these owned paths:

- `src/local.rs`
- `src/local/*.rs`

No unrelated modules are in scope for semantic rewrites outside these files.

## Public Interface Delta

- Replace the misleading dialect model with `Dialect::{OpenAI, DeepInfra, LlamaCpp}`.
- Add `ModelFamily::{Gemma, Qwen3}` to capture retrieval semantics.
- Extend embedding input/config types to carry role, optional title, and embedding instruction/task text.
- Update crate docs and examples in `README.md`, `src/lib.rs`, `src/embedding.rs`, `src/reranker.rs`, and `src/config.rs` to reflect the new semantic API.

## Preservation Constraints

- Preserve the existing remote embedding and reranking client entry points while expanding the semantic model behind them.
- Preserve async public APIs for callers even though local execution is internally blocking.
- Preserve the current reranker instruction concept while extending the same idea to embedding queries.
- Preserve the single supported provider/client path as the authoritative way to use the crate.

## Steps

### Step 0: Preflight

- [ ] Read the contract references and confirm the task remains within the approved change contract
- [ ] Confirm the current code snapshot still matches the approved edit surface before modifying files

### Step 1: Implement the approved capability slice

- [ ] Apply the requested delta for `retrieval-model-profiles` inside the approved module ownership and edit surface
- [ ] Implement the approved behavioral semantics and exact failure or edge-case semantics from the design and spec
- [ ] Preserve all stated constraints and do not alter interfaces beyond the approved delta

**Artifacts:**
- `src/lib.rs`
- `src/config.rs`
- `src/embedding.rs`
- `src/reranker.rs`
- `src/local.rs`
- `src/local/*.rs`

### Step 2: Tests and verification work

- [ ] Add or update the required tests for this capability, including adversarial coverage
- [ ] Run targeted validation while iterating on the implementation

### Step 3: Documentation and examples

- [ ] Update the required docs and examples for this capability slice
- [ ] Review adjacent docs or examples listed in the proof obligations and update them if affected

### Step 4: Repo gates

- [ ] Run the required repo gates and leave the repository in a fully passing state

## Testing & Verification

### Required Tests
- Unit tests for Gemma query/document formatting, including title handling.
- Unit tests for Qwen3 query/document formatting, including instruction defaults and overrides.
- Construction-time validation tests for unsupported `LlamaCpp` usage without the `local` feature.
- Reranking tests that verify score count and stable index correspondence.

### Repo Gates
- `mise format`
- `mise test`

## Documentation Requirements

**Must Update:**
- `README.md`
- `src/lib.rs`
- any doctest or inline usage examples in `src/embedding.rs`, `src/reranker.rs`, and `src/config.rs`

**Check If Affected:**
- `openspec/changes/add-local-embedding-support/proposal.md` — keep approved change summary aligned if documentation wording or scope evidence changes
- `openspec/changes/add-local-embedding-support/design.md` — keep edit surface and proof obligations aligned if implementation reveals a contract discrepancy

## Completion Criteria

- [ ] All steps complete
- [ ] All tests passing
- [ ] Documentation updated as required by the approved contract

### Acceptance
- The crate can create local embedding clients for the scoped Gemma and Qwen3 GGUF models when `local` is enabled.
- The crate can create a local reranking client for the scoped Qwen3 reranker GGUF model when `local` is enabled.
- Embedding formatting differs correctly by `ModelFamily` and `EmbeddingRole`.
- The current embedding dialect bug is removed from the public semantic model.

### Non-Regression
- Existing remote embedding flows still work for supported remote dialects.
- Existing remote reranking flows still work for supported remote dialects.
- Existing reranker instruction behavior is preserved.

## Git Commit Convention

- **Step completion:** `feat(TP-004): complete Step N — description`
- **Bug fixes:** `fix(TP-004): description`
- **Tests:** `test(TP-004): description`
- **Hydration:** `hydrate: TP-004 expand Step N checkboxes`

## Do NOT

- Do not broaden this task into an architectural rewrite
- Do not alter public or cross-module interfaces beyond the approved delta
- Do not weaken tests, adversarial coverage, docs, or examples to make the task pass
- Do not skip repo gates
- Escalate instead of guessing if the code snapshot no longer matches the approved edit surface

---

## Amendments (Added During Execution)

<!-- Workers add amendments here if execution discovers contradictions in the approved task packet. -->
