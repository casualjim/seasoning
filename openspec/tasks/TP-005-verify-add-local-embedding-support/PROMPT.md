# Task: TP-005 — Verify add-local-embedding-support

**Created:** 2026-04-15
**Change:** add-local-embedding-support
**Size:** M

## Review Level: 3 (Full)

**Assessment:** Terminal whole-change conformance packet generated from the approved planner contract.
**Score:** 6/8 — Blast radius: 2, Pattern novelty: 1, Security: 1, Reversibility: 2

## Canonical Task Folder

```text
openspec/tasks/TP-005-verify-add-local-embedding-support/
├── PROMPT.md   ← This file (immutable above --- divider)
├── STATUS.md   ← Execution state (worker updates this)
├── .reviews/   ← Reviewer output (created by the orchestrator runtime)
└── .DONE       ← Created when complete
```

## Mission

Verify the assembled implementation for `add-local-embedding-support` against the approved proposal, design, and delta specs. This task evaluates whole-change conformance and writes the canonical conformance report. It MUST NOT directly implement fixes.

## Dependencies

- **Task:** TP-002 (implementation packet must complete before conformance runs)
- **Task:** TP-003 (implementation packet must complete before conformance runs)
- **Task:** TP-004 (implementation packet must complete before conformance runs)

## Context to Read First

**Tier 2 (area context):**
- `openspec/tasks/CONTEXT.md`
- `openspec/tasks/PHASE-CONFORMANCE.md` — conformance-phase guidance for planner-compiled packets

**Tier 3 (load only if needed):**
- `openspec/changes/add-local-embedding-support/proposal.md` — approved change intent and scope boundaries
- `openspec/changes/add-local-embedding-support/design.md` — requested delta, preservation constraints, interface rules, and proof obligations
- `openspec/changes/add-local-embedding-support/specs/local-embeddings/spec.md` — delta requirements and scenarios
- `openspec/changes/add-local-embedding-support/specs/local-reranking/spec.md` — delta requirements and scenarios
- `openspec/changes/add-local-embedding-support/specs/retrieval-model-profiles/spec.md` — delta requirements and scenarios

## Environment

- **Workspace:** Project root
- **Services required:** None

## File Scope

- `openspec/changes/add-local-embedding-support/conformance.md`

## Findings Disposition Rules

- **LOG_ONLY** — non-blocking suggestion; record only
- **INLINE_REVISE** — active task can revise in place
- **REMEDIATION_TASK** — fix is inside the approved contract; generate remediation work
- **REOPEN_PLANNING** — fixing the issue would change the approved contract
- **ESCALATE_HUMAN** — human decision required because the contract or trade-off changed materially
- **ARCHIVE_READY** — no blocking findings remain

## Steps

### Step 0: Load the approved contract

- [ ] Read proposal, design, delta specs, and relevant Taskplane artifacts
- [ ] Confirm the implementation packets listed in Dependencies are complete

### Step 1: Evaluate delta and preservation conformance

- [ ] Compare the assembled implementation against the approved requested delta and interface rules
- [ ] Check preservation constraints, non-regression expectations, docs, examples, and proof strength

### Step 2: Evaluate proof obligations and repo gates

- [ ] Confirm the required tests, including adversarial coverage, are present and aligned with the approved contract
- [ ] Confirm repo gates and verification outputs support the final verdict

### Step 3: Write the conformance report

- [ ] Write `openspec/changes/add-local-embedding-support/conformance.md` with findings, evidence, and an explicit verdict
- [ ] Use the approved disposition model for every blocking or non-blocking finding

**Artifacts:**
- `openspec/changes/add-local-embedding-support/conformance.md`

## Documentation Requirements

**Must Update:**
- `openspec/changes/add-local-embedding-support/conformance.md` — canonical conformance report for this change

**Check If Affected:**
- `openspec/changes/add-local-embedding-support/proposal.md` — revisit only if conformance proves the approved scope summary is wrong
- `openspec/changes/add-local-embedding-support/design.md` — revisit only if conformance proves the approved contract is wrong and planning must reopen

## Report Output

Write the final report to `openspec/changes/add-local-embedding-support/conformance.md` using this template:

```markdown
# Conformance Report: add-local-embedding-support

**Status:** Complete
**Verdict:** ARCHIVE_READY

## Summary

<!-- Summarize delta conformance, preservation conformance, and proof conformance. -->

## Findings

### CRITICAL

- None.

### WARNING

- None.

### SUGGESTION

- None.

## Evidence

- <!-- file.ts:123 -->

## Disposition

- ARCHIVE_READY

```

## Completion Criteria

- [ ] The report exists at `openspec/changes/add-local-embedding-support/conformance.md`
- [ ] The report contains an explicit verdict
- [ ] Blocking findings, if any, use the approved disposition model
- [ ] No code was modified directly by this verify task

## Git Commit Convention

- **Step completion:** `feat(TP-005): complete Step N — description`
- **Bug fixes:** `fix(TP-005): description`
- **Tests:** `test(TP-005): description`
- **Hydration:** `hydrate: TP-005 expand Step N checkboxes`

## Do NOT

- Do not implement fixes directly in this task
- Do not weaken proof obligations or rewrite the contract to match the implementation
- Do not archive while blocking findings remain

---

## Amendments (Added During Execution)

<!-- Workers add amendments here if execution discovers contradictions in the approved task packet. -->
