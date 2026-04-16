# Conformance Report: add-local-embedding-support

**Status:** Complete
**Verdict:** REMEDIATION_TASK

## Summary

The assembled implementation largely satisfies the requested semantic retrieval delta (explicit dialect/model-family/role semantics, family-owned formatting behavior, local GGUF allowlists, and updated docs/examples). Required formatting and adversarial reranking tests are present, and repo gates (`mise format`, `mise test`) pass in the current lane.

However, conformance is **not archive-ready** because blocking findings remain:

1. Wave 1 modified `.mise` repo-gate files that were outside TP-004’s approved edit surface.
2. The repo-gate implementation now runs **redundant multi-suite nextest runs** (`default`, `local`, and Darwin `metal`) instead of selecting exactly one platform-appropriate local feature probe.
3. Acceptance proof for successful local client construction is incomplete: tests cover negative local-construction paths but do not include positive `--features local` constructor success cases for the supported local models.

## Findings

### CRITICAL

- **F-001 — Out-of-scope repo-gate edits in TP-004 packet**
  - **What:** Wave 1 changed `mise.toml`, `.mise/tasks/test/_default`, and `.mise/tasks/test/rust` even though TP-004 scope/ownership excludes these files.
  - **Why it matters:** This violates approved packet boundaries and weakens contract conformance for the implementation wave.
  - **Disposition:** `REMEDIATION_TASK`

- **F-002 — Repo gate runs redundant test suites instead of single platform-selected feature probe**
  - **What:** Current gate runs `cargo nextest run`, `cargo nextest run --features local`, and (on Darwin) `cargo nextest run --features metal`.
  - **Expected shape (operator guidance):** choose exactly one local feature path per environment: Darwin→`metal`; else CUDA available→`cuda`; else Vulkan available→`vulkan`; else `local`.
  - **Why it matters:** Introduces unnecessary duplicate gate execution and diverges from expected platform/toolchain selection behavior.
  - **Disposition:** `REMEDIATION_TASK`

- **F-003 — Acceptance proof gap for positive local client construction**
  - **What:** Design acceptance requires successful construction of supported local embedding/reranking clients with `local` enabled, but tests only verify failure cases for unsupported models under `feature = "local"`.
  - **Why it matters:** The proof corpus does not fully demonstrate the acceptance obligations for local support.
  - **Disposition:** `REMEDIATION_TASK`

### WARNING

- None.

### SUGGESTION

- Keep adversarial score-count/index coverage and formatting tests as-is; they provide good evidence for retrieval semantics and ordering guarantees.

## Evidence

- `openspec/tasks/TP-004-add-local-embedding-support-retrieval-model-profiles/PROMPT.md:73-98` (approved file scope/edit targets exclude `.mise` and `mise.toml`)
- `mise.toml:16-25` (`tasks.test` runs multiple nextest suites)
- `.mise/tasks/test/_default:9-18` (default + local + Darwin metal runs)
- `.mise/tasks/test/rust:12-21` (same redundant multi-suite behavior)
- `openspec/changes/add-local-embedding-support/design.md:136-137` (acceptance requires successful local client construction)
- `src/embedding.rs:322-346` (`feature = "local"` test covers unsupported-model rejection; no positive supported-model construction test)
- `src/reranker.rs:653-676` (`feature = "local"` test covers unsupported-model rejection; no positive supported-model construction test)
- `src/api.rs:307-429` (Gemma/Qwen3 formatting tests)
- `src/reranker.rs:442-460` and `src/reranker.rs:679-734` (adversarial rerank mismatch/index coverage)

## Disposition

- REMEDIATION_TASK
