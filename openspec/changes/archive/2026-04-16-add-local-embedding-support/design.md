## Context

Seasoning currently mixes transport concerns with model semantics. On the embedding side, `ProviderDialect::DeepInfra` is not actually a distinct implementation: both `OpenAI` and `DeepInfra` use the same OpenAI-style `/embeddings` request shape, so the current dialect abstraction is misleading. At the same time, the crate exposes embedding inputs as raw strings even though the supported retrieval families we care about (`Gemma` and `Qwen3`) require different query/document formatting for correct retrieval behavior.

This change also introduces a second execution backend. Remote calls are async HTTP requests with retries and rate limiting; local llama.cpp calls are synchronous, stateful, and model/context driven. The design needs to support both without pushing formatting or runtime details back onto the caller.

Example local model references for this change are aligned with QMD usage:

- Embeddings:
  - `hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf`
  - `hf:Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf`
- Reranking:
  - `hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf`

These model ids are documented examples, not an exclusive allowlist or support policy boundary for all valid `hf:<repo>/<file>.gguf` artifacts.

## Goals / Non-Goals

**Goals:**
- Represent backend transport/runtime explicitly with `Dialect = OpenAI | DeepInfra | LlamaCpp`.
- Represent retrieval semantics explicitly with `ModelFamily = Gemma | Qwen3`.
- Represent embedding role explicitly with `EmbeddingRole = Query | Document`.
- Move retrieval formatting into the crate so callers provide semantics, not pre-formatted strings.
- Add feature-gated local llama.cpp embedding and reranking support with Hugging Face GGUF sourcing.
- Preserve existing remote functionality while fixing the misleading embedding dialect behavior.

**Non-Goals:**
- Query expansion or general text-generation support.
- Broad support for arbitrary model families beyond Gemma and Qwen3.
- Public free-function formatting utilities for external callers.
- Arbitrary local file-path model loading in the first version; the supported path is Hugging Face hosted GGUF resolution.
- A generalized local runtime manager optimized for multiple external consumers.

## Decisions

### 1. Separate backend dialect from model-family semantics
We will keep a single top-level dialect enum, but redefine it to mean backend/transport profile rather than vendor label:

- `OpenAI`: OpenAI-style remote API
- `DeepInfra`: DeepInfra-specific remote API behavior where it differs, especially for reranking
- `LlamaCpp`: local llama.cpp runtime

Formatting is not owned by `Dialect`. `ModelFamily` owns semantic formatting behavior and `EmbeddingRole` indicates whether the input is a query or document.

### 2. Make embedding semantics first-class in the API
Embedding inputs will no longer be treated as undifferentiated text. The public embedding request surface will include:

- role (`Query` or `Document`)
- text
- optional title
- token count (for remote budgeting and batching)

Embedding configuration will also gain a query-side instruction/task field.

### 3. Put formatting behavior on the enums that drive it
The enums that influence formatting will own the formatting logic through methods rather than through exported free functions. `ModelFamily` is the canonical owner of final string construction, with `EmbeddingRole` contributing role semantics.

Expected behavior:

- `Gemma` query → `task: <task> | query: <text>`
- `Gemma` document → `title: <title-or-none> | text: <text>`
- `Qwen3` query → `Instruct: <instruction>\nQuery: <text>`
- `Qwen3` document → raw text, or `title + "\n" + text` when title is present

The query-side instruction/task is applied only when the role is `Query`.

### 4. Keep reranker instruction support and align it with family-aware behavior
The existing reranker `instruction` field remains. For Qwen3 reranking, it maps onto the model’s instruction-aware pair formatting. Embedding configuration gains a parallel instruction/task concept for query embeddings.

### 5. Add a feature-gated local runtime with opinionated Hugging Face GGUF support
Local support is added behind `local`, with accelerator passthrough features:

- `local`
- `cuda` → implies `local`
- `metal` → implies `local`
- `vulkan` → implies `local`

`llama-cpp-2` and `hf-hub` are optional dependencies enabled through these features. The first version accepts Hugging Face hosted GGUF artifacts in `hf:<repo>/<file>.gguf` form, while documenting QMD-aligned examples for the initial supported retrieval families.

### 6. Use an internal blocking local runtime, not a shared external orchestration layer
Local llama.cpp work will run inside the crate, using lazy model initialization and internal synchronization around mutable context access. Calls from async APIs will delegate to blocking work rather than pretending llama.cpp inference is natively async.

## Requested Delta

Introduce a corrected retrieval contract that separates backend dialect from model-family formatting, adds first-class embedding role and metadata, and adds a feature-gated local llama.cpp backend for the exact Gemma and Qwen3 GGUF models in scope.

## Preservation Constraints

- Preserve the existing remote embedding and reranking client entry points while expanding the semantic model behind them.
- Preserve async public APIs for callers even though local execution is internally blocking.
- Preserve the current reranker instruction concept while extending the same idea to embedding queries.
- Preserve the single supported provider/client path as the authoritative way to use the crate.

## Public Interface Deltas

- Replace the misleading dialect model with `Dialect::{OpenAI, DeepInfra, LlamaCpp}`.
- Add `ModelFamily::{Gemma, Qwen3}` to capture retrieval semantics.
- Extend embedding input/config types to carry role, optional title, and embedding instruction/task text.
- Update crate docs and examples in `README.md`, `src/lib.rs`, `src/embedding.rs`, `src/reranker.rs`, and `src/config.rs` to reflect the new semantic API.

## Module Ownership and Edit Surface

Primary implementation work is owned by:

- `Cargo.toml`
- `README.md`
- `src/lib.rs`
- `src/config.rs`
- `src/embedding.rs`
- `src/reranker.rs`
- `src/error.rs`
- `src/batching.rs`
- `src/service.rs`

New local runtime support may add new modules under one of these owned paths:

- `src/local.rs`
- `src/local/*.rs`

Supporting changes in batching, service orchestration, and error surfaces are in scope where they are required to preserve semantic embedding behavior and local execution correctness.

## Behavioral Semantics

- Embedding requests are semantic, not raw-string-only: the crate receives query/document role and formats the final text internally.
- `ModelFamily::Gemma` uses retrieval prefixes for queries and title/text formatting for documents.
- `ModelFamily::Qwen3` uses instruction-aware query formatting and plain or title-prefixed document formatting.
- `Dialect::LlamaCpp` executes embedding and reranking locally through llama.cpp using supported Hugging Face GGUF artifacts.
- Remote transports remain supported, but transport compatibility does not override model-family formatting rules.

## Failure / Edge Case Semantics

- Constructing a `LlamaCpp` client without the `local` feature fails with an explicit unsupported-configuration error.
- Unsupported dialect/family combinations fail at construction time with explicit validation errors.
- Local reranking returns exactly one score per input document and preserves input-to-score correspondence.
- Embedding instructions apply only to query inputs and are ignored for document inputs.
- Gemma document formatting uses `title: none | text: ...` when no title is supplied.

## Proof Obligations

### Acceptance
- The crate can create local embedding clients for the scoped Gemma and Qwen3 GGUF models when `local` is enabled.
- The crate can create a local reranking client for the scoped Qwen3 reranker GGUF model when `local` is enabled.
- Embedding formatting differs correctly by `ModelFamily` and `EmbeddingRole`.
- The current embedding dialect bug is removed from the public semantic model.

### Non-Regression
- Existing remote embedding flows still work for supported remote dialects.
- Existing remote reranking flows still work for supported remote dialects.
- Existing reranker instruction behavior is preserved.

### Required Tests
- Unit tests for Gemma query/document formatting, including title handling.
- Unit tests for Qwen3 query/document formatting, including instruction defaults and overrides.
- Construction-time validation tests for unsupported `LlamaCpp` usage without the `local` feature.
- Reranking tests that verify score count and stable index correspondence.
- End-to-end local embedding and local reranking tests against real llama.cpp execution paths for documented example GGUF artifacts.

### Documentation and Examples
- `README.md`
- `src/lib.rs`
- any doctest or inline usage examples in `src/embedding.rs`, `src/reranker.rs`, and `src/config.rs`

### Repo Gates
- `mise format`
- `mise test`

## Risks / Trade-offs

- **[Semantic result drift]** Remote and local backends may produce different scores or vectors for the same family due to model/runtime differences → **Mitigation:** scope supported families explicitly, document example models, and test formatting behavior directly.
- **[Native dependency complexity]** `llama-cpp-2` introduces feature- and platform-specific build complexity → **Mitigation:** keep local support feature-gated and pass through `cuda`, `metal`, and `vulkan` explicitly.
- **[Async/runtime impedance mismatch]** llama.cpp inference is blocking and stateful → **Mitigation:** isolate it behind internal blocking execution and synchronization rather than leaking that complexity into public APIs.
- **[Configuration expansion]** Adding role, family, instruction, and title makes the API more explicit and justifies a minor release bump → **Mitigation:** keep naming tight and make invalid combinations fail early with clear errors.
- **[Behavior correction can change outputs]** Existing callers may see different retrieval results once family-aware formatting is applied → **Mitigation:** document the bug fix clearly in proposal, tasks, and release-facing notes.

## Migration Plan

1. Introduce the new enums and configuration fields while preserving the existing remote client entry points.
2. Refactor embedding and reranking formatting to flow through family- and role-aware methods.
3. Add local feature flags and optional dependencies.
4. Implement local llama.cpp embedding support for the scoped Gemma and Qwen3 embedding models.
5. Implement local llama.cpp reranking support for the scoped Qwen3 reranker model.
6. Update docs and examples and call out the semantic bug fix and minor version bump in release messaging.

Rollback is straightforward: disable the `local` feature and revert to the prior remote-only release if local support or semantic changes uncover regressions.

## Closure Status

- Blockers: None
- Known Unknowns: None
- Deferred Design Choices: None

## Open Questions

None. `EmbeddingRole` is the chosen public name for this change, and model family is explicit in configuration for this version.
