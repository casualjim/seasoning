## Why

Seasoning’s current retrieval abstractions are subtly wrong for modern embedding and reranking models. The crate conflates backend transport with semantic model behavior, treats the embedding dialect as if DeepInfra had a distinct implementation when it does not, and does not model the query-versus-document distinction that Qwen3 and EmbeddingGemma require for correct retrieval formatting.

## Change Summary

Add feature-gated local llama.cpp embedding and reranking support for the exact Hugging Face GGUF models used by QMD, and correct the crate’s retrieval semantics by making backend dialect, model family, embedding role, title metadata, and query-side instruction/task text explicit in the public API.

## What Changes

- Add local llama.cpp-backed embedding support for Hugging Face hosted GGUF models behind a feature-gated `LlamaCpp` dialect.
- Add local llama.cpp-backed reranking support for the Qwen3 reranker family using the same Hugging Face GGUF sourcing model.
- Introduce model-family-aware formatting behavior for supported retrieval families (`Gemma` and `Qwen3`) so the crate, not the caller, owns retrieval semantics.
- Extend embedding inputs/configuration to capture retrieval role (`Query` vs `Document`), document title metadata, and embedding instructions/task descriptions where applicable.
- Fix the current embedding dialect bug by making dialects and model-family behavior explicit instead of relying on misleading transport-only labels.
- Keep generation/query-expansion support out of scope for this change.

## Capabilities

### New Capabilities
- `local-embeddings`: Generate embeddings locally through llama.cpp using supported Hugging Face GGUF models.
- `local-reranking`: Score documents locally through llama.cpp using supported Qwen3 reranker GGUF models.
- `retrieval-model-profiles`: Apply model-family-aware retrieval formatting and input semantics for Gemma and Qwen3.

### Modified Capabilities
- None.

## Impact

- Public config and API surface for embedding and reranking clients.
- Formatting and validation logic in `src/embedding.rs`, `src/reranker.rs`, and `src/config.rs`.
- Optional dependencies and Cargo features for `llama-cpp-2` and `hf-hub`, including `cuda`, `metal`, and `vulkan` passthrough.
- Documentation, examples, and release messaging because this is a minor-version semantic expansion and bug fix.
