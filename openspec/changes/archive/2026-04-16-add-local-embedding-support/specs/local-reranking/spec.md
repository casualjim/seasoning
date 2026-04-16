## ADDED Requirements

### Requirement: Feature-gated local reranking backend
The system SHALL provide a local reranking backend through `Dialect::LlamaCpp` when the crate is built with the `local` feature.

#### Scenario: Supported local reranker model
- **WHEN** the crate is built with `local` enabled and a caller configures a reranker client with `Dialect::LlamaCpp` and `hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf`
- **THEN** the client SHALL construct successfully and score documents through the local llama.cpp runtime

#### Scenario: Local reranking without local feature
- **WHEN** the crate is built without the `local` feature and a caller configures a reranker client with `Dialect::LlamaCpp`
- **THEN** client construction SHALL fail with an explicit unsupported-configuration error

### Requirement: Local reranking preserves input-to-score mapping
The local reranking backend SHALL return exactly one score per supplied document in the same input order expected by the public reranking API.

#### Scenario: Score count matches document count
- **WHEN** a caller reranks three documents with the local reranking backend
- **THEN** the backend SHALL return three scores

#### Scenario: Score positions match document positions
- **WHEN** a caller reranks documents in a specific input order with the local reranking backend
- **THEN** each returned score SHALL correspond to the document at the same input index

### Requirement: Instruction-aware Qwen3 reranking
The local reranking backend SHALL support Qwen3 reranker instruction formatting and SHALL allow callers to override the default retrieval instruction.

#### Scenario: Default reranking instruction
- **WHEN** a caller omits a reranking instruction for a Qwen3 local reranker
- **THEN** the backend SHALL apply the crate’s default retrieval instruction when building the model input

#### Scenario: Custom reranking instruction
- **WHEN** a caller supplies a reranking instruction for a Qwen3 local reranker
- **THEN** the backend SHALL use the caller-provided instruction when building the model input
