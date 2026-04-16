# Local Embeddings

## Purpose

Provide local embedding inference capabilities via llama.cpp when the `local` feature is enabled, with GPU acceleration support through feature-gated accelerator backends.

## Requirements

### Requirement: Feature-gated local embedding backend
The system SHALL provide a local embedding backend through `Dialect::LlamaCpp` when the crate is built with the `local` feature.

#### Scenario: Supported local embedding model
- **WHEN** the crate is built with `local` enabled and a caller configures an embedding client with `Dialect::LlamaCpp` and a supported local embedding model
- **THEN** the client SHALL construct successfully and return embeddings from the local llama.cpp runtime

#### Scenario: Local dialect without local feature
- **WHEN** the crate is built without the `local` feature and a caller configures an embedding client with `Dialect::LlamaCpp`
- **THEN** client construction SHALL fail with an explicit unsupported-configuration error

### Requirement: Hugging Face GGUF sourcing for local embeddings
The local embedding backend SHALL resolve supported GGUF models from Hugging Face hosted artifacts instead of requiring callers to pre-download or pre-format model paths.

#### Scenario: Supported Gemma embedding model
- **WHEN** a caller configures the local embedding backend with `hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf`
- **THEN** the backend SHALL resolve the model artifact from Hugging Face and use it for local embedding inference

#### Scenario: Supported Qwen3 embedding model
- **WHEN** a caller configures the local embedding backend with `hf:Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf`
- **THEN** the backend SHALL resolve the model artifact from Hugging Face and use it for local embedding inference

### Requirement: Accelerator feature passthrough
The crate SHALL expose `cuda`, `metal`, and `vulkan` Cargo features for local embeddings, and each accelerator feature SHALL imply `local`.

#### Scenario: CUDA implies local support
- **WHEN** a consumer builds the crate with the `cuda` feature
- **THEN** local llama.cpp embedding support SHALL also be enabled

#### Scenario: Metal implies local support
- **WHEN** a consumer builds the crate with the `metal` feature
- **THEN** local llama.cpp embedding support SHALL also be enabled

#### Scenario: Vulkan implies local support
- **WHEN** a consumer builds the crate with the `vulkan` feature
- **THEN** local llama.cpp embedding support SHALL also be enabled
