# seasoning

Retrieval-focused embedding and reranking infrastructure for Rust.

## What this gives you

- Semantic embedding inputs with explicit query/document roles.
- Model-family-aware formatting for Gemma and Qwen3 retrieval models.
- Remote OpenAI-compatible and DeepInfra backends.
- Feature-gated local llama.cpp execution for the scoped Hugging Face GGUF models.
- Rate limiting, retry handling, and async public APIs.

## Semantic model

Seasoning now separates **backend dialect** from **model-family behavior**:

- `Dialect::{OpenAI, DeepInfra, LlamaCpp}` chooses the transport/runtime.
- `ModelFamily::{Gemma, Qwen3}` chooses retrieval formatting semantics.
- `EmbeddingRole::{Query, Document}` tells the crate how to format each embedding input.

That means callers pass semantic inputs and the crate formats final model text internally.

## Install

Remote-only usage:

```toml
[dependencies]
seasoning = { path = "." }
```

Local llama.cpp usage:

```toml
[dependencies]
seasoning = { path = ".", features = ["local"] }
```

Accelerator passthrough features imply `local`:

```toml
[dependencies]
seasoning = { path = ".", features = ["cuda"] }
# or: ["metal"], ["vulkan"]
```

## Usage

### Embeddings

```rust,no_run
use std::time::Duration;

use secrecy::SecretString;
use seasoning::EmbeddingProvider;
use seasoning::embedding::{
    Client as EmbedClient, Dialect, EmbedderConfig, EmbeddingInput, EmbeddingRole, ModelFamily,
};

# async fn example() -> seasoning::Result<()> {
let embedder = EmbedClient::new(EmbedderConfig {
    api_key: Some(SecretString::from("YOUR_API_KEY")),
    base_url: "https://api.deepinfra.com/v1/openai".to_string(),
    timeout: Duration::from_secs(10),
    dialect: Dialect::DeepInfra,
    model_family: ModelFamily::Qwen3,
    model: "Qwen/Qwen3-Embedding-0.6B".to_string(),
    query_instruction: Some("Given a user query, retrieve matching passages".to_string()),
    embedding_dim: 1024,
    requests_per_minute: 1000,
    max_concurrent_requests: 50,
    tokens_per_minute: 1_000_000,
})?;

let inputs = vec![
    EmbeddingInput {
        role: EmbeddingRole::Query,
        text: "memory safety without garbage collection".to_string(),
        title: None,
        token_count: 6,
    },
    EmbeddingInput {
        role: EmbeddingRole::Document,
        text: "Rust prevents data races at compile time".to_string(),
        title: Some("Rust overview".to_string()),
        token_count: 8,
    },
];

let result = embedder.embed(&inputs).await?;
println!("got {} embeddings", result.embeddings.len());
# Ok(())
# }
```

### Reranking

```rust,no_run
use std::time::Duration;

use secrecy::SecretString;
use seasoning::RerankingProvider;
use seasoning::embedding::{Dialect, ModelFamily};
use seasoning::reranker::{Client as RerankerClient, RerankerConfig};

# async fn example() -> seasoning::Result<()> {
let reranker = RerankerClient::new(RerankerConfig {
    api_key: Some(SecretString::from("YOUR_API_KEY")),
    base_url: "https://api.deepinfra.com/v1".to_string(),
    timeout: Duration::from_secs(10),
    dialect: Dialect::DeepInfra,
    model_family: ModelFamily::Qwen3,
    model: "Qwen/Qwen3-Reranker-0.6B".to_string(),
    instruction: None,
    requests_per_minute: 1000,
    max_concurrent_requests: 50,
    tokens_per_minute: 1_000_000,
})?;

let query = seasoning::RerankQuery {
    text: "memory-safe systems programming".to_string(),
    token_count: 4,
};
let docs = vec![
    seasoning::RerankDocument {
        text: "Rust offers ownership and borrowing".to_string(),
        token_count: 6,
    },
    seasoning::RerankDocument {
        text: "Python emphasizes developer ergonomics".to_string(),
        token_count: 5,
    },
];

let scores = reranker.rerank(&query, &docs).await?;
println!("{scores:?}");
# Ok(())
# }
```

### Local llama.cpp embeddings and reranking

```rust,ignore
use std::time::Duration;

use seasoning::embedding::{Client as EmbedClient, Dialect, EmbedderConfig, ModelFamily};
use seasoning::reranker::{Client as RerankerClient, RerankerConfig};

let embedder = EmbedClient::new(EmbedderConfig {
    api_key: None,
    base_url: String::new(),
    timeout: Duration::from_secs(30),
    dialect: Dialect::LlamaCpp,
    model_family: ModelFamily::Gemma,
    model: "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf".to_string(),
    query_instruction: None,
    embedding_dim: 768,
    requests_per_minute: 1,
    max_concurrent_requests: 1,
    tokens_per_minute: 1_000_000,
})?;

let reranker = RerankerClient::new(RerankerConfig {
    api_key: None,
    base_url: String::new(),
    timeout: Duration::from_secs(30),
    dialect: Dialect::LlamaCpp,
    model_family: ModelFamily::Qwen3,
    model: "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf".to_string(),
    instruction: None,
    requests_per_minute: 1,
    max_concurrent_requests: 1,
    tokens_per_minute: 1_000_000,
})?;
# let _ = (embedder, reranker);
# Ok::<(), seasoning::Error>(())
```

Supported local GGUF artifacts are intentionally narrow for this change:

- `hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf`
- `hf:Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf`
- `hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf`

## Notes

- Local `Dialect::LlamaCpp` construction fails explicitly when the crate is built without the `local` feature.
- Gemma document formatting uses `title: none | text: ...` when no title is supplied.
- Qwen3 query instructions apply only to query embeddings; document embeddings ignore them.
- Existing remote entry points are preserved, but retrieval semantics now come from `ModelFamily` rather than transport labels.

## Modules

- `seasoning::embedding` for embeddings and retrieval formatting inputs
- `seasoning::reranker` for reranking
- `seasoning::reqwestx` for the rate-limited API client
- `seasoning::config` for config structs (no I/O)
