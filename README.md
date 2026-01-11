# seasoning

Embedding + reranking infrastructure as a Rust library.

## What this gives you

- An embedding client with rate limiting and retries.
- A reranker client (DeepInfra API shape).
- Simple config structs you can wire into your own config loader.

## Install

This repo is not published to crates.io. Use a path dependency:

```toml
[dependencies]
seasoning = { path = "." }
```

## Usage

### Embeddings

```rust
use std::time::Duration;

use secrecy::SecretString;
use seasoning::embedding::{
    Client as EmbedClient, EmbedderConfig, EmbeddingInput, ProviderDialect,
};

#[tokio::main]
async fn main() -> seasoning::Result<()> {
    let embedder = EmbedClient::new(EmbedderConfig {
        api_key: Some(SecretString::from("YOUR_API_KEY")),
        base_url: "https://api.deepinfra.com/v1/openai".to_string(),
        timeout: Duration::from_secs(10),
        dialect: ProviderDialect::DeepInfra,
        model: "Qwen/Qwen3-Embedding-0.6B".to_string(),
        embedding_dim: 1024,
        requests_per_minute: 1000,
        max_concurrent_requests: 50,
        tokens_per_minute: 1_000_000,
    })?;

    let inputs = vec![
        EmbeddingInput {
            text: "hello world".to_string(),
            token_count: 2,
        },
        EmbeddingInput {
            text: "another string".to_string(),
            token_count: 2,
        },
    ];

    let result = embedder.embed(&inputs).await?;
    println!("got {} embeddings", result.embeddings.len());
    Ok(())
}
```

Notes:
- `token_count` is required per input so the rate limiter can budget tokens.
- `requests_per_minute`, `max_concurrent_requests`, and `tokens_per_minute`
  control the rate limiter and concurrency.

### Reranking

```rust
use std::time::Duration;

use secrecy::SecretString;
use seasoning::embedding::ProviderDialect;
use seasoning::reranker::{Client as RerankerClient, RerankerConfig};
use seasoning::RerankingProvider;

#[tokio::main]
async fn main() -> seasoning::Result<()> {
    let reranker = RerankerClient::new(RerankerConfig {
        api_key: Some(SecretString::from("YOUR_API_KEY")),
        base_url: "https://api.deepinfra.com/v1".to_string(),
        timeout: Duration::from_secs(10),
        dialect: ProviderDialect::DeepInfra,
        model: "Qwen/Qwen3-Reranker-0.6B".to_string(),
        instruction: None,
        requests_per_minute: 1000,
        max_concurrent_requests: 50,
        tokens_per_minute: 1_000_000,
    })?;

    let query = seasoning::RerankQuery {
        text: "search query".to_string(),
        token_count: 2,
    };
    let docs = vec![
        seasoning::RerankDocument {
            text: "first doc".to_string(),
            token_count: 2,
        },
        seasoning::RerankDocument {
            text: "second doc".to_string(),
            token_count: 2,
        },
        seasoning::RerankDocument {
            text: "third doc".to_string(),
            token_count: 2,
        },
    ];
    let scores = reranker.rerank(&query, &docs).await?;
    println!("{scores:?}");
    Ok(())
}
```

Notes:
- The DeepInfra endpoint is built as `{base_url}/inference/{model}`.
- The reranker client currently supports `ProviderDialect::DeepInfra` only.

## Modules

- `seasoning::embedding` for embeddings
- `seasoning::reranker` for reranking
- `seasoning::reqwestx` for the rate-limited API client
- `seasoning::config` for config structs (no I/O)
