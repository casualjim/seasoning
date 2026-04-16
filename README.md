# seasoning

Retrieval-focused embedding and reranking infrastructure for Rust.

## What this gives you

- Semantic embedding inputs with explicit query/document roles.
- Model-family-aware formatting for Gemma and Qwen3 retrieval models.
- Remote OpenAI-compatible and DeepInfra backends.
- Optional local llama.cpp execution for Hugging Face GGUF models.
- Rate limiting, retry handling, and async public APIs.

## Semantic model

Seasoning separates **backend dialect** from **model-family behavior**:

- `Dialect::{OpenAI, DeepInfra, LlamaCpp}` chooses the backend/runtime.
- `ModelFamily::{Gemma, Qwen3}` chooses retrieval formatting semantics.
- `EmbeddingRole::{Query, Document}` tells the crate how to format each embedding input.

You pass semantic embedding inputs. Seasoning renders the model-specific text and tokenizes it internally before execution.

For example:
- Gemma queries become `task: <task> | query: <text>`.
- Gemma documents become `title: <title-or-none> | text: <text>`.
- Qwen3 queries become `Instruct: <instruction>\nQuery: <text>`.
- Qwen3 documents stay plain text unless a title is present, in which case the title is prepended on its own line.

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
use std::sync::Arc;
use std::time::Duration;

use secrecy::SecretString;
use seasoning::EmbeddingProvider;
use seasoning::embedding::{
    Client as EmbedClient, Dialect, EmbedderConfig, EmbeddingInput, EmbeddingRole, ModelFamily,
    RemoteEmbedderConfig, Tokenizer,
};

# async fn example() -> seasoning::Result<()> {
let embedder = EmbedClient::new(EmbedderConfig::remote(
    ModelFamily::Qwen3,
    Tokenizer::Tiktoken {
        encoding: "cl100k_base".to_string(),
        tokenizer: Arc::new(tiktoken_rs::cl100k_base()?),
    },
    "Qwen/Qwen3-Embedding-0.6B",
    Some("Given a user query, retrieve matching passages".to_string()),
    RemoteEmbedderConfig {
        api_key: Some(SecretString::from("YOUR_API_KEY")),
        base_url: "https://api.deepinfra.com/v1/openai".to_string(),
        timeout: Duration::from_secs(10),
        dialect: Dialect::DeepInfra,
        embedding_dim: 1024,
        requests_per_minute: 1000,
        max_concurrent_requests: 50,
        tokens_per_minute: 1_000_000,
    },
)?)?;

let inputs = vec![EmbeddingInput {
    role: EmbeddingRole::Query,
    text: "memory safety without garbage collection".to_string(),
    title: None,
    token_count: 6,
}];
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
use std::sync::Arc;
use std::time::Duration;

use seasoning::embedding::{Client as EmbedClient, Dialect, EmbedderConfig, ModelFamily, Tokenizer};
use seasoning::reranker::{Client as RerankerClient, RerankerConfig};

let embedder = EmbedClient::new(EmbedderConfig::local(
    ModelFamily::Gemma,
    Tokenizer::HuggingFace {
        model_id: "google/embeddinggemma-300m".to_string(),
        tokenizer: Arc::new(tokenizers::Tokenizer::from_pretrained("google/embeddinggemma-300m", None)?),
    },
    "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf",
    None,
))?;

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

Example local GGUF model ids:


When a local `hf:` GGUF artifact needs to be fetched from Hugging Face, download progress is enabled by default. You can control it with environment variables:

- `SEASONING_HF_HUB_PROGRESS=0|1|false|true|off|on`
- `HF_HUB_DISABLE_PROGRESS_BARS=1|0|true|false|off|on`

If both are set, `SEASONING_HF_HUB_PROGRESS` wins.

- `hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf`
- `hf:Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf`
- `hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf`

## Notes

- Build with the `local` feature to use `Dialect::LlamaCpp`.
- Gemma documents use `title: none | text: ...` when no title is supplied.
- Qwen3 query instructions apply only to query embeddings.
- `ModelFamily` controls embedding formatting semantics and local reranker prompt formatting.

## Modules

- `seasoning::embedding` for embeddings and retrieval formatting inputs
- `seasoning::reranker` for reranking
- `seasoning::reqwestx` for the rate-limited API client
- `seasoning::{AppConfig, Embedding, Reranker}` for config structs
