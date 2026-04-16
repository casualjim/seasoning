//! # Seasoning
//!
//! Retrieval-focused embedding and reranking infrastructure with explicit model
//! semantics, rate limiting, retries, and optional local llama.cpp execution.
//!
//! Config-driven local setups accept the `llama.cpp`, `llamacpp`, `llama-cpp`,
//! or `llama_cpp` dialect spellings when converting into [`Dialect::LlamaCpp`].
//!
//! Seasoning separates backend/runtime selection from retrieval formatting:
//! [`Dialect`] selects transport or local execution, [`ModelFamily`] selects
//! retrieval-family formatting, and [`EmbeddingRole`] identifies whether a
//! semantic embedding input is a query or document.
//!
//! Embedding execution consumes pre-tokenized [`PreparedEmbeddingInput`] values.
//! Callers render semantic inputs first, then tokenize the rendered payload with
//! the tokenizer for the target embedding model.
//!
//! ## Embeddings
//!
//! ```rust,no_run
//! use std::time::Duration;
//!
//! use secrecy::SecretString;
//! use seasoning::EmbeddingProvider;
//! use seasoning::embedding::{
//!     Client as EmbedClient, Dialect, EmbedderConfig, EmbeddingInput, EmbeddingRole,
//!     ModelFamily, PreparedEmbeddingInput,
//! };
//!
//! # async fn example() -> seasoning::Result<()> {
//! let embedder = EmbedClient::new(EmbedderConfig {
//!     api_key: Some(SecretString::from("YOUR_API_KEY")),
//!     base_url: "https://api.deepinfra.com/v1/openai".to_string(),
//!     timeout: Duration::from_secs(10),
//!     dialect: Dialect::DeepInfra,
//!     model_family: ModelFamily::Qwen3,
//!     model: "Qwen/Qwen3-Embedding-0.6B".to_string(),
//!     query_instruction: None,
//!     embedding_dim: 1024,
//!     requests_per_minute: 1000,
//!     max_concurrent_requests: 50,
//!     tokens_per_minute: 1_000_000,
//! })?;
//!
//! let semantic = EmbeddingInput {
//!     role: EmbeddingRole::Query,
//!     text: "memory-safe systems programming".to_string(),
//!     title: None,
//! };
//! let rendered = embedder.render_input(&semantic);
//! let _ = rendered;
//!
//! // Tokenize `rendered` with the tokenizer for the target embedding model.
//! let prepared = vec![PreparedEmbeddingInput::new(vec![1, 2, 3])?];
//! let _ = embedder.embed(&prepared).await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Reranking
//!
//! ```rust,no_run
//! use std::time::Duration;
//!
//! use secrecy::SecretString;
//! use seasoning::RerankingProvider;
//! use seasoning::embedding::{Dialect, ModelFamily};
//! use seasoning::reranker::{Client as RerankerClient, RerankerConfig};
//!
//! # async fn example() -> seasoning::Result<()> {
//! let reranker = RerankerClient::new(RerankerConfig {
//!     api_key: Some(SecretString::from("YOUR_API_KEY")),
//!     base_url: "https://api.deepinfra.com/v1".to_string(),
//!     timeout: Duration::from_secs(10),
//!     dialect: Dialect::DeepInfra,
//!     model_family: ModelFamily::Qwen3,
//!     model: "Qwen/Qwen3-Reranker-0.6B".to_string(),
//!     instruction: None,
//!     requests_per_minute: 1000,
//!     max_concurrent_requests: 50,
//!     tokens_per_minute: 1_000_000,
//! })?;
//!
//! let query = seasoning::RerankQuery {
//!     text: "memory-safe systems programming".to_string(),
//!     token_count: 4,
//! };
//! let documents = vec![seasoning::RerankDocument {
//!     text: "Rust uses ownership and borrowing".to_string(),
//!     token_count: 6,
//! }];
//!
//! let scores = reranker.rerank(&query, &documents).await?;
//! assert_eq!(scores.len(), documents.len());
//! # Ok(())
//! # }
//! ```

mod api;
pub mod batching;
mod config;
pub mod embedding;
mod error;
#[cfg(feature = "local")]
mod local;
mod reqwestx;
pub mod reranker;
pub mod service;

pub use api::{
    AddDecision, BatchItem, BatchingStrategy, Dialect, EmbedOutput, EmbeddingInput,
    EmbeddingProvider, EmbeddingRole, ModelFamily, PreparedEmbeddingInput, ProviderDialect,
    RerankDocument, RerankQuery, RerankingProvider,
};
pub use config::{AppConfig, Embedding, Reranker};
pub use error::{Error, Result};
