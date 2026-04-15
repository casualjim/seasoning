//! # Seasoning
//!
//! Retrieval-focused embedding and reranking infrastructure with explicit model
//! semantics, rate limiting, retries, and optional local llama.cpp execution.
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
//!     ModelFamily,
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
//! let inputs = vec![EmbeddingInput {
//!     role: EmbeddingRole::Query,
//!     text: "memory-safe systems programming".to_string(),
//!     title: None,
//!     token_count: 4,
//! }];
//!
//! let _ = embedder.embed(&inputs).await?;
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
//! let _ = reranker.rerank(&query, &documents).await?;
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
    EmbeddingProvider, EmbeddingRole, ModelFamily, ProviderDialect, RerankDocument,
    RerankQuery, RerankingProvider,
};
pub use config::*;
pub use error::{Error, Result};
