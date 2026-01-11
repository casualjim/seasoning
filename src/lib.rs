//! # Seasoning
//!
//! A Rust library providing embedding and reranking infrastructure with built-in
//! rate limiting, retries, and support for multiple AI provider dialects.
//!
//! ## Features
//!
//! - **Embedding Client**: Generate text embeddings with configurable rate limiting
//!   and automatic retries
//! - **Reranking Client**: Rerank documents based on query relevance
//! - **Rate Limiting**: Token bucket algorithm for both request and token limits
//! - **Retries**: Automatic retry logic with exponential backoff for transient failures
//! - **Provider Support**: Flexible dialect system supporting OpenAI and DeepInfra APIs
//!
//! ## Quick Start
//!
//! ### Embeddings
//!
//! ```rust,no_run
//! use std::time::Duration;
//! use secrecy::SecretString;
//! use seasoning::embedding::{
//!     Client as EmbedClient, EmbedderConfig, EmbeddingInput, ProviderDialect,
//! };
//!
//! # async fn example() -> seasoning::Result<()> {
//! let embedder = EmbedClient::new(EmbedderConfig {
//!     api_key: Some(SecretString::from("YOUR_API_KEY")),
//!     base_url: "https://api.deepinfra.com/v1/openai".to_string(),
//!     timeout: Duration::from_secs(10),
//!     dialect: ProviderDialect::DeepInfra,
//!     model: "Qwen/Qwen3-Embedding-0.6B".to_string(),
//!     embedding_dim: 1024,
//!     requests_per_minute: 1000,
//!     max_concurrent_requests: 50,
//!     tokens_per_minute: 1_000_000,
//! })?;
//!
//! let inputs = vec![
//!     EmbeddingInput {
//!         text: "hello world".to_string(),
//!         token_count: 2,
//!     },
//! ];
//!
//! let result = embedder.embed(&inputs).await?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Reranking
//!
//! ```rust,no_run
//! use std::time::Duration;
//! use secrecy::SecretString;
//! use seasoning::embedding::ProviderDialect;
//! use seasoning::reranker::{Client as RerankerClient, RerankerConfig};
//!
//! # async fn example() -> seasoning::Result<()> {
//! let reranker = RerankerClient::new(RerankerConfig {
//!     api_key: Some(SecretString::from("YOUR_API_KEY")),
//!     base_url: "https://api.deepinfra.com/v1".to_string(),
//!     timeout: Duration::from_secs(10),
//!     dialect: ProviderDialect::DeepInfra,
//!     model: "Qwen/Qwen3-Reranker-0.6B".to_string(),
//!     instruction: None,
//!     requests_per_minute: 1000,
//!     max_concurrent_requests: 50,
//!     tokens_per_minute: 1_000_000,
//! })?;
//!
//! let query = seasoning::RerankQuery {
//!     text: "search query".to_string(),
//!     token_count: 2,
//! };
//! let docs = vec![
//!     seasoning::RerankDocument {
//!         text: "first doc".to_string(),
//!         token_count: 2,
//!     },
//!     seasoning::RerankDocument {
//!         text: "second doc".to_string(),
//!         token_count: 2,
//!     },
//! ];
//! let scores = reranker.rerank(&query, &docs).await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Modules
//!
//! - [`embedding`] - Text embedding generation with rate limiting
//! - [`reranker`] - Document reranking based on query relevance
//! - Configuration types re-exported from the private `config` module

mod api;
pub mod batching;
mod config;
pub mod embedding;
mod error;
mod reqwestx;
pub mod reranker;
pub mod service;

pub use api::{
    AddDecision, BatchItem, BatchingStrategy, EmbedOutput, EmbeddingInput, EmbeddingProvider,
    RerankDocument, RerankQuery, RerankingProvider,
};
pub use config::*;
pub use error::{Error, Result};
