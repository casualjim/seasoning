//! # Seasoning
//!
//! Retrieval-focused embedding and reranking infrastructure with explicit model
//! semantics, rate limiting, retries, and optional local llama.cpp execution.

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
