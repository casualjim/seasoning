//! Configuration types for embedding and reranking services.
//!
//! This module provides serializable configuration structures that can be
//! loaded from external config files or environment variables.

use std::time::Duration;

use secrecy::SecretString;
use serde::{Deserialize, Serialize};

use crate::Error;
use crate::Result;
use crate::embedding::{EmbedderConfig, ProviderDialect};
use crate::reranker::RerankerConfig;

/// Top-level application configuration containing both embedding and reranking settings.
///
/// This struct is designed to be deserialized from configuration files (JSON, YAML, TOML, etc.)
/// and provides all necessary settings for both embedding and reranking services.
///
/// # Example
///
/// ```rust
/// use seasoning::AppConfig;
/// use serde_json::json;
///
/// let config_json = json!({
///     "embedding": {
///         "url": "https://api.deepinfra.com/v1/openai",
///         "model": "Qwen/Qwen3-Embedding-0.6B",
///         "dialect": "deepinfra",
///         "timeout_seconds": 10,
///         "embedding_dim": 1024,
///         "requests_per_minute": 1000,
///         "max_concurrent_requests": 50,
///         "tokens_per_minute": 1000000
///     },
///     "reranker": {
///         "url": "https://api.deepinfra.com/v1",
///         "model": "Qwen/Qwen3-Reranker-0.6B",
///         "dialect": "deepinfra",
///         "timeout_seconds": 10,
///         "requests_per_minute": 1000,
///         "max_concurrent_requests": 50,
///         "tokens_per_minute": 1000000
///     }
/// });
///
/// let config: AppConfig = serde_json::from_value(config_json).unwrap();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// Embedding service configuration
    pub embedding: Embedding,
    /// Reranking service configuration
    pub reranker: Reranker,
}

/// Configuration for the embedding service.
///
/// This struct contains all settings needed to create an embedding client,
/// including API credentials, rate limits, and model parameters.
///
/// # Fields
///
/// - `url`: Base URL for the embedding API endpoint
/// - `api_key`: Optional API key for authentication (skipped during serialization for security)
/// - `model`: Model identifier (e.g., "Qwen/Qwen3-Embedding-0.6B")
/// - `dialect`: Provider dialect string ("openai" or "deepinfra")
/// - `timeout_seconds`: Request timeout in seconds
/// - `embedding_dim`: Dimension of the embedding vectors
/// - `requests_per_minute`: Maximum requests per minute for rate limiting
/// - `max_concurrent_requests`: Maximum number of concurrent requests
/// - `tokens_per_minute`: Maximum tokens per minute for rate limiting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    /// Base URL for the embedding API endpoint
    pub url: String,
    /// API key for authentication (not serialized for security)
    #[serde(skip_serializing)]
    pub api_key: Option<SecretString>,
    /// Model identifier to use for embeddings
    pub model: String,
    /// Tokenizer identifier (e.g. "characters", "tiktoken:o200k_base", "hf:Qwen/Qwen3-Embedding-0.6B")
    #[serde(default)]
    pub tokenizer: String,
    /// Provider dialect ("openai" or "deepinfra")
    pub dialect: String,
    /// Request timeout in seconds
    pub timeout_seconds: u64,
    /// Dimension of embedding vectors returned by the model
    pub embedding_dim: usize,
    /// Maximum tokens per embedding request/batch (used for batching)
    #[serde(default = "default_context_length")]
    pub context_length: usize,
    /// Maximum inputs per embedding batch (used for batching)
    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: usize,
    /// Number of concurrent embedding workers (used for batching)
    #[serde(default = "default_embedding_workers")]
    pub workers: usize,
    /// Maximum number of requests per minute (rate limit)
    pub requests_per_minute: usize,
    /// Maximum number of concurrent requests allowed
    pub max_concurrent_requests: usize,
    /// Maximum number of tokens per minute (rate limit)
    pub tokens_per_minute: u32,
}

/// Configuration for the reranking service.
///
/// This struct contains all settings needed to create a reranking client,
/// including API credentials and model parameters.
///
/// # Fields
///
/// - `url`: Base URL for the reranking API endpoint
/// - `api_key`: Optional API key for authentication (skipped during serialization for security)
/// - `model`: Model identifier (e.g., "Qwen/Qwen3-Reranker-0.6B")
/// - `dialect`: Provider dialect string ("openai" or "deepinfra")
/// - `timeout_seconds`: Request timeout in seconds
/// - `instruction`: Optional instruction text to guide the reranking model
/// - `requests_per_minute`: Maximum requests per minute for rate limiting
/// - `max_concurrent_requests`: Maximum number of concurrent requests
/// - `tokens_per_minute`: Maximum tokens per minute for rate limiting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reranker {
    /// Base URL for the reranking API endpoint
    pub url: String,
    /// API key for authentication (not serialized for security)
    #[serde(skip_serializing)]
    pub api_key: Option<SecretString>,
    /// Model identifier to use for reranking
    pub model: String,
    /// Provider dialect ("openai" or "deepinfra")
    pub dialect: String,
    /// Request timeout in seconds
    pub timeout_seconds: u64,
    /// Optional instruction text to guide the reranking model (omitted from serialization if None)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instruction: Option<String>,
    /// Maximum number of requests per minute (rate limit)
    #[serde(default = "default_reranker_requests_per_minute")]
    pub requests_per_minute: usize,
    /// Maximum number of concurrent requests allowed
    #[serde(default = "default_reranker_max_concurrent_requests")]
    pub max_concurrent_requests: usize,
    /// Maximum number of tokens per minute (rate limit)
    #[serde(default = "default_reranker_tokens_per_minute")]
    pub tokens_per_minute: u32,
}

impl Embedding {
    pub fn to_embedder_config(&self) -> Result<EmbedderConfig> {
        Ok(EmbedderConfig {
            api_key: self.api_key.clone(),
            base_url: self.url.clone(),
            timeout: Duration::from_secs(self.timeout_seconds),
            dialect: parse_provider_dialect(&self.dialect)?,
            model: self.model.clone(),
            embedding_dim: self.embedding_dim,
            requests_per_minute: self.requests_per_minute,
            max_concurrent_requests: self.max_concurrent_requests,
            tokens_per_minute: self.tokens_per_minute,
        })
    }
}

impl Reranker {
    pub fn to_reranker_config(&self) -> Result<RerankerConfig> {
        Ok(RerankerConfig {
            api_key: self.api_key.clone(),
            base_url: self.url.clone(),
            timeout: Duration::from_secs(self.timeout_seconds),
            dialect: parse_provider_dialect(&self.dialect)?,
            model: self.model.clone(),
            instruction: self.instruction.clone(),
            requests_per_minute: self.requests_per_minute,
            max_concurrent_requests: self.max_concurrent_requests,
            tokens_per_minute: self.tokens_per_minute,
        })
    }
}

fn parse_provider_dialect(value: &str) -> Result<ProviderDialect> {
    let normalized = value.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "openai" => Ok(ProviderDialect::OpenAI),
        "deepinfra" => Ok(ProviderDialect::DeepInfra),
        _ => Err(Error::InvalidProviderDialect {
            value: value.to_string(),
        }),
    }
}

fn default_context_length() -> usize {
    32_768
}

fn default_max_batch_size() -> usize {
    15
}

fn default_embedding_workers() -> usize {
    5
}

fn default_reranker_requests_per_minute() -> usize {
    1000
}

fn default_reranker_max_concurrent_requests() -> usize {
    50
}

fn default_reranker_tokens_per_minute() -> u32 {
    1_000_000
}
