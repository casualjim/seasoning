//! Configuration types for embedding and reranking services.
//!
//! This module provides serializable configuration structures that can be
//! loaded from external config files or environment variables.

use secrecy::SecretString;
use serde::{Deserialize, Serialize};

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
///         "timeout_seconds": 10
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
    /// Provider dialect ("openai" or "deepinfra")
    pub dialect: String,
    /// Request timeout in seconds
    pub timeout_seconds: u64,
    /// Dimension of embedding vectors returned by the model
    pub embedding_dim: usize,
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
}
