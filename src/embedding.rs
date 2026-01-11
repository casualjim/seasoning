//! Text embedding generation with rate limiting and retry logic.
//!
//! This module provides a client for generating text embeddings from AI models,
//! with built-in rate limiting, concurrency control, and automatic retries.
//!
//! # Overview
//!
//! The main entry point is the [`Client`] which implements the [`EmbeddingProvider`] trait.
//! The client handles:
//!
//! - Batched embedding requests
//! - Token-aware rate limiting
//! - Request rate limiting
//! - Concurrent request management
//! - Automatic retries with exponential backoff
//! - Multiple provider dialects (OpenAI, DeepInfra)
//!
//! # Example
//!
//! ```rust,no_run
//! use std::time::Duration;
//! use secrecy::SecretString;
//! use seasoning::embedding::{
//!     Client, EmbedderConfig, EmbeddingInput, ProviderDialect,
//! };
//!
//! # async fn example() -> seasoning::Result<()> {
//! // Configure the embedding client
//! let embedder = Client::new(EmbedderConfig {
//!     api_key: Some(SecretString::from("your-api-key")),
//!     base_url: "https://api.deepinfra.com/v1/openai".to_string(),
//!     timeout: Duration::from_secs(30),
//!     dialect: ProviderDialect::DeepInfra,
//!     model: "Qwen/Qwen3-Embedding-0.6B".to_string(),
//!     embedding_dim: 1024,
//!     requests_per_minute: 1000,
//!     max_concurrent_requests: 50,
//!     tokens_per_minute: 1_000_000,
//! })?;
//!
//! // Prepare inputs with token counts
//! let inputs = vec![
//!     EmbeddingInput {
//!         text: "hello world".to_string(),
//!         token_count: 2,
//!     },
//!     EmbeddingInput {
//!         text: "another string".to_string(),
//!         token_count: 2,
//!     },
//! ];
//!
//! // Generate embeddings
//! let result = embedder.embed(&inputs).await?;
//! println!("Generated {} embeddings", result.embeddings.len());
//! # Ok(())
//! # }
//! ```

use std::time::Duration;

use async_trait::async_trait;
use secrecy::SecretString;
use serde::Deserialize;
use serde_json::json;
use tracing::debug;

use crate::EmbeddingProvider;
use crate::Result;
use crate::reqwestx::{ApiClient, ApiClientConfig};
pub use crate::{EmbedOutput, EmbeddingInput};

/// Provider dialect for embedding API compatibility.
///
/// Different providers may have slightly different API shapes or requirements.
/// This enum allows the client to adapt its requests accordingly.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub enum ProviderDialect {
    /// OpenAI-compatible API format
    #[default]
    OpenAI,
    /// DeepInfra API format
    DeepInfra,
}

impl std::fmt::Display for ProviderDialect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProviderDialect::OpenAI => write!(f, "openai"),
            ProviderDialect::DeepInfra => write!(f, "deepinfra"),
        }
    }
}

/// Configuration for the embedding client.
///
/// This struct contains all parameters needed to configure the embedding client,
/// including API credentials, rate limits, and model parameters.
///
/// # Rate Limiting
///
/// The client implements dual rate limiting:
/// - `requests_per_minute`: Limits the number of API requests
/// - `tokens_per_minute`: Limits the total number of tokens processed
///
/// Both limits are enforced using a token bucket algorithm with automatic refill.
///
/// # Example
///
/// ```rust
/// use std::time::Duration;
/// use secrecy::SecretString;
/// use seasoning::embedding::{EmbedderConfig, ProviderDialect};
///
/// let config = EmbedderConfig {
///     api_key: Some(SecretString::from("your-api-key")),
///     base_url: "https://api.deepinfra.com/v1/openai".to_string(),
///     timeout: Duration::from_secs(30),
///     dialect: ProviderDialect::DeepInfra,
///     model: "Qwen/Qwen3-Embedding-0.6B".to_string(),
///     embedding_dim: 1024,
///     requests_per_minute: 1000,
///     max_concurrent_requests: 50,
///     tokens_per_minute: 1_000_000,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct EmbedderConfig {
    /// Optional API key for authentication
    pub api_key: Option<SecretString>,
    /// Base URL for the embedding API endpoint (e.g., `https://api.deepinfra.com/v1/openai`)
    pub base_url: String,
    /// Request timeout duration
    pub timeout: Duration,
    /// Provider dialect for API compatibility
    pub dialect: ProviderDialect,
    /// Model identifier (e.g., "Qwen/Qwen3-Embedding-0.6B")
    pub model: String,
    /// Dimension of the embedding vectors returned by the model
    pub embedding_dim: usize,
    /// Maximum number of requests per minute (rate limit)
    pub requests_per_minute: usize,
    /// Maximum number of concurrent requests allowed
    pub max_concurrent_requests: usize,
    /// Maximum number of tokens per minute (rate limit)
    pub tokens_per_minute: u32,
}

/// Embedding client with rate limiting and retry logic.
///
/// The client handles batched embedding requests with automatic rate limiting,
/// concurrency control, and retries. It implements the [`EmbeddingProvider`] trait.
///
/// # Rate Limiting
///
/// The client uses a dual token bucket algorithm to enforce both request-per-minute
/// and token-per-minute limits. If a limit is reached, requests will automatically
/// wait until capacity is available.
///
/// # Retries
///
/// Failed requests are automatically retried with exponential backoff for:
/// - HTTP status codes: 429, 500, 502, 503, 504
/// - Timeout errors
/// - Connection errors
///
/// # Example
///
/// ```rust,no_run
/// use std::time::Duration;
/// use secrecy::SecretString;
/// use seasoning::embedding::{Client, EmbedderConfig, ProviderDialect};
///
/// # fn example() -> seasoning::Result<()> {
/// let client = Client::new(EmbedderConfig {
///     api_key: Some(SecretString::from("your-api-key")),
///     base_url: "https://api.deepinfra.com/v1/openai".to_string(),
///     timeout: Duration::from_secs(30),
///     dialect: ProviderDialect::DeepInfra,
///     model: "Qwen/Qwen3-Embedding-0.6B".to_string(),
///     embedding_dim: 1024,
///     requests_per_minute: 1000,
///     max_concurrent_requests: 50,
///     tokens_per_minute: 1_000_000,
/// })?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct Client {
    client: ApiClient,
    model: String,
    dimension: usize,
    dialect: ProviderDialect,
}

impl Client {
    /// Create a new embedding client from configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration parameters for the client
    ///
    /// # Returns
    ///
    /// Returns a configured client ready to make embedding requests.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The API key contains invalid characters
    /// - The HTTP client cannot be constructed
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use std::time::Duration;
    /// use secrecy::SecretString;
    /// use seasoning::embedding::{Client, EmbedderConfig, ProviderDialect};
    ///
    /// # fn example() -> seasoning::Result<()> {
    /// let client = Client::new(EmbedderConfig {
    ///     api_key: Some(SecretString::from("your-api-key")),
    ///     base_url: "https://api.deepinfra.com/v1/openai".to_string(),
    ///     timeout: Duration::from_secs(30),
    ///     dialect: ProviderDialect::DeepInfra,
    ///     model: "Qwen/Qwen3-Embedding-0.6B".to_string(),
    ///     embedding_dim: 1024,
    ///     requests_per_minute: 1000,
    ///     max_concurrent_requests: 50,
    ///     tokens_per_minute: 1_000_000,
    /// })?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(config: EmbedderConfig) -> Result<Self> {
        let api_config = ApiClientConfig {
            base_url: config.base_url.clone(),
            api_key: config.api_key.clone(),
            max_concurrent_requests: config.max_concurrent_requests,
            max_requests_per_minute: config.requests_per_minute,
            max_tokens_per_minute: config.tokens_per_minute as usize,
            max_retries: 3,
            timeout: config.timeout,
        };

        let client = ApiClient::new(api_config)?;

        Ok(Self {
            client,
            model: config.model,
            dimension: config.embedding_dim,
            dialect: config.dialect,
        })
    }

    /// Estimate the total token count for a batch of inputs.
    ///
    /// Sums up the pre-calculated token counts from all inputs.
    /// Uses saturating addition to prevent overflow.
    fn estimate_token_count(&self, input: &[EmbeddingInput]) -> u32 {
        let mut tokens: u32 = 0;
        for inp in input {
            tokens = tokens.saturating_add(inp.token_count as u32);
        }
        tokens
    }

    /// Extract text strings from embedding inputs for the API request.
    fn prepare_inputs(&self, input: &[EmbeddingInput]) -> Vec<String> {
        let mut batch_texts = Vec::with_capacity(input.len());
        for inp in input {
            batch_texts.push(inp.text.clone());
        }
        batch_texts
    }
}

/// Internal representation of a single embedding from the API response.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct EmbeddingObject {
    /// Index of the input text this embedding corresponds to
    index: usize,
    /// The embedding vector
    embedding: Vec<f32>,
}

/// Internal representation of the embedding API response.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct EmbedApiResponse {
    /// List of embeddings with their indices
    data: Vec<EmbeddingObject>,
}

#[async_trait]
impl EmbeddingProvider for Client {
    async fn embed(&self, input: &[EmbeddingInput]) -> Result<EmbedOutput> {
        debug!("Embedding input batch_size: {}", input.len());
        let batch_texts = self.prepare_inputs(input);
        let estimated_tokens = self.estimate_token_count(input);

        let payload = match self.dialect {
            ProviderDialect::OpenAI | ProviderDialect::DeepInfra => {
                json!({
                  "input": &batch_texts,
                  "model": self.model,
                  "encoding_format": "float",
                  "dimensions": self.dimension
                })
            }
        };
        let response: EmbedApiResponse = self
            .client
            .post_json("/embeddings", &payload, estimated_tokens)
            .await?;

        let data = response.data;
        let mut embeddings = vec![Vec::new(); data.len()];
        for item in data {
            if item.index < embeddings.len() {
                embeddings[item.index] = item.embedding;
            }
        }

        Ok(EmbedOutput { embeddings })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic;
    use std::time::Duration;

    use secrecy::SecretString;

    #[test]
    fn embedder_new_should_not_panic_on_invalid_api_key() {
        let result = panic::catch_unwind(|| {
            let _ = Client::new(EmbedderConfig {
                api_key: Some(SecretString::from("bad\nkey")),
                base_url: "http://127.0.0.1:1".to_string(),
                timeout: Duration::from_secs(1),
                dialect: ProviderDialect::OpenAI,
                model: "test-model".to_string(),
                embedding_dim: 2,
                requests_per_minute: 1000,
                max_concurrent_requests: 300,
                tokens_per_minute: 1,
            });
        });

        assert!(
            result.is_ok(),
            "Client::new should return Err, not panic, for invalid API keys"
        );
    }
}
