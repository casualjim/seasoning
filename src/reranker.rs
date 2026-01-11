//! Document reranking based on query relevance.
//!
//! This module provides a client for reranking documents using AI models,
//! which score documents based on their relevance to a given query.
//!
//! # Overview
//!
//! The main entry point is the [`Client`] which implements the [`RerankingProvider`] trait.
//! Reranking is useful for improving search results by using a specialized model to
//! score documents based on semantic relevance to a query.
//!
//! # Supported Providers
//!
//! Currently supports:
//! - DeepInfra API dialect
//!
//! # Example
//!
//! ```rust,no_run
//! use std::time::Duration;
//! use secrecy::SecretString;
//! use seasoning::embedding::ProviderDialect;
//! use seasoning::reranker::{Client, RerankerConfig, RerankingProvider};
//!
//! # async fn example() -> eyre::Result<()> {
//! // Configure the reranking client
//! let reranker = Client::new(RerankerConfig {
//!     api_key: Some(SecretString::from("your-api-key")),
//!     base_url: "https://api.deepinfra.com/v1".to_string(),
//!     timeout: Duration::from_secs(10),
//!     dialect: ProviderDialect::DeepInfra,
//!     model: "Qwen/Qwen3-Reranker-0.6B".to_string(),
//!     instruction: None,
//! })?;
//!
//! // Rerank documents based on query
//! let documents = vec![
//!     "Rust is a systems programming language".to_string(),
//!     "Python is great for data science".to_string(),
//!     "Rust has memory safety without garbage collection".to_string(),
//! ];
//!
//! let scores = reranker.rerank("What is Rust?", &documents).await?;
//! println!("Relevance scores: {:?}", scores); // Higher scores = more relevant
//! # Ok(())
//! # }
//! ```

use std::time::Duration;

use async_trait::async_trait;
use eyre::Result;
use http::{
    HeaderValue,
    header::{AUTHORIZATION, CONTENT_TYPE},
};
use reqwest::Client as HttpClient;
use secrecy::{ExposeSecret, SecretString};
use serde::Deserialize;
use serde_json::json;
use tracing::{debug, error};

use crate::embedding::ProviderDialect;

/// Configuration for the reranking client.
///
/// This struct contains all parameters needed to configure the reranking client,
/// including API credentials and model parameters.
///
/// # Example
///
/// ```rust
/// use std::time::Duration;
/// use secrecy::SecretString;
/// use seasoning::embedding::ProviderDialect;
/// use seasoning::reranker::RerankerConfig;
///
/// let config = RerankerConfig {
///     api_key: Some(SecretString::from("your-api-key")),
///     base_url: "https://api.deepinfra.com/v1".to_string(),
///     timeout: Duration::from_secs(10),
///     dialect: ProviderDialect::DeepInfra,
///     model: "Qwen/Qwen3-Reranker-0.6B".to_string(),
///     instruction: Some("Rank by relevance to the query".to_string()),
/// };
/// ```
#[derive(Debug, Clone)]
pub struct RerankerConfig {
    /// Optional API key for authentication
    pub api_key: Option<SecretString>,
    /// Base URL for the reranking API endpoint (e.g., `https://api.deepinfra.com/v1`)
    pub base_url: String,
    /// Request timeout duration
    pub timeout: Duration,
    /// Provider dialect for API compatibility
    pub dialect: ProviderDialect,
    /// Model identifier (e.g., "Qwen/Qwen3-Reranker-0.6B")
    pub model: String,
    /// Optional instruction to guide the reranking model's behavior
    pub instruction: Option<String>,
}

/// Trait for reranking providers.
///
/// This trait abstracts over different reranking implementations,
/// allowing for easy testing and provider swapping.
///
/// # Example Implementation
///
/// ```rust,no_run
/// use async_trait::async_trait;
/// use eyre::Result;
/// use seasoning::reranker::RerankingProvider;
///
/// struct MockReranker;
///
/// #[async_trait]
/// impl RerankingProvider for MockReranker {
///     async fn rerank(&self, query: &str, documents: &[String]) -> Result<Vec<f64>> {
///         // Return mock scores (higher = more relevant)
///         Ok(vec![0.9, 0.5, 0.7])
///     }
/// }
/// ```
#[async_trait]
pub trait RerankingProvider: Send + Sync {
    /// Rerank documents based on their relevance to a query.
    ///
    /// # Arguments
    ///
    /// * `query` - The search query to rank documents against
    /// * `documents` - Slice of document texts to rank
    ///
    /// # Returns
    ///
    /// Returns a vector of relevance scores, one per document, in the same order
    /// as the input documents. Scores are typically in the range [0.0, 1.0],
    /// where higher scores indicate greater relevance to the query.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The query is empty
    /// - The API request fails
    /// - The response cannot be parsed
    /// - Network errors occur
    async fn rerank(&self, query: &str, documents: &[String]) -> Result<Vec<f64>>;
}

/// Reranking client for scoring document relevance.
///
/// The client sends reranking requests to AI models that score documents
/// based on their semantic relevance to a query. This is useful for improving
/// search result quality beyond simple keyword matching or initial retrieval.
///
/// # Supported Dialects
///
/// Currently supports:
/// - [`ProviderDialect::DeepInfra`] - Uses DeepInfra's inference endpoint
///
/// # Example
///
/// ```rust,no_run
/// use std::time::Duration;
/// use secrecy::SecretString;
/// use seasoning::embedding::ProviderDialect;
/// use seasoning::reranker::{Client, RerankerConfig};
///
/// # fn example() -> eyre::Result<()> {
/// let client = Client::new(RerankerConfig {
///     api_key: Some(SecretString::from("your-api-key")),
///     base_url: "https://api.deepinfra.com/v1".to_string(),
///     timeout: Duration::from_secs(10),
///     dialect: ProviderDialect::DeepInfra,
///     model: "Qwen/Qwen3-Reranker-0.6B".to_string(),
///     instruction: None,
/// })?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct Client {
    client: HttpClient,
    api_key: Option<SecretString>,
    base_url: String,
    model: String,
    instruction: Option<String>,
    dialect: ProviderDialect,
}

/// Internal representation of the reranking API response.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RerankApiResponse {
    /// Relevance scores for each document
    scores: Vec<f64>,
    /// Optional token usage information
    #[serde(default)]
    input_tokens: Option<i64>,
}

impl Client {
    /// Create a new reranking client from configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration parameters for the client
    ///
    /// # Returns
    ///
    /// Returns a configured client ready to make reranking requests.
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
    /// use seasoning::embedding::ProviderDialect;
    /// use seasoning::reranker::{Client, RerankerConfig};
    ///
    /// # fn example() -> eyre::Result<()> {
    /// let client = Client::new(RerankerConfig {
    ///     api_key: Some(SecretString::from("your-api-key")),
    ///     base_url: "https://api.deepinfra.com/v1".to_string(),
    ///     timeout: Duration::from_secs(10),
    ///     dialect: ProviderDialect::DeepInfra,
    ///     model: "Qwen/Qwen3-Reranker-0.6B".to_string(),
    ///     instruction: None,
    /// })?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(config: RerankerConfig) -> Result<Self> {
        let mut headers = http::HeaderMap::new();
        if let Some(api_key) = &config.api_key {
            let value = HeaderValue::from_str(&format!("Bearer {}", api_key.expose_secret()))
                .map_err(|err| eyre::eyre!("invalid reranker api key: {err}"))?;
            headers.insert(AUTHORIZATION, value);
        }
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let client = HttpClient::builder()
            .default_headers(headers)
            .user_agent(format!("Context/Reranker; dialect={}", config.dialect))
            .timeout(config.timeout)
            .build()?;

        Ok(Self {
            client,
            api_key: config.api_key,
            base_url: config.base_url,
            model: config.model,
            instruction: config.instruction,
            dialect: config.dialect,
        })
    }

    /// Rerank documents using the DeepInfra API.
    ///
    /// # Arguments
    ///
    /// * `query` - The search query to rank documents against
    /// * `documents` - Slice of document texts to rank
    ///
    /// # Returns
    ///
    /// Returns relevance scores clamped to [0.0, 1.0], one per document.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The documents slice is empty
    /// - The query is empty or contains only whitespace
    /// - The API request fails
    /// - The response cannot be parsed
    async fn rerank_deepinfra(&self, query: &str, documents: &[String]) -> Result<Vec<f64>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        if query.trim().is_empty() {
            return Err(eyre::eyre!("rerank query cannot be empty"));
        }

        let queries = vec![query; documents.len()];
        let mut payload = json!({
            "queries": queries,
            "documents": documents,
        });

        if let Some(instruction) = &self.instruction {
            payload["instruction"] = json!(instruction);
        }

        let endpoint = format!("{}/inference/{}", self.base_url, self.model);
        let mut req = self.client.post(&endpoint).json(&payload);
        if let Some(api_key) = &self.api_key {
            req = req.bearer_auth(api_key.expose_secret());
        }

        let response = req
            .send()
            .await
            .map_err(|e| {
                error!("Failed to send reranking request: {e}");
                e
            })?
            .error_for_status()
            .map_err(|e| {
                error!("Reranking request returned error status: {e}");
                e
            })?
            .json::<RerankApiResponse>()
            .await
            .map_err(|e| {
                error!("Failed to parse reranking response: {e}");
                e
            })?;

        if let Some(input_tokens) = response.input_tokens {
            debug!("Reranking used {} input tokens", input_tokens);
        }

        Ok(response
            .scores
            .into_iter()
            .map(|s| s.clamp(0.0, 1.0))
            .collect())
    }
}

#[async_trait]
impl RerankingProvider for Client {
    async fn rerank(&self, query: &str, documents: &[String]) -> Result<Vec<f64>> {
        match self.dialect {
            ProviderDialect::DeepInfra => self.rerank_deepinfra(query, documents).await,
            ProviderDialect::OpenAI => Err(eyre::eyre!(
                "openai reranking is not configured; use deepinfra or add openai support"
            )),
        }
    }
}
