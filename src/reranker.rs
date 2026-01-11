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
//! use seasoning::RerankingProvider;
//! use seasoning::reranker::{Client, RerankerConfig};
//!
//! # async fn example() -> seasoning::Result<()> {
//! // Configure the reranking client
//! let reranker = Client::new(RerankerConfig {
//!     api_key: Some(SecretString::from("your-api-key")),
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
//! // Rerank documents based on query
//! let documents = vec![
//!     seasoning::RerankDocument {
//!         text: "Rust is a systems programming language".to_string(),
//!         token_count: 2,
//!     },
//!     seasoning::RerankDocument {
//!         text: "Python is great for data science".to_string(),
//!         token_count: 2,
//!     },
//!     seasoning::RerankDocument {
//!         text: "Rust has memory safety without garbage collection".to_string(),
//!         token_count: 2,
//!     },
//! ];
//!
//! let query = seasoning::RerankQuery {
//!     text: "What is Rust?".to_string(),
//!     token_count: 2,
//! };
//! let scores = reranker.rerank(&query, &documents).await?;
//! println!("Relevance scores: {:?}", scores); // Higher scores = more relevant
//! # Ok(())
//! # }
//! ```

use std::time::Duration;

use async_trait::async_trait;
use secrecy::SecretString;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::RerankingProvider;
use crate::embedding::ProviderDialect;
use crate::reqwestx::{ApiClient, ApiClientConfig};
use crate::{Error, Result};
use crate::{RerankDocument, RerankQuery};

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
///     requests_per_minute: 1000,
///     max_concurrent_requests: 50,
///     tokens_per_minute: 1_000_000,
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
    /// Maximum number of requests per minute (rate limit)
    pub requests_per_minute: usize,
    /// Maximum number of concurrent requests allowed
    pub max_concurrent_requests: usize,
    /// Maximum number of tokens per minute (rate limit)
    pub tokens_per_minute: u32,
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
/// # fn example() -> seasoning::Result<()> {
/// let client = Client::new(RerankerConfig {
///     api_key: Some(SecretString::from("your-api-key")),
///     base_url: "https://api.deepinfra.com/v1".to_string(),
///     timeout: Duration::from_secs(10),
///     dialect: ProviderDialect::DeepInfra,
///     model: "Qwen/Qwen3-Reranker-0.6B".to_string(),
///     instruction: None,
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

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct OpenAiRerankResponse {
    data: Vec<OpenAiRerankData>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct OpenAiRerankData {
    index: usize,
    relevance_score: f64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct RerankDeepInfraRequest<'a> {
    queries: Vec<&'a str>,
    documents: Vec<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    instruction: Option<&'a str>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct OpenAiRerankRequest<'a> {
    model: &'a str,
    query: &'a str,
    documents: Vec<&'a str>,
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
    /// # fn example() -> seasoning::Result<()> {
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
    async fn rerank_deepinfra(
        &self,
        query: &RerankQuery,
        documents: &[RerankDocument],
    ) -> Result<Vec<f64>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        if query.text.trim().is_empty() {
            return Err(Error::EmptyRerankQuery);
        }

        let payload = RerankDeepInfraRequest {
            queries: vec![query.text.as_str(); documents.len()],
            documents: documents.iter().map(|d| d.text.as_str()).collect(),
            instruction: self.instruction.as_deref(),
        };

        let token_count = estimate_token_count(query, documents);
        let response: RerankApiResponse = self
            .client
            .post_json(&format!("/inference/{}", self.model), &payload, token_count)
            .await?;

        if let Some(input_tokens) = response.input_tokens {
            debug!("Reranking used {} input tokens", input_tokens);
        }

        Ok(response
            .scores
            .into_iter()
            .map(|s| s.clamp(0.0, 1.0))
            .collect())
    }

    async fn rerank_openai(
        &self,
        query: &RerankQuery,
        documents: &[RerankDocument],
    ) -> Result<Vec<f64>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        if query.text.trim().is_empty() {
            return Err(Error::EmptyRerankQuery);
        }

        let payload = OpenAiRerankRequest {
            model: self.model.as_str(),
            query: query.text.as_str(),
            documents: documents.iter().map(|d| d.text.as_str()).collect(),
        };

        let token_count = estimate_token_count_openai(query, documents);
        let response: OpenAiRerankResponse = self
            .client
            .post_json("/rerank", &payload, token_count)
            .await?;

        let mut scores = vec![0.0f64; documents.len()];
        for item in response.data {
            if let Some(score) = scores.get_mut(item.index) {
                *score = item.relevance_score.clamp(0.0, 1.0);
            }
        }

        Ok(scores)
    }
}

#[async_trait]
impl RerankingProvider for Client {
    async fn rerank(&self, query: &RerankQuery, documents: &[RerankDocument]) -> Result<Vec<f64>> {
        match self.dialect {
            ProviderDialect::DeepInfra => self.rerank_deepinfra(query, documents).await,
            ProviderDialect::OpenAI => self.rerank_openai(query, documents).await,
        }
    }
}

fn estimate_token_count(query: &RerankQuery, documents: &[RerankDocument]) -> u32 {
    // DeepInfra reranking payload repeats the query once per document.
    let query_total = (query.token_count as u32).saturating_mul(documents.len() as u32);
    let docs_total = documents
        .iter()
        .fold(0u32, |acc, d| acc.saturating_add(d.token_count as u32));
    query_total.saturating_add(docs_total)
}

fn estimate_token_count_openai(query: &RerankQuery, documents: &[RerankDocument]) -> u32 {
    let query_total = query.token_count as u32;
    let docs_total = documents
        .iter()
        .fold(0u32, |acc, d| acc.saturating_add(d.token_count as u32));
    query_total.saturating_add(docs_total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use secrecy::SecretString;
    use wiremock::matchers::{body_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn rerank_deepinfra_success() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/inference/test-model"))
            .and(body_json(serde_json::json!({
                "queries": ["q", "q", "q"],
                "documents": ["a", "b", "c"]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "scores": [1.2, -0.1, 0.5],
                "inputTokens": 123
            })))
            .mount(&mock_server)
            .await;

        let client = Client::new(RerankerConfig {
            api_key: None,
            base_url: mock_server.uri(),
            timeout: Duration::from_secs(10),
            dialect: ProviderDialect::DeepInfra,
            model: "test-model".to_string(),
            instruction: None,
            requests_per_minute: 1000,
            max_concurrent_requests: 10,
            tokens_per_minute: 1_000_000,
        })
        .unwrap();

        let query = RerankQuery {
            text: "q".to_string(),
            token_count: 1,
        };
        let documents = vec![
            RerankDocument {
                text: "a".to_string(),
                token_count: 2,
            },
            RerankDocument {
                text: "b".to_string(),
                token_count: 2,
            },
            RerankDocument {
                text: "c".to_string(),
                token_count: 2,
            },
        ];

        let scores = client.rerank(&query, &documents).await.unwrap();
        assert_eq!(scores, vec![1.0, 0.0, 0.5]);
    }

    #[tokio::test]
    async fn rerank_deepinfra_requires_query() {
        let mock_server = MockServer::start().await;

        let client = Client::new(RerankerConfig {
            api_key: None,
            base_url: mock_server.uri(),
            timeout: Duration::from_secs(10),
            dialect: ProviderDialect::DeepInfra,
            model: "test-model".to_string(),
            instruction: None,
            requests_per_minute: 1000,
            max_concurrent_requests: 10,
            tokens_per_minute: 1_000_000,
        })
        .unwrap();

        let query = RerankQuery {
            text: "   ".to_string(),
            token_count: 0,
        };
        let documents = vec![RerankDocument {
            text: "a".to_string(),
            token_count: 1,
        }];

        let err = client.rerank(&query, &documents).await.unwrap_err();
        assert!(matches!(err, Error::EmptyRerankQuery));
    }

    #[tokio::test]
    async fn rerank_deepinfra_sets_authorization_header() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/inference/test-model"))
            .and(header("Authorization", "Bearer test_key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "scores": [0.1]
            })))
            .mount(&mock_server)
            .await;

        let client = Client::new(RerankerConfig {
            api_key: Some(SecretString::from("test_key")),
            base_url: mock_server.uri(),
            timeout: Duration::from_secs(10),
            dialect: ProviderDialect::DeepInfra,
            model: "test-model".to_string(),
            instruction: None,
            requests_per_minute: 1000,
            max_concurrent_requests: 10,
            tokens_per_minute: 1_000_000,
        })
        .unwrap();

        let query = RerankQuery {
            text: "q".to_string(),
            token_count: 1,
        };
        let documents = vec![RerankDocument {
            text: "a".to_string(),
            token_count: 1,
        }];

        let _ = client.rerank(&query, &documents).await.unwrap();
    }

    #[tokio::test]
    async fn rerank_openai_success() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/rerank"))
            .and(body_json(serde_json::json!({
                "model": "test-model",
                "query": "q",
                "documents": ["a", "b", "c"]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [
                    { "index": 1, "relevanceScore": 1.2 },
                    { "index": 0, "relevanceScore": -0.1 },
                    { "index": 2, "relevanceScore": 0.5 }
                ]
            })))
            .mount(&mock_server)
            .await;

        let client = Client::new(RerankerConfig {
            api_key: None,
            base_url: mock_server.uri(),
            timeout: Duration::from_secs(10),
            dialect: ProviderDialect::OpenAI,
            model: "test-model".to_string(),
            instruction: None,
            requests_per_minute: 1000,
            max_concurrent_requests: 10,
            tokens_per_minute: 1_000_000,
        })
        .unwrap();

        let query = RerankQuery {
            text: "q".to_string(),
            token_count: 1,
        };
        let documents = vec![
            RerankDocument {
                text: "a".to_string(),
                token_count: 2,
            },
            RerankDocument {
                text: "b".to_string(),
                token_count: 2,
            },
            RerankDocument {
                text: "c".to_string(),
                token_count: 2,
            },
        ];

        let scores = client.rerank(&query, &documents).await.unwrap();
        assert_eq!(scores, vec![0.0, 1.0, 0.5]);
    }
}
