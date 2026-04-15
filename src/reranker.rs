//! Document reranking based on query relevance.
//!
//! ```rust,no_run
//! use std::time::Duration;
//!
//! use secrecy::SecretString;
//! use seasoning::RerankingProvider;
//! use seasoning::embedding::{Dialect, ModelFamily};
//! use seasoning::reranker::{Client, RerankerConfig};
//!
//! # async fn example() -> seasoning::Result<()> {
//! let client = Client::new(RerankerConfig {
//!     api_key: Some(SecretString::from("your-api-key")),
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
//! let _ = client.rerank(&query, &documents).await?;
//! # Ok(())
//! # }
//! ```

use std::time::Duration;

use async_trait::async_trait;
use secrecy::SecretString;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::RerankingProvider;
#[cfg(feature = "local")]
use crate::local::LocalRerankerClient;
use crate::reqwestx::{ApiClient, ApiClientConfig};
use crate::{Dialect, Error, ModelFamily, Result};
use crate::{RerankDocument, RerankQuery};

/// Configuration for the reranking client.
#[derive(Debug, Clone)]
pub struct RerankerConfig {
    /// Optional API key for authentication.
    pub api_key: Option<SecretString>,
    /// Base URL for the reranking API endpoint.
    pub base_url: String,
    /// Request timeout duration.
    pub timeout: Duration,
    /// Backend dialect for execution.
    pub dialect: Dialect,
    /// Retrieval-model family used by the reranker.
    pub model_family: ModelFamily,
    /// Model identifier.
    pub model: String,
    /// Optional instruction to guide reranking behavior.
    pub instruction: Option<String>,
    /// Maximum number of requests per minute.
    pub requests_per_minute: usize,
    /// Maximum number of concurrent requests allowed.
    pub max_concurrent_requests: usize,
    /// Maximum number of tokens per minute.
    pub tokens_per_minute: u32,
}

#[derive(Clone)]
pub struct Client {
    #[cfg_attr(not(any(test, feature = "local")), allow(dead_code))]
    model_family: ModelFamily,
    instruction: Option<String>,
    backend: Backend,
}

#[derive(Clone)]
enum Backend {
    Remote(RemoteClient),
    #[cfg(feature = "local")]
    Local(LocalRerankerClient),
}

#[derive(Clone)]
struct RemoteClient {
    client: ApiClient,
    model: String,
    dialect: Dialect,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RerankApiResponse {
    scores: Vec<f64>,
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
    pub fn new(config: RerankerConfig) -> Result<Self> {
        validate_reranker_family(config.model_family)?;

        match config.dialect {
            Dialect::OpenAI | Dialect::DeepInfra => {
                let model_family = config.model_family;
                let instruction = config.instruction.clone();
                let remote = RemoteClient::new(config)?;
                Ok(Self {
                    model_family,
                    instruction,
                    backend: Backend::Remote(remote),
                })
            }
            Dialect::LlamaCpp => {
                #[cfg(feature = "local")]
                {
                    Ok(Self {
                        model_family: config.model_family,
                        instruction: config.instruction,
                        backend: Backend::Local(LocalRerankerClient::new(
                            config.model_family,
                            &config.model,
                        )?),
                    })
                }
                #[cfg(not(feature = "local"))]
                {
                    let _ = config;
                    Err(Error::LocalFeatureRequired {
                        dialect: Dialect::LlamaCpp.to_string(),
                    })
                }
            }
        }
    }

    #[cfg_attr(not(any(test, feature = "local")), allow(dead_code))]
    fn local_instruction(&self) -> String {
        self.instruction
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| self.model_family.default_query_instruction().to_string())
    }

    #[cfg_attr(not(any(test, feature = "local")), allow(dead_code))]
    fn format_local_qwen3_input(&self, query: &RerankQuery, document: &RerankDocument) -> String {
        let instruction = self.local_instruction();
        format!(
            "Instruct: {instruction}\nQuery: {}\nDocument: {}",
            query.text, document.text
        )
    }
}

impl RemoteClient {
    fn new(config: RerankerConfig) -> Result<Self> {
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
            dialect: config.dialect,
        })
    }

    async fn rerank_deepinfra(
        &self,
        instruction: Option<&str>,
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
            instruction,
        };

        let token_count = estimate_token_count(query, documents);
        let response: RerankApiResponse = self
            .client
            .post_json(&format!("/inference/{}", self.model), &payload, token_count)
            .await?;

        if let Some(input_tokens) = response.input_tokens {
            debug!("Reranking used {} input tokens", input_tokens);
        }

        ensure_score_count(response.scores.len(), documents.len())?;

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

        order_openai_scores(response.data, documents.len())
    }
}

#[async_trait]
impl RerankingProvider for Client {
    async fn rerank(&self, query: &RerankQuery, documents: &[RerankDocument]) -> Result<Vec<f64>> {
        match &self.backend {
            Backend::Remote(client) => match client.dialect {
                Dialect::DeepInfra => {
                    client
                        .rerank_deepinfra(self.instruction.as_deref(), query, documents)
                        .await
                }
                Dialect::OpenAI => client.rerank_openai(query, documents).await,
                Dialect::LlamaCpp => {
                    unreachable!("local execution is handled outside RemoteClient")
                }
            },
            #[cfg(feature = "local")]
            Backend::Local(client) => {
                if documents.is_empty() {
                    return Ok(Vec::new());
                }
                if query.text.trim().is_empty() {
                    return Err(Error::EmptyRerankQuery);
                }

                let formatted = documents
                    .iter()
                    .map(|document| self.format_local_qwen3_input(query, document))
                    .collect::<Vec<_>>();
                let scores = client.score_texts(&formatted).await?;
                ensure_score_count(scores.len(), documents.len())?;
                Ok(scores)
            }
        }
    }
}

fn validate_reranker_family(model_family: ModelFamily) -> Result<()> {
    if model_family != ModelFamily::Qwen3 {
        return Err(Error::UnsupportedConfiguration {
            message: "reranking currently supports ModelFamily::Qwen3 only".to_string(),
        });
    }

    Ok(())
}

fn ensure_score_count(scores: usize, inputs: usize) -> Result<()> {
    if scores != inputs {
        return Err(Error::RerankScoreCountMismatch { scores, inputs });
    }

    Ok(())
}

fn order_openai_scores(items: Vec<OpenAiRerankData>, inputs: usize) -> Result<Vec<f64>> {
    ensure_score_count(items.len(), inputs)?;

    let mut scores = vec![None; inputs];
    for item in items {
        let slot = scores
            .get_mut(item.index)
            .ok_or(Error::InvalidRerankScoreIndex {
                index: item.index,
                inputs,
            })?;
        if slot.is_some() {
            return Err(Error::InvalidRerankScoreIndex {
                index: item.index,
                inputs,
            });
        }
        *slot = Some(item.relevance_score.clamp(0.0, 1.0));
    }

    scores
        .into_iter()
        .enumerate()
        .map(|(index, score)| {
            score.ok_or(Error::InvalidRerankScoreIndex { index, inputs })
        })
        .collect()
}

fn estimate_token_count(query: &RerankQuery, documents: &[RerankDocument]) -> u32 {
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
            dialect: Dialect::DeepInfra,
            model_family: ModelFamily::Qwen3,
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
    async fn rerank_deepinfra_rejects_score_count_mismatch() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/inference/test-model"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "scores": [0.9]
            })))
            .mount(&mock_server)
            .await;

        let client = Client::new(RerankerConfig {
            api_key: None,
            base_url: mock_server.uri(),
            timeout: Duration::from_secs(10),
            dialect: Dialect::DeepInfra,
            model_family: ModelFamily::Qwen3,
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
                token_count: 1,
            },
            RerankDocument {
                text: "b".to_string(),
                token_count: 1,
            },
        ];

        let err = client.rerank(&query, &documents).await.unwrap_err();
        assert!(matches!(
            err,
            Error::RerankScoreCountMismatch {
                scores: 1,
                inputs: 2
            }
        ));
    }

    #[tokio::test]
    async fn rerank_deepinfra_requires_query() {
        let mock_server = MockServer::start().await;

        let client = Client::new(RerankerConfig {
            api_key: None,
            base_url: mock_server.uri(),
            timeout: Duration::from_secs(10),
            dialect: Dialect::DeepInfra,
            model_family: ModelFamily::Qwen3,
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
            dialect: Dialect::DeepInfra,
            model_family: ModelFamily::Qwen3,
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

    #[test]
    fn reranker_requires_qwen3_family() {
        let result = Client::new(RerankerConfig {
            api_key: None,
            base_url: "https://api.deepinfra.com/v1".to_string(),
            timeout: Duration::from_secs(10),
            dialect: Dialect::DeepInfra,
            model_family: ModelFamily::Gemma,
            model: "test-model".to_string(),
            instruction: None,
            requests_per_minute: 1000,
            max_concurrent_requests: 10,
            tokens_per_minute: 1_000_000,
        });

        assert!(matches!(
            result,
            Err(Error::UnsupportedConfiguration { .. })
        ));
    }

    #[cfg(not(feature = "local"))]
    #[test]
    fn llama_cpp_requires_local_feature() {
        let result = Client::new(RerankerConfig {
            api_key: None,
            base_url: String::new(),
            timeout: Duration::from_secs(1),
            dialect: Dialect::LlamaCpp,
            model_family: ModelFamily::Qwen3,
            model: "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf"
                .to_string(),
            instruction: None,
            requests_per_minute: 1,
            max_concurrent_requests: 1,
            tokens_per_minute: 1,
        });

        assert!(matches!(result, Err(Error::LocalFeatureRequired { .. })));
    }

    #[test]
    fn local_qwen3_instruction_uses_default_and_override() {
        let default_client = Client::new(RerankerConfig {
            api_key: None,
            base_url: "https://api.deepinfra.com/v1".to_string(),
            timeout: Duration::from_secs(10),
            dialect: Dialect::DeepInfra,
            model_family: ModelFamily::Qwen3,
            model: "test-model".to_string(),
            instruction: None,
            requests_per_minute: 1000,
            max_concurrent_requests: 10,
            tokens_per_minute: 1_000_000,
        })
        .unwrap();
        let custom_client = Client::new(RerankerConfig {
            api_key: None,
            base_url: "https://api.deepinfra.com/v1".to_string(),
            timeout: Duration::from_secs(10),
            dialect: Dialect::DeepInfra,
            model_family: ModelFamily::Qwen3,
            model: "test-model".to_string(),
            instruction: Some("rank docs".to_string()),
            requests_per_minute: 1000,
            max_concurrent_requests: 10,
            tokens_per_minute: 1_000_000,
        })
        .unwrap();
        let query = RerankQuery {
            text: "memory safety".to_string(),
            token_count: 2,
        };
        let document = RerankDocument {
            text: "Rust prevents data races".to_string(),
            token_count: 4,
        };

        assert_eq!(
            default_client.format_local_qwen3_input(&query, &document),
            format!(
                "Instruct: {}\nQuery: memory safety\nDocument: Rust prevents data races",
                ModelFamily::Qwen3.default_query_instruction()
            )
        );
        assert_eq!(
            custom_client.format_local_qwen3_input(&query, &document),
            "Instruct: rank docs\nQuery: memory safety\nDocument: Rust prevents data races"
        );
    }

    #[cfg(feature = "local")]
    #[test]
    fn local_reranker_rejects_unsupported_model() {
        let result = Client::new(RerankerConfig {
            api_key: None,
            base_url: String::new(),
            timeout: Duration::from_secs(1),
            dialect: Dialect::LlamaCpp,
            model_family: ModelFamily::Qwen3,
            model: "hf:example/unsupported.gguf".to_string(),
            instruction: None,
            requests_per_minute: 1,
            max_concurrent_requests: 1,
            tokens_per_minute: 1,
        });

        assert!(matches!(
            result,
            Err(Error::UnsupportedLocalModel {
                kind: "reranking",
                ..
            })
        ));
    }

    #[tokio::test]
    async fn rerank_openai_rejects_invalid_index_mapping() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/rerank"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [
                    { "index": 0, "relevanceScore": 0.8 },
                    { "index": 3, "relevanceScore": 0.1 },
                    { "index": 1, "relevanceScore": 0.6 }
                ]
            })))
            .mount(&mock_server)
            .await;

        let client = Client::new(RerankerConfig {
            api_key: None,
            base_url: mock_server.uri(),
            timeout: Duration::from_secs(10),
            dialect: Dialect::OpenAI,
            model_family: ModelFamily::Qwen3,
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

        let err = client.rerank(&query, &documents).await.unwrap_err();
        assert!(matches!(
            err,
            Error::InvalidRerankScoreIndex {
                index: 3,
                inputs: 3
            }
        ));
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
            dialect: Dialect::OpenAI,
            model_family: ModelFamily::Qwen3,
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
