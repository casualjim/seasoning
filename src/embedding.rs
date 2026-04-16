//! Text embedding generation with rate limiting and retrieval-aware formatting.
//!
//! ```rust,no_run
//! use std::time::Duration;
//!
//! use std::sync::Arc;
//!
//! use secrecy::SecretString;
//! use seasoning::EmbeddingProvider;
//! use seasoning::embedding::{
//!     Client, Dialect, EmbedderConfig, EmbeddingInput, EmbeddingRole, ModelFamily,
//!     RemoteEmbedderConfig, Tokenizer,
//! };
//!
//! # async fn example() -> seasoning::Result<()> {
//! let client = Client::new(EmbedderConfig::remote(
//!     ModelFamily::Qwen3,
//!     Tokenizer::Tiktoken {
//!         encoding: "cl100k_base".to_string(),
//!         tokenizer: Arc::new(tiktoken_rs::cl100k_base()?),
//!     },
//!     "Qwen/Qwen3-Embedding-0.6B",
//!     None,
//!     RemoteEmbedderConfig {
//!         api_key: Some(SecretString::from("your-api-key")),
//!         base_url: "https://api.deepinfra.com/v1/openai".to_string(),
//!         timeout: Duration::from_secs(30),
//!         dialect: Dialect::DeepInfra,
//!         embedding_dim: 1024,
//!         requests_per_minute: 1000,
//!         max_concurrent_requests: 50,
//!         tokens_per_minute: 1_000_000,
//!     },
//! )?)?;
//!
//! let inputs = vec![EmbeddingInput {
//!     role: EmbeddingRole::Query,
//!     text: "memory safety".to_string(),
//!     title: None,
//!     token_count: 2,
//! }];
//!
//! let _ = client.embed(&inputs).await?;
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use secrecy::SecretString;
use serde::Deserialize;
use serde_json::json;
use tracing::debug;

use crate::EmbeddingProvider;
use crate::Result;
use crate::api::PreparedEmbeddingInput;
#[cfg(feature = "local")]
use crate::local::LocalEmbeddingClient;
use crate::reqwestx::{ApiClient, ApiClientConfig};
pub use crate::{
    Dialect, EmbedOutput, EmbeddingInput, EmbeddingRole, ModelFamily, ProviderDialect, Tokenizer,
};

/// Remote transport/runtime settings for embeddings.
#[derive(Debug, Clone)]
pub struct RemoteEmbedderConfig {
    /// Optional API key for authentication.
    pub api_key: Option<SecretString>,
    /// Base URL for the embedding API endpoint.
    pub base_url: String,
    /// Request timeout duration.
    pub timeout: Duration,
    /// Remote execution dialect.
    pub dialect: Dialect,
    /// Dimension of the embedding vectors returned by the model.
    pub embedding_dim: usize,
    /// Maximum number of requests per minute.
    pub requests_per_minute: usize,
    /// Maximum number of concurrent requests allowed.
    pub max_concurrent_requests: usize,
    /// Maximum number of tokens per minute.
    pub tokens_per_minute: u32,
}

/// Configuration for the embedding client.
#[derive(Debug, Clone)]
pub struct EmbedderConfig {
    api_key: Option<SecretString>,
    base_url: String,
    timeout: Duration,
    dialect: Dialect,
    model_family: ModelFamily,
    tokenizer: Tokenizer,
    model: String,
    query_instruction: Option<String>,
    embedding_dim: usize,
    requests_per_minute: usize,
    max_concurrent_requests: usize,
    tokens_per_minute: u32,
}

impl EmbedderConfig {
    /// Creates a remote embedding configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if `remote.dialect` is [`Dialect::LlamaCpp`] or if one of the
    /// remote limits is zero.
    pub fn remote(
        model_family: ModelFamily,
        tokenizer: Tokenizer,
        model: impl Into<String>,
        query_instruction: Option<String>,
        remote: RemoteEmbedderConfig,
    ) -> Result<Self> {
        if remote.dialect == Dialect::LlamaCpp {
            return Err(crate::Error::InvalidConfiguration {
                message: "EmbedderConfig::remote requires Dialect::OpenAI or Dialect::DeepInfra"
                    .to_string(),
            });
        }
        if remote.timeout.is_zero() {
            return Err(crate::Error::InvalidConfiguration {
                message: "embedding.timeout_seconds must be greater than zero".to_string(),
            });
        }
        validate_positive_usize(remote.embedding_dim, "embedding.embedding_dim")?;
        validate_positive_usize(remote.requests_per_minute, "embedding.requests_per_minute")?;
        validate_positive_usize(
            remote.max_concurrent_requests,
            "embedding.max_concurrent_requests",
        )?;
        validate_positive_u32(remote.tokens_per_minute, "embedding.tokens_per_minute")?;

        Ok(Self {
            api_key: remote.api_key,
            base_url: remote.base_url,
            timeout: remote.timeout,
            dialect: remote.dialect,
            model_family,
            tokenizer,
            model: model.into(),
            query_instruction,
            embedding_dim: remote.embedding_dim,
            requests_per_minute: remote.requests_per_minute,
            max_concurrent_requests: remote.max_concurrent_requests,
            tokens_per_minute: remote.tokens_per_minute,
        })
    }

    /// Creates a local llama.cpp embedding configuration.
    #[must_use]
    pub fn local(
        model_family: ModelFamily,
        tokenizer: Tokenizer,
        model: impl Into<String>,
        query_instruction: Option<String>,
    ) -> Self {
        Self {
            api_key: None,
            base_url: String::new(),
            timeout: Duration::from_secs(0),
            dialect: Dialect::LlamaCpp,
            model_family,
            tokenizer,
            model: model.into(),
            query_instruction,
            embedding_dim: 0,
            requests_per_minute: 0,
            max_concurrent_requests: 0,
            tokens_per_minute: 0,
        }
    }

    /// Returns the configured execution dialect.
    #[must_use]
    pub fn dialect(&self) -> Dialect {
        self.dialect
    }

    /// Returns the configured retrieval model family.
    #[must_use]
    pub fn model_family(&self) -> ModelFamily {
        self.model_family
    }

    /// Returns the configured tokenizer.
    #[must_use]
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Returns the configured model identifier.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Returns the configured query instruction, if any.
    #[must_use]
    pub fn query_instruction(&self) -> Option<&str> {
        self.query_instruction.as_deref()
    }
}

#[async_trait]
trait EmbeddingBackend: Send + Sync {
    async fn embed_prepared(
        &self,
        input: &[PreparedEmbeddingInput],
        estimated_tokens: u32,
    ) -> Result<EmbedOutput>;
}

#[derive(Clone)]
pub struct Client {
    model_family: ModelFamily,
    query_instruction: Option<String>,
    tokenizer: Tokenizer,
    backend: Arc<dyn EmbeddingBackend>,
}

#[derive(Clone)]
struct RemoteClient {
    client: ApiClient,
    model: String,
    dimension: usize,
    dialect: Dialect,
}

/// Internal representation of a single embedding from the API response.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct EmbeddingObject {
    index: usize,
    embedding: Vec<f32>,
}

/// Internal representation of the embedding API response.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct EmbedApiResponse {
    data: Vec<EmbeddingObject>,
}

impl Client {
    pub fn new(config: EmbedderConfig) -> Result<Self> {
        let model_family = config.model_family;
        let query_instruction = config.query_instruction.clone();
        let tokenizer = config.tokenizer.clone();

        let backend: Arc<dyn EmbeddingBackend> = match config.dialect {
            Dialect::OpenAI | Dialect::DeepInfra => Arc::new(RemoteClient::new(config)?),
            Dialect::LlamaCpp => {
                #[cfg(feature = "local")]
                {
                    Arc::new(LocalEmbeddingClient::new(
                        config.model_family,
                        &config.model,
                    )?)
                }
                #[cfg(not(feature = "local"))]
                {
                    let _ = config;
                    return Err(crate::Error::LocalFeatureRequired {
                        dialect: Dialect::LlamaCpp.to_string(),
                    });
                }
            }
        };

        Ok(Self {
            model_family,
            query_instruction,
            tokenizer,
            backend,
        })
    }

    fn estimate_token_count(prepared: &[PreparedEmbeddingInput]) -> u32 {
        prepared.iter().fold(0u32, |tokens, item| {
            tokens.saturating_add(item.token_count() as u32)
        })
    }

    fn prepare_input(&self, input: &EmbeddingInput) -> Result<PreparedEmbeddingInput> {
        let rendered = self
            .model_family
            .format_embedding_input(input, self.query_instruction.as_deref());
        self.tokenizer.prepare(&rendered)
    }

    fn prepare_inputs(&self, input: &[EmbeddingInput]) -> Result<Vec<PreparedEmbeddingInput>> {
        input.iter().map(|item| self.prepare_input(item)).collect()
    }
}

impl RemoteClient {
    fn new(config: EmbedderConfig) -> Result<Self> {
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
}

#[async_trait]
impl EmbeddingBackend for RemoteClient {
    async fn embed_prepared(
        &self,
        input: &[PreparedEmbeddingInput],
        estimated_tokens: u32,
    ) -> Result<EmbedOutput> {
        if input.is_empty() {
            return Ok(EmbedOutput {
                embeddings: Vec::new(),
            });
        }

        let payload = match self.dialect {
            Dialect::OpenAI | Dialect::DeepInfra => {
                json!({
                    "input": input.iter().map(PreparedEmbeddingInput::token_ids).collect::<Vec<_>>(),
                    "model": self.model,
                    "encoding_format": "float",
                    "dimensions": self.dimension
                })
            }
            Dialect::LlamaCpp => unreachable!("local execution is handled outside RemoteClient"),
        };

        let response: EmbedApiResponse = self
            .client
            .post_json("/embeddings", &payload, estimated_tokens)
            .await?;

        let embeddings = order_embeddings(response.data, input.len())?;

        Ok(EmbedOutput { embeddings })
    }
}

#[cfg(feature = "local")]
#[async_trait]
impl EmbeddingBackend for LocalEmbeddingClient {
    async fn embed_prepared(
        &self,
        input: &[PreparedEmbeddingInput],
        _estimated_tokens: u32,
    ) -> Result<EmbedOutput> {
        self.embed_prepared(input).await
    }
}

fn validate_positive_usize(value: usize, field: &'static str) -> Result<()> {
    if value == 0 {
        return Err(crate::Error::InvalidConfiguration {
            message: format!("{field} must be greater than zero"),
        });
    }

    Ok(())
}

fn validate_positive_u32(value: u32, field: &'static str) -> Result<()> {
    if value == 0 {
        return Err(crate::Error::InvalidConfiguration {
            message: format!("{field} must be greater than zero"),
        });
    }

    Ok(())
}

fn order_embeddings(items: Vec<EmbeddingObject>, inputs: usize) -> Result<Vec<Vec<f32>>> {
    if items.len() != inputs {
        return Err(crate::Error::EmbeddingCountMismatch {
            embeddings: items.len(),
            inputs,
        });
    }

    let mut embeddings = vec![None; inputs];
    for item in items {
        let slot = embeddings
            .get_mut(item.index)
            .ok_or(crate::Error::InvalidEmbeddingIndex {
                index: item.index,
                inputs,
            })?;
        if slot.is_some() {
            return Err(crate::Error::InvalidEmbeddingIndex {
                index: item.index,
                inputs,
            });
        }
        *slot = Some(item.embedding);
    }

    embeddings
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| {
            embedding.ok_or(crate::Error::InvalidEmbeddingIndex { index, inputs })
        })
        .collect()
}

#[async_trait]
impl EmbeddingProvider for Client {
    async fn embed(&self, input: &[EmbeddingInput]) -> Result<EmbedOutput> {
        debug!("Embedding input batch_size: {}", input.len());
        let prepared = self.prepare_inputs(input)?;
        let estimated_tokens = Self::estimate_token_count(&prepared);

        self.backend
            .embed_prepared(&prepared, estimated_tokens)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic;
    use std::time::Duration;

    use secrecy::SecretString;
    use serde_json::json;
    use wiremock::matchers::{body_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn remote_config(
        model_family: ModelFamily,
        tokenizer: Tokenizer,
        model: &str,
        query_instruction: Option<String>,
        remote: RemoteEmbedderConfig,
    ) -> EmbedderConfig {
        EmbedderConfig::remote(
            model_family,
            tokenizer,
            model.to_string(),
            query_instruction,
            remote,
        )
        .unwrap()
    }

    fn local_config(
        model_family: ModelFamily,
        tokenizer: Tokenizer,
        model: &str,
    ) -> EmbedderConfig {
        EmbedderConfig::local(model_family, tokenizer, model.to_string(), None)
    }

    fn test_tokenizer() -> Tokenizer {
        Tokenizer::Tiktoken {
            encoding: "cl100k_base".to_string(),
            tokenizer: std::sync::Arc::new(tiktoken_rs::cl100k_base().unwrap()),
        }
    }

    fn token_ids(tokenizer: &Tokenizer, text: &str) -> Vec<u32> {
        tokenizer.prepare(text).unwrap().token_ids().to_vec()
    }

    fn semantic_input(text: &str, token_count: usize) -> EmbeddingInput {
        EmbeddingInput {
            role: EmbeddingRole::Query,
            text: text.to_string(),
            title: None,
            token_count,
        }
    }

    #[tokio::test]
    async fn embed_openai_success_reorders_embeddings_from_semantic_input() {
        let mock_server = MockServer::start().await;
        let input = vec![
            semantic_input("first query", 3),
            semantic_input("second query", 2),
        ];
        let tokenizer = test_tokenizer();
        let prepared = vec![
            token_ids(
                &tokenizer,
                &format!(
                    "Instruct: {}\nQuery: first query",
                    ModelFamily::Qwen3.default_query_instruction()
                ),
            ),
            token_ids(
                &tokenizer,
                &format!(
                    "Instruct: {}\nQuery: second query",
                    ModelFamily::Qwen3.default_query_instruction()
                ),
            ),
        ];

        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .and(body_json(json!({
                "input": prepared,
                "model": "test-model",
                "encoding_format": "float",
                "dimensions": 2
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [
                    { "index": 1, "embedding": [0.8, 0.9] },
                    { "index": 0, "embedding": [0.1, 0.2] }
                ]
            })))
            .mount(&mock_server)
            .await;

        let client = Client::new(remote_config(
            ModelFamily::Qwen3,
            tokenizer.clone(),
            "test-model",
            None,
            RemoteEmbedderConfig {
                api_key: None,
                base_url: mock_server.uri(),
                timeout: Duration::from_secs(10),
                dialect: Dialect::OpenAI,
                embedding_dim: 2,
                requests_per_minute: 1000,
                max_concurrent_requests: 10,
                tokens_per_minute: 1_000_000,
            },
        ))
        .unwrap();

        let output = client.embed(&input).await.unwrap();
        assert_eq!(output.embeddings, vec![vec![0.1, 0.2], vec![0.8, 0.9]]);
    }

    #[tokio::test]
    async fn embed_deepinfra_success_sets_authorization_header() {
        let mock_server = MockServer::start().await;
        let input = vec![semantic_input("retrieval query", 3)];
        let tokenizer = test_tokenizer();
        let prepared = vec![token_ids(
            &tokenizer,
            &format!(
                "Instruct: {}\nQuery: retrieval query",
                ModelFamily::Qwen3.default_query_instruction()
            ),
        )];

        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .and(header("Authorization", "Bearer test_key"))
            .and(body_json(json!({
                "input": prepared,
                "model": "test-model",
                "encoding_format": "float",
                "dimensions": 3
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [
                    { "index": 0, "embedding": [0.2, 0.4, 0.6] }
                ]
            })))
            .mount(&mock_server)
            .await;

        let client = Client::new(remote_config(
            ModelFamily::Qwen3,
            tokenizer.clone(),
            "test-model",
            None,
            RemoteEmbedderConfig {
                api_key: Some(SecretString::from("test_key")),
                base_url: mock_server.uri(),
                timeout: Duration::from_secs(10),
                dialect: Dialect::DeepInfra,
                embedding_dim: 3,
                requests_per_minute: 1000,
                max_concurrent_requests: 10,
                tokens_per_minute: 1_000_000,
            },
        ))
        .unwrap();

        let output = client.embed(&input).await.unwrap();
        assert_eq!(output.embeddings, vec![vec![0.2, 0.4, 0.6]]);
    }

    #[test]
    fn embedder_new_should_not_panic_on_invalid_api_key() {
        let result = panic::catch_unwind(|| {
            let _ = Client::new(remote_config(
                ModelFamily::Qwen3,
                test_tokenizer(),
                "test-model",
                None,
                RemoteEmbedderConfig {
                    api_key: Some(SecretString::from("bad\nkey")),
                    base_url: "http://127.0.0.1:1".to_string(),
                    timeout: Duration::from_secs(10),
                    dialect: Dialect::OpenAI,
                    embedding_dim: 2,
                    requests_per_minute: 1000,
                    max_concurrent_requests: 300,
                    tokens_per_minute: 1,
                },
            ));
        });

        assert!(
            result.is_ok(),
            "Client::new should return Err, not panic, for invalid API keys"
        );
    }

    #[test]
    fn prepare_inputs_uses_client_query_instruction() {
        let tokenizer = test_tokenizer();
        let client = Client::new(remote_config(
            ModelFamily::Qwen3,
            tokenizer.clone(),
            "test-model",
            Some("custom instruction".to_string()),
            RemoteEmbedderConfig {
                api_key: None,
                base_url: "http://127.0.0.1:1".to_string(),
                timeout: Duration::from_secs(10),
                dialect: Dialect::OpenAI,
                embedding_dim: 2,
                requests_per_minute: 1,
                max_concurrent_requests: 1,
                tokens_per_minute: 1,
            },
        ))
        .unwrap();

        let prepared = client
            .prepare_inputs(&[EmbeddingInput {
                role: EmbeddingRole::Query,
                text: "rust ownership".to_string(),
                title: None,
                token_count: 2,
            }])
            .unwrap();

        assert_eq!(
            prepared[0].token_ids(),
            token_ids(
                &tokenizer,
                "Instruct: custom instruction\nQuery: rust ownership"
            )
        );
    }

    #[cfg(not(feature = "local"))]
    #[test]
    fn llama_cpp_requires_local_feature() {
        let result = Client::new(local_config(
            ModelFamily::Gemma,
            test_tokenizer(),
            "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf",
        ));

        assert!(matches!(
            result,
            Err(crate::Error::LocalFeatureRequired { .. })
        ));
    }

    #[cfg(feature = "local")]
    #[test]
    fn local_embedder_rejects_invalid_model_spec() {
        let result = Client::new(local_config(
            ModelFamily::Gemma,
            test_tokenizer(),
            "not-a-hf-model-spec",
        ));

        assert!(matches!(
            result,
            Err(crate::Error::UnsupportedConfiguration { .. })
        ));
    }

    #[test]
    fn order_embeddings_rejects_out_of_range_index() {
        let err = order_embeddings(
            vec![EmbeddingObject {
                index: 2,
                embedding: vec![0.1, 0.2],
            }],
            1,
        )
        .unwrap_err();

        assert!(matches!(
            err,
            crate::Error::InvalidEmbeddingIndex {
                index: 2,
                inputs: 1
            }
        ));
    }

    #[test]
    fn order_embeddings_rejects_duplicate_index() {
        let err = order_embeddings(
            vec![
                EmbeddingObject {
                    index: 0,
                    embedding: vec![0.1, 0.2],
                },
                EmbeddingObject {
                    index: 0,
                    embedding: vec![0.3, 0.4],
                },
            ],
            2,
        )
        .unwrap_err();

        assert!(matches!(
            err,
            crate::Error::InvalidEmbeddingIndex {
                index: 0,
                inputs: 2
            }
        ));
    }

    #[test]
    fn order_embeddings_rejects_count_mismatch() {
        let err = order_embeddings(
            vec![
                EmbeddingObject {
                    index: 0,
                    embedding: vec![0.1, 0.2],
                },
                EmbeddingObject {
                    index: 1,
                    embedding: vec![0.3, 0.4],
                },
                EmbeddingObject {
                    index: 2,
                    embedding: vec![0.5, 0.6],
                },
            ],
            4,
        )
        .unwrap_err();

        assert!(matches!(
            err,
            crate::Error::EmbeddingCountMismatch {
                embeddings: 3,
                inputs: 4
            }
        ));
    }
}
