//! Text embedding execution with retrieval-aware rendering helpers.
//!
//! ```rust,no_run
//! use std::time::Duration;
//!
//! use secrecy::SecretString;
//! use seasoning::EmbeddingProvider;
//! use seasoning::embedding::{
//!     Client, Dialect, EmbedderConfig, EmbeddingInput, EmbeddingRole, ModelFamily,
//!     PreparedEmbeddingInput,
//! };
//!
//! # async fn example() -> seasoning::Result<()> {
//! let client = Client::new(EmbedderConfig {
//!     api_key: Some(SecretString::from("your-api-key")),
//!     base_url: "https://api.deepinfra.com/v1/openai".to_string(),
//!     timeout: Duration::from_secs(30),
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
//! let semantic = EmbeddingInput {
//!     role: EmbeddingRole::Query,
//!     text: "memory safety".to_string(),
//!     title: None,
//! };
//! let rendered = client.render_input(&semantic);
//! let _ = rendered;
//!
//! // Tokenize `rendered` with the tokenizer for the target embedding model,
//! // then execute with the resulting token ids.
//! let prepared = vec![PreparedEmbeddingInput::new(vec![1, 2, 3])?];
//! let _ = client.embed(&prepared).await?;
//! # Ok(())
//! # }
//! ```

use std::time::Duration;

use async_trait::async_trait;
use secrecy::SecretString;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::EmbeddingProvider;
use crate::Result;
pub use crate::api::PreparedEmbeddingInput;
#[cfg(feature = "local")]
use crate::local::LocalEmbeddingClient;
use crate::reqwestx::{ApiClient, ApiClientConfig};
pub use crate::{
    Dialect, EmbedOutput, EmbeddingInput, EmbeddingRole, ModelFamily, ProviderDialect,
};

/// Configuration for the embedding client.
#[derive(Debug, Clone)]
pub struct EmbedderConfig {
    /// Optional API key for authentication.
    pub api_key: Option<SecretString>,
    /// Base URL for the embedding API endpoint.
    pub base_url: String,
    /// Request timeout duration.
    pub timeout: Duration,
    /// Backend dialect used for execution.
    pub dialect: Dialect,
    /// Retrieval-model family used by rendering helpers.
    pub model_family: ModelFamily,
    /// Model identifier.
    pub model: String,
    /// Optional query instruction or task text used by rendering helpers.
    pub query_instruction: Option<String>,
    /// Dimension of the embedding vectors returned by the model.
    pub embedding_dim: usize,
    /// Maximum number of requests per minute.
    pub requests_per_minute: usize,
    /// Maximum number of concurrent requests allowed.
    pub max_concurrent_requests: usize,
    /// Maximum number of tokens per minute.
    pub tokens_per_minute: u32,
}

#[derive(Clone)]
pub struct Client {
    model_family: ModelFamily,
    query_instruction: Option<String>,
    backend: Backend,
}

#[derive(Clone)]
enum Backend {
    Remote(RemoteClient),
    #[cfg(feature = "local")]
    Local(LocalEmbeddingClient),
}

#[derive(Clone)]
struct RemoteClient {
    client: ApiClient,
    model: String,
    dimension: usize,
    dialect: Dialect,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct EmbeddingRequest<'a> {
    input: Vec<&'a [u32]>,
    model: &'a str,
    encoding_format: &'static str,
    dimensions: usize,
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
        match config.dialect {
            Dialect::OpenAI | Dialect::DeepInfra => {
                let model_family = config.model_family;
                let query_instruction = config.query_instruction.clone();
                let remote = RemoteClient::new(config)?;
                Ok(Self {
                    model_family,
                    query_instruction,
                    backend: Backend::Remote(remote),
                })
            }
            Dialect::LlamaCpp => {
                #[cfg(feature = "local")]
                {
                    Ok(Self {
                        model_family: config.model_family,
                        query_instruction: config.query_instruction,
                        backend: Backend::Local(LocalEmbeddingClient::new(
                            config.model_family,
                            &config.model,
                        )?),
                    })
                }
                #[cfg(not(feature = "local"))]
                {
                    let _ = config;
                    Err(crate::Error::LocalFeatureRequired {
                        dialect: Dialect::LlamaCpp.to_string(),
                    })
                }
            }
        }
    }

    #[must_use]
    pub fn render_input(&self, input: &EmbeddingInput) -> String {
        self.model_family
            .format_embedding_input(input, self.query_instruction.as_deref())
    }

    #[must_use]
    pub fn render_inputs(&self, input: &[EmbeddingInput]) -> Vec<String> {
        input.iter().map(|item| self.render_input(item)).collect()
    }

    fn estimate_token_count(&self, input: &[PreparedEmbeddingInput]) -> u32 {
        input.iter().fold(0u32, |tokens, item| {
            tokens.saturating_add(item.token_count() as u32)
        })
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

    async fn embed_prepared(
        &self,
        prepared: &[PreparedEmbeddingInput],
        estimated_tokens: u32,
    ) -> Result<EmbedOutput> {
        if prepared.is_empty() {
            return Ok(EmbedOutput {
                embeddings: Vec::new(),
            });
        }

        let payload = match self.dialect {
            Dialect::OpenAI | Dialect::DeepInfra => EmbeddingRequest {
                input: prepared
                    .iter()
                    .map(PreparedEmbeddingInput::token_ids)
                    .collect(),
                model: self.model.as_str(),
                encoding_format: "float",
                dimensions: self.dimension,
            },
            Dialect::LlamaCpp => unreachable!("local execution is handled outside RemoteClient"),
        };

        let response: EmbedApiResponse = self
            .client
            .post_json("/embeddings", &payload, estimated_tokens)
            .await?;

        let embeddings = order_embeddings(response.data, prepared.len())?;

        Ok(EmbedOutput { embeddings })
    }
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
    async fn embed(&self, input: &[PreparedEmbeddingInput]) -> Result<EmbedOutput> {
        debug!("Embedding input batch_size: {}", input.len());
        let estimated_tokens = self.estimate_token_count(input);

        match &self.backend {
            Backend::Remote(client) => client.embed_prepared(input, estimated_tokens).await,
            #[cfg(feature = "local")]
            Backend::Local(client) => client.embed_prepared(input).await,
        }
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

    fn prepared_input(token_ids: &[u32]) -> PreparedEmbeddingInput {
        PreparedEmbeddingInput::new(token_ids.to_vec()).unwrap()
    }

    #[tokio::test]
    async fn embed_openai_success_reorders_embeddings_from_token_input() {
        let mock_server = MockServer::start().await;
        let input = vec![prepared_input(&[11, 12, 13]), prepared_input(&[21, 22])];

        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .and(body_json(json!({
                "input": [[11, 12, 13], [21, 22]],
                "model": "test-model",
                "encodingFormat": "float",
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

        let client = Client::new(EmbedderConfig {
            api_key: None,
            base_url: mock_server.uri(),
            timeout: Duration::from_secs(10),
            dialect: Dialect::OpenAI,
            model_family: ModelFamily::Qwen3,
            model: "test-model".to_string(),
            query_instruction: None,
            embedding_dim: 2,
            requests_per_minute: 1000,
            max_concurrent_requests: 10,
            tokens_per_minute: 1_000_000,
        })
        .unwrap();

        let output = client.embed(&input).await.unwrap();
        assert_eq!(output.embeddings, vec![vec![0.1, 0.2], vec![0.8, 0.9]]);
    }

    #[tokio::test]
    async fn embed_deepinfra_success_sets_authorization_header() {
        let mock_server = MockServer::start().await;
        let input = vec![prepared_input(&[5, 8, 13])];

        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .and(header("Authorization", "Bearer test_key"))
            .and(body_json(json!({
                "input": [[5, 8, 13]],
                "model": "test-model",
                "encodingFormat": "float",
                "dimensions": 3
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [
                    { "index": 0, "embedding": [0.2, 0.4, 0.6] }
                ]
            })))
            .mount(&mock_server)
            .await;

        let client = Client::new(EmbedderConfig {
            api_key: Some(SecretString::from("test_key")),
            base_url: mock_server.uri(),
            timeout: Duration::from_secs(10),
            dialect: Dialect::DeepInfra,
            model_family: ModelFamily::Qwen3,
            model: "test-model".to_string(),
            query_instruction: None,
            embedding_dim: 3,
            requests_per_minute: 1000,
            max_concurrent_requests: 10,
            tokens_per_minute: 1_000_000,
        })
        .unwrap();

        let output = client.embed(&input).await.unwrap();
        assert_eq!(output.embeddings, vec![vec![0.2, 0.4, 0.6]]);
    }

    #[test]
    fn embedder_new_should_not_panic_on_invalid_api_key() {
        let result = panic::catch_unwind(|| {
            let _ = Client::new(EmbedderConfig {
                api_key: Some(SecretString::from("bad\nkey")),
                base_url: "http://127.0.0.1:1".to_string(),
                timeout: Duration::from_secs(1),
                dialect: Dialect::OpenAI,
                model_family: ModelFamily::Qwen3,
                model: "test-model".to_string(),
                query_instruction: None,
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

    #[test]
    fn render_input_uses_client_query_instruction() {
        let client = Client::new(EmbedderConfig {
            api_key: None,
            base_url: "http://127.0.0.1:1".to_string(),
            timeout: Duration::from_secs(1),
            dialect: Dialect::OpenAI,
            model_family: ModelFamily::Qwen3,
            model: "test-model".to_string(),
            query_instruction: Some("custom instruction".to_string()),
            embedding_dim: 2,
            requests_per_minute: 1,
            max_concurrent_requests: 1,
            tokens_per_minute: 1,
        })
        .unwrap();

        let rendered = client.render_input(&EmbeddingInput {
            role: EmbeddingRole::Query,
            text: "rust ownership".to_string(),
            title: None,
        });

        assert_eq!(
            rendered,
            "Instruct: custom instruction\nQuery: rust ownership"
        );
    }

    #[cfg(not(feature = "local"))]
    #[test]
    fn llama_cpp_requires_local_feature() {
        let result = Client::new(EmbedderConfig {
            api_key: None,
            base_url: String::new(),
            timeout: Duration::from_secs(1),
            dialect: Dialect::LlamaCpp,
            model_family: ModelFamily::Gemma,
            model: "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf".to_string(),
            query_instruction: None,
            embedding_dim: 768,
            requests_per_minute: 1,
            max_concurrent_requests: 1,
            tokens_per_minute: 1,
        });

        assert!(matches!(
            result,
            Err(crate::Error::LocalFeatureRequired { .. })
        ));
    }

    #[cfg(feature = "local")]
    #[test]
    fn local_embedder_rejects_unsupported_model_for_family() {
        let result = Client::new(EmbedderConfig {
            api_key: None,
            base_url: String::new(),
            timeout: Duration::from_secs(1),
            dialect: Dialect::LlamaCpp,
            model_family: ModelFamily::Gemma,
            model: "hf:example/unsupported.gguf".to_string(),
            query_instruction: None,
            embedding_dim: 768,
            requests_per_minute: 1,
            max_concurrent_requests: 1,
            tokens_per_minute: 1,
        });

        assert!(matches!(
            result,
            Err(crate::Error::UnsupportedLocalModel {
                kind: "embedding",
                ..
            })
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
