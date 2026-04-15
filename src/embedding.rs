//! Text embedding generation with rate limiting and retrieval-aware formatting.

use std::time::Duration;

use async_trait::async_trait;
use secrecy::SecretString;
use serde::Deserialize;
use serde_json::json;
use tracing::debug;

use crate::EmbeddingProvider;
use crate::Result;
use crate::reqwestx::{ApiClient, ApiClientConfig};
#[cfg(feature = "local")]
use crate::local::LocalEmbeddingClient;
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
    /// Retrieval-model family used for input formatting.
    pub model_family: ModelFamily,
    /// Model identifier.
    pub model: String,
    /// Optional query instruction or task text.
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
            },
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

    fn estimate_token_count(&self, input: &[EmbeddingInput]) -> u32 {
        let mut tokens: u32 = 0;
        for inp in input {
            tokens = tokens.saturating_add(inp.token_count as u32);
        }
        tokens
    }

    fn format_inputs(&self, input: &[EmbeddingInput]) -> Vec<String> {
        input
            .iter()
            .map(|item| {
                self.model_family
                    .format_embedding_input(item, self.query_instruction.as_deref())
            })
            .collect()
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

    async fn embed_texts(&self, batch_texts: &[String], estimated_tokens: u32) -> Result<EmbedOutput> {
        if batch_texts.is_empty() {
            return Ok(EmbedOutput {
                embeddings: Vec::new(),
            });
        }

        let payload = match self.dialect {
            Dialect::OpenAI | Dialect::DeepInfra => {
                json!({
                  "input": batch_texts,
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

        if response.data.len() != batch_texts.len() {
            return Err(crate::Error::EmbeddingCountMismatch {
                embeddings: response.data.len(),
                inputs: batch_texts.len(),
            });
        }

        let mut embeddings = vec![Vec::new(); batch_texts.len()];
        for item in response.data {
            if item.index < embeddings.len() {
                embeddings[item.index] = item.embedding;
            }
        }

        Ok(EmbedOutput { embeddings })
    }
}

#[async_trait]
impl EmbeddingProvider for Client {
    async fn embed(&self, input: &[EmbeddingInput]) -> Result<EmbedOutput> {
        debug!("Embedding input batch_size: {}", input.len());
        let batch_texts = self.format_inputs(input);
        let estimated_tokens = self.estimate_token_count(input);

        match &self.backend {
            Backend::Remote(client) => client.embed_texts(&batch_texts, estimated_tokens).await,
            #[cfg(feature = "local")]
            Backend::Local(client) => client.embed_texts(&batch_texts).await,
        }
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
}
