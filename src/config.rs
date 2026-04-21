//! Configuration types for embedding and reranking services.
//!
//! ```rust,no_run
//! use std::sync::Arc;
//!
//! use secrecy::SecretString;
//! use seasoning::{AppConfig, Dialect, Embedding, ModelFamily, Reranker, Tokenizer};
//!
//! let config = AppConfig {
//!     embedding: Embedding {
//!         url: "https://api.deepinfra.com/v1/openai".to_string(),
//!         api_key: Some(SecretString::from("token")),
//!         model: "Qwen/Qwen3-Embedding-0.6B".to_string(),
//!         tokenizer: Tokenizer::Tiktoken {
//!             encoding: "cl100k_base".to_string(),
//!             tokenizer: Arc::new(tiktoken_rs::cl100k_base().unwrap()),
//!         },
//!         dialect: Dialect::DeepInfra,
//!         model_family: ModelFamily::Qwen3,
//!         query_instruction: Some("Given a query, retrieve relevant passages".to_string()),
//!         timeout_seconds: 10,
//!         embedding_dim: 1024,
//!         requests_per_minute: 1000,
//!         max_concurrent_requests: 50,
//!         tokens_per_minute: 1_000_000,
//!     },
//!     reranker: Reranker {
//!         url: "https://api.deepinfra.com/v1".to_string(),
//!         api_key: Some(SecretString::from("token")),
//!         model: "Qwen/Qwen3-Reranker-0.6B".to_string(),
//!         dialect: Dialect::DeepInfra,
//!         model_family: ModelFamily::Qwen3,
//!         timeout_seconds: 10,
//!         instruction: None,
//!         requests_per_minute: 1000,
//!         max_concurrent_requests: 50,
//!         tokens_per_minute: 1_000_000,
//!     },
//! };
//!
//! assert_eq!(config.embedding.dialect, Dialect::DeepInfra);
//! assert_eq!(config.embedding.model_family, ModelFamily::Qwen3);
//! ```

use std::time::Duration;

use secrecy::SecretString;

use crate::embedding::{EmbedderConfig, RemoteEmbedderConfig};
use crate::reranker::RerankerConfig;
use crate::{Dialect, ModelFamily, Result, Tokenizer};

/// Top-level application config for embedding and reranking clients.
#[derive(Debug, Clone)]
pub struct AppConfig {
    /// Embedding client configuration.
    pub embedding: Embedding,
    /// Reranker client configuration.
    pub reranker: Reranker,
}

/// Embedding client configuration.
#[derive(Debug, Clone)]
pub struct Embedding {
    pub url: String,
    pub api_key: Option<SecretString>,
    pub model: String,
    pub tokenizer: Tokenizer,
    pub dialect: Dialect,
    pub model_family: ModelFamily,
    pub query_instruction: Option<String>,
    pub timeout_seconds: u64,
    pub embedding_dim: usize,
    pub requests_per_minute: usize,
    pub max_concurrent_requests: usize,
    pub tokens_per_minute: u32,
}

/// Reranker client configuration.
#[derive(Debug, Clone)]
pub struct Reranker {
    pub url: String,
    pub api_key: Option<SecretString>,
    pub model: String,
    pub dialect: Dialect,
    pub model_family: ModelFamily,
    pub timeout_seconds: u64,
    pub instruction: Option<String>,
    pub requests_per_minute: usize,
    pub max_concurrent_requests: usize,
    pub tokens_per_minute: u32,
}

impl Embedding {
    /// Converts this config into an [`EmbedderConfig`].
    pub fn to_embedder_config(&self) -> Result<EmbedderConfig> {
        match self.dialect {
            Dialect::LlamaCpp => Ok(EmbedderConfig::local(
                self.model_family,
                self.tokenizer.clone(),
                self.model.clone(),
                self.query_instruction.clone(),
            )),
            Dialect::OpenAI | Dialect::DeepInfra => EmbedderConfig::remote(
                self.model_family,
                self.tokenizer.clone(),
                self.model.clone(),
                self.query_instruction.clone(),
                RemoteEmbedderConfig {
                    api_key: self.api_key.clone(),
                    base_url: self.url.clone(),
                    timeout: Duration::from_secs(self.timeout_seconds),
                    dialect: self.dialect,
                    embedding_dim: self.embedding_dim,
                    requests_per_minute: self.requests_per_minute,
                    max_concurrent_requests: self.max_concurrent_requests,
                    tokens_per_minute: self.tokens_per_minute,
                },
            ),
        }
    }
}

impl Reranker {
    /// Converts this config into a [`RerankerConfig`].
    pub fn to_reranker_config(&self) -> Result<RerankerConfig> {
        validate_positive_u64(self.timeout_seconds, "reranker.timeout_seconds")?;
        validate_positive_usize(self.requests_per_minute, "reranker.requests_per_minute")?;
        validate_positive_usize(
            self.max_concurrent_requests,
            "reranker.max_concurrent_requests",
        )?;
        validate_positive_u32(self.tokens_per_minute, "reranker.tokens_per_minute")?;

        Ok(RerankerConfig {
            api_key: self.api_key.clone(),
            base_url: self.url.clone(),
            timeout: Duration::from_secs(self.timeout_seconds),
            dialect: self.dialect,
            model_family: self.model_family,
            model: self.model.clone(),
            instruction: self.instruction.clone(),
            requests_per_minute: self.requests_per_minute,
            max_concurrent_requests: self.max_concurrent_requests,
            tokens_per_minute: self.tokens_per_minute,
            #[cfg(feature = "local")]
            backend: None,
        })
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

fn validate_positive_u64(value: u64, field: &'static str) -> Result<()> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedder_config_preserves_tokenizer_and_fields() {
        let config = Embedding {
            url: String::new(),
            api_key: None,
            model: "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf".to_string(),
            tokenizer: Tokenizer::Characters,
            dialect: Dialect::LlamaCpp,
            model_family: ModelFamily::Gemma,
            query_instruction: Some("retrieve docs".to_string()),
            timeout_seconds: 30,
            embedding_dim: 768,
            requests_per_minute: 1,
            max_concurrent_requests: 1,
            tokens_per_minute: 1_000_000,
        };

        let converted = config.to_embedder_config().unwrap();

        assert_eq!(converted.dialect(), Dialect::LlamaCpp);
        assert_eq!(converted.model_family(), ModelFamily::Gemma);
        assert!(matches!(converted.tokenizer(), Tokenizer::Characters));
        assert_eq!(converted.query_instruction(), Some("retrieve docs"));
    }

    #[test]
    fn reranker_config_preserves_fields() {
        let config = Reranker {
            url: String::new(),
            api_key: None,
            model: "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf"
                .to_string(),
            dialect: Dialect::LlamaCpp,
            model_family: ModelFamily::Qwen3,
            timeout_seconds: 30,
            instruction: Some("rank docs".to_string()),
            requests_per_minute: 1,
            max_concurrent_requests: 1,
            tokens_per_minute: 1_000_000,
        };

        let converted = config.to_reranker_config().unwrap();

        assert_eq!(converted.dialect, Dialect::LlamaCpp);
        assert_eq!(converted.model_family, ModelFamily::Qwen3);
        assert_eq!(converted.instruction.as_deref(), Some("rank docs"));
    }

    #[test]
    fn config_conversion_rejects_zero_limits() {
        let embedding = Embedding {
            url: String::new(),
            api_key: None,
            model: "Qwen/Qwen3-Embedding-0.6B".to_string(),
            tokenizer: Tokenizer::Characters,
            dialect: Dialect::DeepInfra,
            model_family: ModelFamily::Qwen3,
            query_instruction: None,
            timeout_seconds: 0,
            embedding_dim: 1024,
            requests_per_minute: 1,
            max_concurrent_requests: 1,
            tokens_per_minute: 1,
        };
        assert!(matches!(
            embedding.to_embedder_config(),
            Err(crate::Error::InvalidConfiguration { .. })
        ));

        let reranker = Reranker {
            url: String::new(),
            api_key: None,
            model: "Qwen/Qwen3-Reranker-0.6B".to_string(),
            dialect: Dialect::DeepInfra,
            model_family: ModelFamily::Qwen3,
            timeout_seconds: 10,
            instruction: None,
            requests_per_minute: 0,
            max_concurrent_requests: 1,
            tokens_per_minute: 1,
        };
        assert!(matches!(
            reranker.to_reranker_config(),
            Err(crate::Error::InvalidConfiguration { .. })
        ));
    }
}
