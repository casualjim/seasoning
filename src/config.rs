//! Configuration types for embedding and reranking services.
//!
//! ```rust
//! use seasoning::AppConfig;
//! use serde_json::json;
//!
//! let config_json = json!({
//!     "embedding": {
//!         "url": "https://api.deepinfra.com/v1/openai",
//!         "model": "Qwen/Qwen3-Embedding-0.6B",
//!         "dialect": "deepinfra",
//!         "model_family": "qwen3",
//!         "query_instruction": "Given a query, retrieve relevant passages",
//!         "timeout_seconds": 10,
//!         "embedding_dim": 1024,
//!         "requests_per_minute": 1000,
//!         "max_concurrent_requests": 50,
//!         "tokens_per_minute": 1000000
//!     },
//!     "reranker": {
//!         "url": "https://api.deepinfra.com/v1",
//!         "model": "Qwen/Qwen3-Reranker-0.6B",
//!         "dialect": "deepinfra",
//!         "model_family": "qwen3",
//!         "timeout_seconds": 10,
//!         "requests_per_minute": 1000,
//!         "max_concurrent_requests": 50,
//!         "tokens_per_minute": 1000000
//!     }
//! });
//!
//! let config: AppConfig = serde_json::from_value(config_json)?;
//! assert_eq!(config.embedding.dialect, seasoning::Dialect::DeepInfra);
//! assert_eq!(config.embedding.model_family, seasoning::ModelFamily::Qwen3);
//! // `dialect` values also accept `llamacpp`, `llama-cpp`, `llama_cpp`,
//! // and `llama.cpp` via `Dialect` serde aliases.
//! # Ok::<(), serde_json::Error>(())
//! ```

use std::time::Duration;

use secrecy::SecretString;
use serde::{Deserialize, Serialize};

use crate::embedding::EmbedderConfig;
use crate::reranker::RerankerConfig;
use crate::{Dialect, ModelFamily, Result};

/// Top-level application config for embedding and reranking clients.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// Embedding client configuration.
    pub embedding: Embedding,
    /// Reranker client configuration.
    pub reranker: Reranker,
}

/// Serializable embedding client configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    #[serde(default)]
    pub url: String,
    #[serde(skip_serializing)]
    pub api_key: Option<SecretString>,
    pub model: String,
    #[serde(default)]
    pub tokenizer: String,
    pub dialect: Dialect,
    #[serde(default)]
    pub model_family: ModelFamily,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query_instruction: Option<String>,
    pub timeout_seconds: u64,
    pub embedding_dim: usize,
    #[serde(default = "default_context_length")]
    pub context_length: usize,
    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: usize,
    #[serde(default = "default_embedding_workers")]
    pub workers: usize,
    pub requests_per_minute: usize,
    pub max_concurrent_requests: usize,
    pub tokens_per_minute: u32,
}

/// Serializable reranker client configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reranker {
    #[serde(default)]
    pub url: String,
    #[serde(skip_serializing)]
    pub api_key: Option<SecretString>,
    pub model: String,
    pub dialect: Dialect,
    #[serde(default)]
    pub model_family: ModelFamily,
    pub timeout_seconds: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instruction: Option<String>,
    #[serde(default = "default_reranker_requests_per_minute")]
    pub requests_per_minute: usize,
    #[serde(default = "default_reranker_max_concurrent_requests")]
    pub max_concurrent_requests: usize,
    #[serde(default = "default_reranker_tokens_per_minute")]
    pub tokens_per_minute: u32,
}

impl Embedding {
    /// Converts this config into an [`EmbedderConfig`].
    pub fn to_embedder_config(&self) -> Result<EmbedderConfig> {
        validate_positive_u64(self.timeout_seconds, "embedding.timeout_seconds")?;
        validate_positive_usize(self.embedding_dim, "embedding.embedding_dim")?;
        validate_positive_usize(self.context_length, "embedding.context_length")?;
        validate_positive_usize(self.max_batch_size, "embedding.max_batch_size")?;
        validate_positive_usize(self.workers, "embedding.workers")?;
        validate_positive_usize(self.requests_per_minute, "embedding.requests_per_minute")?;
        validate_positive_usize(
            self.max_concurrent_requests,
            "embedding.max_concurrent_requests",
        )?;
        validate_positive_u32(self.tokens_per_minute, "embedding.tokens_per_minute")?;

        Ok(EmbedderConfig {
            api_key: self.api_key.clone(),
            base_url: self.url.clone(),
            timeout: Duration::from_secs(self.timeout_seconds),
            dialect: self.dialect,
            model_family: self.model_family,
            model: self.model.clone(),
            query_instruction: self.query_instruction.clone(),
            embedding_dim: self.embedding_dim,
            requests_per_minute: self.requests_per_minute,
            max_concurrent_requests: self.max_concurrent_requests,
            tokens_per_minute: self.tokens_per_minute,
        })
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
        })
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
    fn embedding_config_accepts_llama_dot_cpp_alias() {
        let config: Embedding = serde_json::from_value(serde_json::json!({
            "url": "",
            "model": "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf",
            "tokenizer": "",
            "dialect": "llama.cpp",
            "model_family": "gemma",
            "query_instruction": "retrieve docs",
            "timeout_seconds": 30,
            "embedding_dim": 768,
            "context_length": default_context_length(),
            "max_batch_size": default_max_batch_size(),
            "workers": default_embedding_workers(),
            "requests_per_minute": 1,
            "max_concurrent_requests": 1,
            "tokens_per_minute": 1_000_000
        }))
        .unwrap();

        let converted = config.to_embedder_config().unwrap();

        assert_eq!(converted.dialect, Dialect::LlamaCpp);
        assert_eq!(converted.model_family, ModelFamily::Gemma);
        assert_eq!(
            converted.query_instruction.as_deref(),
            Some("retrieve docs")
        );
    }

    #[test]
    fn reranker_config_accepts_llama_dot_cpp_alias() {
        let config: Reranker = serde_json::from_value(serde_json::json!({
            "url": "",
            "model": "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf",
            "dialect": "llama.cpp",
            "model_family": "qwen3",
            "timeout_seconds": 30,
            "instruction": "rank docs",
            "requests_per_minute": 1,
            "max_concurrent_requests": 1,
            "tokens_per_minute": 1_000_000
        }))
        .unwrap();

        let converted = config.to_reranker_config().unwrap();

        assert_eq!(converted.dialect, Dialect::LlamaCpp);
        assert_eq!(converted.model_family, ModelFamily::Qwen3);
        assert_eq!(converted.instruction.as_deref(), Some("rank docs"));
    }

    #[test]
    fn config_conversion_rejects_zero_limits() {
        let embedding: Embedding = serde_json::from_value(serde_json::json!({
            "url": "",
            "model": "Qwen/Qwen3-Embedding-0.6B",
            "dialect": "deepinfra",
            "model_family": "qwen3",
            "timeout_seconds": 0,
            "embedding_dim": 1024,
            "requests_per_minute": 1,
            "max_concurrent_requests": 1,
            "tokens_per_minute": 1
        }))
        .unwrap();
        assert!(matches!(
            embedding.to_embedder_config(),
            Err(crate::Error::InvalidConfiguration { .. })
        ));

        let reranker: Reranker = serde_json::from_value(serde_json::json!({
            "url": "",
            "model": "Qwen/Qwen3-Reranker-0.6B",
            "dialect": "deepinfra",
            "model_family": "qwen3",
            "timeout_seconds": 10,
            "requests_per_minute": 0,
            "max_concurrent_requests": 1,
            "tokens_per_minute": 1
        }))
        .unwrap();
        assert!(matches!(
            reranker.to_reranker_config(),
            Err(crate::Error::InvalidConfiguration { .. })
        ));
    }
}
