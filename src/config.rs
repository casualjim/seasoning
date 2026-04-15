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
//! let config: AppConfig = serde_json::from_value(config_json).unwrap();
//! assert_eq!(config.embedding.dialect, "deepinfra");
//! assert_eq!(config.embedding.model_family, "qwen3");
//! // `dialect` values also accept `llamacpp`, `llama-cpp`, and `llama_cpp`
//! // when converted through `to_embedder_config` / `to_reranker_config`.
//! ```

use std::time::Duration;

use secrecy::SecretString;
use serde::{Deserialize, Serialize};

use crate::embedding::EmbedderConfig;
use crate::reranker::RerankerConfig;
use crate::{Dialect, Error, ModelFamily, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub embedding: Embedding,
    pub reranker: Reranker,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    #[serde(default)]
    pub url: String,
    #[serde(skip_serializing)]
    pub api_key: Option<SecretString>,
    pub model: String,
    #[serde(default)]
    pub tokenizer: String,
    pub dialect: String,
    #[serde(default = "default_model_family_string")]
    pub model_family: String,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reranker {
    #[serde(default)]
    pub url: String,
    #[serde(skip_serializing)]
    pub api_key: Option<SecretString>,
    pub model: String,
    pub dialect: String,
    #[serde(default = "default_model_family_string")]
    pub model_family: String,
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
    pub fn to_embedder_config(&self) -> Result<EmbedderConfig> {
        Ok(EmbedderConfig {
            api_key: self.api_key.clone(),
            base_url: self.url.clone(),
            timeout: Duration::from_secs(self.timeout_seconds),
            dialect: parse_provider_dialect(&self.dialect)?,
            model_family: parse_model_family(&self.model_family)?,
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
    pub fn to_reranker_config(&self) -> Result<RerankerConfig> {
        Ok(RerankerConfig {
            api_key: self.api_key.clone(),
            base_url: self.url.clone(),
            timeout: Duration::from_secs(self.timeout_seconds),
            dialect: parse_provider_dialect(&self.dialect)?,
            model_family: parse_model_family(&self.model_family)?,
            model: self.model.clone(),
            instruction: self.instruction.clone(),
            requests_per_minute: self.requests_per_minute,
            max_concurrent_requests: self.max_concurrent_requests,
            tokens_per_minute: self.tokens_per_minute,
        })
    }
}

fn parse_provider_dialect(value: &str) -> Result<Dialect> {
    let normalized = value.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "openai" => Ok(Dialect::OpenAI),
        "deepinfra" => Ok(Dialect::DeepInfra),
        "llamacpp" | "llama-cpp" | "llama_cpp" => Ok(Dialect::LlamaCpp),
        _ => Err(Error::InvalidProviderDialect {
            value: value.to_string(),
        }),
    }
}

fn parse_model_family(value: &str) -> Result<ModelFamily> {
    let normalized = value.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "gemma" => Ok(ModelFamily::Gemma),
        "qwen3" => Ok(ModelFamily::Qwen3),
        _ => Err(Error::InvalidModelFamily {
            value: value.to_string(),
        }),
    }
}

fn default_model_family_string() -> String {
    ModelFamily::Qwen3.as_str().to_string()
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
