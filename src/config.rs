use secrecy::SecretString;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub embedding: Embedding,
    pub reranker: Reranker,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    pub url: String,
    #[serde(skip_serializing)]
    pub api_key: Option<SecretString>,
    pub model: String,
    pub dialect: String,
    pub timeout_seconds: u64,
    pub embedding_dim: usize,
    pub requests_per_minute: usize,
    pub max_concurrent_requests: usize,
    pub tokens_per_minute: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reranker {
    pub url: String,
    #[serde(skip_serializing)]
    pub api_key: Option<SecretString>,
    pub model: String,
    pub dialect: String,
    pub timeout_seconds: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instruction: Option<String>,
}
