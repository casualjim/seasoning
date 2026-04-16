use reqwest::StatusCode;

/// Crate-wide error type.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid api key header value: {0}")]
    InvalidApiKeyHeaderValue(#[source] http::header::InvalidHeaderValue),

    #[error("http client build failed: {0}")]
    HttpClientBuild(#[source] reqwest::Error),

    #[error("request failed: {0}")]
    Request(#[source] reqwest::Error),

    #[error("response body read failed: {0}")]
    ResponseBody(#[source] reqwest::Error),

    #[error(transparent)]
    ResponseJson(#[from] reqwest::Error),

    #[error("request json encode failed: {0}")]
    JsonEncode(#[source] serde_json::Error),

    #[error("response json decode failed: {0}")]
    JsonDecode(#[source] serde_json::Error),

    #[error("rate limiter semaphore closed")]
    SemaphoreClosed,

    #[error("embedder batch channel closed")]
    BatchChannelClosed,

    #[error(
        "batch item token_count {token_count} exceeds max_tokens_per_batch {max_tokens_per_batch}"
    )]
    BatchItemTooLarge {
        token_count: usize,
        max_tokens_per_batch: usize,
    },

    #[error("embedder service requires an active tokio runtime")]
    TokioRuntimeRequired,

    #[error("rerank query cannot be empty")]
    EmptyRerankQuery,

    #[error("provider dialect '{value}' is invalid (expected: openai|deepinfra|llamacpp)")]
    InvalidProviderDialect { value: String },

    #[error("model family '{value}' is invalid (expected: gemma|qwen3)")]
    InvalidModelFamily { value: String },

    #[error("dialect '{dialect}' requires the `local` feature")]
    LocalFeatureRequired { dialect: String },

    #[error("unsupported configuration: {message}")]
    UnsupportedConfiguration { message: String },

    #[error("invalid configuration: {message}")]
    InvalidConfiguration { message: String },

    #[error("local runtime failed: {message}")]
    LocalRuntime { message: String },

    #[error("local runtime channel closed unexpectedly")]
    LocalRuntimeChannelClosed,

    #[error("prepared embedding input token ids cannot be empty")]
    EmptyPreparedEmbeddingInput,

    #[error(
        "embedding input at index {index} contains token id {token_id} that does not fit in i32"
    )]
    InvalidEmbeddingTokenId { index: usize, token_id: u32 },

    #[error("api request failed with status {status}")]
    ApiStatus {
        status: StatusCode,
        body: Option<String>,
    },

    #[error("embedder returned {embeddings} embeddings for {inputs} inputs")]
    EmbeddingCountMismatch { embeddings: usize, inputs: usize },

    #[error("embedder returned an invalid embedding index {index} for {inputs} inputs")]
    InvalidEmbeddingIndex { index: usize, inputs: usize },

    #[error("reranker returned {scores} scores for {inputs} inputs")]
    RerankScoreCountMismatch { scores: usize, inputs: usize },

    #[error("reranker returned an invalid score index {index} for {inputs} inputs")]
    InvalidRerankScoreIndex { index: usize, inputs: usize },
}

/// Crate-wide result alias.
pub type Result<T> = std::result::Result<T, Error>;
