use reqwest::StatusCode;

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

    #[error("embedder result channel closed with {pending} pending batches")]
    EmbedderResultChannelClosed { pending: usize },

    #[error("rerank query cannot be empty")]
    EmptyRerankQuery,

    #[error("provider dialect '{value}' is invalid (expected: openai|deepinfra)")]
    InvalidProviderDialect { value: String },

    #[error("api request failed with status {status}")]
    ApiStatus {
        status: StatusCode,
        body: Option<String>,
    },

    #[error("embedder returned {embeddings} embeddings for {inputs} inputs")]
    EmbeddingCountMismatch { embeddings: usize, inputs: usize },
}

pub type Result<T> = std::result::Result<T, Error>;
