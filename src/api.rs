use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{Error, Result};

const DEFAULT_GEMMA_QUERY_TASK: &str = "search documents";
const DEFAULT_QWEN3_RETRIEVAL_INSTRUCTION: &str =
    "Given a web search query, retrieve relevant passages that answer the query";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PreparedEmbeddingInput {
    pub(crate) token_ids: Vec<u32>,
    pub(crate) text: String,
}

impl PreparedEmbeddingInput {
    pub(crate) fn new(token_ids: Vec<u32>, text: String) -> Result<Self> {
        if token_ids.is_empty() {
            return Err(Error::EmptyPreparedEmbeddingInput);
        }

        Ok(Self { token_ids, text })
    }

    #[must_use]
    pub(crate) fn token_count(&self) -> usize {
        self.token_ids.len()
    }
}

/// One semantic embedding item for [`crate::service::EmbedderService`].
#[derive(Debug, Default)]
pub struct BatchItem<M> {
    /// Caller metadata returned alongside the embedding vectors.
    pub meta: M,
    /// Whether this item is a retrieval query or document.
    pub role: EmbeddingRole,
    /// Semantic text to embed.
    pub text: String,
    /// Optional title metadata for families that use it.
    pub title: Option<String>,
    /// Token count used for batching decisions.
    pub token_count: usize,
}

/// Backend/runtime dialect for embedding and reranking execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Dialect {
    /// OpenAI-compatible remote APIs.
    #[default]
    OpenAI,
    /// DeepInfra remote APIs.
    DeepInfra,
    /// Local llama.cpp execution.
    #[serde(
        rename = "llamacpp",
        alias = "llama-cpp",
        alias = "llama_cpp",
        alias = "llama.cpp"
    )]
    LlamaCpp,
}

impl Dialect {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OpenAI => "openai",
            Self::DeepInfra => "deepinfra",
            Self::LlamaCpp => "llamacpp",
        }
    }
}

impl std::fmt::Display for Dialect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Backwards-compatible alias for the previous public name.
pub type ProviderDialect = Dialect;

/// Preloaded tokenizer instances used by the embedding model layer.
///
/// This mirrors the tokenizer kinds from `niblits`, but only permits already
/// loaded tokenizer values instead of string/model-id configuration.
#[derive(Clone)]
pub enum Tokenizer {
    /// Simple character-based tokenization.
    Characters,
    /// Preloaded OpenAI tiktoken tokenizer.
    Tiktoken {
        encoding: String,
        tokenizer: Arc<tiktoken_rs::CoreBPE>,
    },
    /// Preloaded Hugging Face tokenizer.
    HuggingFace {
        model_id: String,
        tokenizer: Arc<tokenizers::Tokenizer>,
    },
}

impl Tokenizer {
    pub(crate) fn prepare(&self, text: String) -> Result<PreparedEmbeddingInput> {
        let token_ids = match self {
            Self::Characters => {
                return Err(Error::InvalidConfiguration {
                    message: "embedding preparation requires a tokenizer that yields model token ids; the characters tokenizer only counts characters".to_string(),
                });
            }
            Self::Tiktoken { tokenizer, .. } => tokenizer.encode_ordinary(&text),
            Self::HuggingFace { tokenizer, .. } => tokenizer
                .encode(text.as_str(), false)
                .map(|encoding| encoding.get_ids().to_vec())
                .map_err(|error| Error::InvalidConfiguration {
                    message: format!("failed to encode with HF tokenizer: {error}"),
                })?,
        };

        PreparedEmbeddingInput::new(token_ids, text)
    }
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Characters => f.write_str("Characters"),
            Self::Tiktoken { encoding, .. } => write!(f, "Tiktoken({encoding})"),
            Self::HuggingFace { model_id, .. } => write!(f, "HuggingFace({model_id})"),
        }
    }
}

/// Retrieval-family semantics used to format embedding and reranking inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelFamily {
    Gemma,
    #[default]
    Qwen3,
}

impl ModelFamily {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Gemma => "gemma",
            Self::Qwen3 => "qwen3",
        }
    }

    #[must_use]
    pub fn default_query_instruction(self) -> &'static str {
        match self {
            Self::Gemma => DEFAULT_GEMMA_QUERY_TASK,
            Self::Qwen3 => DEFAULT_QWEN3_RETRIEVAL_INSTRUCTION,
        }
    }

    #[must_use]
    pub fn format_embedding_input(
        self,
        input: &EmbeddingInput,
        query_instruction: Option<&str>,
    ) -> String {
        match (self, input.role) {
            (Self::Gemma, EmbeddingRole::Query) => {
                let instruction = normalize_optional_text(query_instruction)
                    .unwrap_or_else(|| self.default_query_instruction().to_string());
                format!("task: {instruction} | query: {}", input.text)
            }
            (Self::Gemma, EmbeddingRole::Document) => {
                let title = normalize_optional_text(input.title.as_deref())
                    .unwrap_or_else(|| "none".to_string());
                format!("title: {title} | text: {}", input.text)
            }
            (Self::Qwen3, EmbeddingRole::Query) => {
                let instruction = normalize_optional_text(query_instruction)
                    .unwrap_or_else(|| self.default_query_instruction().to_string());
                format!("Instruct: {instruction}\nQuery: {}", input.text)
            }
            (Self::Qwen3, EmbeddingRole::Document) => {
                match normalize_optional_text(input.title.as_deref()) {
                    Some(title) => format!("{title}\n{}", input.text),
                    None => input.text.clone(),
                }
            }
        }
    }

    #[must_use]
    pub fn format_reranker_input(
        self,
        query: &RerankQuery,
        document: &RerankDocument,
        instruction: Option<&str>,
    ) -> String {
        match self {
            Self::Qwen3 => {
                let instruction = normalize_optional_text(instruction)
                    .unwrap_or_else(|| self.default_query_instruction().to_string());
                format!(
                    "Instruct: {instruction}\nQuery: {}\nDocument: {}",
                    query.text, document.text
                )
            }
            Self::Gemma => format!("Query: {}\nDocument: {}", query.text, document.text),
        }
    }
}

impl std::fmt::Display for ModelFamily {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Retrieval role for an embedding input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingRole {
    Query,
    #[default]
    Document,
}

impl std::fmt::Display for EmbeddingRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Query => f.write_str("query"),
            Self::Document => f.write_str("document"),
        }
    }
}

/// Input for a single embedding request.
///
/// The caller supplies semantic retrieval metadata and the crate formats and
/// prepares the final model payload internally based on the configured
/// [`ModelFamily`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddingInput {
    /// Whether this input is a retrieval query or indexed document.
    #[serde(default)]
    pub role: EmbeddingRole,
    /// Semantic text content for the embedding request.
    pub text: String,
    /// Optional title metadata used by families that support document titles.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Caller-maintained token count for batching or external accounting.
    pub token_count: usize,
}

/// Output from an embedding request.
///
/// Contains the generated embedding vectors, one per input.
/// The embeddings are returned in the same order as the input texts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbedOutput {
    /// Generated embedding vectors, one per input text
    pub embeddings: Vec<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RerankQuery {
    /// Query text to rank documents against.
    pub text: String,
    /// Pre-computed token count for `text`.
    ///
    /// Tokenization is intentionally out of scope for this crate; callers must provide
    /// the correct count for the target model/tokenizer.
    pub token_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RerankDocument {
    /// Document text to score.
    pub text: String,
    /// Pre-computed token count for `text`.
    ///
    /// Tokenization is intentionally out of scope for this crate; callers must provide
    /// the correct count for the target model/tokenizer.
    pub token_count: usize,
}

/// Batching strategy response for a newly added item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddDecision {
    /// Keep accumulating items in the current batch.
    Continue,
    /// Flush the current batch before starting a new one with the added item.
    Flush,
}

/// Strategy interface for token-aware batch assembly.
pub trait BatchingStrategy: Send {
    /// Records one additional item with the given token count.
    fn add(&mut self, token_count: usize) -> AddDecision;
    /// Resets the strategy after a flush.
    fn flush(&mut self);
    /// Returns the hard item limit for one batch.
    fn max_items_per_batch(&self) -> usize;
    /// Returns the hard token limit for one batch.
    fn max_tokens_per_batch(&self) -> usize;
}

/// Trait for embedding providers.
///
/// This trait abstracts over different embedding implementations,
/// allowing for easy testing and provider swapping.
///
/// # Example Implementation
///
/// ```rust,no_run
/// use async_trait::async_trait;
/// use seasoning::Result;
/// use seasoning::embedding::{EmbedOutput, EmbeddingInput};
/// use seasoning::EmbeddingProvider;
///
/// struct MockEmbedder;
///
/// #[async_trait]
/// impl EmbeddingProvider for MockEmbedder {
///     async fn embed(&self, input: &[EmbeddingInput]) -> Result<EmbedOutput> {
///         let embeddings = input.iter().map(|_| vec![0.0; 1024]).collect();
///         Ok(EmbedOutput { embeddings })
///     }
/// }
/// ```
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Generate embeddings for the given inputs.
    ///
    /// # Arguments
    ///
    /// * `input` - Slice of semantic embedding inputs containing role, text, optional title,
    ///   and token counts
    ///
    /// # Returns
    ///
    /// Returns an [`EmbedOutput`] containing the generated embedding vectors.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The API request fails
    /// - Rate limits are exceeded and retries are exhausted
    /// - The response cannot be parsed
    /// - Network errors occur
    async fn embed(&self, input: &[EmbeddingInput]) -> Result<EmbedOutput>;
}

/// Trait for reranking providers.
///
/// This trait abstracts over different reranking implementations,
/// allowing for easy testing and provider swapping.
///
/// # Example Implementation
///
/// ```rust,no_run
/// use async_trait::async_trait;
/// use seasoning::Result;
/// use seasoning::RerankingProvider;
/// use seasoning::{RerankDocument, RerankQuery};
///
/// struct MockReranker;
///
/// #[async_trait]
/// impl RerankingProvider for MockReranker {
///     async fn rerank(&self, query: &RerankQuery, documents: &[RerankDocument]) -> Result<Vec<f64>> {
///         let _ = (query, documents);
///         Ok(vec![0.9, 0.5, 0.7])
///     }
/// }
/// ```
#[async_trait]
pub trait RerankingProvider: Send + Sync {
    /// Rerank documents based on their relevance to a query.
    ///
    /// # Arguments
    ///
    /// * `query` - The search query to rank documents against
    /// * `documents` - Slice of document texts to rank
    ///
    /// # Returns
    ///
    /// Returns a vector of relevance scores, one per document, in the same order
    /// as the input documents. Scores are typically in the range [0.0, 1.0],
    /// where higher scores indicate greater relevance to the query.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The query is empty
    /// - The API request fails
    /// - The response cannot be parsed
    /// - Network errors occur
    async fn rerank(&self, query: &RerankQuery, documents: &[RerankDocument]) -> Result<Vec<f64>>;
}

fn normalize_optional_text(value: Option<&str>) -> Option<String> {
    let normalized = value?.trim();
    if normalized.is_empty() {
        None
    } else {
        Some(normalized.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepared_embedding_input_rejects_empty_tokens() {
        let err = PreparedEmbeddingInput::new(Vec::new(), String::new()).unwrap_err();
        assert!(matches!(err, Error::EmptyPreparedEmbeddingInput));
    }

    #[test]
    fn prepared_embedding_input_reports_token_count() {
        let input = PreparedEmbeddingInput::new(vec![1, 2, 3], "test".to_string()).unwrap();
        assert_eq!(input.token_count(), 3);
        assert_eq!(input.token_ids, &[1, 2, 3]);
    }

    #[test]
    fn gemma_query_formatting_uses_custom_task() {
        let input = EmbeddingInput {
            role: EmbeddingRole::Query,
            text: "rust async runtime".to_string(),
            title: None,
            token_count: 3,
        };

        let formatted = ModelFamily::Gemma.format_embedding_input(&input, Some("custom task"));

        assert_eq!(formatted, "task: custom task | query: rust async runtime");
    }

    #[test]
    fn gemma_query_formatting_uses_default_task_for_missing_or_blank_instruction() {
        let input = EmbeddingInput {
            role: EmbeddingRole::Query,
            text: "rust async runtime".to_string(),
            title: None,
            token_count: 3,
        };

        let expected = format!(
            "task: {} | query: rust async runtime",
            ModelFamily::Gemma.default_query_instruction()
        );

        assert_eq!(
            ModelFamily::Gemma.format_embedding_input(&input, None),
            expected
        );
        assert_eq!(
            ModelFamily::Gemma.format_embedding_input(&input, Some("   ")),
            expected
        );
    }

    #[test]
    fn gemma_document_formatting_uses_title_or_none() {
        let with_title = EmbeddingInput {
            role: EmbeddingRole::Document,
            text: "Rust enables fearless concurrency".to_string(),
            title: Some("Rust".to_string()),
            token_count: 4,
        };
        let without_title = EmbeddingInput {
            role: EmbeddingRole::Document,
            text: "Rust enables fearless concurrency".to_string(),
            title: None,
            token_count: 4,
        };

        assert_eq!(
            ModelFamily::Gemma.format_embedding_input(&with_title, Some("ignored")),
            "title: Rust | text: Rust enables fearless concurrency"
        );
        assert_eq!(
            ModelFamily::Gemma.format_embedding_input(&without_title, Some("ignored")),
            "title: none | text: Rust enables fearless concurrency"
        );
    }

    #[test]
    fn qwen3_query_formatting_uses_default_and_override() {
        let input = EmbeddingInput {
            role: EmbeddingRole::Query,
            text: "rust ownership".to_string(),
            title: None,
            token_count: 2,
        };

        assert_eq!(
            ModelFamily::Qwen3.format_embedding_input(&input, None),
            format!(
                "Instruct: {}\nQuery: rust ownership",
                ModelFamily::Qwen3.default_query_instruction()
            )
        );
        assert_eq!(
            ModelFamily::Qwen3.format_embedding_input(&input, Some("custom instruction")),
            "Instruct: custom instruction\nQuery: rust ownership"
        );
    }

    #[test]
    fn qwen3_query_formatting_trims_custom_instruction() {
        let input = EmbeddingInput {
            role: EmbeddingRole::Query,
            text: "rust ownership".to_string(),
            title: None,
            token_count: 2,
        };

        assert_eq!(
            ModelFamily::Qwen3.format_embedding_input(&input, Some("  custom instruction  ")),
            "Instruct: custom instruction\nQuery: rust ownership"
        );
    }

    #[test]
    fn qwen3_document_formatting_ignores_query_instruction() {
        let titled = EmbeddingInput {
            role: EmbeddingRole::Document,
            text: "Borrow checking catches aliasing bugs".to_string(),
            title: Some("Borrow Checker".to_string()),
            token_count: 4,
        };
        let untitled = EmbeddingInput {
            role: EmbeddingRole::Document,
            text: "Borrow checking catches aliasing bugs".to_string(),
            title: None,
            token_count: 4,
        };

        assert_eq!(
            ModelFamily::Qwen3.format_embedding_input(&titled, Some("ignored")),
            "Borrow Checker\nBorrow checking catches aliasing bugs"
        );
        assert_eq!(
            ModelFamily::Qwen3.format_embedding_input(&untitled, Some("ignored")),
            "Borrow checking catches aliasing bugs"
        );
    }

    #[test]
    fn qwen3_reranker_formatting_uses_default_and_override() {
        let query = RerankQuery {
            text: "memory safety".to_string(),
            token_count: 2,
        };
        let document = RerankDocument {
            text: "Rust prevents data races".to_string(),
            token_count: 4,
        };

        assert_eq!(
            ModelFamily::Qwen3.format_reranker_input(&query, &document, None),
            format!(
                "Instruct: {}\nQuery: memory safety\nDocument: Rust prevents data races",
                ModelFamily::Qwen3.default_query_instruction()
            )
        );
        assert_eq!(
            ModelFamily::Qwen3.format_reranker_input(&query, &document, Some("rank docs")),
            "Instruct: rank docs\nQuery: memory safety\nDocument: Rust prevents data races"
        );
    }
}
