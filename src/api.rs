use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{Error, Result};

const DEFAULT_GEMMA_QUERY_TASK: &str = "search documents";
const DEFAULT_QWEN3_RETRIEVAL_INSTRUCTION: &str =
    "Given a web search query, retrieve relevant passages that answer the query";

/// Prepared embedding payload ready for execution.
///
/// Token IDs must come from the tokenizer for the exact final rendered payload
/// that will be embedded.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "Vec<u32>", into = "Vec<u32>")]
pub struct PreparedEmbeddingInput {
    token_ids: Vec<u32>,
}

impl PreparedEmbeddingInput {
    /// Create one prepared embedding payload from pre-tokenized model input.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EmptyPreparedEmbeddingInput`] when `token_ids` is empty.
    pub fn new(token_ids: Vec<u32>) -> Result<Self> {
        if token_ids.is_empty() {
            return Err(Error::EmptyPreparedEmbeddingInput);
        }

        Ok(Self { token_ids })
    }

    #[must_use]
    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    #[must_use]
    pub fn token_count(&self) -> usize {
        self.token_ids.len()
    }

    #[must_use]
    pub fn into_token_ids(self) -> Vec<u32> {
        self.token_ids
    }
}

impl AsRef<[u32]> for PreparedEmbeddingInput {
    fn as_ref(&self) -> &[u32] {
        self.token_ids()
    }
}

impl TryFrom<Vec<u32>> for PreparedEmbeddingInput {
    type Error = Error;

    fn try_from(token_ids: Vec<u32>) -> Result<Self> {
        Self::new(token_ids)
    }
}

impl From<PreparedEmbeddingInput> for Vec<u32> {
    fn from(input: PreparedEmbeddingInput) -> Self {
        input.into_token_ids()
    }
}

pub struct BatchItem<M> {
    pub meta: M,
    /// Prepared embedding payload for one document/item.
    pub input: PreparedEmbeddingInput,
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

/// Semantic embedding input.
///
/// Callers format this with [`ModelFamily::format_embedding_input`] or
/// [`crate::embedding::Client::render_input`] before tokenizing the rendered
/// payload for execution.
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddDecision {
    Continue,
    Flush,
}

pub trait BatchingStrategy: Send {
    fn add(&mut self, token_count: usize) -> AddDecision;
    fn flush(&mut self);
    fn max_items_per_batch(&self) -> usize;
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
/// use seasoning::embedding::{EmbedOutput, PreparedEmbeddingInput};
/// use seasoning::EmbeddingProvider;
///
/// struct MockEmbedder;
///
/// #[async_trait]
/// impl EmbeddingProvider for MockEmbedder {
///     async fn embed(&self, input: &[PreparedEmbeddingInput]) -> Result<EmbedOutput> {
///         let embeddings = input.iter().map(|_| vec![0.0; 1024]).collect();
///         Ok(EmbedOutput { embeddings })
///     }
/// }
/// ```
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Generate embeddings for prepared model inputs.
    ///
    /// # Arguments
    ///
    /// * `input` - Slice of prepared embedding inputs containing token IDs for
    ///   the final rendered model payloads
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
    async fn embed(&self, input: &[PreparedEmbeddingInput]) -> Result<EmbedOutput>;
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
        let err = PreparedEmbeddingInput::new(Vec::new()).unwrap_err();
        assert!(matches!(err, Error::EmptyPreparedEmbeddingInput));
    }

    #[test]
    fn prepared_embedding_input_reports_token_count() {
        let input = PreparedEmbeddingInput::new(vec![1, 2, 3]).unwrap();
        assert_eq!(input.token_count(), 3);
        assert_eq!(input.token_ids(), &[1, 2, 3]);
    }

    #[test]
    fn gemma_query_formatting_uses_custom_task() {
        let input = EmbeddingInput {
            role: EmbeddingRole::Query,
            text: "rust async runtime".to_string(),
            title: None,
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
        };
        let without_title = EmbeddingInput {
            role: EmbeddingRole::Document,
            text: "Rust enables fearless concurrency".to_string(),
            title: None,
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
        };
        let untitled = EmbeddingInput {
            role: EmbeddingRole::Document,
            text: "Borrow checking catches aliasing bugs".to_string(),
            title: None,
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
