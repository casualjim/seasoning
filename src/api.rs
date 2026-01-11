use async_trait::async_trait;

use crate::Result;
use serde::{Deserialize, Serialize};

pub struct BatchItem<M> {
    pub meta: M,
    pub text: String,
    pub token_count: usize,
}

/// Input for a single embedding request.
///
/// Each input consists of the text to embed and a pre-calculated token count.
/// The token count is used for rate limiting purposes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddingInput {
    /// The text to generate embeddings for
    pub text: String,
    /// Pre-calculated token count for rate limiting
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

#[derive(Debug, Clone)]
pub struct RerankQuery {
    /// Query text to rank documents against.
    pub text: String,
    /// Pre-computed token count for `text`.
    ///
    /// Tokenization is intentionally out of scope for this crate; callers must provide
    /// the correct count for the target model/tokenizer.
    pub token_count: usize,
}

#[derive(Debug, Clone)]
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
/// use seasoning::embedding::{EmbeddingInput, EmbedOutput};
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
    /// * `input` - Slice of embedding inputs containing text and token counts
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
