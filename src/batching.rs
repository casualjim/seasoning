use std::num::NonZeroUsize;

use crate::api::{AddDecision, BatchItem, BatchingStrategy};
use crate::{Error, Result};

/// Token-aware batching strategy with hard item and token limits.
pub struct TokenAwareBatchingStrategy {
    max_tokens_per_batch: NonZeroUsize,
    max_items_per_batch: NonZeroUsize,
    current_tokens: usize,
    current_items: usize,
}

impl TokenAwareBatchingStrategy {
    /// Creates a batching strategy with non-zero hard limits.
    #[must_use]
    pub fn new(max_tokens_per_batch: NonZeroUsize, max_items_per_batch: NonZeroUsize) -> Self {
        Self {
            max_tokens_per_batch,
            max_items_per_batch,
            current_tokens: 0,
            current_items: 0,
        }
    }
}

impl BatchingStrategy for TokenAwareBatchingStrategy {
    fn add(&mut self, token_count: usize) -> AddDecision {
        if self.current_items > 0
            && (self.current_items >= self.max_items_per_batch.get()
                || self.current_tokens.saturating_add(token_count)
                    > self.max_tokens_per_batch.get())
        {
            self.current_items = 1;
            self.current_tokens = token_count;
            return AddDecision::Flush;
        }

        self.current_items += 1;
        self.current_tokens = self.current_tokens.saturating_add(token_count);
        AddDecision::Continue
    }

    fn flush(&mut self) {
        self.current_items = 0;
        self.current_tokens = 0;
    }

    fn max_items_per_batch(&self) -> usize {
        self.max_items_per_batch.get()
    }

    fn max_tokens_per_batch(&self) -> usize {
        self.max_tokens_per_batch.get()
    }
}

/// Batcher that yields bounded batches from sequential `BatchItem` input.
pub struct TokenAwareBatcher<M> {
    strategy: Box<dyn BatchingStrategy>,
    current: Vec<BatchItem<M>>,
}

impl<M> TokenAwareBatcher<M> {
    /// Creates a batcher with non-zero hard limits.
    #[must_use]
    pub fn new(max_tokens_per_batch: NonZeroUsize, max_items_per_batch: NonZeroUsize) -> Self {
        Self::with_strategy(TokenAwareBatchingStrategy::new(
            max_tokens_per_batch,
            max_items_per_batch,
        ))
    }

    /// Creates a batcher from a custom batching strategy.
    #[must_use]
    pub fn with_strategy(strategy: impl BatchingStrategy + 'static) -> Self {
        Self {
            strategy: Box::new(strategy),
            current: Vec::new(),
        }
    }

    /// Adds one item and optionally returns a flushed batch.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BatchItemTooLarge`] if the item itself exceeds the
    /// configured hard token limit.
    pub fn add(&mut self, item: BatchItem<M>) -> Result<Option<Vec<BatchItem<M>>>> {
        let max_tokens_per_batch = self.strategy.max_tokens_per_batch();
        if item.token_count > max_tokens_per_batch {
            return Err(Error::BatchItemTooLarge {
                token_count: item.token_count,
                max_tokens_per_batch,
            });
        }

        match self.strategy.add(item.token_count) {
            AddDecision::Continue => {
                self.current.push(item);
                Ok(None)
            }
            AddDecision::Flush => {
                let batch = std::mem::take(&mut self.current);
                self.current.push(item);
                Ok(Some(batch))
            }
        }
    }

    /// Flushes the current batch, if any.
    #[must_use]
    pub fn flush(&mut self) -> Option<Vec<BatchItem<M>>> {
        if self.current.is_empty() {
            return None;
        }

        self.strategy.flush();
        Some(std::mem::take(&mut self.current))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::EmbeddingRole;

    fn item<M>(meta: M, text: &str, token_count: usize) -> BatchItem<M> {
        BatchItem {
            meta,
            role: EmbeddingRole::Document,
            text: text.to_string(),
            title: None,
            token_count,
        }
    }

    #[test]
    fn token_aware_batcher_splits_on_token_limit() {
        let mut batcher = TokenAwareBatcher::new(
            NonZeroUsize::new(10).unwrap(),
            NonZeroUsize::new(10).unwrap(),
        );

        assert!(batcher.add(item(1, "a", 6)).unwrap().is_none());

        let batch = batcher.add(item(2, "b", 5)).unwrap().unwrap();

        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].meta, 1);

        let final_batch = batcher.flush().unwrap();
        assert_eq!(final_batch.len(), 1);
        assert_eq!(final_batch[0].meta, 2);
    }

    #[test]
    fn token_aware_batcher_splits_on_item_limit() {
        let mut batcher = TokenAwareBatcher::new(
            NonZeroUsize::new(1_000_000).unwrap(),
            NonZeroUsize::new(2).unwrap(),
        );

        assert!(batcher.add(item(1, "a", 1)).unwrap().is_none());
        assert!(batcher.add(item(2, "b", 1)).unwrap().is_none());

        let batch = batcher.add(item(3, "c", 1)).unwrap().unwrap();

        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0].meta, 1);
        assert_eq!(batch[1].meta, 2);

        let final_batch = batcher.flush().unwrap();
        assert_eq!(final_batch.len(), 1);
        assert_eq!(final_batch[0].meta, 3);
    }

    #[test]
    fn batching_strategy_is_object_safe_for_a_fixed_meta_type() {
        let mut batcher = TokenAwareBatcher::with_strategy(TokenAwareBatchingStrategy::new(
            NonZeroUsize::new(10).unwrap(),
            NonZeroUsize::new(2).unwrap(),
        ));

        assert!(batcher.add(item("a", "a", 5)).unwrap().is_none());

        let batch = batcher.add(item("b", "b", 6)).unwrap().unwrap();

        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].meta, "a");
    }

    #[test]
    fn token_aware_batcher_rejects_oversized_single_item() {
        let mut batcher = TokenAwareBatcher::<()>::new(
            NonZeroUsize::new(10).unwrap(),
            NonZeroUsize::new(2).unwrap(),
        );

        let err = batcher.add(item((), "too-big", 11)).unwrap_err();

        assert!(matches!(
            err,
            Error::BatchItemTooLarge {
                token_count: 11,
                max_tokens_per_batch: 10,
            }
        ));
    }
}
