use crate::api::{AddDecision, BatchItem, BatchingStrategy};
use crate::{Error, Result};

/// Token-aware batching strategy with hard item and token limits.
pub struct TokenAwareBatchingStrategy {
    max_tokens_per_batch: usize,
    max_items_per_batch: usize,
    current_tokens: usize,
    current_items: usize,
}

impl TokenAwareBatchingStrategy {
    /// Creates a strategy for the provided limits.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConfiguration`] when either limit is zero.
    pub fn new(max_tokens_per_batch: usize, max_items_per_batch: usize) -> Result<Self> {
        if max_tokens_per_batch == 0 {
            return Err(Error::InvalidConfiguration {
                message: "max_tokens_per_batch must be greater than zero".to_string(),
            });
        }
        if max_items_per_batch == 0 {
            return Err(Error::InvalidConfiguration {
                message: "max_items_per_batch must be greater than zero".to_string(),
            });
        }

        Ok(Self {
            max_tokens_per_batch,
            max_items_per_batch,
            current_tokens: 0,
            current_items: 0,
        })
    }
}

impl BatchingStrategy for TokenAwareBatchingStrategy {
    fn add(&mut self, token_count: usize) -> AddDecision {
        if self.current_items > 0
            && (self.current_items >= self.max_items_per_batch
                || self.current_tokens.saturating_add(token_count) > self.max_tokens_per_batch)
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
        self.max_items_per_batch
    }

    fn max_tokens_per_batch(&self) -> usize {
        self.max_tokens_per_batch
    }
}

/// Batcher that yields bounded batches from sequential `BatchItem` input.
pub struct TokenAwareBatcher<M> {
    strategy: Box<dyn BatchingStrategy>,
    current: Vec<BatchItem<M>>,
}

impl<M> TokenAwareBatcher<M> {
    /// Creates a batcher with token and item limits.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConfiguration`] when either limit is zero.
    pub fn new(max_tokens_per_batch: usize, max_items_per_batch: usize) -> Result<Self> {
        Ok(Self::with_strategy(TokenAwareBatchingStrategy::new(
            max_tokens_per_batch,
            max_items_per_batch,
        )?))
    }

    /// Creates a batcher from a custom batching strategy.
    pub fn with_strategy(strategy: impl BatchingStrategy + 'static) -> Self {
        Self {
            strategy: Box::new(strategy),
            current: Vec::new(),
        }
    }

    /// Adds one item to the current batch, returning a flushed batch if a limit was hit.
    pub fn add(&mut self, item: BatchItem<M>) -> Option<Vec<BatchItem<M>>> {
        match self.strategy.add(item.input.token_count()) {
            AddDecision::Continue => {
                self.current.push(item);
                None
            }
            AddDecision::Flush => {
                let batch = std::mem::take(&mut self.current);
                self.current.push(item);
                Some(batch)
            }
        }
    }

    /// Flushes the current batch if it has pending items.
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
    use crate::PreparedEmbeddingInput;

    fn prepared(count: usize) -> PreparedEmbeddingInput {
        PreparedEmbeddingInput::new(vec![1; count]).unwrap()
    }

    #[test]
    fn token_aware_batcher_splits_on_token_limit() {
        let mut batcher = TokenAwareBatcher::new(10, 10).unwrap();

        assert!(
            batcher
                .add(BatchItem {
                    meta: 1,
                    input: prepared(6),
                })
                .is_none()
        );

        let batch = batcher
            .add(BatchItem {
                meta: 2,
                input: prepared(5),
            })
            .unwrap();

        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].meta, 1);

        let final_batch = batcher.flush().unwrap();
        assert_eq!(final_batch.len(), 1);
        assert_eq!(final_batch[0].meta, 2);
    }

    #[test]
    fn token_aware_batcher_splits_on_item_limit() {
        let mut batcher = TokenAwareBatcher::new(1_000_000, 2).unwrap();

        assert!(
            batcher
                .add(BatchItem {
                    meta: 1,
                    input: prepared(1),
                })
                .is_none()
        );
        assert!(
            batcher
                .add(BatchItem {
                    meta: 2,
                    input: prepared(1),
                })
                .is_none()
        );

        let batch = batcher
            .add(BatchItem {
                meta: 3,
                input: prepared(1),
            })
            .unwrap();

        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0].meta, 1);
        assert_eq!(batch[1].meta, 2);

        let final_batch = batcher.flush().unwrap();
        assert_eq!(final_batch.len(), 1);
        assert_eq!(final_batch[0].meta, 3);
    }

    #[test]
    fn batching_strategy_is_object_safe_for_a_fixed_meta_type() {
        let mut batcher =
            TokenAwareBatcher::with_strategy(TokenAwareBatchingStrategy::new(10, 2).unwrap());

        assert!(
            batcher
                .add(BatchItem {
                    meta: "a",
                    input: prepared(5),
                })
                .is_none()
        );

        let batch = batcher
            .add(BatchItem {
                meta: "b",
                input: prepared(6),
            })
            .unwrap();

        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].meta, "a");
    }

    #[test]
    fn rejects_zero_batch_limits() {
        assert!(matches!(
            TokenAwareBatchingStrategy::new(0, 2),
            Err(Error::InvalidConfiguration { .. })
        ));
        assert!(matches!(
            TokenAwareBatchingStrategy::new(2, 0),
            Err(Error::InvalidConfiguration { .. })
        ));
        assert!(matches!(
            TokenAwareBatcher::<()>::new(0, 2),
            Err(Error::InvalidConfiguration { .. })
        ));
    }
}
