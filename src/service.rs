use std::collections::VecDeque;
use std::sync::Arc;

use futures::StreamExt;
use futures::stream::FuturesUnordered;
use tokio::sync::mpsc;

use crate::BatchItem;
use crate::EmbeddingProvider;
use crate::batching::TokenAwareBatcher;
use crate::{Error, Result};

/// Embedding output aligned with the source metadata items.
pub struct EmbeddingResult<M> {
    /// Metadata values from the submitted `BatchItem` entries.
    pub items: Vec<M>,
    /// Embedding vectors in the same order as `items`.
    pub embeddings: Vec<Vec<f32>>,
}

async fn embed_batch<M>(
    client: &dyn EmbeddingProvider,
    batch: Vec<BatchItem<M>>,
) -> Result<EmbeddingResult<M>> {
    if batch.is_empty() {
        return Ok(EmbeddingResult {
            items: Vec::new(),
            embeddings: Vec::new(),
        });
    }

    let mut inputs = Vec::with_capacity(batch.len());
    let mut items = Vec::with_capacity(batch.len());
    for item in batch {
        inputs.push(item.input);
        items.push(item.meta);
    }

    let output = client.embed(&inputs).await?;
    if output.embeddings.len() != items.len() {
        return Err(Error::EmbeddingCountMismatch {
            embeddings: output.embeddings.len(),
            inputs: items.len(),
        });
    }

    Ok(EmbeddingResult {
        items,
        embeddings: output.embeddings,
    })
}

/// Background batching service for embedding requests.
///
/// Dropping the returned result receiver cancels background processing.
pub struct EmbedderService<M> {
    batcher: TokenAwareBatcher<M>,
    batch_tx: Option<mpsc::Sender<Vec<BatchItem<M>>>>,
}

impl<M: Send + 'static> EmbedderService<M> {
    /// Creates a new embedding service and result stream.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConfiguration`] if `workers` is zero or batching
    /// limits are invalid.
    pub fn new(
        embedder: Arc<dyn EmbeddingProvider>,
        max_tokens: usize,
        max_batch_size: usize,
        workers: usize,
    ) -> Result<(Self, mpsc::Receiver<Result<EmbeddingResult<M>>>)> {
        if workers == 0 {
            return Err(Error::InvalidConfiguration {
                message: "workers must be greater than zero".to_string(),
            });
        }

        let worker_count = workers;
        let batcher = TokenAwareBatcher::new(max_tokens, max_batch_size)?;
        let (batch_tx, mut prepared_rx) = mpsc::channel::<Vec<BatchItem<M>>>(worker_count * 2);
        let (execution_tx, mut execution_rx) = mpsc::channel::<Vec<BatchItem<M>>>(worker_count * 2);
        let (result_tx, result_rx) = mpsc::channel::<Result<EmbeddingResult<M>>>(worker_count * 2);

        // Ingress actor: forwards prepared batches into the execution queue.
        tokio::spawn(async move {
            while let Some(batch) = prepared_rx.recv().await {
                if execution_tx.send(batch).await.is_err() {
                    break;
                }
            }
        });

        // Execution actor: schedules bounded concurrent embedding work and emits results.
        tokio::spawn(async move {
            let mut in_flight: FuturesUnordered<_> = FuturesUnordered::new();
            let mut pending: VecDeque<Vec<BatchItem<M>>> = VecDeque::new();
            let mut execution_closed = false;
            let pending_limit = worker_count * 2;

            loop {
                while in_flight.len() < worker_count {
                    let Some(batch) = pending.pop_front() else {
                        break;
                    };
                    let embedder = Arc::clone(&embedder);
                    in_flight.push(async move { embed_batch(embedder.as_ref(), batch).await });
                }

                if execution_closed && pending.is_empty() && in_flight.is_empty() {
                    break;
                }

                tokio::select! {
                    biased;
                    Some(result) = in_flight.next(), if !in_flight.is_empty() => {
                        if result_tx.send(result).await.is_err() {
                            // Receiver dropped: stop background work to avoid silently
                            // discarding additional embeddings.
                            break;
                        }
                    }
                    batch = execution_rx.recv(), if !execution_closed && pending.len() < pending_limit => {
                        match batch {
                            Some(batch) => pending.push_back(batch),
                            None => execution_closed = true,
                        }
                    }
                    else => {
                        // No immediate action available; loop to reschedule work and re-evaluate exit.
                    }
                }
            }
        });

        Ok((
            Self {
                batcher,
                batch_tx: Some(batch_tx),
            },
            result_rx,
        ))
    }

    /// Adds one item into the service and flushes a batch if needed.
    pub async fn enqueue(&mut self, item: BatchItem<M>) -> Result<bool> {
        if let Some(batch) = self.batcher.add(item)
            && let Some(tx) = self.batch_tx.as_ref()
        {
            tx.send(batch)
                .await
                .map_err(|_| Error::BatchChannelClosed)?;
            return Ok(true);
        }
        Ok(false)
    }

    /// Flushes pending batched items and closes the batch input channel.
    pub async fn flush(&mut self) -> Result<bool> {
        if self.batch_tx.is_none() {
            return Ok(false);
        }

        let mut sent = false;
        if let Some(batch) = self.batcher.flush()
            && let Some(tx) = self.batch_tx.as_ref()
        {
            tx.send(batch)
                .await
                .map_err(|_| Error::BatchChannelClosed)?;
            sent = true;
        }
        drop(self.batch_tx.take());
        Ok(sent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;
    use std::time::Duration;

    use async_trait::async_trait;
    use tokio::sync::Notify;
    use tokio::time::timeout;

    use crate::{EmbedOutput, PreparedEmbeddingInput};

    struct NoopProvider;

    #[async_trait]
    impl EmbeddingProvider for NoopProvider {
        async fn embed(&self, input: &[PreparedEmbeddingInput]) -> Result<EmbedOutput> {
            Ok(EmbedOutput {
                embeddings: input.iter().map(|_| vec![0.0; 2]).collect(),
            })
        }
    }

    struct BlockingProvider {
        gate: Arc<Notify>,
    }

    #[async_trait]
    impl EmbeddingProvider for BlockingProvider {
        async fn embed(&self, input: &[PreparedEmbeddingInput]) -> Result<EmbedOutput> {
            self.gate.notified().await;
            Ok(EmbedOutput {
                embeddings: input.iter().map(|_| vec![0.0; 2]).collect(),
            })
        }
    }

    fn prepared(count: usize) -> PreparedEmbeddingInput {
        PreparedEmbeddingInput::new(vec![1; count]).unwrap()
    }

    #[test]
    fn service_rejects_zero_workers() {
        let result = EmbedderService::<()>::new(Arc::new(NoopProvider), 64, 8, 0);
        assert!(matches!(result, Err(Error::InvalidConfiguration { .. })));
    }

    #[test]
    fn service_rejects_zero_batch_limits() {
        let result = EmbedderService::<()>::new(Arc::new(NoopProvider), 0, 8, 1);
        assert!(matches!(result, Err(Error::InvalidConfiguration { .. })));
    }

    #[tokio::test]
    async fn intake_can_progress_when_one_batch_is_in_flight() {
        let gate = Arc::new(Notify::new());
        let provider = Arc::new(BlockingProvider { gate: gate.clone() });
        let (mut service, _rx) = EmbedderService::new(provider, 10_000, 1, 1).unwrap();

        let first = service
            .enqueue(BatchItem {
                meta: 1usize,
                input: prepared(1),
            })
            .await
            .unwrap();
        assert!(!first);

        for n in 2usize..=6usize {
            let flushed = timeout(
                Duration::from_millis(75),
                service.enqueue(BatchItem {
                    meta: n,
                    input: prepared(1),
                }),
            )
            .await
            .expect("enqueue should remain non-blocking while queue capacity remains")
            .unwrap();
            assert!(flushed);
        }

        let mut observed_backpressure = false;
        for n in 7usize..=20usize {
            let pushed = timeout(
                Duration::from_millis(75),
                service.enqueue(BatchItem {
                    meta: n,
                    input: prepared(1),
                }),
            )
            .await;

            if pushed.is_err() {
                observed_backpressure = true;
                break;
            }
        }

        assert!(
            observed_backpressure,
            "enqueue should eventually backpressure when execution is stalled"
        );

        gate.notify_waiters();
    }
}
