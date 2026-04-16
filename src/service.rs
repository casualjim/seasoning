use std::collections::VecDeque;
use std::num::NonZeroUsize;
use std::sync::Arc;

use futures::StreamExt;
use futures::stream::FuturesUnordered;
use tokio::sync::mpsc;

use crate::BatchItem;
use crate::EmbeddingProvider;
use crate::batching::TokenAwareBatcher;
use crate::embedding::EmbeddingInput;
use crate::{Error, Result};

/// Embedding output aligned with the source metadata items.
#[derive(Debug)]
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
        inputs.push(EmbeddingInput {
            role: item.role,
            text: item.text,
            title: item.title,
            token_count: item.token_count,
        });
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

/// Background batching service for semantic embedding requests.
///
/// [`BatchItem`] values preserve their embedding role and optional title
/// metadata. Dropping the returned result receiver cancels background
/// processing.
pub struct EmbedderService<M> {
    batcher: TokenAwareBatcher<M>,
    batch_tx: Option<mpsc::Sender<Vec<BatchItem<M>>>>,
}

impl<M: Send + 'static> EmbedderService<M> {
    /// Creates a new embedding service and result stream.
    ///
    /// # Errors
    ///
    /// Returns an error if no Tokio runtime is active.
    ///
    /// This constructor spawns background tasks immediately.
    pub fn new(
        embedder: Arc<dyn EmbeddingProvider>,
        max_tokens: NonZeroUsize,
        max_batch_size: NonZeroUsize,
        workers: NonZeroUsize,
    ) -> Result<(Self, mpsc::Receiver<Result<EmbeddingResult<M>>>)> {
        let worker_count = workers.get();
        let batcher = TokenAwareBatcher::new(max_tokens, max_batch_size);
        let handle =
            tokio::runtime::Handle::try_current().map_err(|_| Error::TokioRuntimeRequired)?;
        let (batch_tx, mut prepared_rx) = mpsc::channel::<Vec<BatchItem<M>>>(worker_count * 2);
        let (execution_tx, mut execution_rx) = mpsc::channel::<Vec<BatchItem<M>>>(worker_count * 2);
        let (result_tx, result_rx) = mpsc::channel::<Result<EmbeddingResult<M>>>(worker_count * 2);

        handle.spawn(async move {
            while let Some(batch) = prepared_rx.recv().await {
                if execution_tx.send(batch).await.is_err() {
                    break;
                }
            }
        });

        handle.spawn(async move {
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
                            break;
                        }
                    }
                    batch = execution_rx.recv(), if !execution_closed && pending.len() < pending_limit => {
                        match batch {
                            Some(batch) => pending.push_back(batch),
                            None => execution_closed = true,
                        }
                    }
                    else => {}
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
    ///
    /// # Errors
    ///
    /// Returns [`Error::BatchChannelClosed`] after [`Self::flush`] has closed the service.
    /// Returns [`Error::BatchItemTooLarge`] when one item exceeds the configured
    /// hard token limit.
    pub async fn enqueue(&mut self, item: BatchItem<M>) -> Result<bool> {
        if self.batch_tx.is_none() {
            return Err(Error::BatchChannelClosed);
        }

        if let Some(batch) = self.batcher.add(item)? {
            let tx = self.batch_tx.as_ref().ok_or(Error::BatchChannelClosed)?;
            tx.send(batch)
                .await
                .map_err(|_| Error::BatchChannelClosed)?;
            return Ok(true);
        }

        Ok(false)
    }

    /// Flushes pending batched items and closes the batch input channel.
    ///
    /// After this returns successfully, subsequent [`Self::enqueue`] calls return
    /// [`Error::BatchChannelClosed`].
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

    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use async_trait::async_trait;
    use tokio::sync::Notify;
    use tokio::time::timeout;

    use crate::{EmbedOutput, EmbeddingInput, EmbeddingRole};

    fn non_zero(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).unwrap()
    }

    fn item<M>(
        meta: M,
        role: EmbeddingRole,
        text: &str,
        title: Option<&str>,
        token_count: usize,
    ) -> BatchItem<M> {
        BatchItem {
            meta,
            role,
            text: text.to_string(),
            title: title.map(str::to_string),
            token_count,
        }
    }

    struct NoopProvider;

    #[async_trait]
    impl EmbeddingProvider for NoopProvider {
        async fn embed(&self, input: &[EmbeddingInput]) -> Result<EmbedOutput> {
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
        async fn embed(&self, input: &[EmbeddingInput]) -> Result<EmbedOutput> {
            self.gate.notified().await;
            Ok(EmbedOutput {
                embeddings: input.iter().map(|_| vec![0.0; 2]).collect(),
            })
        }
    }

    struct RecordingProvider {
        seen: Arc<Mutex<Vec<EmbeddingInput>>>,
    }

    #[async_trait]
    impl EmbeddingProvider for RecordingProvider {
        async fn embed(&self, input: &[EmbeddingInput]) -> Result<EmbedOutput> {
            self.seen.lock().unwrap().extend_from_slice(input);
            Ok(EmbedOutput {
                embeddings: input.iter().map(|_| vec![0.0; 2]).collect(),
            })
        }
    }

    #[test]
    fn service_requires_tokio_runtime() {
        let result = EmbedderService::<()>::new(
            Arc::new(NoopProvider),
            non_zero(64),
            non_zero(8),
            non_zero(1),
        );
        assert!(matches!(result, Err(Error::TokioRuntimeRequired)));
    }

    #[tokio::test]
    async fn service_preserves_batch_item_semantics() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let provider = Arc::new(RecordingProvider { seen: seen.clone() });
        let (mut service, mut rx) =
            EmbedderService::new(provider, non_zero(64), non_zero(8), non_zero(1)).unwrap();

        assert!(
            !service
                .enqueue(item((), EmbeddingRole::Query, "q", Some("query title"), 1,))
                .await
                .unwrap()
        );
        assert!(service.flush().await.unwrap());

        let result = rx.recv().await.unwrap().unwrap();
        assert_eq!(result.items.len(), 1);

        let captured = seen.lock().unwrap();
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].role, EmbeddingRole::Query);
        assert_eq!(captured[0].text, "q");
        assert_eq!(captured[0].title.as_deref(), Some("query title"));
        assert_eq!(captured[0].token_count, 1);
    }

    #[tokio::test]
    async fn enqueue_rejects_items_after_flush() {
        let (mut service, _rx) = EmbedderService::new(
            Arc::new(NoopProvider),
            non_zero(64),
            non_zero(8),
            non_zero(1),
        )
        .unwrap();

        assert!(
            !service
                .enqueue(item((), EmbeddingRole::Document, "a", None, 1))
                .await
                .unwrap()
        );

        assert!(service.flush().await.unwrap());
        assert!(matches!(
            service
                .enqueue(item((), EmbeddingRole::Document, "b", None, 1))
                .await,
            Err(Error::BatchChannelClosed)
        ));
    }

    #[tokio::test]
    async fn enqueue_rejects_oversized_items() {
        let (mut service, _rx) = EmbedderService::new(
            Arc::new(NoopProvider),
            non_zero(8),
            non_zero(8),
            non_zero(1),
        )
        .unwrap();

        let err = service
            .enqueue(item((), EmbeddingRole::Document, "too-big", None, 9))
            .await
            .unwrap_err();

        assert!(matches!(
            err,
            Error::BatchItemTooLarge {
                token_count: 9,
                max_tokens_per_batch: 8,
            }
        ));
    }

    #[tokio::test]
    async fn intake_can_progress_when_one_batch_is_in_flight() {
        let gate = Arc::new(Notify::new());
        let provider = Arc::new(BlockingProvider { gate: gate.clone() });
        let (mut service, _rx) =
            EmbedderService::new(provider, non_zero(10_000), non_zero(1), non_zero(1)).unwrap();

        let first = service
            .enqueue(item(1usize, EmbeddingRole::Document, "a", None, 1))
            .await
            .unwrap();
        assert!(!first);

        for n in 2usize..=6usize {
            let flushed = timeout(
                Duration::from_millis(75),
                service.enqueue(item(
                    n,
                    EmbeddingRole::Document,
                    &format!("item-{n}"),
                    None,
                    1,
                )),
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
                service.enqueue(item(
                    n,
                    EmbeddingRole::Document,
                    &format!("item-{n}"),
                    None,
                    1,
                )),
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
