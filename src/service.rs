use std::sync::Arc;

use futures::StreamExt;
use futures::stream::FuturesUnordered;
use tokio::sync::{Semaphore, mpsc};

use crate::BatchItem;
use crate::EmbeddingProvider;
use crate::batching::TokenAwareBatcher;
use crate::embedding::EmbeddingInput;
use crate::{Error, Result};

pub struct EmbeddingResult<M> {
    pub items: Vec<M>,
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
            text: item.text,
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

pub struct EmbedderService<M> {
    batcher: TokenAwareBatcher<M>,
    batch_tx: Option<mpsc::Sender<Vec<BatchItem<M>>>>,
}

impl<M: Send + 'static> EmbedderService<M> {
    pub fn new(
        embedder: Arc<dyn EmbeddingProvider>,
        max_tokens: usize,
        max_batch_size: usize,
        workers: usize,
    ) -> (Self, mpsc::Receiver<Result<EmbeddingResult<M>>>) {
        let worker_count = workers.max(1);
        let (batch_tx, mut batch_rx) = mpsc::channel::<Vec<BatchItem<M>>>(worker_count * 2);
        let (result_tx, result_rx) = mpsc::channel::<Result<EmbeddingResult<M>>>(worker_count * 2);

        let semaphore = Arc::new(Semaphore::new(worker_count));
        tokio::spawn(async move {
            let mut in_flight: FuturesUnordered<_> = FuturesUnordered::new();
            let mut rx_closed = false;

            loop {
                tokio::select! {
                    batch = batch_rx.recv(), if !rx_closed && in_flight.len() < worker_count => {
                        match batch {
                            Some(batch) => {
                                let embedder = Arc::clone(&embedder);
                                let semaphore = Arc::clone(&semaphore);
                                in_flight.push(async move {
                                    let _permit = semaphore
                                        .acquire_owned()
                                        .await
                                        .map_err(|_| Error::SemaphoreClosed)?;
                                    embed_batch(embedder.as_ref(), batch).await
                                });
                            }
                            None => {
                                rx_closed = true;
                            }
                        }
                    }
                    Some(result) = in_flight.next(), if !in_flight.is_empty() => {
                        let _ = result_tx.send(result).await;
                    }
                    else => {
                        if rx_closed {
                            break;
                        }
                    }
                }
            }
        });

        (
            Self {
                batcher: TokenAwareBatcher::new(max_tokens, max_batch_size),
                batch_tx: Some(batch_tx),
            },
            result_rx,
        )
    }

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
