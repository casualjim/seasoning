use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::mpsc as thread_mpsc;
use std::thread;

use hf_hub::api::sync::Api;
use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use tokio::sync::oneshot;

use crate::{EmbedOutput, Error, ModelFamily, Result};

const GEMMA_EMBEDDING_MODEL: &str =
    "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf";
const QWEN3_EMBEDDING_MODEL: &str =
    "hf:Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf";
const QWEN3_RERANKER_MODEL: &str =
    "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf";

static LLAMA_BACKEND: OnceLock<std::result::Result<LlamaBackend, String>> = OnceLock::new();

#[derive(Clone)]
pub(crate) struct LocalEmbeddingClient {
    sender: thread_mpsc::Sender<EmbeddingCommand>,
}

#[derive(Clone)]
pub(crate) struct LocalRerankerClient {
    sender: thread_mpsc::Sender<RerankerCommand>,
}

enum EmbeddingCommand {
    Embed {
        texts: Vec<String>,
        response: oneshot::Sender<Result<EmbedOutput>>,
    },
}

enum RerankerCommand {
    Score {
        texts: Vec<String>,
        response: oneshot::Sender<Result<Vec<f64>>>,
    },
}

struct LocalEmbeddingRuntime {
    model: LlamaModel,
}

struct LocalRerankerRuntime {
    model: LlamaModel,
}

impl LocalEmbeddingClient {
    pub(crate) fn new(model_family: ModelFamily, model: &str) -> Result<Self> {
        validate_local_embedding_model(model_family, model)?;
        let model_path = resolve_model_path(model)?;
        let (sender, receiver) = thread_mpsc::channel();
        let (init_tx, init_rx) = thread_mpsc::sync_channel(1);
        let thread_name = format!("seasoning-embed-{}", model_family.as_str());

        thread::Builder::new()
            .name(thread_name)
            .spawn(move || match LocalEmbeddingRuntime::new(model_path) {
                Ok(mut runtime) => {
                    let _ = init_tx.send(Ok(()));
                    runtime.run(receiver);
                }
                Err(err) => {
                    let _ = init_tx.send(Err(err));
                }
            })
            .map_err(|err| Error::LocalRuntime {
                message: format!("failed to spawn local embedding worker: {err}"),
            })?;

        match init_rx.recv() {
            Ok(Ok(())) => Ok(Self { sender }),
            Ok(Err(err)) => Err(err),
            Err(_) => Err(Error::LocalRuntimeChannelClosed),
        }
    }

    pub(crate) async fn embed_texts(&self, texts: &[String]) -> Result<EmbedOutput> {
        let (response_tx, response_rx) = oneshot::channel();
        self.sender
            .send(EmbeddingCommand::Embed {
                texts: texts.to_vec(),
                response: response_tx,
            })
            .map_err(|_| Error::LocalRuntimeChannelClosed)?;

        response_rx.await.map_err(|_| Error::LocalRuntimeChannelClosed)?
    }
}

impl LocalRerankerClient {
    pub(crate) fn new(model_family: ModelFamily, model: &str) -> Result<Self> {
        validate_local_reranker_model(model_family, model)?;
        let model_path = resolve_model_path(model)?;
        let (sender, receiver) = thread_mpsc::channel();
        let (init_tx, init_rx) = thread_mpsc::sync_channel(1);
        let thread_name = format!("seasoning-rerank-{}", model_family.as_str());

        thread::Builder::new()
            .name(thread_name)
            .spawn(move || match LocalRerankerRuntime::new(model_path) {
                Ok(mut runtime) => {
                    let _ = init_tx.send(Ok(()));
                    runtime.run(receiver);
                }
                Err(err) => {
                    let _ = init_tx.send(Err(err));
                }
            })
            .map_err(|err| Error::LocalRuntime {
                message: format!("failed to spawn local reranker worker: {err}"),
            })?;

        match init_rx.recv() {
            Ok(Ok(())) => Ok(Self { sender }),
            Ok(Err(err)) => Err(err),
            Err(_) => Err(Error::LocalRuntimeChannelClosed),
        }
    }

    pub(crate) async fn score_texts(&self, texts: &[String]) -> Result<Vec<f64>> {
        let (response_tx, response_rx) = oneshot::channel();
        self.sender
            .send(RerankerCommand::Score {
                texts: texts.to_vec(),
                response: response_tx,
            })
            .map_err(|_| Error::LocalRuntimeChannelClosed)?;

        response_rx.await.map_err(|_| Error::LocalRuntimeChannelClosed)?
    }
}

impl LocalEmbeddingRuntime {
    fn new(model_path: PathBuf) -> Result<Self> {
        let model = load_model(&model_path)?;
        Ok(Self { model })
    }

    fn run(&mut self, receiver: thread_mpsc::Receiver<EmbeddingCommand>) {
        for command in receiver {
            match command {
                EmbeddingCommand::Embed { texts, response } => {
                    let _ = response.send(self.embed_texts(&texts));
                }
            }
        }
    }

    fn embed_texts(&mut self, texts: &[String]) -> Result<EmbedOutput> {
        let mut embeddings = Vec::with_capacity(texts.len());
        for text in texts {
            embeddings.push(self.embed_text(text)?);
        }
        Ok(EmbedOutput { embeddings })
    }

    fn embed_text(&mut self, text: &str) -> Result<Vec<f32>> {
        let tokens = tokenize_nonempty(&self.model, text)?;
        let mut context = self
            .model
            .new_context(llama_backend()?, LlamaContextParams::default().with_embeddings(true))
            .map_err(|err| Error::LocalRuntime {
                message: format!("failed to create llama.cpp embedding context: {err}"),
            })?;
        let mut batch = LlamaBatch::new(tokens.len(), 1);
        batch.add_sequence(&tokens, 0, false).map_err(|err| Error::LocalRuntime {
            message: format!("failed to prepare llama.cpp embedding batch: {err}"),
        })?;
        context.encode(&mut batch).map_err(|err| Error::LocalRuntime {
            message: format!("llama.cpp embedding encode failed: {err}"),
        })?;
        let embedding = context.embeddings_seq_ith(0).map_err(|err| Error::LocalRuntime {
            message: format!("failed to read llama.cpp embedding output: {err}"),
        })?;
        Ok(embedding.to_vec())
    }
}

impl LocalRerankerRuntime {
    fn new(model_path: PathBuf) -> Result<Self> {
        let model = load_model(&model_path)?;
        Ok(Self { model })
    }

    fn run(&mut self, receiver: thread_mpsc::Receiver<RerankerCommand>) {
        for command in receiver {
            match command {
                RerankerCommand::Score { texts, response } => {
                    let _ = response.send(self.score_texts(&texts));
                }
            }
        }
    }

    fn score_texts(&mut self, texts: &[String]) -> Result<Vec<f64>> {
        let mut scores = Vec::with_capacity(texts.len());
        for text in texts {
            scores.push(self.score_text(text)?);
        }
        Ok(scores)
    }

    fn score_text(&mut self, text: &str) -> Result<f64> {
        let tokens = tokenize_nonempty(&self.model, text)?;
        let params = LlamaContextParams::default()
            .with_embeddings(true)
            .with_pooling_type(LlamaPoolingType::Rank);
        let mut context = self
            .model
            .new_context(llama_backend()?, params)
            .map_err(|err| Error::LocalRuntime {
                message: format!("failed to create llama.cpp reranker context: {err}"),
            })?;
        let mut batch = LlamaBatch::new(tokens.len(), 1);
        batch.add_sequence(&tokens, 0, false).map_err(|err| Error::LocalRuntime {
            message: format!("failed to prepare llama.cpp reranker batch: {err}"),
        })?;
        context.encode(&mut batch).map_err(|err| Error::LocalRuntime {
            message: format!("llama.cpp reranker encode failed: {err}"),
        })?;
        let score = context.embeddings_seq_ith(0).map_err(|err| Error::LocalRuntime {
            message: format!("failed to read llama.cpp reranker score: {err}"),
        })?;

        score
            .first()
            .copied()
            .map(f64::from)
            .ok_or_else(|| Error::LocalRuntime {
                message: "llama.cpp reranker returned no score".to_string(),
            })
    }
}

fn validate_local_embedding_model(model_family: ModelFamily, model: &str) -> Result<()> {
    let supported = match model_family {
        ModelFamily::Gemma => GEMMA_EMBEDDING_MODEL,
        ModelFamily::Qwen3 => QWEN3_EMBEDDING_MODEL,
    };

    if model == supported {
        Ok(())
    } else {
        Err(Error::UnsupportedLocalModel {
            kind: "embedding",
            model: model.to_string(),
        })
    }
}

fn validate_local_reranker_model(model_family: ModelFamily, model: &str) -> Result<()> {
    if model_family != ModelFamily::Qwen3 || model != QWEN3_RERANKER_MODEL {
        return Err(Error::UnsupportedLocalModel {
            kind: "reranking",
            model: model.to_string(),
        });
    }

    Ok(())
}

fn resolve_model_path(model: &str) -> Result<PathBuf> {
    let Some(spec) = model.strip_prefix("hf:") else {
        return Err(Error::UnsupportedConfiguration {
            message: format!(
                "local llama.cpp models must be configured as Hugging Face GGUF artifacts (expected hf:<repo>/<file>.gguf, got '{model}')"
            ),
        });
    };

    let Some((repo, filename)) = spec.rsplit_once('/') else {
        return Err(Error::UnsupportedConfiguration {
            message: format!(
                "local Hugging Face GGUF model '{model}' must include both the repository id and file name"
            ),
        });
    };

    Api::new()
        .map_err(|err| Error::LocalRuntime {
            message: format!("failed to initialize hf-hub client: {err}"),
        })?
        .model(repo.to_string())
        .get(filename)
        .map_err(|err| Error::LocalRuntime {
            message: format!("failed to resolve Hugging Face GGUF artifact '{model}': {err}"),
        })
}

fn load_model(model_path: &PathBuf) -> Result<LlamaModel> {
    LlamaModel::load_from_file(llama_backend()?, model_path, &LlamaModelParams::default())
        .map_err(|err| Error::LocalRuntime {
            message: format!("failed to load llama.cpp model from '{}': {err}", model_path.display()),
        })
}

fn tokenize_nonempty(model: &LlamaModel, text: &str) -> Result<Vec<llama_cpp_2::token::LlamaToken>> {
    let tokens = model
        .str_to_token(text, AddBos::Always)
        .map_err(|err| Error::LocalRuntime {
            message: format!("failed to tokenize local llama.cpp input: {err}"),
        })?;

    if tokens.is_empty() {
        return Err(Error::LocalRuntime {
            message: "local llama.cpp input tokenized to an empty sequence".to_string(),
        });
    }

    Ok(tokens)
}

fn llama_backend() -> Result<&'static LlamaBackend> {
    match LLAMA_BACKEND.get_or_init(|| {
        let mut backend = LlamaBackend::init()
            .map_err(|err| format!("failed to initialize llama.cpp backend: {err}"))?;
        backend.void_logs();
        Ok(backend)
    }) {
        Ok(backend) => Ok(backend),
        Err(message) => Err(Error::LocalRuntime {
            message: message.clone(),
        }),
    }
}
