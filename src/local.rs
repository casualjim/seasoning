use std::env::VarError;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc as thread_mpsc;
use std::thread;
use std::time::Duration;

/// Flash attention enabled policy constant.
/// Corresponds to `llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_ENABLED`.
const FLASH_ATTN_ENABLED: std::ffi::c_int = 1;

use hf_hub::api::sync::{Api, ApiBuilder};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::send_logs_to_tracing;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::{
    LlamaBackendDevice, LlamaBackendDeviceType, LogOptions, list_llama_ggml_backend_devices,
};
use tokio::sync::oneshot;

use crate::{EmbedOutput, Error, ModelFamily, Result};

/// Example local GGUF model id for Gemma embeddings.
pub const GEMMA_EMBEDDING_MODEL: &str =
    "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf";
/// Example local GGUF model id for Qwen3 embeddings.
pub const QWEN3_EMBEDDING_MODEL: &str =
    "hf:Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf";
/// Example local GGUF model id for Qwen3 reranking.
pub const QWEN3_RERANKER_MODEL: &str =
    "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf";
const SEASONING_LOCAL_DEBUG_ENV: &str = "SEASONING_LOCAL_DEBUG";
const SEASONING_HF_HUB_PROGRESS_ENV: &str = "SEASONING_HF_HUB_PROGRESS";
const HF_HUB_DISABLE_PROGRESS_BARS_ENV: &str = "HF_HUB_DISABLE_PROGRESS_BARS";

/// Initialize the llama.cpp backend.
///
/// Convenience method for callers that don't need custom backend lifecycle management.
/// The returned backend can be shared across multiple local embed/rerank clients via `Arc`.
pub fn create_backend() -> Result<LlamaBackend> {
    let debug = local_debug_enabled();
    send_logs_to_tracing(LogOptions::default().with_logs_enabled(debug));
    let backend = LlamaBackend::init().map_err(|err| Error::LocalRuntime {
        message: format!("failed to initialize llama.cpp backend: {err}"),
    })?;
    if debug {
        let devices = list_llama_ggml_backend_devices();
        emit_backend_diagnostics(&devices);
    }
    Ok(backend)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LocalModelSelection {
    selected_device_indices: Vec<usize>,
    gpu_offload_enabled: bool,
}

impl LocalModelSelection {
    fn from_devices(devices: &[LlamaBackendDevice]) -> Self {
        let selected_device_indices = preferred_backend_device_indices(devices);
        let gpu_offload_enabled = !selected_device_indices.is_empty();
        Self {
            selected_device_indices,
            gpu_offload_enabled,
        }
    }

    fn apply(&self, params: LlamaModelParams) -> Result<LlamaModelParams> {
        if self.selected_device_indices.is_empty() {
            return Ok(params);
        }

        params
            .with_devices(&self.selected_device_indices)
            .map(|params| params.with_n_gpu_layers(u32::MAX))
            .map_err(|err| Error::LocalRuntime {
                message: format!(
                    "failed to configure llama.cpp GPU devices {:?}: {err}",
                    self.selected_device_indices
                ),
            })
    }
}

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
        token_batches: Vec<Vec<u32>>,
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
    pub(crate) fn new(
        backend: Arc<LlamaBackend>,
        model_family: ModelFamily,
        model: &str,
    ) -> Result<Self> {
        let spec = parse_hf_model_spec(model)?;
        let (sender, receiver) = thread_mpsc::channel();
        let thread_name = format!("seasoning-embed-{}", model_family.as_str());

        thread::Builder::new()
            .name(thread_name)
            .spawn(move || match resolve_model_path_from_spec(&spec) {
                Ok(path) => match LocalEmbeddingRuntime::new(&backend, path) {
                    Ok(runtime) => runtime.run(&backend, receiver),
                    Err(error) => {
                        let message = format!("{error}");
                        for command in receiver {
                            match command {
                                EmbeddingCommand::Embed { response, .. } => {
                                    let _ = response.send(Err(Error::LocalRuntime {
                                        message: message.clone(),
                                    }));
                                }
                            }
                        }
                    }
                },
                Err(error) => {
                    let message = format!("{error}");
                    for command in receiver {
                        match command {
                            EmbeddingCommand::Embed { response, .. } => {
                                let _ = response.send(Err(Error::LocalRuntime {
                                    message: message.clone(),
                                }));
                            }
                        }
                    }
                }
            })
            .map_err(|err| Error::LocalRuntime {
                message: format!("failed to spawn local embedding worker: {err}"),
            })?;

        Ok(Self { sender })
    }

    pub(crate) async fn embed_prepared(&self, token_batches: Vec<Vec<u32>>) -> Result<EmbedOutput> {
        let batch_count = token_batches.len();
        let (response_tx, response_rx) = oneshot::channel();
        self.sender
            .send(EmbeddingCommand::Embed {
                token_batches,
                response: response_tx,
            })
            .map_err(|_| Error::LocalRuntimeChannelClosed)?;

        match tokio::time::timeout(local_timeout(), response_rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(Error::LocalRuntimeChannelClosed),
            Err(_) => Err(Error::LocalRuntime {
                message: format!(
                    "local embedding timed out after {}s (batch count: {batch_count})",
                    local_timeout().as_secs(),
                ),
            }),
        }
    }
}

impl LocalRerankerClient {
    pub(crate) fn new(
        backend: Arc<LlamaBackend>,
        model_family: ModelFamily,
        model: &str,
    ) -> Result<Self> {
        let spec = parse_hf_model_spec(model)?;
        let (sender, receiver) = thread_mpsc::channel();
        let thread_name = format!("seasoning-rerank-{}", model_family.as_str());

        thread::Builder::new()
            .name(thread_name)
            .spawn(move || match resolve_model_path_from_spec(&spec) {
                Ok(path) => match LocalRerankerRuntime::new(&backend, path) {
                    Ok(runtime) => runtime.run(&backend, receiver),
                    Err(error) => {
                        let message = format!("{error}");
                        for command in receiver {
                            match command {
                                RerankerCommand::Score { response, .. } => {
                                    let _ = response.send(Err(Error::LocalRuntime {
                                        message: message.clone(),
                                    }));
                                }
                            }
                        }
                    }
                },
                Err(error) => {
                    let message = format!("{error}");
                    for command in receiver {
                        match command {
                            RerankerCommand::Score { response, .. } => {
                                let _ = response.send(Err(Error::LocalRuntime {
                                    message: message.clone(),
                                }));
                            }
                        }
                    }
                }
            })
            .map_err(|err| Error::LocalRuntime {
                message: format!("failed to spawn local reranker worker: {err}"),
            })?;

        Ok(Self { sender })
    }

    pub(crate) async fn score_texts(&self, texts: Vec<String>) -> Result<Vec<f64>> {
        let text_count = texts.len();
        let (response_tx, response_rx) = oneshot::channel();
        self.sender
            .send(RerankerCommand::Score {
                texts,
                response: response_tx,
            })
            .map_err(|_| Error::LocalRuntimeChannelClosed)?;

        match tokio::time::timeout(local_timeout(), response_rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(Error::LocalRuntimeChannelClosed),
            Err(_) => Err(Error::LocalRuntime {
                message: format!(
                    "local reranking timed out after {}s (text count: {text_count})",
                    local_timeout().as_secs(),
                ),
            }),
        }
    }
}

impl LocalEmbeddingRuntime {
    fn new(backend: &LlamaBackend, model_path: PathBuf) -> Result<Self> {
        let model = load_model(backend, &model_path)?;
        Ok(Self { model })
    }

    fn run(self, backend: &LlamaBackend, receiver: thread_mpsc::Receiver<EmbeddingCommand>) {
        let mut context = match self.model.new_context(backend, {
            let n = 2048u32;
            LlamaContextParams::default()
                .with_embeddings(true)
                .with_n_ctx(Some(
                    NonZeroU32::new(n).expect("context size must be non-zero"),
                ))
                .with_n_batch(n)
        }) {
            Ok(ctx) => ctx,
            Err(err) => {
                let message = format!("failed to create llama.cpp embedding context: {err}");
                for command in receiver {
                    let EmbeddingCommand::Embed { response, .. } = command;
                    let _ = response.send(Err(Error::LocalRuntime {
                        message: message.clone(),
                    }));
                }
                return;
            }
        };

        for command in receiver {
            match command {
                EmbeddingCommand::Embed {
                    token_batches,
                    response,
                } => {
                    let _ = response.send(embed_token_batches(&mut context, &token_batches));
                }
            }
        }
    }
}

impl LocalRerankerRuntime {
    fn new(backend: &LlamaBackend, model_path: PathBuf) -> Result<Self> {
        let model = load_model(backend, &model_path)?;
        Ok(Self { model })
    }

    fn run(self, backend: &LlamaBackend, receiver: thread_mpsc::Receiver<RerankerCommand>) {
        let n = 4096u32;
        let params = LlamaContextParams::default()
            .with_embeddings(true)
            .with_pooling_type(LlamaPoolingType::Rank)
            .with_n_ctx(Some(
                NonZeroU32::new(n).expect("context size must be non-zero"),
            ))
            .with_n_batch(n)
            .with_flash_attention_policy(FLASH_ATTN_ENABLED);
        let mut context = match self.model.new_context(backend, params) {
            Ok(ctx) => ctx,
            Err(err) => {
                let message = format!("failed to create llama.cpp reranker context: {err}");
                for command in receiver {
                    let RerankerCommand::Score { response, .. } = command;
                    let _ = response.send(Err(Error::LocalRuntime {
                        message: message.clone(),
                    }));
                }
                return;
            }
        };

        for command in receiver {
            match command {
                RerankerCommand::Score { texts, response } => {
                    let _ = response.send(score_texts(&self.model, &mut context, &texts));
                }
            }
        }
    }
}

fn embed_token_batches(
    context: &mut LlamaContext,
    token_batches: &[Vec<u32>],
) -> Result<EmbedOutput> {
    if token_batches.is_empty() {
        return Ok(EmbedOutput {
            embeddings: Vec::new(),
        });
    }

    let mut token_sequences = Vec::with_capacity(token_batches.len());
    for (index, token_ids) in token_batches.iter().enumerate() {
        let tokens = token_ids_to_llama_tokens(token_ids, index)?;
        let _ = i32::try_from(tokens.len()).map_err(|_| Error::InvalidConfiguration {
            message: format!(
                "local embedding sequence {index} has {} tokens, which exceeds llama.cpp batch limits",
                tokens.len()
            ),
        })?;
        token_sequences.push(tokens);
    }

    let mut embeddings = Vec::with_capacity(token_sequences.len());
    for (index, tokens) in token_sequences.iter().enumerate() {
        tracing::debug!(
            sequence = index,
            total = token_sequences.len(),
            token_count = tokens.len(),
            "embedding decode start"
        );
        context.clear_kv_cache();
        let mut batch = LlamaBatch::new(tokens.len(), 1);
        batch
            .add_sequence(tokens, 0, false)
            .map_err(|err| Error::LocalRuntime {
                message: format!(
                    "failed to prepare llama.cpp embedding batch sequence {index}: {err}"
                ),
            })?;

        context
            .decode(&mut batch)
            .map_err(|err| Error::LocalRuntime {
                message: format!("llama.cpp embedding decode failed for sequence {index}: {err}"),
            })?;

        let embedding = context
            .embeddings_seq_ith(0)
            .map_err(|err| Error::LocalRuntime {
                message: format!(
                    "failed to read llama.cpp embedding output for sequence {index}: {err}"
                ),
            })?;
        embeddings.push(embedding.to_vec());
        tracing::debug!(
            sequence = index,
            total = token_sequences.len(),
            "embedding decode done"
        );
    }

    Ok(EmbedOutput { embeddings })
}

fn score_texts(
    model: &LlamaModel,
    context: &mut LlamaContext,
    texts: &[String],
) -> Result<Vec<f64>> {
    let mut scores = Vec::with_capacity(texts.len());
    for text in texts {
        scores.push(score_text(model, context, text)?);
    }
    Ok(scores)
}

fn score_text(model: &LlamaModel, context: &mut LlamaContext, text: &str) -> Result<f64> {
    let tokens = tokenize_nonempty(model, text)?;
    tracing::debug!(token_count = tokens.len(), "reranker decode start");
    context.clear_kv_cache();
    let mut batch = LlamaBatch::new(tokens.len(), 1);
    batch
        .add_sequence(&tokens, 0, false)
        .map_err(|err| Error::LocalRuntime {
            message: format!("failed to prepare llama.cpp reranker batch: {err}"),
        })?;
    context
        .decode(&mut batch)
        .map_err(|err| Error::LocalRuntime {
            message: format!("llama.cpp reranker decode failed: {err}"),
        })?;
    tracing::debug!(token_count = tokens.len(), "reranker decode done");
    let score = context
        .embeddings_seq_ith(0)
        .map_err(|err| Error::LocalRuntime {
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

#[derive(Debug)]
struct HuggingFaceModelSpec {
    full_spec: String,
    repo: String,
    filename: String,
}

fn parse_hf_model_spec(model: &str) -> Result<HuggingFaceModelSpec> {
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

    Ok(HuggingFaceModelSpec {
        full_spec: model.to_string(),
        repo: repo.to_string(),
        filename: filename.to_string(),
    })
}

#[cfg(test)]
fn resolve_model_path(model: &str) -> Result<PathBuf> {
    let spec = parse_hf_model_spec(model)?;
    resolve_model_path_from_spec(&spec)
}

fn resolve_model_path_from_spec(spec: &HuggingFaceModelSpec) -> Result<PathBuf> {
    hugging_face_api()?
        .model(spec.repo.clone())
        .get(&spec.filename)
        .map_err(|err| Error::LocalRuntime {
            message: format!(
                "failed to resolve Hugging Face GGUF artifact '{}': {err}",
                spec.full_spec
            ),
        })
}

fn hugging_face_api() -> Result<Api> {
    let progress = resolve_hf_hub_progress_enabled()?;

    ApiBuilder::new()
        .with_progress(progress)
        .build()
        .map_err(|err| Error::LocalRuntime {
            message: format!("failed to initialize hf-hub client: {err}"),
        })
}

fn resolve_hf_hub_progress_enabled() -> Result<bool> {
    let seasoning_progress = read_env_var(SEASONING_HF_HUB_PROGRESS_ENV)?;
    let hf_disable_progress = read_env_var(HF_HUB_DISABLE_PROGRESS_BARS_ENV)?;

    resolve_hf_hub_progress_from_env_values(
        seasoning_progress.as_deref(),
        hf_disable_progress.as_deref(),
    )
}

fn resolve_hf_hub_progress_from_env_values(
    seasoning_progress: Option<&str>,
    hf_disable_progress: Option<&str>,
) -> Result<bool> {
    if let Some(value) = seasoning_progress {
        return parse_bool_env_var(SEASONING_HF_HUB_PROGRESS_ENV, value);
    }

    if let Some(value) = hf_disable_progress {
        return parse_bool_env_var(HF_HUB_DISABLE_PROGRESS_BARS_ENV, value)
            .map(|disabled| !disabled);
    }

    Ok(true)
}

fn read_env_var(name: &'static str) -> Result<Option<String>> {
    match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(VarError::NotPresent) => Ok(None),
        Err(VarError::NotUnicode(_)) => Err(Error::InvalidConfiguration {
            message: format!("{name} must be valid unicode"),
        }),
    }
}

fn parse_bool_env_var(name: &'static str, value: &str) -> Result<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(Error::InvalidConfiguration {
            message: format!("{name} must be one of: 1, 0, true, false, yes, no, on, off"),
        }),
    }
}

fn local_debug_enabled() -> bool {
    std::env::var(SEASONING_LOCAL_DEBUG_ENV)
        .ok()
        .as_deref()
        .map_or(false, |v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
}

const SEASONING_LOCAL_TIMEOUT_ENV: &str = "SEASONING_LOCAL_TIMEOUT";
const DEFAULT_LOCAL_TIMEOUT: Duration = Duration::from_secs(60);

fn local_timeout() -> Duration {
    std::env::var(SEASONING_LOCAL_TIMEOUT_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map_or(DEFAULT_LOCAL_TIMEOUT, Duration::from_secs)
}

fn load_model(backend: &LlamaBackend, model_path: &Path) -> Result<LlamaModel> {
    let devices = list_llama_ggml_backend_devices();
    let selection = LocalModelSelection::from_devices(&devices);
    if local_debug_enabled() {
        emit_model_load_diagnostics(model_path, &selection, &devices);
    }
    let params = selection.apply(LlamaModelParams::default())?;

    LlamaModel::load_from_file(backend, model_path, &params).map_err(|err| Error::LocalRuntime {
        message: format!(
            "failed to load llama.cpp model from '{}': {err}",
            model_path.display()
        ),
    })
}

fn token_ids_to_llama_tokens(token_ids: &[u32], index: usize) -> Result<Vec<LlamaToken>> {
    token_ids
        .iter()
        .map(|token_id| {
            let token = i32::try_from(*token_id).map_err(|_| Error::InvalidEmbeddingTokenId {
                index,
                token_id: *token_id,
            })?;
            Ok(LlamaToken::new(token))
        })
        .collect()
}

fn tokenize_nonempty(model: &LlamaModel, text: &str) -> Result<Vec<LlamaToken>> {
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

fn preferred_backend_device_indices(devices: &[LlamaBackendDevice]) -> Vec<usize> {
    devices
        .iter()
        .filter(|device| {
            matches!(
                device.device_type,
                LlamaBackendDeviceType::Gpu
                    | LlamaBackendDeviceType::IntegratedGpu
                    | LlamaBackendDeviceType::Accelerator
            )
        })
        .map(|device| device.index)
        .collect()
}

fn emit_backend_diagnostics(devices: &[LlamaBackendDevice]) {
    eprintln!("seasoning local: llama.cpp backend initialized");
    if devices.is_empty() {
        eprintln!("seasoning local: no ggml backend devices reported");
        return;
    }

    for device in devices {
        eprintln!(
            "seasoning local: detected {}",
            format_backend_device(device)
        );
    }
}

fn emit_model_load_diagnostics(
    model_path: &Path,
    selection: &LocalModelSelection,
    devices: &[LlamaBackendDevice],
) {
    if selection.selected_device_indices.is_empty() {
        eprintln!(
            "seasoning local: loading '{}' with GPU offload disabled (no GPU/accelerator devices selected)",
            model_path.display()
        );
        return;
    }

    let selected_devices = selection
        .selected_device_indices
        .iter()
        .filter_map(|selected_index| {
            devices
                .iter()
                .find(|device| device.index == *selected_index)
        })
        .map(format_backend_device)
        .collect::<Vec<_>>()
        .join(", ");
    let gpu_offload = if selection.gpu_offload_enabled {
        "all layers"
    } else {
        "disabled"
    };

    eprintln!(
        "seasoning local: loading '{}' with GPU offload={} on {}",
        model_path.display(),
        gpu_offload,
        selected_devices
    );
}

fn format_backend_device(device: &LlamaBackendDevice) -> String {
    format!(
        "device #{} backend={} type={} name='{}' description='{}' memory_free={} memory_total={}",
        device.index,
        device.backend,
        backend_device_type_name(device.device_type),
        device.name,
        device.description,
        format_memory_bytes(device.memory_free),
        format_memory_bytes(device.memory_total)
    )
}

fn backend_device_type_name(device_type: LlamaBackendDeviceType) -> &'static str {
    match device_type {
        LlamaBackendDeviceType::Cpu => "cpu",
        LlamaBackendDeviceType::Accelerator => "accelerator",
        LlamaBackendDeviceType::Gpu => "gpu",
        LlamaBackendDeviceType::IntegratedGpu => "integrated-gpu",
        LlamaBackendDeviceType::Unknown => "unknown",
    }
}

fn format_memory_bytes(bytes: usize) -> String {
    const KIB: usize = 1024;
    const MIB: usize = KIB * 1024;
    const GIB: usize = MIB * 1024;

    if bytes >= GIB {
        format!("{:.1} GiB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.1} MiB", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{:.1} KiB", bytes as f64 / KIB as f64)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    use crate::embedding::{Client as EmbeddingClient, EmbedderConfig};
    use crate::reranker::{Client as RerankerClient, RerankerConfig};
    use crate::{
        EmbeddingInput, EmbeddingProvider, EmbeddingRole, RerankDocument, RerankQuery,
        RerankingProvider, Tokenizer,
    };

    fn tokenizer_for_model_spec(model: &str) -> Result<Tokenizer> {
        let model_id = match model {
            GEMMA_EMBEDDING_MODEL => "google/embeddinggemma-300m",
            QWEN3_EMBEDDING_MODEL => "Qwen/Qwen3-Embedding-0.6B",
            other => panic!("unsupported tokenizer mapping for model: {other}"),
        };
        let tokenizer = tokenizers::Tokenizer::from_pretrained(model_id, None).map_err(|error| {
            Error::LocalRuntime {
                message: format!(
                    "failed to load test tokenizer '{model_id}' for local embedding model '{model}': {error}"
                ),
            }
        })?;
        Ok(Tokenizer::HuggingFace {
            model_id: model_id.to_string(),
            tokenizer: std::sync::Arc::new(tokenizer),
        })
    }

    fn test_embedding_config(model_family: ModelFamily, model: &str) -> Result<EmbedderConfig> {
        Ok(EmbedderConfig::local(
            model_family,
            tokenizer_for_model_spec(model)?,
            model.to_string(),
            None,
        ))
    }

    fn test_reranker_config(model: &str) -> RerankerConfig {
        RerankerConfig {
            api_key: None,
            base_url: String::new(),
            timeout: Duration::from_secs(30),
            dialect: crate::Dialect::LlamaCpp,
            model_family: ModelFamily::Qwen3,
            model: model.to_string(),
            instruction: None,
            requests_per_minute: 1000,
            max_concurrent_requests: 1,
            tokens_per_minute: 1_000_000,
            backend: None,
        }
    }

    fn token_ids_for_text(model: &LlamaModel, text: &str) -> Vec<u32> {
        tokenize_nonempty(model, text)
            .unwrap()
            .into_iter()
            .map(|token| u32::try_from(token.0).expect("llama token ids should be non-negative"))
            .collect()
    }

    fn formatted_embedding_token_count(
        model_family: ModelFamily,
        tokenizer: &Tokenizer,
        input: &EmbeddingInput,
    ) -> usize {
        let rendered = model_family.format_embedding_input(input, None);
        tokenizer.prepare(rendered).unwrap().token_count()
    }

    fn token_count_for_text(model: &LlamaModel, text: &str) -> usize {
        token_ids_for_text(model, text).len()
    }

    fn max_abs_diff(left: &[f32], right: &[f32]) -> f32 {
        assert_eq!(left.len(), right.len());
        left.iter()
            .zip(right)
            .fold(0.0f32, |diff, (lhs, rhs)| diff.max((lhs - rhs).abs()))
    }

    #[test]
    fn parse_hf_model_spec_rejects_non_hf_model_ids() {
        let err = parse_hf_model_spec("example.gguf").unwrap_err();
        assert!(matches!(err, Error::UnsupportedConfiguration { .. }));
    }

    #[test]
    fn token_id_conversion_rejects_out_of_range_values() {
        let err = token_ids_to_llama_tokens(&[u32::MAX], 2).unwrap_err();

        assert!(matches!(
            err,
            Error::InvalidEmbeddingTokenId {
                index: 2,
                token_id: u32::MAX,
            }
        ));
    }

    #[test]
    fn hf_hub_progress_defaults_to_enabled() {
        assert!(resolve_hf_hub_progress_from_env_values(None, None).unwrap());
    }

    #[test]
    fn hf_hub_disable_progress_env_var_disables_progress() {
        assert!(!resolve_hf_hub_progress_from_env_values(None, Some("1")).unwrap());
        assert!(resolve_hf_hub_progress_from_env_values(None, Some("false")).unwrap());
    }

    #[test]
    fn seasoning_progress_env_var_overrides_hf_disable_progress_env_var() {
        assert!(resolve_hf_hub_progress_from_env_values(Some("true"), Some("1")).unwrap());
        assert!(!resolve_hf_hub_progress_from_env_values(Some("off"), Some("0")).unwrap());
    }

    #[test]
    fn invalid_progress_env_var_value_is_rejected() {
        let err = resolve_hf_hub_progress_from_env_values(Some("maybe"), None).unwrap_err();

        assert!(matches!(err, Error::InvalidConfiguration { .. }));
    }

    fn backend_device(
        index: usize,
        backend: &str,
        device_type: LlamaBackendDeviceType,
    ) -> LlamaBackendDevice {
        LlamaBackendDevice {
            index,
            name: format!("device-{index}"),
            description: format!("{backend} device {index}"),
            backend: backend.to_string(),
            memory_total: 8 * 1024 * 1024 * 1024,
            memory_free: 6 * 1024 * 1024 * 1024,
            device_type,
        }
    }

    #[test]
    fn local_model_selection_prefers_non_cpu_devices() {
        let devices = vec![
            backend_device(0, "CPU", LlamaBackendDeviceType::Cpu),
            backend_device(1, "Vulkan", LlamaBackendDeviceType::IntegratedGpu),
            backend_device(2, "CUDA", LlamaBackendDeviceType::Gpu),
        ];

        let selection = LocalModelSelection::from_devices(&devices);

        assert_eq!(selection.selected_device_indices, vec![1, 2]);
        assert!(selection.gpu_offload_enabled);
    }

    #[test]
    fn local_model_selection_falls_back_to_cpu_when_no_accelerator_exists() {
        let devices = vec![backend_device(0, "CPU", LlamaBackendDeviceType::Cpu)];

        let selection = LocalModelSelection::from_devices(&devices);

        assert!(selection.selected_device_indices.is_empty());
        assert!(!selection.gpu_offload_enabled);
    }

    #[test]
    fn backend_device_format_includes_backend_type_and_memory() {
        let device = backend_device(3, "Vulkan", LlamaBackendDeviceType::IntegratedGpu);

        let formatted = format_backend_device(&device);

        assert!(formatted.contains("device #3"));
        assert!(formatted.contains("backend=Vulkan"));
        assert!(formatted.contains("type=integrated-gpu"));
        assert!(formatted.contains("memory_free=6.0 GiB"));
        assert!(formatted.contains("memory_total=8.0 GiB"));
    }

    #[tokio::test]
    async fn local_embedding_clients_embed_supported_models_end_to_end() {
        let mut exercised_models = 0usize;

        for (model_family, model_spec) in [
            (ModelFamily::Gemma, GEMMA_EMBEDDING_MODEL),
            (ModelFamily::Qwen3, QWEN3_EMBEDDING_MODEL),
        ] {
            let Ok(tokenizer) = tokenizer_for_model_spec(model_spec) else {
                eprintln!(
                    "skipping local embedding E2E for {model_spec}: tokenizer fetch unavailable"
                );
                continue;
            };
            let client =
                EmbeddingClient::new(test_embedding_config(model_family, model_spec).unwrap())
                    .unwrap();
            exercised_models += 1;
            let query_a = EmbeddingInput {
                role: EmbeddingRole::Query,
                text: "memory safety in rust".to_string(),
                title: None,
                token_count: 0,
            };
            let query_b = EmbeddingInput {
                role: EmbeddingRole::Query,
                text: "memory safety in rust".to_string(),
                title: None,
                token_count: 0,
            };
            let query_c = EmbeddingInput {
                role: EmbeddingRole::Query,
                text: "tropical fruit smoothie recipes".to_string(),
                title: None,
                token_count: 0,
            };
            let semantic_inputs = vec![
                EmbeddingInput {
                    token_count: formatted_embedding_token_count(
                        model_family,
                        &tokenizer,
                        &query_a,
                    ),
                    ..query_a
                },
                EmbeddingInput {
                    token_count: formatted_embedding_token_count(
                        model_family,
                        &tokenizer,
                        &query_b,
                    ),
                    ..query_b
                },
                EmbeddingInput {
                    token_count: formatted_embedding_token_count(
                        model_family,
                        &tokenizer,
                        &query_c,
                    ),
                    ..query_c
                },
            ];

            let output = client.embed(&semantic_inputs).await.unwrap();

            assert_eq!(
                output.embeddings.len(),
                semantic_inputs.len(),
                "model {model_spec}"
            );
            let dimension = output.embeddings[0].len();
            assert!(
                dimension > 0,
                "model {model_spec} produced empty embeddings"
            );
            assert!(
                output
                    .embeddings
                    .iter()
                    .all(|embedding| embedding.len() == dimension),
                "model {model_spec} returned inconsistent embedding dimensions"
            );
            assert!(
                output
                    .embeddings
                    .iter()
                    .flatten()
                    .all(|value| value.is_finite()),
                "model {model_spec} returned non-finite embedding values"
            );
            assert!(
                max_abs_diff(&output.embeddings[0], &output.embeddings[1]) < 1e-6,
                "model {model_spec} should produce stable embeddings for duplicate queries"
            );
            assert!(
                max_abs_diff(&output.embeddings[0], &output.embeddings[2]) > 1e-6,
                "model {model_spec} should distinguish unrelated queries"
            );
        }

        assert!(
            exercised_models > 0,
            "expected at least one local embedding model E2E path to run"
        );
    }

    #[tokio::test]
    async fn local_reranker_scores_supported_model_end_to_end() {
        let backend = crate::local::create_backend().unwrap();
        let tokenizer_model =
            load_model(&backend, &resolve_model_path(QWEN3_RERANKER_MODEL).unwrap()).unwrap();
        let mut config = test_reranker_config(QWEN3_RERANKER_MODEL);
        config.backend = Some(std::sync::Arc::new(backend));
        let client = RerankerClient::new(config).unwrap();
        let query_text = "how does rust prevent data races";
        let query = RerankQuery {
            text: query_text.to_string(),
            token_count: token_count_for_text(&tokenizer_model, query_text),
        };
        let documents = [
            "Rust prevents data races with ownership and borrowing.",
            "Rust prevents data races with ownership and borrowing.",
            "Bananas are yellow fruit often blended into smoothies.",
        ]
        .into_iter()
        .map(|text| RerankDocument {
            text: text.to_string(),
            token_count: token_count_for_text(&tokenizer_model, text),
        })
        .collect::<Vec<_>>();

        let scores = client.rerank(&query, &documents).await.unwrap();

        assert_eq!(scores.len(), documents.len());
        assert!(scores.iter().all(|score| score.is_finite()));
        assert!(
            (scores[0] - scores[1]).abs() < 1e-6,
            "duplicate documents should receive matching scores: {scores:?}"
        );
        assert!(
            scores[0] > scores[2],
            "relevant document should outrank unrelated text: {scores:?}"
        );
    }

    #[tokio::test]
    async fn local_reranker_preserves_input_to_score_mapping() {
        let backend = crate::local::create_backend().unwrap();
        let tokenizer_model =
            load_model(&backend, &resolve_model_path(QWEN3_RERANKER_MODEL).unwrap()).unwrap();
        let mut config = test_reranker_config(QWEN3_RERANKER_MODEL);
        config.backend = Some(std::sync::Arc::new(backend));
        let client = RerankerClient::new(config).unwrap();
        let query_text = "how does rust prevent data races";
        let query = RerankQuery {
            text: query_text.to_string(),
            token_count: token_count_for_text(&tokenizer_model, query_text),
        };

        let doc_a = "Rust prevents data races with ownership and borrowing.";
        let doc_b = "Ownership rules stop aliasing mutable state across threads.";
        let doc_c = "Bananas are yellow fruit often blended into smoothies.";

        let ordered = [doc_a, doc_b, doc_c]
            .into_iter()
            .map(|text| RerankDocument {
                text: text.to_string(),
                token_count: token_count_for_text(&tokenizer_model, text),
            })
            .collect::<Vec<_>>();
        let permuted = [doc_c, doc_a, doc_b]
            .into_iter()
            .map(|text| RerankDocument {
                text: text.to_string(),
                token_count: token_count_for_text(&tokenizer_model, text),
            })
            .collect::<Vec<_>>();

        let ordered_scores = client.rerank(&query, &ordered).await.unwrap();
        let permuted_scores = client.rerank(&query, &permuted).await.unwrap();

        assert_eq!(ordered_scores.len(), 3);
        assert_eq!(permuted_scores.len(), 3);

        assert!(
            (ordered_scores[0] - permuted_scores[1]).abs() < 1e-6,
            "doc_a score should follow doc_a across permutations: ordered={ordered_scores:?} permuted={permuted_scores:?}"
        );
        assert!(
            (ordered_scores[1] - permuted_scores[2]).abs() < 1e-6,
            "doc_b score should follow doc_b across permutations: ordered={ordered_scores:?} permuted={permuted_scores:?}"
        );
        assert!(
            (ordered_scores[2] - permuted_scores[0]).abs() < 1e-6,
            "doc_c score should follow doc_c across permutations: ordered={ordered_scores:?} permuted={permuted_scores:?}"
        );
    }
}
