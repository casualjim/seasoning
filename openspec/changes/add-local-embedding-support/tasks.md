## 1. API and semantic model cleanup

- [ ] 1.1 Replace the current provider-dialect abstraction with `Dialect = OpenAI | DeepInfra | LlamaCpp` and add `ModelFamily = Gemma | Qwen3`
- [ ] 1.2 Extend embedding configuration and input types to carry query/document role, optional title metadata, and query-side instruction/task text
- [ ] 1.3 Move retrieval formatting into enum-owned methods and fix the current embedding-path bug so remote embedding behavior is explicit and validated

## 2. Local llama.cpp backend support

- [ ] 2.1 Add optional dependencies and Cargo features for `local`, `cuda`, `metal`, and `vulkan`, with accelerator features implying `local`
- [ ] 2.2 Implement Hugging Face GGUF resolution and lazy local embedding execution for the scoped Gemma and Qwen3 embedding models
- [ ] 2.3 Implement lazy local reranking execution for the scoped Qwen3 reranker model, preserving one-score-per-document behavior

## 3. Validation and test coverage

- [ ] 3.1 Add explicit validation for unsupported dialect/family/feature combinations and clear errors for local-without-feature configuration
- [ ] 3.2 Add unit tests for Gemma and Qwen3 embedding formatting, including query/document role, title handling, and instruction defaults/overrides
- [ ] 3.3 Add regression tests for existing remote behavior and feature-gated construction tests for local embedding and reranking clients

## 4. Documentation and release messaging

- [ ] 4.1 Update README and crate examples to show the new semantic embedding inputs, supported model families, and feature-gated local usage
- [ ] 4.2 Document the minor-version semantic expansion and the existing embedding dialect bug fix in the change summary / release-facing notes
