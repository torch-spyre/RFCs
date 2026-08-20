# Model Enablement v2 (RFC #1632)

**Authors:**
- [Ariel Gera](https://github.com/arielge)
- [Benjamin Sznajder](https://github.com/BenjSz)
- [Assaf Toledo](https://github.com/assaftibm)

---

## Summary

This RFC defines the second iteration of the model enablement framework for Spyre accelerators. It covers:

- How models are onboarded via the `hf-adapters` approach
- A structured labeling system to track enablement status per model
- A testing methodology for verifying model correctness across CPU and Spyre
- A CI/CD regression testing strategy with PR-level, nightly, and weekly tiers

---

## Motivation

As the number of models supported on Spyre grows, a clear and consistent process is needed to track which models work, to what degree, and under what conditions. Without a shared labeling vocabulary and a tiered testing strategy, it becomes difficult to communicate model status across teams, gate regressions, and make informed decisions about which models are production-ready.

This RFC builds on the findings of RFC #1632 v1 (Romit Jain & Ashok Pon Kumar Sree Prakash), which established the motivation for using HuggingFace model definitions and outlined early metrics for ops and module coverage. This v2 shifts from a metric-based view to a label-based, test-driven framework grounded in the `hf-adapters` implementation.

---

## Proposed Implementation

### 1. Model Enablement Approach

Model enablement is implemented via the [`hf-adapters`](https://github.com/torch-spyre/hf-adapters) library. The approach is based on minimal runtime patches (monkey-patches) applied to stock HuggingFace Transformers models at load time, requiring no forks or custom model classes. Each adapter replaces only the operations Spyre cannot execute natively — such as RoPE, RMSNorm, KV cache management, and the generation loop.

For a full description of the approach, architecture, and onboarding process, refer to:
- [README.md](https://github.com/torch-spyre/hf-adapters/blob/main/README.md) — overview, supported models, quick start
- [ARCHITECTURE.md](https://github.com/torch-spyre/hf-adapters/blob/main/ARCHITECTURE.md) — adapter internals, key deviations from stock HF, known issues
- [ONBOARDING.md](https://github.com/torch-spyre/hf-adapters/blob/main/ONBOARDING.md) — step-by-step guide for adding a new model

**Exemplars of Adapter Techniques:**
- RoPE replaced with precomputed rotation matrices (Spyre has no `sin`/`cos` ops)
- RMSNorm patched to stay fp16 (Spyre does not support on-device dtype conversion)
- Decoder layers compiled via `torch.compile(block_forward, dynamic=False)` using plain function closures instead of `DynamicCache` objects
- Custom generation loop with 64-block padded decode

### 2. Test Types and Workflows

We defined the following tests:

| Test type | Model type | What it validates |
|---|---|---|
| Model loading | Generative | The causal-LM auto class resolves the adapter, loads and moves the model to Spyre, and attaches `generate()`. It does not run inference. |
| Model loading | Embedding | The embedding auto class resolves the adapter and loads and moves the model to Spyre. It does not run inference. |
| Smoke | Generative | Loads the model, greedily generates five tokens, and verifies that the output is nonempty and nontrivial. It does not compare the output with a CPU reference. |
| Embedding comparison | Embedding | Encodes several prompts through CPU Sentence Transformers and the Spyre backend, then compares token and sentence embeddings using cosine similarity. |
| Token-level comparison | Generative | Runs stock Hugging Face on CPU and the adapter on Spyre for prefill plus four greedy decode steps, comparing logits and selected tokens at every step. This exercises generation, KV-cache updates, masks, position IDs, and decoding. |

These tests are used by the following workflows:

| Workflow | Trigger | What it runs |
|---|---|---|
| PR / regression | PRs, merge queue, pushes to `main`, tags, or manual dispatch | Runs all test types defined above: **Model loading** for generative and embedding models, **Smoke**, **Embedding comparison**, and **Token-level comparison**. It also runs VLM, reranker, masked-LM, question-answering, model/module, adapter-coverage, and static checks. |
| Integration | Manual or upstream-repository dispatch | Runs **Smoke** on four causal models by default. It can optionally run the unit, regression, or trunk tier, selected models, or specific combinations of the test types defined above. |
| Daily | Every day at 06:00 UTC or manual dispatch | Runs slow generative edge-case coverage related to **Smoke** and **Token-level comparison**: EOS handling, block boundaries, short, single-token and mixed prompts, missing pad tokens, sampling, and zero-token generation. These tests compare generated behavior rather than per-step logit vectors. |
| Weekly | Every Saturday at 22:00 UTC or manual dispatch | Fetches and filters the top 10K generative and embedding models and shards them across x1/x2/x4 Spyre runners. Generative models run **Model loading**, **Smoke**, and **Token-level comparison**; embedding models run **Model loading** and **Embedding comparison**. Results are written to ClickHouse. |

#### Weekly Test

The weekly test covers a wide range of models and shows the overall model enablement status. The model list is defined as a concatenation of two sources with duplicates being removed: 
1. The Top 10K models from HuggingFace sorted by number of downloads.
2. A list of pre-defined models with a business incentive.

The same logic is applied to both generative and embedding models. 
As part of the scan, certain models are automatically rejected due to support limitations. For example:
   - Mixture-of-experts architectures
   - Models exceeding 20 billion parameters
   - Any other unsupported architecture or quantization constraints

The rest of the models are tested based on their type: for generative models, the token-level comparison is used; for Embedding models, Embedding comparison.

The results of the weekly are injected to a Clickhouse DB and a dashboard shows the results to users. For each model, the HF Class is recorded, the status of 


## Prior Art

- **RFC #1632 v1** (Romit Jain & Ashok Pon Kumar Sree Prakash): established the motivation for Spyre model enablement tracking, recommended vLLM-based tracing, and proposed ops/modules percentage metrics. This v2 builds on those findings while shifting to an HF-native, label-based approach.
- **`hf-adapters` repository**: the concrete implementation this RFC formalizes. See [README.md](https://github.com/torch-spyre/hf-adapters/blob/main/README.md), [ARCHITECTURE.md](https://github.com/torch-spyre/hf-adapters/blob/main/ARCHITECTURE.md), [ONBOARDING.md](https://github.com/torch-spyre/hf-adapters/blob/main/ONBOARDING.md).


## Unresolved Questions

- **Adapter Minimalism** we want adapters to include as few patches as possible but we have no mechanism that verifies that.
- **Update Automation** we want adapters to update when torch-spyre is changing but currently the process is manual.  

---

## Resolution

*TBD*

### Level of Support
*TBD*

#### Additional Context
*TBD*

### Next Steps
*TBD*

#### Tracking Issue
*TBD*

#### Exceptions
*TBD*
