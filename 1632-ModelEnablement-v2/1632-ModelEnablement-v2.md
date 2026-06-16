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
- A testing methodology for verifying model correctness across CPU, GPU, and Spyre
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

**Summary of key adapter techniques:**
- RoPE replaced with precomputed rotation matrices (Spyre has no `sin`/`cos` ops)
- RMSNorm patched to stay fp16 (Spyre does not support on-device dtype conversion)
- Decoder layers compiled via `torch.compile(block_forward, dynamic=False)` using plain function closures instead of `DynamicCache` objects
- Custom generation loop with 64-block padded decode

### 2. Model ID and Status Labels

Each model is identified by its HuggingFace `model_id` (e.g., `google/gemma-3-4b-it`), which serves as the primary key for tracking, labeling, and test registration. Each model may carry zero or more of the following binary status labels. Labels are independent and non-exclusive — a model can have any combination.

| Label | Description |
|---|---|
| **has adapter** | The model belongs to a model family for which `hf-adapters` has an adapter (i.e., its `config` type is registered in `CONFIG_TO_ADAPTER_MODULE_MAPPING`). This does not imply the model has been tested. |
| **runnable on spyre** | The model can be loaded and executed on Spyre hardware without error. |
| **has reference** | A reference output has been collected by running the model on CPU or GPU. Used as the ground truth for accuracy comparisons. |
| **verified on cpu** | The adapted model has been tested on CPU and its output matches the reference. |
| **verified on gpu** | The adapted model has been tested on GPU and its output matches the reference. |
| **verified on spyre** | The adapted model has been tested on Spyre and its output matches the reference. |

Labels are assigned and updated as testing progresses. A model may hold multiple labels (e.g., `has adapter`, `has reference`, `verified on cpu`, `verified on spyre`).

### 3. Model Registry and Selection

#### Registry File

All tracked models are stored in a checked-in file (`models.csv`) in the repository. Each row represents one model with the following columns:

- `model_id` — the HuggingFace model identifier, used as the primary key
- `type` — either `generative` or `embedding`
- One boolean column per status label (e.g., `has_adapter`, `runnable_on_spyre`, `has_reference`, `verified_on_cpu`, `verified_on_gpu`, `verified_on_spyre`)

Label columns are updated in place as testing progresses. The registry is append-only with respect to model selection: once a model is added, it remains in the file permanently regardless of whether subsequent scans would still surface it. This ensures stability — the tracked set is never silently altered by shifts in HuggingFace popularity rankings.

#### Selection Scan

The selection scan is a script that identifies new candidate models to append to `models.csv`. It runs on demand (e.g., when the team wants to expand coverage) and outputs a summary of the projected registry state after the candidates are added.

Two independent scans are run — one for generative models and one for embedding models — each fetching the top 5k models of that type from HuggingFace sorted by downloads. This keeps the two populations cleanly separated and ensures embedding models are not crowded out by the more numerous generative models.

Each scan proceeds as follows:

1. **Fetch the top 5k models** of the given type from HuggingFace sorted by downloads, forming the candidate pool.

2. **Apply hard filters** to skip models not currently supportable:
   - Mixture-of-experts architectures
   - Models exceeding 20 billion parameters
   - Any other unsupported architecture or quantization constraints

   These filters are expected to evolve as Spyre support expands.

3. **Skip models already in the registry.** Any `model_id` present in `models.csv` is excluded from consideration.

4. **Append remaining candidates to `models.csv`** in download-rank order, with `type` set accordingly and all label columns set to `false`.

#### Scan Output

After appending, the scan prints a summary reflecting the projected state of `models.csv`. The summary includes:

- Total models in registry (before and after)
- Count and percentage of generative vs. embedding models
- Count of models per label (e.g., how many are `verified_on_spyre`)
- Number of models added in this run, per type

### 5. Reference Output Storage

Reference outputs (golden files) are stored directly in the test directory alongside the tests that use them. Each golden file contains the expected output for a specific model under a specific input, and is committed to the repository as a versioned artifact. When a model's reference output needs to be updated — due to a model update, adapter change, or intentional behavior shift — the golden file is regenerated and the diff is reviewed as part of the PR.

This approach keeps tests self-contained: running a test requires no external storage or network access, and changes to expected outputs are visible in code review.

### 6. Testing Methodology

Tests are scoped differently depending on model type. In all cases, the test compares the adapted model's output against the reference output collected under the `has reference` label.

#### Generative Models
- **Input:** a fixed sentence of text
- **Output compared:** the first 5 generated tokens

#### Embedding Models — Single-Vector
- **Input:** a fixed sentence of text
- **Output compared:** the resulting embedding vector

#### Embedding Models — Multi-Vector
- **Input:** a fixed sentence of text
- **Output compared:** the per-token embedding vectors

### 7. CI/CD Regression Testing

Regression tests verify that models previously labeled `verified on spyre` continue to work correctly after code changes. Three tiers are defined:

#### PR-Level Tests
- **Trigger:** every pull request
- **Scope:** 2–3 representative models
- **Purpose:** fast signal to block merging if a critical model regresses
- **Constraint:** must complete quickly enough to not bottleneck the review cycle

#### Nightly Tests
- **Trigger:** every night (scheduled)
- **Scope:** one representative model per adapter family — ensuring every adapter is exercised
- **Purpose:** daily confidence that each adapter continues to function correctly

#### Weekly Tests
- **Trigger:** every weekend (scheduled)
- **Scope:** all models labeled `verified on spyre`
- **Purpose:** comprehensive regression check across the full verified model set

---

## Metrics

- Number of models per label (e.g., count of models with `verified on spyre`)
- Regression test pass rate per tier (PR / nightly / weekly)
- Trend of `verified on spyre` model count over time

---

## Drawbacks

- Maintaining reference outputs adds storage and update overhead as models evolve
- The 5-token comparison for generative models may not catch subtle numerical regressions that manifest only over longer sequences
- PR-level tests covering only 2–3 models risk missing regressions in less-covered model families
- The label system is binary — it does not capture partial or degraded functionality (e.g., a model that runs but with reduced accuracy)

---

## Alternatives

- **Metric-based tracking (v1 approach):** tracking ops% and modules% per model gives finer-grained visibility into enablement progress but does not capture end-to-end correctness. The label-based approach is more actionable for CI/CD.
- **Single test tier:** simpler to operate, but too slow for PR gates or too sparse for weekly confidence.
- **Using vLLM model definitions (v1 recommendation):** v1 argued for tracing vLLM model definitions. This v2 instead uses HuggingFace model definitions via `hf-adapters`, which avoids vLLM dependency and allows direct comparison against HF reference outputs.

---

## Prior Art

- **RFC #1632 v1** (Romit Jain & Ashok Pon Kumar Sree Prakash): established the motivation for Spyre model enablement tracking, recommended vLLM-based tracing, and proposed ops/modules percentage metrics. This v2 builds on those findings while shifting to an HF-native, label-based approach.
- **`hf-adapters` repository**: the concrete implementation this RFC formalizes. See [README.md](https://github.com/torch-spyre/hf-adapters/blob/main/README.md), [ARCHITECTURE.md](https://github.com/torch-spyre/hf-adapters/blob/main/ARCHITECTURE.md), [ONBOARDING.md](https://github.com/torch-spyre/hf-adapters/blob/main/ONBOARDING.md).

---

## How We Teach This

- The status labels provide a shared vocabulary for communicating model readiness across teams — adoption requires documenting the label definitions in the project wiki or README
- The tiered CI/CD structure maps naturally onto existing PR and scheduled pipeline concepts; onboarding a new model to the regression suite means adding it to the appropriate test tier
- New contributors should follow [ONBOARDING.md](https://github.com/torch-spyre/hf-adapters/blob/main/ONBOARDING.md) to add a model, then update its labels as tests pass

---

## Unresolved Questions

- **Label ownership:** who is responsible for updating labels, and is this manual or automated via CI?
- **PR model selection:** which 2–3 models should be in the PR tier, and how should that list be maintained as the supported set grows?
- **Nightly model list:** criteria for a model to be included in the "must-work" nightly list
- **Partial accuracy:** should the label system be extended to capture degraded-but-functional states (e.g., `verified on spyre (degraded)`)?

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
