# RFC: Unified Model Quality Evaluation Framework

**Authors:**
- Lipika Sreedharan
- Antoni Viros i Martin
- Saurabh Srivastava
- Rishika Kedia
- Ajit Samuel John

**Tracking issue:** https://github.com/torch-spyre/torch-spyre/issues/4477

---

## 1. Summary

This RFC proposes a unified, config-driven model quality evaluation framework for IBM Spyre and CPU backends. Evaluation can be done at different layers of the Spyre stack — torch-spyre (low-level hardware backend), spyre-inference (the serving/HTTP layer) and hf-adapters (HuggingFace-compatible model loading for Spyre); this framework targets the hf-adapters layer.

The primary and recommended path for Spyre evaluations is the **vLLM backend** — the eval process connects to a running vLLM server over HTTP, either on IBM Spyre (vllm-spyre) or GPU (vllm-gpu). The recommended cross-platform comparison is **Spyre-vLLM vs GPU-vLLM**, ensuring any quality delta is attributable to the hardware backend rather than differences in request/response processing. A direct hf-adapters backend is also available for in-process Spyre debugging, and a CPU backend is retained for local development only.

The framework evaluates both decoder (generative/LLM) and encoder (embedding/sentence-embedding) models using a single entry point and a single YAML configuration file. It integrates two complementary evaluation paradigms - free-form generation quality (ROUGE, BLEU, METEOR, BERTScore, Exact Match) and standardized academic benchmarks via lm-evaluation-harness (MMLU, ARC, HellaSwag, GSM8K) and produces machine-readable JSON result files consumed by a downstream dashboard.

---

## 2. Motivation

The primary question this framework answers is: **does IBM Spyre produce inference results of expected quality consistent with a reference platform (GPU) across the range of scenarios the runtime stack must support?**

This is a **platform correctness** concern, not a model ranking concern. We do not care whether a model outperforms another model. We care whether a model running on Spyre behaves the way it is supposed to: producing the same quality of output it would produce on a GPU reference, across varied and realistic inference scenarios.

### Inference scenarios that must be validated

| Scenario | Why it matters for Spyre |
|---|---|
| Short context, short generation | Baseline correctness |
| Short context, long generation | Tests generation loop stability |
| Long context, short generation | Tests context handling, attention, padding |
| Long context, long generation | Tests memory pressure and graph shape limits |
| Tool calling / function calling | Tests structured output and special token handling |
| Special characters and non-Latin scripts | Tests tokenizer round-trip fidelity |
| Multimodal inputs (text + image) | Tests VLM inference path *(gap — see Section 9)* |

### Problems addressed

Before this framework, evaluating inference quality on Spyre required bespoke scripts per model and per dataset, with no standardized output format, no reproducibility guarantees, and no shared tooling. Comparing a Spyre run against a GPU reference meant manually diffing ad-hoc outputs.

| Problem | Impact |
|---|---|
| No unified entry point for multi-dataset, multi-model Spyre eval | Each engineer wrote their own runner script |
| No standard output format for dashboard consumption | Dashboard couldn't auto-discover results |
| No reproducible sampling (same N samples every run) | Results not comparable across runs |
| lm-eval harness not wired to Spyre/vLLM backend | Benchmark consistency with reference platform not measurable |
| No embedding model evaluation path | Embedding models had no quality gate |
| No per-run metric metadata in JSON | No shared understanding of metric value meaning (scale, direction) across tools |

---

## 3. Proposed Implementation

### 3.1 Architecture Overview

The framework has four layers — Configuration, Decoder Model Path (generative), Encoder Model Path (embedding) and Output — all dispatched from a single entry point.

![Architecture Diagram — four layers: Configuration dispatches to Decoder path (dataset loading → prompt building → model inference via CPU backend or Spyre backend [Approach 1: hf-adapters | Approach 2: vLLM server] → metric computation → result serialization → benchmark runner) and Encoder path (embedding runner via vLLM /v1/embeddings or local sentence-transformers → MTEB tasks); both feed into Output (result files → results manifest → dashboard)](quality_eval_architecture.png)

The model inference stage supports three backends. The **recommended comparison setup** is **Spyre-vLLM vs GPU-vLLM** - both sides use the same vLLM version and configuration, so any quality delta is attributable to the backend hardware/kernel path rather than differences in prompt templating, tokenization, or output parsing.

- **VLLMAdapter (Spyre)** - connects to a vLLM server running on IBM Spyre hardware (`backend: "vllm-spyre"`). Primary Spyre evaluation path.
- **VLLMAdapter (GPU)** - connects to a vLLM server running on GPU (`backend: "vllm-gpu"`). **Reference platform** for cross-platform comparisons. Replaces CPU as the recommended baseline.
- **SpyreAdapter** - loads the model via `hf_adapters.AutoSpyreModelForCausalLM` directly in the eval process (`backend: "hf-adapters"`). Available for in-process Spyre debugging, but not the primary comparison path.
- **CpuAdapter** - loads the model via HuggingFace Transformers and runs on CPU (`backend: "cpu"`). **Development/debugging only** - CPU inference is too slow for regular use and GPU-vLLM is the correct reference platform. Retained for local development convenience.

All runs, regardless of model type or backend, write to the same `results/` directory and append one entry to a shared `results_index.json` manifest. The dashboard reads this manifest to discover all available runs without scanning the filesystem. The framework ships a single Python entry point and a single YAML configuration file — these are the only two files an engineer needs to interact with to run an evaluation.

### 3.2 Entry Point and Configuration

All runs go through a single dispatcher:

```bash
python quality_eval_pipeline.py --config eval_config.yaml
```

`eval_config.yaml` is the only file that needs to be edited before a run. It has three sections: required fields, decoder model settings (generative) and encoder model settings (embedding).

**Required fields:**

```yaml
model: "ibm-granite/granite-3.3-8b-instruct"
model_type: "decoder"   # or "encoder"
```

**Decoder model settings:**

```yaml
decoder:
  backend: "vllm-spyre"     # "cpu" | "hf-adapters" | "vllm-spyre" | "vllm-gpu"
  base_url: "http://localhost:8000"

  datasets:
    - name: "squad"
      metrics: ["exact_match", "rouge", "bleu", "meteor"]
      num_samples: 200
      max_new_tokens: 50
    - name: "triviaqa"
      metrics: ["exact_match", "rouge", "bleu", "meteor"]
      num_samples: 200
      max_new_tokens: 50
    - name: "xsum"
      metrics: ["rouge", "bleu", "meteor"]
      num_samples: 200
      max_new_tokens: 80
    - name: "gsm8k"
      metrics: ["exact_match", "rouge", "bleu"]
      num_samples: 200
      max_new_tokens: 200
    - name: "wmt"
      metrics: ["bleu", "meteor", "rouge"]
      num_samples: 200
      max_new_tokens: 100
      src_lang: "de"
      tgt_lang: "en"

  max_new_tokens: 100
  max_input_tokens: 256     # spyre backend only
  seed: 42
  dtype: null
  warmup: true

  lm_tasks: ["mmlu", "arc_easy", "hellaswag"]
  lm_backend: "local-completions"   # "cpu" | "spyre" | "local-completions"
  lm_base_url: "http://localhost:8000/v1"
  lm_num_fewshot: 5
  lm_limit: 200
  lm_output_suffix: "lmeval"
```

**max_input_tokens (Spyre direct backend only):** When backend: "hf-adapters", the SpyreAdapter truncates every prompt to `max_input_tokens` tokens and left-pads the entire batch to exactly that length. This ensures Spyre compiles a single graph shape for the whole run, avoiding recompilation across batches with different sequence lengths. Prompts longer than `max_input_tokens` are silently truncated from the left; the corresponding attention mask is set to 0 for all padding positions so truncated tokens do not influence generation. This field has no effect when backend: "cpu", backend: "vllm-spyre", or backend: "vllm-gpu" — for the vLLM backends, sequence length is managed by the server (via --max-model-len).

**backend: "vllm-spyre" prerequisite:** The vLLM Spyre backend requires a running vLLM server before the pipeline is started. The server uses Spyre internally (via torch_spyre) but the eval process itself communicates with it purely over HTTP — no torch or hf_adapters imports in the eval process. The pipeline polls `/v1/models` with retries until the server responds before running any evaluation.

**lm_backend: "local-completions":** When using the vLLM backend, lm-eval benchmarks can also be routed to the same vLLM server by setting lm_backend: "local-completions" and lm_base_url to the server's /v1 endpoint. This uses lm-eval's native OpenAI-compatible HTTP backend — no model is loaded locally, and the Spyre device is not touched by the eval process. This avoids the device-conflict error that occurs when lm_backend: "spyre" is used while a vLLM server already holds the Spyre VFIO device.

**Encoder model settings** (set `model_type: "encoder"` in the required fields above, then):

```yaml
encoder:
  backend: "vllm-spyre"     # "local" (sentence-transformers, in-process) | "vllm-spyre" | "vllm-gpu"
  base_url: "http://localhost:8000"
  tasks:
    - "NFCorpus"
    - "SciFact"
    - "STS12"
    - "STSBenchmark"
    - "Banking77Classification"
    - "ArxivClusteringS2S"
  batch_size: 32
  device: null
  normalize_embeddings: false
  output_folder: "mteb_results"
```

CLI overrides are available without editing the file:

```bash
python quality_eval_pipeline.py --config eval_config.yaml \
    --set decoder.backend=vllm-spyre \
    --set decoder.max_new_tokens=100
```

### 3.3 Generation Quality Pipeline

The generation pipeline has five sequential stages as seen in the architecture diagram above:

#### Stage 1 — Dataset loading

The dataset loader loads and caches N samples from any registered dataset. Samples are shuffled with `seed=42` and cached locally — the same N samples are selected every run for reproducibility. Each sample is a dict with `"input"` and `"reference"` keys.

**Supported datasets:**

| Name | Source | Task |
|---|---|---|
| `xsum` | HuggingFace `EdinburghNLP/xsum` | One-sentence summarization |
| `squad` | HuggingFace `rajpurkar/squad` | Reading-comprehension QA |
| `triviaqa` | HuggingFace `trivia_qa/rc` | Open-domain QA |
| `gsm8k` | HuggingFace `openai/gsm8k` | Math word problems |
| `wmt` | HuggingFace `wmt14/de-en` | Translation |
| `custom_csv` | Local CSV | Any |
| `custom_jsonl` | Local JSONL | Any |
| `prompts_txt` | Local TXT | Any |

New datasets can be added by implementing a loader function and registering it in the dataset registry — no changes to the pipeline or config schema are required.

#### Stage 2 — Prompt building

The prompt builder formats each raw sample into a `(prompt, reference)` pair. Each task has a dedicated builder (e.g. `SummarizationBuilder`, `QABuilder`, `TranslationBuilder`). Builders are resolved by dataset key and expose a `default_metrics` list.

**Note on prompt sensitivity:** Different models may respond differently to the same prompt format. This is partly mitigated by the vLLM backend routing requests through `/v1/chat/completions`, which applies each model's own chat template automatically. The prompt builder layer is also extensible — a model-specific builder can be registered per dataset key without any pipeline changes. Model-specific prompt template support is planned as a follow-up improvement.

#### Stage 3 — Model inference

The model adapter layer defines an abstract `ModelAdapter` interface. Four backends are registered:

- **`VLLMAdapter` (`vllm-spyre`)** — connects to a vLLM server running on IBM Spyre. Generation requests go to `/v1/chat/completions`; log-likelihood scoring uses `/v1/completions` with `echo=True` and `logprobs`. **Primary Spyre evaluation path.**
- **`VLLMAdapter` (`vllm-gpu`)** — same HTTP adapter, pointed at a vLLM server running on GPU. **Reference platform** for Spyre-vs-GPU comparisons. Using the same vLLM version and config on both sides ensures any quality delta is attributable to the backend hardware, not to differences in request/response processing.
- **`SpyreAdapter` (`hf-adapters`)** — uses `hf_adapters.AutoSpyreModelForCausalLM` directly in the eval process. See `max_input_tokens` in Section 3.2 for prompt truncation, batch padding, and graph-shape constraints. Available for in-process Spyre debugging.
- **`CpuAdapter` (`cpu`)** — uses `AutoModelForCausalLM` from HuggingFace Transformers. **Development/debugging only** — not suitable for regular quality comparisons due to speed; use GPU-vLLM as the reference instead.

#### Stage 4 — Metric computation

The metric computation module dispatches to the appropriate scoring library:

| Metric key | Library | Scale |
|---|---|---|
| `rouge` | `rouge-score` (Google) | 0–1 |
| `bleu` | `sacrebleu` (ACL standard) | 0–100 |
| `meteor` | `nltk` | 0–1 |
| `bertscore` | `bert-score` (Zhang et al. 2020) | 0–1 |
| `exact_match` | built-in normalizer | 0 or 1 |

All metrics are computed with standard open-source libraries.

#### Stage 5 — Result serialization

The result serializer writes results to JSON. Every output file follows a fixed schema:

```json
{
  "model": "<model-id>",
  "model_type": "decoder",
  "task_category": "qa",
  "metrics": { "rouge1": 0.412, "exact_match": 0.35 },
  "metric_metadata": {
    "rouge1":      { "scale": "0–1", "higher_is_better": true },
    "exact_match": { "scale": "0–1", "higher_is_better": true }
  },
  "metadata": {
    "backend": "vllm-spyre",
    "dataset": "squad",
    "num_samples": 200,
    "avg_latency_s": 0.43,
    "avg_throughput_tok_s": 112.4,
    "load_time_s": 4.2,
    "total_inference_time_s": 86.0
  }
}
```

The `metric_metadata` block is machine-readable — it tells any downstream consumer whether higher is better and what the scale is, without hardcoding that knowledge in the dashboard. The `scale` field can be used directly by a dashboard to display metric values with context — for example as a label (`0.412 / 1.0`), a progress bar filled to `value / max`, or to normalize metrics onto a common axis for side-by-side comparison charts.

### 3.4 lm-eval Harness Integration

The lm-eval harness module uses `lm-evaluation-harness` (EleutherAI) for standardized academic benchmarks. The framework requires `lm-eval>=0.4` and has been validated against `lm-eval==0.4.8`. Task names follow the lm-eval v0.4 convention: "mmlu"` runs the full 57-subject aggregated benchmark; individual subjects (e.g. "mmlu_philosophy", "mmlu_clinical_knowledge") can also be specified directly to run a targeted subset. Other supported tasks include "arc_easy", "arc_challenge", "hellaswag", "gsm8k", and "truthfulqa_mc1".

It supports three backends:

- **`hf-adapters`** — routes through a `SpyreLM` adapter which wraps `AutoSpyreModelForCausalLM` and exposes the `loglikelihood` and `generate_until` interfaces required by lm-eval.
- **`cpu`** (or `huggingface`) — standard `HFLM` adapter with `device=cpu` and `dtype=bfloat16`, correct on any machine without Spyre.
- **`local-completions`** — lm-eval's native OpenAI-compatible HTTP backend. Points directly at a running vLLM server via `lm_base_url`; no model is loaded locally and the Spyre device is not accessed by the eval process. Use this when the quality eval is already running against a vLLM backend, so that lm-eval benchmarks are routed to the same server without a second device claim.

This path can run standalone or alongside quality metrics in a single pipeline invocation as well.

### 3.5 Embedding Evaluation

The embedding runner wraps the `mteb` library to evaluate any HuggingFace sentence-embedding or bi-encoder model. It runs across all MTEB task types (Retrieval, STS, Classification, Clustering, Reranking, BitextMining) and emits the same top-level JSON schema as the generative framework, with `model_type: "encoder"` and task-appropriate metrics (nDCG@10, Spearman ρ, V-measure, etc.).

Two backends are supported:

- **local** (default) — loads the model in-process via `sentence-transformers` (`SentenceTransformer`) and runs inference locally.
- **vllm-spyre** / **vllm-gpu** — delegates all encoding to a running vLLM server via `/v1/embeddings`. The `VLLMEmbeddingModel` class implements the MTEB `EncoderProtocol` interface over HTTP, with no torch or sentence-transformers dependency in the eval process. Set `encoder.backend: "vllm-spyre"` or `encoder.backend: "vllm-gpu"` and `encoder.base_url` in the config (see Section 3.2).

**Backend compatibility note:** Results from the `local` and `vllm` backends for the same model may not always be directly comparable - vLLM uses its own tokenization and pooling logic which may differ from `sentence-transformers`. Additionally, not all encoder models are supported in vLLM; in such cases only the `local` backend will be available and cross-backend comparison will not be possible.

### 3.6 Results Manifest and Index

The results index module maintains an append-only `results_index.json` manifest. Every completed run adds one entry. The dashboard reads this file to discover all available results without scanning the filesystem.

**Concurrency constraint:** The index writer uses a POSIX advisory file lock (`fcntl.flock`) around the read-modify-write cycle so that parallel evaluation jobs running on the same machine do not corrupt the manifest. Cross-machine concurrent writes (e.g. two CI jobs on different nodes writing to a shared NFS path) are not supported — CI pipelines must serialize evaluation jobs or use separate result directories and merge afterwards.

The index entry shape is:

```json
{
  "timestamp": "2025-09-01T19:00:00Z",
  "model": "ibm-granite/granite-3.3-8b-instruct",
  "model_type": "decoder",
  "task_category": "summarization",
  "dataset": "cnn_dailymail",
  "backend": "vllm-spyre",
  "num_samples": 200,
  "metrics": { "rouge1": 0.412, "bleu_4": 12.3 },
  "result_file": "results/granite-3.3-8b-instruct_spyre_cnn_dailymail.json",
  "status": "ok",
  "error": null
}
```

---

## 4. Dependencies

| Category | Package | Purpose | Scope |
|---|---|---|---|
| **Core** | `torch`, `pyyaml`, `datasets` | Tensor ops, config parsing, HuggingFace dataset loading | `cpu` and `spyre` backends |
| **Generative Eval** | `transformers` | `AutoModelForCausalLM`, tokenizers | Decoder runs |
| | `rouge-score`, `sacrebleu`, `nltk` | Standard quality metrics (ROUGE, BLEU, METEOR) | `rouge`, `bleu`, `meteor` |
| | `bert-score` | Semantic similarity metric | Optional (`bertscore`) |
| **Benchmark Harness** | `lm-eval>=0.4` | EleutherAI benchmark harness (MMLU, ARC) | Academic benchmarks |
| **Embedding Eval** | `mteb>=1.1`, `sentence-transformers>=2.7` | MTEB benchmark suite & embedding models (validated against `mteb==1.7.8`, `sentence-transformers==3.0.1`) | Encoder `local` backend only |
| **Hardware Backend** | `torch_spyre`, `hf_adapters` | IBM Spyre hardware acceleration & model loading | `spyre` backend only |
| **vLLM Backend** | `vllm` (external server process) | OpenAI-compatible inference server; runs on Spyre via `torch_spyre` internally | `vllm` backend only |

All packages except `torch_spyre`, `hf_adapters`, and `vllm` are publicly available on PyPI. `torch_spyre` is the low-level IBM Spyre hardware backend (lives in the `torch-spyre` repo). `hf_adapters` is the HuggingFace-compatible model loading layer for Spyre (lives in the `hf-adapters` repo). Both must be installed from their respective internal repos before running Spyre-backend evaluations. `vllm` is installed in the `torch-spyre` venv on the Spyre machine and runs as a separate server process — the eval code itself has no Python dependency on vLLM and communicates with it purely over HTTP using only the standard library (`urllib`).

---

## 5. Correctness and Reproducibility

### 5.1 Sampling Reproducibility

All dataset loaders shuffle with a fixed `seed=42` and cache the selected sample indices locally. Every re-run with the same config selects exactly the same N samples, making runs comparable across model versions, backends and dates.

### 5.2 Error Isolation

The multi-dataset loop catches exceptions at the per-dataset level. A dataset load failure or metric computation error does **not** abort the run - it is logged, registered as `status: "error"` in the index, and the remaining datasets continue. This ensures a transient HuggingFace Hub timeout or a missing optional dependency (`bert-score`) does not cancel an otherwise valid run.

### 5.3 Metric Fidelity

No custom scoring logic is introduced in the metric computation layer. Every metric delegates to the canonical library:

- ROUGE → Google `rouge-score`
- BLEU → `sacrebleu` (the ACL standard implementation)
- METEOR → `nltk`
- BERTScore → `bert-score` (Zhang et al. 2020, `roberta-large` by default)
- lm-eval accuracy → EleutherAI `lm-evaluation-harness`
- MTEB metrics → `mteb` library

This means metric computation is directly comparable to published numbers that use the same libraries. However, this does not extend to output parsing - for lm-eval benchmarks, different models may require different output parsers to correctly extract answers from generated responses. lm-eval handles this per task, but some models may need additional parsing adjustments to avoid artificially low scores. This is a known challenge when evaluating a variety of models across diverse datasets.

### 5.4 Framework Validation and Reference Outputs

The following results from Spyre runs are provided for reference to demonstrate the framework's output format and metric reporting.

#### 1. Decoder Model: `google/gemma-4-26B-A4B-it` - vLLM backend on IBM Spyre

| Evaluation Stream | Dataset / Task | Samples | Metric | Value |
|---|---|---|---|---|
| **Free-form Generation** | SQuAD | 200 | Exact Match | 0.5850 |
| | | | ROUGE-1 | 0.8266 |
| | | | ROUGE-L | 0.8248 |
| | | | BLEU-4 | 38.99 |
| | | | METEOR | 0.5715 |
| | TriviaQA | 200 | Exact Match | 0.4850 |
| | | | ROUGE-1 | 0.6216 |
| | | | METEOR | 0.4342 |
| | XSum | 200 | ROUGE-1 | 0.2736 |
| | | | ROUGE-L | 0.2003 |
| | | | METEOR | 0.2144 |
| | GSM8K | 200 | Exact Match | 0.5050 |
| **Academic Benchmarks** | ARC Easy | 200 | Accuracy (norm) | 0.6250 |
| | ARC Challenge | 200 | Accuracy (norm) | 0.4250 |
| | HellaSwag | 200 | Accuracy (norm) | 0.5150 |
| | TruthfulQA MC1 | 200 | Accuracy | 0.3250 |
| | MMLU (7 subjects) | 200 each | Mean Accuracy | 0.5485 |

> **Note:** Academic benchmarks use `lm_backend: "local-completions"` routed to the same vLLM server (`lm_limit: 200` per task, zero-shot). MMLU covers 7 subjects (abstract algebra, college mathematics, college physics, clinical knowledge, professional law, moral scenarios, philosophy) — not the full 57-subject suite.

#### 2. Encoder Model: `ibm-granite/granite-embedding-278m-multilingual` - vLLM backend

| Task Category | Evaluated Tasks | Primary Metric | Value |
|---|---|---|---|
| **Retrieval** | NFCorpus, SciFact | Mean nDCG@10 | 0.4616 |
| **STS** | STS12, STSBenchmark | Mean Spearman ρ | 0.7677 |
| **Classification** | Banking77Classification | Mean Accuracy | 0.7806 |
| **Clustering** | ArxivClusteringS2S | Mean V-measure | 0.3290 |

#### 3. Output Schema Verification

The pipeline writes consistent, self-describing JSON records ingested by the shared results index:

**Sample Decoder Result Snippet:**
```json
{
  "model": "google/gemma-4-26B-A4B-it",
  "model_type": "decoder",
  "task_category": "qa",
  "backend": "vllm-spyre",
  "dataset": "squad",
  "metrics": {
    "exact_match": 0.585,
    "rouge1": 0.8266,
    "rougeL": 0.8248,
    "bleu_4": 38.99,
    "meteor": 0.5715
  },
  "metric_metadata": {
    "exact_match": { "scale": "0–1", "higher_is_better": true },
    "rouge1":      { "scale": "0–1", "higher_is_better": true },
    "rougeL":      { "scale": "0–1", "higher_is_better": true },
    "bleu_4":      { "scale": "0–100", "higher_is_better": true },
    "meteor":      { "scale": "0–1", "higher_is_better": true }
  },
  "metadata": { "num_samples": 200, "seed": 42, "backend": "vllm-spyre" }
}
```

**Sample Encoder Result Snippet:**
```json
{
  "model": "ibm-granite/granite-embedding-278m-multilingual",
  "model_type": "encoder",
  "task_category": "retrieval",
  "metrics": {
    "mean_ndcg_at_10_retrieval": 0.4616,
    "mean_spearman_sts": 0.7677,
    "mean_accuracy_classification": 0.7806,
    "mean_v_measure_clustering": 0.329
  },
  "metric_metadata": {
    "ndcg_at_10": { "scale": "0–1", "higher_is_better": true },
    "spearman":   { "scale": "-1–1", "higher_is_better": true }
  },
  "metadata": { "tasks": ["NFCorpus", "MSMARCO", "STS12", "STSBenchmark", "Banking77Classification", "ArxivClusteringS2S"], "backend": "vllm-spyre" }
}
```

---

## 6. Design Decisions

### 6.1 Single YAML Entry Point

**Decision:** All configuration lives in one YAML file; the pipeline reads `model_type` and dispatches accordingly.

**Rationale:** Eliminates per-model bespoke scripts. Any engineer can run a full evaluation by editing one file and running one command. CLI `--set` overrides allow quick one-off changes without touching the file.

### 6.2 Metric Metadata in JSON Output

**Decision:** Every result file carries a `metric_metadata` block encoding `scale` and `higher_is_better` for each computed metric.

**Rationale:** If this is only in the UI, it becomes a presentation detail that can get out of sync with the framework. Encoding it in the JSON makes it machine-readable: any future dashboard, CLI tool, or automated regression system gets the semantics for free without hardcoding them. The `scale` field in particular enables the dashboard to display metric values with context — as a labeled range, a normalized progress bar, or a common-axis comparison chart — without needing a per-metric lookup table.

### 6.3 Two Complementary Evaluation Paradigms

**Decision:** The framework runs both free-form generation quality and log-likelihood benchmark accuracy, optionally in a single command.

**Rationale:** These answer fundamentally different questions. Generation quality (ROUGE, BLEU) tests whether the model produces accurate, fluent output for real tasks. Benchmark accuracy (MMLU, ARC) tests knowledge and reasoning using the same methodology as published leaderboards. Neither subsumes the other — a model that scores well on MMLU can still degrade on summarization on Spyre if quantization affects generation quality, and vice versa.

---

## 7. Relationship to Existing Tooling

**PELE** evaluates ~5 models via HTTP against a running `spyre-inference` instance using a composite PeleScore (60% semantic + 30% grammar + 10% exact match). It tests the **serving stack end-to-end**. This framework tests the **model quality directly** — via a vLLM server (primary Spyre path), via `hf_adapters` (direct in-process Spyre path), or via HuggingFace Transformers (CPU baseline) — independently of the `spyre-inference` serving stack. The two are complementary, not overlapping.

The **[E2E Performance Testing RFC](https://github.com/torch-spyre/RFCs/blob/main/1633-E2EModelPerf/1633-E2EModelPerf.md)** notes that quality evals may be redundant if correctness tests already confirm matching output against a CPU reference. This framework complements that view: correctness tests are a fast regression gate on token-level agreement, while quality evals measure task-level fitness at scale — catching degradation from padding, truncation or generation parameter differences that token-matching alone may not surface. Both are valuable and intended to be run together.

---

## 8. Alternative Approaches

### OLMES (AllenAI)

**Approach:** Use OLMES as the standardization layer on top of lm-eval.

**Advantages:**
- Reproducible prompt formats (5 canonical variants per task)
- Decontamination checking
- Leaderboard-compatible configs

**Disadvantages:**
- Adds an external dependency with enforced prompt formats
- OLMES uses lm-eval under the hood for task orchestration — our current lm-eval integration already wires lm-eval directly to the Spyre/vLLM backend without duplicating any lm-eval logic. Adding OLMES would not reduce what we build; it would add a layer on top
- Its main value-add — canonical 5-variant prompt formats and leaderboard-compatible configs — is not needed for cross-platform consistency comparisons. What matters is that we use the **same prompts and config on both platforms**, which our current setup already guarantees
- Only relevant if the goal becomes publishing numbers against external leaderboards, which is not our goal
- Does not add value for custom dataset evaluation paths which are not covered by OLMES

**Conclusion:** OLMES is worth revisiting if the goal becomes publishing benchmark numbers against public leaderboards. For Spyre platform correctness validation (Spyre-vLLM vs GPU-vLLM), the current lm-eval-direct integration is sufficient and avoids an unnecessary dependency.

### DeepEval

**Approach:** Use DeepEval as the evaluation layer, leveraging its LLM-as-judge metrics (hallucination detection, answer relevancy, faithfulness, contextual precision).

**Advantages:**
- Rich LLM-as-judge metrics beyond n-gram overlap
- Active open-source community with a broad metric library
- Supports custom metrics and CI integration

**Disadvantages:**
- LLM-as-judge metrics require a separate judge model, adding infra overhead and cost
- Not designed for direct Spyre/HuggingFace backend integration — would require a wrapper
- Overkill for the current use case of Spyre-vs-CPU quality comparison using standard reference-based metrics

**Conclusion:** Worth considering if the evaluation scope expands to include RAG pipelines or conversational quality assessment where reference-based metrics are insufficient.

### AWS fmeval

**Approach:** Use Amazon's `fmeval` library for quality evaluation, which supports ROUGE, METEOR, BERTScore, exact match, and toxicity/stereotyping checks.

**Advantages:**
- Covers the same reference-based metrics as this framework
- Includes built-in toxicity and prompt stereotyping checks
- Well-documented with support for custom datasets

**Disadvantages:**
- AWS-centric design — not straightforward to integrate with `hf_adapters` or Spyre backends
- Adds an external managed dependency with its own data format requirements
- No academic benchmark support (MMLU, ARC) — would still require lm-eval alongside it

**Conclusion:** Metric coverage overlaps significantly with this framework but the integration cost with Spyre is high. No compelling reason to adopt it over the current approach.

---

## 9. Future Work

- **Multimodal evaluation** — the framework currently covers text-in/text-out decoder models and text embedding models. VLMs (vision-language models) are a gap. The vLLM backend already supports `/v1/chat/completions` with image content blocks for VLMs; adding multimodal eval requires: (a) a multimodal dataset loader (e.g. `TextVQA`, `MMMU`), (b) a prompt builder that encodes image+text inputs in the OpenAI vision message format, and (c) extending the result schema for multimodal task metrics.
- **Tool-calling / function-calling evaluation** — extend the prompt builder and dataset layer to test structured output and special token handling for tool-use scenarios (e.g. using `ToolBench` or a custom tool-call dataset).
- **Multi-turn evaluation** — extend the prompt builder and dataset layer to support MT-Bench and AlpacaEval for instruction-following quality assessment.
- **Scenario coverage expansion** — add explicit long-context datasets (e.g. `SCROLLS`, `LongBench`) and special-character/multilingual stress sets to systematically cover the inference scenario matrix in Section 2.
- **OLMES integration** — only if the goal becomes publishing benchmark numbers against public leaderboards, which is not the current objective.

---

