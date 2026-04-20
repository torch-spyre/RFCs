# E2E Performance Testing

**Authors:**

- Romit Jain (IBM Research, India)
- Yashasvi Chaurasia (IBM Research, India)
- Ashok Pon Kumar Sree Prakash (IBM Research, India)

## **Summary**

We want to measure the **performance** of models running on Spyre hardware. Performance can be measured in 3 different ways

1. Correctness: Is the model output correct? Correctness is judged by verifying if the model outputs similar tokens as compared to when running on GPU.
2. Quality Evals: Is the model of high quality? This is judged by running various benchmarks like GSM8K, MMLU etc. or usecase specific evals.
3. Benchmarking: Is the model fast? This is judged by measuring metrics like throughput, ITL, TTFT, latency etc.

## **Proposed Implementation**

### Correctness

**Goal:**
Verify that vLLM on Spyre produces the same output as the reference HuggingFace model on CPU/GPU.

**Why this matters:**
Op-level tests catch per-op drift, but only end-to-end comparison catches issues in how ops compose across a full model forward pass.

**What upstream vLLM already provides:**

- `HfRunner` — wraps HuggingFace transformers to generate reference outputs (`tests/conftest.py`)
- `VLLMRunner` — wraps vLLM's `LLM` class to generate outputs under test (`tests/conftest.py`)
- Comparison utilities for greedy output matching and logprob tolerance checks

**What we need to build:**

- Wire `HfRunner` to target the CPU backend (device config, model selection)
- Define the model x prompt matrix (adopt upstream configs and extend them)
- Integrate into CI with pass/fail based on output match and tolerance thresholds
- Optionally, we can cache the results of `HFRunner` instead of running it every time

The tests will reside inside `spyre-inference` repository and will be managed via GHA.

### Benchmarking

**Goal:**
Measure latency and throughput of vLLM on Spyre, and detect regressions over time.

**Why this matters:**
Without baselines, performance regressions are invisible until users report them. We also want our stack to be performant and not leave any performance on the table.

**What upstream vLLM already provides:**

- `vllm bench` CLI with subcommands: `throughput`, `latency`, `serve`, `startup`
- Dataset abstractions (ShareGPT, random, sonnet) for reproducible workloads
- Async HTTP client for serving benchmarks against the OpenAI-compatible API
- Structured output (latency percentiles, tokens/sec, TTFT, ITL)

**What we need to build:**

- Benchmark configs for Spyre-supported models (model, dtype, sequence lengths)
- Infra for recording and storing benchmark results
- CI integration or periodic job that flags regressions beyond a threshold
- A central dashboard reporting the benchmarking numbers
- vLLM bench does not support memory resource tracking as of now, we will have to contribute upstream to enable this

The tests will reside inside `spyre-inference` repository and will be managed via GHA.

### Quality Evals

**Goal:**
Measure the output quality of the model and its capability to perform a task. There are two types of quality evals

1. Standard evals: GSM8K, AIME, MMLU etc.
2. Use case specific evals

**Why this matters:**
A natural question is whether we should also run quality benchmarks (GSM8K, MMLU, GPQA) to verify model accuracy on Spyre. Quality evals test the _model_, not the _backend_. If a model scores 80% on GSM8K on CPU, and our correctness tests confirm Spyre produces matching output, then Spyre inherits that 80%. Running GSM8K or any eval again on Spyre would be redundant verification.

There is an explicit reason to do quality evals, if and only if, correctness check has a noticeable degradation. In that case, we would have no way to judge if the output from Spyre is indeed correct or not.

Apart from this, we may need to perform use-case specific evals, where we may be testing specific prompts, agents with tool calls, document retrieval, etc., when the use case is being delivered on Spyre. Use case specific evals should not devolve down to testing a model on a known data set, which can be reproduced with upstream tests.

**What upstream vLLM already provides:**

- vLLM is compatible with a popular evaluation framework, `lm-evaluation-harness`. Essentially, `lm-evaluation-harness` can run the evals against a running vLLM instance
- `lm-evaluation-harness` is a tool used to run various evals for language models
- `lm-evaluation-harness` provides support for custom datasets and custom evaluation metrics.

**What we need to build:**

Using an existing open source framework would ensure that we inherit the best practices from the community while performing evals for specific usecases.

- Extend support for custom datasets and custom evaluation metrics in `lm-evaluation-harness`
- CI/CD pipelines for running these tests periodically
- Infra for saving results and a dashboard for reporting metrics and regression

The tests will reside inside `spyre-inference` repository and will be managed via GHA.

## Current stack approach

- **PELE**: Deploys spyre-inference on Spyre hardware via K8s, sends prompts over HTTP (standard OpenAI API), compares Spyre outputs against committed GPU baselines using PeleScore (60% semantic + 30% grammar + 10% exact match). Covers ~5 models at 1K-32K context lengths. Covers both correctness and quality evals.
- **zspyre-test-framework**: This framework supports benchmarking tests for encoder and decoder models in addition to correctness tests.
- **fmwork**: Wrapper over vllm bench to run benchmarking tests.
- **OLMES**: Qualtiy evals for multiple different scenarios, including: needle in the haystack, ruler, GSM8K, long cotenxt.
- **BFCL**: Quality evals for tool calling use case. Based on open source datasets
- **WXA4Z**: Quality evals for agentic use case. Based on internal data (?).

## Conclusion

We want to run all the performance tests (correctness, quality, benchmarking) using vLLM as the backend and build utilities around them. We want to maximize upstream usage wherever applicable (`vllm bench`, `lm-evaluation-harness`, `HFRunner`, `VLLMRunner`) so that we are inheriting best practices from upstream.
