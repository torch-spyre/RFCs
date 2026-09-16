# Graph recorder for static `torch.compile` in spyre-inference

**Authors:**
* @R3hankhan123
* @bringlein
* @rishikakedia
* @bohnstingl
* @tdoublep
* @dilipgb

> **Status:** Implemented in spyre-inference
> ([#480](https://github.com/torch-spyre/spyre-inference/pull/480),
> [#638](https://github.com/torch-spyre/spyre-inference/pull/638)).
> Closes [spyre-inference#174](https://github.com/torch-spyre/spyre-inference/issues/174).
>
> **Code:** `spyre_inference/v1/worker/spyre_shape_bucketer.py`,
> `spyre_inference/v1/worker/spyre_model_runner.py`,
> `spyre_inference/platform.py`

---

## **Summary**

Spyre's Inductor backend requires static shapes (`dynamic=False`). vLLM CUDA
graphs are the GPU way to pin those shapes; Spyre has none. The graph recorder
is that analogue for `STOCK_TORCH_COMPILE`, for the **decoder body** (not
attention, not pooling):

* Dummy each body token-count bucket at warmup.
* Pad every runtime batch up onto a warmed bucket via vLLM's
  `BatchDescriptor` / `_determine_batch_execution_and_padding` seam.
* Dummy lm_head row widths separately (one row per sampled request).

The default decoder ladder is sparse: powers of two up to `--max-num-seqs`
(decode), plus one prefill bucket at 512.

## **Motivation**

[#174](https://github.com/torch-spyre/spyre-inference/issues/174) asked for
the CUDA-graph analogue on Spyre: compile complete forward passes for buckets
of request lengths, in the model runner, while paged attention is a separate
path.

Dynamo specializes a new graph on every unseen packed token count `T` and
every lm_head row width. Without a recorder, the first request at a new shape
pays a full compile (tens of seconds) on the serving path and often times out
`execute_model`.

vLLM's CUDA-graph capture dummies each capture size at startup, then pads
runtime batches onto a captured graph. Spyre sets `use_cuda_graph = False`.
The recorder is the same contract for `STOCK_TORCH_COMPILE`: compile once per
bucket at warmup, pad at runtime, never compile mid-request.

Users and developers should think of `compile_sizes` the way CUDA-graph
capture sizes work on GPU: the list is the set of legal body shapes. Anything
the scheduler emits is rounded up onto that list. `--enforce-eager` turns the
recorder off.

## **Proposed Implementation**

### Constraints

* **Compilation mode.** Only `NONE` (`--enforce-eager`) and
  `STOCK_TORCH_COMPILE` are supported. `VLLM_COMPILE` / piecewise CUDA graphs
  are rejected.
* **Packed `T` is the body shape.** With chunked prefill, one forward is
  `[prefill tokens] + [decode tokens]`. Linear / RMSNorm see one flattened
  `[T, H]`. There is no separate decode graph vs prefill graph at the body.
* **Scheduler cap.** After buckets are chosen,
  `max_num_batched_tokens = max(compile_sizes)` so the scheduler cannot emit a
  `T` with no bucket.
* **Decoder only.** Attention kernels and pooling / encoder warmup are
  separate. This RFC is the decoder-body recorder, matching #174.

### Recorders

Two recorders. Each pads *up* to the smallest warmed size that fits; sampling
drops pad.

```
                    scheduler batch
                           │
                           ▼
              ┌────────────────────────┐
              │  1D body T             │  SpyreShapeBucketer
              │  compile_sizes         │  BatchDescriptor pad
              └───────────┬────────────┘
                          │
              ┌───────────▼────────────┐
              │  lm_head rows          │  logits_row_buckets
              │  ≤ max_num_seqs        │  pad, then unpad
              └────────────────────────┘
```

### Body recorder (`SpyreShapeBucketer`)

`TorchSpyrePlatform.apply_config_platform_defaults` fills
`compilation_config.compile_sizes` unless the user already set them.

**Default ladder** (sparse, compile-time first):

* Powers of two from `1` up to `--max-num-seqs` (and `max_num_seqs` itself if
  it is not a power of two). Decode-only steps schedule one token per running
  sequence, so this ladder is exact for pure decode.
* One prefill bucket at 512 (same as sendnn-inference). Full prefill chunks
  fill that budget.

Examples:

| `--max-num-seqs` | `compile_sizes` |
|---|---|
| 4 | `[1, 2, 4, 512]` |
| 6 | `[1, 2, 4, 6, 512]` |

Warmup dummies each body size largest-first (Inductor cache hits on later
smaller shapes), then `mark_warmed_up()`. Runtime padding overrides
GPUModelRunner's `_determine_batch_execution_and_padding` and returns
`CUDAGraphMode.NONE` plus `BatchDescriptor(num_tokens=padded)`. That is the
CUDA-graph seam; we do not mutate
`scheduler_output.total_num_scheduled_tokens`. Pad tokens stay out of
sampling via `num_actual_tokens`.

`--enforce-eager` skips the bucketer.

### lm_head row recorder

The lm_head sits outside every body graph and projects one row per *sampled*
request. A finishing batch walks `1..max_num_seqs`. Those widths pad onto
`logits_row_buckets` (body sizes clipped to `max_num_seqs`); warmup runs
`_dummy_sampler_run` at each width. Pad rows are dropped before sampling. The
prefill body bucket (`512`) is not a reachable row count and is not recorded
for the head.

### Compile granularity

Default `SPYRE_COMPILE_GRANULARITY=block`: each transformer block is
`torch.compile`'d in place (`dynamic=False`). Identical `forward` code objects
share one Inductor artifact (layer 0 still specializes on `residual is None`).
A new `T` therefore recompiles a block, not a whole model.
`SPYRE_COMPILE_GRANULARITY=model` restores a single full-model graph.

### Mixed batches

The sparse body ladder is cheap on the two common cases:

* **Decode-only:** `T == num_running_seqs` → pad to the next power of two
  (3 → 4).
* **Full prefill chunks:** the scheduler fills the 512-token budget → hit the
  prefill graph with no extra pad.

A mixed step has no bucket between `max_num_seqs` and 512. Example:
`{128 prefill + 1 decode} = 129` pads to 512. A denser user list is still
honored:

```bash
vllm serve ... -cc '{"compile_sizes": [1, 2, 4, 64, 128, 256, 512]}'
```

Do not split "decode graphs" and "prefill graphs" as two compilers. The body
only sees `T`. Separate ladders would still pad mixed `T` onto the next
available size — with a hole, that size is 512.

### Configuration

| Knob | Role |
|---|---|
| `--enforce-eager` | Disable compile and decoder recording |
| `-cc.compile_sizes` | Override body buckets; also caps `max_num_batched_tokens` |
| `SPYRE_COMPILE_GRANULARITY` | `block` (default) or `model` |

### Tests

* `tests/runtime/test_platform.py` — default ladders, user `compile_sizes`,
  eager skip
* `tests/runtime/test_shape_bucketer.py` — 1D dispatch, `logits_row_buckets`
* `tests/e2e/test_compile.py` — compiled decoder cosine checks

## **Metrics**

* Warmup wall time vs number of `compile_sizes` (startup cost).
* Zero mid-request Inductor compiles on a covered workload (correctness of
  the ladder + scheduler cap).
* Decode ITL at `T == num_seqs` (no extra pad on the decode-only path).
* Mixed-batch body time vs a denser ladder (padding tax of the 512 hole).

## **Drawbacks**

* **Not a breaking API change**, but it does cap
  `max_num_batched_tokens` to `max(compile_sizes)`. A user who expected a
  larger scheduler budget silently gets 512 (or their own max bucket).
* **UX:** first boot is still blocked on warmup. Restarts are cheap only if
  an Inductor / kernel cache is configured.
* **Runtime pad:** mixed chunked-prefill + decode steps in
  `(max_num_seqs, 512)` pad to 512. That is extra Linear / RMSNorm work on
  the common continuous-batching path.
* **Complexity:** a bucketer, a platform default, a
  `_determine_batch_execution_and_padding` override, and a second row
  recorder for lm_head. Wrong `is_warmed_up` gating compiles during dummy
  and misses the warmed graph at runtime.

## **Alternatives**

| Approach | Why not |
|---|---|
| `VLLM_COMPILE` + CUDA graphs | Spyre is not CUDA; piecewise mode is unsupported. |
| Dense body ladder (`1, 2, 4, 8, …, 512` by 8/16) | ~50 body compiles. Fine padding, unacceptable startup. |
| Decode `[1..max_num_seqs]` only, no 512 | Prefill / mixed `T` has no bucket → mid-request compile, or the scheduler must be capped to `max_num_seqs`. |
| Mutate `scheduler_output.total_num_scheduled_tokens` | Fights upstream accounting. `BatchDescriptor` is the CUDA-graph seam. |
| Background / lazy compile | Startup UX win; first-hit latency spikes. Follow-up, not a replacement. |
| Do nothing | Every new `T` compiles on the serving path. |

## **Prior Art**

* **vLLM CUDA graphs.** Capture a graph per batch descriptor, pad runtime
  batches onto it, execute `CUDAGraphMode.FULL` / piecewise. Same padding
  seam (`_determine_batch_execution_and_padding`, `BatchDescriptor`); we
  return `CUDAGraphMode.NONE` and a padded token count instead of a graph.
  This is the comparison [#174](https://github.com/torch-spyre/spyre-inference/issues/174)
  asked for (also [#121](https://github.com/torch-spyre/spyre-inference/issues/121),
  [#5](https://github.com/torch-spyre/spyre-inference/issues/5),
  [#147](https://github.com/torch-spyre/spyre-inference/issues/147)).
* **vLLM `compilation_config.compile_sizes`.** Already the knob for "which
  shapes to compile." The recorder fills it and honors a user-supplied list.
* **`torch.compile(dynamic=False)`.** Dynamo's specialization model is why
  a finite bucket list is required at all.

## **How we teach this**

* Call it the **graph recorder**, not CUDA graphs. The CUDA-graph vocabulary
  (`capture`, `replay`) is GPU-only; here the artifact is an Inductor graph
  keyed on `T`.
* **Bucket** = one padded size. **Ladder** / **`compile_sizes`** = the sorted
  list. Runtime always pads *up*.
* Document in spyre-inference configuration: default decoder ladder, how to
  override `-cc.compile_sizes`, and that `max_num_batched_tokens` follows
  `max(compile_sizes)`.
* Existing vLLM users already know capture sizes; the teaching point is that
  decode and prefill share one `T` axis under chunked prefill.

## **Unresolved questions**

* Should a production default persist Inductor / kernel cache so restart is
  not a full re-record?
* Should the default ladder grow a few extra sizes in the mixed-batch gap
  (`64, 128, 256`) without returning to a 50-entry list?
* Background warmup so first-boot serve does not block on the full record?

Out of scope: attention-kernel shape recording (explicitly excluded by #174)
and pooling / encoder body buckets (a later, separate change).

## Resolution

Shipped in spyre-inference via [#480](https://github.com/torch-spyre/spyre-inference/pull/480).
Default decoder `compile_sizes` are powers of two up to `--max-num-seqs`, plus
one prefill bucket at 512. User-supplied `compile_sizes` are respected and cap
the scheduler ([#638](https://github.com/torch-spyre/spyre-inference/pull/638)).

### Level of Support

* 4: Acceptance, with Little Feedback.

#### Additional Context

The sparse decode / single-prefill ladder was chosen to cut warmup time after
a dense (~50-entry) ladder proved too slow. Mixed-batch padding to 512 is the
accepted cost.

### Next Steps

Keep the recorder as the body compile path. Follow-ups: persistent cache,
optional mixed-gap sizes, background warmup.

#### Tracking issue

https://github.com/torch-spyre/spyre-inference/issues/174

#### Exceptions

None.
