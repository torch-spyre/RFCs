## Model ops / modules discovery: proposed approach

**Flow:** model list → 1 Discover → 2 Generate → 3 Validate & publish → 4 Fetch & overlay → 5 Run → 6 Record → 7 Watch drift

| Step | Runs on | Owner | Output |
|---|---|---|---|
| 0. Model list | – | each repo | models, pinned revision, capture settings, overlay |
| 1. Discover | CPU (ops and HF modules), Spyre (engine path) | hf-adapters / spyre-inference | raw capture |
| 2. Generate | same job | shared generator (torch-spyre) | raw config YAML |
| 3. Validate & publish | same job | same | blob in Artifactory, `latest` pointer |
| 4. Fetch & overlay | test job | each consumer | ready-to-run config |
| 5. Run | Spyre card | torch-spyre tests | JUnit with `capability.*` |
| 6. Record | ingest | existing ingest | rows in `capabilities` / `capability_runs` |
| 7. Watch drift | – | torch-spyre op owners | drift report, regressions |

**Where things live**

| Repo | Contains |
|---|---|
| **torch-spyre** | Shared generator core (`TorchOpCollector` and the YAML writer, moved from hf-adapters). The test config schema (oot_framework), including the new provenance and `dropped` fields. `test_model_ops_v2.py` and `OP_REGISTRY`. The generic module harness and its capability plugin (moved from hf-adapters). The fetch-and-overlay script. The capability ingest. A small golden model list and overlay for its own trunk and orchestrator runs. |
| **hf-adapters** | HF model list and overlays (`tests/model_lists/`). The HF capture driver (one script that takes a model id) and HF module discovery. The CPU capture workflow that publishes to Artifactory. The nightly that runs `latest`, and the PR check on one micro model. |
| **spyre-inference** | vLLM model list and overlays. The engine-path op capture driver and vLLM module discovery (moved from hf-adapters). The `test_vllm` module test. The Spyre capture workflow that publishes to the same Artifactory store. Its own nightly. |
| **Artifactory** (shared, not a repo) | Raw captured configs, stored by sha256, plus a per-model index with `latest` and `stable` pointers. |

hf-adapters and spyre-inference depend on torch-spyre and not on each other. torch-spyre reads only data from Artifactory, never downstream code. No repo commits generated YAMLs.

**Why discovery runs on CPU, not GPU**

1. **Everything the test YAML needs is device-independent, except one marker.** Op names, shapes, dtypes and strides come from the graph that torch.compile traces with fake tensors, before any backend runs. `device_layout` comes from torch-spyre's layout pass, which already runs on CPU: the committed `-with-device-layout.yaml` was captured on CPU. A GPU never produces `device_layout`.
2. **What the GPU was used for can be done on CPU.**
   - **The host-vs-device marker** (`cpu` vs `cuda:0`) can be recovered with a CPU-backed second device (PyTorch's OpenReg) or a short per-model host list. It affects very few cases: 1 per text model, 12 in Mistral-Small.
   - **Unsupported ops:** CPU runs every op, including ones torch-spyre doesn't support yet, just like a GPU. Spyre itself can't be used, because it stops at the first unsupported op.

| YAML field | Comes from | Same on CPU? | What changes |
|---|---|---|---|
| op name | Dynamo's FX graph | yes | Pin transformers 5.15.0 so the code path matches |
| shape | fake-tensor propagation | yes | Real weights, fixed token ids, forced decode steps |
| dtype | the dtype we load with | yes | Load bf16 explicitly (`from_config` defaults to fp32) |
| stride | fake-tensor propagation; attention depends on the SDPA kernel | yes, with one fix | Force math SDPA (`sdpa_kernel(SDPBackend.MATH)`), because CPU tries flash attention first |
| `device_layout` | torch-spyre's layout pass | yes (a GPU doesn't produce it) | A separate CPU step with torch-spyre installed, re-run when torch-spyre changes |
| host vs device (`cpu` vs `cuda:0`) | `tensor.device` at run time | no, on plain CPU | The generator writes `cuda:0`, the marker the test maps to Spyre. Real host ops come from OpenReg or an overlay host list |

The drivers also change:
- drop `require_cuda()` and `device_map="auto"`
- skip Inductor's C++ compile with a capture-only mode, since we only need the graph
- plan for enough RAM: about 50 GB in bf16 for the 24–26B models

**0. Model list**
- hf-adapters reuses `tests/model_lists/` and `MODEL_PATH` (no new knob). spyre-inference keeps its own list.
- Each entry has: Hub id, pinned revision, capture settings (fixed prompt token ids, decode steps, cache length), and the overlay.

**1. Discover**
- **Ops (HF path) run on CPU.** No GPU is needed (see "Why discovery runs on CPU" above).
- **Capture settings:** hf-adapters' locked env (transformers 5.15.0), bf16, math SDPA, fixed token ids, a few forced decode steps, and one model per process.
- **Skip unchanged models:** if the capture key (model + revision, versions, generator code, settings) is already in the store, skip it.
- **Triggers:** lockfile bumps, generator changes, new models, and a weekly safety run.
- **Modules (HF path) also run on CPU.** Module discovery only records shapes.
- **Engine path (spyre-inference) runs on a Spyre card.** It covers only what the engine can run today.
- **A failure is a result, not a red job:** save what was captured, mark it partial, and record the failure type (trace failure, unsupported model, engine failure, generator bug).
- **Host vs device:** a short spike on PyTorch's OpenReg test device, a CPU-backed second device that gives the exact host/device split. If it doesn't work out, use a per-model host list in the overlay.

**2. Generate**
- The shared generator writes a **raw** config: no xfail, no labels, no dropped ops.
- It adds:
  - provenance fields: repo + SHA, model + revision, versions, capture device
  - an origin per case
  - a `dropped:` list of anything the collector skipped, with the reason
- Devices: the capture device is written as `cuda:0`, the marker the test maps to Spyre. `cpu` is kept only for real host ops.

**3. Validate & publish**
- **Checks:** strict schema check, list any ops missing from `OP_REGISTRY`, reject empty configs.
- **Drift report:** a diff against the previous capture showing ops added or removed and signatures changed.
- **Upload** to Artifactory by sha256, never overwritten, and update the model's index.
- **`latest`:** a capture becomes `latest` if it's valid and not partial.
- **`stable`:** it moves automatically if the diff only adds cases; removed ops need an approval.

**4. Fetch & overlay**
- One fetch script: pick a model and a channel (or an exact digest), verify the sha256, cache locally.
- Apply the repo's own overlay: labels, dropped ops (each with a reason), init overrides, and the host-op list if OpenReg isn't used. No blanket xfail.

**5. Run**
- torch-spyre's `test_model_ops_v2.py` runs ops; the module harness runs modules.
- **Test changes needed:**
  - classify each failure as `not_implemented` or `failed`
  - record ops missing from the registry as `undetermined` instead of skipping them silently
  - fail loudly if the config doesn't load
  - record the config digest
- **Who runs what:**
  - torch-spyre trunk and orchestrator: `stable`, a small golden set
  - hf-adapters nightly: `latest`, all models
  - spyre-inference nightly: its engine captures
  - PR checks: one micro model
  - retries: reuse the same digest

**6. Record**
- Existing JUnit `capability.*` properties go through the existing ingest into `capabilities` / `capability_runs`.
- **Subject** is the full Hub id everywhere, **component** is the repo, and the **config digest** goes in props.
- Discovery failures show up as `undetermined`, with the reason.

**7. Watch drift**
- Capture diffs (step 3) show what changed in the model's ops.
- Verdict changes (passed → failed) show regressions, and torch-spyre op owners triage them.

**What each step writes to the YAML (snippets)**

*Step 1, Discover.* Captures one case per op. These cases become the `edits.ops.include` list:
```yaml
- name: torch.eq
  sample_inputs_func:
    args:
      - tensor: {shape: [1, 24], stride: [24, 1], storage_offset: 0,
                 dtype: torch.int64, device: cuda:0, init: randint, init_args: {high: 1000}}
                                  # cuda:0 = accelerator marker, written even on a CPU capture
      - value: 258880
  description: '.../modeling_gemma4.py#L2238 in get_placeholder_mask, code: ...'
  origin: transformers            # new: which code introduced the op
  tags: [constant, torch.eq.1]
```

*Step 2, Generate.* Wraps the cases in the standard file, with defaults and two new blocks:
```yaml
generation:                       # new: provenance
  generator: hf-adapters@<sha>
  model: google/gemma-4-12B-it
  revision: <sha>
  transformers: 5.15.0
  torch: <version>
  partial: false
test_suite_config:
  labels: [trunk]                 # default; the overlay may replace it
  global:
    supported_dtypes:
      - {name: float16, precision: {atol: 0.005, rtol: 0.005}}
      # ...
    input_config: {seed: 123}
  files:
    - path: ${TORCH_DEVICE_ROOT}/tests/models/test_model_ops_v2.py
      unlisted_test_mode: skip
      tests:
        - names: [TestSpyreModelOps::test_model_ops_db]
          mode: mandatory_success # default; the overlay may replace it
          tags: [model__gemma-4-12B-it]
          edits:
            ops:
              include: [...]      # the cases from step 1
dropped:                          # new: what the collector skipped, and why
  - {op: torch.getitem, call_site: '...#L...', reason: unbound symbolic shape}
```
If device layouts are kept, they are added per tensor by a separate torch-spyre layout step:
```yaml
      - tensor:
          shape: [1, 1, 41]
          dtype: torch.int64
          device_layout: {device_size: [2, 1, 32], stride_map: [32, -1, 1], device_dtype: torch.int32}
```

*Step 3, Validate & publish.* Doesn't change the YAML, since that would change its sha256. It stores the file as `blobs/<sha256>.yaml` and adds an index entry:
```json
{"digest": "sha256:ab12…", "capture_key": "…", "ops": 35, "cases": 260, "partial": false, "channel": "latest"}
```

*Step 4, Fetch & overlay.* Takes the overlay from the model list:
```yaml
overlay:
  labels: [trunk]
  drop_ops:
    - {op: torch.getitem, match: rotate_half, reason: "unbound floordiv slice, see #3773"}
  init_overrides:
    - {op: torch.sum, dtype: bfloat16, init: xavier, reason: "rand inputs overflow, see #5062"}
  host_ops:                        # only if OpenReg isn't used: these cases keep device cpu
    - {op: torch.to, match: cumulative_length, reason: "StaticCache counter is created on the host"}
```
...and applies it to the fetched YAML:
```yaml
source_digest: sha256:ab12…        # new: which stored capture this run used
test_suite_config:
  labels: [trunk]                  # from the overlay
  ...
          edits:
            ops:
              include:
                # the torch.getitem (rotate_half) case is removed
                - name: torch.sum
                  sample_inputs_func:
                    args:
                      - tensor: {..., dtype: torch.bfloat16, init: xavier}   # was: rand
```

*Steps 5–6, Run & Record.* Read the final YAML and add nothing to it. The verdicts carry the link back in JUnit:
```xml
<property name="capability.name" value="torch.sum"/>
<property name="capability.prop.config_digest" value="sha256:ab12…"/>
<property name="capability.prop.origin" value="transformers"/>
```

**Migration**
1. Seed the store with the 11 committed YAMLs, moving xfail and labels into overlays.
2. Run CPU captures and compare them against the seeds (the committed CUDA YAMLs). Only device strings should differ.
3. Switch torch-spyre and hf-adapters to fetch from the store, then delete the committed YAMLs.

**Open items**
- **OpenReg spike:** whether a CPU-backed second device gives the exact host/device split. If not, who maintains the per-model host lists.
- **CPU runner:** enough RAM for the largest models (about 50 GB in bf16).
- **Artifactory:** a repo for this, with write access from the capture job only and read access from all CI, including Spyre runners.
- **Promotions:** who approves `stable` promotions when ops are removed.
- **Module harness:** move the generic part to torch-spyre and `test_vllm` to spyre-inference.

---

## End-to-end example: `ibm-granite/granite-3.3-8b-instruct`

Paths marked *(new)* or *(moved)* are proposed names; everything else exists today.

**Files in each repo**
```
torch-spyre/
  tools/model_ops/                      (new) collector + generator (moved from hf-adapters),
                                              validate.py, publish.py, fetch.py (fetch + overlay)
  tests/models/test_model_ops_v2.py     op test (reads the final YAML)
  tests/models/op_registry.py           OP_REGISTRY
  tests/oot_framework/                  config schema (+ generation / dropped fields)
  tests/models/spyre_tensor_layout_extraction/   optional device_layout step (CPU)
  tests/model_lists/golden.yaml         (new) small golden set + overlays for trunk
  extensions/clickhouse-ingest/         capability ingest

hf-adapters/
  tests/model_lists/model_ops.yaml      (new) models + capture settings + overlays
  utils/model_ops/capture.py            (new) one CPU driver that takes a model id
  utils/module_discovery/auto_generate_module_config.py   HF module discovery (CPU)
  .github/workflows/model_ops_capture.yaml   (new) CPU capture → Artifactory
  Makefile                              capture-model-ops, tests TEST_TYPE=model_ops

spyre-inference/
  tests/model_lists/model_ops.yaml      (new) vLLM models + overlays
  tools/model_ops/capture_engine.py     (moved) engine-path capture (Spyre card)
  tools/module_discovery/auto_generate_module_config_vllm.py   (moved)
  tests/modules/test_vllm_modules.py    (moved) test_vllm
  Makefile                              test-model-ops, test-model-modules

Artifactory (shared)
  model-ops/blobs/<sha256>.yaml                          raw captures, never overwritten
  model-ops/index/ibm-granite/granite-3.3-8b-instruct.json   history + latest/stable
```
No generated YAML is committed anywhere.

**Step 0: model list entry** (`hf-adapters/tests/model_lists/model_ops.yaml`)
```yaml
models:
  - id: ibm-granite/granite-3.3-8b-instruct
    revision: <hub-sha>
    capture:
      prompt_token_ids: [...]     # 41 fixed token ids
      max_new_tokens: 16
      max_cache_len: 2048
    overlay:
      labels: [trunk]
```

**Steps 1–3: capture, generate, validate, publish** (hf-adapters, CPU runner, run when an input changes)
```bash
make capture-model-ops MODEL_PATH=ibm-granite/granite-3.3-8b-instruct
# which runs:
python -m utils.model_ops.capture --model ibm-granite/granite-3.3-8b-instruct \
    --model-list tests/model_lists/model_ops.yaml --out /scratch/raw/
python "$TORCH_DEVICE_ROOT/tools/model_ops/validate.py" /scratch/raw/granite-3.3-8b-instruct.yaml
python "$TORCH_DEVICE_ROOT/tools/model_ops/publish.py"  /scratch/raw/granite-3.3-8b-instruct.yaml
```
```
capture key 7f3c… not in store → capturing on cpu
captured 30 ops, 124 cases, 0 dropped
schema ok; 0 ops missing from OP_REGISTRY
diff vs previous: none (first capture)
published sha256:ab12… → latest, stable
```
The index now holds:
```json
{"model": "ibm-granite/granite-3.3-8b-instruct",
 "channels": {"latest": "sha256:ab12…", "stable": "sha256:ab12…"},
 "captures": [{"digest": "sha256:ab12…", "capture_key": "7f3c…", "transformers": "5.15.0",
               "ops": 30, "cases": 124, "partial": false}]}
```

**Steps 4–6: fetch, overlay, run, record** (hf-adapters nightly, Spyre runner)
```bash
make tests TEST_TYPE=model_ops MODEL_PATH=ibm-granite/granite-3.3-8b-instruct
# which runs:
python "$TORCH_DEVICE_ROOT/tools/model_ops/fetch.py" \
    --model ibm-granite/granite-3.3-8b-instruct --channel latest \
    --overlay tests/model_lists/model_ops.yaml --out /scratch/cfg/
#   → /scratch/cfg/granite-3.3-8b-instruct.yaml  (source_digest: sha256:ab12…, overlay applied)
bash "$RUN_TEST" /scratch/cfg/granite-3.3-8b-instruct.yaml \
    --junit-xml=results/model-ops-granite-3.3-8b-instruct.xml
```
The existing ingest (component `hf-adapters`) then writes one row per case, e.g.:
```
model_ops | ibm-granite/granite-3.3-8b-instruct | torch.matmul | passed | config_digest=sha256:ab12…
```

**torch-spyre trunk** (push to main, same test, `stable` channel, its own overlay)
```bash
python tools/model_ops/fetch.py --model ibm-granite/granite-3.3-8b-instruct --channel stable \
    --overlay tests/model_lists/golden.yaml --out /scratch/cfg/
bash "$RUN_TEST" /scratch/cfg/granite-3.3-8b-instruct.yaml
```

**spyre-inference** (same pattern; capture on a Spyre card, `engine: spyre-inference` in the index)
```bash
make test-model-ops MODEL=ibm-granite/granite-3.3-8b-instruct
```

**When something changes: transformers 5.15.0 → 5.20.2 in hf-adapters' lockfile**
1. The lockfile change gives granite a new capture key, so the CPU capture job runs again.
2. The diff report shows `- torch.matmul` (RoPE now multiplies instead of using `@`) and `- torch.le` (the causal mask is skipped in prefill), with new `torch.mul` cases.
3. The new capture becomes `latest`, so hf-adapters nightly tests it the same night.
4. Because ops were removed, `stable` waits for an approval. torch-spyre trunk keeps running the old `stable` digest until someone approves.
5. Every verdict carries its `config_digest`, so any result can be traced back to the exact capture.
