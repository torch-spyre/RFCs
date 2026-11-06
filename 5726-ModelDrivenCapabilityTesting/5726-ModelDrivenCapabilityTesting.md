# Model-driven op and module capability testing

**Authors:**
* @ashokponkumar
* @anubhavjana

**Tracking issue:** torch-spyre/torch-spyre#5726

## **Summary**
Add two test types, `model_ops` and `model_modules`, to hf-adapters and spyre-inference. Each takes only a model id,
discovers the ops and modules the model executes on that repo's runtime, generates the test configuration on the fly,
runs torch-spyre's existing op and module tests against it, and records the verdicts in the `capabilities` tables.
No generated YAML is committed.

This amends the placement decisions of hf-adapters#686: the generators stay in the repos that own the runtime, the
tests and schema stay in torch-spyre, and the generated op YAMLs are no longer checked into any repository.

## **Motivation**
Operator support for a model is one layer of a wider question: can this model be supported on Spyre, through this
engine. The `capabilities` tables already model it at three levels (`model_ops`, `model_modules`, `model_support`).
Today the op and module levels are fed from committed YAML files, which has three problems.

* **Drift.** The ops a model executes depend on the runtime. In spyre-inference, 15 of 37 PRs merged between
  2026-09-15 and 2026-10-07 changed which ops are recorded or their shapes, and 7 more changed only the recorded
  `file:line` references (analysis posted on hf-adapters#686). A recording is stale within days.
* **Size.** The 11 model-ops YAMLs in torch-spyre total 2.4 MB. Adding spyre-inference recordings would add about
  38,000 lines (spyre-inference#1253, #1267), growing with every model.
* **Ownership.** Only the runtime repo can discover the ops its engine runs, but torch-spyre owns the tests that
  execute them. Committing generated files across that boundary freezes a cross-repo contract in git.

## **Proposed Implementation**

### Flow
```text
models.yaml (per repo)  ->  1 discover  ->  2 generate  ->  3 overlay  ->  4 validate  ->  5 run  ->  6 record
                              downstream repo (CPU or card)                              torch-spyre test   capabilities
```

1. **Discover.** Load the model, run `TorchOpCollector` (ops) or the module-discovery script (modules).
2. **Generate.** Write the config to a scratch directory using the shared generator core.
3. **Overlay.** Apply per-model policy from the repo's `models.yaml`: labels, `mode` (for example `xfail`), dropped
   ops with a required reason, init overrides, capture parameters. Today this policy exists only as hand edits to the
   committed YAMLs (all 11 carry a hand-set `mode: xfail`).
4. **Validate.** Check against the `OOTTestConfig` schema before running.
5. **Run.** Execute torch-spyre's `test_model_ops_v2.py` or the module tests with `PYTORCH_TEST_CONFIG=<generated>`.
6. **Record.** The tests emit JUnit `capability.*` properties; the existing results ingest writes `capabilities` and
   `capability_runs`.

### What exists and what is new
Discovery code is already checked in, in hf-adapters: `TorchOpCollector` and the YAML writer
(`utils/model_ops/utils/torchop_yaml.py`), per-model op drivers (`utils/model_ops/models/*/run_huggingface.py`, plus one
vLLM-path driver for Gemma4), and HF and vLLM module discovery (`utils/module_discovery/`). The module tests and the
`capability.*` plugin are there too. spyre-inference has none of it today; hf-adapters#686 moves the vLLM pieces there.

New in this RFC:
* a driver that takes a model id (today each driver hardcodes its model, prompt and token counts) and reads the repo's
  `models.yaml`;
* the overlay step, the schema validation, and the run-and-record wiring;
* hardening of the existing scripts for unattended use (next section).

### Failure handling
The existing drivers were written for manual runs and have no error handling: an exception during capture discards
everything captured so far. A nightly needs a failure to be a result, not a red job.

* **A discovery failure is a verdict.** An exception becomes a `capability_runs` row with status `undetermined` for the
  model, with the failure class and stage recorded as properties. `undetermined` already exists for tests that broke
  without giving a verdict.
* **Partial captures are kept.** The collector records ops per compiled graph, so ops from graphs captured before a
  failure are available. The driver writes them in a `finally` and marks the config `partial: true`; the model is
  reported as incomplete rather than dropped.
* **Failure classes are distinguished**, because they have different owners: torch or Dynamo could not trace the
  model; the installed transformers does not support the architecture; the engine failed to start (vLLM path).
* **One model per job.** The matrix runs each model separately, so one crash does not stop the others.

A model that does not yet run on Spyre can still be discovered on the HF path, because op capture happens on the FX
graph before any Spyre lowering, and unsupported ops then surface as `not_implemented` in the run step. On the vLLM
path discovery only goes as far as the engine does. A model the engine cannot start gets `undetermined` with the
reason; the HF path remains the way to get its op coverage. The two paths answer different questions: HF says which
ops the model needs, the engine path says whether spyre-inference can run it. Partial-capture behaviour is read from
the code and not yet tested.

### Where things live

| Artifact | Home |
|---|---|
| HF model list + overlay | hf-adapters |
| vLLM model list + overlay | spyre-inference |
| HF discovery and drivers | hf-adapters |
| vLLM discovery and drivers | spyre-inference |
| Generator core, schema, op and module tests | torch-spyre |
| Generated configs | CI artifact only, never committed |
| Signature baseline and drift | `capabilities` / `capability_runs` |

The dependency direction is downstream to torch-spyre only. torch-spyre installs nothing downstream and hf-adapters
and spyre-inference do not depend on each other.

### Invocation

| Repo | Command |
|---|---|
| hf-adapters | `make tests TEST_TYPE=model_ops MODEL=<id>` and `TEST_TYPE=model_module MODEL=<id>` |
| spyre-inference | `make test-model-ops MODEL=<id>` and `make test-model-modules MODEL=<id>` |

With no `MODEL`, every entry in the repo's `models.yaml` for that framework runs. They are suite keys, outside the
per-PR gate: nightly by default, plus one small smoke model when generator, overlay or model-list files change.

### Execution environment
The validation image is the torch-spyre or spyre-inference image the repo already resolves. It ships
`tests/models/test_model_ops_v2.py` and the oot_framework, so the test code is whatever the image was built with
rather than a clone of main.

Measured on 2026-10-25 (CPU, no card, torch-spyre image `amd64-dev-f87aaf2b050d`, `IS_INDUCTOR_SPAWNED_SUBPROCESS=1`):

| Path | Run | Result |
|---|---|---|
| HF op discovery | granite-3.3-8b config, random weights, 2048 static cache | build 117 s, capture 142 s, 28 ops, 112 cases |
| HF module discovery | micro-g3.3-8b-instruct-1b | 27 s, 24 module entries |
| vLLM op discovery | micro-g3.3-8b-instruct-1b, spyre-inference image `weekly-amd64` | fails without a card: segfault in `SpyreCCLBackend`; with `gloo` it fails with `RuntimeContext not created` |

So HF discovery runs on a CPU runner; vLLM discovery needs a Spyre runner. The vLLM result has not yet been
obtained with a card.

### Generation metadata
Generated configs carry real YAML fields (not comments) for provenance: generator repo and SHA, model and revision,
framework, vLLM and transformers versions, image digest. Each case also carries an `origin` (spyre-inference, vllm,
torch-spyre, transformers) derived from its source reference, and a stable signature key. This separates ops that
spyre-inference code introduces from ones upstream already covers.

## **Metrics**
* Models with a current (this-run) `model_ops` and `model_modules` verdict per engine.
* Fraction of runs whose `capability_id`s are new versus the previous run, as a drift signal.
* Nightly wall time per model and generation failure rate.
* Committed model YAML bytes in torch-spyre, hf-adapters and spyre-inference (target: zero).

## **Drawbacks**
* A new generation step in nightly can fail for reasons unrelated to Spyre support. Mitigation: equivalence check
  against the committed YAMLs before deleting them.
* The CPU capture is not yet equivalent to a committed CUDA capture: for granite-3.3-8b it produced 28 of 30 op
  names (missing `torch.le`, `torch.matmul`) and 112 of 124 cases, and records `device: cpu` rather than `cuda:0`.
* vLLM discovery needs card time.
* Model downloads (gated and large models) need a shared HF cache on the runner.
* A failure caused by a torch-spyre change is still visible in a downstream job. The image pin limits how often.
* Overlay `drop_ops` can hide regressions the way hand edits did; each entry needs a reason.
* The existing discovery scripts need hardening (error handling, partial output) before a nightly can depend on them.

## **Alternatives**
* **Commit generated YAMLs (hf-adapters#686 as merged).** Drifts within days and grows the repos.
* **Compact the committed YAMLs.** Reduces size, not drift.
* **Run spyre-inference from torch-spyre CI.** Installs a downstream package into an upstream one.
* **Vendor an ops test into spyre-inference.** Forks the test code and its schema.
* **Keep op coverage only in torch-spyre.** Misses the ops that only the engine path produces (custom layers, KV
  layout, batching).

## **Prior Art**
* The `capabilities` / `capability_runs` tables and the JUnit `capability.*` properties already used by the model-ops,
  model-module and model-support tests.
* hf-adapters module discovery and its `MODULE_CONFIG` knob; `TorchOpCollector`.
* hf-adapters#686 (generator placement) and the discussion on spyre-inference#1149, #1253, #1255 and #1267.

## **How we teach this**
Contributors add a model by adding an entry to their repo's `models.yaml`; they do not generate or commit YAML.
Results are read from the capability views. README sections in hf-adapters and spyre-inference replace the current
generation instructions.

## **Unresolved questions**
* Does the vLLM capture produce a usable config on a card? (Next experiment.)
* Does a partial capture from a failing model write a usable config? (Read from code, untested.)
* What explains the `torch.le` / `torch.matmul` gap between CPU and CUDA captures?
* Which `component` value do spyre-inference rows use, and does it avoid collisions with existing writers?
* Where is the generated YAML referenced from a run (CI artifact link, `run_properties`)?
* A model listed in both repos has two overlays; is a consistency check enough?
* Who owns promoting entries from `xfail` to passing?
* Suite key naming: hf-adapters uses `model_module`, the capability `test_type` is `model_modules`, and the
  orchestrator already has a `model_ops` column.
* Fate of the committed model YAMLs in torch-spyre and their owners' sign-off.

## Resolution
