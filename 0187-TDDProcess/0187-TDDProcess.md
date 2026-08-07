# Test-Driven Development Process for Adding New OOT Device Support

**Based on:** [RFC: Test Suite Configuration for running upstream PyTorch tests from OOT devices](https://github.com/torch-spyre/torch-spyre/pull/1135)

**Authors:** Ashok Pon Kumar Sree Prakash

---

## Overview

This document describes a test-driven development (TDD) process for onboarding a new out-of-tree (OOT) device backend onto PyTorch's upstream test suite using the YAML-based test configuration framework defined in PR #1135. The process uses the framework's `force_xfail`, `unlisted_test_mode`, and per-test `mode` controls to implement a red-green-refactor cycle at every stage of device maturation.


---

## Prerequisites

- PyTorch source cloned and built
- Your device backend registered as a `privateuse1` backend
- Environment variables configured:
  - `TORCH_TEST_CONFIG` — path to your YAML test config file
  - `TORCH_ROOT` — path to PyTorch source
  - `TORCH_DEVICE_ROOT` — path to your device plugin repo

---

## Phase 0: Bootstrap — Validate the Framework Plumbing

**Goal:** Run the upstream and device test suite with zero tests executing. Confirm the config loading, env var resolution, and test collection pipeline work end-to-end.

### Key principle: glob all test files

Use a glob pattern to cover all upstream and device test files. You do **not** need to know which files contain tests for which ops — the framework's `supported_ops` filter handles that. Tests in files that don't exercise any of your declared ops simply produce no variants.

### Steps

1. Create a minimal YAML config and a glob over all test files in upstream and device:

    ```yaml
    test_suite_config:
      files:
        - path: ${TORCH_ROOT}/test/**/*.py
          unlisted_test_mode: skip
          tests: []
        - path: ${TORCH_DEVICE_ROOT}/tests/**/*.py
          unlisted_test_mode: skip
          tests: []
    ```

2. Set up env vars and run the suite from the PyTorch test directory:

    ```bash
    export TORCH_TEST_CONFIG="$PROJECT_ROOT/torch-spyre/tests/test_suite_config.yaml"
    export TORCH_ROOT="$PROJECT_ROOT/pytorch"
    export TORCH_DEVICE_ROOT="$PROJECT_ROOT/torch-spyre"

    cd $TORCH_DEVICE_ROOT/test/
    pytest -v
    ```

3. **Expected result:** All tests are collected and skipped. Zero failures, zero passes, all skipped. This confirms the config loading, glob resolution, and test collection pipeline work.

### Exit criteria

- [ ] pytest runs without errors
- [ ] All tests show as "skipped"
- [ ] `${TORCH_ROOT}` and `${TORCH_DEVICE_ROOT}` glob in `path` resolves correctly

---

## Phase 1: Non-Op/Module Tests — Enable Broader Test Coverage

**Goal:** Identify and enable upstream PyTorch tests that are **not** decorated with `@ops` or `@modules` but are still relevant for validating device support — such as plain functional tests, tensor operation tests, device transfer tests, compilation tests, and integration tests.

### Why this matters

The `@ops` and `@modules` frameworks cover operator and module correctness, but a significant portion of PyTorch's upstream test suite consists of tests that don't use these decorators. These include:

- **Tensor creation and manipulation** — `torch.zeros`, `torch.cat`, `torch.reshape`, indexing, slicing
- **Device transfer semantics** — `.to()`, `.cpu()`, `.clone()`, cross-device copies
- **Autograd behavior** — gradient computation, backward pass, gradient accumulation
- **Compilation and JIT** — `torch.compile`, graph breaks, tracing behavior
- **Serialization** — `torch.save`/`torch.load` round-trips with device tensors
- **Memory management** — allocation, deallocation, device memory reporting
- **Distributed operations** — collective ops, process groups (if applicable)

These tests validate foundational device behavior that ops and modules depend on. A device that passes all `@ops` tests but fails basic tensor creation or device transfer is not production-ready.

### Approach: Testing team enables, component teams collaborate

This phase is driven by the **testing team** rather than individual component teams. The testing team:

1. **Surveys the upstream test landscape** — Categorize non-`@ops`/`@modules` tests by functional area (tensor ops, autograd, compile, serialization, etc.)
2. **Identifies high-value test groups** — Prioritize tests that validate foundational device behavior required by all backends
3. **Creates test selector patterns** — Define `test_selectors` using `name_regex` and `markers` to collect these tests by category
4. **Provides starter YAML config blocks** — Give component teams ready-to-use config snippets they can drop into their YAML files
5. **Works with component teams** to triage failures — Some failures may indicate genuine device gaps, others may be tests that assume CPU-specific behavior

### Step 1.1: Categorize non-`@ops`/`@modules` upstream tests

The testing team surveys PyTorch's test suite and groups non-decorated tests into functional categories:

| Category | Example test files | Example tests |
|---|---|---|
| Tensor creation | `test_tensor_creation.py` | `test_zeros`, `test_ones_like`, `test_empty` |
| Indexing/slicing | `test_indexing.py` | `test_basic_indexing`, `test_advanced_indexing` |
| Device transfer | `test_cuda.py`, `test_device.py` | `test_to_device`, `test_copy_between_devices` |
| Autograd | `test_autograd.py` | `test_grad_fn`, `test_backward` |
| Compile/JIT | `test_compile.py`, `test_jit.py` | `test_simple_compile`, `test_graph_break` |
| Serialization | `test_serialization.py` | `test_save_load_device_tensor` |

### Step 1.2: Define test selector patterns for each category

For each category, the testing team provides a `test_selectors` block that component teams can include:

```yaml
# Example: enabling tensor creation and device transfer tests
files:
  - path: ${TORCH_ROOT}/test/**/*.py
    test_selectors:
      include:
        - name_regex:
            - "TestTensorCreation::.*"
            - "TestDeviceTransfer::.*"
            - ".*test_to_device.*"
            - ".*test_empty.*"
            - ".*test_zeros.*"
    unlisted_test_mode: xfail
```

Component teams add these blocks incrementally as they are ready to tackle each category.

### Step 1.3: Component teams triage and promote

For each category of non-`@ops`/`@modules` tests enabled by the testing team:

1. **Run with `unlisted_test_mode: xfail`** — See what passes and what fails
2. **Triage failures with the testing team** — Distinguish between:
   - Tests that expose genuine device gaps → fix the backend
   - Tests that assume CPU-specific behavior → work with testing team to adapt or exclude
   - Tests that need device-specific setup → add to device config
3. **Promote passing tests to `mandatory_success`** — Same red-green-refactor cycle as op tests
4. **Tag by category** for selective execution:

```bash
# Run only tensor creation tests
pytest -m tensor_creation -v

# Run device transfer + serialization tests
pytest -m "device_transfer or serialization" -v
```

### Step 1.4: Iterate across categories

The testing team and component teams work through categories incrementally, prioritizing based on:

- **Foundational first** — Tensor creation and device transfer before compilation or distributed
- **Model dependencies** — Categories required by target models get priority
- **Shared wins** — Fixes that benefit multiple tests are prioritized

### Exit criteria

- [ ] Testing team has surveyed and categorized non-`@ops`/`@modules` upstream tests
- [ ] Test selector patterns are defined for each category
- [ ] Component teams have starter config blocks for high-priority categories
- [ ] Initial triage completed for at least tensor creation and device transfer categories
- [ ] Passing tests promoted to `mandatory_success` with category tags

---

## Phase 2: First Op — The Initial Red-Green Cycle

**Goal:** Get the first operator (`add`) passing on a single dtype (`float16`). This validates the full path from config → test collection → device execution → result reporting.

### Key principle: devs never need to browse PyTorch code

The developer need not have to browse PyTorch tests to see which ones are relevant, but be able to select and use them based on behaviours they want. The framework automatically discovers and runs **all upstream and device tests** that exercise those the features like op, dtype, module etc. There is no need to know test file names, class names, or method names upfront.

### Step 2.1: Declare the op and let the framework discover tests (RED)

Add `add` to `supported_ops`. Use `test_selectors` to collect only tests that exercise your declared ops, and `unlisted_test_mode: xfail` so discovered tests run with failures absorbed.

```yaml
test_suite_config:
  files:
    - path: ${TORCH_ROOT}/test/**/*.py
      test_selectors:                        # ← only collect tests for my ops
        include:
          - has_ops: true
      unlisted_test_mode: xfail              # ← run discovered tests, failures expected
    - path: ${TORCH_DEVICE_ROOT}/tests/**/*.py
      test_selectors:                        # ← only collect tests for my ops
        include:
          - has_ops: true
      unlisted_test_mode: xfail              # ← run discovered tests, failures expected
  global:
    supported_dtypes:
      - name: float16
    supported_ops:
      - name: add
```

```bash
pytest -v
# Expected: all test variants for (add, float16) across all files → xfail
#           tests without @ops or for other ops → deselected (not collected, not run)
```

**What happens:** The framework globs all `**/*.py` test files. `test_selectors` with `has_ops: true` deselects any test that doesn't use the `@ops` decorator — these are never collected or run. Only `@ops`-decorated tests are collected, and since `add` is in `supported_ops`, `(add, float16)` variants are generated and run as xfail. You immediately see the full scope of upstream coverage for your op — without ever opening a PyTorch test file, and without noise from unrelated tests.

**Why this is "red":** The op isn't implemented yet. Every variant runs and fails, but `unlisted_test_mode: xfail` absorb the failures so the suite itself is green.

### Step 2.2: Implement the op in your backend (GREEN)

Implement `aten::add` for `float16` in your device runtime. Once it produces correct results, re-run the tests:

```bash
pytest -v
# Expected: most (add, float16) variants → PASSED
#           some may still xfail (via unlisted_test_mode) — that's fine
```

Review the output. Tests that pass are your green baseline. Tests that still fail remain absorbed by `unlisted_test_mode: xfail`.

### Step 2.3: Lock in passing tests (REFACTOR)

From the test output, identify the tests that pass and promote them to `mandatory_success` so they become regression guards:

```yaml
files:
  - path: ${TORCH_ROOT}/test/**/*.py
    test_selectors:
      include:
        - has_ops: true
    unlisted_test_mode: xfail
    tests:
      - names:
          - TestBinaryUfuncs::test_scalar_support
          - TestBinaryUfuncs::test_contig_vs_transposed
          - TestCommon::test_compare_cpu
        mode: mandatory_success
        tags:
          - phase_2
  - path: ${TORCH_DEVICE_ROOT}/tests/**/*.py
    test_selectors:
      include:
        - has_ops: true
    unlisted_test_mode: xfail
```

Now these tests **must** pass on every run. Remaining unlisted tests continue to run as xfail, giving you visibility into what's left.

### Step 2.4: Analyze failures with testing team tooling

When an op is enabled and tests fail, the testing team provides tooling to help understand and triage those failures. This includes:

- **Failure categorization** — Automatically group failures by root cause (e.g., missing kernel, wrong dtype, numerical mismatch, unsupported tensor layout)
- **Failure reports** — Generate summaries showing which tests fail, why, and what backend work is needed to fix them
- **Priority ranking** — Identify which failures are quick wins vs. deep backend issues, so component teams can attack the highest-impact problems first

This tooling is developed and maintained by the testing team, and is used by component teams each time a new op is enabled to efficiently work through the xfail → mandatory_success promotion cycle.

### Exit criteria

- [ ] At least one op declared in `supported_ops` — tests pass
- [ ] Passing tests promoted to `mandatory_success`
- [ ] Full upstream and device test coverage for the op is visible (via xfail output)
- [ ] Tags enable selective test execution (`pytest -m phase_2`)

---

## Phase 3: Add Dtypes — Widen the Matrix

**Goal:** Expand dtype support for existing ops, one dtype at a time.

### Step 3.1: Add dtype to global list (RED)

Add the new dtype. Since `unlisted_test_mode: xfail` is already in place, all new `(add, bfloat16)` variants are automatically discovered and run as xfail — no config changes to individual tests needed.

```yaml
global:
  supported_dtypes:
    - name: float16
    - name: bfloat16     # ← new dtype, variants auto-discovered as xfail
  supported_ops:
    - name: add
```

```bash
pytest -v
# Existing (add, float16) variants: PASSED (mandatory_success)
# New (add, bfloat16) variants: xfail (auto-discovered)
```

If specific `bfloat16` variants crash (segfault, hang), suppress only those:

```yaml
tests:
  - names:
      - TestBinaryUfuncs::test_problematic_test
    mode: skip
    # reason: segfault on bfloat16 — tracking issue #42
```

### Step 3.2: Implement dtype support and promote (GREEN)

Once `bfloat16` works, promote passing tests to `mandatory_success` (same as Phase 2 Step 2.3). Remaining variants stay as xfail.

### Exit criteria

- [ ] Each supported dtype has at least one passing op
- [ ] Passing tests promoted to `mandatory_success` for each dtype

---

## Phase 4: Device-Specific Corner Case Tests

**Goal:** Add custom tests in your device repo that cover device-specific corner cases not exercised by upstream PyTorch tests — while using the same YAML config framework for consistency.

### Why this matters

Upstream PyTorch tests validate generic operator correctness, but your device may have corner cases unique to its hardware or compiler:

- Tensor shape/stride constraints (e.g., alignment requirements, maximum dimensions)
- Device-specific memory layouts or padding behavior
- Compiler-specific edge cases (e.g., fusion boundaries, graph breaks)
- Precision behavior at hardware limits (e.g., denormals, overflow/underflow)
- Multi-device or device-to-host transfer semantics
- Ops that work differently under compilation vs eager mode

These tests live in `${TORCH_DEVICE_ROOT}/test/` and are picked up by the second file entry in your YAML config.

### Step 4.1: Write device-specific tests using `@ops` (RED)

Write corner case tests in your device repo that use the same `@ops` decorator as upstream tests. This ensures they participate in the framework's op/dtype filtering automatically.

```python
# ${TORCH_DEVICE_ROOT}/tests/test_device_corner_cases.py

from torch.testing._internal.common_device_type import ops
from torch.testing._internal.common_methods_invocations import op_db

class TestDeviceCornerCases(TestCase):

    @ops(op_db, allowed_dtypes=(torch.float16, torch.bfloat16))
    def test_large_tensor_alignment(self, device, dtype, op):
        """Test ops with tensors that stress device alignment constraints."""
        # Large tensor that may trigger alignment-specific code paths
        x = torch.randn(1, 4097, device=device, dtype=dtype)
        result = op(x)
        ...

    @ops(op_db, allowed_dtypes=(torch.float16,))
    def test_denormal_inputs(self, device, dtype, op):
        """Test ops with denormal float16 values near hardware limits."""
        x = torch.tensor([1e-7, -1e-7], device=device, dtype=dtype)
        result = op(x)
        ...
```

Because these tests use `@ops`, they are automatically filtered by `test_selectors` — no special config needed. They show up in the xfail output alongside upstream tests.

### Step 4.2: Write non-`@ops` device tests and list them explicitly

For corner cases that don't map to a single op (e.g., device transfer, compilation behavior), write plain tests:

```python
# ${TORCH_DEVICE_ROOT}/tests/test_device_specifics.py

class TestDeviceSpecifics(TestCase):

    def test_host_to_device_transfer(self):
        """Verify tensor transfer from CPU to device preserves values."""
        x = torch.randn(100, dtype=torch.float16)
        y = x.to("privateuse1")
        assert torch.equal(x, y.cpu())

    def test_compile_graph_break_recovery(self):
        """Ensure device handles graph breaks gracefully under torch.compile."""
        ...
```

### Step 4.3: Tag for selective execution

Tag device-specific tests so they can be run independently or alongside model tests:

```bash
# Run only device corner case tests
pytest -m device_corner_cases -v

# Run model tests AND device corner cases together
pytest -m "granite_8b or device_corner_cases" -v
```

### Step 4.4: Promote and maintain

As with upstream tests, use the red-green-refactor cycle:

1. Write the test → it fails (RED)
2. Fix the backend → it passes (GREEN)
3. Add to `mandatory_success` → it becomes a regression guard (REFACTOR)

### Exit criteria

- [ ] Device-specific `@ops` tests are discovered automatically via `test_selectors`
- [ ] Non-`@ops` device tests are written and runnable
- [ ] Corner case tests are tagged for selective execution
- [ ] All device-specific tests pass with `mandatory_success`

---

## Phase 5: Model-Driven Op Expansion

**Goal:** Support all ops required by a specific model, using tags for traceability.

### Step 5.1: Profile the model to get the op and module lists

Use `torch.fx` tracing, `torch.profiler`, or manual analysis to identify which **ops** and **modules** the model requires.

For a vLLM model like `granite-8b`, inspect the model architecture to extract both:

```
Ops:      add, mul, sub, matmul, softmax, layer_norm, embedding, linear, gelu, ...
Modules:  Linear, RMSNorm, RotaryEmbedding, SiluAndMul, Attention, MLP, ...
```

Modules come from two sources:
- **PyTorch built-ins** (`nn.Linear`, `nn.Embedding`) — already in PyTorch's upstream `module_db`
- **vLLM custom modules** (`RMSNorm`, `RotaryEmbedding`, `PagedAttention`) — inherit from `nn.Module` via vLLM's `CustomOp` base class, but are not in PyTorch's `module_db`

### Step 5.2: Add ops and modules to global config (RED)

Add both ops and modules. The framework discovers and runs all upstream and device tests for them automatically.

```yaml
global:
  supported_ops:
    - name: add           # already passing
    - name: mul
      force_xfail: true   # new — not yet implemented
    - name: sub
      force_xfail: true
    - name: matmul
      force_xfail: true
    - name: softmax
      force_xfail: true

  supported_modules:
    - name: Linear         # PyTorch built-in — upstream @modules tests exist
    - name: LayerNorm
    - name: RMSNorm        # vLLM custom — needs custom module_db entry
      force_xfail: true
    - name: RotaryEmbedding
      force_xfail: true
    - name: SiluAndMul
      force_xfail: true
```

`supported_modules` works exactly like `supported_ops`:
- Only modules listed here generate `@modules` test variants
- `force_xfail` flips `mandatory_success` → `xfail` at the variant level
- Per-module `dtypes` and `precision` overrides are supported

```bash
pytest -v
# add variants: PASSED (already implemented)
# mul, sub, matmul, softmax op variants: xfail (auto-discovered)
# Linear, LayerNorm module variants: xfail (from upstream @modules tests)
# RMSNorm, RotaryEmbedding module variants: xfail (from custom module_db)
```

### Step 5.3: Create a custom module_db for vLLM modules

PyTorch's upstream `module_db` only contains built-in modules. For vLLM modules (`RMSNorm`, `RotaryEmbedding`, etc.), create a custom `module_db` in your device repo that follows the same `ModuleInfo` pattern:

```python
# ${TORCH_DEVICE_ROOT}/tests/vllm_module_db.py

from torch.testing._internal.common_modules import ModuleInfo

# vLLM modules all inherit from nn.Module (via CustomOp),
# so they can be tested with the same @modules framework.

vllm_module_db = [
    ModuleInfo(
        module_cls=RMSNorm,
        module_inputs_func=rms_norm_inputs,
        supported_dtypes=lambda device: (torch.float16, torch.bfloat16),
    ),
    ModuleInfo(
        module_cls=RotaryEmbedding,
        module_inputs_func=rotary_embedding_inputs,
        supported_dtypes=lambda device: (torch.float16, torch.bfloat16),
    ),
    ModuleInfo(
        module_cls=SiluAndMul,
        module_inputs_func=silu_and_mul_inputs,
        supported_dtypes=lambda device: (torch.float16, torch.bfloat16),
    ),
    # ... more vLLM modules
]
```

Then write tests using `@modules` that parametrize over this custom DB:

```python
# ${TORCH_DEVICE_ROOT}/tests/test_vllm_modules.py

from torch.testing._internal.common_modules import modules
from vllm_module_db import vllm_module_db

class TestVLLMModules(TestCase):

    @modules(vllm_module_db)
    def test_module_forward(self, device, dtype, module_info):
        """Test forward pass for vLLM modules on device."""
        module = module_info.module_cls(...).to(device=device, dtype=dtype)
        inputs = module_info.module_inputs_func(module_info, device=device, dtype=dtype)
        result = module(*inputs)
        ...

    @modules(vllm_module_db)
    def test_module_forward_backward(self, device, dtype, module_info):
        """Test forward + backward for vLLM modules on device."""
        ...
```

Because these tests use `@modules`, the framework filters them by `supported_modules` — just like `@ops` tests are filtered by `supported_ops`.

### Step 5.4: Configure module tests in the YAML

Module tests support the same config patterns as op tests. Use `edits.modules` (analogous to `edits.ops`) to include/exclude modules for specific tests:

```yaml
files:
  - path: ${TORCH_ROOT}/test/**/*.py
    test_selectors:
      include:
        - has_modules: true              # ← also collect upstream @modules tests
    unlisted_test_mode: xfail
    tests:
      - names:
          - TestModule::test_forward
        mode: mandatory_success
        tags:
          - granite_8b
        edits:
          modules:
            include:
              - name: RMSNorm            # inject if not in upstream module_db
            exclude:
              - name: Dropout            # not relevant for this model
  - path: ${TORCH_DEVICE_ROOT}/test/**/*.py
    test_selectors:
      include:
        - has_ops: true
        - has_modules: true
    unlisted_test_mode: xfail
    tests:
      - names:
          - TestVLLMModules::test_module_forward
          - TestVLLMModules::test_module_forward_backward
        mode: mandatory_success
        tags:
          - granite_8b
```

`edits.modules` works exactly like `edits.ops`:

| Field | When to use |
|---|---|
| `include` | Inject a module not in `supported_modules` or not in the test's `@modules` list for this specific test |
| `exclude` | Remove a module from the test's `@modules` list for this specific test |

### Step 5.5: Implement ops and modules (GREEN)

For each op and module, repeat the cycle:

1. Implement the op/module in your backend
2. Flip `force_xfail: false` for that op/module
3. Run `pytest -m granite_8b -v`
4. Fix failures, add tolerance overrides if needed
5. Commit the config change alongside the backend change

### Step 5.6: Analyze failures with testing team tooling

The same failure analysis tooling from Phase 2 (Step 2.4) applies here for both ops and modules. When a new module is enabled, the testing team tooling helps component teams:

- **Categorize module-specific failures** — e.g., missing backward pass, incorrect output shape, dtype mismatch, device transfer issues within the module graph
- **Distinguish op vs. module failures** — A module test failure may be caused by an underlying op not yet supported, rather than a module-level issue. The tooling traces failures back to root causes
- **Track cross-module dependencies** — Some modules compose others (e.g., `Attention` uses `Linear` + `Softmax`). The tooling identifies which upstream module or op fix would unblock the most downstream failures

### Step 5.7: Graduate the model (REFACTOR)

When all ops and modules for `granite-8b` pass:

```bash
pytest -m granite_8b -v
# Expected: all op AND module tests PASSED, zero xfail
```

This is your model-level acceptance gate. The tagged test suite covers both operator correctness (via `@ops`) and module correctness (via `@modules`) as a regression guard.

### Adding another model

When a second model shares ops/modules with the first, simply add tags:

```yaml
- names:
    - TestBinaryUfuncs::test_scalar_support
    - TestVLLMModules::test_module_forward
  mode: mandatory_success
  tags:
    - granite_8b
    - llama_3_8b    # ← new model reuses same op and module tests
```

New ops or modules unique to the second model go through the same cycle: add to `supported_ops`/`supported_modules` with `force_xfail: true` → tests auto-discovered as xfail → implement → flip → promote.

### Exit criteria

- [ ] All ops for the target model pass with `force_xfail: false`
- [ ] All modules for the target model pass with `force_xfail: false`
- [ ] Custom `module_db` created for vLLM modules not in upstream `module_db`
- [ ] All model tests (ops + modules) are tagged and selectable via `pytest -m <model>`
- [ ] No `force_xfail: true` remains for ops or modules the model depends on

---

## Phase 6: Mature Device — Flip to Mandatory Success

**Goal:** Since we've been using `unlisted_test_mode: xfail` with a glob from Phase 2, all upstream and device tests for your supported ops are already running. This phase is about flipping the default from "failures expected" to "failures are regressions".

### Step 6.1: Review the xfail landscape

By now you have a clear picture from test output:
- Tests promoted to `mandatory_success` — your regression guards
- Tests still running as xfail — known gaps
- Tests marked `skip` — crashes/hangs

Categorize remaining xfails:

| Observation | Action |
|---|---|
| Test passes unexpectedly | Promote to `mandatory_success` |
| Test fails — fixable soon | Keep as xfail, track in backlog |
| Test crashes (segfault, hang) | Mark as `skip` with comment |
| Test fails with tolerance issues | Add precision overrides |

### Step 6.2: Flip `unlisted_test_mode` to `mandatory_success` (GREEN)

When most tests pass, invert the config — list only the exceptions:

```yaml
files:
  - path: ${TORCH_ROOT}/test/**/*.py
    unlisted_test_mode: mandatory_success   # ← everything must pass
    tests:
      - names:
          - TestFoo::test_known_broken
        mode: xfail
      - names:
          - TestBar::test_crashes_device
        mode: skip
  - path: ${TORCH_DEVICE_ROOT}/test/**/*.py
    unlisted_test_mode: mandatory_success
```

You've flipped from allowlist to blocklist: instead of listing what works, you list what doesn't. Any **new** test added upstream or to your device repo that exercises your supported ops will automatically run and must pass.

### Exit criteria

- [ ] `unlisted_test_mode: mandatory_success` for the glob
- [ ] Only genuinely broken tests are listed as `skip` or `xfail`
- [ ] New upstream tests automatically run (and must pass)

---

## Summary: The TDD Progression

| Stage | `unlisted_test_mode` | `force_xfail` | What it proves |
|---|---|---|---|
| **Phase 0: Bootstrap** | `skip` | N/A | Framework plumbing works |
| **Phase 1: Non-op/module tests** | `xfail` | N/A | Broader test coverage enabled, testing team + component teams collaborate |
| **Phase 2: First op** | `xfail` | N/A | Single op passes, all tests auto-discovered |
| **Phase 3: Add dtypes** | `xfail` | `false` | Op works across dtypes |
| **Phase 4: Device corner cases** | `xfail` | `false` | Device-specific edge cases covered |
| **Phase 5: Model ops + modules** | `xfail` | per-op/module | Full model op and module set passes |
| **Phase 6: Mature device** | `mandatory_success` | `false` | Everything passes, exceptions listed |

Each transition is a deliberate, testable step. The YAML config acts as both the **test specification** and the **living documentation** of your device's capabilities at any point in time.

---

## Quick Reference: Config Patterns

### "I just added an op but it's not stable yet"
```yaml
- name: new_op
  force_xfail: true
```

### "This specific test crashes my device"
```yaml
- names:
    - TestFoo::test_bar
  mode: skip
  # reason: segfault — tracking issue #123
```

### "This op needs looser tolerance on float16"
```yaml
- name: my_op
  dtypes:
    - name: float16
      precision:
        atol: 1e-3
        rtol: 1e-3
```

### "I just added a module but it's not stable yet"
```yaml
supported_modules:
  - name: RMSNorm
    force_xfail: true
```

### "Exclude a module from a specific test"
```yaml
- names:
    - TestModule::test_forward
  edits:
    modules:
      exclude:
        - name: Dropout
```

### "Run only tests for a specific model"
```bash
pytest -m granite_8b -v
```

---

## Proposed Enhancements to the RFC

The following enhancements are needed to fully support the TDD workflow described above. These should be addressed as updates to the [RFC PR #1135](https://github.com/torch-spyre/torch-spyre/pull/1135).

- [ ] **`test_selectors` — declarative collection-time filtering with `include` / `exclude`**

  ### Problem

  The current RFC has no way to say "auto-discover and run only tests that exercise my declared ops, don't touch anything else." The two available options both fall short:

  - **`unlisted_test_mode: xfail`** — auto-discovers everything, but *executes* all tests including those without `@ops` decorators (plain device tests, module tests). This wastes time and produces noisy output in early phases.
  - **`unlisted_test_mode: skip`** — avoids execution, but requires devs to know and explicitly list every test they want to run — defeating the "devs never need to browse PyTorch code" principle.

  Currently, `supported_ops` filters at the variant generation level (patching `@ops.op_list`), but tests without `@ops` decorators are unaffected — they are collected and handled based on `unlisted_test_mode` regardless.

  ### Proposed solution: `test_selectors` with `include` / `exclude`

  A new file-level field with two sub-fields that filter at **collection time** using pytest's deselection mechanism (`pytest_collection_modifyitems`). Non-matching tests are never collected, never run, never show as skipped — just "X deselected" in the summary.

  ```yaml
  test_selectors:
    include:
      - ...   # selector groups (OR'd)
    exclude:
      - ...   # selector groups (OR'd), always wins
  ```

  #### Composition model

  Inspired by Kubernetes label selectors:

  - **Fields within a selector group** are AND'd together
  - **Multiple selector groups** within `include` or `exclude` are OR'd together
  - **`exclude`** always wins over `include` (applied after)

  #### Available filter fields

  | Field | Type | Matches |
  |---|---|---|
  | `has_ops` | bool | Test uses the `@ops` decorator |
  | `ops_in_supported` | bool | Test's `@ops` list intersects `global.supported_ops` |
  | `has_modules` | bool | Test uses the `@modules` decorator |
  | `modules_in_supported` | bool | Test's `@modules` list intersects `global.supported_modules` |
  | `markers` | list of strings | Test has ALL listed markers |
  | `name_regex` | list of strings | Test node ID matches ANY listed regex (Python `re` syntax) |
  | `has_params` | bool | Test is parametrized |

  #### Evaluation flow

  ```
  1. test_selectors.include → determines which tests are collected (OR across groups, AND within group)
  2. test_selectors.exclude → removes tests from collected set (OR across groups, AND within group)
  3. unlisted_test_mode → governs mode for collected tests not in explicit `tests` list
  4. tests[].mode → overrides mode for explicitly listed tests
  ```

  #### Examples

  **Early phase — only tests exercising my declared ops:**

  ```yaml
  files:
    - path: ${TORCH_ROOT}/test/**/*.py
      test_selectors:
        include:
          - has_ops: true
            ops_in_supported: true
      unlisted_test_mode: xfail
    - path: ${TORCH_DEVICE_ROOT}/tests/**/*.py
      test_selectors:
        include:
          - has_ops: true
            ops_in_supported: true
      unlisted_test_mode: xfail
  ```

  Only tests with `@ops` whose op list intersects `supported_ops` are collected. Plain device tests, module tests, tests for other ops — all deselected. Clean signal.

  **Only tests for a specific model tag:**

  ```yaml
  test_selectors:
    include:
      - markers:
          - granite_8b
  ```

  **Tests for my ops OR tests marked smoke:**

  ```yaml
  test_selectors:
    include:
      - has_ops: true          # group 1: tests with @ops for my supported ops
        ops_in_supported: true
      - markers:               # group 2: OR tests marked smoke
          - smoke
  ```

  Reads as: collect tests that *(have @ops AND ops intersect supported_ops)* OR *(are marked smoke)*.

  **Tests for my ops that are also marked regression:**

  ```yaml
  test_selectors:
    include:
      - has_ops: true
        ops_in_supported: true
        markers:
          - regression
  ```

  All three conditions AND'd within the single group.

  **Include specific non-@ops tests by name pattern:**

  ```yaml
  test_selectors:
    include:
      - has_ops: true
        ops_in_supported: true
      - name_regex:
          - "test_compare_cpu"
          - "test_dtypes"
  ```

  Collects @ops tests for my ops, plus specific known tests by name.

  **Exclude slow and autograd tests from whatever is collected:**

  ```yaml
  test_selectors:
    include:
      - has_ops: true
        ops_in_supported: true
    exclude:
      - markers:
          - slow
      - name_regex:
          - "test_autograd.*"
  ```

  Reads as: collect my ops tests, but exclude anything marked slow OR matching test_autograd.*.

  **No selectors (current RFC behaviour):**

  If `test_selectors` is omitted, all tests in matched files are collected (existing behaviour). This maintains backward compatibility.

  ### Interaction with `unlisted_test_mode`

  `test_selectors` and `unlisted_test_mode` operate at different levels:

  | Concern | Controlled by |
  |---|---|
  | **Which tests are collected** | `test_selectors.include` / `test_selectors.exclude` (deselection) |
  | **How collected-but-unlisted tests behave** | `unlisted_test_mode` (skip/xfail/xfail_strict/mandatory_success) |
  | **How explicitly listed tests behave** | `tests[].mode` |

  This separation means you can use `test_selectors.include` to narrow collection to relevant tests, then use `unlisted_test_mode: xfail` to run them all with failures absorbed, and promote individual tests to `mandatory_success` via the `tests` list — exactly the TDD flow described in this document.

- [ ] **`supported_modules` and `edits.modules` — module-level filtering analogous to ops**

  ### Problem

  The current RFC only supports op-level filtering via `supported_ops` and `edits.ops`. PyTorch's upstream testing framework also has a `@modules` decorator (in `common_modules.py`) that parametrizes tests over `module_db`, analogous to how `@ops` parametrizes over `op_db`. There is no way to control which modules generate test variants.

  Additionally, vLLM and other frameworks define custom `nn.Module` subclasses (e.g., `RMSNorm`, `RotaryEmbedding`) that are not in PyTorch's `module_db` but need to be tested with the same framework.

  ### Proposed solution

  **`global.supported_modules`** — analogous to `global.supported_ops`:

  ```yaml
  global:
    supported_modules:
      - name: Linear
      - name: RMSNorm
        force_xfail: true
        dtypes:
          - name: float16
            precision:
              atol: 1e-3
              rtol: 1e-3
      - name: RotaryEmbedding
        force_xfail: true
  ```

  Behaviour mirrors `supported_ops`:
  - Only modules listed here generate `@modules` test variants
  - `force_xfail` flips `mandatory_success` → `xfail` at the variant level
  - Per-module `dtypes` and `precision` overrides are supported

  **`edits.modules`** — analogous to `edits.ops`, at the test entry level:

  ```yaml
  tests:
    - names:
        - TestModule::test_forward
      edits:
        modules:
          include:
            - name: RMSNorm
          exclude:
            - name: Dropout
  ```

  | Field | When to use |
  |---|---|
  | `include` | Inject a module not in `supported_modules` or not in the test's `@modules` list |
  | `exclude` | Remove a module from the test's `@modules` list for this specific test |

  **`test_selectors` fields** — `has_modules` and `modules_in_supported` for collection-time filtering (see `test_selectors` enhancement above).
