# RFC: Test Suite Configuration for running upstream pytorch tests from OOT devices

**Authors:**

- Anubhav Jana (IBM Research, India)
- Ashok Pon Kumar Sree Prakash (IBM Research, India)

*Reference implementation: IBM Spyre*

---

## 1. Motivation

PyTorch provides a large suite of upstream tests that validate operator correctness across devices. For out-of-tree (OOT) device backends registered via `privateuse1`, reusing these upstream tests is preferable to writing new ones — it ensures the same correctness bar and reduces maintenance burden.

However, OOT devices typically support a subset of ops and dtypes, and some tests may be known to fail, crash, or require special tolerance settings. Running the full upstream suite without filtering would result in thousands of failures and crashes that obscure real signal. While the upstream test refactoring is happening, we want to enable a way to selectively enable or edit the tests out of tree even before refactoring of all tests are complete.

This RFC defines a YAML-based configuration schema that allows an OOT device team to:

- Declare which ops and dtypes their device supports
- Declare which devices to test against (CPU, custom device, CUDA, etc.)
- Select which upstream tests to run, skip, or mark as expected failures
- Filter tests based on flexible criteria (ops, markers, patterns, etc.)
- Allow the same framework to control, parameterise device specific custom tests
- Express per-op and per-test tolerance overrides
- Tag tests with model names and other metadata for traceability
- Gradually expand test coverage as the device matures

---

## 2. Background

### 2.1 How upstream PyTorch tests work

Upstream PyTorch tests use the `@ops` decorator to parametrize test methods across all ops in `op_db` and all dtypes supported by those ops:

```python
@ops(binary_ufuncs, allowed_dtypes=(torch.float32, torch.float16))
def test_scalar_support(self, device, dtype, op):
    ...
```

At collection time, `@ops` generates one test variant per `(op, dtype)` combination. For a device to participate, it must register a `TestBase` subclass via `TORCH_TEST_DEVICES` and implement `instantiate_test`.

### 2.2 The Spyre test framework

The Spyre framework hooks into this mechanism via `SpyreTestBase` (which can eventually be contributed back to `PrivateUse1TestBase`) which:

1. Loads the YAML config on first `instantiate_test` call
2. Patches `@ops.op_list` directly to restrict which ops generate variants (`_OOTOpListPatcher`)
3. Patches `@onlyOn` to allow the `spyre` device type (`_OOTOnlyOnPatcher`)
4. Injects extra dtypes into `@ops.allowed_dtypes` (`_OOTDtypePatcher`)
5. Applies skip, xfail, or mandatory_success to each generated variant
6. Adds custom markers to tests for provenance.

---

## 3. Configuration File

The configuration is a YAML file pointed to by the `PYTORCH_TEST_CONFIG` environment variable. The downstream can have multiple
such config files. `PYTORCH_TEST_CONFIG` will govern which config needs to be used by upstream.

### 3.1 Top-level structure

```yaml
test_suite_config:
  files:
    - ...   # one entry per upstream test file
  
  global:
    devices:
      - cpu
      - spyre
    
    supported_dtypes:
      - name: float16
      - name: int64
      - ...
    supported_ops:
      - ...
```



| Field | Required | Description |
|---|---|---|
| `test_suite_config` | Yes | Root key |
| `files` | Yes | List of test file entries |
| `global` | No | Device-wide capability declaration |

---

## 4. File Entry

Each entry under `files` corresponds to one test file.

```yaml
- path: ${TORCH_ROOT}/test/test_binary_ufuncs.py # or ${TORCH_ROOT}/test/.*py
  unlisted_test_mode: skip
  tests:
    - ...
```

| Field | Required | Default | Description |
|---|---|---|---|
| `path` | Yes | — | Path to the test file or a glob. Supports `${TORCH_ROOT}` and `${TORCH_DEVICE_ROOT}` tokens resolved from env vars `PYTORCH_ROOT` and `TORCH_SPYRE_ROOT` |
| `unlisted_test_mode` | No | `skip` | Mode applied to tests not listed under `tests`
| `tests` | No | `[]` | List of test entries with explicit configuration |

### 4.1 `unlisted_test_mode`

Controls the behaviour for tests that are **not explicitly listed** in `tests`.

| Value | Behaviour |
|---|---|
| `skip` | Skip entirely (**Default**). Use when the file is under active development and most tests are not yet ready |
| `xfail` | Run but mark as expected failure. Use when the device broadly supports the op set but individual tests may still fail |
| `xfail_strict` | Run and mark as `xfail(strict=True)`. Fails the suite if the test unexpectedly passes — use when you want to be notified of unexpected improvements |
| `mandatory_success` | Must pass. Use with caution — any new test added to the test file will immediately break the suite |

**When to use each:**

```
New device, early stage:
  unlisted_test_mode: skip             <- only run what you explicitly list and skip the unlisted tests

Device broadly working, tracking regressions:
  unlisted_test_mode: xfail             <- run everything, failures expected

Stable device, enforcing correctness:
  unlisted_test_mode: mandatory_success  <- everything must pass
```

---

## 5. Test Entry

Each entry under `tests` configures a specific upstream test method. The same test can have multiple entries to define different combinations of behaviour if relevant. The final set will be the union of all tests.

`tests.names` is an array of TEST_CLASS::test_method, so that the same config can be applied to multiple tests/class level etc if required.

```yaml
tests: 
  - names:
      - TestBinaryUfuncs::test_scalar_support
      - TestBinaryUfuncs::test_contig_vs_transposed
    mode: xfail
    tags:
      - model_name_depending_on_this_test_1
    selectors: # [Planned]
      include:
        - has_ops: true
          ops_in_supported: true
      exclude:
        - markers:
            - slow
    edits:
      ops:
        include:
          - name: add
            description: "add description for ops (optional)"
        exclude:
          - name: gcd
      modules:                          # ← NEW field
        include:
          - name: torch.nn.BatchNorm2d
            description: "add batchnorm even though not in global.supported_modules"
        exclude:
          - name: torch.nn.Linear
            description: "Linear causes OOM on this test"
      dtypes:
        include:
          - name: float16
            description: "add description for dtypes include (optional)"
          - name: int64
        exclude:
          - name: bfloat16
            description: "add description for dtypes exclude (optional)"
```

| Field | Required | Default | Description |
|---|---|---|---|
| `names` | Yes | — | List of `ClassName::method_name` identifying the upstream test |
| `mode` | No | `mandatory_success` | How to treat this test's variants |
| `tags` | No | `[]` | Pytest mark labels applied to all variants of this test |
| `selectors` | No | — | **[New, Planned]** Per-test filtering criteria. Replaces the former top-level `test_selectors`. Same schema — see §5.4 |
| `edits` | No | — | Per-test overrides for ops, modules, and dtypes |

### 5.1 Test `mode`

Applied at the **variant level** — each `(test, op, dtype)` combination is treated independently.

| Value | Behaviour |
|---|---|
| `mandatory_success` | Variant must pass. Fails the suite if it does not |
| `xfail` | Variant is expected to fail. Passes the suite either way |
| `xfail_strict` | Variant must fail. Fails the suite if it unexpectedly passes |
| `skip` | Variant is skipped entirely with a skip message |

**`mode` vs `unlisted_test_mode` precedence:**

```
test listed with explicit mode    → test mode governs
test listed without mode          → mandatory_success
test not listed at all            → unlisted_test_mode governs
```

### 5.2 Markers

Markers are registered as pytest marks on every variant of the test. This enables test selection based on various filters like model:

```bash
pytest test_binary_ufuncs.py -m model_name_depending_on_this_test_1
pytest test_binary_ufuncs.py -m "model_a or model_b"
pytest test_binary_ufuncs.py -m "not model_a"
```

Markers must be valid Python identifiers (no spaces or special characters).

### 5.3 Edits

#### 5.3.1 `edits.ops`

Controls which ops are included in `@ops.op_list` for this specific test.

```yaml
edits:
  ops:
    include:
      - name: add    # inject add into @ops.op_list for this test
        description: "optional description"
    exclude:
      - name: gcd    # remove gcd from @ops.op_list for this test
        description: "optional description"
```

| Field | When to use |
|---|---|
| `include` | The test uses a pre-filtered op list (e.g. `binary_ufuncs_with_references`) that excludes an op you want to test or a particular op is not in global supported_ops, but anyway you want to override and test it. Injects the op into `@ops.op_list` at instantiation time |
| `exclude` | The op is in `supported_ops` and in the test's `@ops.op_list`, but you want to suppress it for this specific test only |

> **Note on `include`:** This is only needed when the test uses a filtered list that excludes your op or the op is not in globally supported ops like and you want to selectively enable for this test alone. For example, `binary_ufuncs_with_references = [op for op in binary_ufuncs if op.ref is not None]` excludes ops without a reference implementation. If `gcd` has no `ref`, you cannot test it via `test_scalar_support` without injecting it via `include`.

Both `include` and `exclude` are lists of dicts with `name` and `description (optional)` field, kept consistent for future extensibility (e.g. adding per-op precision overrides at the test level).

#### 5.3.2 `edits.modules` 

Controls which modules are included in `@modules.module_list` for this specific test. Mirrors the structure of `edits.ops`.

```yaml
edits:
  modules:
    include:
      - name: torch.nn.BatchNorm2d
        description: "add batchnorm even though not in global.supported_modules"
    exclude:
      - name: torch.nn.Linear
        description: "Linear causes OOM on this test"
```

| Field | When to use |
|---|---|
| `include` | The module is not in `global.supported_modules`, but you want to test it for this specific test only. Injects the module into `@modules.module_list` at instantiation time |
| `exclude` | The module is in `global.supported_modules` and in the test's module list, but you want to suppress it for this test only |

Both fields accept a list of dicts with `name` (required, fully-qualified Python class path e.g. `torch.nn.BatchNorm2d`) and `description` (optional), consistent with `edits.ops` and `edits.dtypes`.

#### 5.3.3 `edits.dtypes`

Controls which dtype variants are generated for this test.

```yaml
edits:
  dtypes:
    include:
      - name: float16
      - name: int64
    exclude:
      - name: bfloat16
```

| Field | When to use |
|---|---|
| `include` | Inject a dtype that the upstream `@ops(allowed_dtypes=(...))` does not include. Without this, the variant is never generated and cannot be tested |
| `exclude` | Suppress a dtype variant for this test only. Useful when a specific `(test, dtype)` combination is known to crash or produce incorrect results |

**Dtype precedence chain:**

The effective dtypes for a given test variant are computed as:

```
effective_dtypes =
    (global.supported_dtypes ∩ op.dtypes ∩ test.allowed_dtypes)     <- base intersection
    + edits.dtypes.include <- injected dtypes
    - edits.dtypes.exclude                                          <- removed dtypes
```

Where:

- `global.supported_dtypes` — hardware capability.
- `op.dtypes` — op-level dtype override from `global.supported_ops[op].dtypes`. If not specified, defaults to `global.supported_dtypes`.
- `test.allowed_dtypes` — upstream `@ops(allowed_dtypes=(...))` constraint from the test source code.
- `edits.dtypes.include` — can be mutually exclusive to `global.supported_dtypes`, not necessarily a subset. It can be an additional dtype to
test for a particular op without affecting other tests.
- `edits.dtypes.exclude` — applied last, after all inclusions.

### 5.4 Test Selectors *[Planned]*

**Status:** Not yet implemented

Provides flexible test selection based on multiple criteria with include/exclude logic. This allows declarative control over which tests run without modifying test files.

```yaml
- names:
    - TestBinaryUfuncs::test_scalar_support
  mode: mandatory_success
  selectors:
    include:
      # OR between list items
      - has_ops: true
        ops_in_supported: true
        # AND within each dict
      
      - markers:
          - cuda
          - slow
        name_patterns:
          - "test_conv*"
          - "test_matmul*"
    
    exclude:
      # OR between list items
      - markers:
          - slow
      
      - name_patterns:
          - "test_deprecated_*"
      
      - has_ops: false  # Exclude non-op tests when global op list is active
```

#### Selection Criteria

| Criterion | Type | Description |
|---|---|---|
| `has_ops` | bool | Test has `@ops` decorator |
| `ops_in_supported` | bool | Test's ops are in `global.supported_ops` |
| `markers` | list[str] | Pytest markers (e.g., `slow`, `cuda`, `skipif`) |
| `name_patterns` | list[str] | Glob patterns for test names (e.g., `test_add*`) |
| `module_patterns` | list[str] | Glob patterns for module paths (e.g., `test_ops*.py`) |
| `decorators` | list[str] | Specific decorator names to match |

#### Logic

- **Include section**: Test must match at least ONE include rule (OR logic)
  - Within each rule dict, ALL conditions must match (AND logic)
- **Exclude section**: Test must NOT match ANY exclude rule
- If no include rules specified: all tests included by default
- Exclude rules are applied after include rules

#### Use Cases

**Scenario 1: Run only op-based tests that are supported**
```yaml
- names:
    - TestBinaryUfuncs::test_scalar_support
  selectors:
    include:
      - has_ops: true
        ops_in_supported: true
```

**Scenario 2: Exclude slow tests**
```yaml
- names:
    - TestBinaryUfuncs::test_contig_vs_transposed
  selectors:
    exclude:
      - markers:
          - slow
      - name_patterns:
          - "test_deprecated_*"
```

**Scenario 3: Run specific test patterns for a model**
```yaml
- names:
    - TestModule::test_forward
  selectors:
    include:
      - markers:
          - model_resnet
        name_patterns:
          - "test_conv*"
          - "test_batchnorm*"
```

**Scenario 4: Filter non-op tests when using global op list**
```yaml
- names:
    - TestBinaryUfuncs::test_scalar_support
  selectors:
    exclude:
      - has_ops: false  # Skip tests without @ops decorator
```

---

## 6. Global Configuration

Declares the device-wide capability.

```yaml
global:
  devices:
    - cpu
    - spyre
    - cuda  # optional
  
  supported_dtypes:
    - name: float16
    - name: int64
  
  supported_ops:
    - name: add
      dtypes:
        - name: float16
          precision:
            atol: 1e-3
            rtol: 1e-3
        - name: int64
    - name: mul
    - name: sub
    - name: gcd
      force_xfail: true
```

| Field | Required | Description |
|---|---|---|
| `devices` | No | **[Planned]** List of devices to test against. See §6.1 |
| `supported_dtypes` | Yes | Device-wide supported dtypes |
| `supported_ops` | Yes | List of ops the device supports. Only ops listed here generate test variants |

### 6.1 `devices` *[Planned Feature]*

**Status:** Not yet implemented

Centralizes device configuration for all tests. Instead of hardcoding device names in individual tests or test base classes, the device list is declared once in the global configuration.

```yaml
global:
  devices:
    - cpu     # Reference device for comparison
    - spyre   # Primary test target
    - cuda    # Optional: if available
```

**Behaviour:**
- Tests will be parametrized across all listed devices
- Device availability is checked at runtime — unavailable devices are skipped with a clear message
- Per-test device override may be supported via test-level `edits` (future extension)

**Default:** If `devices` is not specified, defaults to `[cpu, spyre]` for backward compatibility.

**Use cases:**
- Test ops on multiple backends in a single run
- Compare results between reference (CPU) and target device (spyre)
- Conditionally include CUDA tests when hardware is available

### 6.2 `supported_dtypes`

The complete set of dtypes the device hardware supports. No test variant will run with a dtype outside this list will run, unless in the test specific config, it is explicitly set to include.

If omitted, no dtype filtering is applied at the global level.

### 6.3 `supported_ops`

Each entry declares one op the device supports and configures how tests exercising that op behave.

| Field | Required | Default | Description |
|---|---|---|---|
| `name` | Yes | — | Op name matching `OpInfo.name` in upstream `op_db` |
| `force_xfail` | No | `false` | If `true`, flips any `mandatory_success` variant for this op to `xfail`. Has no effect on variants already marked `xfail` or `xfail_strict` |
| `dtypes` | No | `global.supported_dtypes` | Op-level dtype override |

#### 6.3.1 `force_xfail` behaviour

`force_xfail` operates at the **variant level**, not the test level. Since `@ops` generates one variant per `(op, dtype)` combination, `force_xfail` on an op affects only variants for that specific op:

```
test_scalar_support_add_float16:
  test mode: mandatory_success, add.force_xfail: false  ->  mandatory_success

test_scalar_support_gcd_float16:
  test mode: mandatory_success, gcd.force_xfail: true   ->  xfail (flipped)

test_scalar_support_gcd_float16:
  test mode: xfail,             gcd.force_xfail: true   ->  xfail (unchanged)

test_scalar_support_gcd_float16:
  test mode: xfail_strict,      gcd.force_xfail: true   ->  xfail_strict (unchanged)
```

`force_xfail` only flips `mandatory_success` -> `xfail`. It never changes `xfail`, `xfail_strict`, or `skip`.

> **When to use `force_xfail: true`:** When an op is in `supported_ops` (so variants are generated) but is not yet stable enough to require passing. This allows tracking which tests exercise the op without committing to a correctness guarantee.

#### 6.3.2 Op-level `dtypes`

Narrows the dtype variants generated for this op across all tests.

Each dtype entry can optionally specify tolerance overrides:

```yaml
dtypes:
  - name: float16
    precision:
      atol: 1e-3
      rtol: 1e-3
  - name: int64       # no precision override, uses framework default
```

Precision overrides apply to all test variants for this `(op, dtype)` combination.

---

## 7. Non-Op Test Filtering *[Planned Feature]*

**Status:** Not yet implemented

### 7.1 Problem

When running with a global supported ops list and glob patterns like `**/*.py`, the framework collects **all** test files, including:

1. Tests decorated with `@ops` that match supported ops ✓
2. Tests decorated with `@ops` but with unsupported ops ✗
3. Tests without `@ops` decorator (utility tests, sanity checks) ✗

Currently, tests in categories 2 and 3 may fail or produce confusing errors because they don't match the filtering criteria.

### 7.2 Solution

Automatically filter out tests that don't have op-related decorators when a global op list is active. This prevents test collection/execution failures for non-op tests.

### 7.3 Implementation Strategy

Use `selectors` within a test entry to declaratively filter non-op tests:

```yaml
- names:
    - TestBinaryUfuncs::test_scalar_support
  selectors:
    include:
      - has_ops: true
        ops_in_supported: true
    
    # Alternatively, exclude non-op tests explicitly:
    exclude:
      - has_ops: false
```

**Pytest integration:**
- Use `pytest_collection_modifyitems` hook to inspect collected tests
- Check for `@ops` decorator presence
- Check if test's ops intersect with `global.supported_ops`
- Skip tests that don't match with clear skip message

### 7.4 Use Case Example

```yaml
test_suite_config:
  files:
    - path: ${TORCH_ROOT}/test/**/*.py  # Glob pattern - collects everything
      unlisted_test_mode: skip
      tests:
        - names:
            - TestBinaryUfuncs::test_scalar_support
          selectors:                    # ← per-test, not top-level
            include:
              - has_ops: true
                ops_in_supported: true
  
  global:
    supported_ops:
      - name: add
      - name: mul
```

**Result:**
- `test_add` with `@ops([add, mul, sub])` → runs (has ops, intersection with supported)
- `test_utility_function` without `@ops` → skipped (no ops decorator)
- `test_gcd` with `@ops([gcd])` → skipped (has ops, but `gcd` not in supported)

**Skip message:**
```
SKIPPED [1] test_utils.py::test_utility_function: Test has no @ops decorator (filtered by selectors)
SKIPPED [1] test_ops.py::test_gcd: Test ops ['gcd'] not in supported ops (filtered by selectors)
```

---

## 8. Scenarios

### 8.1 New model to be supported

A model depends on `add` and `mul`. You want to run the tests that exercise these ops and verify they pass.

```yaml
test_suite_config:
  files:
    - path: ${TORCH_ROOT}/test/*.py
      unlisted_test_mode: skip
      tests:
        - names: 
            - TestBinaryUfuncs::test_scalar_support
          mode: mandatory_success
          tags:
            - my_model
        - names: 
            - TestBinaryUfuncs::test_contig_vs_transposed
          mode: mandatory_success
          tags:
            - my_model

  global:
    supported_dtypes:
      - name: float16
      - name: int64
    supported_ops:
      - name: add
      - name: mul
```

Another model team wants to reuse the same op tests — they add their tag without changing anything else:

```yaml
- names: 
  - TestBinaryUfuncs::test_scalar_support
  mode: mandatory_success
  tags:
    - my_model
    - another_model    # <- added, no other change needed
```

### 8.2 New op supported by device

`gcd` is newly supported. You want to run all upstream tests that exercise `gcd`, with failures expected while it stabilises.

```yaml
test_suite_config:
  files:
    - path: ${TORCH_ROOT}/test/test_binary_ufuncs.py
      unlisted_test_mode: xfail       # run everything, failures expected
      tests:
        - names:
          - ....
          - ....
          mode: mandatory_success

  global:
    supported_dtypes:
      - name: float16
      - name: int64
    supported_ops:
      - name: gcd
        force_xfail: true             # all variants expected to fail initially
```

As `gcd` stabilises, flip `force_xfail: false` and move specific tests to `mandatory_success`.

### 8.3 Known crash — suppress a specific test

`test_add` causes a segfault. Block it entirely:

```yaml
- names: 
    - TestBinaryUfuncs::test_add
  mode: skip
  # Signal 11 - Segmentation fault
```

### 8.4 Tolerance override for a specific op

`add` passes on `float16` but requires looser tolerance:

```yaml
global:
  supported_ops:
    - name: add
      dtypes:
        - name: float16
          precision:
            atol: 1e-3
            rtol: 1e-3
```

### 8.5 Test uses a filtered op list — inject an op

`test_scalar_support` uses `binary_ufuncs_with_references` which only includes ops with a `ref`. If `gcd` has no `ref`, it is excluded from that list. To test it anyway:

```yaml
- names: 
    - TestBinaryUfuncs::test_scalar_support
  mode: xfail
  edits:
    ops:
      include:
        - name: gcd     # gcd has no ref so binary_ufuncs_with_references excludes it
                        # include injects it into @ops.op_list for this test only
```

### 8.6 Filter non-op tests with glob pattern *[Planned]*

Running all test files but only want op-based tests:

```yaml
test_suite_config:
  files:
    - path: ${TORCH_ROOT}/test/**/*.py  # Collect all test files
      unlisted_test_mode: skip
      tests:
        - names:
            - TestBinaryUfuncs::test_scalar_support
          selectors:                    # ← per-test selector
            include:
              - has_ops: true
                ops_in_supported: true
  
  global:
    supported_ops:
      - name: add
      - name: mul
```

Result: Only tests with `@ops` decorator and ops in the supported list will run.

### 8.7 Multi-device testing *[Planned]*

Test ops on both CPU and spyre:

```yaml
global:
  devices:
    - cpu
    - spyre
  
  supported_ops:
    - name: add
```

Each test variant will run on both devices, enabling cross-device validation.

### 8.8 Module include/exclude for a test *(New scenario)*

`TestModule::test_forward` should run `BatchNorm2d` even though it is not in `global.supported_modules`, but must skip `Linear` due to OOM:

```yaml
- path: ${TORCH_ROOT}/test/test_modules.py
  unlisted_test_mode: skip
  tests:
    - names:
        - TestModule::test_forward
      mode: mandatory_success
      edits:
        modules:
          include:
            - name: torch.nn.BatchNorm2d
              description: "add batchnorm even though not in global.supported_modules"
          exclude:
            - name: torch.nn.Linear
              description: "Linear causes OOM on this test"
```

---

## 9. Field Reference Summary

### File entry

| Field | Type | Required | Default |
|---|---|---|---|
| `path` | string | Yes | — |
| `unlisted_test_mode` | enum | No | `skip` |
| `tests` | list | No | `[]` |

### Test entry

| Field | Type | Required | Default |
|---|---|---|---|
| `names` | list of strings | Yes | — |
| `mode` | enum | No | `mandatory_success` |
| `tags` | list of strings | No | `[]` |
| `selectors` | selector dict | No | — | ← **New** (moved from top-level) [Planned] |
| `edits.ops.include` | list of `{name, description?}` | No | `[]` |
| `edits.ops.exclude` | list of `{name, description?}` | No | `[]` |
| `edits.modules.include` | list of `{name, description?}` | No | `[]` | ← **New** |
| `edits.modules.exclude` | list of `{name, description?}` | No | `[]` | ← **New** |
| `edits.dtypes.include` | list of `{name, description?}` | No | `[]` |
| `edits.dtypes.exclude` | list of `{name, description?}` | No | `[]` |

### ~~Test Selectors~~ *(removed from top-level — see `selectors` in Test entry above)*

~~| Field | Type | Required | Default |~~
~~|---|---|---|---|~~
~~| `include` | list of selector dicts | No | `[]` |~~
~~| `exclude` | list of selector dicts | No | `[]` |~~

#### Selector Dict Fields

| Field | Type | Description |
|---|---|---|
| `has_ops` | bool | Test has `@ops` decorator |
| `ops_in_supported` | bool | Test ops are in supported ops list |
| `markers` | list of strings | Pytest marker names |
| `name_patterns` | list of strings | Glob patterns for test names |
| `module_patterns` | list of strings | Glob patterns for module paths |
| `decorators` | list of strings | Decorator names to match |

### Global

| Field | Type | Required | Default |
|---|---|---|---|
| `devices` | list of strings | No | `[cpu, spyre]` |
| `supported_dtypes` | list of dtype entries | Yes | no filtering |
| `supported_ops` | list of op entries | Yes | no filtering |

### Supported op entry

| Field | Type | Required | Default |
|---|---|---|---|
| `name` | string | Yes | — |
| `force_xfail` | bool | No | `false` |
| `dtypes` | list of `{name, precision?}` | No | `supported_dtypes` |

### Precision

| Field | Type | Required | Default |
|---|---|---|---|
| `atol` | float | No | framework default |
| `rtol` | float | No | framework default |

---

## 10. Validation Rules

1. `names` must match `ClassName::method_name` pattern
2. `mode` and `unlisted_test_mode` must be one of `mandatory_success`, `xfail`, `xfail_strict`, `skip`
3. All dtype strings must be valid PyTorch dtype names
4. `edits.dtypes.include` may be subset of `global.supported_dtypes` or mutually exclusive to `global.supported_dtypes`
5. `supported_ops[*].dtypes` must be a subset of `global.supported_dtypes`
6. If `supported_ops[*].dtypes` ∩ `global.supported_dtypes` is empty, a warning is emitted
7. `tags` must be valid Python identifiers (used as pytest mark names)
8. `path` tokens (`${TORCH_ROOT}`, `${TORCH_DEVICE_ROOT}`) must resolve via environment variables at load time
9. `devices` must be valid device type strings (e.g., `cpu`, `cuda`, `spyre`, `privateuse1`)
10. Test selector patterns must be valid glob patterns
11. Test selector marker names must be valid pytest marker names
12. **[New]** `edits.modules.include` and `edits.modules.exclude` entries must have a `name` field that is a fully-qualified Python class path (e.g. `torch.nn.BatchNorm2d`)
13. **[New]** `selectors` within a test entry follows the same validation rules as the former top-level `test_selectors`

---

## 11. Environment Variables

| Variable | Description |
|---|---|
| `PYTORCH_TEST_CONFIG` | Path to the YAML config file |
| `PYTORCH_ROOT` | Resolves `${TORCH_ROOT}` token in `path` |
| `TORCH_SPYRE_ROOT` | Resolves `${TORCH_DEVICE_ROOT}` token in `path` |
| `PYTORCH_TESTING_DEVICE_ONLY_FOR` | Must be set to `privateuse1` |
| `TORCH_TEST_DEVICES` | Must point to `spyre_test_base_common.py` |
| `PYTORCH_TEST_WITH_SLOW` | Must be set to `1` to enable slow tests like `test_compare_cpu` |

## 12. Using the framework

### Running Tests

#### Single Command Line to run the test framework (Recommended Approach):

The orchestrator script resides in torch-spyre/tests/

- Please login to your spyre-enabled pod and `cd` to the `torch-spyre` directory (provide relative/absolute path based on your current path in the pod, in that case provide paths accordingly) 

Command format: `bash /path/to/torch-spyre/tests/run_test.sh /path/to/tests/config`

Below is an example:

```bash
bash tests/run_test.sh tests/configs/test_suite_config.yaml
```

This will run all the tests mentioned in the config. Please feel free to comment out / add / delete files that you dont want the framework to run.

#### Environment Setup (Backward Compatibility -- in case you want to export your own environment variables)

Export the required environment variables before running tests: (Please make sure you have torch-spyre and pytorch already cloned inside `DTI_PROJECT_ROOT` and are built properly).

```bash
# Set home directory (if using tmpfs)
export HOME=/dev/shm
mkdir -p dt-inductor

# Project root directory
export DTI_PROJECT_ROOT=/dev/shm/dt-inductor
cd $DTI_PROJECT_ROOT

# PyTorch test configuration
export PYTORCH_TESTING_DEVICE_ONLY_FOR="privateuse1"

# Spyre test framework paths
export PYTHONPATH="$DTI_PROJECT_ROOT/torch-spyre/tests:$PYTHONPATH"
export TORCH_TEST_DEVICES="$DTI_PROJECT_ROOT/torch-spyre/tests/spyre_test_base_common.py"

# Test configuration file
export PYTORCH_TEST_CONFIG="$DTI_PROJECT_ROOT/torch-spyre/tests/test_suite_config.yaml"

# Source code locations
export PYTORCH_ROOT="$DTI_PROJECT_ROOT/pytorch"
export TORCH_SPYRE_ROOT="$DTI_PROJECT_ROOT/torch-spyre"
```

**Note:** Replace `torch-spyre` with your actual fork repository name, if required.

#### Running Tests

Navigate to the PyTorch test directory and run tests:

```bash
cd $PYTORCH_ROOT/test/
python3 -m pytest test_binary_ufuncs.py -v
```

#### Running Specific Test Subsets

```bash
# Run only tests tagged with a specific model
python3 -m pytest test_binary_ufuncs.py -v -m model_name_depending_on_this_test_1

# Run tests matching a specific pattern
python3 -m pytest test_binary_ufuncs.py -v -k "test_scalar_support"

# [Planned] Use per-test selectors for complex filtering
# This will be automatic based on selectors in each test entry
```

#### Environment Variables Reference

| Variable | Description |
|----------|-------------|
| `PYTORCH_TESTING_DEVICE_ONLY_FOR` | Must be set to `privateuse1` |
| `TORCH_TEST_DEVICES` | Path to `spyre_test_base_common.py` |
| `PYTORCH_TEST_CONFIG` | Path to the YAML test configuration file |
| `PYTORCH_ROOT` | Path to PyTorch source repository |
| `TORCH_SPYRE_ROOT` | Path to torch-spyre repository |

---

## 13. Planned Features Summary

The following features are documented in this RFC but not yet implemented:

### 13.1 Global `devices` Configuration (§6.1)

**Target:** Centralize device configuration

**Benefits:**
- Test across multiple devices in single run
- Compare CPU vs custom device results
- Conditional device inclusion based on availability


### 13.2 Test Selectors (§5.4)

**Target:** Advanced declarative test filtering, scoped per test entry

**Benefits:**
- Filter by ops, markers, patterns, decorators
- Include/exclude logic with OR/AND combinations
- Different tests in the same file can use different selector logic
- No test file modification needed


### 13.3 Non-Op Test Filtering (§7)

**Target:** Automatically skip tests without `@ops` decorator

**Benefits:**
- Prevents failures when using glob patterns with global op list
- Clear skip messages for filtered tests
- Reduces test collection noise

---

## Appendix A: Complete Configuration Sample

The following example demonstrates every supported field in a single config file, **including planned features**. It is intended as a reference — real configs will use only the fields relevant to their device and test coverage stage.

```yaml
test_suite_config:

  # NOTE: test_selectors has been removed from the top level.
  # Test filtering is now declared per-test via the `selectors` field — see tests below.

  # ── File Entries ───────────────────────────────────────────────────────────
  files:

    # ── File 1: upstream binary ufunc tests ──────────────────────────────────
    - path: ${TORCH_ROOT}/test/test_binary_ufuncs.py

      # unlisted_test_mode controls tests NOT listed under `tests`.
      # skip = only tests explicitly listed below will run.
      # Options: skip | xfail | xfail_strict | mandatory_success
      unlisted_test_mode: skip

      tests:

        # ── Entry 1: two tests sharing the same mode, tags, selectors, and edits ──
        - names:
            - TestBinaryUfuncs::test_scalar_support
            - TestBinaryUfuncs::test_contig_vs_transposed

          # mode applies to every (op × dtype) variant of all tests in names.
          # Default when absent: mandatory_success
          # Options: mandatory_success | xfail | xfail_strict | skip
          mode: xfail

          # tags become pytest marks on every variant.
          # Select with: pytest -m model_1
          #              pytest -m "model_1 or model_2"
          #              pytest -m "not model_1"
          tags:
            - model_1
            - model_2

          # selectors: per-test filtering [Planned] — replaces former top-level test_selectors
          # OR between list items, AND within each dict
          selectors:
            include:
              - has_ops: true
                ops_in_supported: true
            exclude:
              - markers:
                  - slow
                  - requires_internet
              - has_ops: false

          edits:

            ops:
              # include: inject an op into @ops.op_list for these tests.
              include:
                - name: add
                  description: "inject add — binary_ufuncs_with_references excludes it if ref is None"

              # exclude: remove an op from @ops.op_list for these tests only.
              exclude:
                - name: gcd
                  description: "gcd causes buffer alignment errors for this test"

            # modules: NEW — inject or suppress modules for these tests
            modules:
              include:
                - name: torch.nn.BatchNorm2d
                  description: "add batchnorm even though not in global.supported_modules"
              exclude:
                - name: torch.nn.Linear
                  description: "Linear causes OOM on this test"

            dtypes:
              # include: inject a dtype into @ops.allowed_dtypes for these tests.
              include:
                - name: float32   # not in global.supported_dtypes — injected for this test only

              # exclude: suppress a dtype variant for these tests only.
              exclude:
                - name: bfloat16
                  description: "bfloat16 unsupported on Spyre for this test"

        # ── Entry 2: single test, mode skip ────
        - names:
            - TestBinaryUfuncs::test_add
          mode: skip
          # Signal 11 - Segmentation fault on Spyre

    # ── File 2: upstream test_ops.py ─────────────────────────────────────────
    - path: ${TORCH_ROOT}/test/test_ops.py
      unlisted_test_mode: skip
      tests:

        # ── Entry: module-level test (no @ops, plain device arg) ─────────────
        - names:
            - TestCommon::test_compare_cpu
          # export PYTORCH_TEST_WITH_SLOW=1 required for this test
          mode: mandatory_success
          edits:
            dtypes:
              exclude:
                - name: float32
                  description: "corrupted double-linked list — test_compare_cpu__refs__conversions_float_spyre_float32"

    # ── File 3: upstream test_modules.py ─────────────────────────────────────
    - path: ${TORCH_ROOT}/test/test_modules.py
      unlisted_test_mode: skip
      tests:

        # ── Entry: module test with edits.modules ─────────────────────────────
        - names:
            - TestModule::test_forward
          mode: mandatory_success
          edits:
            modules:
              include:
                - name: torch.nn.BatchNorm2d
                  description: "add batchnorm even though not in global.supported_modules"
              exclude:
                - name: torch.nn.Linear
                  description: "Linear causes OOM on this test"

  # ── Global: device-wide capability declaration ─────────────────────────────
  global:

    # devices [Planned]: list of devices to test against
    # Default: [cpu, spyre]
    devices:
      - cpu     # Reference device
      - spyre   # Primary target
      - cuda    # Optional: if available

    # supported_dtypes: positive list of dtypes the hardware supports.
    supported_dtypes:
      - name: float16
      - name: int64

    # supported_ops: only ops listed here generate test variants.
    supported_ops:

      - name: add
        force_xfail: false
        dtypes:
          - name: float16
            precision:
              atol: 1e-3
              rtol: 1e-3
          - name: int64

      - name: mul

      - name: sub

      - name: gcd
        force_xfail: true
```

---

## Appendix B: Feature Implementation Roadmap

### Phase 1: Current (Implemented)
- File-level configuration with glob patterns
- Test-level mode and tag configuration
- Op and dtype edits (include/exclude)
- Module edits (include/exclude) via `edits.modules`
- Global supported ops and dtypes
- Force xfail at op level
- Precision overrides per op/dtype

### Phase 2: Planned 
- Global `devices` configuration (§6.1)
- Non-op test filtering (§7)
- Per-test `selectors`
- Advanced test selectors — markers, patterns
- Per-test device overrides
- Subprocess execution for test isolation
- Configuration merge logic for layered configs

---

### Dtype effective set (reference)

```
effective_dtypes =
    (global.supported_dtypes ∩ op.dtypes ∩ test.allowed_dtypes)  ← base
    + edits.dtypes.include                                         ← additive, no global ceiling
    - edits.dtypes.exclude                                         ← always applied last
```

### Mode precedence (reference)

```
test listed with explicit mode    → test mode governs
test listed, no mode field        → mandatory_success (default)
test not listed at all            → unlisted_test_mode governs
op has force_xfail: true          → flips mandatory_success → xfail at variant level
```

### pytest selection with tags (reference)

```bash
# Run only variants tagged model_1
pytest test_binary_ufuncs.py -m model_1

# Run variants for either model
pytest test_binary_ufuncs.py -m "model_1 or model_2"

# Run model_2 variants not also tagged model_1
pytest test_binary_ufuncs.py -m "model_2 and not model_1"

# Exclude model_1 entirely
pytest test_binary_ufuncs.py -m "not model_1"
```

### Test selectors example (reference) *[Planned]*

```bash
# Configured per test entry via `selectors`, executed automatically:
# - names:
#     - TestBinaryUfuncs::test_scalar_support
#   selectors:
#     include:
#       - has_ops: true
#         ops_in_supported: true
#     exclude:
#       - markers:
#           - slow

# No command-line flags needed — filtering is automatic
pytest test_binary_ufuncs.py
```