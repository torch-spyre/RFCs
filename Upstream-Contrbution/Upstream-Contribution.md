# RFC: Upstreaming OOT Test Framework Features to PyTorch Core

**Authors:** Anubhav Jana, Mehant, Ashok Pon Kumar Sree Prakash (IBM Research)  
**Reference Implementation:** IBM Spyre (torch-spyre)  

---

## Overview

The OOT (out-of-tree) test framework developed for IBM Spyre solves a problem
that every `privateuse1` backend team faces: how to selectively run upstream
PyTorch tests against a device that supports a subset of ops and dtypes, without
either (a) crashing on unsupported variants or (b) maintaining a large fork of
test files.

Several of the mechanisms developed for this framework are device-agnostic and
belong upstream. This RFC enumerates each candidate, describes what it does,
where exactly it should live in the PyTorch codebase, and how it should be
shaped for upstream consumption.

---

## Candidate Features

### Feature 1 — `bypass_device_restrictions` on `DeviceTypeTestBase`

#### What it is

A boolean class attribute on `DeviceTypeTestBase` that, when `True`, causes
`@onlyOn` (and derived decorators like `@onlyCUDA`, `@onlyMPS`) to skip their
device check and run the test body anyway.

#### Current state

Already present as a stub in the upstream `DeviceTypeTestBase` (set to `False`)
and in `PrivateUse1TestBase`. The `onlyOn.__call__` method already checks for
it:

```python
# torch/testing/_internal/common_device_type.py  (current upstream)
class DeviceTypeTestBase(TestCase):
    bypass_device_restrictions: bool = False

class onlyOn:
    def __call__(self, fn):
        @wraps(fn)
        def only_fn(slf, *args, **kwargs):
            if slf.device_type not in self.device_type:
                if getattr(slf, "bypass_device_restrictions", False):
                    return fn(slf, *args, **kwargs)   # <-- already there
                raise unittest.SkipTest(reason)
```

#### What is missing upstream

`PrivateUse1TestBase.bypass_device_restrictions` is hardcoded to `False` with
no mechanism for a downstream backend to opt in without subclassing and
overriding the attribute.

#### Proposed change

Allow the attribute to be set to `True` on `PrivateUse1TestBase` via an
environment variable or config, so a backend can opt in without subclassing.
Alternatively (and more robustly), expose it as a writable class attribute that
`PrivateUse1TestBase.setUpClass` sets based on a registered capability.

**File:** `torch/testing/_internal/common_device_type.py`  
**Class:** `PrivateUse1TestBase`  
**Change type:** Attribute + optional env-var hook

```python
class PrivateUse1TestBase(DeviceTypeTestBase):
    # Opt-in: set PYTORCH_PRIVATEUSE1_BYPASS_DEVICE_RESTRICTIONS=1
    # or override this attribute in your subclass.
    bypass_device_restrictions: bool = bool(
        os.getenv("PYTORCH_PRIVATEUSE1_BYPASS_DEVICE_RESTRICTIONS", "0") == "1"
    )
```

---

### Feature 2 — `_OOTNativeDeviceTypesPatcher`: inject `"privateuse1"` into `NATIVE_DEVICES` (if @onlyNativeDeviceTypes still exists for certain tests)

#### What it is

`NATIVE_DEVICES` is a module-level tuple in `common_device_type.py`. The
decorators `@onlyNativeDeviceTypes` and `@onlyNativeDeviceTypesAnd` check
`self.device_type in NATIVE_DEVICES` at call time. The registered backend name
(e.g. `"openreg"`) is already in `NATIVE_DEVICES`, but `PrivateUse1TestBase`
resets `device_type` to the literal string `"privateuse1"` in `setUpClass`,
causing the check to miss.

The fix is a one-line tuple extension:

```python
# oot_upstream_patchers.py — _OOTNativeDeviceTypesPatcher.patch()
if "privateuse1" not in _cdt.NATIVE_DEVICES:
    _cdt.NATIVE_DEVICES = _cdt.NATIVE_DEVICES + ("privateuse1",)
```

#### Proposed change

This should be done unconditionally inside
`PrivateUse1TestBase.setUpClass`, immediately after the `device_type` reset.

**File:** `torch/testing/_internal/common_device_type.py`  
**Method:** `PrivateUse1TestBase.setUpClass`  
**Change type:** One-line addition

```python
@classmethod
def setUpClass(cls):
    cls.device_type = torch._C._get_privateuse1_backend_name()
    cls.device_mod = getattr(torch, cls.device_type, None)
    ...
    cls.primary_device = f"{cls.device_type}:{cls.device_mod.current_device()}"

    # Ensure "privateuse1" is in NATIVE_DEVICES so @onlyNativeDeviceTypes passes.
    # TorchTestBase uses the literal "privateuse1" in some paths, while
    # NATIVE_DEVICES contains the registered backend name.  Both must be present.
    import torch.testing._internal.common_device_type as _cdt
    if "privateuse1" not in _cdt.NATIVE_DEVICES:
        _cdt.NATIVE_DEVICES = _cdt.NATIVE_DEVICES + ("privateuse1",)
```

---

### Feature 3 — YAML-driven test configuration schema (`OOTTestConfig`)

#### What it is

A Pydantic-validated YAML schema that lets a `privateuse1` backend declare:

- Which ops it supports (`global.supported_ops`)
- Which dtypes it supports (`global.supported_dtypes`)
- Which upstream test files to run, and with what mode (`mandatory_success`,
  `xfail`, `xfail_strict`, `skip`)
- Per-test overrides for op lists, dtype lists, and tolerance

The schema is defined in `oot_test_models.py` and parsed by `oot_test_parsing.py`.

#### What should go upstream

The schema itself does not need to move wholesale into PyTorch core. What does
belong upstream is:

1. **A stable loading point**: `PrivateUse1TestBase` should check for a
   `PYTORCH_TEST_CONFIG` environment variable and, if set, load and cache a
   config object. The *format* of that config can remain the backend's
   responsibility; the *loading hook* is the upstream contract.

2. **`instantiate_test` override in `PrivateUse1TestBase`**: The override that
   reads the config, patches decorators, and applies modes should be a supported
   override point, not an internal OOT trick. The current `instantiate_test` in
   `DeviceTypeTestBase` is a `@classmethod`; `PrivateUse1TestBase` can override
   it cleanly.

**File:** `torch/testing/_internal/common_device_type.py`  
**Class:** `PrivateUse1TestBase`  
**Change type:** New `instantiate_test` override + config loading hook

```python
class PrivateUse1TestBase(DeviceTypeTestBase):

    _oot_config = None          # populated once by _load_oot_config()
    _oot_config_loaded = False

    @classmethod
    def _load_oot_config(cls):
        """Load YAML config from PYTORCH_TEST_CONFIG env var, if set.

        Returns None if the env var is unset or the config cannot be parsed.
        Backend teams subclass this to plug in their own config loader.
        """
        if cls._oot_config_loaded:
            return cls._oot_config
        cls._oot_config_loaded = True
        path = os.getenv("PYTORCH_TEST_CONFIG")
        if path is None:
            return None
        # Default implementation: backends can override with their own parser.
        cls._oot_config = cls._parse_oot_config(path)
        return cls._oot_config

    @classmethod
    def _parse_oot_config(cls, path: str):
        """Override in your TestBase to parse the config at `path`.

        Returns an object that instantiate_test can interrogate.
        The base implementation returns None (no filtering).
        """
        return None

    @classmethod
    def instantiate_test(cls, name, test, *, generic_cls=None):
        """Override point for config-driven test instantiation.

        If _load_oot_config() returns a config object, backends can inspect it
        here to apply op/dtype filtering, mode decorators, etc.
        Falls back to DeviceTypeTestBase.instantiate_test when no config.
        """
        config = cls._load_oot_config()
        if config is None:
            return super().instantiate_test(name, test, generic_cls=generic_cls)
        # Backend-specific logic lives here (or in a further subclass).
        return super().instantiate_test(name, test, generic_cls=generic_cls)
```

---

### Feature 4 — Decorator patchers as supported `PrivateUse1TestBase` utilities

#### What it is

The OOT framework contains a family of patchers that mutate PyTorch decorator
instances at `instantiate_test` time:

| Patcher | Decorator targeted | What it mutates |
|---|---|---|
| `_OOTOnlyOnPatcher` | `@onlyOn` | `val.device_type` list |
| `_OOTDtypePatcher` | `@ops(allowed_dtypes=...)` | `ops_instance.allowed_dtypes` |
| `_OOTOpListPatcher` | `@ops(op_list)` | `ops_instance.op_list[:]` |
| `_OOTOpDtypeExpander` | `@ops` + `OpInfo.dtypes` | `op_info.__dict__["dtypes"]` |
| `_OOTModuleListPatcher` | `@modules` | `modules_instance.module_info_list[:]` |
| `_OOTModuleDtypePatcher` | `@modules(allowed_dtypes=...)` | `modules_instance.allowed_dtypes` |
| `_OOTPrecisionOverridePatcher` | test function | `fn.precision_overrides / tolerance_overrides` |

These patchers exist because PyTorch deepcopies the test method before calling
`instantiate_test`, so a one-time global patch would not propagate. Mutating
the decorator instance post-copy but pre-instantiation is the only correct
approach.

#### What should go upstream

The patchers are currently OOT-private. They should become first-class utilities
in `common_device_type.py`, exported for use by any `PrivateUse1TestBase`
subclass.

Two options:

**Option A (preferred) — move patchers into `common_device_type.py` as
underscore-prefixed helpers:**

```python
# torch/testing/_internal/common_device_type.py

def _patch_only_on_for_privateuse1(test, backend_name: str) -> None:
    """Mutate @onlyOn on `test` to also accept `backend_name` and 'privateuse1'."""
    ...  # logic from _OOTOnlyOnPatcher.patch()

def _filter_ops_for_privateuse1(test, supported_ops: set[str]) -> None:
    """Filter @ops.op_list to `supported_ops`."""
    ...  # logic from _OOTOpListPatcher.patch()

def _inject_dtypes_for_privateuse1(test, extra_dtypes: set[torch.dtype]) -> None:
    """Add `extra_dtypes` to @ops.allowed_dtypes."""
    ...  # logic from _OOTDtypePatcher.patch()
```

**Option B — expose a `PatchContext` or `TestPatcher` protocol** so backends can
compose these into their own `instantiate_test` without copy-pasting the
closure-walking logic.

Either way the closure-walking logic (walking `__wrapped__` chains to find
decorator instances) is fragile and deserves a dedicated helper rather than
being silently re-implemented by each backend.

**File:** `torch/testing/_internal/common_device_type.py`  
**Change type:** New module-level helper functions, no existing API changes

---

### Feature 5 — Platform and op/dtype pytest markers via `parametrize_fn` wrapping

#### What it is

`_OOTPlatformMarkerPatcher`, `_OOTOpMarkerPatcher`, and
`_OOTModuleMarkerPatcher` attach structured pytest marks to generated test
variants:

- `platform__x86_64`, `platform__s390x`, `platform__aarch64` — identifies
  which CPU arch the test was collected on
- `op__add`, `op__mul` — identifies which op a variant exercises
- `dtype__float16`, `dtype__int64` — identifies which dtype a variant exercises
- `module__Linear`, `module__Conv2d` — identifies which module a variant exercises

This enables fine-grained pytest filtering:

```bash
pytest test_ops.py -m "op__add and dtype__float16"
pytest test_ops.py -m "not platform__s390x"
```

#### What should go upstream

The `op__*` and `dtype__*` markers are universally useful for any backend
team doing CI. They should be added to the `@ops` and `@modules`
`_parametrize_test` methods in `common_device_type.py` directly, not patched
on top.

The `platform__*` marker belongs in `PrivateUse1TestBase.instantiate_test`,
where it can be guarded by a feature flag.

**File:** `torch/testing/_internal/common_device_type.py`  
**Classes/methods:** `ops._parametrize_test`, `modules._parametrize_test`,
`PrivateUse1TestBase.instantiate_test`  
**Change type:** Additive — new markers attached to generated `test_wrapper`

Concrete change to `ops._parametrize_test` (already yields `test_wrapper`):

```python
# Inside ops._parametrize_test, after constructing test_wrapper:
import pytest, regex as re

op_safe = re.sub(r"[^a-zA-Z0-9_]", "_", op.name).strip("_")
if op_safe:
    test_wrapper = pytest.mark.__getattr__(f"op__{op_safe}")(test_wrapper)

if dtype is not None:
    dtype_safe = re.sub(r"[^a-zA-Z0-9_]", "_",
                        str(dtype).replace("torch.", "")).strip("_")
    if dtype_safe:
        test_wrapper = pytest.mark.__getattr__(f"dtype__{dtype_safe}")(test_wrapper)

yield (test_wrapper, test_name, param_kwargs, decorator_fn)
```

This is a no-op for users who do not use `-m op__*` and adds no runtime cost.

---

### Feature 6 — `_OOTCpuMovePatcher`: class-method CPU-tensor wrapper

#### What it is

Wraps specified test class methods (typically `assertEqual`) so that all tensor
arguments are moved to CPU before the call. This lets a `privateuse1` backend
reuse assertion helpers that internally call CUDA/CPU-only ops.

#### What should go upstream

This is too backend-specific for the core module. The right upstream surface is
a documented override point on `DeviceTypeTestBase`:

```python
class DeviceTypeTestBase(TestCase):

    # Override in your TestBase to list method names whose tensor args
    # should be moved to CPU before the call.
    _cpu_move_methods: list[str] = []
```

Then `PrivateUse1TestBase.setUpClass` (or `instantiate_test`) iterates
`_cpu_move_methods` and applies the wrapping, using a shared implementation
that lives in `common_device_type.py`.

**File:** `torch/testing/_internal/common_device_type.py`  
**Change type:** New class attribute + helper; no behavior change unless attribute is populated

---

### Feature 7 — `_OOTPrecisionOverridePatcher`: config-driven tolerance injection

#### What it is

Injects `fn.precision_overrides` and `fn.tolerance_overrides` onto the test
function before `instantiate_test` reads them, allowing a YAML config to
specify per-`(op, dtype)` tolerance without modifying test source files.

#### What should go upstream

The *mechanism* (`precision_overrides` / `tolerance_overrides` attributes on the
test function) is already upstream. What is missing is a documented way for a
backend to inject them programmatically inside `instantiate_test`.

Proposed: add a `_get_extra_tolerance_overrides` hook to
`DeviceTypeTestBase.instantiate_test`:

```python
@classmethod
def _get_extra_tolerance_overrides(cls, test_name: str, op_name: str | None,
                                   dtype: torch.dtype | None
                                  ) -> dict[torch.dtype, tol] | None:
    """Return extra tolerance overrides to apply before instantiation.

    Override in subclass to supply config-driven tolerances.
    Return None (default) for no extra overrides.
    """
    return None
```

`instantiate_test` calls this hook and merges the result into
`test.tolerance_overrides` before `_apply_precision_override_for_test` runs.

**File:** `torch/testing/_internal/common_device_type.py`  
**Method:** `DeviceTypeTestBase.instantiate_test` (new hook)  
**Change type:** New override point, backward-compatible

---

### Feature 8 — Mode-based test result control (`mandatory_success`, `xfail`, `skip`)

#### What it is

The OOT framework assigns one of four modes to each test variant:

| Mode | Behavior |
|---|---|
| `mandatory_success` | Must pass |
| `xfail` | Expected to fail; suite passes either way |
| `xfail_strict` | Must fail; unexpected pass fails the suite |
| `skip` | Skipped entirely |

Modes are applied as pytest decorators wrapping the test method after
`instantiate_test` generates it. The config can assign modes at file,
test-class, test-method, op, or dtype granularity.

#### What should go upstream

The mode concept is a clean abstraction for any backend that is in incremental
stabilization. The right upstream form is a hook in
`PrivateUse1TestBase.instantiate_test` that calls a user-overridable method to
determine the mode for each generated variant, then applies the corresponding
pytest decorator.

```python
class PrivateUse1TestBase(DeviceTypeTestBase):

    @classmethod
    def _get_test_mode(cls, test_name: str, op_name: str | None,
                       dtype: torch.dtype | None) -> str:
        """Return mode for the given test variant.

        Override to supply config-driven modes.
        Valid return values: 'mandatory_success', 'xfail', 'xfail_strict', 'skip'.
        Default: 'mandatory_success'.
        """
        return "mandatory_success"

    @classmethod
    def instantiate_test(cls, name, test, *, generic_cls=None):
        # ... (existing logic) ...
        # After generating each variant, apply its mode:
        mode = cls._get_test_mode(test_name, op_name=..., dtype=...)
        if mode == "skip":
            setattr(cls, test_name, unittest.skip("OOT: skip")(getattr(cls, test_name)))
        elif mode == "xfail":
            setattr(cls, test_name, pytest.mark.xfail(strict=False)(getattr(cls, test_name)))
        elif mode == "xfail_strict":
            setattr(cls, test_name, pytest.mark.xfail(strict=True)(getattr(cls, test_name)))
        # mandatory_success: no decoration needed
```

**File:** `torch/testing/_internal/common_device_type.py`  
**Class:** `PrivateUse1TestBase`  
**Change type:** New hook method + wiring inside `instantiate_test` override

---

## Summary Table

| # | Feature | Target file | Change type | Priority |
|---|---|---|---|---|
| 1 | `bypass_device_restrictions` env-var opt-in | `common_device_type.py` | Attribute + env hook | Low (already stubbed) |
| 2 | `"privateuse1"` in `NATIVE_DEVICES` | `common_device_type.py` | One-line in `setUpClass` | High (correctness bug) |
| 3 | YAML config loading hook | `common_device_type.py` | New `_load_oot_config` + `_parse_oot_config` override points | Medium |
| 4 | Decorator patchers as shared utilities | `common_device_type.py` | New helper functions | High (fragile, re-implemented by every backend) |
| 5 | `op__*` / `dtype__*` pytest markers | `common_device_type.py` | Additive in `ops._parametrize_test` | Medium |
| 6 | CPU tensor move wrapper | `common_device_type.py` | New `_cpu_move_methods` attribute + helper | Low |
| 7 | Config-driven tolerance injection | `common_device_type.py` | New `_get_extra_tolerance_overrides` hook | Medium |
| 8 | Mode-based result control | `common_device_type.py` | New `_get_test_mode` hook + wiring | High |

---

## Implementation Order

The recommended contribution order, based on impact vs. invasiveness:

1. **Feature 2** — fixing `NATIVE_DEVICES` is a one-line correctness fix with no
   API surface. Ship it as a standalone PR.

2. **Feature 4** — the decorator patchers solve a problem every `privateuse1`
   backend hits. Moving them into `common_device_type.py` as underscore-private
   helpers is low-risk and immediately useful.

3. **Feature 8** — the mode hook gives backends a clean way to express partial
   support without maintaining a fork. The hook has a safe default and requires
   no changes to existing test infrastructure.

4. **Feature 5** — the `op__*` / `dtype__*` markers are purely additive. They
   can be gated behind a `PYTORCH_TEST_ADD_OP_MARKERS=1` env flag initially.

5. **Features 3, 6, 7** — these are more design-heavy and can follow once the
   hook architecture from Feature 8 is established.

6. **Feature 1** — already half-done upstream; a small follow-up PR.

---

## Open Questions

1. **Should `_parse_oot_config` return a typed object or a duck-typed protocol?**
   A protocol (`OOTConfigProtocol`) would let backends use their own config
   classes without depending on Pydantic.

2. **Should the YAML schema itself be upstreamed?**
   The full Pydantic models (`OOTTestConfig`, `FileEntry`, `TestEntry`, etc.)
   are useful documentation of the config surface. They could live in a
   `torch/testing/_internal/oot_test_config.py` as an optional reference
   implementation, with backends free to replace the parser.

3. **`op__*` marker registration**: Pytest warns on unregistered marks.
   The markers would need to be registered either in `conftest.py` via
   `pytest_configure` or declared in `pyproject.toml`. The upstream pytest
   config would need one additional hook or a `filterwarnings` entry.

4. **Thread safety of `NATIVE_DEVICES` mutation**: The tuple reassignment
   in Feature 2 is not atomic. In practice `setUpClass` runs before any
   test threads start, so this is not a real issue, but a comment is warranted.

5. **`bypass_device_restrictions` and `@onlyNativeDeviceTypes`**:
   `bypass_device_restrictions` currently gates `@onlyOn` but not
   `@onlyNativeDeviceTypes` (which checks `NATIVE_DEVICES` directly). Feature 2
   fixes `@onlyNativeDeviceTypes`. These two features should be reviewed
   together to ensure the bypass semantics are consistent.

---

## `common_modules.py` — Module Testing Infrastructure

The OOT framework's `_OOTModuleListPatcher`, `_OOTModuleMarkerPatcher`, and
`_OOTModuleDtypePatcher` all target the `@modules` decorator defined in
`common_modules.py`. This section covers the gaps in that file that make
those patchers necessary in the first place, and what should change upstream
to eliminate the need for them.

---

### Feature 9 — `dtypesIfPrivateUse1` on `ModuleInfo`

#### What it is

`ModuleInfo` defines dtype overrides per device type (`dtypesIfMPS`,
`dtypesIfHpu`) but has no `dtypesIfPrivateUse1` field. The `supported_dtypes`
method is a dispatch table:

```python
# torch/testing/_internal/common_modules.py  (current)
def supported_dtypes(self, device_type):
    if device_type == 'mps':
        return self.dtypesIfMPS
    elif device_type == 'hpu':
        return self.dtypesIfHpu
    else:
        return self.dtypes   # <-- privateuse1 falls through here
```

When `device_type == "privateuse1"`, `ModuleInfo.supported_dtypes` returns the
generic `self.dtypes` — typically `floating_types()` which does not include
`float16`. OOT backends with narrower dtype support (e.g. float16-only) must
patch `op_info.__dict__["dtypesIfPrivateUse1"]` at instantiation time to inject
their dtypes. That is the job of `_OOTOpDtypeExpander`, and the same problem
exists on the module side.

#### Proposed change

Add `dtypesIfPrivateUse1` to `ModuleInfo.__init__` and `supported_dtypes`,
mirroring the existing `dtypesIfMPS` and `dtypesIfHpu` pattern exactly.

**File:** `torch/testing/_internal/common_modules.py`  
**Class:** `ModuleInfo`  
**Change type:** New optional constructor parameter + one branch in `supported_dtypes`

```python
class ModuleInfo:
    def __init__(
        self,
        module_cls,
        *,
        module_inputs_func,
        dtypes=floating_types(),
        dtypesIfMPS=(torch.float16, torch.float32),
        dtypesIfHpu=(torch.bfloat16, torch.float32),
        dtypesIfPrivateUse1=None,   # <-- new
        ...
    ):
        ...
        self.dtypesIfPrivateUse1 = dtypesIfPrivateUse1

    def supported_dtypes(self, device_type):
        if device_type == 'mps':
            return self.dtypesIfMPS
        elif device_type == 'hpu':
            return self.dtypesIfHpu
        elif device_type == 'privateuse1' and self.dtypesIfPrivateUse1 is not None:
            return self.dtypesIfPrivateUse1   # <-- new branch
        else:
            return self.dtypes
```

`None` as default preserves backward compatibility: unset means "use `self.dtypes`
as before." OOT backends can populate this field in their own `ModuleInfo`
registrations without touching the upstream `module_db`.

This eliminates `_OOTOpDtypeExpander`'s `__dict__` write for the module path
entirely, replacing a fragile implementation-detail hack with a first-class API.

---

### Feature 10 — `modules._parametrize_test` should attach `module__*` and `dtype__*` pytest markers

#### What it is

`ops._parametrize_test` (once Feature 5 is upstreamed) will attach `op__*` and
`dtype__*` markers to each generated test wrapper. The `modules._parametrize_test`
method in `common_modules.py` generates variants the same way but currently
attaches no markers at all.

```python
# torch/testing/_internal/common_modules.py  (current)
class modules(_TestParametrizer):
    def _parametrize_test(self, test, generic_cls, device_cls):
        for module_info in self.module_info_list:
            ...
            for (training, dtype) in product(training_flags, dtypes):
                ...
                @wraps(test)
                def test_wrapper(*args, **kwargs):
                    return test(*args, **kwargs)
                # <-- no markers attached
                yield (test_wrapper, test_name, param_kwargs, decorator_fn)
```

#### Proposed change

Apply the same marker injection pattern proposed in Feature 5, but for
`module_info.name` instead of `op.name`.

**File:** `torch/testing/_internal/common_modules.py`  
**Class:** `modules._parametrize_test`  
**Change type:** Additive — new markers on `test_wrapper`, no existing API change

```python
import pytest
import re as _re

# Inside modules._parametrize_test, after constructing test_wrapper:
module_safe = _re.sub(r"[^a-zA-Z0-9_]", "_", module_info.name).strip("_")
# module_info.name may contain "." (e.g. "nn.Linear") — already normalized by
# formatted_name but here we sanitize for pytest mark name rules.
if module_safe:
    test_wrapper = pytest.mark.__getattr__(f"module__{module_safe}")(test_wrapper)

if dtype is not None:
    dtype_safe = _re.sub(
        r"[^a-zA-Z0-9_]", "_", str(dtype).replace("torch.", "")
    ).strip("_")
    if dtype_safe:
        test_wrapper = pytest.mark.__getattr__(f"dtype__{dtype_safe}")(test_wrapper)

yield (test_wrapper, test_name, param_kwargs, decorator_fn)
```

This mirrors Feature 5 for `@ops` and eliminates `_OOTModuleMarkerPatcher`
entirely, since those markers would be attached by the parametrizer itself.

---

### Feature 11 — `ModuleInfo.supported_dtypes` should handle the registered backend name, not just `"privateuse1"`

#### What it is

`PrivateUse1TestBase.setUpClass` resets `device_type` from `"privateuse1"` to
the registered backend name (e.g. `"openreg"`, `"spyre"`). The call chain is:

```
modules._parametrize_test
    -> module_info.supported_dtypes(device_cls.device_type)
```

`device_cls.device_type` at parametrize time is `"privateuse1"` (before
`setUpClass` has run), but at test execution time `self.device_type` is the
registered name. This creates an asymmetry: dtype lookup at parametrize time
uses `"privateuse1"`, while `@onlyOn` checks at execution time use the
registered name.

Currently, both values must separately be handled — either the config patches
`op_info.__dict__["dtypesIfPrivateUse1"]` (for the parametrize-time lookup) and
the `onlyOn` patcher handles the execution-time check.

#### Proposed change

`ModuleInfo.supported_dtypes` (and `OpInfo.supported_dtypes`) should normalize
the device type string before the dispatch:

**File:** `torch/testing/_internal/common_modules.py` (and `common_methods_invocations.py`)  
**Change type:** Two-line normalization in `supported_dtypes`

```python
def supported_dtypes(self, device_type):
    # Normalize: treat the registered privateuse1 backend name the same as
    # the literal "privateuse1" string, which setUpClass assigns before tests run.
    import torch
    _pu1 = torch._C._get_privateuse1_backend_name()
    if device_type == _pu1:
        device_type = "privateuse1"

    if device_type == 'mps':
        return self.dtypesIfMPS
    elif device_type == 'hpu':
        return self.dtypesIfHpu
    elif device_type == 'privateuse1' and self.dtypesIfPrivateUse1 is not None:
        return self.dtypesIfPrivateUse1
    else:
        return self.dtypes
```

This is a one-time normalization that makes all downstream dtype-dispatching
code consistent regardless of when in the test lifecycle it is called.

The `_get_privateuse1_backend_name()` call is guarded by the conditional — it
only runs when a `privateuse1` backend is actually registered — so the overhead
for CPU/CUDA-only runs is zero.

---

### Feature 12 — `modules` parametrizer should support `filter_fn` for op-list-style filtering

#### What it is

`_OOTModuleListPatcher` filters `modules.module_info_list` in-place before
`_parametrize_test` iterates it. This is necessary because there is no
first-class way for a backend's `instantiate_test` to say "only generate
variants for these modules."

For `@ops`, the equivalent is handled by `_OOTOpListPatcher`, which also
filters `op_list` in-place. Both patchers exist because neither parametrizer
exposes a filter hook.

#### Proposed change

Add an optional `filter_fn` parameter to both `ops` and `modules` that is
called on each item before variant generation. This replaces in-place mutation
with a clean functional filter.

**File:** `torch/testing/_internal/common_modules.py` (and `common_device_type.py` for `ops`)  
**Classes:** `modules.__init__`, `modules._parametrize_test`; `ops.__init__`, `ops._parametrize_test`  
**Change type:** New optional parameter, backward-compatible (default `None`)

```python
class modules(_TestParametrizer):
    def __init__(
        self,
        module_info_iterable,
        allowed_dtypes=None,
        train_eval_mode=TrainEvalMode.train_and_eval,
        skip_if_dynamo=True,
        filter_fn=None,   # <-- new: callable(ModuleInfo) -> bool
    ):
        self.module_info_list = list(module_info_iterable)
        ...
        self.filter_fn = filter_fn

    def _parametrize_test(self, test, generic_cls, device_cls):
        module_list = self.module_info_list
        if self.filter_fn is not None:
            module_list = [m for m in module_list if self.filter_fn(m)]

        for module_info in module_list:
            ...
```

And identically for `ops`:

```python
class ops(_TestParametrizer):
    def __init__(
        self,
        op_list,
        *,
        dtypes=OpDTypes.supported,
        allowed_dtypes=None,
        skip_if_dynamo=True,
        filter_fn=None,   # <-- new: callable(OpInfo) -> bool
    ):
        self.op_list = list(op_list)
        ...
        self.filter_fn = filter_fn
```

A `PrivateUse1TestBase` subclass can then pass its supported-op set at
decoration time instead of patching the list at instantiation time:

```python
# In the backend's test base
@classmethod
def _make_ops_filter(cls):
    supported = cls._load_oot_config().resolved_supported_ops()  # set[str]
    if supported is None:
        return None
    return lambda op: op.name in supported
```

This is strictly cleaner than in-place list mutation and does not depend on
PyTorch deepcopying the decorator before `instantiate_test` runs.

---

### Feature 13 — `ModuleInfo` should support `dtypesIfPrivateUse1` in `module_db` entries via `DecorateInfo`

#### What it is

The existing `module_db` uses `DecorateInfo` with `device_type=` filtering for
device-specific skips and expected failures. There is no analogous mechanism for
supplying device-specific dtype overrides via the database at registration time.
OOT backends that want to narrow the dtype set for specific modules either
(a) subclass `ModuleInfo` and replace entries in `module_db`, or (b) use
`_OOTModuleDtypePatcher` to mutate `allowed_dtypes` at instantiation time.

Neither approach is clean. Option (a) requires maintaining a fork of `module_db`
entries. Option (b) is the patching approach this RFC aims to replace.

#### Proposed change

Expose `dtypesIfPrivateUse1` as a first-class `ModuleInfo` field (Feature 9)
and allow it to be populated per-entry in `module_db` for known backends.
For backends that are not yet upstream, the right surface is the `filter_fn`
hook (Feature 12) combined with a custom `module_inputs_func` registered via
`TORCH_TEST_DEVICES`.

No change to `module_db` itself is needed for OOT backends. The combination of
Features 9 and 12 gives OOT backends a clean path:

- Feature 9 provides the `dtypesIfPrivateUse1` hook on `ModuleInfo` for
  backends that contribute their `ModuleInfo` upstream
- Feature 12 provides the `filter_fn` hook for backends that filter the
  existing `module_db` at test time without modifying it

---

## Updated Summary Table

| # | Feature | Target file | Change type | 
|---|---|---|---|
| 1 | `bypass_device_restrictions` env-var opt-in | `common_device_type.py` | Attribute + env hook | 
| 2 | `"privateuse1"` in `NATIVE_DEVICES` | `common_device_type.py` | One-line in `setUpClass` | 
| 3 | YAML config loading hook | `common_device_type.py` | New `_load_oot_config` + `_parse_oot_config` | 
| 4 | Decorator patchers as shared utilities | `common_device_type.py` | New helper functions | 
| 5 | `op__*` / `dtype__*` markers on `@ops` | `common_device_type.py` | Additive in `ops._parametrize_test` | 
| 6 | CPU tensor move wrapper | `common_device_type.py` | New `_cpu_move_methods` attribute | 
| 7 | Config-driven tolerance injection | `common_device_type.py` | New `_get_extra_tolerance_overrides` hook | 
| 8 | Mode-based result control | `common_device_type.py` | New `_get_test_mode` hook | 
| 9 | `dtypesIfPrivateUse1` on `ModuleInfo` | `common_modules.py` | New constructor param + dispatch branch |
| 10 | `module__*` / `dtype__*` markers on `@modules` | `common_modules.py` | Additive in `modules._parametrize_test` | 
| 11 | Normalize registered backend name in `supported_dtypes` | `common_modules.py` + `common_methods_invocations.py` | Two-line normalization | 
| 12 | `filter_fn` on `ops` and `modules` parametrizers | `common_device_type.py` + `common_modules.py` | New optional parameter |
| 13 | `dtypesIfPrivateUse1` in `module_db` (design note) | `common_modules.py` | No immediate code change; design guidance | 

