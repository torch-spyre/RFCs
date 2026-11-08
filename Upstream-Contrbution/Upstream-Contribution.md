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

## Mental model: two layers of filtering

Before enumerating features it helps to distinguish two layers of filtering the
OOT framework performs, because they require different upstream mechanisms:

**Layer 1 — global capability filtering** (`global.supported_ops`,
`global.supported_dtypes`): "My device supports `add` and `float16` everywhere."
This is static and can be expressed as a predicate passed at decoration time
(`filter_fn` on `@ops` / `@modules`, `dtypesIfPrivateUse1` on `ModuleInfo`).

**Layer 2 — per-test edits** (`edits.ops.include/exclude`,
`edits.dtypes.include/exclude`): "For *this specific test*, inject `gcd` even
though it is not in `global.supported_ops`" or "for *this specific test*,
suppress `float32` even though it is globally supported." These overrides are
keyed on the test method name and can only be applied inside `instantiate_test`,
after the test has been identified by name.

Layer 1 can be replaced by proper upstream extension points. Layer 2 cannot be
eliminated — it will always require mutation of decorator state inside
`instantiate_test`. What upstream can provide for Layer 2 is a clean hook
(`_get_test_config`) and a utility that performs the mutation, rather than
leaving every backend to re-implement the fragile closure-walking logic.

---

## Candidate Features

### Feature 1 — `bypass_device_restrictions` on `DeviceTypeTestBase`

#### What it is

A boolean class attribute on `DeviceTypeTestBase`. When `True`, `@onlyOn`
skips its device check and runs the test body regardless of device. When
`False` (the default), a test decorated with `@onlyOn("cuda")` running on a
`privateuse1` device is **silently skipped** via `unittest.SkipTest` — it does
not fail, it does not run, it produces no signal at all.

```python
class onlyOn:
    def __call__(self, fn):
        @wraps(fn)
        def only_fn(slf, *args, **kwargs):
            if slf.device_type not in self.device_type:
                if getattr(slf, "bypass_device_restrictions", False):
                    return fn(slf, *args, **kwargs)   # True: run anyway
                raise unittest.SkipTest(reason)       # False (default): silent skip
            return fn(slf, *args, **kwargs)
        return only_fn
```

#### Why this is not sufficient for OOT use

`bypass_device_restrictions` is a **class-level, all-or-nothing switch**. There
is no way to say "bypass for test A but not test B." For a backend in
incremental bring-up — which is the normal state — this is too coarse. Turning
it on globally would run every `@onlyOn("cuda")` test, including ones that are
known to crash or produce wrong results.

The OOT framework uses `_OOTOnlyOnPatcher` instead, which injects
`backend_name` and `"privateuse1"` into `val.device_type` on a per-test basis,
driven by the YAML config. Only tests explicitly listed in the config get the
injection; unlisted tests remain skipped.

#### What belongs upstream

`PrivateUse1TestBase.bypass_device_restrictions` is currently hardcoded to
`False`. The upstream gap is not the mechanism (it already exists) but that
there is no way to opt in without subclassing. An env-var hook is sufficient:

**File:** `torch/testing/_internal/common_device_type.py`  
**Class:** `PrivateUse1TestBase`  
**Change type:** One-line env-var read

```python
class PrivateUse1TestBase(DeviceTypeTestBase):
    bypass_device_restrictions: bool = (
        os.getenv("PYTORCH_PRIVATEUSE1_BYPASS_DEVICE_RESTRICTIONS", "0") == "1"
    )
```

This is useful for backends that are confident enough to say "run every
device-restricted test against us." For incremental bring-up, the per-test
patching mechanism (Layer 2 / Feature 4b) is the right tool.

Already partially implemented upstream; only an env-var
hook is missing.

---

### Feature 2 — inject `"privateuse1"` into `NATIVE_DEVICES` (for tests that still use it -- we can remove it later if required)

#### What it is

`NATIVE_DEVICES` is a module-level tuple checked by `@onlyNativeDeviceTypes`
and `@onlyNativeDeviceTypesAnd` at call time:

```python
if self.device_type not in NATIVE_DEVICES:
    raise unittest.SkipTest(...)
```

`PrivateUse1TestBase.setUpClass` resets `cls.device_type` from `"privateuse1"`
to the registered backend name (e.g. `"spyre"`). `NATIVE_DEVICES` already
contains the registered name — but at test execution time `self.device_type` is
`"privateuse1"` and `NATIVE_DEVICES` contains `"spyre"`, so the check still
fails. Both strings must be present.

#### Proposed change

One line in `PrivateUse1TestBase.setUpClass`, immediately after the
`device_type` reset.

**File:** `torch/testing/_internal/common_device_type.py`  
**Method:** `PrivateUse1TestBase.setUpClass`  
**Change type:** One-line addition

```python
@classmethod
def setUpClass(cls):
    cls.device_type = torch._C._get_privateuse1_backend_name()
    ...
    cls.primary_device = f"{cls.device_type}:{cls.device_mod.current_device()}"

    if "privateuse1" not in NATIVE_DEVICES:
        NATIVE_DEVICES = NATIVE_DEVICES + ("privateuse1",)
```

---

### Feature 3 — YAML config loading hook on `PrivateUse1TestBase`

#### What it is

The OOT framework reads a YAML config (pointed to by `PYTORCH_TEST_CONFIG`) that
declares supported ops, dtypes, per-test modes, and per-test edits. The config
is loaded once and consulted on every `instantiate_test` call.

#### What belongs upstream

The YAML schema and Pydantic models do not need to move upstream — backends
should be free to use whatever config format they prefer. What belongs upstream
is the *loading hook* and the *instantiate_test override point*, so that
`PrivateUse1TestBase` is a usable base class rather than something each backend
must fully replace.

**File:** `torch/testing/_internal/common_device_type.py`  
**Class:** `PrivateUse1TestBase`  
**Change type:** New `_load_oot_config`, `_parse_oot_config`, and `instantiate_test` override points

```python
class PrivateUse1TestBase(DeviceTypeTestBase):

    _oot_config = None
    _oot_config_loaded = False

    @classmethod
    def _load_oot_config(cls):
        """Load config from PYTORCH_TEST_CONFIG env var, if set.
        Returns None if unset. Backends override _parse_oot_config to supply
        their own parser; the base implementation returns None.
        """
        if cls._oot_config_loaded:
            return cls._oot_config
        cls._oot_config_loaded = True
        path = os.getenv("PYTORCH_TEST_CONFIG")
        if path is None:
            return None
        cls._oot_config = cls._parse_oot_config(path)
        return cls._oot_config

    @classmethod
    def _parse_oot_config(cls, path: str):
        """Override to parse the config file at `path`.
        Return any object; instantiate_test will receive it via _load_oot_config.
        Base implementation returns None (no filtering).
        """
        return None

    @classmethod
    def instantiate_test(cls, name, test, *, generic_cls=None):
        config = cls._load_oot_config()
        if config is None:
            return super().instantiate_test(name, test, generic_cls=generic_cls)
        # Backends override this to apply config-driven filtering and modes.
        return super().instantiate_test(name, test, generic_cls=generic_cls)
```

**Priority:** Medium — enables the rest of the config-driven features.

---

### Feature 4 — Patcher utilities

The OOT framework contains patchers that mutate decorator state inside
`instantiate_test`. They split cleanly into two groups based on Layer 1 / Layer
2 from the mental model above.

#### 4a — Patchers that should be REPLACED by upstream extension points

These patchers exist because the upstream infrastructure has gaps. Once those
gaps are filled (Features 9, 11, 12), these patchers become unnecessary:

| Patcher | Gap it fills | Upstream replacement |
|---|---|---|
| `_OOTOnlyOnPatcher` | `@onlyOn` has no per-test opt-in | Feature 1 (`bypass`) + Feature 4b for per-test case |
| `_OOTOpListPatcher` | `@ops` has no filter hook | Feature 12 (`filter_fn` on `@ops`) |
| `_OOTModuleListPatcher` | `@modules` has no filter hook | Feature 12 (`filter_fn` on `@modules`) |
| `_OOTOpDtypeExpander` | `OpInfo` has no `dtypesIfPrivateUse1` | Feature 11 (normalization) |
| `_OOTModuleDtypePatcher` | `ModuleInfo` has no `dtypesIfPrivateUse1` | Feature 9 + 11 |
| `_OOTNativeDevicesPatcher` | `NATIVE_DEVICES` missing `"privateuse1"` | Feature 2 |

None of this patcher logic belongs upstream. The right upstream contribution is
the extension points that make the patchers unnecessary.

#### 4b — Patchers that CANNOT be eliminated: the per-test edits case

`edits.ops.include/exclude` and `edits.dtypes.include/exclude` are per-test
overrides. They cannot be expressed as a `filter_fn` on `@ops` because
`filter_fn` is a single predicate shared across all tests using that `@ops`
decorator. The override must be applied inside `instantiate_test`, after the
test method has been identified by name.

Example of what these edits express:

```yaml
# test_scalar_support uses binary_ufuncs_with_references, which excludes
# ops with no ref. gcd has no ref, so it is not in that list. But we want
# to test it anyway for this specific test.
- names:
    - TestBinaryUfuncs::test_scalar_support
  edits:
    ops:
      include:
        - name: gcd
    dtypes:
      exclude:
        - name: float32   # known crash for this (test, dtype) pair only
```

The mechanism required is:

1. A hook `_get_test_config(test_name)` on `PrivateUse1TestBase` that returns
   per-test op/dtype edits
2. A utility `_apply_test_edits(test, include_ops, exclude_ops,
   include_dtypes, exclude_dtypes)` that performs the decorator mutation

The mutation itself (walking `__wrapped__` chains to find `@ops` instances,
mutating `op_list` and `allowed_dtypes`) **cannot be eliminated** for Layer 2
because `@ops` has already been applied as a decorator by the time
`instantiate_test` runs. What upstream can provide is the utility so each
backend does not re-implement the fragile closure-walking logic independently.

**File:** `torch/testing/_internal/common_device_type.py`  
**Change type:** New hook + utility; the actual mutation logic moves from OOT into upstream

```python
class PrivateUse1TestBase(DeviceTypeTestBase):

    @classmethod
    def _get_test_config(cls, test_name: str) -> dict | None:
        """Return per-test edits for `test_name`, or None.

        Expected return shape (all fields optional):
        {
            "include_ops":    set[str],
            "exclude_ops":    set[str],
            "include_dtypes": set[torch.dtype],
            "exclude_dtypes": set[torch.dtype],
        }
        Override in your subclass to supply config-driven edits.
        """
        return None


def _apply_test_edits(
    test,
    *,
    include_ops: set[str] | None = None,
    exclude_ops: set[str] | None = None,
    include_dtypes: set[torch.dtype] | None = None,
    exclude_dtypes: set[torch.dtype] | None = None,
) -> None:
    """Mutate @ops / @modules decorator state on `test` to apply per-test edits.

    Must be called inside instantiate_test, after deepcopy but before
    _parametrize_test runs. Walks the __wrapped__ chain to locate the @ops
    or @modules instance and mutates op_list / allowed_dtypes in-place.
    """
    # ... closure-walking logic from _OOTOnlyOnPatcher / _OOTOpListPatcher ...
```

`instantiate_test` on `PrivateUse1TestBase` then calls both hooks:

```python
@classmethod
def instantiate_test(cls, name, test, *, generic_cls=None):
    config = cls._load_oot_config()
    if config is not None:
        edits = cls._get_test_config(name)
        if edits:
            _apply_test_edits(test, **edits)
    return super().instantiate_test(name, test, generic_cls=generic_cls)
```

**Priority:** High — this is the one piece of OOT patcher logic that genuinely
belongs upstream as a utility, because there is no cleaner alternative for the
per-test case.

---

### Feature 5 — `op__*` / `dtype__*` pytest markers on `@ops` and `@modules`

#### What it is

`_OOTOpMarkerPatcher` and `_OOTModuleMarkerPatcher` wrap `parametrize_fn` to
attach structured markers to each generated variant:

- `op__add`, `op__mul` — identifies which op a variant exercises
- `dtype__float16`, `dtype__int64` — identifies which dtype a variant exercises
- `module__nn_Linear` — identifies which module a variant exercises

Enables: `pytest test_ops.py -m "op__add and dtype__float16"`

#### Proposed change

Attach these markers directly inside `ops._parametrize_test` and
`modules._parametrize_test`. No patching needed — the markers become part of
the parametrizer itself.

**File:** `torch/testing/_internal/common_device_type.py` (`ops`)  
**File:** `torch/testing/_internal/common_modules.py` (`modules`)  
**Change type:** Additive — new markers on `test_wrapper`, zero cost for users
who do not use `-m op__*`

```python
# ops._parametrize_test, after constructing test_wrapper:
import pytest, re as _re

op_safe = _re.sub(r"[^a-zA-Z0-9_]", "_", op.name).strip("_")
if op_safe:
    test_wrapper = pytest.mark.__getattr__(f"op__{op_safe}")(test_wrapper)
if dtype is not None:
    dtype_safe = _re.sub(r"[^a-zA-Z0-9_]", "_",
                         str(dtype).replace("torch.", "")).strip("_")
    if dtype_safe:
        test_wrapper = pytest.mark.__getattr__(f"dtype__{dtype_safe}")(test_wrapper)
```

**Priority:** Medium — purely additive, can be gated behind
`PYTORCH_TEST_ADD_OP_MARKERS=1` initially.

---

### Feature 6 — CPU tensor move wrapper (`_OOTCpuMovePatcher`)

#### What it is

Wraps specified test class methods (e.g. `assertEqual`) to move all tensor
arguments to CPU before the call. Needed when a backend's `assertEqual`
internally dispatches to CPU-only ops.

#### Proposed change

A documented `_cpu_move_methods` class attribute on `DeviceTypeTestBase`,
populated by the backend's `TestBase`. `PrivateUse1TestBase.setUpClass`
applies the wrapping using a shared helper in `common_device_type.py`.

**File:** `torch/testing/_internal/common_device_type.py`  
**Change type:** New class attribute + helper; no behavior change unless attribute is set

```python
class DeviceTypeTestBase(TestCase):
    _cpu_move_methods: list[str] = []
```

**Priority:** Low — backend-specific enough that it can remain OOT until
multiple backends independently need it.

---

### Feature 7 — Config-driven tolerance injection (`_OOTPrecisionOverridePatcher`)

#### What it is

Injects `fn.precision_overrides` and `fn.tolerance_overrides` onto test
functions before `instantiate_test` reads them, allowing a config to specify
per-`(op, dtype)` tolerance without modifying upstream test files.

#### Proposed change

A `_get_extra_tolerance_overrides` hook on `DeviceTypeTestBase`. Called inside
`instantiate_test` and merged into `test.tolerance_overrides` before
`_apply_precision_override_for_test` runs.

**File:** `torch/testing/_internal/common_device_type.py`  
**Change type:** New override point, backward-compatible

```python
@classmethod
def _get_extra_tolerance_overrides(
    cls, test_name: str, op_name: str | None, dtype: torch.dtype | None
) -> dict[torch.dtype, tol] | None:
    """Return extra tolerance overrides, or None. Override in subclass."""
    return None
```

**Priority:** Medium — follows naturally from Feature 4b's `_get_test_config`
hook since tolerance overrides are just another per-test edit.

---

### Feature 8 — Mode-based test result control

#### What it is

The OOT framework assigns one of four modes to each test variant:

| Mode | Behavior |
|---|---|
| `mandatory_success` | Must pass |
| `xfail` | Expected to fail; suite passes either way |
| `xfail_strict` | Must fail; unexpected pass fails the suite |
| `skip` | Skipped entirely |

Modes are determined by the YAML config at variant level (`(test, op, dtype)`
granularity) and applied as pytest decorators after `instantiate_test` generates
the variant.

#### Proposed change

A `_get_test_mode` hook on `PrivateUse1TestBase`, called after each variant is
generated.

**File:** `torch/testing/_internal/common_device_type.py`  
**Class:** `PrivateUse1TestBase`  
**Change type:** New hook + wiring inside `instantiate_test` override

```python
@classmethod
def _get_test_mode(
    cls, test_name: str, op_name: str | None, dtype: torch.dtype | None
) -> str:
    """Return 'mandatory_success', 'xfail', 'xfail_strict', or 'skip'.
    Override to supply config-driven modes. Default: 'mandatory_success'.
    """
    return "mandatory_success"
```

Inside `instantiate_test`, after `setattr(cls, test_name, instantiated_test)`:

```python
mode = cls._get_test_mode(test_name, op_name=..., dtype=...)
if mode == "skip":
    setattr(cls, test_name, unittest.skip("OOT: skip")(getattr(cls, test_name)))
elif mode == "xfail":
    setattr(cls, test_name, pytest.mark.xfail(strict=False)(getattr(cls, test_name)))
elif mode == "xfail_strict":
    setattr(cls, test_name, pytest.mark.xfail(strict=True)(getattr(cls, test_name)))
```

**Priority:** High — the most immediately useful hook for any backend in
incremental bring-up.

---

### Feature 9 — `dtypesIfPrivateUse1` on `ModuleInfo`

#### What it is

`ModuleInfo.supported_dtypes` dispatches on device type but has no branch for
`privateuse1`, falling through to the generic `self.dtypes`:

```python
def supported_dtypes(self, device_type):
    if device_type == 'mps':
        return self.dtypesIfMPS
    elif device_type == 'hpu':
        return self.dtypesIfHpu
    else:
        return self.dtypes   # privateuse1 falls through here
```

#### Proposed change

Add `dtypesIfPrivateUse1=None` to `ModuleInfo.__init__` and a new branch in
`supported_dtypes`. `None` default preserves full backward compatibility.

**File:** `torch/testing/_internal/common_modules.py`  
**Change type:** New optional constructor parameter + one branch

```python
elif device_type == 'privateuse1' and self.dtypesIfPrivateUse1 is not None:
    return self.dtypesIfPrivateUse1
```

**Priority:** High — eliminates `_OOTOpDtypeExpander`'s fragile
`op_info.__dict__` write for the module path.

---

### Feature 10 — `module__*` / `dtype__*` markers on `modules._parametrize_test`

Mirrors Feature 5 exactly for the `@modules` parametrizer. See Feature 5 for
the proposed code shape.

**File:** `torch/testing/_internal/common_modules.py`  
**Priority:** Medium — ship in the same PR as Feature 5.

---

### Feature 11 — Normalize registered backend name in `supported_dtypes`

#### What it is

`PrivateUse1TestBase.setUpClass` resets `device_type` to the registered name
(e.g. `"spyre"`). `ModuleInfo.supported_dtypes` and `OpInfo.supported_dtypes`
check the literal string `"privateuse1"`. These two values are never equal, so
`dtypesIfPrivateUse1` would never be returned even after Feature 9 is applied,
unless the normalization is also added.

#### Proposed change

Two-line normalization at the top of `supported_dtypes` in both
`common_modules.py` and `common_methods_invocations.py`:

```python
def supported_dtypes(self, device_type):
    _pu1 = torch._C._get_privateuse1_backend_name()
    if device_type == _pu1:
        device_type = "privateuse1"
    # ... rest of dispatch unchanged ...
```

The `_get_privateuse1_backend_name()` call only has overhead when a
`privateuse1` backend is actually registered.

**File:** `torch/testing/_internal/common_modules.py` +
`common_methods_invocations.py`  
**Priority:** High — required for Feature 9 to work at all; ship together.

---

### Feature 12 — `filter_fn` on `@ops` and `@modules` (Layer 1 global filtering)

#### What it is

Both `@ops` and `@modules` store their item lists as instance state. The only
way to restrict which variants are generated is to mutate those lists before
`_parametrize_test` runs — which is what `_OOTOpListPatcher` and
`_OOTModuleListPatcher` do.

A `filter_fn` parameter replaces in-place mutation with a functional filter
applied at variant-generation time. This is the Layer 1 solution: "my device
globally supports this set of ops/modules."

**Important limitation:** `filter_fn` operates at parametrizer level and sees
all tests uniformly. It cannot express per-test overrides like
`edits.ops.include/exclude`. Those require Feature 4b.

**File:** `torch/testing/_internal/common_device_type.py` (`ops`) +
`torch/testing/_internal/common_modules.py` (`modules`)  
**Change type:** New optional parameter, backward-compatible (default `None`)

```python
class ops(_TestParametrizer):
    def __init__(self, op_list, *, dtypes=..., allowed_dtypes=None,
                 skip_if_dynamo=True, filter_fn=None):
        self.op_list = list(op_list)
        self.filter_fn = filter_fn
        ...

    def _parametrize_test(self, test, generic_cls, device_cls):
        op_list = self.op_list
        if self.filter_fn is not None:
            op_list = [op for op in op_list if self.filter_fn(op)]
        for op in op_list:
            ...
```

And identically for `modules`. A `PrivateUse1TestBase` subclass would supply
this at decoration time:

```python
@ops(binary_ufuncs,
     filter_fn=lambda op: op.name in cls._supported_ops())
def test_scalar_support(self, device, dtype, op):
    ...
```

**Priority:** High — eliminates `_OOTOpListPatcher` and
`_OOTModuleListPatcher` entirely for the global case.

---

## Summary Table

| # | Feature | Target file | What it replaces |
|---|---|---|---|
| 1 | `bypass_device_restrictions` env-var | `common_device_type.py` | N/A (already stubbed) | Low |
| 2 | `"privateuse1"` in `NATIVE_DEVICES` | `common_device_type.py` | `_OOTNativeDeviceTypesPatcher` | **Critical** |
| 3 | Config loading hook | `common_device_type.py` | OOT-private config loading | Medium |
| 4a | Extension points replacing patchers | multiple | `_OOTOnlyOnPatcher`, `_OOTOpListPatcher`, etc. | — (see individual features) |
| 4b | `_apply_test_edits` utility + `_get_test_config` hook | `common_device_type.py` | Per-test `edits.ops/dtypes` mutation | **High** |
| 5 | `op__*`/`dtype__*` markers on `@ops` | `common_device_type.py` | `_OOTOpMarkerPatcher` | Medium |
| 6 | CPU tensor move wrapper | `common_device_type.py` | `_OOTCpuMovePatcher` | Low |
| 7 | Tolerance injection hook | `common_device_type.py` | `_OOTPrecisionOverridePatcher` | Medium |
| 8 | Mode-based result control hook | `common_device_type.py` | OOT mode system | **High** |
| 9 | `dtypesIfPrivateUse1` on `ModuleInfo` | `common_modules.py` | `_OOTModuleDtypePatcher` / `_OOTOpDtypeExpander` | **High** |
| 10 | `module__*`/`dtype__*` markers on `@modules` | `common_modules.py` | `_OOTModuleMarkerPatcher` | Medium |
| 11 | Normalize backend name in `supported_dtypes` | `common_modules.py` + `common_methods_invocations.py` | Root cause of dtype mismatch | **High** |
| 12 | `filter_fn` on `@ops` and `@modules` | `common_device_type.py` + `common_modules.py` | `_OOTOpListPatcher`, `_OOTModuleListPatcher` (global case) | **High** |



## Some Open Thoughts / Discussions:

1. **`filter_fn` vs `filter_fn` per device type**: Should `filter_fn` on `@ops`
   receive the `device_cls` as a second argument so backends can filter
   differently per device? Or should the predicate be constructed outside with
   the device already bound? The latter is simpler and avoids coupling the
   parametrizer to the device type.

2. **`_get_test_config` return type**: A plain `dict` is flexible but untyped.
   A `TypedDict` or dataclass (`TestEdits`) would be more robust. The base class
   can define the type without depending on Pydantic.

3. **Pytest marker registration**: `op__*` / `dtype__*` markers will trigger
   pytest's unregistered-mark warning. They need to be registered either in
   `conftest.py` via `pytest_configure` or via a `filterwarnings` entry. PyTorch
   already has a `conftest.py`; adding a `pytest.ini_options.markers` pattern
   there is sufficient.

4. **`_apply_test_edits` and the `edits.ops.include` case**: When `include_ops`
   contains an op that is not in `@ops.op_list` at all (e.g. `gcd` excluded by
   `binary_ufuncs_with_references`), the utility must *inject* it by finding the
   op in `op_db` by name. This requires access to `op_db`, which is in
   `common_methods_invocations.py`. The dependency direction needs to be
   confirmed — `common_device_type.py` should not import from
   `common_methods_invocations.py`. The utility may need to live in a new
   `common_privateuse1.py` that imports from both.

5. **Thread safety of `NATIVE_DEVICES` mutation** (Feature 2): The tuple
   reassignment is not atomic but `setUpClass` runs before any test threads
   start. A comment noting this is sufficient.