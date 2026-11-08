# RFC: Pluggable Device-Transfer Hook for safetensors Custom Device Support

**Authors:**
* @amritahs-ibm

## **Summary**

Add a pluggable device-transfer hook registry to safetensors, implemented in the Rust core (`lib.rs`). Third-party device packages register a Python callable at import time via `_register_device_transfer_hook()`. A new `Device::Custom(String, Option<usize>)` variant lets any alphanumeric device string be parsed and round-tripped through the Rust `Device` enum.

With this change, `safe_open()`, `load_file()`, and `load_model()` — as well as numpy/paddle/flax/tf bindings — transparently support custom devices such as `"spyre"` without monkey-patching safetensors.

## **Motivation**

`safetensors.torch.safe_open()` only accepts device strings hard-coded in the Rust core: `"cpu"`, `"cuda"`, `"mps"`, `"npu"`, etc. Any other string is rejected by the `Device` enum's `FromPyObject` implementation before any Python code can intercept it:

```python
# Fails today on unpatched safetensors:
f = st.safe_open("model.safetensors", framework="pt", device="spyre")
# SafetensorError: device spyre is invalid
```

Without a standard extension point, device packages are forced to:
- Monkey-patch safetensors functions at import time
- Re-implement file parsing from scratch
- Bypass safetensors entirely and lose format compatibility

This is fragile, non-composable, and breaks on upstream changes. HuggingFace Transformers calls `safe_open()` internally when loading models — so there is no upstream hook available without patching. Without a standard solution, every accelerator backend creates an incompatible workaround and the ecosystem fragments.

AI accelerators (IBM Spyre, Intel Gaudi, AMD Instinct, etc.) need to integrate with standard model loading workflows. This RFC provides the single canonical mechanism for them to do so.

## **Proposed Implementation**

### Hook Registry in Rust

A `Lazy<RwLock<HashMap<String, DeviceTransferHook>>>` global (`DEVICE_HOOKS`) is added to `lib.rs`. `once_cell::sync::Lazy` + `std::sync::RwLock` allows concurrent reads (from all `get_tensor()` calls) with exclusive writes only during registration, which happens once at import time.

```rust
type DeviceTransferHook =
    Arc<dyn Fn(Py<PyAny>, &str, Py<PyAny>) -> PyResult<Py<PyAny>> + Send + Sync>;

static DEVICE_HOOKS: Lazy<RwLock<HashMap<String, DeviceTransferHook>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));
```

### `Device::Custom` Variant

A new variant is added to the `Device` enum with full `FromPyObject` / `IntoPyObject` / `Display` support:

```rust
enum Device {
    Cpu, Cuda(usize), Mps, /* ... existing variants ... */
    Custom(String, Option<usize>),  // NEW
    Anonymous(usize),
}
```

- Parsing accepts `"spyre"` or `"spyre:0"` and validates that the device-type segment is alphanumeric + underscore only.
- `Display` round-trips correctly: `"spyre:1"` → `Device::Custom("spyre", Some(1))` → `"spyre:1"`.
- All existing named device strings (`cpu`, `cuda:0`, `mps`, etc.) parse identically to before.

### Registration API

Two new `#[pyfunction]`s are exposed from Rust and re-exported through `__init__.py`:

- `_register_device_transfer_hook(device_type, hook, overwrite=False)` — validates the hook is callable, wraps it in `Arc<dyn Fn>`, stores in `DEVICE_HOOKS`. Raises `SafetensorError` if a hook is already registered for `device_type` and `overwrite=False` (the default). Pass `overwrite=True` for interactive use (e.g. re-running a notebook cell) where re-registration is expected.
- `_is_custom_device(device_type)` — query returning `bool`; used by `load_model()` to auto-select the assign path without needing a Python-side mirror dict.

### Hook Interface

Each hook is a Python callable with signature:

```python
def my_hook(cpu_tensor, name: str, device) -> tensor: ...
```

| Argument | Type | Description |
|---|---|---|
| `cpu_tensor` | Framework tensor on CPU | Zero-copy `PyMemoryView` (mmap backend) or `PyByteArray` (pread backend), wrapped into a CPU framework tensor via `create_tensor` |
| `name` | `str` | Tensor key in the safetensors file |
| `device` | `str` | Target custom device string, e.g. `"spyre:0"` |

The hook returns a tensor on the target device.

### Eager Validation at Open Time

When `Open::new()` encounters a `Device::Custom` variant, it immediately checks `DEVICE_HOOKS` and returns a clear, actionable error if no hook is registered — surfacing the "forgot to import" mistake at open time, not on the first `get_tensor()` call:

```
SafetensorError: safetensors: device type 'spyre' is not natively supported
and no transfer hook has been registered for it.
If you are using a third-party device package, make sure it is
imported before calling safe_open (e.g., 'import torch_spyre').
```

### Zero-Copy Tensor Dispatch

`Open::get_tensor()` checks for `Device::Custom` before the existing native-device dispatch table and calls `get_tensor_bytes_for_hook()`:

- **mmap backend (default)**: Uses `PyMemoryView_FromMemory` (CPython C API) to expose the OS page cache slice directly as a read-only `memoryview` — zero heap allocation. The view is wrapped into a CPU framework tensor and passed to the hook.
- **pread backend**: Allocates a `PyByteArray` and `pread`s bytes directly into it — one copy from disk, no second copy. Wrapped and passed to the hook in the same way.
- **Torch/Paddle storage**: Returns an explicit error directing the caller to use `backend='pread'` (planned follow-up).

The `torch.UntypedStorage.from_file` code path is explicitly skipped for `Device::Custom` so the mmap hook path is always taken.

### `load_model()` Assign Path

Custom device models typically start with parameters on the meta device (shape/dtype only, no storage). PyTorch's `load_state_dict()` uses `copy_()` internally — a no-op when the destination is a meta tensor on a custom device, meaning tensors silently fail to load.

A new `assign` parameter (`Optional[bool]`, defaults to `None`) and `_assign_tensors_to_model()` helper are added to `load_model()`. `None` auto-detects via `_is_custom_device()`; an explicit `True`/`False` always overrides:

```python
def load_model(model, filename, strict=True, device="cpu", *, assign=None, backend="mmap"):
    state_dict = load_file(filename, device=device, backend=backend)
    dev_type = _device_type(device) if not isinstance(device, int) else "cuda"
    if assign is None:
        _assign = _is_custom_device(dev_type)   # auto-detect
    else:
        _assign = assign                         # explicit caller override
    if _assign:
        missing, unexpected = _assign_tensors_to_model(model, state_dict, strict=False)
    else:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
```

`_assign_tensors_to_model()` directly sets `parent._parameters[attr_name]` and `parent._buffers[attr_name]` with shape/dtype validation before assignment, and correctly accounts for tied weights in `unexpected_keys`.

### Files Changed

| File | Change | +/− |
|---|---|---|
| `bindings/python/src/lib.rs` | Hook registry, `Device::Custom`, `_register_device_transfer_hook` (with `overwrite`), `_is_custom_device`, `get_tensor_bytes_for_hook`, `get_tensor` dispatch, `PySafeSlice` slice dispatch | +251 / −9 |
| `bindings/python/py_src/safetensors/torch.py` | `_device_type`, `_resolve_parent`, `_assign_tensors_to_model`, `load_model` assign path, `assign: Optional[bool]` param | +146 / −1 |
| `bindings/python/py_src/safetensors/__init__.py` | Re-export `_register_device_transfer_hook`, `_is_custom_device` | +2 / −0 |
| `bindings/python/Cargo.toml` | Add `once_cell = "1.19"` dependency | +1 / −0 |
| `bindings/python/tests/test_pt_comparison.py` | `test_spyre_slice` — slice hook dispatch tests (mmap + pread, 5 slice patterns) | +95 / −0 |

### New Symbols Added

**Rust (`lib.rs`):**
- `DEVICE_HOOKS` — global hook registry (`Lazy<RwLock<HashMap<String, DeviceTransferHook>>>`)
- `_register_device_transfer_hook(device_type, hook, overwrite=False)` — `#[pyfunction]` to register a hook
- `_is_custom_device()` — `#[pyfunction]` to query registry
- `Device::Custom(String, Option<usize>)` — new enum variant
- `Open::get_tensor_bytes_for_hook()` — zero-copy / pread byte extraction for `get_tensor` dispatch
- `PySafeSlice::name` — new field forwarding tensor key to hook
- `PySafeSlice::slice_custom_device()` — slice hook dispatch helper
- `PySafeSlice::slice_bytes_to_cpu_tensor()` — produces CPU tensor from slice bytes (avoids double-dispatch)

**Python (`torch.py`):**
- `_device_type()` — extract device type string from `str`, `int`, or `torch.device`
- `_resolve_parent()` — resolve parent module and attr name for dotted parameter paths
- `_assign_tensors_to_model()` — direct parameter/buffer assignment with shape+dtype validation

### Slice API Hook Dispatch (`get_slice`)

`PySafeSlice.__getitem__` (the code path taken when a caller uses the slice API — e.g. `f.get_slice("weight")[:, :dim]`) previously returned a CPU tensor for custom devices. A second patch adds full hook dispatch:

- `name: String` field added to `PySafeSlice` so the tensor key is forwarded to the hook.
- `slice_custom_device()` reads the requested byte ranges into a contiguous CPU buffer via `slice_bytes_to_cpu_tensor()` (which forces `Device::Cpu` to avoid double-dispatch), then calls `hook(cpu_tensor, name, device)`.
- A `Device::Custom` guard is inserted in `__getitem__` before the existing MPS guard, so custom devices are handled on all platforms including macOS/aarch64.

`slice_bytes_to_cpu_tensor()` is necessary because reusing `slice_bytes_to_tensor()` with `&self.device` would call `create_tensor(..., Custom)`, which invokes the hook inside the slice step — producing a device tensor that is then passed to the hook again, causing a "src on spyre dst on spyre" copy error.

### Consumer Implementation (torch-spyre)

The `torch-spyre` package registers the hook in `torch_spyre/model_utils.py` at import time via `_monkey_patch.py`. The hook applies the optimal `dim_order=[1,0]` DMA layout for 2D Linear weight tensors and the default contiguous layout for embeddings and normalisation weights:

```python
def spyre_layout_hook(cpu_tensor: torch.Tensor, name: str, device) -> torch.Tensor:
    if cpu_tensor.dtype != torch.float16 and cpu_tensor.dtype.is_floating_point:
        cpu_tensor = cpu_tensor.to(dtype=torch.float16)
    if not cpu_tensor.is_contiguous():
        cpu_tensor = cpu_tensor.contiguous()
    swap_dims = _should_swap_dims(name, list(cpu_tensor.shape))
    layout = SpyreTensorLayout(..., [1, 0] if swap_dims else None)
    dst = spyre_empty_with_layout(...)
    copy_tensor(cpu_tensor, dst, non_blocking=False)
    return dst

# Registered once at import time:
safetensors._register_device_transfer_hook(DEVICE_NAME, spyre_layout_hook)
```

User code requires no changes beyond importing `torch_spyre` after `torch`:

```python
import torch          # must come first — initialises _import_device_backends
import torch_spyre    # registers hook; torch is already fully initialised
import safetensors.torch as st

# Single-file model:
with st.safe_open("model.safetensors", framework="pt", device="spyre") as f:
    for key in f.keys():
        tensor = f.get_tensor(key)  # → Spyre tensor, optimal DMA layout

# Multi-shard models (iterate over each shard file):
for shard_file in shard_files:
    with st.safe_open(shard_file, framework="pt", device="spyre") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            _assign_tensor_to_model(model, key, tensor)
```

## **Metrics**

Benchmark results from Granite 3.3 8B on Spyre (8 July 2025, 3 runs per mode). **BASE** = unpatched safetensors (current main); **PATCH** = this PR.

### Tensor Loading Paths

**mmap (zero-copy) path:**
```
File → OS page cache (mmap) → PyMemoryView (zero-copy) → create_tensor (CPU) → Hook → Custom Device DMA
```

**pread path:**
```
File → PyByteArray (one pread copy) → create_tensor (CPU) → Hook → Custom Device
```

Zero overhead for native devices. The `Device::Custom` check in `get_tensor()` is a single `matches!()` that short-circuits immediately for `Cpu`, `Cuda`, `Mps`, etc. The hot path is unchanged.

### Correctness

| Mode | BASE | PATCH |
|---|---|---|
| `safe_open` | 0/3 PASS — `device spyre is invalid` | **3/3 PASS** |
| `load_model` | 0/3 PASS — `device spyre is invalid` | **3/3 PASS** |
| `load_file` | 0/3 PASS — `device spyre is invalid` | **3/3 PASS** |
| `to` (CPU load + `.to("spyre")`) | 3/3 PASS | 3/3 PASS |

`safe_open`, `load_model`, and `load_file` were completely broken on BASE. The PATCH fixes all three.

### Load Time (seconds, avg / min / max across 3 runs)

| Mode | BASE Avg | BASE Min | BASE Max | PATCH Avg | PATCH Min | PATCH Max |
|---|---|---|---|---|---|---|
| `safe_open` | N/A (ERROR) | — | — | **28.0** | 26.5 | 30.1 |
| `load_model` | N/A (ERROR) | — | — | **27.3** | 26.6 | 28.4 |
| `load_file` | N/A (ERROR) | — | — | **30.4** | 26.8 | 33.2 |
| `to` (baseline) | 39.4 | 34.8 | 44.4 | 45.4 | 36.7 | 54.2 |

The three new PATCH modes load **27–30 s** on average, compared to the `to` baseline of **39.4 s** — a **~28–31% improvement** over the only previously-working path.

### Time to First Token / Generation Throughput

| Mode | BASE Gen 1st (s) | PATCH Gen 1st (s) | BASE Gen Avg (s) | PATCH Gen Avg (s) |
|---|---|---|---|---|
| `safe_open` | N/A | **87.7** | N/A | **5.2** |
| `load_model` | N/A | **89.7** | N/A | **5.3** |
| `load_file` | N/A | **90.8** | N/A | **7.0** |
| `to` (baseline) | 95.7 | 98.0 | 5.0 | 5.8 |

First-token latency for all three new modes (**87.7–90.8 s**) is lower than the `to` baseline (**95.7 s**).

### `to` Mode Regression (only directly comparable pair)

| Metric | BASE Avg | PATCH Avg | Δ (s) | Δ (%) |
|---|---|---|---|---|
| Load time | 39.4 s | 45.4 s | +6.0 s | +15.2% |
| Time to first token | 95.7 s | 98.0 s | +2.3 s | +2.4% |
| Gen avg per token | 5.0 s | 5.8 s | +0.8 s | +16.0% |

The apparent regression in `to` mode is driven largely by run 1's outlier load of 54.2 s. PATCH minimum values (load 36.7 s, gen avg 4.8 s) closely match the BASE results, indicating run-to-run variance rather than a systematic slowdown. The `to` path itself is unmodified by this PR.

## **Drawbacks**

- **`once_cell` dependency added to Cargo.toml** — a small addition, but the crate is widely used in the Rust ecosystem.
- **Torch/Paddle storage hook path not yet supported** — opening a file with a Torch or Paddle storage backend and a custom device returns an error directing the user to `backend='pread'`. Full support is a follow-up item.
- **Not a breaking change** — `Device::Custom` is only reached when the previous match arm would have returned `Err("device X is invalid")`. All existing named device strings parse identically to before. `load_model()`'s new `assign` parameter defaults to `False`.
- **Implementation cost** — +213 lines in Rust, +138 lines in Python. The Python additions are pure helpers with no new public-facing classes.

## **Alternatives**

1. **Teach Rust about every accelerator device by name** — Doesn't scale; every new accelerator would require a safetensors release and creates tight coupling between safetensors and each hardware vendor.
2. **Always go CPU first, `.to(device)` in Python** — Works but wastes a full CPU heap allocation for every tensor and forfeits zero-copy DMA optimization for hardware that supports it.
3. **Separate `safe_open_custom()` API** — Fragments the ecosystem; callers and upstream frameworks such as HuggingFace Transformers would need to know which API to call per device type.
4. **PyTorch-level dispatch hook only** — Doesn't address the `Device` enum rejection in Rust; `safe_open(device="spyre")` still fails before any Python hook can run.
5. **Per-framework Python shadow** — A torch-only Python intercept doesn't cover numpy/paddle/flax/tf bindings and requires maintaining a parallel shadow of the Rust `Device` parsing logic.

## **Prior Art**

- **PyTorch PrivateUse1 / custom device dispatch** — PyTorch exposes a `PrivateUse1` device slot and a C++ dispatcher hook mechanism for custom hardware backends. This RFC mirrors that design at the safetensors layer, providing a similarly lightweight registration interface.
- **`torch.serialization` / `map_location`** — PyTorch's `torch.load()` accepts a `map_location` callable to redirect tensor storage during deserialization. The hook in this RFC plays the same role for the safetensors format.
- **safetensors PR #804** — <https://github.com/safetensors/safetensors/pull/804>
- **torch-spyre consumer implementation PR #2751** — <https://github.com/torch-spyre/torch-spyre/pull/2751/files>

## **How we teach this**

- **Naming**: The mechanism is called a *device transfer hook* to match the existing safetensors concept of a *device* and align with PyTorch's use of "hook" for user-registered callbacks.
- **Documentation**: The `_register_device_transfer_hook` and `_is_custom_device` functions should be documented in the safetensors Python API reference. If promoted to public API (underscore removed), they belong in the top-level `safetensors` namespace.
- **For device package authors**: A short "Custom Device Support" guide in the safetensors docs would explain the two-step pattern: register the hook in `__init__.py` at import time; implement `hook(cpu_tensor, name, device) -> tensor`. The torch-spyre PR #2751 serves as a reference implementation.
- **For end users**: No learning required — the hook is transparent. Users only need to `import torch` before `import <device_package>`. The error message when a hook is missing is actionable and includes a concrete fix suggestion.

## **Unresolved questions**

1. Should `_register_device_transfer_hook` and `_is_custom_device` be promoted to stable public API (remove leading underscore)?
2. Currently, when a file is opened with a Torch or Paddle storage backend (instead of the default `mmap` or `pread`), the hook path is not yet supported and returns an error telling the user to switch to `backend='pread'`. Should full Torch/Paddle storage support be added in a follow-up, or is this fallback message sufficient for now?
3. Should `_assign_tensors_to_model()` be exposed publicly for device packages that call it directly?

## Resolution

### Level of Support

#### Additional Context

### Next Steps

#### Tracking issue

#### Exceptions
