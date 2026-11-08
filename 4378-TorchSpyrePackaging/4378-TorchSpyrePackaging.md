# RFC: Generic native-library preloading hook for OOT accelerators

**Authors:**
* Mehant Kammakomati @kmehant (IBM Research)


## Summary

PyTorch already solves "how does a device backend's native runtime `.so`s
get loaded without `LD_LIBRARY_PATH`" twice — `_preload_cuda_deps()` and
`_preload_rocm_deps()` in `torch/__init__.py` — but both are hardcoded,
vendor-specific, and private. Out-of-tree (OOT) `PrivateUse1` backends like
torch-spyre have no equivalent hook and must reimplement the same
`ctypes.CDLL(..., RTLD_GLOBAL)` preload dance themselves.

## Problem

torch-spyre's runtime (`libflex.so`, `libspyre_comms.so`, ...)
currently must be installed system-wide (via RPM + `LD_LIBRARY_PATH`) before
`import torch_spyre` works. Bundling these into the wheel would make it
self-contained, but there's no supported PyTorch API to preload sibling
package `.so`s before the backend's C extension loads — only the two
private, hardcoded implementations for CUDA and ROCm.

## Proposed upstream change

Extract the shared logic of `_preload_cuda_deps`/`_preload_rocm_deps` into a
small public utility, e.g.:

```python
# torch/utils/_preload.py
def preload_shared_libraries(
    package: str,          # e.g. "torch_spyre_runtime"
    lib_dir: str = "lib",  # relative to the package's install location
    patterns: list[str] = ("*.so*",),
    ordered: bool = False, # set True if load order matters (like cublasLt before cublas)
) -> None: ...
```

- Locate `package` via `importlib.util.find_spec` (same anchor
  `_preload_rocm_deps` already uses for `_rocm_sdk_core`).
- Glob `patterns` under `<package>/<lib_dir>` and `ctypes.CDLL(path,
  mode=ctypes.RTLD_GLOBAL)` each, in listing order (or a caller-supplied
  order).
- No-op if `package` isn't installed — safe for environments where the
  backend's native libs are already on the system loader path.

`_preload_cuda_deps`/`_preload_rocm_deps` become thin wrappers calling this,
and any `PrivateUse1` backend can call it from its own `torch.backends`
autoload entrypoint (torch-spyre already has one:
`torch_spyre = torch_spyre:_autoload`).

### Scope

Two-file diff: new `torch/utils/_preload.py` + refactor of the two existing
call sites to use it. No new subsystem, no ABI/API surface beyond one
function.

### Open questions for upstream discussion

- Should this live in `torch.utils` (public) or stay `torch._utils`
  (private, but importable by OOT backends by convention)?
- Does load ordering need to be caller-specified, or is directory-listing
  order sufficient for non-CUDA backends?

## Alternative A: no PyTorch changes, separate sibling pip package

torch-spyre can get the same self-contained-wheel benefit entirely on its
own side, today:

1. Package the extracted runtime `.so`s (`opt/ibm/spyre/{runtime,senlib,
   deeptools,spyre-comms}/lib/*.so*`) into a separate pip package, e.g.
   `torch-spyre-runtime`, installed as a sibling of `torch_spyre` in
   site-packages (mirrors TheRock's `_rocm_sdk_core` layout and `nvidia_cuda_runtime_cuXX` packages).
2. `torch-spyre-runtime` will have separate wheels for each of x86_64, aarch64, s390x or ppc64le and dynamically fetched during the pip install.
2. In `torch_spyre/__init__.py`, before `_autoload_impl()` imports
   `torch_spyre._C`, add a private preload helper:

   ```python
   import ctypes, glob, importlib.util, os

   def _preload_spyre_runtime_deps():
       spec = importlib.util.find_spec("torch_spyre_runtime")
       if spec is None or spec.origin is None:
           return  # system install path (RPM + LD_LIBRARY_PATH) — no-op
       lib_dir = os.path.join(os.path.dirname(spec.origin), "lib")
       for so in sorted(glob.glob(os.path.join(lib_dir, "*.so*"))):
           try:
               ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
           except OSError:
               pass
   ```

3. Call `_preload_spyre_runtime_deps()` at the top of `_autoload_impl()`,
   before the `torch_spyre._C` import.

This is exactly the `_preload_rocm_deps` pattern, implemented entirely
inside torch-spyre — no upstream PR needed, no dependency on the RFC being
accepted.

## Alternative B: no PyTorch changes, bundle `.so`s inside torch_spyre itself

Instead of a second pip package, ship the `.so`s directly inside the
`torch_spyre` wheel, e.g. under `torch_spyre/lib/`:

1. Copy the runtime `.so`s into `torch_spyre/lib/` at build time (or check
   them in via a `SPYRE_LIB_STAGE_DIR`-style build step) and add them to
   `package_data`/`MANIFEST.in` so `setup.py` ships them in the wheel.
2. Point `torch_spyre._C`'s loader at them:
   - **Preload** (safer, same trick as Alternative A): in
     `torch_spyre/__init__.py`, resolve the lib dir via
     `Path(__file__).parent / "lib"` (no `find_spec` needed — it's the same
     package) and `ctypes.CDLL(..., RTLD_GLOBAL)` each `.so` before
     importing `torch_spyre._C`.

`torch-spyre` as a whole will be distributed in separate wheels for each of arch like mentioned in alternative A for `torch-spyre-runtime`