# Injecting Layout Metadata into the Triton → KTIR Lowering

**Authors:**
* @fabianlim
* @mudhakar
* @lchu6
* @tnakaike
* @moriohara
* @raghukiran1224

The Spyre Triton backend referenced throughout lives in the
[`torch-spyre/triton`](https://github.com/torch-spyre/triton) fork, under
[`third_party/spyre`](https://github.com/torch-spyre/triton/tree/main/third_party/spyre).
Reference for the Path A flow:
[Triton Layout Design (nakaike)](https://github.com/tnakaike/torch-spyre/blob/nakaike/triton-layout-doc/docs/source/compiler/triton_layout.md).


> **Status:** draft — **design proposal, not yet completely verified against an
> implementation.** Grounded in reading existing code (`SpyreOptions`,
> `_make_ktir`, the exploratory `torch-spyre/nakaike/triton` plumbing, Triton's
> `compile`/`jit` seams).

---

## 1. Summary

This RFC proposes a **general channel for passing Spyre-specific metadata into
the Triton → KTIR lowering**: carry it as fields on **`SpyreOptions`**, the
Triton backend's option object that both `triton.compile` and `triton.jit`
already funnel through and that the lowering already reads.

The **immediate, motivating use case is physical tensor layout.** Spyre's lowering
needs to expand a logical 2-D tensor descriptor into the 3-D stick-tiled device
shape (e.g. a `[M, N]` fp16 tensor → `[N/64, M, 64]`). The Triton kernel source
stays **logical** — the layout is not baked into the descriptor operands, because
that form causes problems in the lowering. We add an ordered `tensor_layouts`
field, keyed by kernel argument name, that the lowering reads to perform the
expansion.

But `tensor_layouts` is only the first field. The same channel accommodates
**future lowering hints** — `LoopSpec`/`OpSpec`-sourced information the logical
Triton IR does not carry (e.g. loop tiling counts, per-core distribution such as
cross-core reduction grouping). Each is another declared field on
`SpyreOptions`, consumed by the relevant lowering pass.

The key property: `SpyreOptions` is the **one object both entry points funnel
through, and the one object the lowering already reads.** Extending it is the
minimal injection point — no new compile entry point, no new transport mechanism,
and it generalizes to metadata beyond layout.

---

## 2. Motivation

There are two ways a Triton kernel reaches the Spyre lowering:

* **Path A (Inductor).** `torch.compile` → Inductor → `SpyreTritonKernel` emits
  Triton source for the kernel.
* **Path B (standalone).** A hand-written Triton kernel compiled directly, with
  no Inductor scheduler, no generated wrapper.

**Neither path passes layout metadata into the lowering today.** Standalone
kernels (e.g. the fork's `examples/triton/softmax.py`) compile to KTIR right now
with no externally supplied layout — `LowerDescriptorMemory` derives a layout
from the logical descriptor alone. The exploratory `nakaike/triton` Path A
serializes a layout payload into Inductor's `triton_meta`.

The gap, stated precisely: **there is no channel to hand a chosen physical layout
to the lowering** — on either path. The moment a kernel needs a layout other than
what the lowering derives on its own (a specific stick dimension, a transpose,
anything the Inductor layout optimizer would otherwise pick), there is nowhere to
put it. The
[Inductor→KTIR design](https://github.com/torch-spyre/interface-specs/blob/inductor-ktir/InductorKTIR/design.md)
likewise leaves "how a Triton-for-Spyre kernel obtains its tensor layout" open.

This RFC adds that channel. The value: one mechanism serves **both** paths and
**both** entry points, so the lowering pass stays fixed and any kernel — Inductor
or standalone — can specify its layout with a single declared field.

---

## 3. Background: what the lowering needs, and why not constexpr

The Triton → KTIR lowering consumes four kinds of information. Three of them are
already present for a standalone kernel; only **physical layout** is missing:

| Information | Path A source | Path B (standalone) |
|---|---|---|
| Computation (ops) | kernel body | kernel body — present |
| Parallelism (core distribution) | `iteration_space` core divisors | launch `grid` — present |
| Tiling (block sizes) | `LoopSpec` / `OpSpec` | kernel `tl.constexpr` block sizes — present |
| **Physical layout (device shape, strides)** | `TensorArg.device_size` + `device_coordinates` | **must be provided** |

Standalone kernels use **tensor descriptors only** (no raw pointer arithmetic — a
hard constraint for Spyre), so the only gap is the physical layout. That is what
this RFC injects.

Why layout is non-trivial: Triton kernels operate on **logical** shapes, but
Spyre memory is **stick-tiled**. A `[M, N]` fp16 tensor with the stick on `N` is
physically `[N/64, M, 64]` (num_sticks × rows × intra-stick). Which dimension is
the stick dimension is a per-tensor decision. The KTIR lowering
(`LowerDescriptorMemory`) needs each tensor's device shape and the host↔device
mapping to construct `ktdp.construct_memory_view`. (BLOCK sizes on the stick
dimension must be ≥ `stick_size` and a multiple of it — 64 for fp16/bf16.)

The payload per tensor is:

* `device_size` — physical, stick-last shape (e.g. `[N/64, M, 64]`)
* `stride_map` — device-dim → host-memory offset
* `device_coordinates` — the access mapping (sympy expressions)
* `device_dtype` — Spyre hardware dtype

This is **richer than a `tl.constexpr` can carry** (it includes sympy
expressions and integer vectors, not a single scalar). It must therefore travel
as compile metadata, not as a kernel argument. It is also the same data the
existing `TensorArg` already carries on the SDSC path.

The descriptors in the Triton source stay logical (`shape=[M, K]`); baking the
3-D device shape into the descriptor operands is explicitly **not** the chosen
form — it complicates the lowering. The lowering applies the 2-D → 3-D expansion
itself, driven by the metadata.

---

## 4. Proposed Implementation

### 4.1 The channel: `SpyreOptions` as the metadata carrier

`SpyreOptions` is defined in
[`third_party/spyre/backend/compiler.py`](https://github.com/torch-spyre/triton/blob/main/third_party/spyre/backend/compiler.py)
(in the [`torch-spyre/triton`](https://github.com/torch-spyre/triton) fork) and
already carries `grid` — a structured Python value the lowering reads. The
proposal is to treat `SpyreOptions` as the **general home for lowering metadata**:
each piece of metadata the lowering needs is a declared field. The first such
field is `tensor_layouts`:

```python
@dataclass
class SpyreOptions:
    grid: Tuple[int, ...] = (32,)
    lx_size: int = 2 * 1024 * 1024
    tensor_layouts: dict = None    # ordered {arg_name: layout}, in signature order
    # future hints land here as additional fields, consumed by their passes
```

Flow into the lowering — identical in shape to how `grid` already flows:

```
options = {"tensor_layouts": {...}, "grid": (...)}     # a plain dict
            │
triton.compile(src, target=spyre, options=options)
            │
SpyreBackend.parse_options(options)   →  SpyreOptions(grid=..., tensor_layouts=...)
            │
backend.add_stages → _make_ktir(mod, metadata, options)   ← the ktir stage
            │
LowerDescriptorMemory reads options.tensor_layouts → 2-D → 3-D stick expansion
```

### 4.2 Why an ordered dict keyed by argument name

`tensor_layouts` has one entry per **tensor** argument (scalars like `M, N, K`
are skipped). It is keyed by kernel argument name and ordered in signature order,
so it carries both identifications at once:

* the **key** is the readable arg name — idiomatic Triton (`signature` and
  `triton.heuristics` key the same way);
* the **order** maps to `TensorArg.arg_index` — how Path A and the lowering
  identify tensors.

Names are authoritative; order is the bridge to the `arg_index` schema. (A plain
`dict` is insertion-ordered in Python ≥ 3.7; an ordered dict makes the intent
explicit and `str()`s deterministically for the cache key.)

### 4.3 Why `SpyreOptions` is the right seam

1. **Both entry points produce a `SpyreOptions`.**
   * `triton.compile(src, options=...)` → `parse_options(options)` → `SpyreOptions`.
   * `kernel[grid](..., tensor_layouts=...)` → `JITFunction.run` →
     `parse_options(kwargs)` → `SpyreOptions`. (`jit.run` calls `_do_compile` →
     the same `triton.compile`.)

2. **The lowering already consumes `SpyreOptions`.** `_make_ktir(mod, metadata,
   options)` reads `options.grid` today and passes it to the C++ pass. Reading
   `options.tensor_layouts` and passing it to `LowerDescriptorMemory` is the same
   move — no new plumbing into the lowering.

3. **The only change is one declared field.** Why declaration is required:
   * `triton.jit`: `JITFunction.run` raises `KeyError` for any launch kwarg that
     is neither a kernel parameter nor a `SpyreOptions` field. Without the field,
     `tensor_layouts=...` is rejected outright.
   * `triton.compile`: `parse_options` silently *drops* keys that are not
     `SpyreOptions` fields, so the payload would never reach the lowering.

   Declaring the field is **necessary and sufficient** for both surfaces.

4. **No hashability concern.** The Triton cache key uses `str(options)`, not
   `hash(options)`, so a structured (dict) payload is fine — it needs a stable
   string form, not hashability.

### 4.4 Two paths, one lowering

Both paths emit **logical** Triton source and feed the **same**
`tensor_layouts` payload (on `SpyreOptions`) to `_make_ktir`. The lowering pass is
identical. Only **who supplies the layout, and how it reaches `SpyreOptions`**,
differs:

```
Path A (Inductor):
  FX graph → Inductor → SpyreTritonKernel → Triton source (logical descriptors)
                                          → layout from FixedTiledLayout
                                          → options["tensor_layouts"]
                                          → triton.compile → _make_ktir

Path B (standalone):
  hand-written Triton kernel (logical descriptors)
                                          → layout supplied by author
                                          → options["tensor_layouts"]
                                          → triton.compile / triton.jit → _make_ktir
```

| | Path A | Path B |
|---|---|---|
| Layout source | derived from `FixedTiledLayout` (Inductor layout optimizer) | supplied by the kernel author |
| Entry point | `triton.compile` (via the Inductor wrapper) | `triton.compile` **or** `triton.jit` |
| How it reaches `SpyreOptions` | Inductor lifts it into `options` | author passes `options=` (compile) or `tensor_layouts=` launch kwarg (jit) |
| Inductor involved? | yes | **no** |

(Neither path implements this today — see §2. The table describes the proposed
end state.)

### 4.5 Path B usage

We **assume the author already has the layout payload** — the
`OpSpec`/`LoopSpec`/`TensorArg` data structures and their construction
(`SpyreKernel.create_op_spec`, `wrap_op_specs_in_loop`) and serialization already
exist in torch-spyre. The only Path B question is **how to inject it**:

```python
layouts = {"a_ptr": la, "b_ptr": lb, "c_ptr": lc}   # ordered, signature order

# Option 1 — triton.compile (compile only; tools, tests, KTIR inspection)
compiled = triton.compile(
    ASTSource(matmul_kernel, signature, constants, attrs),
    target=GPUTarget("spyre", ...),
    options={"tensor_layouts": layouts, "grid": (32,)},
)

# Option 2 — triton.jit (author runs the kernel; tensor_layouts is a launch kwarg)
matmul_kernel[grid](a, b, c, M, N, K,
                    BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
                    tensor_layouts=layouts)
```

### 4.6 Semantic equivalence

```
standalone_kernel + tensor_layouts(supplied by author)
        ≡
inductor_kernel  + tensor_layouts(built by SpyreTritonKernel)
```

Same logical Triton source. Same payload on the same
`SpyreOptions.tensor_layouts`. Same `_make_ktir`. The only thing the standalone
author supplies that Inductor supplies automatically is the layout payload.

---

## 5. Metrics

* **Lowering invariance:** the same Triton source + same `tensor_layouts`
  produces byte-identical KTIR regardless of entry point (`compile` vs `jit`) and
  regardless of path (A vs B).
* **Surface size:** the integration is a single declared field plus a read in
  `_make_ktir` — measured as net lines changed in the Triton backend.
* **Path B viability:** a hand-written Triton kernel compiles to correct KTIR and
  runs, using only `tensor_layouts` (no Inductor).

---

## 6. Drawbacks

* **Not a breaking change**, but it does add a backend option field that becomes
  part of the compile cache key (via `str(options)`). The payload's string form
  must be stable, or cache keys churn.
* **`SpyreOptions` can grow into a grab-bag.** Treating it as the metadata home
  means hints accrete as fields over time. Mitigation: a field qualifies only if
  it is metadata the *lowering* consumes (not a kernel argument, not host-side
  config), and each field pairs with the pass that reads it.
* **Author burden (Path B).** The standalone author must produce a correct layout
  payload. This RFC assumes that is given; producing it ergonomically is out of
  scope.

---

## 7. Alternatives

* **Bake the 3-D device shape into the descriptor operands** (emit
  `shape=[N/64, M, 64]` in the Triton source). This is the approach the
  `SpyreTritonKernel` / `SpyreTritonScheduling` codegen work takes, performing the
  2-D → 3-D expansion at Python codegen time — a competing idea tracked in
  [torch-spyre#2482](https://github.com/torch-spyre/torch-spyre/issues/2482). It
  makes Triton generation simpler, but emitting the device shape into the
  descriptor operands causes problems in the Triton → KTIR lowering, which is the
  motivation for this RFC's metadata-channel alternative.
* **Carry layout as `tl.constexpr` kernel arguments.** A constexpr is a scalar; it
  cannot hold `device_coordinates` (sympy expressions) or stride vectors. Rejected
  as insufficiently rich.
* **A bespoke `spyre_compile()` entry point.** A second compile path competing
  with `triton.compile`/`triton.jit`. Rejected: it forks the toolchain and the
  lowering would need a separate seam; `SpyreOptions` already unifies both
  existing entry points.
* **`triton.autotune` to carry tiling/core-division.** Autotune's purpose is
  benchmark-driven search; Spyre's splits are computed deterministically, not
  searched. Rejected.

---

## 8. Prior Art

* **Exploratory `nakaike/triton` plumbing**
  ([torch-spyre#2482](https://github.com/torch-spyre/torch-spyre/issues/2482)).
  `SpyreTritonKernel.codegen_body()` serializes a layout payload into Inductor's
  `triton_meta`, and `async_compile.triton()` lifts it into
  `triton.compile(options=...)`.
* **`grid` on `SpyreOptions`.** Already a structured Python value carried on
  `SpyreOptions` and consumed by a C++ KTIR pass (`add_distribute_work`, bound in
  [`third_party/spyre/triton_spyre.cc`](https://github.com/torch-spyre/triton/blob/main/third_party/spyre/triton_spyre.cc)).
  `tensor_layouts` follows the identical pattern.
* **`triton.heuristics` / `triton.Config`.** Triton's own mechanisms for attaching
  compile-time metadata to a `@triton.jit` function, keyed by argument name — the
  precedent for name-keyed metadata.

---

## 9. How we teach this

* **Terminology:** `tensor_layouts` — a per-input physical-layout map consumed by
  the lowering. Distinguish from `spyre_hint` (an Inductor/FX tiling annotation on
  the SDSC path) — different layer, different purpose.
* **Mental model:** "the kernel is logical; lowering metadata rides
  `SpyreOptions`; the lowering applies it." For layout specifically: authors think
  in logical shapes, the compiler handles the physical stick mapping. The same
  model extends to any future hint.
* **Docs:** a standalone-kernel guide showing both `triton.compile` and
  `triton.jit` injection.

---

## 10. Unresolved questions

1. **`LowerDescriptorMemory` does not yet consume `tensor_layouts`.** This is the
   core unimplemented piece. Today the pass derives a layout from the logical
   descriptor alone; it has no parameter for, and no logic to apply, an
   externally supplied `tensor_layouts`. The ops must be extended to take the
   layout payload and use it for the 2-D → 3-D `construct_memory_view` expansion.
   Until that lands, the channel carries metadata that nothing reads.
2. **`tensor_layouts` field type / serialization.** The cache key uses
   `str(options)`, so the payload needs a stable string form but not hashability.
   Confirm it crosses `parse_options` intact (dict-of-layout vs. the existing
   serialization wrappers `OpSpecDict` / `TensorArgDict` / `SympyExpr`), and
   confirm how a dict key maps to `TensorArg.arg_index` inside the lowering.
3. **Future hints.** Which `LoopSpec`/`OpSpec`-sourced hints beyond layout should
   this channel carry, and on what cadence — e.g. loop tiling counts, per-core
   distribution such as cross-core reduction grouping? Each is a candidate
   `SpyreOptions` field paired with its consuming pass.

---

## Resolution

*Pending RFC discussion.*

### Level of Support

*5: Unclear Resolution (draft).*

#### Tracking issue

<https://github.com/torch-spyre/torch-spyre/issues/2600>
