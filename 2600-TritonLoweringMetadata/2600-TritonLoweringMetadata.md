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


> **Status:** draft. The primary mechanism (`tl.spyre_tensor_layout` +
> `RewriteDescriptorLayout`) is **prototyped** in
> [torch-spyre/triton#18](https://github.com/torch-spyre/triton/issues/18)
> (pointwise) and [#19](https://github.com/torch-spyre/triton/issues/19) (matmul
> loop synthesis); output (C) physicalization is the known remaining gap. The
> `SpyreOptions` alternative in §4.5 is design-only.

---

## 1. Summary

This RFC proposes a way to **inject physical tensor layout into the Triton → KTIR
lowering** while keeping the Triton kernel source **logical**. Spyre's lowering
needs to expand a logical 2-D tensor descriptor into the 3-D stick-tiled device
shape (e.g. a `[M, N]` fp16 tensor → `[N/64, M, 64]`); the layout is not baked
into the descriptor operands (that form causes problems in the lowering).

The mechanism — **prototyped in
[torch-spyre/triton#18](https://github.com/torch-spyre/triton/issues/18) and
[#19](https://github.com/torch-spyre/triton/issues/19)** — is an **IR
annotation plus a rewrite pass**:

* **`tl.spyre_tensor_layout(desc, layout)`** — a new Triton builtin that stamps a
  descriptor's physical device layout (the OpSpec `device_coordinates` map) onto
  the `tt.make_tensor_descriptor` as **op attributes**. The kernel keeps writing
  logical `shape=[M, N]` descriptors.
* **`RewriteDescriptorLayout`** — a new pass that runs **last** in
  `add_convert_ttir_to_ktdp`, reads the annotations, and physicalizes the KTDP IR
  (memory views, access tiles, the compute chain, and — per #19 — matmul loop
  synthesis) to the device layout.

The layout thus rides *with the IR* as an annotation that the author writes
inline (or an upstream component emits mechanically), not as a hand-written 3-D
descriptor and not baked into the descriptor operands. A considered alternative —
carrying the layout on the backend's `SpyreOptions` option object — is documented
in §4.5.

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

This RFC adds that mechanism. The value: one annotation + one pass serves **both**
paths and **both** entry points, so any kernel — Inductor or standalone — can
specify its layout while keeping its source logical.

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

### 4.1 The annotation: `tl.spyre_tensor_layout`

`tl.spyre_tensor_layout(desc, layout)` stamps a descriptor's physical device
layout onto its `tt.make_tensor_descriptor`. The kernel keeps writing **logical**
descriptors; the annotation rides alongside as op attributes. It is
spyre-target-guarded (raises on any other backend).

`layout` is the OpSpec `device_coordinates` map: one entry per **physical** dim,
each either a bare `src` int (identity on that logical dim) or an `(src, op, arg)`
tuple, where `op` is `"floordiv"` / `"mod"` / `"identity"` and `arg` is the
divisor / modulus.

```python
# [M, N] tensor stick-tiled on N → device_size = [N//64, M, N%64]
tl.spyre_tensor_layout(desc, [
    (1, "floordiv", 64),   # phys dim 0: logical dim 1 (N) // 64  → stick index
    0,                     # phys dim 1: logical dim 0 (M)        (identity)
    (1, "mod", 64),        # phys dim 2: logical dim 1 (N) % 64   → stick lane
])
```

The stick dim appears twice — `floordiv` (stick index) and `mod` (lane) — exactly
the OpSpec `[floor(c/64), …, c%64]` shape.

**Carried as op attributes, not SSA values.** This is deliberate: SSA values
written `[N//64, M, N%64]` constant-fold in the static case to `[4, 512, 0]`,
erasing the stick identity (`//64`/`%64` structure gone, `512 % 64 = 0`
indistinguishable from a real coordinate). Attributes survive folding, so static
and dynamic paths carry identical metadata. (A separate op rather than a new
operand on `tt.make_tensor_descriptor`, which has `SameVariadicOperandSize` and
many consumers.)

### 4.2 The pass: `RewriteDescriptorLayout`

Runs **last** in `add_convert_ttir_to_ktdp`, after `LowerDescriptorMemory` and
`LowerComputeOps`. This ordering is deliberate: `tt.dot` must already be
`linalg.matmul` before operands are physicalized — `tt.dot` enforces 2-D operands
and would reject a rank-3 physical tensor.

For each annotated descriptor the pass:

1. rebuilds `ktdp.construct_memory_view` with the physical shape/strides
   (`floordiv` dims get extent `ceil(N/div)`, `mod` dims get the modulus, identity
   dims unchanged — e.g. `[M, N] → [ceil(N/64), M, 64]`);
2. rebuilds `ktdp.construct_access_tile` with the physical block shape and
   remapped index operands (identity / `divsi` / `remsi`);
3. retypes `ktdp.load` results and propagates the physical rank forward through
   the elementwise compute chain to `ktdp.store`;
4. erases the marker.

Descriptors without an annotation are untouched — the pass is a no-op on
unannotated kernels. `LowerDescriptorMemory` leaves an `UnrealizedConversionCast`
bridge so the marker's descriptor operand stays valid after lowering; the pass
traces it to reach the memory view.

**Matmul ([#19](https://github.com/torch-spyre/triton/issues/19)).** When a
physicalized operand feeds a `linalg.matmul`, the pass extends to three phases:
*physicalize* (above, but `retypeChain` stops at the matmul to preserve the
accumulator type), *synthesize loops* (replace the matmul with `scf.for` nests
that extract 2-D slices per stick and accumulate — handling both parallel-stick
and K-reduction layouts), then *erase markers*. Output (C) physicalization is the
known remaining gap.

### 4.3 Why an annotation + pass

* **Source stays logical.** A `[M, N]` add reads as a `[M, N]` add; the kernel is
  not rewritten for stick tiling.
* **Layout is generated, not authored.** The annotation *is* the OpSpec
  `device_coordinates`, so an upstream component (Inductor) can emit it
  mechanically, and a kernel can be retargeted to a different stick layout without
  touching its body.
* **The fixed lowering does the work.** The 2-D → 3-D expansion lives in one pass
  that both Inductor-emitted and hand-written kernels share; the descriptors they
  emit are identical (logical), and only the annotation differs in origin.

### 4.4 Two paths, one lowering

Both paths emit **logical** Triton source carrying the **same**
`tt.spyre_tensor_layout` annotation, and `RewriteDescriptorLayout` physicalizes
both identically. Only **who writes the annotation** differs:

```
Path A (Inductor):
  FX graph → Inductor → emit logical Triton source
                      + tl.spyre_tensor_layout from OpSpec device_coordinates
                      → triton.compile → RewriteDescriptorLayout

Path B (standalone):
  hand-written logical Triton kernel
                      + tl.spyre_tensor_layout written by the author
                      → triton.compile / triton.jit → RewriteDescriptorLayout
```

| | Path A | Path B |
|---|---|---|
| Annotation source | emitted from OpSpec `device_coordinates` (Inductor layout optimizer) | written inline by the kernel author |
| Entry point | `triton.compile` (via the Inductor wrapper) | `triton.compile` **or** `triton.jit` |
| Inductor involved? | yes | **no** |

(The pointwise mechanism is prototyped in #18; matmul loop synthesis in #19;
output physicalization is the known remaining gap.)

### 4.5 Path B usage and the `SpyreOptions` alternative

In Path B the author writes the annotation inline, next to the logical
descriptor:

```python
x_desc = tl.make_tensor_descriptor(x_ptr, shape=[M, N], strides=[N, 1],
                                   block_shape=[BLOCK_M, BLOCK_N])
tl.spyre_tensor_layout(x_desc, [(1, "floordiv", 64), 0, (1, "mod", 64)])
```

> ⚠️ Pass `layout` inline at the call site — binding it to a local first makes
> the `@triton.jit` code generator try to tensor-convert the keyword strings and
> fail. (Rough edge to smooth in a follow-up.)

**Alternative considered — carry the layout on `SpyreOptions`.** Rather than an
IR annotation, the layout could ride as a field on the backend's `SpyreOptions`
option object (which already carries `grid`), keyed by argument name, read by the
lowering the way `grid` is. Both `triton.compile` (via `parse_options`) and
`triton.jit` (via the launch kwarg → `parse_options`) funnel through
`SpyreOptions`, so a single declared field would serve both entry points, and the
cache key (`str(options)`) tolerates a structured payload.

The annotation approach (#18/#19) was prototyped instead because it keeps the
layout *attached to the specific descriptor in the IR* — surviving folding, and
mechanically emittable as the OpSpec `device_coordinates` — rather than as a
side-table keyed by argument name that the pass must re-associate with each
descriptor. The `SpyreOptions` route remains a viable carrier if a future need
(e.g. layout for arguments with no in-body descriptor) calls for it.

### 4.6 Semantic equivalence

```
standalone_kernel + tl.spyre_tensor_layout(written by author)
        ≡
inductor_kernel  + tl.spyre_tensor_layout(emitted from OpSpec)
```

Same logical Triton source. Same `tt.spyre_tensor_layout` annotation. Same
`RewriteDescriptorLayout` pass. The only thing the standalone author supplies
that Inductor supplies automatically is the annotation.

---

## 5. Metrics

* **Lowering invariance:** the same logical Triton source + same
  `tt.spyre_tensor_layout` annotation produces byte-identical KTIR regardless of
  entry point (`compile` vs `jit`) and regardless of path (A vs B).
* **No-op on unannotated kernels:** `RewriteDescriptorLayout` leaves a kernel
  with no annotations untouched (the existing suite passes unchanged).
* **Path B viability:** a hand-written, annotated Triton kernel compiles to
  correct KTIR and runs (no Inductor).

---

## 6. Drawbacks

* **A new `tt` op + a new pass.** `tl.spyre_tensor_layout` is Spyre-only surface
  in the Triton fork (target-guarded), and `RewriteDescriptorLayout` adds a pass
  to the pipeline. Both are scoped — the op is inert on other backends, the pass
  is a no-op without annotations.
* **Inline-only annotation.** The `layout` must be passed inline at the call site;
  a named local makes the `@triton.jit` codegen tensor-convert the keyword strings
  and fail. A rough edge to smooth.
* **Coverage gaps.** Output (C) physicalization and gather/scatter are not yet
  handled by the rewrite (see §10).
* **Author burden (Path B).** The standalone author must produce a correct layout
  payload. This RFC assumes that is given; producing it ergonomically is out of
  scope.

---

## 7. Alternatives

* **Carry the layout on `SpyreOptions`** instead of as an IR annotation — see
  §4.5. A viable carrier (one declared field serves both entry points), but the
  annotation keeps the layout attached to the specific descriptor and survives
  folding; prototyped as the primary for those reasons.
* **Bake the 3-D device shape into the descriptor operands** (emit
  `shape=[N/64, M, 64]` in the Triton source). This is the approach the
  `SpyreTritonKernel` / `SpyreTritonScheduling` codegen work takes, performing the
  2-D → 3-D expansion at Python codegen time — a competing idea tracked in
  [torch-spyre#2482](https://github.com/torch-spyre/torch-spyre/issues/2482). It
  makes Triton generation simpler, but emitting the device shape into the
  descriptor operands causes problems in the Triton → KTIR lowering — which is the
  motivation for keeping the source logical and physicalizing in a pass.
* **Carry layout as `tl.constexpr` kernel arguments.** A constexpr is a scalar; it
  cannot hold `device_coordinates` (sympy expressions) or stride vectors. Rejected
  as insufficiently rich (which is also why the annotation uses op attributes).
* **A bespoke `spyre_compile()` entry point.** A second compile path competing
  with `triton.compile`/`triton.jit`. Rejected: it forks the toolchain; the
  annotation rides the existing `triton.compile` path unchanged.

---

## 8. Prior Art

* **Exploratory `nakaike/triton` plumbing**
  ([torch-spyre#2482](https://github.com/torch-spyre/torch-spyre/issues/2482)).
  `SpyreTritonKernel.codegen_body()` serializes a layout payload into Inductor's
  `triton_meta`, and `async_compile.triton()` lifts it into
  `triton.compile(options=...)`.
* **`grid` on `SpyreOptions`.** A structured Python value carried on
  `SpyreOptions` and consumed by a C++ KTIR pass (`add_distribute_work`, bound in
  [`third_party/spyre/triton_spyre.cc`](https://github.com/torch-spyre/triton/blob/main/third_party/spyre/triton_spyre.cc)) —
  the precedent for the §4.5 `SpyreOptions` alternative.
* **`UnrealizedConversionCast` bridging.** Standard MLIR practice for keeping a
  value valid across a dialect-conversion boundary; here it keeps the marker's
  `!tt.tensordesc` operand live until `RewriteDescriptorLayout` consumes it.

---

## 9. How we teach this

* **Terminology:** `tl.spyre_tensor_layout` — a per-descriptor physical-layout
  annotation consumed by `RewriteDescriptorLayout`. Distinguish from `spyre_hint`
  (an Inductor/FX tiling annotation on the SDSC path) — different layer, different
  purpose.
* **Mental model:** "the kernel is logical; the layout rides the descriptor as an
  annotation; one pass physicalizes it." Authors think in logical shapes; the
  compiler handles the physical stick mapping.
* **Docs:** a standalone-kernel guide showing the annotation alongside a logical
  descriptor, for both `triton.compile` and `triton.jit`.

---

## 10. Unresolved questions

1. **Output (C) physicalization.** Matmul loop synthesis (#19) physicalizes
   inputs but accumulates into a logical `[M, N]` result, so an annotated output
   descriptor mismatches at the store. The fix is symmetric to the input side
   (accumulate into the physical C slice, insert back) — required for a real
   round-trip kernel.
2. **Gather/scatter.** `RewriteDescriptorLayout` handles load/store only and
   hard-errors on a gather/scatter user (unreachable today; gather is the SDSC
   path). Needs a plan to physicalize indirect access.
3. **Inline-only `layout`.** The annotation must be passed inline; a named local
   breaks `@triton.jit` codegen. Smooth this rough edge.
4. **Future hints beyond layout.** Whether other `LoopSpec`/`OpSpec`-sourced
   information the logical IR lacks (loop tiling counts, per-core distribution such
   as cross-core reduction grouping) should also ride annotations, or take the
   `SpyreOptions` route (§4.5).

---

## Resolution

*Pending RFC discussion.*

### Level of Support

*5: Unclear Resolution (draft).*

#### Tracking issue

<https://github.com/torch-spyre/torch-spyre/issues/2600>
