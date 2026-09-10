# The Triton → KTIR Lowering Pipeline

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

> **Status:** draft — **describes the current pipeline; this flow may still
> change.** Grounded in the pass declarations in
> [`Passes.td`](https://github.com/torch-spyre/triton/blob/main/third_party/spyre/include/Dialect/KTDP/Transforms/Passes.td)
> and the backend entry point
> [`compiler.py`](https://github.com/torch-spyre/triton/blob/main/third_party/spyre/backend/compiler.py).
> Pass set, ordering, and contracts are expected to evolve as op coverage and
> the KTIR contract mature.

---

## 1. Summary

Spyre compiles Triton kernels by lowering **TTIR** (Triton IR) to **KTIR** (the
`ktdp` MLIR dialect for the IBM Spyre accelerator). This RFC documents the
lowering pipeline: the ordered set of MLIR passes that turn an optimized TTIR
module into KTIR, the per-pass input/output contract, and the invariants each
pass relies on.

The pipeline today is:

```
TTIR
  │
  ▼
LowerDescriptorMemory   tt.descriptor_* → ktdp memory ops
  │
  ▼
LowerComputeOps         tt.reduce / tt.dot / shape ops → linalg / tensor
  │
  ▼
ConvertFunctions        tt.func/return → func.*, !tt.ptr args → index
  │
  ▼
DistributeWork          tt.get_program_id → ktdp.get_compute_tile_id, fold num_programs
  │
  ▼
canonicalize + CSE
  │
  ▼
KTIR
```

`LowerDescriptorMemory` → `LowerComputeOps` → `ConvertFunctions` are grouped
behind a single entry point (`add_convert_ttir_to_ktdp`); `DistributeWork` and
the cleanup passes are added separately.

---

## 2. Motivation

There are two motivations for documenting this pipeline as an RFC:

1. **It is the contract every Spyre Triton kernel passes through.** Whatever
   produces the TTIR — a standalone hand-written kernel today, or a future
   Inductor path that emits Triton (prototyped as `SpyreTritonKernel` on the
   `nakaike/triton` branch, not yet on `main`) — reaches the device through these
   same passes. Anyone adding op coverage, debugging a lowering failure, or
   extending the KTIR contract needs the per-pass input/output guarantees written
   down.

2. **The flow is expected to change.** Op coverage is incomplete, the KTIR
   contract is still maturing, and metadata-driven extensions are in flight (e.g.
   layout metadata via `SpyreOptions`, tracked separately). A documented baseline
   makes those changes reviewable as diffs against a known pipeline rather than
   against tribal knowledge.

---

## 3. Proposed Implementation

### 3.1 Entry point

`SpyreBackend.add_stages` (in
[`compiler.py`](https://github.com/torch-spyre/triton/blob/main/third_party/spyre/backend/compiler.py))
registers two stages on top of Triton's standard frontend:

* **`ttir`** (`_make_ttir`) — standard Triton TTIR optimization passes (inliner,
  canonicalizer, combine, reorder-broadcast, CSE, symbol-DCE).
* **`ktir`** (`_make_ktir`) — the Spyre lowering, below.

`_make_ktir` builds the pass manager:

```python
spyre.passes.ttir_to_ktdp.add_convert_ttir_to_ktdp(pm)   # 3 passes (3.2–3.4)
spyre.passes.ttir_to_ktdp.add_distribute_work(pm, grid)  # 3.5
passes.common.add_canonicalizer(pm)
passes.common.add_cse(pm)
```

`grid` comes from `SpyreOptions.grid`.

Each pass peels off **one concern**; the next pass's input is the previous
pass's output. The intuition per pass (full per-op contracts live in
[`Passes.td`](https://github.com/torch-spyre/triton/blob/main/third_party/spyre/include/Dialect/KTDP/Transforms/Passes.td)):

* **`LowerDescriptorMemory` — memory.** Turns the kernel's tensor-descriptor
  loads and stores into Spyre memory views (`ktdp.construct_memory_view` +
  access tiles). This is where a logical tensor becomes a device memory access —
  and the point at which a chosen physical (stick-tiled) layout would be applied.
  Layout metadata feeding this pass is a separate RFC (torch-spyre#2600); here it
  matters as the pass's role in the pipeline.

* **`LowerComputeOps` — compute.** Turns the actual math and shape ops into
  `linalg`/`tensor`: `tt.dot` → `linalg.matmul`, `tt.reduce` → `linalg.reduce`,
  and the shape family (`reshape`, `broadcast`, `trans`, …) into their `tensor`
  equivalents.

* **`ConvertFunctions` — the function boundary.** Rewrites `tt.func`/`tt.return`
  to `func.*` and turns `!tt.ptr` arguments into `index`. Must run **after** the
  memory passes, which still need the pointer args; this is why it comes late.

* **`DistributeWork` — parallelism.** Maps `tt.get_program_id` to the Spyre core
  id (`ktdp.get_compute_tile_id`) and folds `tt.get_num_programs` against the
  `grid`. It trusts the kernel to distribute work via its own per-core loop — it
  does not synthesize a wrapping loop — and enforces a small grid contract (axes
  read densely from 0, grid rank matches).

* **Cleanup (`canonicalize` + `CSE`).** Folds the redundant arithmetic the
  lowering leaves behind. The module is now KTIR.

The net effect: **no `triton` dialect op or type survives the pipeline.**

---

## 4. Planned directions

The baseline above lowers standard, portable Triton. Two extensions are planned.

### 4.1 Spyre-only Triton

Some Spyre lowerings are cleaner if upstream Triton semantics are **relaxed for
the Spyre fork only** — behavior NVIDIA/AMD do not carry. The first concrete case
is `tl.cat` / `tl.join`: upstream rejects unequal-size concat, but Spyre can lower
the relaxed form to a *distributed* memory view across cores
([torch-spyre/triton#1](https://github.com/torch-spyre/triton/issues/1)). A
related relaxation removes the `TRITON_MAX_TENSOR_NUMEL` size check, which is a
GPU-oriented limit that does not apply to Spyre
([torch-spyre/triton#3](https://github.com/torch-spyre/triton/issues/3)).

The intuition: the lowering pipeline should be free to accept patterns that only
make sense on Spyre, so kernel authors can express disjoint-partition loads and
large tensors directly instead of through awkward workarounds. The risk is
portability — a kernel using a Spyre-only relaxation will behave differently on
upstream Triton — so these relaxations must be **surfaced to authors as
Spyre-only**, not silently accepted
([torch-spyre/triton#2](https://github.com/torch-spyre/triton/issues/2)).

### 4.2 A reduce semantic over tiles

`tt.reduce` today lowers to `linalg.reduce` — a reduction *within* a tile. Spyre
also needs reductions *across* tiles/cores (all-reduce, reduce-to-one). The KTDP
dialect now has the target ops for this:
`ktdp.inter_tile_produce` / `ktdp.inter_tile_reduce`
([ktir-mlir-frontend#25](https://github.com/torch-spyre/ktir-mlir-frontend/pull/25)).

The plan is to introduce a reduce semantic in the lowering that recognizes a
cross-tile reduction and maps it onto `inter_tile_reduce`, rather than forcing it
through the intra-tile `linalg.reduce` path. This is what connects a Triton-level
reduction across `program_id`s to the Spyre inter-core communication primitive.

---

## 5. Metrics

* **Coverage:** the set of `tt.*` ops the pipeline lowers without falling back —
  tracked against the fixtures in `third_party/spyre/test`.
* **Contract stability:** per-pass input/output contracts hold across changes; a
  pass change that breaks a downstream pass's input assumption is caught by the
  pipeline tests.
* **Round-trip correctness:** lowered KTIR for the example/fixture kernels matches
  the numerical reference.

---

## 6. Drawbacks

* **A documented pipeline can drift from the code.** The pass declarations in
  `Passes.td` are the source of truth; this RFC must be kept in sync or it
  misleads. Mitigation: treat `Passes.td` as canonical and keep this RFC to
  rationale + ordering, not a line-by-line mirror.
* **Ordering coupling.** `ConvertFunctions` must run after the memory passes;
  `DistributeWork` after functions are converted. These constraints are implicit
  in the pass order and easy to violate when adding a pass.

---

## 7. Alternatives

* **No separate document — rely on `Passes.td`.** The `.td` header already
  carries per-pass contracts. But it does not capture cross-pass ordering
  rationale or the relationship to the two kernel-generation paths, which is what
  an RFC adds.
* **A single monolithic TTIR→KTIR pass.** Rejected: the staged pipeline lets each
  pass declare a narrow input/output contract and be tested in isolation
  (individual pass bindings already exist for this).

---

## 8. Prior Art

* **`Passes.td` pipeline header**
  ([link](https://github.com/torch-spyre/triton/blob/main/third_party/spyre/include/Dialect/KTDP/Transforms/Passes.td)) —
  the canonical declaration of the passes and their contracts.
* **MLIR dialect conversion** — the staged `tt.* → ktdp/linalg/tensor/func`
  lowering follows standard MLIR progressive-lowering practice.
* **Layout metadata RFC** (torch-spyre#2600) — proposes feeding chosen layout into
  `LowerDescriptorMemory`; depends on this pipeline as its substrate.

---

## 9. How we teach this

* **Mental model:** "TTIR comes in; each pass peels off one concern — memory,
  compute, functions, work distribution — and what's left is KTIR." No `triton`
  dialect survives the pipeline.
* **Per-pass contract:** every pass states the dialects/ops it expects and what it
  leaves behind; the next pass's input is the previous pass's output.
* **Docs:** keep `Passes.td` as the canonical contract; this RFC for the
  end-to-end narrative and ordering rationale.

---

## 10. Unresolved questions

1. **Pass set / ordering changes.** Which passes are added, removed, or reordered
   as op coverage and the KTIR contract mature? This RFC documents the current
   baseline; the flow is expected to change.
2. **Relationship to metadata extensions.** How do metadata-driven passes (layout
   and future hints, via `SpyreOptions`) slot into this pipeline — as parameters
   to existing passes, or as new passes?

---

## Resolution

*Pending RFC discussion.*

### Level of Support

*5: Unclear Resolution (draft).*

#### Tracking issue

<https://github.com/torch-spyre/torch-spyre/issues/2604>
