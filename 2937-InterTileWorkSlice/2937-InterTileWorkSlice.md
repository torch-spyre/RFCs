# Injecting Work-Slice Metadata into the Triton → KTIR Lowering

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
The mechanism in this RFC is prototyped in
[torch-spyre/triton#43](https://github.com/torch-spyre/triton/pull/43)
(`LowerInterTile` + `tl.inter_tile`), building on the cross-tile KTDP ops added in
[torch-spyre/ktir-mlir-frontend#25](https://github.com/torch-spyre/ktir-mlir-frontend/pull/25).

> **Status:** draft. The primary mechanism (`tl.inter_tile` + `WORK_SLICES`
> constexpr + `tl.wk_slice_coord`, lowered by `LowerInterTile`) is **prototyped**
> in [torch-spyre/triton#43](https://github.com/torch-spyre/triton/pull/43) for
> `all_reduce` and `reduce_to_one`; `broadcast` and `reduce_scatter` are deferred
> (their KTDP delivery ops have not landed). This RFC is the **work-slice analog**
> of [RFC 2600](https://github.com/torch-spyre/RFCs/pull/22) (physical layout):
> same shape of solution — annotate the logical kernel, physicalize in a pass —
> applied to *which tiles cooperate* rather than *how a tensor is stored*.

---

## 1. Summary

This RFC proposes a way to **inject work-slice (tile-group) metadata into the
Triton → KTIR lowering** while keeping the Triton kernel source **logical and
topology-independent**. A Spyre kernel whose iteration space is sliced across
tiles (cores) leaves each tile with a *partial* result that must be combined
across a group of cooperating tiles before the kernel finishes — a cross-tile
reduction or collective. Triton has no notion of combining values *across* tiles;
every `tt` op sees only its own tile's data.

The mechanism — **prototyped in
[torch-spyre/triton#43](https://github.com/torch-spyre/triton/pull/43)** — is a
**frontend builtin plus a lowering pass**, carrying the tile-group structure as
**compile-time metadata** (not baked into the kernel body):

* **`tl.inter_tile(x, axis, combiner, mode, *, WORK_SLICES)`** — a new Triton
  builtin that expresses *intent*: "combine `x` across the tiles that differ only
  on `axis`, with `combiner`, delivering per `mode`." It embeds the tile→slice
  map (`WORK_SLICES`) and the per-axis slice counts (`W`) as **op attributes** on
  an emitted `tt.inter_tile_reduce`.
* **`WORK_SLICES`** — a `tl.constexpr` kernel parameter: a list indexed by tile id,
  each entry a per-tile slice-index dict (e.g. `{"out": 1, "in": 0}`). It is the
  single source of truth for which tiles cooperate. It rides *with the kernel* as
  compile metadata, the way `tl.spyre_tensor_layout`'s layout rides with the
  descriptor in [RFC 2600](https://github.com/torch-spyre/RFCs/pull/22).
* **`tl.wk_slice_coord(WORK_SLICES, axis)`** — a companion builtin returning the
  current tile's slice index on `axis` as a runtime `i32`, so the kernel body
  recovers its own coordinates from `WORK_SLICES` rather than re-deriving them
  from `pid` with a hand-written `pid // N` / `pid % N` radix.
* **`LowerInterTile`** — a new pass that reads the `WORK_SLICES`/`W` attributes and
  expands each `tt.inter_tile_reduce` into the future-based KTDP producer/consumer
  pair (`ktdp.inter_tile_produce` + `ktdp.inter_tile_reduce`), resolving the
  abstract `axis` into the concrete cooperating-tile groups.

The work-slice structure thus rides *with the IR* as metadata an upstream
component can emit mechanically (the same `device_coordinates`-style data the
`LoopSpec`/`OpSpec` already carries), not as a hand-written tile-group loop
in the kernel body.

---

## 2. Motivation

There are two ways a Triton kernel reaches the Spyre lowering (the same two as
RFC 2600):

* **Path A (Inductor).** `torch.compile` → Inductor → `SpyreTritonKernel` emits
  Triton source for the kernel.
* **Path B (standalone).** A hand-written Triton kernel compiled directly, with no
  Inductor scheduler, no generated wrapper.

**Neither path has a channel to express cross-tile cooperation today.** When the
SDSC `LoopSpec`/`OpSpec` distributes a reduction across cores (split-K, a row-wise
softmax spread across column blocks, an all-reduce), the resulting per-tile
partials must be combined — but the Triton source the kernel author or Inductor
emits only describes *one tile's* computation. There is nowhere to say "these
tiles cooperate, combine their partials with `add`, deliver the result to all of
them (or to one)."

The gap, stated precisely: **there is no channel to hand the chosen tile-group
structure to the lowering** — on either path. The work division *is* known
(Inductor's layout/loop optimizer chose it; a standalone author chose it), but the
logical Triton IR cannot express it. This is the exact shape of the gap RFC 2600
identified for *physical layout*; here it is for *work slices*.

A second, related gap is **topology coupling in the kernel body.** Without a
channel, a kernel must reconstruct its own tile coordinates from the launch `pid`
with a hand-written radix (`pid_in = pid % NUM_IN_TILES`, `pid_out = pid //
NUM_IN_TILES`). That formula silently encodes the same layout the reduction
groups depend on; if the two drift apart the compiler cannot detect it.

This RFC adds the mechanism. The value: one builtin + one constexpr + one pass
serves **both** paths and **both** entry points, so any kernel — Inductor or
standalone — can specify its cross-tile cooperation while keeping its source
logical and topology-independent.

---

## 3. Background: what the lowering needs, and why not the grid

The KTDP cross-tile collective ops (`ktir-mlir-frontend#25`) are *future-based*:
a `ktdp.inter_tile_produce` publishes each tile's partial into a `tile_future` for
a set of cooperating tiles (`producer_tiles_per_group`), and a delivery op
(`ktdp.inter_tile_reduce`) consumes that future on a `consumer_tiles_per_group`,
combining the partials. Constructing these ops requires knowing, per collective:

| Information | Path A source | Path B (standalone) |
|---|---|---|
| Computation (the partial) | kernel body | kernel body — present |
| The combiner (`add`/`max`/…) | `OpSpec` reduce `type_` | `tl.inter_tile(combiner=…)` — present |
| The delivery mode | `LoopSpec` reduction kind | `tl.inter_tile(mode=…)` — present |
| **Which tiles cooperate (the tile→slice map)** | `LoopSpec` core distribution | **must be provided** |

The combiner and mode are scalar choices the `tl.inter_tile` call carries directly.
The missing piece is the **tile→slice map**: *which* tiles must combine their
partials, and how they partition into groups.

**Why the launch `grid` is not enough.** Triton's `grid` says how many program
instances run, but it carries no named work-slice axes and no tile→slice mapping.
"Reduce across `in`" is meaningless without knowing which tiles differ *only* on
`in`. Deriving groups from the bare grid shape under an assumed core ordering is
unsafe: two tiles the work division actually placed in different output slices
could be grouped together, combining unrelated partials. The lowering needs the
authoritative map.

The payload per collective is modeled on the same `OpSpec`/`LoopSpec` structures
the SDSC backend already builds in `torch-spyre` — `numWkSlicesPerDim` and
`coreIdToWkSlice` mirror the core-distribution fields the `LoopSpec` carries, so
the metadata is a re-use of an existing representation, not a new invention:

* `numWkSlicesPerDim` (`W`) — map (axis name → slice count), e.g.
  `{"out": 16, "in": 2}`
* `coreIdToWkSlice` (`C`) — a list indexed by tile id; `C[t]` is that tile's
  slice-index dict, e.g. `{"out": 8, "in": 0}`
* (optional) `depWkSlices` (`D`) — per-tile synchronization granularity (which
  producers each consumer waits on); absent ⇒ full barrier

This is **richer than a `tl.constexpr` scalar can carry** (it is a structured
list-of-maps plus a derived count map), which is exactly why — as in RFC 2600 —
it travels as op-attribute metadata rather than a kernel scalar argument. It is
also the same data the SDSC `LoopSpec` already holds about core distribution.

The kernel body stays logical: it computes one tile's partial and calls
`tl.inter_tile`. The 2-D → cross-tile expansion (producer/consumer groups, futures,
combiner regions) lives in `LowerInterTile`, driven by the metadata.

---

## 4. Proposed Implementation

### 4.1 The builtin: `tl.inter_tile`

`tl.inter_tile(partial, axis, combiner, mode, *, WORK_SLICES)` emits a single
`tt.inter_tile_reduce` op carrying the intent and the work-slice metadata. The
kernel keeps computing a **single tile's** partial; the cross-tile combine rides
as op attributes. It is spyre-target-guarded (`@requires_backend("spyre")`).

* **`partial`** — this tile's partial value. It carries a leading **unit**
  within-group axis (e.g. `tensor<1×BLOCK_M×BLOCK_N×f32>`); the delivery op
  collapses it so the result rank is one less than the partial rank (a `ktdp`
  verifier requirement).
* **`axis`** — the reduction axis: a work-slice dim name (one of the keys in
  `WORK_SLICES`, e.g. `"in"`, `"out"`). Tiles that agree on every dim *but* `axis`
  cooperate.
* **`combiner`** — `"add"` / `"max"` / `"mul"` (a shorthand with a known identity:
  `0` / `−inf` / `1`), or a custom region with explicit identity operand(s).
* **`mode`** — `"all_reduce"` (every cooperating tile receives the result) or
  `"reduce_to_one"` (only the slice-0 tile per group does). `"broadcast"` and
  `"reduce_scatter"` are reserved but deferred (§10).
* **`WORK_SLICES`** — the `tl.constexpr` tile→slice map (§4.2). The frontend
  derives `W[β] = max(C[t][β] for all t) + 1` at compile time and embeds both `C`
  and `W` as op attributes.

```python
result = tl.inter_tile(
    partial,                  # tensor<1 × BLOCK_M × BLOCK_N × f32>  (unit lead dim)
    axis="x",
    combiner="add",
    mode="all_reduce",
    WORK_SLICES=WORK_SLICES,  # the constexpr tile→slice map
)                             # tensor<BLOCK_M × BLOCK_N × f32>  (unit dim collapsed)
```

### 4.2 The metadata: `WORK_SLICES` constexpr

`WORK_SLICES` is a `tl.constexpr` kernel parameter — a **list indexed by tile id**,
each entry a per-tile slice-index dict, baked in at compile time (capitalized by
convention, like the `BLOCK_*` constexprs):

```python
# 8 tiles, 4 reduction groups × 2 within-group tiles.
#   "x" = reduction-axis slice (group label)
#   "n" = within-group slice index (which member of the group)
WORK_SLICES = [
    {"x": 0, "n": 0}, {"x": 0, "n": 1},
    {"x": 1, "n": 0}, {"x": 1, "n": 1},
    {"x": 2, "n": 0}, {"x": 2, "n": 1},
    {"x": 3, "n": 0}, {"x": 3, "n": 1},
]
```

**Both the group label *and* the within-group index are explicit.** A tile's entry
names its slice on *every* axis, not just the reduction axis: `"x"` says which
group the tile belongs to, and `"n"` says which member it is *within* that group.
This is a deliberate design requirement, not redundancy — the within-group index
is what `pick₀` (`reduce_to_one`) and `DEP_WORK_SLICES` (§4.8) key off, and what
lets `tl.wk_slice_coord` recover *any* coordinate the body needs (§4.3). A map
that named only the reduction axis would force the kernel to re-derive the
within-group position from `pid`, reintroducing exactly the topology coupling this
mechanism removes.

It is the **single source of truth** for tile cooperation. `tl.inter_tile`
receives the whole list (`C = WORK_SLICES`) and derives `W`. Two collectives with
different partitions simply pass two different `WORK_SLICES` constexprs — no
function-level annotation, mirroring how RFC 2600 attaches a *different*
`tl.spyre_tensor_layout` per descriptor.

**Carried as op attributes, not SSA values** — the same reasoning as RFC 2600's
layout. The map is structured (a list of maps) and must survive folding; an SSA
encoding would constant-fold and lose the per-tile identity. Attributes survive,
so the static and dynamic paths carry identical metadata, and an upstream
component can emit them mechanically from the `LoopSpec`.

### 4.3 The companion builtin: `tl.wk_slice_coord`

A kernel using `tl.inter_tile` still needs its *own* logical coordinates to pick
which block to load/store. `tl.wk_slice_coord(WORK_SLICES, axis)` returns the
current tile's slice index on `axis`:

```python
pid_out = tl.wk_slice_coord(WORK_SLICES, "out")   # runtime i32
pid_in  = tl.wk_slice_coord(WORK_SLICES, "in")
if pid_in == 0:                                    # == pick₀ for reduce_to_one
    c_desc.store([pid_out * BLOCK_M, 0], result)
```

It *is* `WORK_SLICES[tl.program_id(0)][axis]` — the lookup the author cannot write
directly because `program_id` is a runtime scalar, not a Python index.
`WORK_SLICES` and `axis` are `constexpr`; the **result is a runtime `i32`**, the
same kind `tl.program_id(0)` returns. (A per-tile `constexpr` is not achievable:
Triton compiles one TTIR with no per-tile Python specialization, so the lookup
cannot fold to a distinct `arith.constant` per tile.) Since the column `[ws[axis]
for ws in WORK_SLICES]` is known at compile time, it lowers to a constant table
indexed by `program_id(0)` — a `cmpi`/`select` chain — needing **no new IR op**.

This is what makes the kernel **topology-independent**: the layout radix lives only
in `WORK_SLICES` and never in a `pid // N` / `pid % N` formula in the body, so
there is nothing to drift out of sync with the groups the pass reads. The
`reduce_to_one` store guard `if tl.wk_slice_coord(...) == 0` directly asserts
slice-0 membership, matching the pass's `pick₀` by construction.

### 4.4 The pass: `LowerInterTile`

`LowerInterTile` runs in the TTIR → KTIR pipeline **after `LowerComputeOps`** (so
the partial is already a `linalg`/`tensor` value) and is **independent of the
descriptor-layout passes** (`RewriteDescriptorLayout` from RFC 2600) — the two
metadata channels compose but do not interact. It slots in before
`ConvertFunctions`.

At a high level the pass is a single walk that expands each `tt.inter_tile_reduce`
independently: it reads `W`/`C` from the op attributes, partitions the tiles into
the cooperating groups the `axis` names, and emits the future-based
`ktdp.inter_tile_produce` + `ktdp.inter_tile_reduce` pair — wiring the combiner
(shorthand `add`/`max`/`mul` or a transcribed region), selecting the consumer set
per `mode` (`all_reduce` → all members; `reduce_to_one` → the slice-0 member), and
collapsing the unit within-group axis on the result. A degenerate axis
(`W[axis] == 1`) folds away, and a module with no `tt.inter_tile_reduce` is
returned unchanged — a no-op on kernels that do not use the builtin, exactly as
`RewriteDescriptorLayout` is a no-op on unannotated kernels.

The full step-by-step lowering (group affine-set construction for both contiguous
and strided layouts, `pick₀` selection, identity materialization, result retyping)
is **prototyped and exercised end-to-end in
[torch-spyre/triton#43](https://github.com/torch-spyre/triton/pull/43)** — the
three fixtures of §4.6 each round-trip through it to numerically correct KTIR.

### 4.5 Why a builtin + constexpr + pass

* **Source stays logical and single-tile.** The kernel computes one tile's
  partial and calls `tl.inter_tile`; it is not rewritten into an explicit
  tile-group loop with hand-managed futures.
* **Cooperation is generated, not authored into the body.** `WORK_SLICES` *is* the
  `LoopSpec` core distribution, so an upstream component (Inductor) can emit it
  mechanically, and a kernel can be retargeted to a different tile partition
  without touching its body.
* **The fixed lowering does the work.** The single-tile → cross-tile expansion
  lives in one pass that both Inductor-emitted and hand-written kernels share.

This is the same three-part argument RFC 2600 makes for layout
(`tl.spyre_tensor_layout` + `RewriteDescriptorLayout`); §4.7 makes the parallel
explicit.

### 4.6 Worked examples

All three kernels below are the prototype fixtures in
[torch-spyre/triton#43](https://github.com/torch-spyre/triton/pull/43)
(`third_party/spyre/test/fixtures/inter_tile_reduce/`). **Every one recovers its
coordinates via `tl.wk_slice_coord` — none decodes `pid` by hand.**

#### Example 1 — `all_reduce` (add)

8 tiles, 4 row-groups × 2 within-group tiles. Each tile loads its column-block,
then all tiles in the same `x`-group cooperate so every tile receives the sum.

```python
@triton.jit
def inter_tile_add_kernel(x_ptr, output_ptr, M, N,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                          NUM_N_TILES: tl.constexpr,
                          WORK_SLICES: tl.constexpr):
    pid_m = tl.wk_slice_coord(WORK_SLICES, "x")   # group label
    pid_n = tl.wk_slice_coord(WORK_SLICES, "n")   # within-group column

    x_desc   = tl.make_tensor_descriptor(x_ptr,      shape=[M, N], strides=[N, 1],
                                         block_shape=[BLOCK_M, BLOCK_N])
    out_desc = tl.make_tensor_descriptor(output_ptr, shape=[M, N], strides=[N, 1],
                                         block_shape=[BLOCK_M, BLOCK_N])

    offset_m, offset_n = pid_m * BLOCK_M, pid_n * BLOCK_N
    partial_2d = x_desc.load([offset_m, offset_n])          # BLOCK_M × BLOCK_N
    partial    = tl.reshape(partial_2d, [1, BLOCK_M, BLOCK_N])  # unit lead dim

    result = tl.inter_tile(partial, axis="x", combiner="add",
                           mode="all_reduce", WORK_SLICES=WORK_SLICES)
    out_desc.store([offset_m, offset_n], result)            # BLOCK_M × BLOCK_N
```

`WORK_SLICES[t] = {"x": t // 2, "n": t % 2}`. `axis="x"` groups tiles by row
label; `W = {"x": 4, "n": 2}`, `gsize = 2`, `ngroups = 4`. Lowers to a
`ktdp.inter_tile_produce` + `ktdp.inter_tile_reduce` pair with a `linalg.add`
combiner; the `1×BLOCK_M×BLOCK_N` partial's unit axis is collapsed to
`BLOCK_M×BLOCK_N`.

#### Example 2 — `reduce_to_one` (split-K matmul)

`C[M,N] = A[M,K] @ B[K,N]`, with `K` split across `NUM_IN_TILES` K-shard tiles per
output block. Each tile accumulates its K-shard partial; `reduce_to_one` on the
`in`-axis returns the sum to `pick₀` (the `pid_in == 0` tile), which writes `C`.

```python
@triton.jit
def matmul_splitk_kernel(a_ptr, b_ptr, c_ptr, M, K, N,
                         BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
                         BLOCK_N: tl.constexpr, NUM_IN_TILES: tl.constexpr,
                         WORK_SLICES: tl.constexpr):
    pid_out = tl.wk_slice_coord(WORK_SLICES, "out")   # output block
    pid_in  = tl.wk_slice_coord(WORK_SLICES, "in")    # K-shard

    K_SHARD: tl.constexpr = K // NUM_IN_TILES // BLOCK_K
    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[K, 1],
                                       block_shape=[BLOCK_M, BLOCK_K])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[K, N], strides=[N, 1],
                                       block_shape=[BLOCK_K, BLOCK_N])
    c_desc = tl.make_tensor_descriptor(c_ptr, shape=[M, N], strides=[N, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])

    k_start = pid_in * K_SHARD
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(k_start, k_start + K_SHARD):
        a_tile = a_desc.load([pid_out * BLOCK_M, k * BLOCK_K])
        b_tile = b_desc.load([k * BLOCK_K, 0])
        acc = tl.dot(a_tile, b_tile, acc)

    partial = tl.reshape(acc, [1, BLOCK_M, BLOCK_N])
    result  = tl.inter_tile(partial, axis="out", combiner="add",
                            mode="reduce_to_one", WORK_SLICES=WORK_SLICES)

    if pid_in == 0:                                   # == pick₀
        c_desc.store([pid_out * BLOCK_M, 0], result)
```

`WORK_SLICES[t] = {"out": t // NUM_IN_TILES, "in": t % NUM_IN_TILES}` — `out`
outermost, so groups by `out` are contiguous (`out=0 → {0,1}`, `out=1 → {2,3}`).
`reduce_to_one`'s `pick₀` is the `in==0` member, and the `pid_in == 0` store guard
matches it by construction. (The production case `splitk_M64_N512_K6144` has the
same shape: 8 out-groups × 4 in-shards.)

#### Example 3 — softmax (two sequential `all_reduce`s)

Row-wise softmax distributed across 16 column-block tiles per row-group, using two
`tl.inter_tile` calls — rowmax (`combiner="max"`) then rowsum (`combiner="add"`) —
each lowered to an independent produce/reduce pair. This is the two-combiner
pattern the SDSC fuser synthesizes for a softmax distributed over its reduction
axis (`max` then `add`).

```python
@triton.jit
def softmax_inter_tile(output_ptr, input_ptr, M, N,
                       BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr,
                       NUM_MB_TILES: tl.constexpr, WORK_SLICES: tl.constexpr):
    pid_out = tl.wk_slice_coord(WORK_SLICES, "out")   # row-block
    pid_mb  = tl.wk_slice_coord(WORK_SLICES, "mb")    # column-block within row-group

    row0, col0 = pid_out * BLOCK_ROWS, pid_mb * BLOCK_COLS
    in_desc  = tl.make_tensor_descriptor(input_ptr,  shape=[M, N], strides=[N, 1],
                                         block_shape=[BLOCK_ROWS, BLOCK_COLS])
    out_desc = tl.make_tensor_descriptor(output_ptr, shape=[M, N], strides=[N, 1],
                                         block_shape=[BLOCK_ROWS, BLOCK_COLS])

    x_f32 = in_desc.load([row0, col0]).to(tl.float32)

    # all-reduce #1: true row max from per-tile column-block maxes
    partial_max = tl.reshape(tl.max(x_f32, axis=1), [1, BLOCK_ROWS])
    rowmax = tl.inter_tile(partial_max, axis="out", combiner="max",
                           mode="all_reduce", WORK_SLICES=WORK_SLICES)

    x_shifted = x_f32 - tl.reshape(rowmax, [BLOCK_ROWS, 1])
    exp_x = tl.exp(x_shifted)

    # all-reduce #2: true row sum (shares the same WORK_SLICES)
    partial_sum = tl.reshape(tl.sum(exp_x, axis=1), [1, BLOCK_ROWS])
    rowsum = tl.inter_tile(partial_sum, axis="out", combiner="add",
                           mode="all_reduce", WORK_SLICES=WORK_SLICES)

    softmax_out = exp_x / tl.reshape(rowsum, [BLOCK_ROWS, 1])
    out_desc.store([row0, col0], softmax_out.to(tl.float16))
```

`WORK_SLICES[t] = {"mb": t % 16, "out": t // 16}`, 32 tiles = 2 out-groups × 16
mb-tiles. The two `tl.inter_tile` calls share the **same** `WORK_SLICES` and lower
to **two** independent produce/reduce pairs — the evidence that multiple
collectives with the same partition compose without a function-level annotation.

### 4.7 Two paths, one lowering — and the parallel to RFC 2600

Both paths emit **logical, single-tile** Triton source carrying the **same**
`tl.inter_tile` call and `WORK_SLICES` constexpr, and `LowerInterTile` expands both
identically. Only **who writes the metadata** differs:

```
Path A (Inductor):
  FX graph → Inductor → emit logical Triton source
                      + WORK_SLICES from LoopSpec core distribution
                      → triton.compile → LowerInterTile

Path B (standalone):
  hand-written logical Triton kernel
                      + WORK_SLICES written by the author
                      → triton.compile / triton.jit → LowerInterTile
```

This RFC is the **work-slice analog** of RFC 2600. The two channels are
structurally identical and orthogonal:

| | RFC 2600 (layout) | This RFC (work slices) |
|---|---|---|
| What it injects | physical tensor layout (`device_coordinates`) | tile-group cooperation (`coreIdToWkSlice`) |
| Frontend surface | `tl.spyre_tensor_layout(desc, layout)` | `tl.inter_tile(..., WORK_SLICES=…)` + `tl.wk_slice_coord` |
| Carried as | op attributes on `tt.make_tensor_descriptor` | op attributes on `tt.inter_tile_reduce` |
| Pass | `RewriteDescriptorLayout` (runs last) | `LowerInterTile` (after `LowerComputeOps`) |
| No-op when absent | unannotated descriptors untouched | kernels with no `tt.inter_tile_reduce` untouched |
| SDSC source | `OpSpec device_coordinates` | `LoopSpec` core distribution |

A kernel can use **both** at once — physicalized descriptors *and* cross-tile
reduction — since the passes are independent (§4.4). Together they cover the two
columns RFC 2600's §3 table left as gaps for the lowering: physical layout (RFC
2600) and core distribution (this RFC).

### 4.8 Optional: per-tile synchronization (`DEP_WORK_SLICES`)

Beyond *membership*, an optional `DEP_WORK_SLICES` constexpr controls
**synchronization granularity** — which producers each consumer actually waits on
within its group (a ring or pairwise reduction instead of a full barrier). It is a
map from a consumer's within-group **local index** to the producer local indices
it depends on, embedded as a `depWkSlices` op attribute and rendered as the KTDP
delivery op's `producer_dependency_per_consumer`. **Absent ⇒ full barrier** (every
consumer waits on all producers in its group) — the default the three examples
use. This is a forward-looking knob; the prototype focuses on membership.

---

## 5. Metrics

* **Lowering invariance:** the same logical single-tile Triton source + same
  `WORK_SLICES` produces equivalent KTIR regardless of entry point (`compile` vs
  `jit`) and path (A vs B).
* **No-op on non-collective kernels:** `LowerInterTile` leaves a kernel with no
  `tt.inter_tile_reduce` untouched (the existing suite passes unchanged).
* **Topology independence:** every fixture kernel recovers its coordinates via
  `tl.wk_slice_coord` — none hard-codes a `pid // N` radix — and stays correct when
  `WORK_SLICES` changes without a body edit.
* **Path B viability:** a hand-written, `WORK_SLICES`-annotated kernel compiles to
  correct KTIR and runs numerically (all-reduce, reduce-to-one, softmax fixtures
  on the multi-core interpreter; no Inductor).

---

## 6. Drawbacks

* **A new `tt` op + two builtins + a new pass.** `tl.inter_tile` /
  `tl.wk_slice_coord` are Spyre-only surface in the Triton fork (target-guarded),
  and `LowerInterTile` adds a pass. All are scoped — inert on other backends, a
  no-op without `tt.inter_tile_reduce`.
* **Author burden (Path B).** A standalone author must produce a correct
  `WORK_SLICES` map. This RFC assumes that is given; producing it ergonomically is
  out of scope (the same posture RFC 2600 takes for the layout payload).

---

## 7. Alternatives

* **Decode tiles from `pid` by hand.** Reconstruct coordinates with
  `pid_in = pid % NUM_IN_TILES` etc., and derive reduction groups from the launch
  `grid` shape. Rejected: it couples the kernel's pid-formula to the work
  partition with no channel for the compiler to check, and the bare grid carries
  no named axes or tile→slice map, so groups cannot be safely recovered (§3).
* **Carry work slices on `SpyreOptions`.** As RFC 2600 considers for layout, the
  map could ride as a backend option field keyed by argument, read by the pass.
  Viable (one declared field serves both entry points), but the constexpr +
  op-attribute route keeps the metadata *attached to the specific collective in the
  IR*, survives folding, and is mechanically emittable from the `LoopSpec` — the
  same reasons RFC 2600 prototyped the annotation over the option. 
* **Surface the producer/consumer split at the frontend (Approach B).** Expose the
  `ktdp` produce/deliver pair directly instead of the fused `tl.inter_tile`
  metadata op. Rejected for now: the split is a lowering-internal concern; the
  fused op expresses *intent* and lets the pass choose the structure. Revisit if
  disjoint producer/consumer placement becomes a frontend need.
* **Inductor → KTIR directly** ([interface-specs#12](https://github.com/torch-spyre/interface-specs/pull/12)).
  A competing direction that bypasses Triton entirely, generating KTIR from
  Inductor. That removes the need for *any* Triton-level injection channel (layout
  or work slices) on Path A — but it does not serve Path B (standalone Triton
  kernels), and it forgoes the shared `triton.compile` toolchain. This RFC and RFC
  2600 take the position that one Triton-level channel serves both paths; #12 is
  the alternative of not routing through Triton at all. Called out as the principal
  architectural fork.

---

## 8. Prior Art

* **RFC 2600 — layout injection** ([torch-spyre/RFCs#22](https://github.com/torch-spyre/RFCs/pull/22)).
  The direct sibling: `tl.spyre_tensor_layout` + `RewriteDescriptorLayout` inject
  physical layout the same way this RFC injects work slices. The annotation +
  rewrite-pass shape, the op-attribute-over-SSA reasoning, and the two-paths-one-
  lowering argument are all borrowed from it.
* **The Triton → KTIR pipeline** ([RFC 2604](https://github.com/torch-spyre/RFCs/pull/23)).
  `LowerInterTile` slots into the documented pass pipeline after `LowerComputeOps`;
  RFC 2604 is the pipeline this pass extends.
* **Reflecting OpSpec/LoopSpec to Triton** ([RFC 2602](https://github.com/torch-spyre/RFCs/pull/24)).
  The SDSC `LoopSpec` core distribution is the upstream source of `WORK_SLICES` on
  Path A — this RFC is the channel by which that `LoopSpec` reaches the lowering.
* **Future-based KTDP collectives** ([ktir-mlir-frontend#25](https://github.com/torch-spyre/ktir-mlir-frontend/pull/25),
  design in [#23](https://github.com/torch-spyre/ktir-mlir-frontend/pull/23)). The
  `ktdp.inter_tile_produce` / `ktdp.inter_tile_reduce` ops, their group affine
  sets, and `producer_dependency_per_consumer` are the lowering target; this RFC
  pins a fixed (affine, full-barrier) subset of their generality.
* **`grid` on `SpyreOptions`.** The precedent for a structured Python value carried
  into a KTIR pass (`add_distribute_work`) — the basis for the §7 `SpyreOptions`
  alternative.

---

## 9. How we teach this

* **Terminology:** `tl.inter_tile` — a cross-tile collective expressing intent
  (axis/combiner/mode); `WORK_SLICES` — the per-tile slice map that names which
  tiles cooperate; `tl.wk_slice_coord` — recover *this* tile's coordinate from that
  map. Distinguish from `tl.spyre_tensor_layout` (RFC 2600): same idea (logical
  source, metadata-driven pass), different axis (cooperation vs. storage).
* **Mental model:** "the kernel computes one tile's partial and says *combine this
  across the group*; `WORK_SLICES` names the group; one pass builds the collective."
  Authors think in a single tile and a named axis; the compiler builds the
  producer/consumer structure.
* **Docs:** a standalone-kernel guide showing `tl.inter_tile` + `tl.wk_slice_coord`
  + a `WORK_SLICES` constexpr, with the three worked examples (all-reduce,
  reduce-to-one split-K, softmax), for both `triton.compile` and `triton.jit`.

---

## 10. Unresolved questions

1. **`broadcast` / `reduce_scatter` modes.** Reserved in the `mode` set but
   rejected with a "not yet supported" diagnostic until their KTDP delivery ops
   (`inter_tile_consume`, `inter_tile_reduce_scatter`) land. The frontend shape is
   defined so the lowering is ready when they do.
2. **Group-varying synchronization.** `DEP_WORK_SLICES` (§4.8) is keyed by
   within-group local index, so it expresses only *group-uniform* dependencies
   (ring, pairwise, full barrier). Group-dependent pairings (e.g. a butterfly)
   would need a `g`-parameterized form.

---

## Resolution

*Pending RFC discussion.*

### Level of Support

*5: Unclear Resolution (draft).*

#### Tracking issue

<https://github.com/torch-spyre/torch-spyre/issues/2937>
