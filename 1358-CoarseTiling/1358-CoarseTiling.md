# Coarse-Tiling Loop IR — Design Rationale

**Authors:**
* @mudhakar
* @dgrove-oss
* @tardieu


A conceptual companion to the reference document
[coarse_tiling_loops.md](https://github.com/torch-spyre/torch-spyre/blob/main/docs/source/compiler/coarse_tiling_loops.md).
The reference doc describes *what* each attribute is and *how* the layers
wire together — optimized for the implementer who needs the
contract. This doc describes *why the design has the shape it does*:
what problem it solves, and what choices fall out from that.

> **Status:** living document.

---

## 1. What coarse tiling does

Coarse tiling is a program transformation for **working-set reduction**.
Take a sequence of ops that share an iteration-space dimension; split
that dimension into K chunks; emit the body inside a counted outer loop.
Each iteration sees a smaller per-tile working set, so most intermediate
tensors fit in scratchpad rather than HBM. Result: less off-chip
traffic, effective scratchpad utilization.

The transformation is conceptually simple. The mechanics — carrying the
"split-into-K-chunks" decision through Inductor's pipeline without
losing it — are not. The rest of this doc is about those mechanics.

---

## 2. The shape of the problem

Coarse tiling has to navigate several mismatches between its natural
shape and what the surrounding pipeline assumes. Three of them are
load-bearing for the design.

A small running example anchors the three pressures below:

```python
def f(a, b, c):
    y = a + b
    z = y * c
    return z
```

All tensors are `[1024, 4096]`. The goal is to tile this as **K=2
outer × M=4 inner** — eight iterations of a nested loop, each
computing on a `[512, 1024]` tile, with the intermediate `y` staying
in scratchpad across iterations. (The reference doc has the full
walkthrough; this doc uses the example only as a reference point.)

### Flat list vs. tree

Inductor's pipeline is flat-list-shaped end to end — every pass
operates on `list[ir.Operation]` or `list[BaseSchedulerNode]`; that's
the framework's contract. The example's `[add, mul]` is a 2-element
list at every Inductor stage. The hardware bundle, by contrast, is
tree-shaped: the same example will land as
`scf.for(K=2) { scf.for(M=4) { add; mul } }` — a tree, expected by
the MLIR affine dialect, the SDSC bundler, and the device firmware.
Coarse tiling has to bridge that.

The pressure rules out the obvious approaches. Inductor's IR can't be
made a tree without forcing every existing pass to walk a new
container — fighting the framework. Structure-building can't be
deferred to codegen-time either: late reconstruction is silently
fragile against any earlier pass that disturbs the layout (reordering,
fusing across the boundary, interleaving an unrelated op). And one
structure can't serve every layer — the scheduler dispatches on
`BaseSchedulerNode` subclasses; codegen output must be picklable (FX
graph cache) and emittable as literal Python source (generated
wrapper); no single object satisfies both.

The response: loop identity is carried as a per-op tag at Layer 1,
becomes a scheduler-native wrapper at Layer 2 in the unique window
where order is settled but the boundary can still be defended, and a
serializable tree node at Layer 3. **Three structures, one concept** —
each lives in the type system its layer demands.

### Full tensor vs. tile

Inductor's existing infrastructure assumes ops compute over their full
input tensor sizes. Coarse tiling changes that: inside the loop, ops
compute over per-tile sizes; outside the loop, consumers still expect
the full tensor. In the example, `add` and `mul` originally compute
over `[1024, 4096]`; after coarse tiling, each computes over a
`[512, 1024]` tile per iteration. But `a`, `b`, `c`, `z` are still
`[1024, 4096]` from the caller's perspective. The intermediate `y`
goes the other way — `[512, 1024]` per iteration, kept in scratchpad.
The shape mismatch propagates through `data.ranges`, layout sizes,
iteration spaces, and hardware addresses.

This pins coarse tiling's position in the pre-scheduling pipeline
between two pass groups. It must run **after layout finalization** —
those passes resolve tensor shapes into their final stick-layout
terms; without them, coarse tiling would split the wrong dimension or
produce a non-stick-aligned inner size. And it must run **before
work-division and scratchpad allocation** — those passes need to see
the per-tile (reduced) iteration space, not the full pre-tile space.
Otherwise work-division would size cores for the full range, and
scratchpad allocation would size for the full iteration space —
defeating the very working-set reduction coarse tiling exists to
enable. The slot is rigid:

```
layouts fixed   →   coarse tiling   →   work division, scratchpad
```

The response at the loop boundary is asymmetric. Inside the loop, ops
see per-tile data, and the existing pre-scheduling passes operate on
the reduced ranges — exactly what coarse tiling wants from them. At
the boundary, only the producer side (tile → full) needs explicit
shape adaptation; the consumer side (full → tile) is just addressing
arithmetic. **Addressing is cheap; conversion is expensive.**
Adaptation runs only where it's genuinely needed.

### Scoped vs. global optimization

Inductor's analysis — dependency, fusion, lowering — operates over ops
without knowing they belong to a loop. But the design needs the loop
body to be visible two different ways. For some purposes it has to be
treated as a unit: in the example, fusion of `add` and `mul` should
stay inside the loop body, never reaching across the perimeter to
something outside; the intermediate `y` needs scratchpad reuse across
iterations; the loop dispatches as a single codegen call. For others,
the inner ops still need individual treatment: scratchpad planning
sizes `y`'s buffer per-op; work-division partitions `add` and `mul`'s
iteration spaces independently; dependency analysis tracks per-op
reads and writes.

The response: make the loop wrapper **opaque** to scope-breaking
optimizations (a perimeter that fusion can't cross and scheduling
can't dissolve) while keeping the inner ops **legible** to
scope-respecting ones. Same loop, two views, depending on what's
asking.

---

## 3. The three loop identities

Across the three layers, the same concept — *"these ops are inside a
counted loop"* — wears three faces:

| Layer | Identity | Form |
|---|---|---|
| 1 | `loop_group_id` + `loop_count` + `loop_tiled_dims` on `ir.Operation` | Per-op tag |
| 2 | `CountedLoopSchedulerNode` | Scheduler-side wrapper (a perimeter) |
| 3 | `LoopSpec` | Codegen-output tree node |

### At Layer 1, the loop is a tag

`loop_group_id` is the identity element — a tuple-path like `(0,)`,
`(0, 0)`, `(0, 1)`, where each element is one nesting level. Same tuple
= same innermost group; shared prefix = siblings inside a common outer
loop. The tuple structure handles the two jobs an identity tag must do:
distinguish multiple loops in one graph, and encode nesting depth
(which a single int couldn't). The shape is the same as a filesystem
path — different positions in a tree get distinguishable names.

Today's `coarse_tile()` API only stamps tuples like `(g, 0, 0, …)` —
one chain per outer group, no siblings — but the data model and
scheduler-side reconstruction are general; sibling inner loops are
forward-looking but representable.

`loop_count` and `loop_tiled_dims` ride alongside the tag as **loop
parameters**, not identity. They're stamped on each op in a group for
convenience (so the post-fusion pass needs no side table) but logically
belong to the loop. The scheduler asserts they're consistent across ops
sharing a group.

The tag-on-each-op approach is what §2 derived from the flat-list
constraint: the IR list stays flat, the structure lives in the elements.
A side table would add a synchronization burden across every IR-mutating
pass; restructuring the list would fight the framework.

### At Layer 2, the loop is a perimeter

`CountedLoopSchedulerNode` is a `FusedSchedulerNode` subclass that wraps
a contiguous run of constituent `SchedulerNode`s. It's created by
`build_loop_scheduler_nodes` from the Layer-1 tags, and its job is to
defend the loop boundary against two threats:

| Direction | Threat | Defense |
|---|---|---|
| Inside-out | `Scheduler.process_grouped_nodes()` unpacking the wrapper (dissolution) | `unpack()` returns `[self]` |
| Outside-in | `spyre_fuse_nodes` merging an external node into the body | `can_fuse` returns `False` |

Together these draw a **perimeter** around the loop's body. Inside: the
constituent nodes. Outside: everything else. Nothing crosses in either
direction. Cross-group isolation falls out: each loop has its own
perimeter; no shared boundaries, no migration.

`FusedSchedulerNode` is the right base class because the alternative,
`GroupedSchedulerNode`, is unconditionally dissolved by
`process_grouped_nodes()` and isn't dispatched to `codegen_node()` at
all. Wrong type-system fit.

### At Layer 3, the loop is a serializable tree node

`LoopSpec` has three fields:

- **`count`** — how many iterations.
- **`body`** — what to run each iteration. A list whose elements can
  themselves be `LoopSpec`s — that recursion is how nested loops are
  represented.
- **`tiled_symbols`** — which iteration-space axes this loop sweeps.

> *For each of `count` iterations, run `body`, advancing the
> iteration-space symbols in `tiled_symbols` by one tile each step.*

Why three fields and not two: K=4 over a 2D tensor could mean 4 row
strips, 4 column strips, or 4 diagonal blocks (both axes in lockstep).
`tiled_symbols` disambiguates by recording which axes the iteration
index advances. It's a *list* because one loop level can sweep multiple
axes simultaneously (the diagonal case), and on the loop rather than the
body op because different loop levels sweep different axes — the body
is direction-agnostic.

`LoopSpec` lives in a different type system than
`CountedLoopSchedulerNode`. The codegen output must be picklable for
the FX graph cache and emittable as literal Python source for the
generated wrapper; `BaseSchedulerNode` is neither. The transition
between Layer 2 and Layer 3 happens at one specific function:
`SuperDSCScheduling._codegen_counted_loop` takes the
`CountedLoopSchedulerNode` and produces the `LoopSpec`.

(Side note: a transitional `tiled_symbols` field also exists on
`OpSpec` for legacy code paths. Conceptually it belongs only on the
loop. Known minor wart — see PR #2250 review notes.)

---

## 4. The data perimeter

§3 introduced the loop's perimeter as a control-flow concept: the
scheduler can't merge or dissolve across it. The same perimeter governs
**data flow**.

Inside the perimeter, ops produce and consume per-tile buffers. Outside,
ops expect full tensors. Wherever data flows across the boundary,
something has to make the shapes line up. That's the job of
`insert_tiling_propagation`: enforce that tile-shape doesn't leak out
as data.

The perimeter is **shape-asymmetric**. On the producer side (tile →
full), a tiled op writes per-tile data while an outside consumer wants
full data — a genuine shape mismatch that needs adaptation. On the
consumer side (full → tile), the loop body reads from full HBM tensors
using tile-sized windows via `affine.apply` — no conversion, just
addressing. **Addressing is cheap; conversion is expensive.** The
asymmetry is deliberate: only producer-side crossings need adaptation.

For each tiled `ComputedBuffer`, `insert_tiling_propagation` classifies
by consumer topology and applies the cheapest treatment that maintains
correctness:

| Case | Inside consumers | Outside consumers | Treatment |
|---|---|---|---|
| 1 | ✓ | ✗ | Mark `per_tile_fixed` — flag only, no IR change |
| 2 | ✓ | ✓ | Allocate full HBM buffer; insert a copy op (loop-tagged) that publishes each tile into the right slice |
| 3 | ✗ | ✓ | Rewire the tiled op to write directly into a full HBM buffer via `MutationLayoutSHOULDREMOVE` |

**Case 1** is where most of coarse tiling's working-set-reduction win
comes from: an intermediate `y` flowing from one tiled op to another
stays in scratchpad. The flag tells `scratchpad_planning` it can place
the buffer in LX, and tells the unroller to skip address advance and
`device_size` update for it.

**Case 2** is the explicit bridge. The inserted copy op carries the
same `loop_group_id` / `loop_count` / `loop_tiled_dims` so the scheduler
wraps it inside the same `CountedLoopSchedulerNode`; its `tiled_symbols`
machinery writes each iteration's per-tile output into the right slice
of the full buffer. After K iterations, the full buffer is complete and
ready for outside consumers. Each iteration *publishes* its tile — it's
an interleaved per-tile publish, not a single bulk copy.

**Case 3** would be Case 2 if the per-tile buffer weren't vestigial.
Since no inside consumer needs the per-tile shape, the tiled op's store
target is just redirected to the full buffer's slice. **Runtime cost of
this rewire: zero** — the op was going to write somewhere anyway, just
to a different address. That's why "rewire" is the honest term: a
metadata redirect of the write target, not added data movement. Calling
both Cases 2 and 3 "copy" would hide the cost asymmetry that's the
reason Case 3 exists as a distinct case.

A unified treatment that always allocated a full buffer and always
inserted a copy would handle all three correctly but waste HBM in Case
1 (defeating working-set reduction) and waste a copy op in Case 3.

---

## 5. The codegen seam: address binding

Once the loop has reached `LoopSpec` form at Layer 3, one question
remains: who unrolls it — the frontend, or something downstream? This
is gated by `config.unroll_loops`.

When `unroll_loops=True` (today's default), the frontend unrolls.
Each `LoopSpec(K, body)` becomes K body copies — addresses advanced by
`iter * stride`, `device_size` set to per-tile shape, `tiled_symbols`
cleared. The bundle.mlir ends up with K plain `sdsc_execute` calls,
addresses baked into each `sdsc_*.json`.

When `unroll_loops=False`, the frontend defers. Bundle.mlir keeps the
loop intact — an `scf.for` block wrapping `affine.apply` per tiled
tensor, then `sdsc_execute`. A downstream stage evaluates the address
arithmetic per iteration.

`unroll_loops=False` is strictly more capable (smaller bundle, symbolic
K, late-bound input addresses), but the backend doesn't yet fully
support the symbol-table machinery it requires, so it's opt-in.

The fork happens at one place, gated by one flag. **Nothing upstream of
`generate_bundle` knows or cares which path is active.** When backend
support lands, flip the default; `unroll_loop_specs` becomes dead code.
The frontend's internal model never has to change to accommodate
evolving backend capabilities.
