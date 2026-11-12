# User Directives on Values: LX Residency and Work Division

**Authors:**

* @spectre-ns

**Discussion:** [torch-spyre#6012](https://github.com/torch-spyre/torch-spyre/issues/6012)
(@isuruf, @mtsokol, @dgrove-oss, @cyang49)

**Status:** Draft. It has no RFC number yet. When it moves to
[`torch-spyre/rfcs`](https://github.com/torch-spyre/rfcs) it becomes
`NNNN-SpyreDirectives/NNNN-SpyreDirectivesRFC.md`.

## Summary

Add a value-level directive that constrains the buffer holding a tensor: where
it lives, and how the op that produces it is divided across cores.

```python
from torch_spyre._inductor import spyre_directive

scores = spyre_directive(q @ k.T, lx=True)  # must be LX-resident
scaled = spyre_directive(                    # 4 x 8 cores
    scores * scale, work_div={scores: ((0, 4), (1, 8))}
)
out = spyre_directive(scaled @ v, lx=False)  # must stay in HBM
```

* `lx=` sets **residency only**. The LX planner still chooses the address.
* `work_div=` sets the **core division of the producing op** as splits on
  axes of the op's operands, keyed by the operand tensors themselves. It
  needs no
  `declare_tensor_dim` or `name_tensor_dims`: the directive is
  self-contained.
* Every directive is **mandatory**. The compile either meets it or raises.
  Nothing is overridden silently, and there is no best-effort "preferred"
  level.
* A directive attaches to a **value**, not a scope. The buffer it names is the
  realized output of the op that produced the wrapped tensor, for that
  buffer's whole lifetime.
* Outside `torch.compile`, and on any device other than Spyre, it returns
  its argument unchanged.
* Tensors the user cannot see (restickify copies, decomposition
  intermediates, relayout copies) take no directives. The solver fills the
  gaps around the user's directives.

The residency half is the outcome the discussion in #6012 converged on. The
work-division half extends the same wrapper to the control `spyre_hint`
already gives over regions, as in @spectre-ns's first example in the thread.
`spyre_hint(work_div=...)` stays for regions. Exact LX addresses (option B in
the issue) and region-wide "no spill" constraints are out of scope; the
reasons are recorded under [Alternatives](#alternatives).

## Motivation

`spyre_hint(work_div={...})` gives users authoritative control of the core
division for the ops in a scope (see "User Work-Division Hints" in
`docs/source/compiler/work_division_planning.md`). `for_each_tile` gives
them control of coarse tiling. Users have no matching way to say where a
buffer lives. The LX planner (`LAYOUT_SOLVER`, co-optimization, LX
relayout, HBM-pool planning) decides residency on its own.

Projects such as vLLM write hand-tuned kernels for attention and other hot
paths. For them the compiler's output has to be predictable and repeatable,
not just good on average. Today:

* A kernel author cannot guarantee that a working set, such as a softmax
  running max and sum or a block reused across `for_each_tile` trips, stays
  in LX.
* The planner's choice can change when the cost model, the solver, its time
  budget or a neighbouring op changes. Performance can then regress with no
  change to the kernel.
* Fixing such a regression means changing compiler internals instead of
  annotating the kernel.

A residency directive turns "the solver happened to keep it in LX" into "the
compile guarantees it is in LX, or tells me why it cannot".

Residency and division are coupled. A buffer's per-core footprint, and
whether its readers can read it from LX at all (the `core div mismatch`
rule), depend on how its producer and readers are divided. A kernel author
who pins a buffer into LX often wants to pin the division that makes it fit
too. Today that takes two mechanisms with different targets: a
`spyre_hint` scope around the producer, with dimension names declared up
front (`declare_tensor_dim`) and attached to its inputs
(`name_tensor_dims`), plus the residency directive on the value. One wrapper
on the value says both, and reads its dimensions straight off the tensors
in the expression, so a division is written where it is used.

## Requirements

The discussion asked for requirements before the interface is settled. These
are the ones the proposal is measured against.

| ID | Requirement | Source in #6012 |
|---|---|---|
| R1 | The user can force a buffer into LX, or keep it out. | Issue |
| R2 | The directive names exactly one buffer, without a side channel that names buffers in a scope. | @mtsokol |
| R3 | The directive covers graph inputs as well as op outputs. | @mtsokol |
| R4 | The same model code runs unchanged in eager mode, on the host and on other devices. | @spectre-ns, @mtsokol |
| R5 | A directive is either met or the compile fails with a reason. No silent override. | Issue, @spectre-ns, @cyang49 |
| R6 | The directive survives later passes. The final plan is checked against it. | @mtsokol |
| R7 | The interface exposes no implementation-defined detail (LX size, stick alignment, reserved regions) that would lock in today's planner. | @spectre-ns, @cyang49 |
| R8 | The user can turn directives off without editing the model. | @cyang49 |
| R9 | Each solver can honour the directive with the machinery it already has. | Issue |
| R10 | The user can fix the core division of the op that produces one value, in the same place as its residency. | @spectre-ns |
| R11 | A value-level division and a scope-level `work_div` hint go through one validation path, so they cannot disagree on what is legal. | This RFC |
| R12 | A value-level division is self-contained: it needs no dimension declarations elsewhere in the program, and it can reach a reduction dimension. | @spectre-ns |

## Proposed Implementation

### User-facing API

```python
def spyre_directive(
    t: torch.Tensor,
    *,
    lx: bool | None = None,
    work_div: Mapping[str, AxisSplits] | None = None,
) -> torch.Tensor:
    """Return ``t``. Under torch.compile on Spyre, constrain the buffer that
    holds ``t``: its residency (``lx``) and its producer's core division
    (``work_div``). At least one key must be given."""
```

`AxisSplits` is a tuple of `(axis, split)` pairs. A single pair may be
written without the outer tuple, so `(0, 4)` means `((0, 4),)`.

* Every key is keyword-only, so later keys can join without breaking calls.
* It returns `t` itself in eager mode and under any backend other than the
  Spyre Inductor backend. The wrapper adds no copy, no sync and no
  device-specific import to the model.
* It is exported next to `spyre_hint` from `torch_spyre._inductor`.

### Semantics of `lx`

**`lx=True`.** The buffer is LX-resident from the op that writes it to its
last reader, at a planner-chosen address. If the planner cannot meet that,
the compile raises `Unsupported` with:

* the directive's source location (file and line of the wrapper call),
* the buffer name and the op that produced it,
* the reason, which is one of:
  * a residency rule from `BUFFER_RESIDENCY_RULES`
    (`scratchpad/residency_rules.py`), quoted with its reason label,
  * the capacity reason the solver already records (minimum footprint
    larger than LX, or no packing with every other forced buffer live at the
    same time),
  * a conflict with another directive or with a `work_div` hint, naming
    both.

**`lx=False`.** The buffer stays in HBM. Every buffer can live in HBM, so
this directive can always be met. An `lx=False` buffer cannot be the parent
of an LX in-place reuse; the planner gives the child its own slot or leaves
it in HBM.

**Lifetime.** The wrapper has no scope, so the directive covers the buffer's
whole lifetime. This matches the planner as it stands: a buffer is either in
LX for its whole lifetime or not at all (the "no eviction" assumption in
`docs/source/compiler/scratchpad_planning.md`). If eviction is ever added,
`lx=True` means "never evicted".

### Semantics of `work_div`

`work_div` fixes the core division of the op that writes the buffer, as a
mapping `{operand: ((axis, split), ...)}`:

```python
# x: (M, K), w: (K, N)
y = spyre_directive(x @ w, work_div={x: ((0, 2), (1, 2)), w: (1, 4)})
```

Each key is a tensor the producing op reads, the object itself. A tensor
hashes by identity, so the dictionary needs no names: the key is the same
object the expression reads, wherever it was created. Each pair splits one
axis of that tensor, counted in the tensor's own dimensions, and `1` leaves
it unsplit. Above, `M` (axis 0 of `x`) is split 2 ways, the reduction
dimension `K` (axis 1 of `x`) 2 ways and `N` (axis 1 of `w`) 4 ways: 16
cores, each computing a partial `M/2 x N/4` block over half of `K`.

Operand axes reach every loop of an op that reads a tensor: each output
dimension and each reduction dimension indexes some operand axis. Ops that
read no tensor (`zeros`, `full`, `arange`) take their division from a
`spyre_hint` scope.

Several entries can name the same loop of the op. Here `x` axis 1 and `w`
axis 0 are both `K`. Entries that resolve to the same loop must give the
same split, or the compile raises and names both; a loop named once is
enough. Loops no entry reaches stay unsplit.

A key must be the object the op reads. In `x.float() @ w` the matmul reads
the result of `x.float()`, not `x`, so `x` as a key raises ("not read by
the producing op"); bind the converted tensor first (`xf = x.float()`) and
key by `xf`. A view as a key is rejected for the same reason a directive on
a view is (see
[Which buffer a directive names](#which-buffer-a-directive-names)): key by
the tensor the view was taken of, and count axes in its dimensions. For
`x @ w.T` that means `w` with `w`'s own axes. Writing `w.T` as the key
would not even work by accident, because each `.T` is a new object.

**Validation.** The resolved splits go through the checks `_apply_user_hint`
(`work_division.py`) already runs for scope hints: every split is a positive
integer, splits evenly divide the stick-adjusted dimension size, at most one
reduction dimension is split, and the splits respect the op's legal split
domains and its memory-span floor. The one difference is strictness. A scope
hint whose next split would take the op past `SENCORES` is skipped and only
logged at INFO. A directive raises instead, because a directive never yields
silently (R5). Each failure raises `Unsupported` with the directive's source
location, like an `lx` failure.

**Resolution.** Every entry is resolved to one iteration variable of the
producing op, in the work-division pass, where those variables exist: the
variable that indexes that axis in the op's read of the operand's buffer.
An axis indexed by no single variable cannot be divided this way: for
example, one indexed by a flattened combination of several variables after
a reshape, or a broadcast axis of size 1 with a non-unit split. An operand
that the producing op does not read raises too, which happens when a
decomposition reads it only in an earlier op (see
[Which buffer a directive names](#which-buffer-a-directive-names)). Each
error names the entry and the reason.

**Which ops.** Only ops that work division plans at all: `Pointwise` and
`Reduction` IR nodes. A `work_div` on the output of an extern or fallback
kernel, or of TopK (which uses constraint-based division), raises, matching
the limitations listed in `work_division_planning.md`.

**Whole-op pin.** As with a resolved scope hint, the directive pins the
whole op to the committed division. Under co-optimization the joint solver
does not re-slice it, and dimensions the directive leaves at `1` stay
unsplit even if that leaves cores idle. The neighbours stay free. A reader
with a different division gets an LX relayout or reads from HBM, unless the
buffer also carries `lx=True` (see [Combining keys](#combining-keys)).

**Precedence with scope hints.** A value directive and a `spyre_hint` scope
can both reach the same op. Once both are resolved to iteration variables,
their splits are merged per variable. If they give different splits for the
same variable, the compile raises and names both. There is no "inner wins"
rule, so moving a wrapper into or out of a scope can never change a division
silently.

### Combining keys

`lx=True` with `work_div` is the case the coupling in the
[Motivation](#motivation) is about. The buffer is placed in LX with the
per-core blocks its pinned producer writes. Every reader must then read it
from LX: the solver must give each reader a division that matches, or
schedule an LX relayout copy for it. Without the residency directive it
could instead have pushed the buffer to HBM. If neither works, the
`core div mismatch` rule applies and the compile raises, naming the
directive and the reader.

`lx=False` with `work_div` is legal and needs no coupling: the buffer is in
HBM, so any reader division can read it.

### Global opt-out

There is no per-directive "warn" or "preferred" mode (see
[Alternatives](#alternatives)). Two environment variables drop directives
before planning, one per key:

* `SPYRE_INDUCTOR_IGNORE_LX_DIRECTIVES=1` drops every `lx` key.
* `SPYRE_INDUCTOR_IGNORE_WORK_DIV_HINTS=1`, which exists today for scope
  hints, also drops every `work_div` key on a directive. A directive's
  division is a work-division hint in every other respect, so one switch
  covers both.

This answers @cyang49's question about relaxing a directive without editing
the kernel (R8).

### Which buffer a directive names

The directive names the buffer that **holds the wrapped value**: the
realized output of the op that produced it.

**Views are rejected.** Inductor never realizes a view as its own buffer. It
inlines the view into whatever reads it, so a directive on `.T` would land
on the reader's buffer and not the producer's. @mtsokol observed exactly
this on CPU. The fold step (below) raises when the producer is a view op, and the error suggests
moving the wrapper inside the view:

```python
spyre_directive((a @ b).T, lx=True)   # raises
spyre_directive(a @ b, lx=True).T     # OK: the matmul output is in LX
```

**Decomposed ops: the last op only.** When the wrapped value comes from an
op that decomposes into several (softmax into max, subtract, exp, sum and
divide), the directive reaches only the op that writes the final value. The
intermediates are hidden tensors (below). A `work_div` entry that names the
softmax input raises, because the final divide does not read it. A user who
needs the whole decomposition divided uses a `spyre_hint` scope.

**Pointwise producers are realized.** Inductor can inline a pointwise
producer into its consumer, which would leave no buffer to place and no op
to divide. The directive's lowering realizes its input, so the wrapped value
always has a buffer and a producer.

**Graph inputs make a copy.** The caller allocates a graph input in HBM
before the program runs. `lx=True` on an input therefore means "copy the
input into LX at the start of the program, and have every reader read the
copy". AOT tracing creates fresh placeholders and drops their `custom`
meta, so the marker op (below) is not folded for an input. Instead its
lowering produces the copy, and the copy carries the directive like any op
output. The directive's copy replaces the input-boundary clone that
`ScratchpadAllocator._eligible_clone_inputs` would otherwise insert for
that input, so the input is never cloned twice.

* `lx=False` on an input is a no-op, because an input is already in HBM.
* `work_div` on an input with `lx=True` divides the copy. The copy reads
  the input, so its entries name the input itself:
  `spyre_directive(x, lx=True, work_div={x: (0, 4)})`.
* `work_div` on an input without `lx=True` raises. An input has no
  producing op to divide, and inserting a copy the user did not ask for
  would add HBM traffic.

**Hidden tensors take no directives.** Restickify copies, LX relayout
copies, decomposition intermediates and other buffers the compiler creates
cannot be named by the user. This is option 2 in the thread, which
@cyang49 agreed to: the solver fills the gaps between the buffers the user
constrained. The limitation is documented in the user guide. A primitives
module for hand-placed restickifications (option 1) stays open as future
work if real kernels need it.

### Carrying the directive through tracing

`spyre_directive` flattens `work_div` into three parallel lists (one
operand tensor, axis and split per pair) and calls a
`torch.library.custom_op` marker, `spyre::directive(t, lx, operands, axes,
splits)`, whose fake and eager kernels return their input. Because the
operands are tensor arguments, each one is an FX node from then on. A pass in `CustomPreGradPasses`
(`torch_spyre/_inductor/passes.py`, empty today) folds each marker into its
producer before AOT tracing:

1. Reject the marker if its producer is a view op.
2. Check that every operand is an argument of the producer, then copy the
   directive onto it:
   `producer.meta["custom"]["spyre_directive"] = {"lx": ..., "work_div": ...,
   "source": ...}`, with each operand recorded by a tag on its own node (or
   by input position for a graph input), not by the node itself. This is the dictionary `spyre_hint` annotations already
   live in, and AOT tracing carries it onto the ATen node
   (`copy_fx_custom_meta` in `pass_utils.py` keeps it through later
   rewrites).
3. Point the marker's users at the producer and erase the marker.

A marker whose input is a graph placeholder is left in place and lowered to
the input copy described above.

@mtsokol prototyped the `lx` key this way on CPU. After AOT tracing `aten.mm`
carries `custom={'lx_directive': True}`, the graph has no extra node, and the
stub-op variant runs end to end through Inductor with output matching
eager. `torch._dynamo.comptime` can reach the producer node during Dynamo
tracing too, but it is a private API and cannot handle inputs. The stub op
uses only mechanisms the codebase already has: marker custom ops (such as
`spyre::tile_dim_marker`) and the pre-grad pass hook.

If two directives name the same buffer, they must agree on every key they
both set, or the fold raises.

### Operand keys under tracing

A CPU probe on torch 2.14.1 confirmed that tensor keys need nothing beyond
public APIs. `model` calls a helper `attention_block(q, k)`, which computes
`s = q.sin()` and calls
`spyre_directive(s @ k, work_div={s: ((0, 2), (1, 2)), k: (1, 4)})`. Under
`torch.compile(fullgraph=True)`:

* The dictionary traces without a graph break, from inside the helper.
* The marker's operand arguments are the nodes `s`, `s` and `l_w_`, which
  are exactly the arguments of the `matmul` it wraps, with axes `[0, 1, 1]`
  and splits `[2, 2, 4]`.
* The compiled output matches eager.

In eager mode the dictionary costs a hash per key and the directive returns
its input. It still checks that each key is a tensor and each axis is in
range, so a malformed entry fails in a plain eager run.

### One accessor for work-division hints

Scope hints reach their readers through `get_op_hints(op)`, which collects
the `_hint_<id>` entries in the op's `custom` meta. Four readers each merge
the `work_div` dictionaries themselves:

* `_work_div_hint_by_name` and `has_work_div_hint` in `work_division.py`,
* `_hinted_work_div_syms` in `work_division_constraints.py`,
* `_row_hint` in `materialize_hinted_views.py`,
* the `work_div` check in `wsr/propagate_named_dims.py`.

Scope hints are keyed by dimension name and directives by operand axis, so
the two meet at iteration variables. One accessor
resolves both to `dict[Symbol, int]` for an op: scope hints through
`_resolve_work_div_hint` and `work_div_loop_info` as today, directives as
described under [Resolution](#semantics-of-work_div). It applies the
precedence rule and records which source each split came from. The readers
move onto it, and `_apply_user_hint` validates the merged result. A
directive then reaches the work-division pass, the co-optimization pin
(`has_resolved_work_div_hint` in the allocator) and the constraint
collection with no new code in any of them, and R11 holds by construction.

### Solver interaction

The allocator reads the `lx` key off each buffer's origin node when it
builds the `LifetimeBoundBuffer` list, and records it as a field next to
`residency_reason`. The `work_div` key needs nothing new here: by the time
the allocator runs, the op is pinned to its committed division like any
other resolved hint.

| Solver | `lx=False` | `lx=True` |
|---|---|---|
| All | Set `residency_reason` to a directive reason. `MemoryPlanSolver.excluded` already honours it, so no solver changes. | Check the residency rules first. A buffer barred by a rule raises before the solve. |
| CP-SAT (`CpSatLayoutSolver`) | Pinned non-resident, as every barred buffer already is. | Fix the buffer's residency literal to true. An INFEASIBLE solve raises and names the forced buffers. |
| Greedy, first-fit, best-fit | Partitioned out by `partition()`. | Place forced buffers first, then the rest. A forced buffer that does not fit raises. |
| Simulated annealing and SA co-optimizer | Partitioned out. | Forced buffers are never moved out of LX by a move. An infeasible start raises. |

Option A has no fixed address, so no solver has to solve around reserved
regions.

**Core-division coupling.** The `core div mismatch` rule is the one rule
whose outcome depends on choices the solve makes. Under co-optimization,
which is on by default, the joint solver must choose divisions for a forced
buffer's producer and readers that agree. That restricts the neighbours'
divisions, which is part of what the user asked for. Without
co-optimization the divisions are fixed before the solve, so a mismatch is
a rule violation and raises. When the producer's division is itself pinned
by a `work_div` key or a scope hint, only the readers are free.

### Interaction with later passes

The solve must satisfy each directive, and later passes must keep it
(R6). Two of the passes that run after the solve never change a buffer's
residency, so they need no change:

* LX relayout adds LX copies for readers that need a different per-core
  split. It leaves the source where it is.
* HBM-pool planning only places buffers that are not in LX.

Read-copy elision (`elide_proven_read_copies`) is the exception: it removes
read copies once the final plan proves them safe. A user who writes
`spyre_directive(x.clone(), lx=True)` asked for that copy, so the pass must
skip a copy that carries a directive.

A check after the last pass that touches LX allocations walks every
directive and raises if one no longer holds: the buffer's residency for
`lx`, and the producer's committed tensor work division for `work_div`.
That keeps the guarantee true as passes change, instead of asking each pass
to know about directives.

### Diagnostics

* At `INFO`, the compiler logs each directive with its buffer, its source
  line, the division it committed and whether it was met.
* Every failure raises (see the semantics above). There is no "applied but
  overridden" state.
* `SPYRE_DUMP_COST=1` adds a line per directive with the predicted cost of
  the plan. Cost relative to the unconstrained optimum, which @spectre-ns
  suggested, needs a second solve without the directives. It is opt-in,
  behind a separate flag, because it doubles solve time.

### Example

```python
import torch.nn.functional as F
from torch_spyre._inductor import spyre_directive


def mlp(x, w_up, w_down):
    # Keep the intermediate in LX, divided by rows so each core holds a
    # contiguous block of tokens.
    h = spyre_directive(x @ w_up, lx=True, work_div={x: (0, 32)})
    a = spyre_directive(F.silu(h), lx=True, work_div={h: (0, 32)})
    # Graph output: forbid output-boundary cloning from promoting the
    # producer to LX.
    return spyre_directive(a @ w_down, lx=False)
```

A directive on a buffer that one of the residency rules bars raises with
that rule. Inside `for_each_tile` this matters for the motivating cases:

* A buffer whose writes or reads advance with the loop variable is barred
  by `tiled (advancing)`, so a directive on a slice that walks a KV cache
  raises today.
* A counted-loop carry, such as a running max or sum, may be LX-resident
  under the `mutation target` rule's loop-carry exception.
* An operand passed whole to every trip (`dims=None`) has a fixed address
  and is not barred by either rule.

## Trade study

The interface choices discussed in #6012, scored against the
[requirements](#requirements).

### Where the directive attaches

| | Context manager (`spyre_hint(lx_residency=...)`) | Value wrapper (`spyre_directive(x, lx=...)`) |
|---|---|---|
| R2: names one buffer | No. It tags every op in scope, so it needs names for the buffers (output names, `named_dims`, variable names), and decompositions and renames break those names. | Yes. The wrapped value is the name. |
| R3: graph inputs | Only with an extra naming scheme. | Yes, through the input copy. |
| R4: runs unchanged elsewhere | Yes. | Yes. |
| Region use cases | Natural. It was motivated by coarse tiling, which needed regions (@dgrove-oss). | Needs one wrapper per value. |
| Fit with existing code | Same mechanism as `work_div`. | New marker op and pre-grad pass, built from existing parts. |

Residency is a property of a value, so the wrapper fits it. The open point
from @dgrove-oss stands: we must be sure that one value per directive covers
the use cases. See [Unresolved questions](#unresolved-questions).

### Where a work division attaches

| | Scope (`spyre_hint(work_div=...)`) | Value (`spyre_directive(x, work_div=...)`) |
|---|---|---|
| Reach | Every op in the scope, including a whole `for_each_tile` body. | The one op that writes the value. |
| Dimension names | Needs `declare_tensor_dim` and `name_tensor_dims` on the inputs. | None. Dimensions come from the tensors in the expression. |
| Reduction dimensions | Yes, by name. | Yes, through an operand axis. |
| With residency | A separate annotation in a different place. | The same call, so residency and the division that makes it fit are read together (R10). |
| Ops in scope that lack the dimension | Get a default split, silently. | Not possible: the directive names one op, and a dimension it cannot resolve raises. |
| Over `SENCORES` | Split skipped, INFO log. | Raises. |

The two are not alternatives. A scope is the tool for regions and
`for_each_tile` bodies, which is the case @dgrove-oss raised. A value
directive is the tool for one buffer whose division matters, most often
because its residency depends on it. Both resolve through the same accessor
and `_apply_user_hint`, so they cannot drift apart (R11).

### How a work division names its dimensions

| | Dimension names (`spyre_hint`) | Tensor keys (proposed) | Variable-name strings | Operand positions |
|---|---|---|---|---|
| Example | `{"M": 2, "K": 2}` after `name_tensor_dims` | `{x: ((0, 2), (1, 2))}` | `{"x": ((0, 2), (1, 2))}` | `{0: ((0, 2), (1, 2))}` |
| R12: self-contained | No. Names are declared and attached elsewhere. | Yes. | Yes. | Yes. |
| Readability | Best for a model that already names its dimensions. | Reads like the expression. | Reads like the expression. | Depends on the op's argument order, which a decomposition can change. |
| Mechanism | Public (`fx.traceback.annotate`). | Public: the keys become marker-op arguments. | Private: `torch._dynamo.comptime` reads the caller's locals. | Public. |
| Renaming a variable | No effect. | No effect. | Breaks the directive. | No effect. |

Tensor keys are proposed. They read like the expression, and they need no
lookup at all: the key is the operand. Strings were probed too.
`comptime`'s `ctx.get_local(name, stacklevel=1)` does find a helper's
locals, and `InstructionTranslator.current_tx()`, which `spyre_hint` uses,
returns the outermost compiled frame instead, so it would look names up in
the wrong function. Strings would still depend on a private API and on
variable names staying stable, for no gain in readability. Operand
positions are rejected: `x @ w` and `torch.matmul(x, w)` reach different
ops after decomposition, so a position means little to the user.

### What the residency directive fixes

| | Residency only (A) | Residency and address (B) |
|---|---|---|
| R7: no implementation detail | Yes. | No. It ties user code to LX capacity, stick alignment and reserved regions. |
| Exact control | No. Fragmentation can still affect what fits. | Yes, including deliberate reuse of a region. |
| Solver work | One fixed literal or a placement order. | Reserved regions, plus overlap checks between pinned buffers whose lifetimes overlap. |
| Cost to keep valid | Low. | High: every LX layout change breaks pinned kernels. |

@cyang49 notes that address-level control is very hard to program against,
brings no performance benefit over a residency guarantee, and makes code
brittle. A is proposed. B stays possible later as a separate key, but
nothing in this RFC reserves room for it.

### How strong the directive is

| | Mandatory | Preferred (best effort) |
|---|---|---|
| R5: met or fails | Yes. | No. |
| Defined in today's cost model | Yes: a constraint. | No. Eviction has no cost of its own (the model prices the downstream compute and traffic), so "how much do we prefer buffer A over buffer B" has no answer. |
| Implementation | Check validity only. | Needs a new objective term and weights. |

Mandatory only, as @spectre-ns proposed and @cyang49 agreed. The same
reasoning applies to `work_div`: a preferred division would need a weight
against the cost model's own choice, which nobody has asked for.

## Metrics

* Every directive in the test suite is met or raises. A CI test asserts
  each case, for each key and for the combination.
* For kernels that adopt directives, the predicted and measured runtime
  stays the same across solver, cost-model and time-budget changes that do
  not touch the kernel.
* Solve time with directives is no worse than without them on the model zoo.
  Fixed literals and pinned divisions remove choices, so CP-SAT is not
  expected to slow down.

## Drawbacks

* **A new user-facing API.** Once kernels use it, its semantics are hard to
  change. Keeping `lx` to residency limits that exposure.
* **Two ways to set a division.** Users must learn when to use a scope and
  when to use a value, and how the two merge. The merge rule (agree or
  raise) keeps the cost of a mistake to a compile error.
* **Division legality is not portable.** A `work_div` must divide the
  stick-adjusted dimension size and respect `SENCORES`, so a kernel that
  compiles at one shape or core count can raise at another. Scope hints have
  the same property today; the value form makes it easier to reach.
* **Hard failures move with the compiler.** A directive met today can start
  raising when a residency rule or a split domain is added or tightened,
  because a rule change now breaks a compile instead of only changing a
  plan. The global opt-outs soften this, but the error is the point: the
  user asked for a guarantee.
* **Less room for the solver.** Each forced buffer and pinned op removes
  choices from the solve. A bad directive can make the rest of the graph
  slower, and the relative-cost diagnostic is the only signal.
* **Users cannot reach hidden tensors.** A kernel whose working set is mostly
  restickify or relayout copies gains little.

## Alternatives

* **Region-wide "no spill" constraint** (@cyang49). Rejected. Many rules
  govern LX residency, so the constraint would be very hard to satisfy for
  scopes of more than a few buffers, and the diagnostics would be poor when
  several ops conflict. @cyang49 agreed that logs give the same insight.
* **Exact LX addresses** (option B). See the
  [trade study](#what-the-residency-directive-fixes).
* **Preferred residency.** See the [trade study](#how-strong-the-directive-is).
* **Work division through scope hints only.** Leave `work_div` on
  `spyre_hint` and give `spyre_directive` residency alone. It keeps one way
  to set a division, but residency and the division it depends on are then
  written in two places, and a one-op division still needs dimension names
  and a scope that may catch other ops.
* **Hand-placed restickifications** (option 1 in the thread). A
  `torch_spyre.primitives` module would let a kernel insert its own
  restickify ops. It ties kernels to one configuration and version, like
  hand-written vector intrinsics, and might need an LX context switch around
  the kernel so outside state cannot invalidate its directives. Deferred
  until a kernel needs it.
* **Do nothing.** Kernel authors keep tuning by changing compiler internals,
  and regressions from solver changes stay invisible until they are
  measured.

## Prior Art

* `spyre_hint(work_div={...})` in this repository: an authoritative,
  scope-based directive with a global opt-out. This RFC reuses its
  validation and its opt-out and differs from it in two ways: it names a
  value, not a scope, and it never skips a directive with a log line.
* Op-based layout directives prototyped earlier in this project by
  @cyang49.
* GPU programming exposes shared memory as an explicit allocation. That is
  closer to option B, and it is the model @cyang49 argues against for Spyre.

## How we teach this

* A "Controlling LX residency and work division" section in
  `docs/source/user_guide/`, with the examples above, the view rule, the
  input-copy rule and the hidden-tensor limitation.
* In `docs/source/compiler/work_division_planning.md`, "User Work-Division
  Hints" gains the value form, the merge rule with scopes and the stricter
  `SENCORES` behaviour.
* A note in `docs/source/compiler/scratchpad_planning.md` saying where
  directives enter the allocator, and that the residency rules table lists
  what can make an `lx=True` directive fail.
* The error message names the rule or the failed split check, so users can
  look it up.

## Unresolved questions

To resolve before this RFC is accepted:

1. **Single value versus regions.** Does a per-value directive cover every
   use case for residency, or do some kernels need a region form
   (@dgrove-oss)? Work division already has one, the scope hint.
2. **Name and location.** `spyre_directive` with a keyword per key, or a
   shorter name? `work_div` matches `spyre_hint`; @spectre-ns's first
   example in the thread used `core_division`.
3. **`LX_PLANNING=0`.** Should `lx=True` raise when LX planning is off, or
   be dropped like the opt-out does?

To resolve during implementation:

1. Where exactly the post-pass check runs, given that `CustomPostFusionPasses`
   still changes allocations (`hbm_pool_planning`).
2. Whether the input copy reuses `GraphEditor` (`graph_editor.py`), as
   input-boundary cloning does.
3. How a directive survives `materialize_hinted_views` and other passes that
   clone or rewrite nodes. `copy_fx_custom_meta` covers the passes that
   already use it; the post-pass check catches the rest.
4. How a directive written inside a `for_each_tile` body reaches its buffer.
   The body is traced as a `scan` combine subgraph, so the fold pass must
   walk subgraphs, and the directive must ride the same recovery path that
   `spyre_hint` annotations take when the scan becomes a `while_loop`
   (`_recover_hop_subgraph_hints` in `propagate_hints.py`). A directive on a
   loop carry must name the carry, not the per-trip value it is built from.
5. How a `work_div` maps through `mm_to_bmm_pass`, which rewrites a 2D
   matmul into a 3D bmm after the fold. The split must follow the original
   operand dimensions, not the bmm's.
6. How an operand tag survives AOT tracing and decomposition. A tag on the
   operand's producing node rides `custom` meta like the directive itself;
   a graph input loses its `custom` meta when AOT creates new placeholders,
   so its entry is recorded by input position. The work-division pass then
   finds the read of the tagged buffer among the producing op's reads.

Out of scope:

* Exact addresses, region constraints and hand-placed restickifications, as
  above.
* Directives for buffers created after the solve.
