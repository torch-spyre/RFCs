# Gemma 4 Mixture-of-Experts on Spyre — Per-Expert Grouping (Approach B)

**Authors:**
* @ani300

## **Summary**

Gemma 4's MoE FFN (`google/gemma-4-26B-A4B-it`: 128 experts, top-8 trained,
`moe_intermediate_size=704`, `hidden_size=2816`, 30 all-MoE layers) is being
ported to run on IBM Spyre. An already-in-progress path ("Approach A") runs the
whole MoE FFN on-device by looping directly over the `[T,K]` topk results, doing
one expert-weight fetch **per routed row**. This RFC specifies **Approach B**:
the follow-on that turns those per-row fetches into **once-per-expert** fetches
by grouping routed rows into contiguous per-expert segments and driving a
fixed-trip loop over a single static device program.

Approach B cannot be built today because the per-segment device grouped program
cannot be tiled until a new backend primitive exists. This document is a
precise, actionable spec of the **five backend deliverables** (torch-spyre /
deeptools) that grouping needs, so the work can be picked up as soon as the
primitives land. Nothing here is implemented yet — it is a specification and a
set of collaboration asks.

## **Motivation**

The shipped Gemma 4 MoE adapter keeps expert weights host-resident, selects the
per-row weight on CPU, and moves small `[N,·]` slices to device. That was forced
by two findings: (a) fancy-indexing the expert stack did not lower on-device at
the time, and (b) keeping all experts resident **and** materializing the per-row
`[N,H,2M]` weight tensor simultaneously exhausted the card (~46 GB resident +
~46 GB materialized on a ~103 GB card).

The redesign's insight: the OOM was never "experts in HBM" per se — it was
*indexing/materializing the whole stack at once*. If the on-device weight select
is **tiled** so each iteration touches only a scratchpad window, the giant
per-row tensor never materializes and the select runs on Spyre.

Approach A realizes that with a per-row loop. Approach B is the efficiency
follow-on. With top-8 routing over 128 experts, adjacent routed rows frequently
share an expert; fetching that expert's weight slab once per row wastes HBM
bandwidth. Grouping reorders rows so all rows for one expert are contiguous,
then fetches each expert's `[H,2M]` / `[M,H]` slabs **once per tile** instead of
once per row — the dominant cost in a bandwidth-bound MoE FFN. This is what makes
a 26B-parameter, 128-expert model tractable to run and later scale to the
trained top-8 on Spyre.

## **Proposed Implementation**

Approach B has two stages that share a **byte-identical device static program**;
only the *producer* of the grouping tables differs. Stage 1 groups on the x86
host CPU; Stage 2 moves grouping onto the in-Spyre RISC-V CPU.

### Terminology

The router has already run. For each of the `T` tokens it picked `K` experts, so
there are `N = T·K` **(token, expert) pairs**. Three flat arrays of length `N`,
in the same order, describe them:

- `token_of_row[i]` — which token the `i`-th pair belongs to (`0..T-1`, each
  appearing `K` times).
- `expert_of_row[i]` — which expert the `i`-th pair routes to (`0..E-1`,
  `E = 128`; data-dependent, unsorted).
- `row_weight[i]` — the scalar router weight for the pair (post
  softmax → top-`K` → renormalize → `per_expert_scale`), i.e.
  `row_weight = w.reshape(N)` from the `[T,K]` router-weight tensor `w`. Every
  expert output row is scaled by its `row_weight` before the scatter-combine.

  > **Note (blocker #5):** the `w.reshape(N)` flatten is itself the layout-abort
  > trigger described in deliverable #5 — a `[T=64,K=4]→[N=256]` reshape across
  > the partial (`K=4 < 64`) stick. It blocks the router weight from reaching the
  > device in any form (A's device region, a device-side flatten, or the RISC-V
  > producer), so grouping does **not** route around it.

### B-Stage 1 — grouping on the host CPU

**Grouping** is three plain integer-array operations (no model knowledge):

1. **Sort the pairs by expert.** `sort_perm = argsort(expert_of_row)` `[N]`.
   Applying it yields the reordered views:
   - `expert_sorted = expert_of_row[sort_perm]` — now non-decreasing, e.g.
     `[0,0,0,2,2,5,5,5,5,…]`; experts with no rows simply don't appear.
   - `gathered_sorted = activations[sort_perm]` — token activations reordered to
     match.
   - `token_of_row_sorted = token_of_row[sort_perm]` — kept for the final
     scatter-back.
   - `row_weight_sorted = row_weight[sort_perm]` — router weights reordered the
     same way; applied at the end.

2. **Count rows per expert.** `counts = bincount(expert_sorted, minlength=E)`
   `[E]`. With top-8 over 128 experts, most counts are small, some zero.

3. **Counts → segment boundaries.** `group_off [E+1]` with `group_off[0]=0`,
   `group_off[1:] = cumsum(counts)`. Expert `e` owns the half-open row range
   `group_off[e]:group_off[e+1]` — its **segment**.

Worked micro-example (`T=3`, `K=2`, `N=6`, `E=6`):

```
expert_of_row      = [2, 0, 2, 5, 0, 2]      # token0→{2,0}, token1→{2,5}, token2→{0,2}
sort_perm          = [1, 4, 0, 2, 5, 3]      # positions of experts 0,0,2,2,2,5
expert_sorted      = [0, 0, 2, 2, 2, 5]      # non-decreasing after the sort
counts (E=6)       = [2, 0, 3, 0, 0, 1]      # expert0:2 rows, expert2:3, expert5:1
group_off (E+1=7)  = [0, 2, 2, 5, 5, 5, 6]   # expert2 owns 2:5, expert5 owns 5:6
```

(The adapter already ships these as `_moe_permute` (step 1) and `_group_offsets`
(steps 2–3).)

#### From variable-size segments to a fixed-trip static loop

Segment sizes are **data-dependent** — a Spyre device loop may have a runtime
trip count, but the program it executes must be **static** (same work every
iteration). So round every expert's segment length up to a whole number of
`TILE`-row tiles, inserting inert padding rows. Two invariants hold afterward:

1. **`TILE`-alignment** — each expert's padded segment starts and ends on a
   `TILE` boundary, so a tile never spans two experts.
2. **Single-expert tiles** — hence a single `tile_expert[tile]` scalar names the
   weight slab for a whole tile.

```python
# INPUT (from the grouping section):
#   sort_perm     [N]    argsort(expert_of_row)           — the by-expert order
#   expert_sorted [N]    expert_of_row[sort_perm]         — non-decreasing
#   counts        [E]    bincount(expert_sorted, E)       — rows per expert
#   gathered_sorted [N,H], token_of_row_sorted [N]        — reordered activations
#   row_weight_sorted [N] row_weight[sort_perm]           — reordered router weights
TILE = 32                                     # rows per tile (module constant)

# 1. Tiles each expert needs = ceil(counts / TILE). Experts with 0 rows need 0.
tiles_per_expert = (counts + TILE - 1) // TILE          # [E]  int
padded_counts    = tiles_per_expert * TILE              # [E]  each a TILE multiple

# 2. Padded segment boundaries (same shape/role as group_off, TILE-aligned).
pad_off = torch.zeros(E + 1, dtype=torch.long)          # [E+1]
pad_off[1:] = torch.cumsum(padded_counts, 0)            # expert e -> pad_off[e]:pad_off[e+1]
N_pad   = int(pad_off[-1].item())                       # total padded rows
N_TILES = N_pad // TILE                                 # runtime loop count

# 3. tile_expert[t] = the one expert that owns padded tile t.
experts     = torch.arange(E)
tile_expert = experts.repeat_interleave(tiles_per_expert)   # [N_TILES]

# 4. Scatter the real (unpadded) rows into their padded slots.
seg_start_of_row = pad_off[:-1].repeat_interleave(counts)   # [N] padded seg start per row
intra_base       = group_off[:-1].repeat_interleave(counts)  # [N] unpadded seg start per row
intra            = torch.arange(N) - intra_base             # [N] j within segment
dst_pos          = seg_start_of_row + intra                 # [N] -> padded index

gathered_pad = gathered_sorted.new_zeros(N_pad, H)          # padding rows = 0
gathered_pad[dst_pos] = gathered_sorted                     # real rows into their slots
# padding rows carry the SINK token id T (one past the last real token) so the
# on-device per-tile index_add routes them to a throwaway sink row out[T] that
# is sliced off after the loop — no data-dependent mask in the static program.
token_of_row_pad = token_of_row_sorted.new_full((N_pad,), T)
token_of_row_pad[dst_pos] = token_of_row_sorted
# the router weights ride along into the same padded slots; padding rows get a
# weight of 0, so even the sink-row accumulation for them is zero.
row_weight_pad = row_weight_sorted.new_zeros(N_pad)         # [N_pad] padding = 0
row_weight_pad[dst_pos] = row_weight_sorted
```

Worked micro-example continuing `T=3, K=2, E=6` (`counts = [2,0,3,0,0,1]`),
`TILE = 2`:

```
counts            = [2, 0, 3, 0, 0, 1]
tiles_per_expert  = [1, 0, 2, 0, 0, 1]      # ceil([2,0,3,0,0,1] / 2)
padded_counts     = [2, 0, 4, 0, 0, 2]      # * TILE
pad_off (E+1=7)   = [0, 2, 2, 6, 6, 6, 8]   # cumsum, leading 0; N_pad = 8
N_TILES           = 4                        # 8 / 2
tile_expert       = [0, 2, 2, 5]            # expert 2 spans two tiles (3 rows -> 2 tiles, 1 pad)
# padded row block (8 rows): [e0 e0 | e2 e2 | e2 PAD | e5 PAD]
```

#### The host→device side-channel

Two small integer tables travel from host to device alongside the padded,
reordered activations `gathered_pad [N_pad,H]` and per-row router weights
`row_weight_pad [N_pad]`:

- `pad_off [E+1]` — the `TILE`-aligned segment boundaries.
- `tile_expert [N_TILES]` — for each padded tile, the single expert id it owns.

`row_weight_pad` is not an integer table — it is a per-row column that rides with
`gathered_pad`, consumed by the device program's final row-scale multiply.

#### The device static program

Single program, hint-looped over the runtime `N_TILES`:

```python
out = gathered_pad.new_zeros(T + 1, H)       # [T+1,H] accumulator; row T is the padding sink
with spyre_hint(tiles={"row": TILE}):        # exactly N_TILES iterations
    e    = tile_expert[tile]                 # ONE expert id for the whole tile
    W_gu = gate_up_dev[e]                     # ONE slab [H,2M] — per_tile_fixed: once/tile
    W_dn = down_dev[e]                         # ONE slab [M,H]  — per_tile_fixed
    seg  = gathered_pad[rows]                 # [TILE,H] this tile's rows (one expert + any pad)
    w_r  = row_weight_pad[rows]               # [TILE] this tile's router weights
    dst  = token_of_row_pad[rows]             # [TILE] destination token per row (T for padding)
    gu   = torch.bmm(seg.unsqueeze(1), W_gu)  # [TILE,1,2M]
    g, u = gu.chunk(2, dim=-1)                # [TILE,1,M]
    act  = F.gelu(g, approximate="tanh") * u  # gelu-tanh SwiGLU
    seg_out = torch.bmm(act, W_dn).squeeze(1) # [TILE,H]
    seg_out = seg_out * w_r.reshape(TILE, 1)  # per-row router-weight scale (blocker #5)
    out.index_add_(0, dst, seg_out)           # ON-DEVICE scatter-combine, per tile
out = out[:T]                                 # drop the padding sink row -> [T,H]
```

Two consequences of doing the combine per tile, on-device:

- **Padding rows are routed to a sink row, not masked.** They carry destination
  token id `T`, so their (zero) contribution lands in the throwaway `out[T]`,
  sliced off after the loop. This avoids a data-dependent mask, which would
  reintroduce a variable-length gather the static tile program cannot express.
- **`out` is the one loop-carried accumulator** (an `index_add_` reduction into a
  fixed `[T+1,H]` buffer), so it is *not* `per_tile_fixed`; the expert slabs
  `W_gu`/`W_dn` still are (one fetch per tile).

**Win vs. Approach A:** one weight slab per tile instead of per row, marked
loop-invariant via `per_tile_fixed`. The cost is padding rows — with top-8 over
128 experts and short prompts, many segments are far smaller than `TILE`, so
padding overhead (and the right `TILE`) is the key tuning trade-off.

### B-Stage 2 — grouping on the RISC-V CPU inside Spyre

Move Stage-1 grouping (argsort + bincount + cumsum + capacity-pad + offset-table
build) onto the **in-Spyre RISC-V CPU**, so topk results never round-trip to the
x86 host. The device static program is **byte-identical to Stage 1** — only the
producer of `pad_off` / `tile_expert` / `sort_perm` moves. This defines a
**RISC-V ↔ device-program ABI**: the RISC-V code writes the offset table plus
permutation into a known HBM/scratchpad region the static program reads, with a
defined sync/fence contract.

### The five backend deliverables (the collaboration asks)

1. **Per-segment operand-select tiling primitive.** A way to tile the loop by
   per-expert segment so one static program iterates `N_TILES` times, each
   binding one `expert_id → one weight slab`. Today `tiles={...}` binds only to an
   op's **output** ranges (`wsr/coarse_tile.py:758-768`, `wsr/coarse_tile_hints.py`);
   there is no hint to make a per-tile scalar `tile_expert[tile]` select the
   *weight operand* from `group_off`. This is the core new primitive — either a
   backend **grouped-GEMM op** or a `group_off`-driven **per-tile operand-select
   hint**.
2. **Windowed HBM→scratchpad indirect-gather + per-tile scatter-reduce
   correctness.** Harden the indirect-gather execution path: `expert_w[expert_ids]`
   reaches SDSC but is `xfail` on divergence by default
   (`tests/inductor/indirect_access_common.py:413-434`) and the literal MoE case
   is skipped for output-span overflow (`test_moe` skipped,
   `tests/inductor/test_indirect_access_gather.py:447-465`). **Also Approach A's
   dependency.** This ask **also covers the on-device per-tile `index_add_`
   scatter-combine** both approaches wire through the tiled loop: a scatter-reduce
   into a fixed `[T+1,H]` accumulator, padding rows routed to sink row `out[T]`.
   The spec **assumes `index_add`/`scatter-reduce` lowers on device** at these
   shapes; confirm it does (or land it), since neither the per-tile combine nor
   the sink-row scheme has an off-device fallback.
3. **`per_tile_fixed` for the weight operand.** Confirm/extend that the
   loop-invariant-load flag fires for the expert weight slab within a segment tile
   (mechanism exists: `insert_restickify.py:281-345`).
4. **RISC-V grouping ABI (Stage 2).** Memory region, layout, and
   synchronization/fence contract for the RISC-V-produced grouping outputs —
   `pad_off` / `tile_expert` / `sort_perm` plus the per-row padded columns
   (`gathered_pad`, `row_weight_pad`, and `token_of_row_pad` with its sink id `T`).
   Ties to Spyre correction-path/host-compute ordering (HostCompute inline,
   H2D→Compute auto-barriered); a RISC-V→device handoff needs an explicit fence
   design.
5. **Restickify on a reshape across a sub-stick dim.** *(The FIRST blocker
   Approach A's on-card gate actually hit, ahead of #2.)* A reshape that flattens
   a small trailing dim into the outer dim aborts the layout pass when that
   trailing dim is a **partial stick** (< one 64-element stick). Router weights are
   `[T=64,K=4]`; `K=4` occupies a stick padded to 64, so `w.reshape(N=256,1)` —
   the `row_weight = w.reshape(N)` flatten feeding the per-row weight multiply (and
   identically Approach A's `out * tw.reshape(N,1)`) — makes the layout pass read
   the `[T,K]` buffer with a flat `[256]` index. It blocks **both approaches**: the
   flatten *is* this reshape. The read is the cross-stick expression
   `Mod(d0,4) + floor(d0/4)` (mod **4**, not mod 64), which
   `is_stick_expr_offset_free` rejects, so `_multi_arg_pointwise_layouts` raises:

   ```
   InductorError: Unsupported: Spyre backend does not support:
     Multi-arg pointwise (buf1): no supported output layout found
     with size=[256, 2816] and coordinates=[d0, d1]
   propagate_layouts.py:1099  (load-bearing gate at :985)
   ```

   **Not fixable adapter-side.** Every materialization strategy (`.contiguous()`,
   `* 1.0`, `+ 0.0`, or multiplying in `[T,K,H]` space) merely relocates the abort
   — the copy's *input read* of the `[T,K]` partial-stick buffer with a flat index
   is itself the unrepresentable expression. Only a device-native `[N,·]` operand
   compiles. **The fix is a backend restickify** (HBM round-trip) at the
   `[T,K]→[N]` reshape, re-laying the partial-stick buffer `N`-outermost before any
   flat consumer reads it. The comment at `propagate_layouts.py:987-989` already
   anticipates this but the code only inserts it for the index-symbol case;
   `AllSameNode` cannot rescue it today (asserts `out_layouts` non-empty at
   `optimize_restickify.py:183`; the abort at `:1098` fires before it is
   constructed). **Distinct bug class from #2** — no gather, no bmm; they share
   only the raise site.

   Reproducers (no model load, no real weights, any Spyre host):
   `repros/gemma4_moe/gatherbcast_layout_repro.py` (its `region` stage reproduces
   the abort verbatim) and `repros/gemma4_moe/rowweight_reshape_probe.py` (a 7-way
   isolation pinning the trigger to the `[T,K]↔N` reshape).

## **Metrics **

- **HBM bandwidth per MoE layer:** expert-weight bytes fetched per forward pass —
  the primary win. Target: from `N` slab fetches (Approach A) down to `N_TILES`
  slab fetches (Approach B), i.e. up to a `TILE`× reduction, minus padding
  overhead.
- **Padding overhead ratio:** `N_pad / N` — rows actually computed vs. useful
  rows, as a function of `TILE` and prompt shape. The core tuning metric.
- **Correctness gate:** single-layer **fp16-vs-fp32 rel-err** (`mean_rel < 0.02`,
  `max_rel < 0.5`).
- **E2E gate:** `tests/spyre/test_e2e_token_compare_spyre.py -k 26B-A4B`
  top-1 token agreement (non-blocking xfail during bring-up).

## **Drawbacks**

- **Not a breaking change** — it is a new device formulation of an FFN that is
  otherwise unimplemented on Spyre; no existing user-facing API changes.
- **Blocked on backend work.** Approach B cannot be built until deliverable #1
  (per-segment operand-select) lands; #5 blocks compilation of *both* approaches
  today. Building the host-grouping side prematurely would only reproduce the
  existing non-working path.
- **Padding cost.** With top-8 over 128 experts and short prompts, many segments
  are much smaller than `TILE`, so a meaningful fraction of computed rows are
  padding. Wrong `TILE` erodes the bandwidth win.
- **Implementation complexity.** Introduces a per-tile operand-select primitive,
  an on-device scatter-reduce assumption, and (Stage 2) a RISC-V↔device ABI with
  an explicit fence contract — each a non-trivial backend surface.

## **Alternatives**

- **Approach A (per-row loop) — in progress.** Simpler; no grouping, no argsort,
  no padding. Correct but does one weight fetch per row. Approach B is strictly
  the bandwidth optimization on top; if B's primitives never land, A remains the
  fallback.
- **Original host-resident path (the shipped `4B` path).** Keeps experts on the
  host and materializes per-row weights — the path that OOM'd the card and
  motivated this redesign. Rebuilding host grouping without primitive #1 would
  reproduce it.
- **Impact of not doing this:** MoE FFNs stay bandwidth-bound at per-row weight
  fetches (Approach A), limiting throughput on the 26B-A4B model and any future
  MoE checkpoint on Spyre.

## **Prior Art**

- **Grouped-GEMM / expert-parallel MoE** is standard on GPUs (Megablocks,
  vLLM/SGLang fused MoE, Triton grouped-GEMM kernels): sort tokens by expert, run
  a segment-tiled batched GEMM, scatter back. This RFC adapts that well-worn
  pattern to Spyre's static-program + stickified-layout constraints — the novelty
  is not the algorithm but the backend primitives Spyre needs to express it.
- The capacity/padding-to-tile scheme mirrors the expert-capacity padding used in
  Switch-Transformer-style routing, here applied per-tile rather than per-expert
  for a fixed-trip static loop.

## **How we teach this**

- Terminology to standardize: **(token, expert) pair / row**, **segment**
  (one expert's contiguous rows), **tile** (a `TILE`-row single-expert unit),
  **`tile_expert`** (the per-tile expert id), **sink row** (the `out[T]` padding
  drain).
- Frame it as two stages sharing one device program: *"grouping is just sorting
  integers and recording where the runs begin"* (host/RISC-V producer), and *"the
  device program is a fixed-trip loop that fetches one expert per tile."*
- The five deliverables are the teaching spine for the backend team; the two
  reproducers are the executable teaching artifacts for blocker #5.

## **Unresolved questions**

- **To resolve via the RFC process:** the shape of primitive #1 — a backend
  **grouped-GEMM op** vs. a `group_off`-driven **per-tile operand-select hint**.
- **To resolve during implementation:** whether on-device `index_add`/
  scatter-reduce lowers correctly at these shapes (deliverable #2); whether
  `per_tile_fixed` fires for the weight operand as-is (deliverable #3); the exact
  RISC-V↔device ABI and fence contract (deliverable #4); the right `TILE` and the
  padding-overhead trade-off.
- **Out of scope:** the dense `hf_gemma4.py` path; restoring the trained top-8
  (`K` is pinned to 4 during bring-up because on-device `topk(k>4)` SIGABRT'd —
  once routing runs on-device with a working `topk`, or grouping is RISC-V-side
  where `k` is unconstrained, lift `K` 4→8 and re-validate); the pre-existing
  ARCHITECTURE.md / README adapter-count discrepancy.

## Resolution

*Pending — this RFC is open for review.*

### Level of Support
5: Unclear Resolution.

#### Additional Context
Specification and collaboration asks only; nothing is implemented. Acceptance
depends on the torch-spyre / deeptools team scoping the five backend
deliverables.

### Next Steps
Land the backend primitives in dependency order — deliverable #5 (unblocks
compilation of both approaches), then #2 (gather + scatter correctness), then #1
(per-segment operand-select), #3, and #4 (RISC-V ABI). Pick up host grouping
(B-Stage 1) once #1 is available.

#### Tracking issue
<github issue URL>

#### Exceptions
Approach B is deferred until primitive #1 lands; Approach A proceeds in the
meantime as the fallback path.
