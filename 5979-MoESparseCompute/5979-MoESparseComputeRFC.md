# Sparse MoE Compute on Spyre — Gemma4 26B-A4B

> **Status:** draft. Tracking issue: [#5979](https://github.com/torch-spyre/torch-spyre/issues/5979).

## **Summary**

This RFC proposes sparse kernels for the MoE (sparse FFN) block on Spyre, using Gemma4 26B-A4B as the target model.

* **Chunked prefill:** each expert computes only on the tokens routed to it, instead of every expert computing on every token. The chunk size is chosen so that each expert gets about 512 tokens on average (`Tq = 512·E/K` = 8192). Each expert's tokens are processed in 512-token tiles in a dynamic loop. A tile with fewer than 512 real tokens is padded with dummy tokens.
* **Decode:** there is no per-token loop. The whole batch runs through each *selected* expert in a dynamic loop over the `E_sel` distinct experts chosen anywhere in the batch, so each selected expert's weights are fetched once.
* **Shared experts** run on all tokens in both phases.
* The router needs two **new ops** (`non_zero_pos_to_idx`, `count_non_zero_div_ceil`) and changes to two existing ones.
* **Router list build (revised):** the per-expert token lists are built from the token view directly. Only the `Tq·K` real (token, expert) entries are scattered (65,536 sticks instead of 1,048,576), into an expert-major list, with no sink writes and no per-expert weight scatter. This removes the router's main bottleneck, the `Tq·E` scatters (A16a/A16b of the earlier revision), and makes the router about 4–6× faster per prefill chunk of `Tq` tokens (see [Router cost](#router-cost)). It adds no new op.

Across devices, the block uses tensor parallelism: each device holds a `1/D` slice of every expert's intermediate dim, and an AllReduce at the end combines the partial outputs.

## **Motivation**

The current prefill path is dense: every expert processes every token in the chunk, and routing weights are applied as a `[Tq,E]` mask at combine time. That does `E/K = 16×` the necessary FLOPs. The extra work is free while `Tq ≲ 512` keeps the block memory-bound, but it becomes the bottleneck as the chunk grows.

The current decode path loops over `B·K` (token, expert) pairs and runs one GEMV each. An expert picked by several tokens in the batch has its weights fetched several times. Decode is bound by weight bandwidth, so those repeated fetches are wasted.

This RFC removes both inefficiencies:

* **Prefill:** with the chunk sized so `Tqe = K/E·Tq ≈ 512`, each per-expert GEMM has arithmetic intensity ≈ 512 flop/byte at df16 (`6·512·dmodel·dff_e` FLOP over `6·dmodel·dff_e` bytes), close to Spyre's balance point (to be confirmed). Only `K/E` of the dense work is done. Padding to `G = 512` adds fewer than `G` dummy tokens per expert, so the computed token count stays below `Tq·K + G·E`: at most **2×** the ideal at `Tqe = 512`, versus 16× for dense.
* **Decode:** each selected expert's weights are fetched **once** (`E_sel ≤ min(E, B·K)` fetches). A `B`-row GEMM costs about the same as a GEMV when memory-bound. The extra FLOPs (`B·E_sel` instead of `B·K` token-expert rows) are negligible at `B ≤ 32`.

## **Proposed Implementation**

### Configuration and notation

**Model:**

| Symbol | Meaning |
|---|---|
| `dmodel` | model / hidden dim |
| `E` | routed experts per layer — **128** |
| `K` | experts selected per token — **8** |
| `Es` | shared experts per layer |
| `dff_e` | routed-expert intermediate dim |
| `dff_s` | shared-expert intermediate dim |

**Workload:**

| Symbol | Meaning |
|---|---|
| `Tq` | tokens in one prefill chunk — chosen so `Tqe ≈ 512` ⇒ `Tq = 512·E/K` = **8192** |
| `Tqe` | `= K/E · Tq`, average tokens routed to one expert in a chunk — target **512** |
| `G` | token granularity of per-expert compute — **512** |
| `cnt[e]` | tokens actually routed to expert `e` in this chunk (`Σ_e cnt[e] = Tq·K`) |
| `nc[e]` | `= max(1, ⌈cnt[e]/G⌉)`, 512-token tiles for expert `e` — **dynamic** loop bound, always ≥ 1 |
| `B` | decode batch — target **1–32** |
| `E_sel` | distinct experts selected across the decode batch, `≤ min(E, B·K)` — **dynamic** loop bound (from B13b) |

**Layout:**

| Symbol | Meaning |
|---|---|
| `S` | stick length — elements fetched/written as one unit. Dtype-dependent: **64** at df16, **32** at fp32 ⇒ a stick is a fixed 128 B |

Every tensor has one **stick dim**: the axis along which contiguous groups of `S` elements are accessed together. It is stored as `⌈dim/S⌉` sticks; a stick dim shorter than `S` occupies a partial stick and wastes the remainder. The **Stick dim** column below names that axis for each op's output.

**Loop markers:** loops marked **(dynamic)** have a trip count produced on device by the router, not known at compile time.

**Index tensors:** gathers, scatters and indirect accesses take their index tensors directly (int32, or fp32 holding integers). The backend inserts the index-to-address conversion (`idx2Addr`) right in front of each op that needs it, so the frontend never puts `idx2Addr` in the bundle. The tables still list these conversions for completeness, marked *(inserted by the backend)*, each right in front of the op it feeds; they carry the number of the op they feed with an `i` suffix.

### Path A — Chunked prefill (selected tokens per expert)

Each expert computes only on the tokens routed to it. Its routed tokens are processed in tiles of `G = 512`. For each tile:

1. Gather the tile's input rows.
2. Run the expert FFN.
3. Gather the same rows of a partial-output buffer `moe_out[Tq+1,dmodel]`.
4. Add, and scatter back.

A tile with fewer than 512 real tokens is padded with **dummy tokens**. These read a zero pad row appended to the input and write to a sink row in `moe_out`, so their results are discarded. An expert with `cnt[e] = 0` still runs one all-dummy tile, because A12 gives 1 tile for a count of 0. Every loop count is therefore ≥ 1, at the cost of at most one extra weight fetch per unused expert. After the routed experts, the shared expert(s) run on all `Tq` tokens and add into the same buffer.

The router turns the dense `[Tq,E]` routing weights into per-expert packed token lists:

* `non_zero_pos_to_idx` gives each (token, expert) entry its slot in the expert's list. Adding a constant per-expert base turns the slot into the entry's address in one expert-major list.
* `topk` over E keeps, for each token, the addresses of its K selected entries (a `[Tq,K]` list, the token view).
* A scatter of only those `Tq·K` entries (A17b) packs the token ids into the per-expert lists. Unwritten slots keep the pad token from a bulk init.
* In the expert loop, each tile's token ids address the gathers directly, and each token's routing weight is gathered by token id from a weight tensor with one stick per (token, expert).

A step-by-step visual walkthrough of this router on a toy routing is in [the router walkthrough](5979-MoESparseComputeRouterWalkthrough.md).

A scatter or gather moves whole sticks, so an element placed on its own must occupy a full stick (a dummy dim in the stick). The earlier version of this router did that for all `Tq·E` entries (128 MB per scatter, 15 of every 16 writes into the sink row); this version does it only for the `Tq·K` real entries, and the replicated source is a constant.

Input: `x : [Tq, dmodel]`, stick dim `dmodel`

Constants, precomputed: `tok_ids : [Tq, 1, 32]` int32, token id `t` replicated in its own stick (feeds the scatter A17b, broadcast over K); `seed : [1, E]` fp32 = `e·Tq` (A13)

**Router**

| # | Op | Inputs | Output shape | Precision | Stick dim | Notes |
|---|---|---|---|---|---|---|
| A1 | `router_gemm` | `x[Tq,dmodel]` (in) × `W_router[dmodel,E]` | `[Tq, E]` | df16 | `E` | |
| A2 | `softmax` (dim=E) | `[Tq,E]` (A1) | `[Tq, E]` | df16 | `E` | reduces **along** the stick dim |
| A3 | `restickify` | `[Tq,E]` (A2) | `[Tq, E]` | df16 | `Tq` | stick dim `E`→`Tq`; moves the selection/reduction axis `E` off the stick |
| A4 | `topk_idx` (K=8, dim=E) | `[Tq,E]` (A3) | `idx [Tq,K]` | df16 | `Tq` | index output only; selects **across** sticks. Exact in df16 (`E = 128` needs 7 bits) |
| A5 | `keep_by_index` | `[Tq,E]` (A3), `idx[Tq,K]` (A4) | `[Tq, E]` | df16 | `Tq` | keeps the top-K scores, others → 0 (`maskval = 0`) |
| A6 | `sum_nonstick` (dim=E) | `[Tq,E]` (A5) | `[Tq, 1]` | df16 | `Tq` | renormalize denominator |
| A7 | `realdiv` | `[Tq,E]` (A5) ÷ `[Tq,1]` (A6) | `[Tq, E]` | df16 | `Tq` | renormalized routing weights |
| A8 | `restickify` | `[Tq,E]` (A7) | `[Tq, E]` | df16 | `E` | stick dim `Tq`→`E`; brings `E` back into the stick |
| A9 | `df16tofp32` | `[Tq,E]` (A8) | `[Tq, E]` | fp32 | `E` (32) | 32 `E` per stick. Output is **staggered** along `E`: each 64-wide df16 stick yields fp32 sticks holding `0..3, 8..11, …` then `4..7, 12..15, …`. fp32 is needed because slots and list addresses (up to `E·Tq`) aren't exact in df16 |
| A10 | `unstagger_stick_ea` | `[Tq,E]` (A9) | `score [Tq, E]` | fp32 | `E` (32) | restores contiguous `E` order |
| A11 | `non_zero_pos_to_idx` (dim=Tq, padval=−∞) **[NEW]** | `score[Tq,E]` (A10) | `run_pos [Tq, E]` | fp32 | `E` (32) | scans **across** sticks along `Tq`, independently per `E` lane. Non-zero entry → number of non-zeros before it in this `E` column (its slot in the expert's list); zero entry → `padval`. `padval` must be −∞ so that A13 cannot turn it into a valid address |
| A12 | `count_non_zero_div_ceil` (dim=Tq, divisor=G) **[NEW]** | `score[Tq,E]` (A10) | `nc [1, E]` | fp32 | `E` (32) | counts non-zeros along `Tq` per `E` lane (`cnt[e]`), divides by `G` and rounds up: `nc[e] = max(1, ⌈cnt[e]/G⌉)`. **A count of 0 gives 1**, so `nc[e] ≥ 1` |
| A13 | `add` | `run_pos` (A11), `seed[1,E]` (constant, broadcast over Tq) | `key [Tq, E]` | fp32 | `E` (32) | `key = e·Tq + slot`: the entry's address in the expert-major list `[E·Tq]`. −∞ stays −∞. Exact in fp32 (max `E·Tq` = 1,048,576 < 2²⁴) |
| A14 | `restickify` | `key[Tq,E]` (A13) | `[Tq, E]` | fp32 | `Tq` | stick dim `E`→`Tq`, as A3, so that A15 selects across sticks |
| A15 | `topk` (values, K=8, dim=E) | `key[Tq,E]` (A14) | `addr_k [Tq, K]` | fp32 | `Tq` | each token has exactly K finite keys, so the top K are its selected entries; their order does not matter. This is the step from the token view to the expert view |
| A17a | `init` (fill = `Tq`) | — | `tok_dst [E·Tq, 32]` | int32 | dummy (32) | expert-major list, capacity `Tq` per expert: slot `s` of expert `e` is row `e·Tq + s`. Every slot starts as the pad token, so slots past `cnt[e]` need no masking later. Best done as a broadcast copy of a `[1,32]` constant (one device kernel), about 2× faster than a runtime fill |
| A17bi | `idx2Addr` *(inserted by the backend)* | `addr_k[Tq,K]` (A15) | `[Tq, K]` | int32 (input fp32) | `Tq` | not emitted by the frontend: the backend converts `addr_k` in front of A17b. `Tq·K` = 65,536 entries, fp32 input (see [New and modified ops](#new-and-modified-ops)) |
| A17b | `scatter` (token id) | `tok_ids[Tq,1,32]` (constant, broadcast over K), index `addr_k` (A15), dest A17a | `tok_dst [E·Tq, 32]` | int32 | dummy (32) | one stick store per real (token, expert) pair: **65,536 stores** (16× fewer than each earlier scatter), no sink writes. Replaces A14a, A15, A16a/b, A17a/b of the earlier version |
| A17c | `restickify` | `nc[1,E]` (A12) | `[1, E, 32]` | fp32 | dummy (32) | one stick per expert; source of the `nc[e]` loop bound (as before) |
| A17w | `restickify` | `w_pad[Tq+16,E]` (A8 with 16 zero rows appended; row `Tq` is the pad token's zero row) | `w_dummy [Tq+16, E, 64]` | df16 | dummy (64) | one replicated stick per (token, expert) weight, rows outermost. Suggestion: appending 16 rows rather than 1 keeps the work split evenly over all cores (with `Tq+1` rows the split falls back to few cores); rows past `Tq` are never read |
| A18 | `zero_init` | — | `moe_out [Tq+1, dmodel]` | df16 | `dmodel` | partial-output space. Row `Tq` is a **sink** for dummy-slot writes |
| A19 | `pad` | `x[Tq,dmodel]` (in) | `x_pad [Tq+1, dmodel]` | df16 | `dmodel` | appends 1 dummy token at row `Tq`, all zeros. Padded slots hold the pad token `Tq`, so the pad token must be zero: the bias-free FFN then outputs 0 and the sink row stays finite |

Each tile's token ids go straight to the gathers and the scatter of the loop (A22, A28, A30, A32); the backend inserts the conversion of the tile's `G` ids in front of each of them.

**Routed experts**

`nc[e] = max(1, ⌈cnt[e]/G⌉) ≥ 1` (A12 → A17c). Tile slots run up to `nc[e]·G − 1 ≤ Tq − 1`, since `nc[e]·G ≤ Tq` (`cnt[e] ≤ Tq`, `Tq` is a multiple of `G`, and `G ≤ Tq`), so tiles never read past `Tq+1` rows. Slots from `cnt[e]` up to the end of the last tile hold the pad id `Tq`. Those dummy tokens read `x_pad[Tq]` and write `moe_out[Tq]` (the sink).

| # | Op | Inputs | Output shape | Precision | Stick dim | Notes |
|---|---|---|---|---|---|---|
| — | **loop `e = 0 .. E-1`:** | | | | | |
| — | &nbsp;&nbsp;**loop `c = 0 .. nc[e]-1` (dynamic):** | | | | | tile = slots `[c·G, (c+1)·G)` of expert `e`, i.e. rows `[e·Tq + c·G, +G)` of `tok_dst` (contiguous, stick-aligned since `G` is a multiple of the stick) |
| A21t | `restickify` (lane 0) | `tok_dst[e·Tq + c·G : +G]` (A17b) | `tok [G]` | int32 | `G` | token ids of the tile; slots past `cnt[e]` already hold `Tq` (A17a). |
| A22i | `idx2Addr` *(inserted by the backend)* (base = `x_pad`) | `tok [G]` (A21t), one tile | `[G]` (one tile) | int32 | `G` | token id → row address in `x_pad`, right in front of A22; not emitted by the frontend |
| A22 | `gather` (input) | `x_pad[Tq+1,dmodel]` (A19), `tok` (A21t) | `exp_in [G, dmodel]` | df16 | `dmodel` | row gather — whole `dmodel` sticks per token |
| A23 | `gate_proj` | `exp_in[G,dmodel]` (A22) × `W_gate[e][dmodel,dff_e]` | `[G, dff_e]` | df16 | `dff_e` | regular `512 × dmodel × dff_e` GEMM |
| A24 | `up_proj` | `exp_in[G,dmodel]` (A22) × `W_up[e][dmodel,dff_e]` | `[G, dff_e]` | df16 | `dff_e` | weights for expert `e` reused across its `nc[e]` tiles |
| A25 | `silu` | `[G,dff_e]` (A23) | `[G, dff_e]` | df16 | `dff_e` | on A23 only |
| A26 | `mul` | `[G,dff_e]` (A25) × `[G,dff_e]` (A24) | `[G, dff_e]` | df16 | `dff_e` | `silu(gate) * up` |
| A27 | `down_proj` | `[G,dff_e]` (A26) × `W_down[e][dff_e,dmodel]` | `[G, dmodel]` | df16 | `dmodel` | |
| A28i | `idx2Addr` *(inserted by the backend)* (base = `w_dummy`) | `tok [G]` (A21t), one tile | `[G]` (one tile) | int32 | `G` | token id → address of `w_dummy[tok, e]`, right in front of A28; not emitted by the frontend |
| A28 | `gather` (routing weight) | `w_dummy[Tq+16,E,64]` (A17w), `tok` (A21t) | `w_tile [G, 64]` | df16 | dummy (64) | `G` single sticks `w_dummy[tok, e]`: the token's weight for expert `e`, already replicated across the stick. |
| A29 | `mul` | `[G,dmodel]` (A27) × `w_tile[G,64]` (A28) | `[G, dmodel]` | df16 | `dmodel` | one weight stick broadcast across each token's `dmodel` sticks |
| A30i | `idx2Addr` *(inserted by the backend)* (base = `moe_out`) | `tok [G]` (A21t), one tile | `[G]` (one tile) | int32 | `G` | token id → row address in `moe_out`, right in front of A30; not emitted by the frontend |
| A30 | `gather` (partial out) | `moe_out[Tq+1,dmodel]` (A18 / A32), `tok` (A21t) | `[G, dmodel]` | df16 | `dmodel` | same tokens as A22 |
| A31 | `add` | `[G,dmodel]` (A29) + `[G,dmodel]` (A30) | `[G, dmodel]` | df16 | `dmodel` | |
| A32i | `idx2Addr` *(inserted by the backend)* (base = `moe_out`) | `tok [G]` (A21t), one tile | `[G]` (one tile) | int32 | `G` | token id → row address in `moe_out`, right in front of A32; not emitted by the frontend |
| A32 | `scatter` | `[G,dmodel]` (A31), `tok` (A21t) | `moe_out [Tq+1, dmodel]` (in place) | df16 | `dmodel` | write back to the same rows. Dummy slots all hit sink row `Tq` |
| — | &nbsp;&nbsp;**end loop `c`** | | | | | |
| — | **end loop `e`** | | | | | |

**Shared experts** — all `Tq` tokens

| # | Op | Inputs | Output shape | Precision | Stick dim | Notes |
|---|---|---|---|---|---|---|
| — | **loop `s = 0 .. Es-1`:** | | | | | |
| A33 | `shared_gate` | `x[Tq,dmodel]` (in) × `W_gate_s[s][dmodel,dff_s]` | `[Tq, dff_s]` | df16 | `dff_s` | |
| A34 | `shared_up` | `x[Tq,dmodel]` (in) × `W_up_s[s][dmodel,dff_s]` | `[Tq, dff_s]` | df16 | `dff_s` | |
| A35 | `shared_silu` | `[Tq,dff_s]` (A33) | `[Tq, dff_s]` | df16 | `dff_s` | on A33 only |
| A36 | `shared_mul` | `[Tq,dff_s]` (A35) × `[Tq,dff_s]` (A34) | `[Tq, dff_s]` | df16 | `dff_s` | |
| A37 | `shared_down` | `[Tq,dff_s]` (A36) × `W_down_s[s][dff_s,dmodel]` | `[Tq, dmodel]` | df16 | `dmodel` | |
| A38 | `add` | `moe_out[0:Tq,dmodel]` (A32) `+= [Tq,dmodel]` (A37) | **`[Tq, dmodel]`** (in place) | df16 | `dmodel` | every real token updated; sink row `Tq` excluded. After the last `s`: `moe_out[0:Tq]` is the final MoE block output |
| — | **end loop `s`** | | | | | |

Weight stick dims: `W_router[dmodel,E]` → `E`; `W_gate/W_up[E,dmodel,dff_e]` → `dff_e`; `W_down[E,dff_e,dmodel]` → `dmodel`.

A33–A37 do not depend on the router or the routed loop (only A38 touches `moe_out`), so they can run concurrently or interleave to fill bubbles — e.g. overlap with the router (A13–A17).

#### Router cost

Per prefill chunk (`Tq` = 8192 tokens, one MoE layer), at E = 128, K = 8. The routed-expert compute (A22–A32) is the same in both versions.

| | Earlier version (A14a–A21, A28) | Revised (A13–A17, A21t, A28) |
|---|---|---|
| Scattered sticks | 2 × 1,048,576 (A16a/b), 15/16 into the sink | 65,536 (A17b), none into a sink: 32× fewer |
| Destination fills | 2 × 134 MB (A15) | 1 × 128 MB (A17a) |
| Full-list restickifies | 2 × 134 MB (A17a/b) | none; each tile restickifies only its own `G` slots (A21t) |
| Per tile | one restickify (A28) | one restickify (A21t) and one gather (A28) |
| Router time per chunk | 1× | about 4–6× less |

All ops used by the revised list build exist already (`add`, `restickify`, `topk`, `init`, `scatter`, `gather`); the only op-level changes are the ones listed below.

### Path B — Decode (whole batch × selected experts)

There is no batch loop. One iteration covers one **selected** expert and all `B` tokens. The loop runs over the `E_sel` distinct experts picked anywhere in the batch, so its trip count is dynamic. Every token goes through every selected expert; a token that didn't pick that expert gets routing weight 0.

The router works out which experts at least one token picked (`max` over the batch; a max of 0 means not selected), then packs their ids into a list (`non_zero_pos_to_idx` + a 16 KB scatter). The expert loop then indexes each selected expert's weights and its column of routing weights directly with the expert id from that list.

Input: `x : [B, dmodel]`, stick dim `dmodel`

Constant: `exp_ids : [E]` int32 = `0 .. E-1`, precomputed. Feeds the expert-id scatter (B16) directly

**Router**

| # | Op | Inputs | Output shape | Precision | Stick dim | Notes |
|---|---|---|---|---|---|---|
| B1 | `router_gemm` | `x[B,dmodel]` (in) × `W_router[dmodel,E]` | `[B, E]` | df16 | `E` | GEMV-ish; `B ≤ 32` |
| B2 | `softmax` (dim=E) | `[B,E]` (B1) | `[B, E]` | df16 | `E` | reduces **along** the stick dim |
| B3 | `restickify` | `[B,E]` (B2) | `[B, E]` | df16 | `B` | stick dim `E`→`B`. `B ≤ 32` ⇒ partial sticks whenever `S > B` |
| B4 | `topk_index` (K=8, dim=E) | `[B,E]` (B3) | `idx [B,K]` | df16 | `B` | exact in df16 |
| B5 | `keep_by_index` | `[B,E]` (B3), `idx[B,K]` (B4) | `[B, E]` | df16 | `B` | keeps the top-K scores, others → 0 (same as prefill A5) |
| B6 | `sum_nonstick` (dim=E) | `[B,E]` (B5) | `[B, 1]` | df16 | `B` | renormalize denominator |
| B7 | `realdiv` | `[B,E]` (B5) ÷ `[B,1]` (B6) | `w_dense [B, E]` | df16 | `B` | renormalized routing weights; 0 where token `b` didn't pick expert `e` |
| B8 | `restickify` | `w_dense[B,E]` (B7) | `[B, E]` | df16 | `E` | stick dim `B`→`E`; df16 restickify keeps `E` contiguous (no stagger) |
| B9 | `max` (dim=B) | `[B,E]` (B8) | `sel [1, E]` | df16 | `E` | reduces **across** sticks: `sel[e]` = largest routing weight any token gives expert `e`. Weights are ≥ 0, so **`sel[e] = 0` ⇒ expert `e` not selected** |
| B10 | `df16tofp32` | `sel[1,E]` (B9) | `[1, E]` | fp32 | `E` (32) | output staggered along `E`, as in prefill A9 |
| B11 | `unstagger_stick_ea` | `[1,E]` (B10) | `[1, E]` | fp32 | `E` (32) | restores contiguous `E` order |
| B12 | `restickify` | `[1,E]` (B11) | `[E, 32]` | fp32 | dummy (32) | puts a dummy dim into the stick — one full stick per expert |
| B13a | `non_zero_pos_to_idx` (dim=E) **[NEW]** | `[E,32]` (B12) | `run_pos [E, 32]` | fp32 | dummy (32) | same op as prefill A11, scanning along `E`. Selected expert → its slot `j` among the selected experts; unselected → dummy location `E` (`padval = E`) |
| B13b | `count_non_zero_div_ceil` (dim=E, divisor=1) **[NEW]** | `[E,32]` (B12) | `E_sel [1, 32]` | fp32 | dummy (32) | same op as prefill A12 with divisor 1, i.e. a plain count: `E_sel` = number of selected experts. The 0 → 1 case never triggers, since every token selects `K` experts |
| B15 | `init` | — | `[E+1, 32]` | int32 | dummy (32) | scatter destination. Row `E` is the sink for unselected experts. Slots `≥ E_sel` are never read by the loop, so the fill value doesn't matter |
| B16i | `idx2Addr` *(inserted by the backend)* | `run_pos[E,32]` (B13a) | `run_addr [E, 32]` | int32 (input fp32) | dummy (32) | slot → address in the `[E+1,32]` scatter destination (B15). Not emitted by the frontend: the backend converts `run_pos` in front of B16. Input is fp32 (see [New and modified ops](#new-and-modified-ops)) |
| B16 | `scatter` (expert id) | `exp_ids[E]` (constant), index `run_pos` (B13a), dest (B15) | `sel_list [E+1, 32]` | int32 | dummy (32) | packs selected expert ids into slots `0..E_sel-1`. Only `E·128 B` = 16 KB at `E = 128` ⇒ cheap |

The loop reads slots `j = 0 .. E_sel-1` of `sel_list` (B16): `sel_list[j]` is the expert id that indexes `W_gate`, `W_up`, `W_down` and the column of `w_dense`.

**Routed experts**

| # | Op | Inputs | Output shape | Precision | Stick dim | Notes |
|---|---|---|---|---|---|---|
| B18 | `zero_init` | — | `acc [B, dmodel]` | df16 | `dmodel` | 0 tensor moved to device |
| — | **loop `j = 0 .. E_sel-1` (dynamic):** | | | | | |
| B19i | `idx2Addr` *(inserted by the backend)* (base = `W_gate`) | `sel_list[j]` (B16), one expert id | `addr_gate[j] [1, 32]` | int32 | dummy (32) | expert id → base address of that expert in `W_gate`, right in front of B19; not emitted by the frontend |
| B19 | `gate_gemm` | `x[B,dmodel]` (in), `W_gate[E,dmodel,dff_e]`, `sel_list[j]` | `[B, dff_e]` | df16 | `dff_e` | 3rd input is the expert id — the weights of that expert are fetched (indirect access). **Needs tweak — see [New and modified ops](#new-and-modified-ops)** |
| B20i | `idx2Addr` *(inserted by the backend)* (base = `W_up`) | `sel_list[j]` (B16), one expert id | `addr_up[j] [1, 32]` | int32 | dummy (32) | expert id → base address of that expert in `W_up`, right in front of B20; not emitted by the frontend |
| B20 | `up_gemm` | `x[B,dmodel]` (in), `W_up[E,dmodel,dff_e]`, `sel_list[j]` | `[B, dff_e]` | df16 | `dff_e` | **Needs tweak — see [New and modified ops](#new-and-modified-ops)** |
| B21 | `silu` | `[B,dff_e]` (B19) | `[B, dff_e]` | df16 | `dff_e` | on B19 only |
| B22 | `mul` | `[B,dff_e]` (B21) × `[B,dff_e]` (B20) | `[B, dff_e]` | df16 | `dff_e` | `silu(gate) * up` |
| B23i | `idx2Addr` *(inserted by the backend)* (base = `W_down`) | `sel_list[j]` (B16), one expert id | `addr_down[j] [1, 32]` | int32 | dummy (32) | expert id → base address of that expert in `W_down`, right in front of B23; not emitted by the frontend |
| B23 | `down_gemm` | `[B,dff_e]` (B22), `W_down[E,dff_e,dmodel]`, `sel_list[j]` | `[B, dmodel]` | df16 | `dmodel` | **Needs tweak — see [New and modified ops](#new-and-modified-ops)** |
| B24i | `idx2Addr` *(inserted by the backend)* (base = `w_dense`) | `sel_list[j]` (B16), one expert id | `addr_w[j] [1, 32]` | int32 | dummy (32) | expert id → address of column `e` of `w_dense[B,E]` (B7), right in front of B24; not emitted by the frontend |
| B24 | `mul` | `[B,dmodel]` (B23), `w_dense[B,E]` (B7), `sel_list[j]` | `[B, dmodel]` | df16 | `dmodel` | indirect access: fetches the `[B]` weight column of expert `sel_list[j]`, broadcast per token across `dmodel` sticks; zeroes non-routed tokens |
| B25 | `add` | `acc[B,dmodel]` (B18) `+= [B,dmodel]` (B24) | `[B, dmodel]` (in place) | df16 | `dmodel` | `acc` stays local across the loop; written out once |
| — | **end loop `j`** | | | | | |

**Shared experts** — all `B` tokens

| # | Op | Inputs | Output shape | Precision | Stick dim | Notes |
|---|---|---|---|---|---|---|
| — | **loop `s = 0 .. Es-1`:** | | | | | |
| B26 | `shared_gate` | `x[B,dmodel]` (in) × `W_gate_s[s][dmodel,dff_s]` | `[B, dff_s]` | df16 | `dff_s` | |
| B27 | `shared_up` | `x[B,dmodel]` (in) × `W_up_s[s][dmodel,dff_s]` | `[B, dff_s]` | df16 | `dff_s` | |
| B28 | `shared_silu` | `[B,dff_s]` (B26) | `[B, dff_s]` | df16 | `dff_s` | on B26 only |
| B29 | `shared_mul` | `[B,dff_s]` (B28) × `[B,dff_s]` (B27) | `[B, dff_s]` | df16 | `dff_s` | |
| B30 | `shared_down` | `[B,dff_s]` (B29) × `W_down_s[s][dff_s,dmodel]` | `[B, dmodel]` | df16 | `dmodel` | |
| B31 | `add` | `acc[B,dmodel]` (B25) `+= [B,dmodel]` (B30) | **`[B, dmodel]`** (in place) | df16 | `dmodel` | after the last `s`: final MoE block output |
| — | **end loop `s`** | | | | | |

### Multi-chip parallelization

**Strategy:** tensor parallelism (TP) across `D` devices (`D` = 4 or 8), with no expert parallelism. Every device holds a `1/D` slice of the intermediate dim of **every** expert, routed and shared:

| Weight | Split | Per-device slice |
|---|---|---|
| `W_gate`, `W_up` `[E, dmodel, dff_e]` | columns of `dff_e` | `[E, dmodel, dff_e/D]` |
| `W_down` `[E, dff_e, dmodel]` | rows of `dff_e` | `[E, dff_e/D, dmodel]` |
| `W_gate_s`, `W_up_s` / `W_down_s` | `dff_s`, same way | `dff_s/D` |
| `W_router` | replicated | `[dmodel, E]` |

**Rationale:** `dff_e ≈ 2048`, so each device keeps `dff_e/D` = 512 (`D = 4`) or 256 (`D = 8`). That is still 4–8 full df16 sticks, enough for efficient GEMMs. Every device runs the same experts on the same tokens with the same slice size, so the work is **balanced by construction**. With expert parallelism, load would follow the routing distribution and the most-loaded device would set the latency.

**Execution:**
- `x` is replicated, and every device runs the full router (prefill A1–A19, decode B1–B16) and gets identical routing.
- Each device then runs the routed-expert and shared-expert loops on its slices. The expert-id indexing of `W_gate`, `W_up` and `W_down` resolves to the device-local slices.
- Each device's `down_proj` output is a partial sum over its `dff_e/D` slice. The routing-weight `mul` and the `gather`/`add`/`scatter` into `moe_out` are linear, so partial sums can accumulate locally.
- A single **AllReduce** (sum over `D` devices) at the end of the MoE block combines routed and shared partials. It moves `[Tq, dmodel]` in prefill and `[B, dmodel]` in decode, df16.

In the op tables, read `dff_e` / `dff_s` as `dff_e/D` / `dff_s/D` per device.

**Pseudocode:**

```python
# D devices; device d holds slice F_d = [d*dff_e/D, (d+1)*dff_e/D) of every expert
# (and the matching dff_s/D slice of each shared expert)
on each device d in parallel:
    route = router(x)                        # replicated: same result on every device
    out_d = zeros(T, dmodel)                 # T = Tq (prefill) or B (decode)
    for e in experts(route):                 # prefill: all E, nc[e] tiles; decode: E_sel
        t, w = tokens(route, e), weights(route, e)
        h = silu(x[t] @ W_gate[e][:, F_d]) * (x[t] @ W_up[e][:, F_d])   # [|t|, dff_e/D]
        out_d[t] += w * (h @ W_down[e][F_d, :])                         # partial over F_d
    for s in range(Es):
        h = silu(x @ W_gate_s[s][:, Fs_d]) * (x @ W_up_s[s][:, Fs_d])
        out_d += h @ W_down_s[s][Fs_d, :]
    y = all_reduce_sum(out_d)                # one collective per MoE block
```

### New and modified ops

1. Matmul (decode B19, B20, B23): indirect access into the weight tensor; a `[B × dmodel]` GEMM rather than a GEMV.
2. `non_zero_pos_to_idx` (prefill A11, decode B13a): **new op**. One output: running position, with a dummy location for zero entries, passed as the op's `padval` parameter (prefill uses −∞, decode `E`). The op interface specifies a single scan dimension, which cannot be split across cores. The values written by the scatter come from a constant id tensor (`tok_ids` / `exp_ids`).
3. `count_non_zero_div_ceil` (prefill A12, decode B13b): **new op**. Count non-zeros along the reduction dimension, divide by `G` and round up, with 0 → 1. Decode uses divisor 1.
4. Index-to-address conversion: inserted by the backend in front of every gather, scatter and indirect access that takes an index tensor, so the frontend does not emit `idx2Addr`. The conversion must accept fp32 indices holding integer values (prefill `addr_k` from A15, decode `run_pos` from B13a); the other index tensors are int32.
5. Dynamic loops with data-dependent loop count (prefill tile loop `nc[e]` from A17c, decode expert loop `E_sel` from B13b):
   - a) SDSCbundle interface: `sdscbundle.load_value_at_address` reads the loop count from a tensor address, stored in FP32 as the first value of a stick, with `max_value` the largest count it can hold. Its result is the upper bound of the `scf.for`
   - b) Backend program looping where the loop count comes from a tensor read

`df16tofp32` outputs are staggered along the stick; `unstagger_stick_ea` restores contiguous order (see [RFC 2971 — FP32 Element Arrangement](../2971-FP32ElementArrangement/2971-FP32ElementArrangementRFC.md)).

## **Metrics**

* MoE block latency for prefill (per chunk) and decode (per step), compared with the current dense prefill and per-pair decode paths.
* Prefill: tokens actually computed per expert, compared with `Tq·K`. This measures padding overhead.
* Prefill: router time, in particular the list build A13–A17 and the per-tile router work (A21t, A28).
* Decode: weight bytes fetched per step, compared with `E_sel · (per-expert weight size)`.

## **Alternatives**

* **Dense prefill (current).** No gather, scatter or permutation, and perfectly regular GEMMs, but 16× the FLOPs. It remains the better choice for small chunks (`Tq ≲ 512`) where the block is memory-bound.
* **Per-(token, expert) decode loop (current).** A simpler router, but `B·K` GEMVs, with an expert's weights fetched again for every token that picked it.

## **Prior Art**

The prefill structure follows the usual sparse-MoE kernel pattern of per-expert gather → FFN → gather/add/scatter into a shared output buffer, with an expert loop followed by a shared-expert loop.

## **How we teach this**

* Op tables give stick dim and precision for every intermediate, so layout changes (restickify, df16 ↔ fp32, unstagger) are explicit.
* Loops marked **(dynamic)** have trip counts produced on device.

## **Unresolved questions**

* ~~How to optimize the prefill router scatters (A16a/b).~~ Addressed by the revised list build (A13–A17).

## Resolution

Pending review.

#### Tracking issue

https://github.com/torch-spyre/torch-spyre/issues/5979
