# Sparse MoE Compute on Spyre — Gemma4 26B-A4B

> **Status:** draft. Tracking issue: [#5979](https://github.com/torch-spyre/torch-spyre/issues/5979).

## **Summary**

This RFC proposes sparse kernels for the MoE (sparse FFN) block on Spyre, using Gemma4 26B-A4B as the target model.

* **Chunked prefill:** each expert computes only on the tokens routed to it, instead of every expert computing on every token. The chunk size is chosen so that each expert gets about 512 tokens on average (`Tq = 512·E/K` = 8192). Each expert's tokens are processed in 512-token tiles in a dynamic loop. A tile with fewer than 512 real tokens is padded with dummy tokens.
* **Decode:** there is no per-token loop. The whole batch runs through each *selected* expert in a dynamic loop over the `E_sel` distinct experts chosen anywhere in the batch, so each selected expert's weights are fetched once.
* **Shared experts** run on all tokens in both phases.
* The router needs three **new ops** (`non_zero_pos_to_idx`, `round_up`, `is_any_non_zero`) and changes to two existing ones.

Multi-chip parallelization will be added in a later revision.

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

### Path A — Chunked prefill (selected tokens per expert)

Each expert computes only on the tokens routed to it. Its routed tokens are processed in tiles of `G = 512`. For each tile:

1. Gather the tile's input rows.
2. Run the expert FFN.
3. Gather the same rows of a partial-output buffer `moe_out[Tq+1,dmodel]`.
4. Add, and scatter back.

A tile with fewer than 512 real tokens is padded with **dummy tokens**. These read a zero pad row appended to the input and write to a sink row in `moe_out`, so their results are discarded. An expert with `cnt[e] = 0` still runs one all-dummy tile, because A12 rounds 0 up to `G`. Every loop count is therefore ≥ 1, at the cost of at most one extra weight fetch per unused expert. After the routed experts, the shared expert(s) run on all `Tq` tokens and add into the same buffer.

The router turns the dense `[Tq,E]` routing weights into per-expert packed token lists:

* `non_zero_pos_to_idx` gives each (token, expert) entry its slot in the expert's list.
* A scatter packs the routing weights and token ids into those slots.
* `idx2Addr` converts the packed token ids into row addresses for the input and output gathers.

Input: `x : [Tq, dmodel]`, stick dim `dmodel`

Constant: `tok_ids : [Tq]` = `0 .. Tq-1`, precomputed. Feeds the token-id scatter (A16b) directly and is the same for every expert

**Router**

| # | Op | Inputs | Output shape | Precision | Stick dim | Notes |
|---|---|---|---|---|---|---|
| A1 | `router_gemm` | `x[Tq,dmodel]` (in) × `W_router[dmodel,E]` | `[Tq, E]` | df16 | `E` | |
| A2 | `softmax` (dim=E) | `[Tq,E]` (A1) | `[Tq, E]` | df16 | `E` | reduces **along** the stick dim |
| A3 | `restickify` | `[Tq,E]` (A2) | `[Tq, E]` | df16 | `Tq` | stick dim `E`→`Tq`; moves the selection/reduction axis `E` off the stick |
| A4 | `topk_idx` (K=8, dim=E) | `[Tq,E]` (A3) | `idx [Tq,K]` | df16 | `Tq` | index output only; selects **across** sticks. Exact in df16 (`E = 128` needs 7 bits) |
| A5 | `mask_by_index` | `[Tq,E]` (A3), `idx[Tq,K]` (A4) | `[Tq, E]` | df16 | `Tq` | non-top-K scores → 0. **Needs tweak — see [New and modified ops](#new-and-modified-ops)** |
| A6 | `sum_nonstick` (dim=E) | `[Tq,E]` (A5) | `[Tq, 1]` | df16 | `Tq` | renormalize denominator |
| A7 | `realdiv` | `[Tq,E]` (A5) ÷ `[Tq,1]` (A6) | `[Tq, E]` | df16 | `Tq` | renormalized routing weights |
| A8 | `restickify` | `[Tq,E]` (A7) | `[Tq, E]` | df16 | `E` | stick dim `Tq`→`E`; brings `E` back into the stick |
| A9 | `df16tofp32` | `[Tq,E]` (A8) | `[Tq, E]` | fp32 | `E` (32) | 32 `E` per stick. Output is **staggered** along `E`: each 64-wide df16 stick yields fp32 sticks holding `0..3, 8..11, …` then `4..7, 12..15, …`. fp32 is needed because token ids up to `Tq = 8192` aren't exact in df16 |
| A10 | `unstagger_stick_ea` | `[Tq,E]` (A9) | `score [Tq, E]` | fp32 | `E` (32) | restores contiguous `E` order |
| A11 | `non_zero_pos_to_idx` (dim=Tq) **[NEW]** | `score[Tq,E]` (A10) | a) `run_pos [Tq, E]`<br>b) `cnt [1, E]` | fp32 | `E` (32) | scans **across** sticks along `Tq`, independently per `E` lane. a) non-zero entry → number of non-zeros before it in this `E` column (its slot in the expert's list); zero entry → dummy location `Tq` (the `(Tq+1)`-th row). b) non-zeros per expert = `cnt[e]` |
| A12 | `round_up` (to G) | `cnt[1,E]` (A11b) | `cnt_r [1, E]` | fp32 | `E` (32) | `cnt_r[e] = max(G, ⌈cnt[e]/G⌉·G)`. **0 rounds to `G`**, so `nc[e] = cnt_r[e]/G ≥ 1` |
| A13 | `idx2Addr` | `run_pos[Tq,E]` (A11a) | `run_addr [Tq, E]` | int32 | `E` (32) | slot → address in the `[Tq+1,E,32]` scatter destination (A15). Fed to the A16 scatters as is, with no dummy dim |
| A14a | `restickify` | `score[Tq,E]` (A10) | `[Tq, E, 32]` | fp32 | dummy (32) | puts a dummy dim into the stick — one full stick per (token, expert) entry |
| A14b | `restickify` | `cnt_r[1,E]` (A12) | `[1, E, 32]` | fp32 | dummy (32) | one stick per expert; source of the `nc[e]` loop bound |
| A15 | `init` (fill = `Tq`) | — | `[Tq+1, E, 32]` | fp32 | dummy (32) | every entry points at the dummy pad token (row `Tq` of `x_pad`, A19). Initial destination for A16a and A16b (one copy each) |
| A16a | `scatter` (score) | `[Tq,E,32]` (A14a), `run_addr[Tq,E]` (A13), dest (A15) | `[Tq+1, E, 32]` | fp32 | dummy (32) | entry `(t,e)` → row `run_pos[t,e]`, expert `e`. Zero entries all land in dummy row `Tq` |
| A16b | `scatter` (token id) | `tok_ids[Tq]` (constant), `run_addr[Tq,E]` (A13), dest (A15) | `[Tq+1, E, 32]` | fp32 | dummy (32) | entry `(t,e)` writes token id `t` to row `run_pos[t,e]`, expert `e`; the same `tok_ids` is used for every `e`. **Slowest op:** each scatter is `Tq·E·128 B` = 128 MB at `E = 128`, ~256 MB for A16a+A16b ⇒ can take **~10+ ms**. WIP to optimize |
| A17a | `restickify` | `[Tq+1,E,32]` (A16a) | `w_list [Tq+1, E]` | fp32 | `Tq` | brings `Tq` into the stick, drops the dummy dim. Column `e` = weights in slot order |
| A17b | `restickify` | `[Tq+1,E,32]` (A16b) | `tok_list [Tq+1, E]` | fp32 | `Tq` | as above. Column `e` = token ids routed to `e`, packed into slots `0..cnt[e]-1`; remaining slots = `Tq` (pad) |
| A18 | `zero_init` | — | `moe_out [Tq+1, dmodel]` | df16 | `dmodel` | partial-output space. Row `Tq` is a **sink** for dummy-slot writes |
| A19 | `pad` | `x[Tq,dmodel]` (in) | `x_pad [Tq+1, dmodel]` | df16 | `dmodel` | appends 1 dummy token at row `Tq`, all zeros. Dummy slots of `w_list` keep the A15 fill value (`Tq`) rather than 0, so the pad token must be zero: the bias-free FFN then outputs 0 and the sink row stays finite |
| A20 | `idx2Addr` (base = `x_pad`) | `tok_list[Tq+1,E]` (A17b) | `addr_in [Tq+1, E]` | int32 | `Tq` | token id → row address in `x_pad`. **Needs tweak — see [New and modified ops](#new-and-modified-ops)** |
| A21 | `idx2Addr` (base = `moe_out`) | `tok_list[Tq+1,E]` (A17b) | `addr_out [Tq+1, E]` | int32 | `Tq` | token id → row address in `moe_out` |

**Routed experts**

`nc[e] = cnt_r[e]/G ≥ 1` (A12 → A14b). Tile slots run up to `nc[e]·G − 1 ≤ Tq − 1`, since `cnt_r[e] ≤ Tq` (`cnt[e] ≤ Tq`, `Tq` is a multiple of `G`, and `G ≤ Tq`), so tiles never read past `Tq+1` rows. Slots from `cnt[e]` up to the end of the last tile hold the pad id `Tq`. Those dummy tokens read `x_pad[Tq]` and write `moe_out[Tq]` (the sink).

| # | Op | Inputs | Output shape | Precision | Stick dim | Notes |
|---|---|---|---|---|---|---|
| — | **loop `e = 0 .. E-1`:** | | | | | |
| — | &nbsp;&nbsp;**loop `c = 0 .. nc[e]-1` (dynamic):** | | | | | slice `[c·G : (c+1)·G, e]` of `addr_in`, `addr_out`, `w_list` |
| A22 | `gather` (input) | `x_pad[Tq+1,dmodel]` (A19), `addr_in[·,e]` (A20) | `exp_in [G, dmodel]` | df16 | `dmodel` | row gather — whole `dmodel` sticks per token |
| A23 | `gate_proj` | `exp_in[G,dmodel]` (A22) × `W_gate[e][dmodel,dff_e]` | `[G, dff_e]` | df16 | `dff_e` | regular `512 × dmodel × dff_e` GEMM |
| A24 | `up_proj` | `exp_in[G,dmodel]` (A22) × `W_up[e][dmodel,dff_e]` | `[G, dff_e]` | df16 | `dff_e` | weights for expert `e` reused across its `nc[e]` tiles |
| A25 | `silu` | `[G,dff_e]` (A23) | `[G, dff_e]` | df16 | `dff_e` | on A23 only |
| A26 | `mul` | `[G,dff_e]` (A25) × `[G,dff_e]` (A24) | `[G, dff_e]` | df16 | `dff_e` | `silu(gate) * up` |
| A27 | `down_proj` | `[G,dff_e]` (A26) × `W_down[e][dff_e,dmodel]` | `[G, dmodel]` | df16 | `dmodel` | |
| A28 | `restickify` | `w_list[c·G:(c+1)·G, e]` (A17a) | `w_tile [G, 32]` | fp32 | dummy (32) | stick dim `Tq`→dummy; one weight stick per token |
| A29 | `mul` | `[G,dmodel]` (A27) × `w_tile[G,32]` (A28) | `[G, dmodel]` | df16 | `dmodel` | one weight stick broadcast across each token's `dmodel` sticks |
| A30 | `gather` (partial out) | `moe_out[Tq+1,dmodel]` (A18 / A32), `addr_out[·,e]` (A21) | `[G, dmodel]` | df16 | `dmodel` | same tokens as A22 |
| A31 | `add` | `[G,dmodel]` (A29) + `[G,dmodel]` (A30) | `[G, dmodel]` | df16 | `dmodel` | |
| A32 | `scatter` | `[G,dmodel]` (A31), `addr_out[·,e]` (A21) | `moe_out [Tq+1, dmodel]` (in place) | df16 | `dmodel` | write back to the same rows. Dummy slots all hit sink row `Tq` |
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

A33–A37 do not depend on the router or the routed loop (only A38 touches `moe_out`), so they can run concurrently or interleave to fill bubbles — e.g. overlap with the A16 scatters.

### Path B — Decode (whole batch × selected experts)

There is no batch loop. One iteration covers one **selected** expert and all `B` tokens. The loop runs over the `E_sel` distinct experts picked anywhere in the batch, so its trip count is dynamic. Every token goes through every selected expert; a token that didn't pick that expert gets routing weight 0.

The router works out which experts at least one token picked (`is_any_non_zero`), then packs their ids into a list (`non_zero_pos_to_idx` + a 16 KB scatter). From that list it builds address sticks for each expert's weights and for that expert's column of routing weights.

Input: `x : [B, dmodel]`, stick dim `dmodel`

Constant: `exp_ids : [E]` = `0 .. E-1`, precomputed. Feeds the expert-id scatter (B16) directly

**Router**

| # | Op | Inputs | Output shape | Precision | Stick dim | Notes |
|---|---|---|---|---|---|---|
| B1 | `router_gemm` | `x[B,dmodel]` (in) × `W_router[dmodel,E]` | `[B, E]` | df16 | `E` | GEMV-ish; `B ≤ 32` |
| B2 | `softmax` (dim=E) | `[B,E]` (B1) | `[B, E]` | df16 | `E` | reduces **along** the stick dim |
| B3 | `restickify` | `[B,E]` (B2) | `[B, E]` | df16 | `B` | stick dim `E`→`B`. `B ≤ 32` ⇒ partial sticks whenever `S > B` |
| B4 | `topk_index` (K=8, dim=E) | `[B,E]` (B3) | `idx [B,K]` | df16 | `B` | exact in df16 |
| B5 | `mask_by_index` | `[B,E]` (B3), `idx[B,K]` (B4) | `[B, E]` | df16 | `B` | non-top-K scores → 0 (same as prefill A5). **Needs tweak — see [New and modified ops](#new-and-modified-ops)** |
| B6 | `sum_nonstick` (dim=E) | `[B,E]` (B5) | `[B, 1]` | df16 | `B` | renormalize denominator |
| B7 | `realdiv` | `[B,E]` (B5) ÷ `[B,1]` (B6) | `w_dense [B, E]` | df16 | `B` | renormalized routing weights; 0 where token `b` didn't pick expert `e` |
| B8 | `restickify` | `w_dense[B,E]` (B7) | `[B, E]` | df16 | `E` | stick dim `B`→`E`; df16 restickify keeps `E` contiguous (no stagger) |
| B9 | `is_any_non_zero` (dim=B) | `[B,E]` (B8) | `sel [1, E]` | df16 | `E` | reduces **across** sticks: `sel[e] = 1` if any token picked expert `e`, else 0 |
| B10 | `df16tofp32` | `sel[1,E]` (B9) | `[1, E]` | fp32 | `E` (32) | output staggered along `E`, as in prefill A9 |
| B11 | `unstagger_stick_ea` | `[1,E]` (B10) | `[1, E]` | fp32 | `E` (32) | restores contiguous `E` order |
| B12 | `restickify` | `[1,E]` (B11) | `[E, 32]` | fp32 | dummy (32) | puts a dummy dim into the stick — one full stick per expert |
| B13 | `non_zero_pos_to_idx` (dim=E) **[NEW]** | `[E,32]` (B12) | a) `run_pos [E, 32]`<br>b) `E_sel [1, 32]` | fp32 | dummy (32) | same op as prefill A11, scanning along `E`. a) selected expert → its slot `j` among the selected experts; unselected → dummy location `E`. b) number of selected experts = `E_sel` |
| B14 | `idx2Addr` | `run_pos[E,32]` (B13a) | `run_addr [E, 32]` | int32 | dummy (32) | slot → address in the `[E+1,32]` scatter destination (B15) |
| B15 | `init` | — | `[E+1, 32]` | fp32 | dummy (32) | scatter destination. Row `E` is the sink for unselected experts. Slots `≥ E_sel` are never read by the loop, so the fill value doesn't matter |
| B16 | `scatter` (expert id) | `exp_ids[E]` (constant), `run_addr` (B14), dest (B15) | `sel_list [E+1, 32]` | fp32 | dummy (32) | packs selected expert ids into slots `0..E_sel-1`. Only `E·128 B` = 16 KB at `E = 128` ⇒ cheap |
| B17a | `idx2Addr` (base = `W_gate`) | `sel_list[E+1,32]` (B16) | `addr_gate [E+1, 32]` | int32 | dummy (32) | expert id → base address of that expert in `W_gate`; one address stick per slot. **Needs tweak — see [New and modified ops](#new-and-modified-ops)** |
| B17b | `idx2Addr` (base = `W_up`) | `sel_list[E+1,32]` (B16) | `addr_up [E+1, 32]` | int32 | dummy (32) | as above for `W_up` |
| B17c | `idx2Addr` (base = `W_down`) | `sel_list[E+1,32]` (B16) | `addr_down [E+1, 32]` | int32 | dummy (32) | as above for `W_down` |
| B17d | `idx2Addr` (base = `w_dense`) | `sel_list[E+1,32]` (B16) | `addr_w [E+1, 32]` | int32 | dummy (32) | expert id → address of column `e` of `w_dense[B,E]` (B7, stick dim `B`) — the `[B]` weights of all tokens for that expert |

The loop reads slots `j = 0 .. E_sel-1` of `addr_gate` / `addr_up` / `addr_down` / `addr_w`.

**Routed experts**

| # | Op | Inputs | Output shape | Precision | Stick dim | Notes |
|---|---|---|---|---|---|---|
| B18 | `zero_init` | — | `acc [B, dmodel]` | df16 | `dmodel` | 0 tensor moved to device |
| — | **loop `j = 0 .. E_sel-1` (dynamic):** | | | | | |
| B19 | `gate_gemm` | `x[B,dmodel]` (in), `W_gate[E,dmodel,dff_e]`, `addr_gate[j]` | `[B, dff_e]` | df16 | `dff_e` | 3rd input is the address stick — weights fetched at `addr_gate[j]`. **Needs tweak — see [New and modified ops](#new-and-modified-ops)** |
| B20 | `up_gemm` | `x[B,dmodel]` (in), `W_up[E,dmodel,dff_e]`, `addr_up[j]` | `[B, dff_e]` | df16 | `dff_e` | **Needs tweak — see [New and modified ops](#new-and-modified-ops)** |
| B21 | `silu` | `[B,dff_e]` (B19) | `[B, dff_e]` | df16 | `dff_e` | on B19 only |
| B22 | `mul` | `[B,dff_e]` (B21) × `[B,dff_e]` (B20) | `[B, dff_e]` | df16 | `dff_e` | `silu(gate) * up` |
| B23 | `down_gemm` | `[B,dff_e]` (B22), `W_down[E,dff_e,dmodel]`, `addr_down[j]` | `[B, dmodel]` | df16 | `dmodel` | **Needs tweak — see [New and modified ops](#new-and-modified-ops)** |
| B24 | `mul` | `[B,dmodel]` (B23), `w_dense[B,E]` (B7), `addr_w[j]` (B17d) | `[B, dmodel]` | df16 | `dmodel` | indirect access: fetches the `[B]` weight column at `addr_w[j]`, broadcast per token across `dmodel` sticks; zeroes non-routed tokens |
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

*To be updated soon.*

### New and modified ops

1. `mask_by_index` (prefill A5, decode B5): mask the elements **not** in the index list.
2. Matmul (decode B19, B20, B23): indirect access into the weight tensor; a `[B × dmodel]` GEMM rather than a GEMV.
3. `non_zero_pos_to_idx` (prefill A11, decode B13): **new op**. Two outputs: running position (with a dummy location `Tq` / `E` for zero entries) and count. The index values come from a constant id tensor (`tok_ids[Tq]` / `exp_ids[E]`) fed straight to the scatter.
4. `round_up` (prefill A12): **new op**. Round to a multiple of `G`, with 0 → `G`.
5. `is_any_non_zero` (decode B9): **new op**. Reduce across sticks to a 0/1 flag.
6. `idx2Addr` (prefill A13, A20, A21; decode B14, B17a–d): accept fp32 input. It doesn't today.

`df16tofp32` outputs are staggered along the stick; `unstagger_stick_ea` restores contiguous order (see [RFC 2971 — FP32 Element Arrangement](../2971-FP32ElementArrangement/2971-FP32ElementArrangementRFC.md)).

## **Metrics**

* MoE block latency for prefill (per chunk) and decode (per step), compared with the current dense prefill and per-pair decode paths.
* Prefill: tokens actually computed per expert, compared with `Tq·K`. This measures padding overhead.
* Prefill: router time, in particular the A16a/b scatters.
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

* How to optimize the prefill router scatters (A16a/b).
* Multi-chip parallelization (to be added).

## Resolution

Pending review.

#### Tracking issue

https://github.com/torch-spyre/torch-spyre/issues/5979
