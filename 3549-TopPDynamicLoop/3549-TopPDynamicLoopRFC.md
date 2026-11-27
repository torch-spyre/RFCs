# Top-p on Spyre with a Dynamic Loop

> Tracking issue: [#3549](https://github.com/torch-spyre/torch-spyre/issues/3549).

## **Summary**

This RFC proposes how to compute the top-p (nucleus) candidates of a batch of rows on Spyre, for a sampler that runs on the device afterwards.

* Top-k on Spyre returns at most 128 elements per call. Larger sets are taken **128 at a time**: a `topk` with k = 128, then a `maskbyindex` that writes 0 over the elements just taken, so the next `topk` finds the 128 after them.
* How many groups of 128 top-p needs depends on the data: usually one or a few, all of the vocabulary in the worst case. The groups are taken in a **dynamic loop whose bound the loop body updates**. The bound starts at one iteration, and each iteration raises it by one while some row's accumulated probability is still below its `p`.
* Each iteration also makes the **exact cut** within its group: it keeps the candidates of each row up to the one that brings its mass to `p`, and sets the others to 0.
* The whole computation is one bundle. Its outputs are what the sampler needs: candidate token ids and their probabilities, with 0 for every candidate not kept.

The cumsum it needs is a native op (backend PR 5023), so no other new op is needed. The RFC relies on a loop bound read from device memory with `sdscbundle.load_value_at_address`, and it proposes one sentence for the spec to allow writing that bound from inside the loop it bounds.

## **Motivation**

Moving sampling onto the device ([#5841](https://github.com/torch-spyre/torch-spyre/issues/5841)) removes the copy of the full `[batch, vocab]` logits to the host on every decode step, which grows with the batch. Top-p is the part of sampling that doesn't map onto the device ops as they are:

* `topk` returns at most k = 128 per call ([#5854](https://github.com/torch-spyre/torch-spyre/issues/5854)), and top-p without a top-k can need far more candidates.
* The design in [#3549](https://github.com/torch-spyre/torch-spyre/issues/3549) takes `K_max` candidates in a fixed number of `⌈K_max/128⌉` peeling passes and applies top-p afterwards. It notes that top-p could stop early once every row reached its `p`, but that a compiled graph had no data-dependent loop termination. To be safe for any distribution, a fixed pass count either costs the worst case on every step or caps the candidates and changes the result.

Bundles now support loops whose upper bound is a value read from device memory ([interface-specs#23](https://github.com/torch-spyre/interface-specs/pull/23)). The loop reads the value and decides at the end of each iteration whether to go round again. If the loop body writes that value, the loop stops as soon as every row has reached `p`. Peaked distributions, the common case, then take one or two groups of 128, and only the rare flat ones pay for more.

## **Proposed Implementation**

### Configuration and notation

| Symbol | Meaning |
|---|---|
| `V` | vocabulary size, e.g. 131072 or 262144 |
| `B` | rows in the batch. 32 rows fill one FP32 stick; more rows take more sticks |
| `G` | candidates per group, **128** (the largest k of one `topk`) |
| `Gmax` | `= ⌈V/G⌉`, the groups that cover the whole vocabulary: the worst case, `p` reached only with every token. The `max_value` of the loop bound |
| `R` | `= Gmax·G`, candidate rows of the outputs: `V` rounded up to a multiple of 128, so `R = V` for a vocabulary that is a multiple of 128 |
| `p[b]` | top-p threshold of row `b` |
| `S` | stick length: 32 elements at FP32, 128 B |

Every tensor is FP32 with the batch dim `B` in the stick, `[N, B]` meaning `N` rows of `⌈B/32⌉` sticks each:

* `topk` and `maskbyindex` scan a dim that cannot be in the stick, nor split across cores. That dim is `V` in the loop.
* Token ids above 1024 are not exact in DF16 ([#5847](https://github.com/torch-spyre/torch-spyre/issues/5847)), so the ids, and with them the whole computation, are FP32.

### What the sampler needs

The sampler this RFC targets draws with the exponential race: `token[b] = argmax_r prob[r,b] / q[r,b]`, with `q ~ Exp(1)` from the device RNG ([#5851](https://github.com/torch-spyre/torch-spyre/issues/5851)). That is the form vLLM's sampler uses, and it needs:

* the token ids of the candidates;
* their probabilities, **not renormalized**: dividing every candidate by the kept mass does not change the `argmax`;
* nothing else. A candidate with probability 0 never wins, so the outputs can have the fixed size `R`, with 0 for candidates not kept and for groups not taken.

The bundle **outputs** exactly these, plus the number of groups per row, which a sampler can use to limit its work:

| Output | Shape | Precision | Stick dim | Role |
|---|---|---|---|---|
| `cand_prob` | `[R, B]` | fp32 | `B` | kept probabilities in descending order; 0 for candidates past the cut and for groups not taken |
| `cand_tok` | `[R, B]` | fp32 | `B` | token ids, integers held in FP32; meaningful where `cand_prob > 0` |
| `ngroups` | `[1, B]` | fp32 | `B` | groups that hold a kept candidate, `≤ Gmax`. Not needed for correctness |

Its **inputs** are:

| Input | Shape | Precision | Stick dim | Role |
|---|---|---|---|---|
| `probs` | `[V, B]` | fp32 | `B` | softmax output, after temperature. Masked in place by the loop, so consumed |
| `p` | `[1, B]` | fp32 | `B` | threshold per row |
| `zero` | `[1, B]` | fp32 | `B` | constant 0 in every lane: initial values, and the value of a candidate past the cut |
| `one` | `[1, B]` | fp32 | `B` | constant 1 in every lane: initial `ngroups` |
| `gmax` | `[1, B]` | fp32 | `B` | constant `Gmax` in every lane: the clamp in T10 |

The three constants are precomputed, like any constant tensor. They could also be one `[3, B]` input, read at three offsets, which saves two arguments. It is a small gain at best.

An inverse-CDF sampler would also need the kept mass: `acc` (see below), as one more output.

### Algorithm

```python
# inputs probs [V,B], p [1,B]; outputs cand_prob [R,B], cand_tok [R,B], ngroups [1,B]
cand_prob[:] = 0; ngroups[:] = 1; acc[:] = 0
g = 0
while True:                                         # one device loop, bound read from memory
    rows = slice(g*G, (g+1)*G)
    cand_tok[rows]  = topkindex(probs, k=G, dim=V)  # next 128 of each row, descending
    cand_prob[rows] = topkvalue(probs, k=G, dim=V)
    probs = maskbyindex(probs, cand_tok[rows], maskval=0)          # in place
    cs = cumsum(cand_prob[rows], dim=0)             # within the group
    before = acc + cs - cand_prob[rows]             # mass before each candidate
    cand_prob[rows] = where(before >= p, 0, cand_prob[rows])       # exact cut
    acc += cs[G-1]                                  # the group's sum: last row of cs
    cont = acc < p                                  # 1 for a row still below p
    ngroups = min(ngroups + cont, Gmax)             # the min is a precaution
    bound = ngroups.max()                           # g+2 if a row continues, else g+1
    g += 1
    if bound <= g: break                            # the loop's own exit test
```

**Prep**

| # | Op | Inputs | Output shape | Precision | Stick dim | Notes |
|---|---|---|---|---|---|---|
| P1 | `identity` | `zero`, broadcast over `R` | `cand_prob [R, B]` | fp32 | `B` | groups not taken stay 0 |
| P2 | `identity` | `one` | `ngroups [1, B]` | fp32 | `B` | the first group always runs |
| P3 | `identity` | `zero` | `acc [1, B]` | fp32 | `B` | probability taken so far |

The loop bound needs no initial value: the loop reads it only at the end of an iteration, after the body wrote it.

**Loop over groups** — `bound = load_value_at_address(bound_addr) {max_value = Gmax}`

| # | Op | Inputs | Output shape | Precision | Stick dim | Notes |
|---|---|---|---|---|---|---|
| — | **loop `g = 0 .. bound-1` (dynamic):** | | | | | the body writes `bound` (T11) |
| T1 | `topkindex` (k=G, dim=V) | `probs[V,B]` | `cand_tok[gG:(g+1)G, B]` | fp32 | `B` | ids of the next 128 of each row. The output address is affine in `g` |
| T2 | `topkvalue` (k=G, dim=V) | `probs[V,B]` | `cand_prob[gG:(g+1)G, B]` | fp32 | `B` | the same scan, so the same order as T1, descending |
| T3 | `maskbyindex` (maskval=0) | `probs[V,B]`, `cand_tok` group (T1) | `probs [V, B]` (in place) | fp32 | `B` | the next T1/T2 find the 128 after these |
| T4 | `cumsum` (dim=G) | `cand_prob` group (T2) | `cs [G, B]` | fp32 | `B` | the native `cumsum` op (see [Op changes](#op-changes)), instead of the decomposition of `torch.cumsum` on device ([#3389](https://github.com/torch-spyre/torch-spyre/pull/3389)) |
| T5 | `sub`, `add` | `cs` (T4), `cand_prob` group, `acc` | `before [G, B]` | fp32 | `B` | `acc + cs − cand_prob`: the mass before each candidate. `acc` is broadcast over the group |
| T6 | `greaterequal` | `before` (T5), `p` | `drop [G, B]` | fp32 | `B` | 1 for a candidate after the mass reached `p`, else 0 |
| T7 | `where3` | `drop` (T6), `zero`, `cand_prob` group | `cand_prob` group (in place) | fp32 | `B` | the exact cut: 0 past it |
| T8 | `add` | `acc`, `cs` row `G−1` (T4) | `acc [1, B]` (in place) | fp32 | `B` | the last row of the cumsum is the group's sum, before the cut |
| T9 | `lesserthan` | `acc` (T8), `p` | `cont [1, B]` | fp32 | `B` | 1 for a row still below `p`, else 0 |
| T10 | `add`, `minimum` | `ngroups`, `cont` (T9), `gmax` | `ngroups [1, B]` (in place) | fp32 | `B` | `min(ngroups + cont, Gmax)`. Exact: integers in FP32. The `min` is a precaution (see below) |
| T11 | `max` (dim=B) | `ngroups` (T10) | `bound` | fp32 | `B` | the bound is the first value of its stick. With `B` not a whole number of sticks, the lanes past `B` are left out, by coordinate masking or a lane mask (see below). With `B = 1`, T11 goes and `ngroups` is the bound |
| — | **end loop `g`** | | | | | exits when `bound ≤ g+1` |

The compares (T6, T9) write 1.0 for true and 0.0 for false, so `cont` adds to `ngroups` directly. `lesserthan` is a backend op that the spec's op table doesn't list yet.

After iteration `g`, every row still below `p` has `ngroups[b] = g+2`, and every row that reached it has `ngroups[b] ≤ g+1`. So `bound = g+1` exactly when the whole batch is done, which is when the loop exits.

**The cut.** The groups are in descending order one after the other, not only within each: every group comes from what the earlier ones left. So `before` is the mass of all of the row's candidates ahead of this one, and a candidate is kept while that is below `p`. The rule also drops the whole group of a row that reached `p` in an earlier iteration, while the loop goes on for the rest of the batch, because its `before` starts at `acc ≥ p`.

The example [`top_p_bundle.mlir`](top_p_bundle.mlir) is the whole bundle, with `V = 131072` (so `Gmax = 1024`, `R = V`), written with the native cumsum. Each SDSC is named after its op.

### Interface change: a bound written in its own loop

The SuperDSC-Bundle spec says a loop bounded by `sdscbundle.load_value_at_address` reads the value and decides at the end of each iteration whether to go round again. This RFC relies on a value that an SDSC inside the loop writes, so the spec should say so explicitly. Proposed sentence, under the third kind of upper bound in "Loops":

> The value can also be written by an SDSC inside the loop it bounds: what an iteration writes decides whether the next one runs.

`scf.for` reads its bounds once, so the bundle relies on this reading: no pass may treat the bound as a value fixed for the loop. The loop needs no loop-carried value, which the spec doesn't support. All the state (`probs`, `acc`, `ngroups`, the bound) is in device memory.

### Caveats

* **`maskval` is 0, not -inf.** Probabilities are ≥ 0, so a masked 0 still loses to every candidate that is left, and the sums stay finite. With -inf, a group taken after the whole vocabulary would put -inf into `acc` and NaN into the cut. That happens only when `p` is reached with every token, but `Gmax` is sized for exactly that. A masked token picked again has probability 0, so it is never sampled.
* **In-place `maskbyindex`.** T3 writes `probs` over itself. The backend is expected to support an op whose output is its input. The frontend has to emit it that way, which should not need more than lifting a restriction there: a body cannot alternate between two buffers instead, because an address can only be affine in `g`, and `g mod 2` isn't.
* **Cost per group.**
  * T1, T2 and T3 each make a full pass over `probs`: 16 MB at V = 131072 and 32 MB at V = 262144, for up to 32 rows. Each further 32 rows add as much again.
  * The scanned dim cannot be split across cores. k = 128 splits as 4 per core over 32 cores ([#5854](https://github.com/torch-spyre/torch-spyre/issues/5854)).
  * The cut (T4–T7) works on the group's 128 rows only.
* **Worst case.** A flat distribution, or `p` close to 1, can take up to `Gmax` groups, every one with three passes over `probs`. `Gmax = ⌈V/128⌉` keeps the result exact in every case. A frontend that prefers a bound on cost can give a smaller `max_value`, which caps the candidates at `max_value·128` and changes the result for the rows that would need more.
* **The clamp to `Gmax`** (T10) is a precaution. Once all of the vocabulary is taken, `acc` is the whole mass, so `p ≤ 1` stops the loop. Only rounding can leave `acc` just below a `p` of 1: then the loop still ends after `Gmax` iterations, at `max_value`, and the clamp keeps `ngroups` at `Gmax` instead of `Gmax + 1`.
* **The batch runs as long as its slowest row.** A row that is done keeps taking groups until the last row is done. The cut sets its extra groups to 0.
* **A batch that isn't a whole number of sticks.** The lanes past `B` in the last stick hold whatever the buffers contain. Every op except T11 works lane by lane, so those lanes only produce candidates nobody reads. T11 is the one op that combines lanes: it reduces over the stick to get the bound. Its input is never NaN: P2 sets every lane of `ngroups` to 1, and T10 only adds the 0 or 1 of a compare and clamps to `Gmax`. But a lane past `B` could keep the loop going up to `Gmax`, so T11 must leave those lanes out.
  * **`B` fixed at compile time:** T11 uses coordinate masking, which leaves the lanes past `B` out of the reduction.
  * **`B` symbolic:** coordinate masking is set when the bundle is compiled, so it cannot follow the batch. Instead, T11 first multiplies `ngroups` by an input `lanes [1, B]` that holds 1 in the lanes of real rows and 0 in the others, made with the batch. The lanes past `B` then give 0, and the real rows give at least 1, so the `max` is the bound of the real rows. That is one more `mul` SDSC per iteration, on one stick per 32 rows.
  * An alternative without the `mul`: whoever writes `p` writes 0 in the lanes past `B`. `lesserthan(acc, 0)` is then 0 in those lanes, also when `acc` is NaN (a NaN compares false), so their `ngroups` stays at 1, below the bound of any real row. It relies on the padding of `p`, and on the NaN behaviour of the compare.
* **Element arrangement.** If `probs` comes from a DF16 softmax, restickified to `[V,B]` and then converted to FP32, its `B` lanes are in the staggered FP32 arrangement ([RFC 2971](../2971-FP32ElementArrangement/2971-FP32ElementArrangementRFC.md)). Every per-row tensor must have the same arrangement: `p`, `acc`, `ngroups` and the outputs. The sampler can work in that arrangement and restore the order only for the sampled tokens.
* **P1 writes all of `cand_prob`**, `R` rows, once per step: a pass the size of `probs`, smaller than one group. A sampler that reads only the groups the loop took can skip it, by looping over `max ngroups` groups itself.

### Op changes

* **T4 uses the native `cumsum` op** ([#3388](https://github.com/torch-spyre/torch-spyre/issues/3388)), now in the backend (PR 5023), together with `cumprod`, `cummax` and `cummin`. The frontend should lower `torch.cumsum` to it instead of the decomposition ([#3389](https://github.com/torch-spyre/torch-spyre/pull/3389)): one SDSC instead of `log2 n` full passes. What held a cumsum op back was the SDSC interface: it had no way to name a dim that an op scans along without reducing it. `non_zero_pos_to_idx` added that, as `scanDim_` in the op's attributes, and the cumulative ops are specified the same way: the same shape and layout in and out, `scanDim_` not in the stick and not split across cores ([interface-specs#24](https://github.com/torch-spyre/interface-specs/pull/24)).
* **A topk with both outputs**, values and indices from one scan, would save T2's pass over `probs`: two passes per group instead of three. It needs enabling both in the frontend, where `topk` is decomposed into the two ops, and in the backend.

## **Metrics**

* Latency of the bundle per decode step, against the fixed-pass design of #3549 for the same candidates, and against the host path (copy of the logits and host sampling).
* Groups taken per step on real traffic (`max ngroups`): how often the loop stops after one or two groups.

## **Alternatives**

* **Fixed peeling** ([#3549](https://github.com/torch-spyre/torch-spyre/issues/3549)): `⌈K_max/128⌉` groups every step, then top-p. The same ops without a dynamic loop, but always the worst-case cost for the chosen cap.
* **A top-k first, then top-p within it:** a fixed small number of groups, but a different result whenever the top-p set is larger.
* **Host:** copy the logits and sample on the host, as today.

## **Prior Art**

Top-p in PyTorch and Hugging Face sorts the probabilities, takes a cumsum and cuts where it passes `p`. This RFC does the same, with the sort replaced by groups of 128 taken until the cut.

## **How we teach this**

* The op tables give the stick dim and precision of every tensor. The loop marked **(dynamic)** has a bound that its own body writes.
* "Group" means the 128 candidates taken by one `topk`.

## **Unresolved questions**

* The frontend side of the in-place `maskbyindex` (T3).
* The spec sentence for a bound written inside its own loop ([interface-specs#24](https://github.com/torch-spyre/interface-specs/pull/24)), and a hardware check of it.
* Combining with a top-k per row: also drop the candidates past `k[b]`, and take `max_value = ⌈max k/128⌉`.

## Resolution

Pending review.

#### Tracking issue

https://github.com/torch-spyre/torch-spyre/issues/3549
