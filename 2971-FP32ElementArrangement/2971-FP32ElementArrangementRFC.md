# FP32 Element Arrangement

**Authors:** @joyalbin (owner), @moriohara, @lupalby, @pradghos, @msrivats, @avery-blanchard, @manid2

> **Status:** draft. RFC number is the FP32 support epic
> ([#2971](https://github.com/torch-spyre/torch-spyre/issues/2971)).

## Summary

Widening a tensor's elements on device (16-bit `DL16`/`BF16` ↔ 32-bit `FP32`)
does **not** reshuffle them into standard stick order — the wider elements come
out **staggered**: all values are correct, but their position within the stick no
longer matches logical order. The inductor backend tracks this as an **Element
Arrangement (EA)** per layout and gates op legality on it (`is_ea_compatible`,
`validate_ops`); this RFC specifies those rules, today only in code (epic
[#2971](https://github.com/torch-spyre/torch-spyre/issues/2971)).

There are two ways to deal with a staggered tensor.

The cheap one is to keep FP32 **ephemeral**: a transient scoped to one
precision-sensitive op, entered by an upcast and left by a downcast. The downcast
puts the elements back in standard order at no cost, so nothing has to be
rearranged. Every op inside the bracket is one that does not care about position.

The general one is to **rearrange explicitly**. The backend can now move a tensor
from one element arrangement to another, so a staggered tensor can be put back in
standard order, or a standard one can be made staggered, without a width
conversion. This costs bandwidth, so it is not the first choice, but it is
available where the bracket is not.

The ephemeral bracket is therefore the fast path rather than the only path. What
"completing FP32" needs: turn the up/downcast brackets on for `layernorm` and
`softmax` (they strip them today), enable other roles such as RMSNorm, and decide
per case whether an unbracketed staggered value is rearranged or refused.

## What staggering is

Spyre packs tensors into **sticks** — 128-byte units, holding 64 elements at
16-bit or 32 at 32-bit. Widening cannot keep elements both in place and in order:
putting them back in order would redistribute them across sticks, so the
conversion leaves them out of order within the stick instead.

The elements move in groups of four. Take one 16-bit stick as 16 such groups,
`g0` to `g15`. Widening doubles their width, so they no longer fit in one stick
and spill into two:

```
16-bit, STANDARD  (64 elements = 16 groups of 4, in one stick):
  stick:   [ g0 g1 g2 g3 g4 g5 g6 g7 g8 g9 g10 g11 g12 g13 g14 g15 ]

        │  upcast 16-bit → FP32
        ▼

STAGGERED FP32  (32 elements = 8 groups per stick):
  stick A: [ g0 g2 g4 g6 g8 g10 g12 g14 ]     <- even groups
  stick B: [ g1 g3 g5 g7 g9 g11 g13 g15 ]     <- odd groups
```

Every value is present. Only the position within the stick changes, and no op may
depend on what that position is — only on its being the same for every tensor
carrying the same EA. That gives the legality rule: **an op is safe on staggered
inputs if and only if it never consults position within the stick.**

```
unary point-wise    exp([g0 g2 g4 ...]) = [exp g0, exp g2, ...]    OK  position never consulted

binary, both staggered THE SAME:
  [g0 g2 g4 ...] + [h0 h2 h4 ...] -> g0+h0, g2+h2, ...             OK  permutation cancels

binary, staggered + STANDARD (no broadcast):        added slot-by-slot
  [g0 g2 g4 ...] + [h0 h1 h2 ...] -> g0+h0, g2+h1, ...            BAD  logical indices don't line up

full-dim reduction over stick:
  sum(g0 g2 g4 ...) + sum(g1 g3 g5 ...) = sum of all               OK  order irrelevant
```

So the **safe set** is unary point-wise, binary with identically-staggered
operands (or one a stick-dim broadcast), and full-dim stick reductions. Anything
else needs the operands put in a common arrangement first (see
[Rearranging on device](#rearranging-on-device)).

## The ephemeral bracket

The cheapest way to use FP32 is not to let a staggered tensor outlive the op that
needs it:

```
16-bit  --upcast-->  FP32 (staggered)  --op-->  FP32 (staggered)  --downcast-->  16-bit
```

FP32 then lives only in temporary intermediate tensors, and every stored tensor
stays 16-bit. Two things follow:

* **The downcast is a free un-stagger.** `DL16_TO_FP32 → STANDARD` hands the
  consumer a standard 16-bit tensor as a side effect of narrowing it. A bracket
  that ends this way never pays to rearrange, which is why it is the fast path.
* **The safe set is the role set.** A precision-sensitive op decomposes into
  exactly the ops that are legal on staggered tensors. Softmax → `max, sub, exp,
  sum, realdiv`; RMSNorm → `mean(x²), rsqrt, mul`; layernorm adds its `EXX2`
  partial reduction.

Flows that keep FP32 for longer (FP32 in storage, an upcast whose result is
persisted, a downcast on its own) used to be ruled out partly because a staggered
tensor would have had no way back to standard order. That is no longer so: such a
flow can rearrange. What is left to decide about them is whether FP32 belongs in
storage at all, and that depends on what the user asked for, or on the precision
of the operations before and after.

## Rearranging on device

A data-movement op (`identity` and its `shuffle` alias) may have a different
element arrangement on its input and its output. The backend implements the
difference. It works out the largest piece of a stick that both arrangements hold
the same way, moves the tensor one piece at a time, and writes each piece where
the output arrangement wants it. Nothing else about the op changes.

So an arrangement can be applied or undone on its own. A staggered tensor can be
put back in standard order without narrowing it, and a standard one can be
staggered to match another operand.

Two limits. The pieces have to hold whole elements and be a size the load and
store units can move; where only the order of whole sticks differs, whole sticks
move. A permutation at a granularity the hardware does not support is refused. So
is an arrangement that would put one element in two places.

The staggered arrangements produced by fp16-fp32 upcasts and downcasts move
elements in groups of four, so they are within these limits in both directions.

The cost is bandwidth. Moving four elements at a time takes eight accesses where a
standard tensor takes one, so a rearranged 32-bit tensor costs eight times the
traffic, and a 16-bit one more. Avoiding a rearrangement is always faster. The
point is that it is now a choice rather than a wall.

## EA values

EA is an `ElementArrangement` enum on each `SpyreTensorLayout`:

| EA | Meaning | Produced by |
|---|---|---|
| `STANDARD` | sequential stick order | no/same-size conversion, a restoring width conversion, or a rearrangement |
| `DL16_TO_FP32` | staggered FP32 | widening `STANDARD` 16-bit → `FP32`, or a rearrangement |
| `FP32_TO_DL16` | staggered 16-bit | narrowing `STANDARD` `FP32` → 16-bit, or a rearrangement |
| `EXX2` | reduction mode, two values/stick | layernorm partial reduction |
| `QFP8CH` | FP8 quant output — **out of scope** | FP8 activation quantization |
| `QFP8WT` | FP8 quant output — **out of scope** | FP8 weight quantization |

`DL16_TO_FP32`/`FP32_TO_DL16` form `STAGGERED_EAS` — conversions that must
preserve the input device layout.

## Assignment and propagation

**At a conversion**, a width change *creates* a staggered EA from `STANDARD`, or
*restores* `STANDARD` from the opposite staggered tag; any other input EA is
`Unsupported`. A rearrangement is not a conversion and does not touch element
width, but it sets EA the same way, and it can go straight from one staggered EA
to the other:

| Operation | creates | restores |
|---|---|---|
| widen 16-bit → `FP32` | `STANDARD` → `DL16_TO_FP32` | `FP32_TO_DL16` → `STANDARD` |
| narrow `FP32` → 16-bit | `STANDARD` → `FP32_TO_DL16` | `DL16_TO_FP32` → `STANDARD` |
| rearrange, width unchanged | `STANDARD` → either staggered EA | either staggered EA → `STANDARD` |

**Through ops**, EA propagates forward: unary point-wise **preserves**, a full-dim
stick reduction **clears** to `STANDARD`, and a multi-arg op's output follows the
predicate below.

> Staggering is a byte-width property, identical for DL16 or BF16 — so
> `DL16_TO_FP32` tags `BF16 → FP32` too
> ([#2843](https://github.com/torch-spyre/torch-spyre/issues/2843) historically
> got this wrong). Runtime EA reporting is
> [#2788](https://github.com/torch-spyre/torch-spyre/issues/2788).

## Compatibility predicate: `is_ea_compatible`

Can these operand EAs coexist on one multi-arg point-wise op?

```python
def is_ea_compatible(eas):
    unique = set(eas)
    if len(unique) <= 1:            # all operands share one EA (incl. all-STANDARD)
        return True
    non_standard = unique - {ElementArrangement.STANDARD}
    return len(non_standard) == 1 and ElementArrangement.EXX2 not in non_standard
```

| Case | Operand EAs | Verdict |
|---|---|---|
| 1 | All identical | ✅ permutation absent or cancels |
| 2 | One non-STANDARD EA (≠ `EXX2`) + `STANDARD` | ✅ broadcast pattern |
| 3 | Two+ distinct non-STANDARD EAs | ❌ can't pair different permutations directly |
| 4 | `EXX2` as the non-STANDARD EA | ❌ reduction mode, not an ordering |

The predicate says which EAs can be used together as they are. Case 3, and the BAD
case above, can still be compiled by rearranging one operand to match the other,
which turns them into case 1. Whether to spend the traffic or raise `Unsupported`
is a cost decision.

## Enforcement: `validate_ops`

`validate_ops` runs after propagation and raises `Unsupported` when a multi-input
point-wise op's operand EAs fail the predicate. `layernormnorm`/`layernormscale`
carrying `EXX2` are skipped.

The predicate is only half the check — it governs EA-*set* membership. That a
case-2 `STANDARD` operand actually broadcasts at the stick dim (size 1) is
enforced in `_multi_arg_pointwise_layouts`, where concrete layouts exist; that is
where the BAD case above is rejected.

## FP32 allowlist (`SPYRE_FP32_OPS`)

A separate list: which ops may run in FP32 at all. Already well beyond the
original softmax/layernorm set:

```
add, sub, mul, where, realdiv, relufwd, reciprocal, mean, sum, max, min,
layernormscale, abs, neg, exp, sigmoid, exx2, layernormnorm, identity, sqrt,
rsqrt, topkvalue, topkindex, floor, to_dtype, maximum, minimum, prod
```

An op not in the list receiving a FP32 input is a compile-time
`Unsupported`, not a silent downcast.

## Completeness: what's missing

> **This section may be out of date and needs a review.** Some of the gaps below
> may have been closed since it was written.

The invariant to hold on to is:

> A value an op cannot legally consume must raise a compile-time `Unsupported`,
> never a silent downcast; and no staggered value may reach a graph boundary.

Gaps:

1. **Arrangement check at graph boundaries (missing).** `validate_ops` is per-op;
   nothing checks globally that no staggered value escapes. With rearrangement
   available, a path can be closed either by the downcast it already has or by an
   inserted rearrangement, so the check has to decide which, but the invariant
   it enforces is unchanged.
2. **Flagship brackets.** `layernorm`/`softmax` still strip the up/downcasts and
   run 16-bit; removing the strip is the primary "turn on FP32" work.
3. **RMSNorm.** Not enabled — open question whether it works via allowlisted
   primitives or needs a fused lowering like layernorm's `EXX2`.
4. **Stick-offset conversions.** A width change re-lays-out sticks, so the convert
   re-accounts for padding. It handles trailing padding but bails (`return []`)
   when the tensor doesn't start on a stick boundary — an acceptable constraint,
   but the bail should be a hard-fail, not a silent drop.
5. **Standalone eager conversion.** Eager `.to(fp32)` up/downcasts do not check
   that no staggered value escapes, and are not guaranteed to produce standard
   FP32. They can be made to: a rearrangement after the upcast gives standard
   FP32. What is missing is the decision to spend the traffic, not the means.

**Debug aid.** A staggered tensor can be inspected by rearranging it to standard
on device and copying that. The older host-side route (copy the staggered tensor
verbatim and undo the permutation on the host) still works and needs nothing from
the device, but it hard-codes the hardware permutation and has to track hardware
generations, where the device route reads the arrangement off the layout.

## Related Issues

Under epic [#2971](https://github.com/torch-spyre/torch-spyre/issues/2971) (FP32
support): [#2843](https://github.com/torch-spyre/torch-spyre/issues/2843)
bf16→fp32 tagging, [#2788](https://github.com/torch-spyre/torch-spyre/issues/2788)
runtime EA reporting, [#3223](https://github.com/torch-spyre/torch-spyre/issues/3223)
predicate unit tests.
