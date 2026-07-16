# FP32 Element Arrangement

**Authors:** @joyalbin (owner), @moriohara, @lupalby, @pradghos, @msrivats

> **Status:** draft. RFC number is the FP32 support epic
> ([#2971](https://github.com/torch-spyre/torch-spyre/issues/2971)).

## Summary

Widening a tensor's elements on device (16-bit `DL16`/`BF16` ↔ 32-bit `FP32`)
does **not** reshuffle them into standard stick order — the wider elements come
out **staggered** (all values correct, but within-stick position no longer matches
logical order; "anywhere valid"). The inductor backend tracks this as an
**Element Arrangement (EA)** per layout and gates op legality on it
(`is_ea_compatible`, `validate_ops`); this RFC specifies those rules, today only
in code (epic [#2971](https://github.com/torch-spyre/torch-spyre/issues/2971)).

What makes it tractable: FP32 is **ephemeral** — never stored, only a transient
scoped to one precision-sensitive op, entered by an upcast and left by a downcast
that un-staggers for free. "Completing FP32" is therefore bounded: turn on the
up/downcast brackets for
`layernorm`/`softmax` (they strip them today), enable other roles (RMSNorm), and
add a bracket-closure check.

## What staggering is

Spyre packs tensors into **sticks** — 128-byte units (64 elements at 16-bit, 32
at 32-bit). Widening can't keep elements in place *and* in order: restoring
standard order would redistribute them across sticks (expensive), so the
conversion instead leaves them out of order within the stick — *staggered*:

```
Logical tensor:  e0 e1 e2 e3 e4 e5 e6 e7      (what the model "means")

16-bit, STANDARD  (64 elems / 128B stick; shown as 8):
  stick: [ e0 e1 e2 e3 e4 e5 e6 e7 ]

        │  upcast 16-bit → FP32  (elements double in width → span 2 sticks)
        ▼

STANDARD FP32 would be  (needs a reshuffle — expensive):
  stick A: [ e0 e1 e2 e3 ]
  stick B: [ e4 e5 e6 e7 ]

STAGGERED FP32 = what the hardware actually emits  (no reshuffle):
  stick A: [ e0 e2 e4 e6 ]     <- even slots
  stick B: [ e1 e3 e5 e7 ]     <- odd slots
```

Every value is present; only within-stick position is scrambled, and the
permutation is illustrative — nothing may depend on it, only on its being
*consistent* per EA value. The legality rule follows: **an op is safe on
staggered inputs iff it never consults within-stick position.**

```
unary point-wise    exp([e0 e2 e4 e6]) = [exp e0, exp e2, ...]    OK  position never consulted

binary, both staggered THE SAME:
  [e0 e2 e4 e6] + [f0 f2 f4 f6] -> e0+f0, e2+f2, ...              OK  permutation cancels

binary, staggered + STANDARD (no broadcast):        added slot-by-slot
  [e0 e2 e4 e6] + [f0 f1 f2 f3] -> e0+f0, e2+f1, e4+f2, ...      BAD  logical indices don't line up

full-dim reduction over stick:
  sum(e0 e2 e4 e6) + sum(e1 e3 e5 e7) = e0+e1+...+e7              OK  order irrelevant
```

So the **safe set** is unary point-wise, binary with identically-staggered
operands (or one a stick-dim broadcast), and full-dim stick reductions. Everything
else needs rearrangement, which the compiler rejects.

## FP32 is ephemeral

FP32 is never stored on Spyre — it is a transient scoped to a **single**
precision-sensitive op, bracketed by an upcast in and a downcast out:

```
16-bit  --upcast-->  FP32 (staggered)  --op-->  FP32 (staggered)  --downcast-->  16-bit
```

So FP32 lives only in LX scratchpad and compute; every persisted tensor stays
16-bit. Two consequences close the design:

* **The exit downcast is the free un-stagger.** `DL16_TO_FP32 → STANDARD` hands
  the consumer a standard 16-bit tensor — so **no rearrangement primitive is ever
  needed**.
* **The safe set *is* the role set.** A precision-sensitive op decomposes into
  exactly what's legal on staggered tensors. Softmax → `max, sub, exp, sum,
  realdiv`; RMSNorm → `mean(x²), rsqrt, mul`; layernorm adds its `EXX2` partial
  reduction.

Only the symmetric bracket is in scope; FP32-native flows, persisted upcasts, and
standalone downcasts would put FP32 in storage and let a staggered tensor persist,
so they are excluded.

## EA values

EA is an `ElementArrangement` enum on each `SpyreTensorLayout`:

| EA | Meaning | Produced by |
|---|---|---|
| `STANDARD` | sequential stick order | no/same-size conversion, or a restoring width conversion |
| `DL16_TO_FP32` | staggered FP32 | widening `STANDARD` 16-bit → `FP32` |
| `FP32_TO_DL16` | staggered 16-bit | narrowing `STANDARD` `FP32` → 16-bit |
| `EXX2` | reduction mode, two values/stick | layernorm partial reduction |
| `QFP8CH` | FP8 quant output — **out of scope** | FP8 quantization |

`DL16_TO_FP32`/`FP32_TO_DL16` form `STAGGERED_EAS` — conversions that must
preserve the input device layout.

## Assignment and propagation

**At a conversion**, a width change *creates* a staggered EA from `STANDARD` or
*restores* `STANDARD` from the opposite staggered tag; any other input EA is
`Unsupported`:

| Width conversion | creates | restores |
|---|---|---|
| widen 16-bit → `FP32` | `STANDARD` → `DL16_TO_FP32` | `FP32_TO_DL16` → `STANDARD` |
| narrow `FP32` → 16-bit | `STANDARD` → `FP32_TO_DL16` | `DL16_TO_FP32` → `STANDARD` |

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
| 3 | Two+ distinct non-STANDARD EAs | ❌ can't pair different permutations |
| 4 | `EXX2` as the non-STANDARD EA | ❌ reduction mode, not an ordering |

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

An op not in the list receiving a staggered FP32 input is a compile-time
`Unsupported`, not a silent downcast.

## Completeness: what's missing

Because FP32 is ephemeral, completeness is one question: are the brackets closed
and enforced? Four gaps:

1. **Bracket-closure check (missing).** `validate_ops` is per-op; nothing verifies
   *global* closure — that every upcast is matched by a downcast on all paths and
   no staggered FP32 reaches a graph boundary unclosed. That is the invariant:
   > A value an op cannot legally consume must raise a compile-time `Unsupported`,
   > never a silent downcast; and no staggered FP32 may reach a graph boundary
   > unclosed.
2. **Flagship brackets.** `layernorm`/`softmax` still strip the up/downcasts and
   run 16-bit; removing the strip is the primary "turn on FP32" work.
3. **RMSNorm.** Not enabled — open question whether it works via allowlisted
   primitives or needs a fused lowering like layernorm's `EXX2`.
4. **Stick-offset conversions.** A width change re-lays-out sticks, so the convert
   re-accounts for padding. It handles trailing padding but bails (`return []`)
   when the tensor doesn't start on a stick boundary — an acceptable constraint,
   but the bail should be a hard-fail, not a silent drop.

**Debug aid.** With no device-side un-stagger, inspection is host-side: copy the
staggered FP32 to host verbatim and reverse the permutation there for golden
comparison. It hard-codes the hardware permutation — debug-only, generation-aware.

## Related Issues

Under epic [#2971](https://github.com/torch-spyre/torch-spyre/issues/2971) (FP32
support): [#2843](https://github.com/torch-spyre/torch-spyre/issues/2843)
bf16→fp32 tagging, [#2788](https://github.com/torch-spyre/torch-spyre/issues/2788)
runtime EA reporting, [#3223](https://github.com/torch-spyre/torch-spyre/issues/3223)
predicate unit tests.
