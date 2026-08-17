# RFC: MoE Expert Execution Planning on Spyre

**Author:** Adnan Hoque

## Summary

This RFC proposes a compiler-owned planning pipeline for Mixture-of-Experts
(MoE) value execution on Spyre.

Models express one strategy-neutral routed-expert FFN operation. The compiler
then selects an execution strategy and carries that decision through explicit,
immutable plans for temporal program shape, work division, scheduling,
placement, operand binding, code generation, and backend acceptance.

The first optimized strategy is dense all-expert persistent execution: one
static device loop processes all experts while the input activation and output
accumulator remain on chip, expert weights and routing scalars advance through
HBM, and only the final accumulated output is drained.

The architecture is designed so that future active-dense and grouped
strategies reuse the same semantic operation, selector, scheduling, placement,
binding, failure, and verification machinery.

## Motivation

A semantic MoE op sequence does not determine a good physical program.
Expressing gate, up, activation, down, route weighting, and expert reduction as
ordinary tensor operations can lead to:

- static unrolling of one complete FFN chain per expert;
- host-dispatched expert partitions;
- materialized `[T,E,F]` or `[T,E,H]` intermediates in HBM;
- repeated activation and accumulator transfers;
- compiler limits caused by a large generated program; and
- strategy choices embedded in model code.

These are compiler decisions, not model semantics. They span several compiler
phases, and a correct solution requires those phases to agree on program shape,
core ownership, memory lifetime, transport, and changing operand addresses.

The compiler currently has mechanisms that can express parts of this schedule,
but no single contract owns the complete decision. Decisions can consequently
travel through graph patterns, private attributes, or post-hoc rewrites. That
makes the behavior difficult to reason about, extend, and verify.

This proposal makes expert execution an explicit compiler planning problem.

## Goals

- Keep execution strategy out of model and adapter code.
- Give each compiler decision class one owning phase.
- Carry cross-phase decisions as typed, immutable values.
- Make planning pure and strategy materialization transactional.
- Represent schedule order explicitly rather than infer it from placement.
- Resolve loop operand addresses only after division and placement.
- Require placement and codegen to prove conformance with work division.
- Make backend acceptance atomic and verify the emitted program.
- Preserve existing behavior for graphs without the semantic MoE operation.
- Provide one architecture for persistent, active-dense, and grouped execution.

## Non-goals

- Router logits, softmax, top-k selection, masking, or normalization.
- Shared-expert execution and the final routed-plus-shared addition.
- Arbitrary expert subgraphs.
- A new composite native math kernel.
- Runtime-counted or indexed expert execution in the initial implementation.
- Selection policy based on unvalidated performance constants.

## Semantic operation

The model-facing operation describes the routed-expert value equation:

```text
spyre::moe_ffn(
    x,
    gate_weight,
    up_weight,
    down_weight,
    normalized_routing_weight,
    top_k,
    activation
) -> output
```

For a gated expert FFN:

```text
output[t,:] = sum_e routing_weight[t,e] * (
    activation(x[t,:] @ gate_weight[e])
    * (x[t,:] @ up_weight[e])
) @ down_weight[e]
```

The operation consumes normalized routing weights. Unselected experts have
zero weight. Route weighting occurs after the down projection.

The logical contract is independent of physical weight packing, expert
partitioning, loop structure, core division, and memory placement. It has an
eager decomposition that defines correctness and serves as the CPU oracle.

## Compiler pipeline

```text
MoE semantics
  -> ExpertExecutionIntent
  -> TemporalProgramPlan
  -> DivisionPlan
  -> SchedulePlan
  -> PlacementPlan
  -> ResolvedLoopOperandBindings
  -> LoweredExecutionPlan
  -> BackendExecutionContract
  -> BackendAcceptance
  -> EmissionRecord
```

Each arrow is a phase boundary. A phase consumes immutable inputs and produces
a new immutable result. It does not mutate an upstream plan.

### Expert execution intent

The selector chooses one strategy for one attempt. Initial strategies are:

```text
dense_all_expert_persistent
ordinary_dense
```

The intent records the selected strategy, reason, shape, residency intent, and
constraints. It does not contain division, allocation, or address decisions.

The selector uses conservative feasibility checks. A precheck may reject a
strategy only when a lower bound proves it impossible. Passing the precheck is
not proof of feasibility; authoritative division and placement still decide.

### Temporal program plan

`TemporalProgramPlan` is a tagged union.

`NoTemporalBankLoopPlan` delegates program shape to an existing lowering. It is
used by the ordinary dense fallback.

`StaticPersistentExpertLoopPlan` declares:

- a static expert loop and trip count;
- loop-invariant operands;
- streamed operands;
- loop-local values;
- carried reductions;
- logical binding requirements;
- division constraints; and
- placement constraints.

A runtime-counted variant is reserved for future strategies and cannot be
represented by weakening the static plan.

### Division plan

The work divider is the only phase that chooses work division and physical core
ownership. It consumes constraints such as:

```text
equal_rows(operation_group, row_axis)
required_core_count(operation_group, count)
compatible_partition(operation_group, logical_axis)
```

`DivisionPlan` records per-operation divisions, physical ownership maps,
constraint proofs, cost terms, and rejected candidates. Later phases may verify
this result but may not re-divide operations.

### Schedule plan

Scheduling is explicit and occurs before placement. `SchedulePlan` records:

- operation order;
- preheader, counted-loop, and drain regions;
- dependency and mutation ordering proofs;
- atomic operation groups;
- barriers; and
- proof that the selected divisions are preserved.

This prevents the allocator or relayout planner from becoming an implicit
second scheduler.

### Placement plan

Placement owns memory spaces, storage classes, access policies, lifetimes,
aliasing, capacity, and legal transport.

These concepts remain separate:

```text
memory space: LX or HBM
storage class: input, output, temporary, pool, external
access policy: resident, streamed, loop-invariant, loop-carried
```

Placement may realize a legal relayout between already-decided ownership maps.
It never changes a work division. A transport-free constraint may forbid such
a relayout.

`PlacementPlan` includes capacity, alias, residency, transport, and division-
conformance proofs. A change in physical ownership must name the exact legal
transport that realizes it.

Loop-carried lifetime extension is represented directly. It must not fabricate
a synthetic read to influence allocation scoring.

### Loop operand binding

Temporal planning states a logical `LoopOperandBindingRequirement`:

```text
operand
binding kind
advance cadence
address unit
```

It does not guess a physical base, step, or bound.

After division and placement, binding resolution produces:

```text
operand
base logical allocation or graph argument
base offset
induction source
trip-count source
step
valid range
on-violation behavior
```

The initial binding kind is `sequential_affine`. Codegen consumes the resolved
binding. Address-expression analysis verifies the emitted result; it is not the
source of the decision.

An indexed binding is a future explicit capability. It must not be encoded by
overloading the affine form.

### Backend contract and emission verification

`LoweredExecutionPlan` describes the intended operation sequence, bindings,
core maps, allocations, schedule, and required backend capabilities.

The compiler extracts a narrow `BackendExecutionContract`. The backend accepts
or rejects it as one unit. It may not silently:

- ignore an operand binding;
- replace an advancing operand with a fixed address;
- drop a selected core map;
- change loop bounds; or
- weaken a residency requirement.

After emission, `EmissionRecord` captures the actual loop, operations,
bindings, core maps, and bundle identity, together with verification against
the accepted contract. A mismatch is a compiler error. It is not a reason to
silently select a slower strategy.

## Persistent dense expert strategy

The persistent strategy lowers the semantic operation into one logical expert
body enclosed by one static device loop.

```text
preheader:
    copy x from HBM to LX once
    initialize output accumulator in LX once

for expert in [0, E):
    gate = x @ gate_weight[expert]
    up = x @ up_weight[expert]
    hidden = activation(gate) * up
    down = hidden @ down_weight[expert]
    accumulator += down * routing_weight[:, expert, :]

drain:
    copy accumulator to the graph output once
```

Required physical properties are:

- one compiled copy of the expert body;
- one static counted loop;
- one shared activation resident through the loop;
- one accumulator resident through the loop;
- streamed gate, up, and down weights;
- a routing scalar advanced once per expert and applied after down;
- no HBM activation or accumulator intermediate; and
- one final output drain.

The initial strategy does not require indexed access. Experts execute in static
order, so expert weights and routing scalars use sequential affine bindings.
Indexed access becomes relevant only when the execution order is selected by a
runtime table.

The large logical tensors `[T,E,F]` and `[T,E,H]` are never required as physical
allocations. The expert dimension is temporal: loop-local values are allocated
for one expert, and the weighted output is immediately combined into `[T,H]`.

## Ordinary dense fallback

`ordinary_dense` lowers the same semantic equation through existing device
operations. It is selected when persistent execution is unavailable or proven
impossible.

The fallback is compiler-owned. The model does not select or dispatch expert
partitions. If compiler limits require partitioning, that remains an internal
lowering decision with the same post-down weighting and accumulation semantics.

The eager CPU decomposition is a correctness oracle, not a production fallback.
If all device strategies fail, compilation terminates with a structured error.

## Pure planning and transactional materialization

Planning reads the semantic FX graph and produces plans without mutating graph
nodes, metadata, work division, or allocator state.

Strategy materialization operates on an isolated graph clone:

1. digest the source graph;
2. clone it;
3. materialize one selected plan on the clone;
4. verify the candidate and source digest;
5. commit the candidate only after verification; and
6. discard the complete candidate on failure.

A failed strategy leaves no state for the next attempt. This rule applies to
graph rewrites and to compiler-side plan state.

## Failure and provenance

Every plan failure carries the strategy, phase, reason, attempt identity, and
structured payload. Reasons include unsupported semantics, capacity,
ownership, constraints, scheduling, transport, aliasing, binding, backend
capability, and compile limits.

The selector attempts a strategy at most once in one compilation attempt chain.
The initial ladder is:

```text
dense_all_expert_persistent -> ordinary_dense
```

Minimal provenance records selected and failed strategies, reason codes, and
the final plan, contract, and emission identities. Verbose costs and capacity
ledgers are optional. Provenance observes decisions; it does not carry them.

## Compatibility and safety

Graphs without `spyre::moe_ffn` must retain existing compiler behavior and pay
no symbolic-planning cost for this feature.

The compatibility suite includes same-source differential fixtures for:

- flat reductions with and without same-loop consumers;
- nested reductions;
- ordinary dependency-aware read copies;
- BMMs with squeezed dimensions;
- scratchpad allocation without loop-carried lifetimes; and
- generic relayout planning.

Feature-specific admission rules require positive and near-miss negative tests.
The existing flat-reduction path is not globally restricted to accommodate the
persistent strategy.

One integration fixture must compose:

- an invariant activation whose ownership requires legal LX relayout;
- a loop-carried accumulator with lifetime extension;
- the counted expert loop; and
- the final drain.

The fixture verifies schedule preservation, allocation overlap, division
conformance, relayout legality, and absence of HBM intermediates.

## Acceptance criteria

### Structure

- one static expert loop;
- one activation HBM-to-LX preheader copy;
- one LX-resident loop-carried accumulator;
- correct advancing bindings for weights and routing scalars;
- route weighting after down;
- zero HBM activation or accumulator intermediates;
- zero HBM restickification in the activation path; and
- one final output drain.

### Correctness

- compare against the eager semantic decomposition;
- execute two distinct nonbinary routing-weight payloads with one compiled
  callable;
- verify both outputs independently; and
- verify the output delta between payloads.

### Performance

- compare persistent and ordinary dense execution over the same semantic
  boundary;
- use the same compiler, backend, device, tensor payload, and timing protocol;
- register warmups, sample count, statistic, and exclusions before timing; and
- report absolute measurements and the matched ratio.

Historical measurements are not permanent pass thresholds. Strategy selection
uses only machine constants with a reproducible measurement method.

## Implementation sequence

1. Add the semantic operation and eager oracle.
2. Add immutable plans, deterministic identities, structured failures, and
   provenance.
3. Add pure strategy selection and transactional FX materialization.
4. Add shared-LHS projection lowering as an internal target.
5. Add the static persistent temporal plan and explicit schedule plan.
6. Feed ownership constraints into the work divider.
7. Add placement, lifetime, transport, and division-conformance proofs.
8. Resolve affine operand bindings after placement.
9. Add atomic backend acceptance and emission verification.
10. Run compatibility, structure, correctness, and matched performance gates.

Each step must be independently testable and must leave unrelated graphs
unchanged.

## Alternatives

### Static unrolling

Unrolling one FFN chain per expert requires no counted-loop planning, but it
duplicates compiler IR and backend program structure and can exceed compilation
limits. It also does not guarantee on-chip activation placement.

### Model-driven expert chunks

Host-dispatched expert chunks bound compiler size, but expose execution policy
to model code and introduce boundaries across which activation and accumulator
state may be reloaded.

### Post-hoc compiler rewrites

Private attributes and late alignment passes can realize a prototype quickly,
but create multiple owners for work division and placement decisions. They are
not an extensible inter-phase contract.

### Native composite kernel

A native kernel could implement the persistent schedule directly. It remains a
valid optimization if backend sequencing becomes the limiting factor, but it
does not remove the need for strategy selection, semantic correctness,
placement requirements, failure handling, or comparison with other strategies.
This proposal therefore keeps the backend contract narrow enough to permit a
future native implementation without putting policy in the model.

### No dedicated planning architecture

Continuing to add strategy-specific model paths and compiler markers duplicates
the same loop, placement, and binding decisions for every execution form.

## Drawbacks

- The proposal adds compiler data structures and validation code before all
  strategies use them.
- Deterministic plan identity and provenance increase implementation surface.
- Transactional graph cloning adds compilation work for MoE graphs.
- Explicit constraints can initially reject schedules that an implicit rewrite
  might have attempted.
- The persistent path spans frontend, work division, allocator, codegen, and
  backend contracts, so the implementation cannot be reviewed as a local
  matmul change.

These costs buy a single ownership model, inspectable failures, and reuse across
future strategies.

## Extension model

Active-dense adds a selector strategy and an explicit indexed or masked binding
without changing model semantics.

Grouped execution additionally introduces typed routing preparation, including
expert ids, group offsets, row extents, permutation, padding, and combine
semantics. It reuses the semantic operation, selector, division, scheduling,
placement, binding, failure, and verification pipeline.

The architectural invariant is:

> Adding an expert execution strategy must not require model-specific execution
> code or duplicate loop, placement, and operand-binding infrastructure.

## Unresolved questions

- Which physical fields belong in the stable backend contract rather than the
  compiler-local lowered plan?
- Which existing compiler representation should carry immutable plans between
  the relevant passes?
- Where should minimal compilation provenance be retained?
- Which machine constants are sufficiently reproducible for selector policy?
- Should ordinary dense partitioning be represented explicitly in the temporal
  plan or remain internal to its existing lowering?
