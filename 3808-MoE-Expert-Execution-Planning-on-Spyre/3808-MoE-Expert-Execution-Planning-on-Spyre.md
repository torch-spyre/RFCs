# RFC: MoE Expert Execution Planning on Spyre

**Author:** Adnan Hoque

Date: 2026-08-17

## 1. Decision

Torch-Spyre will represent routed-expert value execution as a strategy-neutral
semantic operation followed by a typed, immutable planning pipeline.

Each compiler phase owns one decision class:

- the model defines semantics;
- the selector chooses an execution strategy;
- temporal planning chooses program shape;
- the work divider chooses core ownership;
- the scheduler chooses an executable operation order without changing that
  ownership;
- placement chooses memory, lifetime, and legal transport;
- binding resolution turns logical loop requirements into physical operand
  bindings;
- codegen emits the chosen plan; and
- the backend accepts the execution contract atomically.

Later phases may verify earlier decisions. They may not silently replace them.
Failure returns to the selector as structured data. Emission is verified against
the accepted backend contract.

## 2. Scope

This RFC covers the compiler architecture for the routed-expert value path:

- dense all-expert chunked execution;
- dense all-expert persistent execution;
- per-pair indirect execution; and
- future active-dense and grouped execution.

In the Stage-1 operation sequence, this is A9-A15: gate, up, activation,
gated product, down, post-down route weighting, and expert reduction. The
operation consumes normalized routing weights and produces the routed expert
output `[T,H]`.

It does not own A1-A8 (router GEMM, softmax, top-k, masking, or
renormalization), A16-A21 (the shared expert and final routed-plus-shared add),
arbitrary expert subgraphs, a native composite DDL, or a new grouped backend
primitive. This RFC introduces a frontend semantic operation and compiler
contracts; it does not require a new native backend math operation.

The model-facing operation is:

```text
spyre::routed_expert_ffn(
    x,
    expert_weights,
    normalized_routing_weights,
    RoutedExpertFFNSemantics
)
```

The model describes the equation. It never selects an execution strategy.

### Compatibility with Stage 1

This architecture is not a prerequisite for Stage-1 enablement. The existing
dense all-expert chunked path remains the functional fallback while the typed
pipeline is implemented. Integrating the semantic operation may simplify that
path, but Stage-1 correctness does not depend on persistent-state planning,
typed provenance, or backend contract extraction.

## 3. Architectural rules

1. Decisions cross phase boundaries only as typed data.
2. Every artifact is immutable and records all of its inputs by digest.
3. Each decision class has one authoring phase.
4. Division constraints and placement constraints are different types.
5. Loop operand bindings are requirements first and physical bindings only
   after division and placement.
6. Codegen consumes resolved bindings; it does not discover them.
7. The backend accepts or rejects a complete contract, never selected fields.
8. Silent fallback is a compiler bug.
9. A backend emission that violates an accepted contract is a hard compiler
   error, not a reason to try a slower strategy.
10. Graphs without the MoE semantic operation retain existing behavior and pay
    no planning cost for this feature.
11. Planning is pure. A strategy attempt may not mutate the source graph.
12. Materialization is transactional. A failed attempt is discarded in full,
    and the next attempt starts from the original semantic graph.
13. Placement is tied to an exact scheduled order. Any later reordering must be
    verified against the placement facts or rejected.

## 4. Pipeline

```text
RoutedExpertFFNSemantics
  -> ExpertExecutionIntent
  -> TemporalProgramPlan
  -> DivisionPlan
  -> SchedulePlan
  -> PlacementPlan
  -> ResolvedLoopOperandBindings
  -> LoweredExecutionPlan
  -> BackendExecutionContract
  -> BackendAcceptance
  -> TransactionalMaterialization
  -> BackendEmission
  -> EmissionRecord
  -> RuntimeLoadRecord when executed
```

Any planning phase may return `PlanFailure` to the selector. The selector may
start one new attempt with another strategy. Planning artifacts are pure, and
candidate graph materialization occurs on an isolated copy only after backend
contract acceptance. A failed candidate is discarded without committing graph
state. The attempted plan remains inspectable.

## 5. Artifact identity and lineage

Every artifact carries:

```text
ArtifactHeader
  artifact_type
  semantic_fingerprint
  producer_phase
  input_artifact_digests: map[role -> digest]
```

Every strategy-attempt artifact also carries:

```text
PlanAttemptHeader
  ArtifactHeader
  plan_attempt_id
  machine_constant_set_id
```

The semantic fingerprint is the digest of the canonical semantic payload; its
header is excluded. The digest of each later artifact covers its canonical
header and payload and serves as its identity. Symbolic expressions use a
deterministic encoding before hashing.

Nested records are values inside their parent artifact. They use stable local
ids, not independent headers.

Digests provide lineage verification and tamper detection. They do not
authenticate the producing toolchain.

## 6. Semantic operation and strategy selection

### `RoutedExpertFFNSemantics`

The semantic descriptor contains only logical facts for the routed-expert
value path:

```text
expert topology: gated or plain
activation: registered enum
routing: top-k, weighting position, and logical weight representation
logical weight schema
E, T, H, F
input, weight, and reference dtypes
```

Physical packing, LX placement, loop structure, and core division do not belong
in the descriptor.

For the gated form, the semantic equation is:

```text
Y[t,:] = sum_e alpha[t,e] * ((
    activation(X[t,:] @ Wg[e]) * (X[t,:] @ Wu[e])
) @ Wd[e])
```

The routing contract consumes normalized weights produced outside this
operation, defines selected experts, and treats unselected weights as zero. It
does not compute logits, top-k indices, masks, or renormalization. A physical
singleton representation such as `alpha[E,T,1]` is a later ABI choice, not
part of the equation.

Every accepted descriptor has an eager reference decomposition. That
decomposition defines correctness and provides the CPU test oracle. It is not a
production fallback.

Semantic validation also requires a `dense_all_expert_chunked` device lowering
for the descriptor. This proves representability, not that one unpartitioned
E128 program will fit compiler limits.

### `ExpertExecutionIntent`

The selector chooses exactly one strategy per attempt:

```text
dense_all_expert_persistent
dense_all_expert_chunked
per_pair_indirect
reserved: active_dense, grouped
```

The intent carries the strategy, selection reason, machine-constant identity,
residency requirements, and logical constraints. It contains no work division,
allocation, address, or backend schedule.

Before creating an intent, the selector may produce a typed
`StrategyFeasibilityCheck` using semantic shapes and retained machine
constants. It records:

```text
candidate strategy
proven minimum resource requirement
available resource bound
proof inputs and machine-constant identity
feasible, impossible, or unknown
```

The selector may reject a strategy early only when a necessary lower bound
proves it impossible, such as the minimum invariant activation, accumulator,
and live temporary exceeding available LX. An estimate, fragmentation model,
or incomplete ownership model may not reject a strategy. `unknown` proceeds to
normal planning. Exact capacity, aliasing, ownership, and transport feasibility
remain placement decisions.

## 7. Program shape

`TemporalProgramPlan` is a tagged union:

### `NoTemporalBankLoopPlan`

Used when an existing lowering owns program shape, including per-pair indirect
execution. It carries strategy identity and constraints but invents no temporal
loop.

### `StaticPartitionedProgramPlan`

The dense-all-expert-chunked fallback. It represents the all-expert reference
equation as one or more compiler-owned, statically bounded expert ranges. It is
the production-plan counterpart of the existing `Ec=32` path, which reuses one
compiled 32-expert callable across four expert banks. Under this contract,
partitioning becomes a compiler decision; the model never dispatches chunks.
The plan declares the ordered partitions and final accumulation semantics.

This variant makes the fallback honest: dense all-expert execution need not
rely on an uncompilable monolithic E128 graph.

### `StaticPersistentExpertLoopPlan`

The dense all-expert persistent form. It declares:

```text
static bank loop and trip count
loop-invariant operands and preheader copies
streamed operands and binding requirements
loop-local values
loop-carried reductions
preheader, loop body, and single drain
lifetime facts
division constraints
placement constraints
```

### `RuntimeCountedBankLoopPlan`

Reserved for active-dense, per-pair indirect loops with runtime work counts, or
grouped execution. It is not implemented by this RFC and cannot be represented
by weakening the static-loop type.

## 8. Constraints and division

`DivisionConstraint` describes requirements that the work divider owns, for
example:

```text
equal_rows(operation_group, logical_row_axis)
required_core_count(operation_group, count)
compatible_partition(operation_group, logical_axis)
```

`PlacementConstraint` describes memory and transport requirements, for
example:

```text
loop_invariant_lx(buffer)
loop_carried_lx(buffer)
streamed_hbm(operand)
transport_free(operation_group)
no_hbm_intermediate(operation_group)
single_final_drain(buffer)
```

`transport_free` is not a work-division constraint. The divider chooses legal
ownership. Placement may realize transport between fixed divisions unless a
placement constraint forbids it. Placement never re-divides operations.

`DivisionPlan` records:

```text
per-operation work divisions
physical ownership maps
constraint-satisfaction proof
cost terms used
rejected candidates and reasons
```

This artifact is the sole authoritative answer for work division.

## 9. Scheduling

`SchedulePlan` is an immutable artifact produced from `TemporalProgramPlan`
and `DivisionPlan`. It records:

```text
exact operation order
loop nesting and preheader/body/drain boundaries
dependency and mutation ordering proof
atomic operation groups
barriers and synchronization requirements
division-preservation proof
```

Scheduling may choose an executable order among dependency-equivalent
operations. It may not change a division, core ownership map, temporal loop,
binding requirement, or placement constraint. If the selected divisions
cannot be scheduled legally, scheduling returns `PlanFailure` rather than
rewriting them.

The schedule digest identifies the exact order used by lifetime and capacity
analysis. Placement consumes this artifact directly. Any later pass that
changes operation order, loop boundaries, atomic groups, or mutation order
must either reproduce all schedule-dependent proofs or fail the plan. A late
schedule change may never silently retain stale placement facts.

## 10. Placement

`PlacementPlan` consumes the exact `TemporalProgramPlan`, `DivisionPlan`, and
`SchedulePlan`. It owns all memory kinds, lifetimes, aliasing, and legal
transport. Each allocation separates three concepts:

```text
memory_space: LX or HBM
storage_class: graph_input, graph_output, temporary, pool, external
access_policy: resident, streamed, loop_invariant, loop_carried
```

An allocation also carries a logical allocation id, offset, size, alignment,
owner map, and lifetime interval.

The plan contains capacity, alias, residency, transport, and division-
conformance proofs. For every operation it records the authoritative
`DivisionPlan` ownership map, the realized ownership map, and any explicit
legal transport connecting already-decided maps. A legal relayout may connect
those maps. It may not change them. Contract construction later verifies this
proof again; digest lineage alone is not treated as semantic conformance.

If capacity, aliasing, ownership, schedule order, division conformance, or a
transport-free constraint cannot be satisfied, placement returns
`PlanFailure`.

Loop-carried lifetime extension uses the general `lifetime_end_override`
mechanism. It never fabricates a read to influence allocator scoring.

## 11. Loop operand bindings

Temporal planning creates a `LoopOperandBindingRequirement` containing:

```text
local id
operand id
kind: sequential_affine; reserved: indexed
advance cadence
address unit
```

It contains no base, step, or bound because those are not known before physical
planning.

After division and placement, binding resolution produces a
`ResolvedLoopOperandBinding`:

```text
operand id
base logical allocation or graph-argument id
base offset
induction source
trip-count source
step in explicit address units
valid range
on-violation behavior
```

The resolved base is never a generated `%arg_N` position. Codegen maps the
logical id to its final argument only during emission.

Codegen consumes this binding. TensorArg advance analysis verifies that the
emitted address expression equals it. An unknown binding kind or mismatched
advance fails closed; it never becomes a fixed-base operand.

## 12. Codegen, backend acceptance, transactional materialization, and emission

`LoweredExecutionPlan` is the intended program:

```text
ordered operations
resolved bindings
selected core maps
logical allocations and address requirements
required backend capabilities
```

Construction of `LoweredExecutionPlan` is pure. It consumes the exact
`SchedulePlan`, `DivisionPlan`, `PlacementPlan`, and resolved bindings without
mutating the source graph. It verifies that each selected core map equals the
authoritative division or is connected by a legal transport explicitly
recorded in placement.

`BackendExecutionContract` is the narrow backend ABI. It contains loop
bounds, ordered operations, bindings, core maps, memory requirements,
capability requirements, and source identities. It uses logical allocations
plus offsets and alignment; absolute runtime addresses are not part of the
stable ABI.

Contract construction independently verifies:

```text
operation order and loop boundaries against SchedulePlan
core maps against DivisionPlan
legal transport and residency against PlacementPlan
emitted binding requirements against resolved bindings
```

An input digest proves lineage but does not replace these semantic checks.

The backend returns `BackendAcceptance` for the complete contract. It may not
ignore a binding, drop a core map, replace an advancing base with a fixed base,
or weaken residency silently.

After backend acceptance, the compiler materializes the candidate on an
isolated graph transaction. The accepted plan is the only source of inserted
loops, copies, reductions, bindings, and layout operations. Planning artifacts
remain immutable. If materialization or an expected backend capability check
fails, the transaction and all attempt-local compiler state are discarded and
the selector receives `PlanFailure`. A later attempt starts from the original
semantic graph, not the failed candidate.

After transactional materialization and backend emission:

1. the backend supplies the raw bundle and SDSC inventory;
2. the compiler contract verifier compares the actual program with the
   accepted contract;
3. the verifier creates an immutable `EmissionRecord`; and
4. only a verified emission may be committed to the compilation and caches;
   and
5. when the program is loaded, runtime creates a separate immutable
   `RuntimeLoadRecord` that names the emission digest, loaded binary, and device.

`EmissionRecord` contains the contract digest, normalized and raw bundle
hashes, actual bindings and core maps, backend identity, verifier identity, and
verification result.

`RuntimeLoadRecord` contains the emission digest, runtime identity, binary
identity, and device identity. Runtime never mutates the emission artifact.

A mismatch between emission and an accepted contract is a structured compiler
error. The selector must not hide it by choosing a slower strategy.

## 13. Failure, fallback, and provenance

Planning failures use one enum:

```text
UNSUPPORTED_TOPOLOGY
UNSUPPORTED_ACTIVATION
STRATEGY_UNAVAILABLE
COMPILE_LIMIT
LX_CAPACITY
ALIAS_CONFLICT
UNREPRESENTABLE_OWNERSHIP
CONSTRAINT_UNSATISFIABLE
SCHEDULE_UNSATISFIABLE
TRANSPORT_REQUIRED
BINDING_UNRESOLVABLE
BACKEND_CAPABILITY
CONTRACT_REJECTED
MATERIALIZATION_FAILED
```

The selector attempts a strategy at most once for one semantic fingerprint and
machine-constant set. Each retry receives a new attempt id.

The initial ladder is:

```text
dense_all_expert_persistent -> dense_all_expert_chunked
```

`dense_all_expert_chunked` uses `StaticPartitionedProgramPlan` when required.
If it also fails, compilation terminates with the complete attempt record.
Production compilation never silently transfers to the eager CPU oracle.

Every compilation containing `spyre::routed_expert_ffn` emits a minimal
provenance artifact containing the semantic fingerprint, attempted strategies,
feasibility-check outcomes, reason codes, attempt ids, selected plan digest,
accepted contract digest, and bundle digest. This artifact cannot be disabled.
Verbose candidate costs and capacity ledgers are optional.

## 14. General mechanisms and required redesign

The following mechanisms are general compiler capabilities and require
non-MoE tests:

- shared-LHS expert matmul contracts and lowering as internal targets;
- `lifetime_end_override`;
- affine-symbol serialization for squeezed advancing dimensions; and
- selected core-map propagation into SuperDSC.

The following designs are rejected and must be replaced:

- model or adapter calls to strategy-specific
  `activation_stationary_*` ops;
- post-divider ownership alignment in `lx_relayout`;
- the global flat-reduction restructure and same-loop-consumer prohibition;
- private attributes used as inter-phase contracts;
- parallel legacy and dependency-aware read-copy dialects;
- codegen-side binding discovery; and
- silent engagement, decline, or fallback.

The implementation starts from the typed contracts rather than adapting a
strategy-specific compiler path.

## 15. Compatibility gates

1. On the same source SHA, registering this feature while no
   `spyre::routed_expert_ffn` exists must leave the named legacy fixture corpus
   byte-identical.
2. Legacy flat reductions, matched same-loop consumers, nested reductions,
   ordinary read-copy planning, squeezed BMM dimensions, and ordinary
   scratchpad allocation are explicit differential fixtures.
3. One dependency-aware read-copy implementation must reproduce existing
   behavior before the legacy branch is deleted.
4. Plan-only symbolic analysis and metadata allocation must not execute for
   graphs without the semantic operation.
5. Every special admission rule has a positive test and a near-miss negative
   test.
6. Minimal MoE provenance is always emitted.
7. A composition fixture combines a hoisted loop-invariant preheader, an
   atomic relayout group, loop-carried accumulator lifetime, and a
   post-scheduling demotion. It must prove that the complete strategy attempt
   fails atomically: no partial LX allocation, copy, loop, or lifetime change
   reaches the source graph.
8. A retry fixture forces the persistent attempt to fail after candidate
   materialization begins, then proves that the chunked attempt receives a
   pristine semantic graph and produces the same result as a compilation that
   never attempted the persistent strategy.

## 16. Acceptance protocol

Dense all-expert persistent acceptance is ordered and fail-closed.

### Structure

- one static expert loop;
- one activation HBM-to-LX preheader copy;
- activation and accumulator remain LX-resident through the loop;
- gate, up, down, and alpha use correct per-trip HBM bindings;
- alpha is applied after down;
- zero HBM activation or accumulator intermediates;
- zero HBM restickify operations in the activation path; and
- one final output drain.

### Correctness

- compare with the eager semantic decomposition;
- test at least two distinct nonbinary route-weight payloads on the same
  compiled callable; and
- verify the output delta as well as each output independently.

### Performance

- measure the selected path and `dense_all_expert_chunked` fallback over the
  same A9-A15 boundary on the same compiler SHA, native extension, DeepTools
  build, device, payload, and protocol;
- register warmups, sample counts, statistic, exclusions, and tolerance before
  timing;
- use one representative device for the landing gate; broader reproduction is
  a follow-on validation task; and
- publish absolute results and the matched ratio. No historical latency is a
  permanent pass criterion.

The cost model is enabled only with machine constants derived from calibrated
measurements. Reported hardware guidance is not substituted for a measurement.

## 17. Implementation sequence

1. Land schemas, canonical serialization, provenance, and the routed-expert
   semantic operation without enabling a new strategy. Existing Stage-1
   lowering remains unchanged.
2. Map the existing `Ec=32` dense all-expert chunked callable into
   `StaticPartitionedProgramPlan` as the guaranteed on-device fallback for
   accepted semantics.
3. Land the four general compiler mechanisms independently.
4. Implement `StaticPersistentExpertLoopPlan`, plan-gated invariant hoisting,
   streamed operands, and loop-carried reductions.
5. Feed constraints into the real work divider; do not add an alignment pass.
6. Implement `SchedulePlan` and require placement to consume its exact order.
7. Implement placement and division-conformance proofs, then binding
   resolution.
8. Implement backend atomic acceptance, isolated candidate materialization,
   and emission verification.
9. Add conservative selector feasibility checks, the bounded strategy ladder,
   and replacement of adapter-side strategy selection.
10. Run structural, correctness, and matched one-AIU performance gates.

Each step must leave unrelated graphs unchanged and independently testable.

## 18. Extension rule

The current strategy mapping is explicit:

```text
Stage-1 dense prefill / Ec=32 chunked -> StaticPartitionedProgramPlan
persistent dense Path A               -> StaticPersistentExpertLoopPlan
per-pair indirect decode Path B       -> NoTemporalBankLoopPlan today
future active-dense or grouped        -> RuntimeCountedBankLoopPlan + indexed binding
```

Active-dense may use either a fixed masked static loop or the reserved runtime
counted plan plus indexed bindings. The choice is explicit in the strategy
plan.

Grouped execution additionally requires typed routing preparation: expert ids,
group offsets, row extents, permutation, padding policy, and combine semantics.
It may require one backend indexed-binding capability. It must reuse the same
semantic operation, selector, division, placement, binding, failure, and
provenance infrastructure.

Adding either strategy must require no model-specific execution code and no
duplicate loop, placement, or binding implementation.
