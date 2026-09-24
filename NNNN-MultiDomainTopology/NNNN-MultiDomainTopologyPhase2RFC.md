# Multi-Domain Sharded Reduction for Intra-Device Spyre Tensors (Phase 2)

**Authors:** Tuan M. Hoang Trong, Mudhakar Srivatsa

> **Status:** Draft. The RFC number and tracking issue are TBD. This RFC builds
> on [Multi-Domain Topology for Intra-Device Spyre Tensor
> Sharding](NNNN-MultiDomainTopologyRFC.md) (Phase 1).

## **Summary**

This RFC proposes a reduction model for the eight-way intra-device sharded Spyre
tensor introduced in the Phase 1 `MultiDomainTopology` RFC. It adds support for
`sum`, `mean`, and value-only `max` over one ordinary global-shape tensor on
`spyre:0` whose backend-owned storage already holds eight semantic shards in
eight logical HBM groups. The user does not construct ranks, a process group, or
a DTensor.

Reductions divide into two classes:

- a reduction that **excludes** the shard dimension runs independently on every
  shard and returns another ordinary global-shape Spyre tensor with remapped
  semantic sharding; and
- a reduction that **includes** the shard dimension produces one domain-local
  partial per shard, combines the eight partials in deterministic order through
  an explicitly homed, globally reachable HBM rendezvous, and returns one
  ordinary Spyre tensor whose value is complete and whose storage is a single
  designated copy.

The initial cross-shard algorithm is blocking, deterministic, centralized, and
host-orchestrated. Its centralized global-HBM path is the correctness oracle
against which any future direct communication path must be compared.

A central purpose of this RFC is to extend the Phase 1 DeepTools contract to a
two-stage reduction interface. At compile time, Inductor describes two program
stages — local partial production and deterministic combination — through
SuperDSC JSON and SuperDSC-Bundle MLIR, identifying every partial, scratch, and
output tensor start as a symbol and every scratch range as an external symbolic
range. DeepTools must preserve those symbols and ranges through lowering and
emit, for **every** host-compute module (HCM) in both stages, an authoritative
final ordered correction-input manifest. At runtime, torch-spyre validates the
actual tensor arguments and its own scratch and output allocations, resolves
their live per-endpoint addresses in that final order, and supplies them to the
DeepTools-generated correction mechanism before each stage runs.

Concrete addresses, scratch slots, and translation state remain runtime state.
They are not embedded in tensor semantics, compiler guards, or cache keys. This
RFC deliberately does not model the reduction as a distributed collective, and
it defers direct `Partial -> Shard` reduce-scatter to Phase 3.

## **Motivation**

Phase 1 established sharded storage, blocking migration, and shard-local
pointwise execution, but explicitly excluded cross-shard reductions. Reductions
are the next capability real models need, and a reduction over the shard
dimension is fundamentally different from a pointwise operation: the final value
of each output element depends on contributions that live in all eight logical
HBM groups. Combining those contributions requires either a communication
mechanism between localities or an explicit rendezvous through shared memory.

Modeling this as an eight-rank collective all-reduce would expose internal HBM
locality as a user-visible distributed topology, require a process group and
rank-local storage that one process owning all eight shards does not have, and
tie correctness to distributed-runtime behavior. Keeping one ordinary tensor and
one device preserves the Phase 1 user model.

Existing components provide primitives but not the complete two-stage semantic
reduction:

- torch-spyre lowers `sum`, `mean`, and `max` today, but through the current
  dense-core, single-stage `OpSpec` path; it has no notion of a domain-local
  partial, an explicit combiner stage, or receipt-bound scratch;
- Flex can allocate and place memory, but the production path does not yet
  provide the hard-bound, authoritative placement of a shared rendezvous buffer
  and a separate designated output;
- DeepTools has non-unified allocation and symbolic-address mechanisms and
  per-HCM positional correction, but symbolic cross-core output has an inspected
  blocked path, and its final ordering and independent external symbolic-range
  tracking must be validated before they become a two-stage runtime ABI; and
- the checked-in 1p5 UMI DMA and compute paths are incomplete, the public Senlib
  `Dva` API currently fixes `VA[39]` to zero, and there is no verified global
  HBM cache coherence or arbitrary-participant on-device barrier.

The last group of facts is why the initial algorithm is conservative. It uses a
centralized HBM rendezvous, a blocking host-orchestrated visibility boundary
between stages, and no HBM atomics, rather than assuming coherence, a device
barrier, or a native collective that the current stack does not prove. An
explicit, versioned cross-layer contract is required so that shard, partial,
scratch-slot, endpoint, and receipt identities are never inferred from vector
position, group number, or endpoint number.

## Goals and non-goals

### Goals

- Reduce one ordinary global-shape sharded Spyre tensor with `sum`, `mean`, and
  value-only `max` and return an ordinary global-shape Spyre tensor.
- Classify a reduction as shard-local or cross-shard from proven placement, and
  reject any case that cannot be proven one of the two.
- Preserve sharding through a shard-local reduction, remapping the shard
  dimension by the exact PyTorch dimension-removal rule.
- Materialize a cross-shard reduction as one designated single copy through a
  verified HBM rendezvous, with deterministic combine order.
- Keep semantic state (`Complete`/`Partial`) separate from physical placement in
  types, guards, diagnostics, and tests.
- Keep all reduction metadata address-free and resolve physical addresses only
  at launch, per stage, from live receipts.
- Extend the DeepTools contract to authoritative final per-HCM manifests for a
  two-stage producer/combiner artifact.
- Preserve the Phase 1 pointwise, unified, and single-domain paths.
- Fail closed, atomically, and without exposing any partial, stale, or
  seven-participant result.

### Non-goals

The Phase 2 milestone does not provide:

- modified DTensor storage, `DeviceMesh`, c10d, a `ProcessGroup`, fake ranks, or
  any change in `../pytorch`;
- direct `Partial -> Shard` reduce-scatter, which is a hard Phase 3 boundary;
- direct ring, LX, core-to-core, or card/rank RDMA communication transport;
- HBM atomics, global HBM coherence, selectable `VA[39]`, or an
  arbitrary-endpoint device barrier without separately verified capabilities;
- a public `Partial` tensor, a generic collective API, or arbitrary
  user-defined reductions;
- index-producing `max`, `argmax`, or `(values, indices)` output;
- uneven, empty, dynamic, aliased, viewed, mutated, or autograd-enabled sharded
  storage;
- asynchronous producer/consumer overlap, pipelined scratch reuse, or
  cancellation; or
- performance parity with a future direct-communication implementation.

## **Proposed Implementation**

### User-visible contract

The input and output are ordinary `torch.Tensor` objects on `spyre:0`. For an
admitted `sum`, `mean`, or value-only `max` on a Phase 1 sharded base tensor:

```python
x = torch.spyre.to_sharded(x_cpu, placement=Shard(1))   # Phase 1
y = torch.sum(x, dim=(1, 2))                             # Phase 2
y_cpu = y.to("cpu")
```

Shapes, `keepdim`, dtypes, and admitted numerical behavior match the frozen
PyTorch-compatible policy. A full reduction with `keepdim=False` produces a
zero-dimensional Spyre tensor with a real designated allocation and receipt, not
a host scalar. Cross-shard execution is initially blocking; `non_blocking=True`
is rejected or explicitly synchronized and reported as blocking.

The path admits only a Phase 1 sharded base tensor with full storage and offset
zero, exactly eight equal non-empty shards, static global shape and axes, a
supported dtype and layout, and a normalized, in-range, duplicate-free reduction
dimension set. Any other case is a stable `UNSUPPORTED_REDUCTION` error before
allocation or device work. There is no fallback that treats eight local partials
as eight complete outputs, and no silent fallback to unified storage or host
reduction; a safe fallback must be an explicit semantic gather, operation, and
redistribution.

### Independence from distributed PyTorch

Phase 2 has no dependency on a modified DTensor implementation, `DeviceMesh`,
c10d or a `ProcessGroup`, fake ranks, native DTensor storage or collectives,
`distribute_tensor()`, `parallelize_module()`, or any change in `../pytorch`.
Public `Shard`, `Replicate`, and `Partial` concepts may be used as semantic
vocabulary, and selected public PyTorch reduction rules may be differential test
oracles. They are not runtime storage, dispatch, or communication dependencies.

### Architectural ownership

```text
PyTorch reduction request
  global shape, dtype, dims, keepdim, Shard(dim) input
                         |
                         v
                  torch-spyre
  admission, local/cross-shard classification, shard-dim remapping,
  partial layouts and numerical policy, scratch/output intent,
  endpoint lookup, two-stage schedule, SuperDSC/bundle generation
                         |
             +-----------+-----------+
             |                       |
             v                       v
       Flex / Senlib              DeepTools
  hard-bound scratch and         two-stage programs,
  designated-output              symbolic partial/scratch/output
  allocation, receipts,          ranges, final per-HCM manifests
  topology, visibility,                  |
  translation, submission                |
             |                       |
             +-----------+-----------+
                         v
                    UMI runtime
        Stage A/Stage B submission, visibility
        operations, address activation, correction,
        completion aggregation, epoch retirement
```

The ownership boundaries are normative:

- **torch-spyre** owns reduction admission, semantic `Complete`/`Partial` state,
  classification and remapping, explicit semantic participant mappings, scratch
  and output intent, endpoint lookup from the validated locality profile, the
  deterministic two-stage schedule, and cache/guard state that excludes live
  addresses.
- **Flex** realizes hard-bound scratch and designated-output allocation
  requests, preserves the requested logical HBM group across the device
  boundary, and returns immutable receipts that prove physical placement.
- **Senlib and qualified target data** provide topology, access, translation,
  visibility, and barrier facts, and must qualify them by test rather than
  assert them by routing data.
- **DeepTools** owns two-stage program structure, preservation of symbolic
  scratch/partial/output ranges through lowering, and the final per-HCM
  correction-input manifest for every HCM in both stages.
- **The runtime** resolves live scratch, output, and input allocations against
  each stage's final manifest, enforces the stage boundary, and retains
  resources until terminal completion.

`SpyreCode` serialization remains domain-agnostic: it may validate HCM IDs,
shapes, and counts, but must not interpret shard ordinals, choose an HBM group,
resolve a receipt, allocate a translation slot, or apply topology policy.

### Phase 1 prerequisite

Phase 2 starts only after the relevant Phase 1 storage, migration, execution,
correction, and lifetime contracts pass, exported as an immutable, versioned
capability record accepted by all component owners. Acceptance is transactional:
a missing field or capability fails with `PHASE1_CAPABILITY_MISSING` before any
allocation or device work, and a profile-identity or fingerprint change after
planning forces rejection or full replanning. There is no weaker fallback.

### Terminology and invariants

Semantic value state and physical storage are orthogonal, and the distinction
must appear in types, guards, diagnostics, and tests. A **partial** is an
internal semantic value whose contributions are not yet a complete global
result. A **rendezvous** is explicitly allocated scratch where participant-local
partials become visible to the combiner. An **epoch** is one operation
generation used to reject stale slots and completion. A **receipt** is an
immutable allocation identity resolved to a live address only at runtime.

The implementation must preserve these invariants:

1. The input and output are ordinary tensors on `spyre:0`; the internal
   `SpyrePartial` value never escapes into eager dispatch or becomes
   user-visible, and has no rank, process group, or collective handle.
2. Semantic `Complete`/`Partial` state is separate from physical placement.
   `Materialized` is not a placement kind; materialization is an atomic
   `Partial -> Complete` transition that either completes or exposes no result.
3. A cross-shard reduction produces exactly eight partials, one per semantic
   participant, each with one and only one owner.
4. Every semantic-placement, shard, replica, partial, scratch-slot, receipt,
   HBM-group, and endpoint identity is explicit and looked up, even when numeric
   values coincide. Container vector position is never identity.
5. Scratch slots are aligned, pairwise non-overlapping, single-writer, and
   epoch-tagged; the combiner reads each slot exactly once in deterministic
   order.
6. Scratch and designated output are separate allocations; scratch/output
   aliasing and in-place partial overwrite are prohibited.
7. Generated execution uses explicit profile-provided endpoint IDs, never
   `range(32)` and never `core_id == chunk_index`.
8. Concrete addresses and transient translation slots do not enter compiler
   cache keys unless generated code embeds a slot.
9. The runtime resolves and validates every live binding, per stage, before
   correction or launch.
10. Any failed participant fails the complete logical operation; there is no
    partial, chunk-zero, or seven-partial success.
11. Stage B cannot begin before every Stage A producer and the required
    visibility operation reach terminal success for the current epoch.
12. Submitted work retains storage, receipts, scratch, output, translation
    state, programs, manifests, and completion objects until terminal
    completion.

### Reduction classification and remapping

The request is normalized into an immutable form (operation, global shape,
dtype, normalized unique sorted dimensions, `keepdim`, output dtype,
determinism, input placement). For input placement `Complete(Shard(s))` and
normalized reduction dimensions `R`:

```text
if s not in R:  classification = SHARD_LOCAL_REDUCTION
else:           classification = CROSS_SHARD_REDUCTION
```

A shard-local reduction preserves sharding but may move the shard index:

```text
s_out = s if keepdim else s - count(r in R where r < s)
```

Every shard's origin and extent are transformed by the same mapping, and each
output shard's local layout and encoded bytes are derived independently — never
by copying a global layout or dividing a global byte count by eight. The output
shard is allocated in the corresponding input shard's logical HBM group unless a
separately approved policy says otherwise.

For a cross-shard reduction, every producer reduces all requested dimensions
locally, so the partial shape is already the final global output shape; only the
values remain partial across the eight participants. The combiner performs an
elementwise reduction across the eight partials.

### Internal `SpyrePartial`

A cross-shard planner creates an address-free internal `SpyrePartial` value that
records the operation, global and partial shapes, reduced dimensions, `keepdim`,
the frozen numerical policy identity, a partial-layout signature, and the
explicit participant/shard/replica/partial ownership mappings. A layout
signature describes encoding only; it does not identify an HBM group, scratch
slot, receipt, endpoint, or live address. Equal partial encodings may share one
layout class while occupying eight distinct, non-overlapping ranges. The full
schema is frozen in the cross-repository fixture referenced by the
implementation guide; it is not reproduced here.

### Numerical policies

The supported dtype matrix is frozen before implementation, and an unsupported
combination fails before allocation. The accumulation dtype is never silently
changed.

- **`sum`** produces one local sum per participant in the declared accumulation
  dtype, visits partials in fixed `partial_ordinal` order `0..7`, writes the
  final result once, and casts only as the frozen result-dtype matrix requires.
  No HBM atomic add is used.
- **`mean`** is represented as a sum plus one exact global count. Each
  participant writes an accumulation-dtype local sum, the combiner sums all
  local sums in deterministic order, and one final division by the exact global
  reduction count is applied once — never per participant, which would change
  rounding and be wrong for future uneven sharding.
- **Value-only `max`** writes one local maximum per participant and applies an
  ordered elementwise max over partials `0..7`, under a frozen PyTorch-compatible
  NaN and signed-zero policy established by fixtures before the operation is
  enabled. `argmax`, indices, and `(values, indices)` output are deferred.

Deterministic mode fixes the participant and partial order, producer endpoint
assignment and work partition, the local reduction tree, the global combine
order, the accumulation and result dtypes, and prohibits atomics,
timing-dependent consumption, and producer/consumer overlap. The baseline order
is `partial_ordinal = [0, 1, 2, 3, 4, 5, 6, 7]`.

### Scratch rendezvous and designated output

The planner selects one scratch-home logical HBM group and one designated-output
logical HBM group. They may be the same group but are separate allocations, each
independently validated. Selection considers, in order, a validated
locality-profile mapping, verified read/write behavior for every selected
endpoint, a supported requester/XLAT/access-mode and fixed-zero `VA[39]` policy,
usable capacity after reservations and qualified sparing, and a deterministic
configured home-group policy. Phase 2 v1 runs no route or global cost optimizer;
an unqualified endpoint or runtime configuration is a correctness failure, not
an expensive candidate.

Scratch holds eight aligned, pairwise-disjoint slots, one per partial, with one
declared writer endpoint each, constructed deterministically and validated with
checked integer arithmetic before allocation. The initial plan is one physically
verified `Bind` allocation in the selected logical HBM group — not a Flex
`Interleave`. The designated output begins invalid and becomes
`Complete(SingleCopy(...))` only after the combiner, output visibility, and
aggregate completion succeed. Allocation is one transaction: prevalidate the
plan and all range arithmetic, allocate and verify scratch, allocate and verify
output, freeze both receipts, and expose the executable operation only after
every step succeeds; on failure, release every acquired allocation and return no
output and no partially executable plan.

### Endpoint access, visibility, and barriers

Phase 2 consumes the exact immutable, address-free topology snapshot defined by
Phase 1; it does not define a reduction-only topology vocabulary. For each
source, scratch, or output allocation, torch-spyre looks up the receipt's actual
`logical_hbm_group_id` and obtains exactly four qualified compute endpoints. The
profile is not proof of requester permissions, XLAT behavior, or memory
visibility; those remain separately versioned hardware capabilities and tests.
Every producer and combiner endpoint choice is serialized in the plan; the
runtime does not select or remap endpoints.

The minimum producer-to-consumer ordering is:

```text
producer compute terminal success
  -> producer HBM write terminal success
  -> producer cache flush/fence, if required
  -> all-producer host stage boundary
  -> consumer cache invalidate/discard, if required
  -> consumer HBM reads
```

The design must not equate XLAT activation with data visibility, command
submission with completion, route existence with permission, a control barrier
with an HBM memory fence, or successful 1p0 emulation with native coherence. The
visibility protocol is an owner-approved, versioned capability ID; there is no
implicit `none` policy. If no tested protocol exists, cross-shard execution is
disabled while shard-local reductions remain eligible.

The initial stage barrier is host-orchestrated: successful completion of all
Stage A producers plus the required visibility protocol is the barrier. This
avoids relying on an arbitrary-participant device barrier, whose current
inspected implementation assumes a consecutive worker set. An on-device barrier
may replace the host boundary only after its participants, semantics, namespace,
generation, count, timeout, cache behavior, and cancellation interaction are
specified and tested. Every reduction gets a unique epoch that every slot,
publication state, barrier generation, HCM binding, and completion object
references; an epoch mismatch is `SCRATCH_EPOCH_STALE` and is never repaired by
reading the current bytes.

### Two-stage execution protocol

The initial native algorithm is deterministic, blocking, centralized, and
host-orchestrated, expressed as two artifacts or two explicit HCM stage groups.
After full pre-enqueue validation of the entire logical operation:

- **Stage A (local partial production):** for each of the eight replicas,
  resolve the source and scratch bindings, validate receipts, epoch, ranges, and
  permissions, compute the local reduction on the declared four-endpoint work
  division, and write one finalized partial exactly once to the assigned slot.
- **Host visibility boundary:** wait for every producer to reach terminal state,
  suppress Stage B on any failure, execute the approved producer fence and
  consumer invalidation, and publish all eight slots for this epoch only after
  those steps succeed.
- **Stage B (deterministic combination):** resolve the eight slot addresses and
  the designated output address, read all partials in order `0..7`, apply the
  operation's combine rule, write the output exactly once, execute the output
  visibility operation, and atomically mark the output valid and `Complete`.

A shard-local reduction uses a single-stage artifact of eight independent
replicas with no scratch or cross-domain stage. The complete blocking launch
sequence is specified in the implementation guide.

### Compile-time and runtime DeepTools contract

The reduction program is correct only if its compile-time description and its
runtime tensor bindings identify the same region for every participating
endpoint, in every stage. This extends the Phase 1 two-part interface to two
stages and to receipt-backed scratch.

At compile time, Inductor emits, for each stage, the exact physical endpoint IDs
and local ordinals, `numCoresUsed_`/`coreIdsUsed_` and work-slice mappings
consistent with those identities, per-endpoint data stages and allocation
coordinates, `nonUnifiedAllocInHBM_` where independent addresses are required,
symbolic starts in `startAddressCoreCorelet_`, the partial/scratch/workspace/
output byte ranges, explicit stage dependencies, reduction dimensions and scale
semantics, local and final layouts, the deterministic combine order, and opaque
semantic binding identities. Scratch reads and writes are external symbolic
ranges, not slices of one pool; the frontend must not post-codegen clone a
four-endpoint template for the correctness oracle, and must not use the
card/rank/RDMA collective library as the intra-device implementation.

Stages use distinct versioned ABI identifiers:

```text
spyre.reduction.local_sharded.v1
spyre.reduction.hbm_partials.v1
spyre.reduction.hbm_combine.v1
```

Each symbolic binding carries an address-free `ReductionBindingIdentity` naming
its ABI, HCM stage, binding role (`input`, `partial_write`, `partial_read`,
`workspace`, `output`), the applicable shard/replica/partial identity, the
expected logical HBM group and receipt-identity class, the compute endpoint and
local ordinal, the byte offset and valid length, the access mode, the local
layout signature, the profile fingerprint, the translation and `VA[39]`
policies, and the epoch class. Inapplicable identity fields are absent, not
filled with ambiguous sentinels.

DeepTools may derive, expand, and reorder symbols while compiling. It must
therefore emit the **authoritative** final per-HCM manifest only after those
transformations are complete. For each HCM, `parameter_index` values are unique
and contiguous from zero, `parameter_count == len(parameters)`, every final
symbol has exactly one binding identity, duplicate or missing semantic
identities are rejected, and the HCM's `vdci.inputSym_` order equals manifest
parameter order. The frontend-generated `address_binding_schema.json` is **not**
authoritative for Phase 2. The positional invariant, per HCM, is:

```text
DeepTools final per-HCM manifest order
  == HCM inputSym_ order
  == SpyreCode ishape[0] element order
  == runtime symbolic_inputs order
```

At runtime, for each HCM parameter, torch-spyre selects the declared binding
role, resolves the relevant live input, scratch, workspace, or output receipt,
validates allocation and receipt liveness, semantic slot/shard/replica/partial
and endpoint mappings, physical logical-HBM placement, layout signature, access
mode, alignment, valid range, topology, and epoch, reserves a compatible
translation mapping, adds the declared byte offset with checked arithmetic,
produces the DVA for the declared requester and fixed-zero policy, and stores it
at the exact `parameter_index`. It then requires, per HCM:

```text
symbolic_inputs.size() == hcm_manifest.parameter_count
symbolic_inputs.size() == ComputeOnHostCommand.ishape[0]
```

No live DVA enters the compiled artifact, and reallocation at different
addresses reuses compatible code.

#### Required DeepTools support

This RFC specifically asks DeepTools to define and support the following
behavior, beyond the Phase 1 single-stage asks:

1. preserve symbolic scratch, partial, workspace, and output offsets through
   every backend IR, including the currently blocked symbolic cross-core output
   path, so no live address is baked into code;
2. compute exact HBM transfer sizing from the explicit shape, layout, and range
   records rather than from a first-core assumption or a temporary sizing path;
3. track independent external symbolic ranges with overlap safety, and reject
   legacy one-pool treatment for receipt-backed external ranges;
4. preserve every endpoint, range, role, stage, and opaque binding identity
   through final ordering;
5. emit a versioned authoritative final per-HCM correction-input manifest for
   **every** HCM in both Stage A and Stage B, with the exact input order and
   count expected by the correction mechanism; and
6. define the runtime handoff by which torch-spyre supplies each HCM's ordered
   scalar address vector and receives a clear failure for a count, type, symbol,
   stage, artifact-version, or HCM-identity mismatch.

Inductor supplies the static two-stage program and binding identities;
torch-spyre validates the actual tensors and its own scratch and output
allocations and resolves live addresses; DeepTools supplies the artifact
metadata and program-correction mechanism. This division lets one compiled
artifact run correctly with new allocations without teaching DeepTools about
PyTorch storage or Flex receipts.

### Failure atomicity and lifetime

A reduction operation uses an explicit monotonic state machine
(`PLANNED -> ALLOCATED -> PREVALIDATED -> STAGE_A_ACTIVE -> STAGE_A_TERMINAL ->
VISIBILITY_COMPLETE -> STAGE_B_ACTIVE -> STAGE_B_TERMINAL -> OUTPUT_VISIBLE ->
COMPLETE`), and any state before `COMPLETE` may transition to `FAILED`.
`COMPLETE` is the only state that exposes a valid output.

Every wait category has an owner-configured finite deadline recorded in the
plan, and timeout behavior is fail-closed: record the timed-out stage and epoch,
stop dependent work, request cancellation only if a qualified primitive exists,
settle every submitted operation to terminal status, keep all resources alive
until settling completes, leave the output invalid, poison the scratch epoch,
and return a stable stage-specific error. A retry is always a new operation with
a new epoch; it never resumes a timed-out state. One failed participant fails
the complete logical operation — there is no partial success, unified fallback,
chunk-zero fallback, reorder retry, or seven-partial result. Phase 2 does not
promise cancellation; if it is later exposed, it must settle every submitted
message and retain resources to terminal status rather than deallocate
immediately.

### Single-domain software oracle

A single-domain (1p0) target may emulate the full semantic decomposition —
eight distinct semantic-placement slots, eight partial owners, eight scratch
slots, and one deterministic combiner — while every physical backend domain is
zero. Semantic identities must remain distinct; rewriting every slot to zero
would not exercise planning, ownership, binding, ordering, or failure handling.
Each shard, scratch, and output allocation stays a separately owned single-chunk
domain-0 buffer, and no multi-chunk address reaches the PF or VF scheduler.

The emulator proves reduction classification and remapping, semantic placement
mappings, `SpyrePartial` construction, non-overlapping slot planning, the sum,
mean-count, and value-only max policies, deterministic ordering, final
manifest/binding invariants, epochs and stale-slot rejection, designated
single-copy semantics, and aggregate failure and output-validity behavior. It
cannot prove physical eight-group placement, cross-group routing or permissions,
native DVA/XLAT behavior, producer-to-combiner visibility, a native fence or
barrier, coherence, or 1p5 UMI readiness. Every 1p0 trace is tagged
`is_emulated = true` and `native_cross_group_behavior_exercised = false`, and
passing the emulator never closes a native accessibility or visibility gate.

### Rollout and future phases

All Phase 2 paths are fail-closed and disabled by default until their dependency
packages and capability gates pass. A disabled or rejected cross-shard path does
not silently fall back to unified storage, host reduction, a distributed
collective, or incomplete local outputs. Rollout uses independently controllable
levels, each with a separately auditable backend gate that records its decision,
capability-record version, profile identity, operation, dtype, axes, and layout:

1. planning and schema validation only, with no allocation or device work;
2. the 1p0 semantic oracle, independent of native access gates;
3. native shard-local reductions, independent of any cross-group rendezvous;
4. the native centralized-HBM cross-shard `sum`, after the access, visibility,
   and correction gates pass on an allowlisted hardware/firmware combination;
5. native cross-shard `mean` and value-only `max`, after the numerical policies
   are frozen; and
6. an optional hierarchical HBM schedule, enabled only where it is measurably
   faster and proven equivalent to the centralized oracle.

Each native level roll-out follows owner-only hardware tests with mandatory
address-free traces, CI qualification on nominal and supported spared
topologies, opt-in canary with the centralized path forced for comparison, and
default enablement only for the tested matrix. Each native level has a kill
switch that prevents new operations but lets submitted work reach terminal
status before releasing resources. This RFC establishes the reduction
correctness foundation; follow-on Phase 3 work may add communication-aware
Inductor propagation, ring/LX/direct transfers, compact replication, overlap,
replicated outputs, resharding, and — in particular — direct `Partial -> Shard`
reduce-scatter. Every such path must preserve and test against the centralized
oracle's global semantics, accumulation dtype, deterministic order, visibility,
failure, and lifetime behavior, and the centralized implementation remains
runnable as the oracle unless a later explicit product decision removes it after
equivalent coverage exists.

## **Metrics**

Phase 2 is successful when:

- shard-local `sum`, `mean`, and value-only `max` match CPU for every supported
  non-shard axis, preserve remapped sharding, and allocate no global scratch;
- cross-shard `sum` matches CPU through one centralized designated-HBM copy,
  using topology-selected endpoints and a verified visibility protocol;
- cross-shard `mean` sums first and divides once by the exact global count, and
  value-only `max` matches the frozen NaN and signed-zero policy;
- Stage B provably cannot begin before all producers and the visibility
  operation succeed;
- every input, partial, scratch, workspace, and output address is resolved from
  a live receipt in each HCM's final manifest order, and reallocation reuses the
  artifact without recompilation;
- injected failures at every stage never expose a partial, stale, or
  seven-participant result, never leak or hang, and always require a new epoch to
  retry;
- no distributed runtime is initialized and no collective is invoked; and
- Phase 1 pointwise, unified, and 1p0 compatibility suites do not regress.

Performance measurements — Stage A/Stage B latency, rendezvous bandwidth, and
combine cost — establish the centralized-oracle baseline that Phase 3
communication paths must beat while matching its semantics. They inform later
optimization but do not replace the correctness gates.

## **Drawbacks**

### Centralized rendezvous is not performance-optimal

Routing every partial through one homed HBM buffer and a host-orchestrated
barrier is deliberately conservative. It trades bandwidth and latency for a
correctness oracle that does not depend on unproven coherence, atomics, or
device barriers. Faster direct communication is explicitly deferred to Phase 3.

### Many dependent hardware gates

Native cross-shard execution depends on unverified capabilities — cross-group
read/write permission, a producer-to-consumer visibility protocol, fixed-zero
`VA[39]` for every requester, and complete 1p5 UMI DMA and compute. Each is a
gate, and the native path stays disabled until owner evidence closes it.

### Cross-stack complexity

Two-stage ABIs, receipt-bound scratch, epochs, and authoritative final manifests
add versioned boundaries across torch-spyre, Flex, Senlib, UMI, and DeepTools.
Implicit conventions would be unsafe when allocation order, topology, or symbol
ordering changes, but the explicit contracts increase implementation work.

### Restricted operator coverage

The milestone covers three reductions on one narrow tensor class. This is
intentional: it establishes the two-stage rendezvous, visibility, and
correction foundation before general placement propagation, communication, or
asynchronous lifetimes.

## **Alternatives**

### Use a native distributed reduce-scatter or all-reduce

A collective models rank-local tensors and a process group that one process
owning all eight shards does not have. It would expose internal HBM locality as
a distributed topology and tie correctness to distributed-runtime behavior.

### Combine partials with HBM atomic add

Atomics are not verified on the target and would make the result order-dependent
and non-deterministic, defeating the deterministic-oracle goal.

### Reduce on the host after gathering all shards

Gathering to host and reducing there is only the explicit safe-fallback
semantics, not the native path; using it silently would defeat the purpose of
intra-device reduction and hide the missing native capability.

### Return a sharded result by relabeling partials

Relabeling eight partials as a sharded `Complete` result, copying placement
metadata onto an unrelated allocation, or invoking a native collective is
exactly `Partial -> Shard` reduce-scatter. It is a hard Phase 3 boundary and must
not be smuggled into Phase 2.

### Divide each local partial before combining for `mean`

Per-participant division changes rounding and is incorrect for future uneven
sharding. `mean` must sum first and divide once by the exact global count.

### Treat scratch as a public placement or `Materialized` as a placement kind

Scratch is not a public tensor placement, and materialization is a semantic
transition, not a placement. Conflating them would leak internal state into the
tensor model.

### Direct ring, LX, or core-to-core reduction now

These transports are not verified in the current stack and belong to Phase 3.
Phase 2 uses HBM rendezvous as the auditable oracle first.

### Do nothing

The backend would remain unable to reduce a sharded tensor without gathering to
one domain first, blocking model coverage and leaving no oracle for future
intra-device communication.

## **Prior Art**

PyTorch DTensor contributes the `Shard`, `Replicate`, and `Partial` vocabulary
and the reduction-remapping semantics reused here as differential oracles, while
this RFC keeps one-process, intra-device composite storage distinct from
rank-local distributed storage and from collectives. SPMD compilers and
reduction-tree runtimes provide analogous partial/combine and deterministic-order
concepts; this design differs because all localities remain within one PyTorch
device and one process, combined through explicit shared HBM rather than a
network collective. This RFC builds directly on the Phase 1
[Multi-Domain Topology RFC](NNNN-MultiDomainTopologyRFC.md) and preserves the
[Tensors with Device-Specific Layouts RFC](../0047-TiledTensors/0047-TiledTensorsRFC.md)
rule that device encoding stays below ordinary PyTorch tensor semantics: a
reduction returns an ordinary global-shape tensor, not a distributed object.

## **How we teach this**

Documentation should begin with one statement:

```text
PyTorch sees one tensor and one reduction; Spyre computes one partial per
HBM shard and combines them into one designated complete copy.
```

Distinguish a **shard-local reduction** (stays sharded, shard index remapped)
from a **cross-shard reduction** (eight partials combined through an HBM
rendezvous into one designated single copy). Use **partial**, **rendezvous**,
**designated output**, and **epoch**; do not call partials ranks, do not call the
rendezvous a collective, and do not describe the designated output as
`Replicate`. Developer documentation should explain the layers independently:

```text
Semantic state       Complete vs Partial, and the target distribution
Classification       shard-local vs cross-shard, and the remap formula
Partial and scratch  per-participant local value and its rendezvous slot
Two-stage schedule   producers, the visibility boundary, and the combiner
Binding manifest     which live address supplies each per-HCM parameter
```

Detailed schemas, identity tables, source change maps, work packages, test
matrices, and debugging procedures belong in the Phase 2 implementation guide
rather than in this RFC.

## **Unresolved questions**

This section lists open decisions that can change a public API, cross-component
ABI, correctness rule, or ownership boundary. Qualification experiments that
test the proposed design are listed separately as validation gates; later
optimizations are listed under rollout and future phases. Every unanswered
question below is a gate: the owning team records `verified`, `unsupported`, or
`deferred` with evidence, capability or ABI version, tested hardware/firmware
matrix, and a revalidation trigger. Only `verified` closes a gate; a question is
never converted into an optimistic default.

### DeepTools interface decisions

The following require agreement with the DeepTools owners before the two-stage
compile-time and runtime interface can be frozen:

1. Can DeepTools preserve symbolic scratch, partial, workspace, and output
   offsets through every backend stage, including the currently blocked symbolic
   cross-core output path?
2. Which source field is the authoritative mapping from a SuperDSC `core_id` to
   the 32 topology-provided compute endpoints?
3. Can the local partial layout and transfer-size rules replace every temporary
   cross-core reduction sizing path without a hidden first-core assumption?
4. What versioned authoritative final per-HCM manifest will DeepTools emit for
   each HCM in **both** stages, including final input position, original symbol,
   opaque binding key, required scalar representation, and expected input count?
5. Which symbol transformations may DeepTools perform, and what invariants ensure
   that non-unified placement, independent external symbolic ranges, sparse
   endpoint IDs, stage identity, and binding keys survive them?

### Other design decisions

1. Which runtime component owns live XLAT slot reservation, eviction, and
   conflict isolation across both stages and concurrent operations?
2. Does each endpoint require its own corrected DVA, or can a group share one
   interpretation under the supported requester and fixed-zero policy?
3. Is an extra output visibility operation required before D2H or downstream
   compute consumes the designated output?
4. What is the terminal failure contract after only a subset of producers has
   completed, and what cancellation or quarantine behavior applies to
   already-submitted work?
5. Are topology or HMI remappings launch-patchable, or are they code-affecting
   and recompilation-requiring?

### Implementation validation gates

The following do not block architectural review, but the feature remains
disabled on a target until evidence answers them:

1. Which exact producer and combiner
   `(logical_hbm_group_id, access_port_id, compute_endpoint_id)` records have
   production read and write permission?
2. What exact producer fence, cache flush, consumer invalidation, and completion
   sequence establishes producer-to-combiner visibility, and which completion
   result proves all producer HBM writes are terminal?
3. Is fixed `VA[39] = 0` valid for every Phase 2 compute and DMA requester?
4. Will Phase 2 use only host-orchestrated sequencing, or is a mask/list
   arbitrary-endpoint barrier available and generation-safe?
5. Can Flex and Senlib physically hard-bind both scratch and designated output to
   every proposed home group and return authoritative placement evidence?
6. Which accumulation and result dtypes work end to end for `sum` and `mean`,
   and what exact NaN and signed-zero semantics does value-only `max` implement?
7. What are the maximum HCM input, correction-site, packet, and artifact sizes?
8. Which target-profile fields distinguish code-affecting 1p0 and 1p5 lowering
   from runtime-only validation, so that a 1p0 artifact never reuses a 1p5 cache
   entry?

Compiler-visible placement and communication propagation, compact address
binding, hierarchical HBM scheduling, asynchronous execution, uneven shards,
general views, and direct intra-device communication remain non-blocking
follow-on work under **Rollout and future phases**.

## Resolution

TBD.

### Level of Support

TBD.

#### Additional Context

This RFC chooses an explicit, centralized HBM rendezvous as the initial
correctness oracle. Compact code generation, hierarchical scheduling, and
communication optimization are accepted only after they are shown equivalent to
the explicit partial, rendezvous, and combine path. The centralized
implementation remains runnable as the Phase 3 correctness oracle.

### Next Steps

1. Assign an RFC number and open a tracking issue.
2. Agree with DeepTools owners on the two-stage reduction ABIs, symbolic
   scratch/partial/output preservation, the authoritative final per-HCM manifest
   for both stages, and the `SpyreCode` runtime handoff.
3. Resolve the remaining native access, visibility, barrier, and `VA[39]` gates
   with Flex and Senlib owners.
4. Freeze the versioned reduction semantics, numerical policies, `SpyrePartial`,
   scratch/output, receipt, epoch, and final-binding contracts.
5. Validate the 1p0 semantic oracle and native shard-local reductions, which do
   not depend on the cross-group rendezvous gate.
6. Implement transactional scratch/output allocation, the host-orchestrated
   visibility boundary, and the blocking two-stage execution state machine.
7. Validate the native centralized-HBM cross-shard `sum`, then `mean` and
   value-only `max`, against CPU with shard-unique patterns, no atomics, and no
   native collective.
8. Export the Phase 3 oracle capability record and keep centralized HBM
   execution as the Phase 3 baseline.

#### Tracking issue

TBD.

#### Exceptions

Direct `Partial -> Shard` reduce-scatter, direct ring/LX/core-to-core or
card/rank RDMA communication, HBM atomics, global coherence, selectable
`VA[39]`, arbitrary-endpoint barriers, asynchronous overlap and pipelined scratch
reuse, cancellation, index-producing `max`, uneven or dynamic shards, and any
change to `../pytorch` or distributed runtime remain outside the Phase 2
milestone. They require separate design and qualification.
