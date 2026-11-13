# Hierarchical DTensor execution over local and distributed mesh coordinates

**Authors:**

* Joshua Rosenkranz (@JRosenkranz)
* Tuan Hoang Trong (@thoangtrvn)

**Status:** Draft for discussion in torch-spyre/RFCs, 2026-11-13.

## **Summary**

Separate DTensor's logical mesh and placement semantics from process-rank
ownership, and support hierarchical tensor distribution: inter-device tensor
parallelism combined with intra-device sharding across memory domains. An
execution context can own several mesh coordinates in one process, with
execution and redistribution selected per mesh axis. This would preserve
DTensor's global metadata, `Shard`, `Replicate`, `Partial`, and operator
propagation across both levels. Constructing a logical mesh or executing a
wholly local placement must not require a process group, fake ranks, or
additional operating-system processes. Distributed axes can use existing
communication backends, and existing process-group-backed DTensor execution
remains the default.

Spyre needs both levels at the same time: tensor parallelism across accelerators
and sharding across each accelerator's eight logical HBM groups. One process may
control a device and own its eight local coordinates while participating in
distributed execution along another axis. Compute endpoints remain backend
resources, separate from logical placement coordinates. A wholly local executor
is the first implementation milestone; combined local and distributed execution
is a requirement of the proposed architecture.

## **Motivation**

Tensor placement and process topology describe different things. `Shard(0)`
specifies a partition of a tensor's logical dimension; it need not imply a
separate process for each partition. A single accelerator may expose several
memory localities, and a single process may own the data and execution resources
for every locality.

The ordinary DTensor interface associates `DeviceMesh` entries with process
ranks, computes the current rank's mesh coordinate, and carries a rank-local
tensor alongside the global specification. That model works well for
process-based distributed execution. It does not provide a public way for one
process to own all coordinates of an eight-way mesh within one PyTorch device.

An out-of-tree backend consequently has to implement a separate global-tensor
sharding system or adapt private simulation infrastructure. Maintaining another
set of placement and operator rules duplicates work and makes interoperability
harder. DTensor already supplies useful semantics for pointwise operations,
shard-local reductions, partial results, and redistribution.

Spyre's proposed native profile illustrates the distinction: eight semantic
shards map to eight HBM groups, while 32 compute endpoints perform the work.
Several endpoints may process one shard. Treating each endpoint as a mesh
coordinate, or each memory group as an independent PyTorch device, would conflate
storage placement with execution parallelism. Other backends that manage several
local buffers or memory domains could benefit from the same separation.

The goal is to make logical DTensor placement useful without requiring a
physical process for every mesh coordinate, and to let planning and compilation
describe that placement before device execution is initialized.

For Spyre, local sharding must compose with inter-device tensor parallelism on
the same tensor. Supporting only one mode at a time would leave the motivating
use case incomplete. A shared logical mesh should describe both levels while
allowing each axis to use the appropriate local or distributed executor.

## **Proposed Implementation**

### Logical mesh and execution binding

Introduce a logical mesh description containing its shape, dimension names, and
coordinate namespace. It describes placement cardinality and coordinates rather
than process ranks, live device addresses, or compute endpoint IDs. Mesh and
placement metadata must be usable for shape propagation and compilation without
initializing c10d or an accelerator runtime.

Bind that description to a separate execution context when a tensor is
materialized. The context supplies coordinate ownership, backend capabilities,
axis-specific execution bindings, and the operations needed to realize a
placement. The initial local context owns all coordinates in one process and
uses one PyTorch device. It has no distributed transport dependency. A
hierarchical context additionally binds distributed axes to their communication
executor and local axes to the device backend. The resulting ownership map can
assign several coordinates to each participating process.

Existing `DeviceMesh` execution can adapt to the shared logical interface while
preserving its public rank-based behavior. This RFC does not propose changing the
meaning of the rank array in an existing `DeviceMesh` or permitting duplicate
ranks to stand for additional coordinates. Combining local and distributed axes
is a required capability of the design, even though implementation begins with
the simpler wholly local case.

The public names and registration API below are illustrative and require
maintainer agreement. The design requirement is the separation between logical
placement and execution, rather than these specific class names.

### Wholly local example

```python
import torch
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

# Proposed interface: LogicalMesh, LocalExecutionContext, and the execution
# keyword are not available public APIs today.
mesh = LogicalMesh(shape=(8,), mesh_dim_names=("hbm",))
execution = LocalExecutionContext(mesh, backend="spyre", device="spyre:0")

source = torch.randn(1024, 64, dtype=torch.float32)
x = distribute_tensor(source, mesh, [Shard(0)], execution=execution)
y = x + x                                      # Shard(0)
row_sums = y.sum(dim=1)                         # Shard(0)
column_partials = y.sum(dim=0)                  # Partial(sum)
column_sums = column_partials.redistribute(placements=[Replicate()])
host_result = column_sums.full_tensor().cpu()
```

One process owns all eight coordinates. The backend partitions and encodes the
input, executes the operations, and combines the partials within the device.
`full_tensor()` materializes an ordinary complete tensor; the explicit `.cpu()`
performs the host transfer. This preserves the distinction between gathering a
value and changing its device.

### Hierarchical distribution

Use **inter-device tensor parallelism** for partitioning across accelerators,
**intra-device sharding** for partitioning within each accelerator's memory
domains, and **hierarchical tensor distribution** for their composition. The
word "local" alone is insufficient to distinguish a host, process, or device
boundary.

For example, four Spyre devices with eight HBM groups each form a logical mesh
of shape `(4, 8)`:

| Mesh axis | Meaning | Execution mechanism |
| --- | --- | --- |
| `tp` | Partition across Spyre devices | Inter-device communication and collectives. |
| `hbm` | Partition within each device's HBM groups | Local allocation, computation, and redistribution. |

```python
# Conceptual definition; hierarchical execution binding is proposed.
mesh = LogicalMesh(shape=(4, 8), mesh_dim_names=("tp", "hbm"))
placements = [Shard(1), Shard(0)]
```

With one process per device, process `t` owns coordinates `(t, 0)` through
`(t, 7)`. There are four processes and 32 logical coordinates. Process count is
an execution choice rather than the logical mesh's cardinality.

For a global tensor of shape `[1024, 256]`, `Shard(1)` on `tp` partitions columns
across devices, and `Shard(0)` on `hbm` partitions rows within each device. Each
device owns a `[1024, 64]` partition; each HBM coordinate owns a `[128, 64]`
component. The backend maps those coordinates to memory groups and compute
endpoints without exposing either as process ranks.

Operations propagate placements across both axes. With reductions using
`keepdim=False`:

| Operation | Output placements on (`tp`, `hbm`) | Required combine |
| --- | --- | --- |
| `x + x` | `[Shard(1), Shard(0)]` | None. |
| `x.sum(dim=0)` | `[Shard(0), Partial(sum)]` | Combine the eight local HBM contributions per device to obtain `[Shard(0), Replicate()]`. |
| `x.sum(dim=1)` | `[Partial(sum), Shard(0)]` | Combine contributions across devices for each HBM coordinate to obtain `[Replicate(), Shard(0)]`. |

The first reduction renumbers the surviving tensor dimension, so the `tp`
placement becomes `Shard(0)`. Reducing all elements produces
`[Partial(sum), Partial(sum)]`; a valid plan can combine locally along `hbm`
and then across devices along `tp` to obtain `[Replicate(), Replicate()]`.
Each stage combines the semantic participants of its own axis exactly once.
After the HBM combine, the TP stage consumes one combined contribution per
device; copies replicated across HBM coordinates are not additional
contributions. The selected reduction order must respect the numerical policy,
since floating-point reassociation can change rounding.

### Axis-specific execution and collective matching

An axis operation acts on coordinates that differ along that axis while the
other coordinates remain fixed. In the example, an `hbm` operation acts on the
eight coordinates owned by one device; a `tp` operation matches coordinates
across devices for a fixed HBM coordinate.

The execution plan must identify these participant sets and give corresponding
communication operations a consistent identity and order across processes.
Participants must also agree on component order, payload layout, and the
reduction operation.
Multiple local components may share a process group or be batched into a
transport operation, provided their logical ranges and participant identities
remain correct. Merely reusing a group for several buffers does not establish
collective matching. Local HBM coordinates do not require extra process ranks
or independent process groups.

These requirements also apply after compiler transformations. Lowering, fusion
and transport batching must preserve the logical participant sets, matching
identity, component order and producer-to-consumer dependencies of the selected
execution plan.

Composition also requires explicit dependencies between local and distributed
stages. A transport operation must not read unfinished local results, and a
local consumer must not read an incomplete received result. The execution
context applies the completion, visibility, and failure contract across the
whole hierarchical operation.

### Coordinate ownership and local storage

Each DTensor retains global shape, stride, dtype, mesh, and placements. Its local
representation must also describe every locally owned coordinate and the
corresponding logical range and payload. The representation may use a
Tensor-compatible aggregate of local tensors or a backend aggregate storage
handle. Backend storage is not required to be a Python list of separately
dispatched tensors.

Coordinate identity is explicit and stable. A backing-buffer position, physical
domain ID, or core number cannot substitute for it. Materialized payloads must
cover the ranges prescribed by the placement, preserve local logical shape, and
remain associated with their owning execution context.

Existing `to_local()` behavior stays unchanged for existing rank-based DTensors.
For contexts owning several local coordinates, access to one component must
name its coordinate.
An accessor such as `local_shards()` could expose coordinate-indexed components;
an unqualified `to_local()` must reject an ambiguous multi-coordinate request.
The accessor's name and return type are open for review. Inspection must make
clear which returned handles alias the DTensor and which operations are valid on
them; it must not silently expose an aggregate as an ordinary contiguous tensor.

### Execution and redistribution interface

DTensor continues to select eligible input and output placements and derive
global metadata. The execution context realizes the resulting local computation
and placement transitions. Its contract includes:

| Responsibility | Required behavior |
| --- | --- |
| Ownership | Report the locally owned coordinates and establish a complete coordinate-to-owner mapping without missing or duplicate coordinate ownership. |
| Materialization | Create payloads for the specified logical ranges, preserving dtype and local logical geometry. |
| Local computation | Execute a propagated operation over the required components, or lower the operation jointly across them. |
| Redistribution | Implement each requested placement transition over the declared axis participants, using its local or distributed executor. |
| Gathering | Materialize a complete ordinary tensor without implicitly changing its device. |
| Completion | Retain payloads through terminal completion and publish a result only after all required work succeeds. |
| Visibility and failure | Make producer results visible before dependent reads; fail the whole logical operation if a required participant fails. |

Allowing joint lowering is essential for accelerator backends. The interface
must not require a Python loop that invokes every operator separately for each
coordinate. A backend may compile the global operation and its placement into a
single execution plan with explicit local work slices.

For compiled execution, this requires visibility of the relevant locally owned
component computations and their redistribution dependencies within one
compilation scope. The operator-propagation and compilation section defines the
proposed representation and the point at which backend lowering selects its
realization.

Blocking completion does not by itself establish producer-to-consumer memory
visibility. The executor must enforce the ordering and visibility needed by its
local operations and redistributions. On failure, it must not expose a partial
result as complete, and it must retain resources for already submitted work
until that work reaches a terminal state.

Placement transitions retain their existing mathematical meaning. A local
`Shard -> Replicate` gathers the prescribed shards. `Partial(sum) -> Replicate`
combines one contribution from each logical coordinate. A backend must reject a
transition it cannot implement; registering a local executor does not imply
support for every redistribution operation.

`Replicate` requires each coordinate to observe the same complete value. A backend
may implement this with separate copies or a supported shared backing, but shared
backing requires explicit access and mutation/aliasing semantics. A complete value
allocated at one memory locality is not automatically a replica at every
coordinate.

### Operator propagation and compilation

Ordinary compiled DTensor execution evaluates placement propagation and dispatch
during tracing, exposing local tensor computation and required redistribution
operations to the compiler. DTensor metadata supports reconstruction and
compatibility checks; the compiler need not optimize the Python DTensor wrapper
itself. This proposal extends the meaning of locally owned computation from one
coordinate to every coordinate owned by the process.

The proposed initial compiler representation contains ordinary tensor operations
on the locally owned components, together with explicit semantic redistribution
operations. A supported compiled region contains all relevant local components
in one compilation scope. For a wholly local eight-coordinate mesh, that scope
includes computation over all eight components. A hierarchical process
contributes the components it owns and the communication operations connecting
them to other processes; compilation need not encompass every process.

Conceptually, the compilation path is:

```text
DTensor program and logical placements
  -> placement propagation and executor-capability checks
  -> traceable local components and axis-specific redistributions
  -> captured functional graph, through AOT tracing where applicable
  -> Inductor/backend lowering and scheduling
  -> runtime bindings, execution and completion
```

Each component has its own local logical shape and stride. Capture retains the
association between component values, coordinates, logical ranges and placements,
together with the information needed to reconstruct the resulting DTensor.
Backend layout and locality constraints that affect code generation must survive
optimization. A backend may retain aggregate runtime storage, but the compiler
must have equivalent visibility of its components and relationships. This
representation does not require a Python launch per coordinate or independent
physical allocations for every graph value.

A traceable multi-component payload is one candidate implementation. Its abstract
shape behavior, flattening and reconstruction must integrate with Dynamo and AOT
tracing. Any new redistribution operation needs an abstract implementation and a
supported decomposition or backend lowering. Calling an opaque Python executor
during tracing, or retaining the only copy of placement information on a wrapper
that tracing removes, is insufficient. The precise representation and
registration points require agreement with DTensor and compiler maintainers.

Redistributions retain their semantic axis, participant coordinates, source and
destination placements, and reduction operation until an appropriate lowering
selects their realization. Local HBM redistribution uses the local executor. A
distributed-axis operation uses its communication executor with consistent
matching and component order. Existing c10d operations do not automatically
acquire local-HBM semantics when supplied with a different mesh.

Lowerings must describe buffer reads and writes, aliases, communication effects
and completion dependencies accurately. Compiler scheduling and memory reuse
must preserve those dependencies. Distributed collectives cannot be removed or
reordered independently across processes in a way that breaks matching. Local
transformations may be optimized as ordinary tensor operations when their value,
placement, alias and visibility requirements remain satisfied. Shared backing for
replicas requires the declared access and mutation semantics.

Spyre lowering may jointly optimize component computation and local
redistribution and emit one or several device programs. Multiple component
operations in the captured graph do not prescribe multiple kernel launches.
Local operation boundaries may be decomposed or fused once the selected lowering
preserves their semantics. Coarse opaque backend regions remain an alternative,
with the explicit tradeoff that Inductor cannot optimize computation hidden
inside them.

Reuse the existing placement strategies and tensor metadata machinery where
applicable. Refactor assumptions that derive one local shape from the current
rank so they instead use explicit coordinates and execution ownership. Factory
and view arguments may have different local shapes at different coordinates;
support must be explicit rather than inferred from one representative component.

Strategy selection must filter candidates through the executor's supported
operations and placement transitions before considering their costs. The
mathematical strategy definitions remain shared; backend capability and cost
policies determine which realizations are feasible. If none is supported, reject
the operation during planning or compilation before attempting its allocation,
redistribution, or kernel submission.

Normal DTensor dispatch includes a C++ path. Both that path and Python fallback
handling must select the appropriate execution context and preserve the same
placement semantics. A Python dispatcher patch alone is not a complete design.

Abstract evaluation and compatibility checking must work without initializing
c10d or an accelerator runtime. Global and local geometry, placements,
ownership, and code-affecting backend execution or layout assumptions participate
in compilation compatibility. This may include local encoding, topology or
endpoint mappings. Concrete allocation addresses and transient runtime handles
remain runtime bindings. Rebinding compatible storage must not require
recompiling solely because its addresses changed. Redistribution cost estimates
must come from a compatible backend policy rather than assuming GPU/NCCL costs
apply to local HBM operations.

### Example of the compiler-visible computation

Consider the wholly local example with a global `[1024, 64]` tensor sharded on
dimension zero over eight HBM coordinates. Each coordinate owns a local logical
`[128, 64]` component. The table describes the proposed graph semantics, not new
public operator names.

| Source operation | Computation represented in the captured region | Result placement |
| --- | --- | --- |
| `y = x + x` | Eight component additions; all are visible to the local backend together. | `Shard(0)` |
| `p = y.sum(dim=0)` | Eight component reductions, each producing a `[64]` contribution. | `Partial(sum)` |
| `p.redistribute(placements=[Replicate()])` | An HBM-axis combine of the eight contributions, making the complete value observable at every coordinate. | `Replicate()` |

A CPU reference can lower the combine to ordinary reductions and copies. Spyre
can lower the same semantics into a joint execution plan with its own local
encoding, endpoint work and required visibility operations. Four endpoints
serving one coordinate do not become four additional DTensor participants. A
result stored at one HBM group meets `Replicate` only if every coordinate has the
required access through copies or a supported shared representation.

In the hierarchical `(tp, hbm)` example, process `t` captures its eight `(t, h)`
components. A TP-axis operation combines the corresponding components across
devices while holding `h` fixed. The compiled plan preserves participant
identity, component order and dependencies between local and distributed stages.
Packing several components into a transport operation is permitted only when
their logical ranges and matching remain intact. A subsequent TP combine must
not count HBM replicas as additional contributions.

### Backend responsibilities and the Spyre use case

PyTorch owns logical placement semantics. An out-of-tree accelerator backend
owns their hardware realization. For Spyre, torch-spyre retains local tensor
encoding, physical placement requests, and endpoint work division; Flex retains
allocation enforcement, addressing, submission, and runtime ordering.

The coordinate-to-HBM-group mapping comes from the backend's validated topology.
Several endpoints can serve one coordinate without increasing its semantic
participant count. Endpoint-local work must finalize the contribution prescribed
for that coordinate before a cross-coordinate reduction combines the eight
contributions.

When tensor parallelism is also active, the distributed executor handles
communication between devices while the local backend handles HBM operations
within each device. The same DTensor specification describes both levels; the
backend's physical endpoint work division does not create another mesh axis.

This proposal does not prescribe `Bind` or `Interleave` as the allocation policy.
Existing allocation machinery may realize an explicit placement plan if it
preserves its ranges, ownership, and locality requirements. It does not infer
tensor shards from equal byte division.

### Initial scope and compatibility

Introduce the interface experimentally, beginning with a static one-dimensional
mesh, equal non-empty shards, one local owner, and inference operations sufficient
to demonstrate pointwise computation and sum-based placement transitions. Provide
a CPU reference executor so the logical model can be exercised independently of
accelerator hardware. A production accelerator executor is a separate
integration against the agreed contract.

The next required milestone composes the local executor with an existing
distributed executor on a static two-dimensional mesh such as (`tp`, `hbm`).
The interface must account for that composition from the outset. A local-only
prototype establishes the first step but does not complete the Spyre use case;
the architecture must not be stabilized with hierarchical execution excluded.

The first executor uses blocking completion and rejects unsupported views,
mutation, RNG, and autograd operations explicitly. General support for those
features requires the corresponding ownership and semantic contracts before
stabilization. The existing process-based DTensor path keeps its current behavior
and feature coverage.

For this initial inference scope, materialization also rejects source tensors
requiring gradients; successful construction must not imply an unsupported
autograd connection.

Execution contexts are compatible only when their coordinate ownership and
logical mesh identities agree. Mixing tensors from incompatible contexts requires
an explicit transfer or rejection. In particular, an operation must not silently
combine distributed and wholly local tensors merely because their mesh shapes
match. Conversion between the two execution models is outside the first stage.

A tensor whose own context binds both local and distributed axes is a supported
design target, distinct from implicitly mixing tensors created under unrelated
contexts. Arbitrary conversion between such contexts can remain deferred.

## **Metrics**

Measure whether the local executor produces the same global values and placement
transitions as the existing DTensor semantics, while operating with one process
and without c10d initialization or simulated process-group state. Check that
existing process-based behavior is preserved and that new operators can reuse
shared placement strategies rather than requiring a second propagation system.

For hierarchical execution, evaluate the same global values and placements on
a mesh where each process owns multiple local coordinates. Verify that
redistribution selects the correct mechanism per axis, matches communication
participants consistently, and does not count local replicas or compute
endpoints as additional reduction contributions.

Evaluate eager and compiled execution against the CPU reference, including
gathered results and complete values observed at every replicated coordinate.

Establish compiled execution through an actual Inductor lowering, alongside
comparison with the eager reference. Successful tracing with an eager compiler
backend establishes capture support but does not demonstrate Inductor scheduling,
local redistribution lowering or native execution. Before declaring a compiled
scope supported, inspect the resulting graph and execution plan for the intended
component coverage, axis-specific redistribution and dependency structure.

Hardware evaluation should compare the executor with the backend's existing
sharding implementation using matched shapes, dtypes, layouts, topology, and
execution settings. This RFC makes no measured performance claim; native Spyre
accessibility and visibility require independent backend validation.

## **Drawbacks**

This adds an execution abstraction to a system whose local-tensor and mesh APIs
are currently rank-oriented. It affects storage representation, coordinate-aware
shape handling, dispatch, redistribution, and compilation; it cannot be delivered
through a custom collective backend or a sharding-rule registration alone.

Multiple local components complicate tensor extraction, views, aliasing, RNG,
and autograd. A narrow experimental scope reduces the initial implementation
burden but also limits which existing DTensor programs can run locally. Clear
capability errors and documentation are necessary to avoid implying feature
parity.

Composing local and distributed executors adds communication matching, buffer
packing, and cross-stage dependency obligations. Failure handling must cover
both local work and transport without publishing partially valid results.

Exposing both logical meshes and execution bindings increases API surface.
Maintainers may prefer an extension of existing mesh abstractions. The design
must preserve existing public behavior and avoid two independent placement
propagators or loosely synchronized copies of the tensor's logical specification.

## **Alternatives**

**Keep sharding entirely in each accelerator backend.** This supports immediate
hardware work and avoids changes to DTensor, but duplicates logical placement and
operator propagation and limits interoperability. It remains a practical path
while an upstream interface is discussed.

**Nest backend-local sharding beneath an existing distributed DTensor.** An
outer rank-based DTensor could carry a backend tensor whose storage is itself
locally sharded. This may support some combined workloads without a new public
mesh model. However, the HBM axis remains invisible to DTensor's placement,
redistribution, and strategy selection. The proposal makes both levels explicit
in one logical specification so their semantics and planning can be shared.

**Use LocalTensorMode directly as the production implementation.** It already
exercises multiple represented shards in one process. However, its current
contract is debugging/simulation; it retains fake process-group metadata and
patches rank/coordinate behavior. Its implementation is useful prior art, but a
supported execution contract should be deliberate rather than inherited from
testing machinery.

**Use one process per memory locality.** This aligns each coordinate with
DTensor's existing rank-local tensor model and can simplify reuse of current
tracing, shape propagation and compiler lowering. For Spyre, it requires
qualifying and coordinating several processes sharing one accelerator,
including device ownership, placement and allocation accounting, submission
ordering, cross-locality movement, and lifetime and failure handling. Joint
optimization across the localities would also need coordination across their
compilation scopes. This shifts complexity from compiler capture into the
backend/runtime and does not meet the proposed wholly local single-process
requirement. It remains an alternative to evaluate for backends whose resource
and process models support it.

**Use virtual-rank specialization during capture.** Specializing or tracing a
local computation for different logical coordinates could reuse parts of
rank-oriented DTensor tracing without creating extra operating-system processes.
It still needs to produce a coherent graph or execution plan for all locally
owned coordinates, preserving their geometry, aliases and redistribution
dependencies. Simulated rank metadata does not itself provide local execution or
collective matching. This could be an internal capture technique if its output
meets the compiler contract above. Existing `compile_on_one_rank` support compiles
on one rank for execution on multiple ranks; it does not by itself establish
simultaneous multi-coordinate execution in one process.

**Extend DeviceMesh with coordinate ownership.** This could reduce the number of
public abstractions if rank-based mesh construction and local mesh construction
remain unambiguous. It is a viable API alternative. It still requires a
multi-coordinate payload and execution interface; permitting repeated rank
entries alone would not establish them.

**Register a custom process-group backend.** This can implement communication
operations but leaves mesh coordinates and local payload selection tied to
process ranks. It does not provide the proposed ownership model by itself.

**Expose each memory locality as a PyTorch device.** This may fit backends whose
localities actually have independent device semantics. It does not fit one Spyre
device's shared runtime, streams, and allocation lifecycle, and it still does not
address multiple locally owned DTensor coordinates on its own.

**Expose placement propagation as a standalone compiler utility.** This could
reuse part of DTensor without changing its runtime representation. It is a
potential smaller contribution, but would leave tensor execution, extraction,
and redistribution as separate backend APIs rather than providing the DTensor
programming model requested here.

## **Prior Art**

PyTorch's private [LocalTensor and LocalTensorMode][localtensor] represent several
logical-rank tensors in one process and implement local equivalents of
collectives. The [DTensor testing harness][localtensor-harness] disables process
spawning while using simulated process-group metadata. This demonstrates that
existing DTensor behavior can be exercised over several local components.

There is also a [compilation test][localtensor-compile] using
`torch.compile(backend="eager", fullgraph=True)`. It establishes tracing/testing
support, not optimized accelerator execution. These observations refer to the
PyTorch revision inspected on 2026-10-30; they do not establish a public local
executor contract.

[PyTorch issue #199239][accelerator-testing] requests accelerator-agnostic
LocalTensor and fake-backend support for a PrivateUse1 accelerator. This is a
related testing-infrastructure contribution, distinct from hardware-backed local
DTensor execution.

[torch-spyre PR #5754][spyre-sharding] implements an initial global-shape tensor
with backend-owned shard storage and launch-time shard addressing.
[Spyre RFC #44][spyre-phase1] proposes explicit HBM-group placement and separate
execution endpoints; [RFC #45][spyre-phase2] adds reduction and visibility
contracts. These are concrete motivating designs, not evidence that native
execution or the DTensor integration proposed here is already complete.

## **How we teach this**

Teach three separate concepts: a global tensor's placement, ownership of its
logical mesh coordinates, and the resources used to execute them. Existing
distributed users continue to use rank-based DeviceMesh execution. A local
accelerator user selects a backend that owns all coordinates within one process.
The placement meanings stay the same.

Teach the combined case as hierarchical tensor distribution: inter-device
tensor parallelism on `tp`, intra-device sharding on `hbm`, and several local
coordinates owned by each process. The example must distinguish 32 logical
coordinates from four participating processes and from compute endpoints.

Use an example that follows `Shard(0)` through pointwise addition, a reduction to
`Partial(sum)`, and explicit redistribution to `Replicate`. Explain that a
redistribution expresses a data transformation; the executor chooses whether it
uses local operations or distributed transport. Document component extraction,
gathering, and explicit host transfer separately.

Add an experimental local-execution section to the DTensor documentation, plus
backend integration guidance. Prefer “logical mesh coordinate” and “local
execution context” over virtual ranks, since no process-group identity is
required. The term “local” refers to coordinate ownership, not to a guarantee of
contiguous memory or one physical backing allocation.

## **Unresolved questions**

Before agreeing to the architecture:

* Should a logical mesh be a new type, or a clearly separated construction mode
  of an existing mesh abstraction?
* Where should execution-context registration and per-axis binding live, and how
  should they integrate with existing distributed executors, PrivateUse1
  backends, and C++ DTensor dispatch?
* Should the first upstream contribution be the local executor interface, or an
  extraction of reusable logical propagation machinery that enables it?
* What representation reaches the AOT-to-backend boundary for multiple locally
  owned components, including output reconstruction and FakeTensor behavior?
* Which local redistribution operations have a shared semantic representation,
  when are they decomposed, and where can a backend select joint lowering?
* How do the chosen graph and effect representations preserve placement
  constraints, aliases, distributed collective matching and completion
  dependencies through compiler optimization?

Before stabilizing the implementation:

* What is the public component-extraction API, and what ownership, aliasing, and
  mutation guarantees does it provide?
* How should execution-context identity and backend capabilities be represented
  in FakeTensor, export, serialization, and compilation guards?
* What contracts are required for autograd, RNG, asynchronous completion, views,
  and replicated shared backing?
* How should hierarchical executors expose collective matching, batching,
  dependencies, and failure handling when a process owns several coordinates?

Combined local and distributed ownership is a required design capability.
Elastic remapping, unequal or dynamic shards, and general conversion between
execution contexts remain follow-on topics. The first stage must not introduce
assumptions that permanently prevent them.

## Resolution

Pending discussion in torch-spyre/RFCs. This proposal has not been submitted to
PyTorch's upstream RFC process, and no upstream decision is claimed.

### Level of Support

Not assessed; awaiting DTensor maintainer and community feedback.

#### Additional Context

The requested decision is whether logical mesh semantics and multiple local
coordinate ownership should be supported through a shared DTensor execution
interface that composes inter-device tensor parallelism with intra-device
sharding. Illustrative API names and experimental scope remain proposals.

### Next Steps

Discuss the mesh/execution boundary and compiler representation with DTensor and
compiler maintainers, agree on the capture and redistribution-lowering contract
for the smallest experimental contribution, and then prepare an implementation
proposal and tracking issue. Retain the CPU reference and accelerator realization
as separate stages so logical correctness does not depend on native hardware
availability.

Plan hierarchical local/distributed composition as the required step after the
wholly local reference, including per-axis execution and collective matching.

#### Tracking issue

Spyre discussion: [torch-spyre/torch-spyre#5983](https://github.com/torch-spyre/torch-spyre/issues/5983).

A separate upstream tracking issue has not been opened. Issue #199239 is related
prior art, not the tracking issue for this proposal.

#### Exceptions

The initial local executor is restricted as described above. Broader DTensor
feature support and native Spyre qualification are not implied by this RFC.
The initial milestone's local-only scope does not remove the requirement to
support combined local and distributed execution in the resulting design.

[localtensor]: https://github.com/pytorch/pytorch/blob/5f9684c2db892325470f836a2725c1cd6e68e65f/torch/distributed/_local_tensor/__init__.py#L917
[localtensor-harness]: https://github.com/pytorch/pytorch/blob/5f9684c2db892325470f836a2725c1cd6e68e65f/torch/testing/_internal/distributed/_tensor/common_dtensor.py#L1111
[localtensor-compile]: https://github.com/pytorch/pytorch/blob/5f9684c2db892325470f836a2725c1cd6e68e65f/test/distributed/test_local_tensor.py#L982
[accelerator-testing]: https://github.com/pytorch/pytorch/issues/199239
[spyre-sharding]: https://github.com/torch-spyre/torch-spyre/pull/5754
[spyre-phase1]: https://github.com/torch-spyre/RFCs/blob/75d3c99a0b6f371fe7f5cc60c7763c6b459fc701/NNNN-MultiDomainTopology/NNNN-MultiDomainTopologyRFC.md
[spyre-phase2]: https://github.com/torch-spyre/RFCs/blob/80516af43b503d368957945e3e600812fb2ac11d/NNNN-MultiDomainTopology/NNNN-MultiDomainTopologyPhase2RFC.md
