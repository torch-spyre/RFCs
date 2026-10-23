# Multi-Domain Topology for Intra-Device Spyre Tensor Sharding

**Authors:** Tuan M. Hoang Trong, Mudhakar Srivatsa

> **Status:** Draft. The RFC number and tracking issue are TBD.

## **Summary**

This RFC proposes a capability-driven `MultiDomainTopology` contract and an
intra-device sharding model that allow one Python process to use multiple logical
HBM groups and compute endpoints without exposing those resources as separate
PyTorch devices or distributed ranks. The initial qualified profile contains
eight logical HBM groups and 32 healthy compute endpoints.

The user continues to interact with one ordinary, global-shape tensor on
`spyre:0`. Internally, torch-spyre partitions its storage into eight semantic
shards, allocates each shard in a logical HBM group, and executes one pointwise
replica per shard. The initial correctness milestone supports blocking host
migration and same-placement pointwise `torch.add` through one explicit
32-endpoint program.

A central purpose of this RFC is to define the DeepTools contract needed to
make that program correct. At compile time, Inductor describes endpoint-local
work, layout, non-unified allocation, and symbolic tensor starts through the
SuperDSC-Bundle interface. DeepTools must preserve those symbols and emit both
the compiled program-correction mechanism and its final binding order. At
runtime, torch-spyre validates the actual tensor arguments, resolves their live
per-endpoint addresses in that order, and supplies those values to the
DeepTools-generated correction mechanism before launch.

The design separates five concerns:

1. PyTorch-compatible shard intent and global tensor semantics;
2. shard-local Spyre layouts;
3. physical allocation and immutable allocation receipts;
4. topology-selected execution planning; and
5. launch-time resolution of live addresses.

Concrete addresses remain runtime state. They are not embedded in tensor
semantics, compiler guards, or cache keys.

This proposal deliberately does not model the eight HBM groups as a distributed
`DeviceMesh`. PyTorch Tensor Parallel (TP) contributes shard vocabulary and
partitioning semantics, but torch-spyre owns the composite storage and runtime
execution model.

## **Motivation**

A multi-domain Spyre target presents one accelerator with several internal
memory and compute localities. Requiring users to model the initial profile as
eight devices, eight processes, or a Python list of shards would expose hardware
topology in ordinary model code and would not match PyTorch's view of one
accelerator.

The intended user model remains:

```python
x = x_cpu.to("spyre")
y = y_cpu.to("spyre")
z = torch.add(x, y)
z_cpu = z.to("cpu")
```

For eligible tensors under an enabled sharding policy, the physical
representation changes while the logical representation does not:

```text
Host                                      multi-domain Spyre target

one contiguous logical tensor            one global-shape Spyre tensor
+-----------------------------+           +----+----+----+----+
| shard 0 | shard 1 | ... | 7 |   H2D     | D0 | D1 | ... | D7 |
+-----------------------------+  ------>  +----+----+----+----+
                                          eight HBM allocations
                                          eight execution replicas
                                          32 compute endpoints
```

Logical contiguity and physical contiguity are different properties. A tensor
may retain ordinary global sizes, strides, and contiguity while its device
storage consists of independent allocations. No code may use
`is_contiguous()`, `data_ptr()`, or a global byte count as proof that the device
storage has one linear physical base.

Existing components provide useful primitives but not the complete semantic
model:

- torch-spyre represents Spyre as one PrivateUse1 device and has an experimental
  byte-interleaved placement path, but that path does not represent semantic
  tensor shards;
- Flex can represent multiple chunks in a `CompositeAddress`, but the production
  path does not yet provide the explicit eight-group allocation and immutable
  semantic receipt required here;
- the current multi-domain UMI path does not provide the required production
  multi-chunk migration and compute support; and
- DeepTools has non-unified allocation and symbolic-address mechanisms, but its
  final symbol ordering and explicit multi-endpoint behavior must be validated
  before they become a runtime ABI.

An explicit cross-layer contract is therefore required. Inferring shard identity
from chunk order, HBM group number, or endpoint number would be fragile under
allocator reordering, topology changes, and core sparing.

## Goals and non-goals

### Goals

- Preserve one ordinary global-shape tensor on one `spyre` device.
- Represent eight semantic shards with independently encoded local layouts.
- Allocate each shard in an explicitly selected logical HBM group.
- Scatter from one CPU tensor and gather back to one CPU tensor with exact
  round-trip fidelity.
- Execute supported shard-local pointwise operations over all eight shards and
  all 32 healthy endpoints.
- Keep semantic metadata address-free and resolve physical addresses only at
  launch.
- Preserve the existing unified-allocation path and Spyre I behavior.
- Reject unsupported or ambiguous cases before DMA or compute submission.

### Non-goals

The initial milestone does not provide:

- eight PyTorch-visible devices, virtual ranks, or a c10d process group;
- native DTensor storage or `parallelize_module()` execution;
- cross-card tensor parallelism or changes to `spyreccl`;
- cross-shard reductions or compiler-inserted communication;
- arbitrary views, aliases, mutation, autograd, or asynchronous execution;
- uneven, empty, symbolic, or dynamic shard geometry;
- general broadcasting, matmul, convolution, or sharded intermediates; or
- compact template replication or a reduced address ABI.

## **Proposed Implementation**

### User-visible contract

The canonical object is an ordinary `torch.Tensor`:

```python
assert isinstance(x, torch.Tensor)
assert x.device.type == "spyre"
assert x.shape == x_cpu.shape
```

For the initial admitted input class, global strides and logical contiguity are
also preserved. These properties describe the logical tensor, not a single
physical device range.

Phase 1 uses stock public PyTorch APIs and requires no modified PyTorch build,
c10d initialization, or distributed runtime. Public TP placement values are
inputs to torch-spyre planning, not a dependency on distributed execution.

Rollout begins with the proposed explicit blocking API
`torch.spyre.to_sharded()`, accepting a concrete public
`torch.distributed.tensor.Shard(dim)` placement:

```python
x = torch.spyre.to_sharded(x_cpu, placement=Shard(0))
```

After native allocation, migration, error handling, and pointwise execution are
qualified, an opt-in policy may route eligible `.to("spyre")` calls through the
same implementation. Ordinary `.to("spyre")` remains unified by default during
bring-up.

A plain `.to("spyre")` call has no shard dimension and must not infer one from a
tensor name, shape, or future use. A documented policy may eventually select a
default such as `Shard(0)`; other dimensions require explicit intent.

The initial path accepts only static, contiguous, non-aliased base tensors with
zero storage offset, exactly eight equal non-empty shards, supported dtype and
memory format, and a valid local Spyre layout for every shard. Explicit sharding
fails when these conditions are not met. A transparent policy may use unified
storage only when that fallback is explicitly configured and diagnosed.

### Architectural ownership

```text
PyTorch placement intent
  global shape, dtype, strides, Shard(dim)
                         |
                         v
                  torch-spyre
  shard ranges, local layouts, placement signature,
  copy orchestration, execution plan, output placement
                         |
             +-----------+-----------+
             |                       |
             v                       v
       Flex / Senlib              DeepTools
  physical allocation,       generated programs and
  receipt, topology,          final binding manifests
  translation, submission             |
             |                       |
             +-----------+-----------+
                         v
                    UMI runtime
          migration, address activation,
          correction, launch, completion
```

The ownership boundaries are normative:

- **torch-spyre** owns logical tensor meaning and the mapping from global ranges
  to semantic shards.
- **Flex** realizes explicit physical allocation requests and returns immutable
  evidence of what was allocated.
- **Senlib and qualified target data** provide topology and address-translation
  facts.
- **DeepTools** owns final generated-program structure and final per-HCM symbol
  order.
- **The runtime** resolves live allocations against the final manifest and
  submits work.

No component reconstructs another component's contract from vector position,
numeric coincidence, environment variables, or assumed endpoint numbering.

### Terminology and invariants

A **logical tensor** is the one global-shape tensor visible to PyTorch. A
**semantic shard** is one logical range of that tensor. A **logical HBM group**
is one of eight allocation localities and is not a physical HBM stack, HMI
service node, XLAT entry, distributed rank, or compute endpoint. An **execution
replica** processes one semantic shard on topology-selected endpoints.

The implementation must preserve these invariants:

1. One sharded allocation remains one tensor on `spyre:0`.
2. Physical placement does not change global shape, strides, storage offset, or
   logical contiguity.
3. The eight shard ranges cover the tensor exactly once.
4. In the normal Phase 1 configuration, the eight shards map bijectively to
   logical HBM groups `0..7`.
5. Every shard has its own local layout and one stable allocation-receipt entry.
6. Shard, receipt, HBM-group, replica, and endpoint identities are explicit even
   when some numeric values happen to match.
7. Receipt identity does not depend on `CompositeAddress` vector order.
8. Concrete addresses and transient translation slots do not enter compiler
   cache keys unless generated code embeds a slot.
9. The runtime validates every live binding before correction or launch.
10. A multi-chunk tensor is never reduced to its first chunk or treated as one
    global physical base.
11. Any failed participant fails the complete logical operation.
12. Newly allocated destinations remain invalid until all H2D entries complete;
    pointwise outputs remain invalid until all endpoint work completes. Submitted
    work retains storage, receipts, translation state, and completion resources
    until it reaches a terminal state.

### Multi-domain topology contract

`MultiDomainTopology` is the generic, immutable, versioned capability contract
between hardware discovery, torch-spyre planning, compilation, and launch. It
uses stable logical namespaces rather than a product or hardware-generation
name. The public Python API, serialized placement metadata, compiler guards, and
SuperDSC interface refer to this contract or its fingerprint, not to a hardware
codename.

The initial qualified profile reports four physical HBM stacks exposed as eight
logical DVA/MCI HBM groups. It reports 16 working physical core slots with two
working sub-core endpoints each, for 32 healthy compute endpoints. Each logical
HBM group has four XLAT regions. These namespaces must remain distinct.

The initial profile reports 12 GiB per logical HBM group, but allocation must use
the capacity reported after channel sparing, firmware reservations, and allocator
reservations. The maximum XLAT window is virtual aperture, not physical
capacity.

The initial implementation uses one immutable, versioned, qualified
`MultiDomainTopology` profile that maps each logical HBM group to four compute
endpoints. The profile is deployment input derived from Senlib and hardware
qualification; it is not arbitrary user tuning. It includes a target identity
and fingerprint used for compiler and launch validation.

The implementation must not calculate endpoint IDs from an HBM-group or shard
number. An unknown target, malformed profile, unavailable endpoint, or uncovered
sparing state fails closed. Dynamic topology and cost-based endpoint selection
are follow-on capabilities.

The current public Senlib DVA representation fixes `VA[39]` to zero. The initial
path must either qualify that behavior for every requester it uses or remain
disabled until an explicit public address-mode API is available.

### Semantic placement and storage

PyTorch's public `Shard` type supplies placement vocabulary. A version-tested
adapter validates the shard dimension and delegates range calculation to public
`Shard.local_shard_size_and_offset()`. The backend must not silently substitute
a second partitioning algorithm if that API is incompatible. This use of
`Shard` does not create a `DeviceMesh`, a `ProcessGroup`, or native rank-local
DTensor storage.

For each shard, torch-spyre records address-free metadata sufficient to drive all
later stages:

- global logical origin and extent;
- compact local logical shape and strides;
- local layout identity and encoded byte requirement;
- semantic placement slot and shard identity;
- requested logical HBM group;
- stable allocation-request and receipt identities; and
- the `MultiDomainTopology` fingerprint.

The collection of eight records is canonical. Downstream components consume it
rather than recomputing shard ranges.

Each shard receives an independently constructed `SpyreTensorLayout`. The
implementation must not divide one globally encoded byte buffer into eighths
unless a layout proves that this is equivalent. Equal shards may share a layout
class or compiled template only after their code-affecting geometry is shown to
be equivalent.

The storage object owns the composite allocation, immutable allocation receipt,
semantic placement, validity state, and any resources retained by in-flight
work. `data_ptr()` remains an opaque ownership token; it is never a globally
addressable device base.

### Allocation and migration

The generic byte-count allocator cannot infer semantic shard placement.
torch-spyre therefore creates an explicit allocation plan while global shape,
dtype, strides, placement, and local layouts are available. Flex allocates all
eight entries transactionally and returns a receipt preserving stable shard
identity and verified logical HBM-group placement.

If any entry fails, all earlier entries are released and no tensor or receipt is
returned. Recording a requested domain without verifying the physical result is
not sufficient.

Host-to-device transfer performs, for each shard:

```text
global host slice
  -> compact local staging
  -> shard-local encoding and padding
  -> matching receipt entry in one logical HBM group
```

Device-to-host transfer reverses the transformation, removes local padding, and
writes each shard into its global logical range. Completion means all eight
transfers completed. A failed transfer must not produce a valid destination or a
partially successful host result.

The first implementation is blocking. `non_blocking=True` is rejected or
explicitly synchronized until the runtime can retain storage, receipts, staging,
translation state, and completion objects for the full asynchronous lifetime.

### Pointwise execution

The first required operation is same-shape binary `torch.add`:

```text
Shard(d), Shard(d) -> Shard(d)
```

Before choosing output placement or allocating storage, the implementation
derives the global output shape, strides, and dtype from ATen/meta semantics; it
does not copy input zero's metadata blindly.

The compiler preserves the global tensor shape. Spyre-private lowering may
produce shard-local iteration spaces only after proving that:

- input placements and shard boundaries match;
- each output element depends only on the corresponding input shard;
- broadcasting does not cross shard boundaries;
- all shard-local layouts and work divisions are valid; and
- output placement is known before allocation.

### Compile-time and runtime DeepTools contract

The multi-domain program is correct only if its compile-time description and
runtime tensor bindings identify the same tensor region for every participating
endpoint. This is a two-part interface:

1. **At compile time, Inductor to DeepTools:** Inductor communicates the proven
   shard-local execution through SuperDSC JSON and SuperDSC-Bundle MLIR. The
   interface describes endpoint-local work and layout, identifies every tensor
   start as a symbol, and establishes how those symbols enter the compiled
   program.
2. **At runtime, torch-spyre to the DeepTools-generated program-correction
   mechanism:** torch-spyre resolves each symbol from the actual input and output
   tensors supplied to the invocation. It passes the resulting ordered address
   vector to the correction mechanism emitted in `SpyreCode`, which substitutes
   or corrects the symbolic starts before device execution.

DeepTools therefore owns more than parsing SuperDSC JSON. It must compile the
multi-domain description into an artifact whose correction inputs retain an
unambiguous, machine-readable relationship to the original tensor symbols.
DeepTools does not allocate the user tensors and must not infer their live
addresses at compile time.

For every tensor allocation participating in the operation, the compile-time
contract carries these fields:

| SuperDSC field | Required meaning |
|---|---|
| `N_` | Global operation dimensions retained for operation semantics. |
| `dataStageParam_[0].ss_` and `.el_` | Tensor dimensions consumed or produced by one endpoint. |
| `primaryDsInfo_` | Local tensor layout class, including `layoutDimOrder_`, `stickDimOrder_`, and `stickSize_`. |
| Allocate-node `layoutDimOrder_` and `maxDimSizes_` | Dimension ordering and bounds of the local allocation. |
| Allocate-node `coordinates_.coordInfo` | Affine coordinate folds that encode local layout progression, including local memory strides and the core fold. |
| `numCoresUsed_` and `coreIdsUsed_` | The 32 topology-selected compute endpoints; IDs need not be dense. |
| `numWkSlicesPerDim_` and `coreIdToWkSlice_` | The logical work partition assigned to each endpoint. |
| Allocate-node `nonUnifiedAllocInHBM_ = true` | Size and interpret HBM data from the endpoint-local data stage rather than as slices of one allocation sized by global `N_`. |
| Allocate-node `isStartAddrSymbolic_ = 1` | Resolve starts at runtime rather than embedding live addresses. |
| Allocate-node `startAddressCoreCorelet_` | Map each participating endpoint to its independent symbolic start address. |

Conceptually, the required JSON shape is:

```text
SuperDSC {
  numCoresUsed_: 32
  numWkSlicesPerDim_: ...
  coreIdToWkSlice_: {endpoint_id: logical_work_slice, ...}
  dscs_[0] {
    numCoresUsed_: 32
    coreIdsUsed_: [topology-selected endpoint IDs]
    N_: global operation dimensions
    dataStageParam_[0]: {ss_: local dimensions, el_: local dimensions}
    primaryDsInfo_: {local layout and stick geometry}
    scheduleTree_: [
      AllocateNode {
        component_: "hbm"
        nonUnifiedAllocInHBM_: true
        isStartAddrSymbolic_: 1
        layoutDimOrder_: ...
        maxDimSizes_: ...
        coordinates_.coordInfo: local affine layout/stride folds
        startAddressCoreCorelet_[endpoint_id, 0, 0]: symbol
      }
    ]
  }
}
```

`nonUnifiedAllocInHBM_` is an allocation interpretation, not a complete sharding
contract. When it is `false`, per-endpoint starts refer to subrectangles of one
allocation and may be derived from a common base plus global layout offsets.
When it is `true`, every endpoint start denotes an independently located local
HBM tensor, and DeepTools sizes that tensor from the local data stage. The
address rule becomes:

```text
address(endpoint, local_coordinate)
  = runtime_start[endpoint]
  + local_layout_offset(local_coordinate)
```

There is no required base-to-base stride between endpoints. This is the
**non-unified placement** property. The stride and coordinate mapping *within*
each local tensor remains explicit in `primaryDsInfo_`, `layoutDimOrder_`, and
`coordinates_.coordInfo`.

The Phase 1 SuperDSC uses one proven local layout class for all equal shards. The
existing interface does not describe arbitrary, independently varying internal
layouts or strides for every endpoint. If local layouts differ, Inductor must
partition replicas into layout-equivalent DSC/template classes or reject the
operation; supporting fold-indexed per-endpoint layouts would require a separate
SuperDSC schema extension.

The following information is not encoded by `nonUnifiedAllocInHBM_` and remains
in torch-spyre's address-free execution and binding metadata: graph tensor
argument and access mode, shard and replica identity, logical HBM group,
endpoint-local byte offset and valid length, expected placement and layout, and
the topology fingerprint. The live allocation-receipt identity is attached only
when an actual tensor is presented at runtime. DeepTools need not understand
Flex receipts or PyTorch shard semantics; it must preserve an opaque binding key
that lets torch-spyre join the final correction input back to this metadata.

For each tensor and endpoint, the compile-time handoff establishes this chain:

```text
Inductor binding metadata:
  binding key B -> graph tensor argument A, replica R, expected receipt entry Q,
                   byte offset O, valid length L, access mode, placement checks

SuperDSC JSON:
  endpoint E uses symbolic start S for tensor allocation T

SuperDSC-Bundle MLIR:
  operand K supplies symbol S and carries binding key B

DeepTools output:
  final HCM correction input K -> symbol S and opaque binding key B
```

DeepTools may transform, duplicate, or reorder symbols while compiling. It must
therefore emit, for every HCM, the **final** ordered correction-input manifest
after those transformations. Each entry identifies its final input position,
original symbol, and opaque binding key. The manifest also states the exact
number and integer representation of correction inputs expected by the
DeepTools-generated host-compute operation. A provisional order emitted by the
Inductor frontend is not authoritative after DeepTools transforms the program.

At runtime, torch-spyre handles an actual invocation as follows:

```text
PyTorch invocation tensors
  -> validate tensor role, dtype, shape, layout, placement, topology, and access
  -> use binding key B to select the actual tensor and semantic receipt entry
  -> resolve that entry's live endpoint-visible address plus byte offset O
  -> place the address at final correction-input position K
  -> pass the complete ordered vector to the generated host-compute operation
  -> DeepTools program correction substitutes S before the device program runs
```

This lookup applies equally to input, output, and intermediate tensor symbols.
Compilation cannot capture a live address, because a cached artifact can be used
with different tensor allocations. Conversely, torch-spyre must not invent a
symbol order from tensor argument order, endpoint number, receipt vector order,
or a frontend sidecar. Missing, duplicate, stale, or incompatible bindings fail
before program correction and launch.

The positional invariant is:

```text
DeepTools final per-HCM manifest order
  == HCM inputSym_ order
  == SpyreCode ishape[0] element order
  == runtime symbolic_inputs order
```

DeepTools must preserve `nonUnifiedAllocInHBM_`, allocation coordinates, sparse
endpoint IDs, symbolic starts, and opaque binding keys through parsing,
allocation reconstruction, semantic hashing, lowering, and program correction.
The canonical JSON Boolean representation of `nonUnifiedAllocInHBM_` must be
accepted deliberately. `nonUnifiedAllocInHBM_ = true` alone is insufficient:
local data stages, layout coordinates, endpoint work slices, address symbols,
bundle operands, final correction-input manifests, and runtime bindings must
describe the same execution.

#### Required DeepTools support

This RFC specifically asks DeepTools to define and support the following
behavior:

1. accept the multi-domain SuperDSC fields and non-unified allocation semantics
   defined above;
2. preserve the association among endpoint, tensor start symbol, and opaque
   Inductor binding key even if compilation reorders or duplicates symbols;
3. generate `SpyreCode` containing the program-correction mechanism required to
   substitute symbolic starts before execution;
4. emit a versioned final per-HCM correction-input manifest after all DeepTools
   transformations, including the exact input order and count expected by that
   mechanism; and
5. define the runtime handoff by which torch-spyre supplies the ordered scalar
   address vector and receives a clear failure for a count, type, symbol, or
   artifact-version mismatch.

The runtime boundary is deliberately not a PyTorch tensor API. torch-spyre owns
interpretation and validation of the actual invocation tensors and converts
them to endpoint-visible addresses. The DeepTools-generated host-compute
operation consumes only the validated address vector and uses it to correct the
compiled program. This division lets one compiled artifact run correctly with
new tensor allocations without teaching DeepTools about PyTorch storage or Flex
allocation receipts.

Bring-up first compares eight independent four-endpoint launches with CPU. The
Phase 1 correctness artifact is then one explicit non-unified program containing
all 32 topology-selected endpoints and their work slices. Endpoint IDs are read
from the qualified `MultiDomainTopology`, never generated as `range(32)`.

For binary add, the initial runtime ABI supplies one live address per tensor and
endpoint:

```text
32 lhs addresses + 32 rhs addresses + 32 output addresses = 96 values
```

This explicit ABI avoids assuming that all four requesters in one replica observe
an identical effective base. A compact 24-base ABI or backend replication of one
four-endpoint template is a later optimization. Either requires hardware proof
and equivalence to the explicit artifact.

DeepTools emits the final correction-input manifest for every HCM after all
symbol and program transformations. At launch, torch-spyre validates the
artifact and manifest versions, topology fingerprint, tensor roles, placement
signatures, receipt identities, logical HBM groups, layouts, ranges, access
modes, and parameter counts. It then resolves live addresses in the manifest's
final order and passes them to the DeepTools-generated host-compute operation,
which performs program correction. A frontend-generated provisional sidecar is
not the final positional authority.

### Failure and compatibility policy

Validation is transactional and occurs before device submission whenever
possible:

- allocation failure rolls back every completed allocation;
- migration failure invalidates the complete destination and never returns a
  partial host value;
- manifest or binding mismatch aborts before program correction;
- one endpoint failure fails the complete pointwise operation; and
- uncertain post-submission writes poison the destination until it is safely
  released or restored.

There is no silent fallback to chunk zero, one address per tensor, a fabricated
unified base, or a different placement. A safe fallback must be an explicit
semantic gather, operation, and redistribution.

The existing unified allocation and execution path remains unchanged for tensors
without semantic sharding. Existing single-domain targets remain on their
current allocator and runtime paths. Capability guards prevent multi-domain
contracts from reaching unsupported targets.

### Single-domain software oracle

A single-domain target may emulate the software decomposition using eight
semantic placement slots backed by eight independently owned allocations in its
one physical memory domain. Semantic identities must remain distinct; rewriting
every shard identity to zero would not exercise planning, scatter/gather,
binding, or output placement.

The emulator can validate software ownership, layout, migration, binding, and
failure semantics. It cannot satisfy native multi-domain placement, DVA/XLAT,
topology, coherence, endpoint, or performance acceptance criteria. Qualification
on a native target implementing `MultiDomainTopology` therefore remains
mandatory even when the same semantic tests pass under single-domain emulation.

### Rollout and future phases

The feature is controlled by a dedicated, default-off semantic-sharding
capability independent of the existing byte-interleaving experiment. Rollout is:

1. freeze versioned semantic, topology, receipt, and binding contracts;
2. enable planning and inspection without execution;
3. validate software decomposition with mocks or single-domain emulation;
4. enable native transactional allocation and blocking scatter/gather;
5. validate eight independent pointwise replicas;
6. enable the explicit 32-endpoint/96-address artifact; and
7. enable opt-in transparent `.to("spyre")` after all native gates pass.

This RFC establishes the storage, migration, and pointwise correctness
foundation. Follow-on work may add:

- HBM-mediated cross-shard reductions using explicit partial values and a
  deterministic combiner;
- compiler-visible `Unified`, `Shard`, `Replicate`, and `Partial` placement
  propagation;
- a read-only public placement-inspection API after the placement and receipt
  contracts are stable;
- explicit redistribution and communication nodes;
- topology- and cost-aware HBM, ring/LX, or direct communication;
- compact backend replication and reduced address ABIs; and
- asynchronous operations, uneven shards, degraded topology, broader operators,
  views, autograd, and training.

These extensions must preserve the explicit Phase 1 path as a correctness oracle
until equivalence is established.

## **Metrics**

Phase 1 is successful when:

- blocking H2D/D2H round trips reproduce CPU values at every shard boundary;
- all eight allocations are verified in their requested logical HBM groups on
  the initial native multi-domain target;
- `torch.add` covers every shard exactly once and matches CPU results;
- the explicit program uses all 32 healthy endpoints under the qualified
  `MultiDomainTopology`;
- binary add receives exactly 96 validated runtime bindings in final manifest
  order;
- compatible tensors allocated at different live addresses reuse the same
  compiled artifact;
- injected planning, allocation, migration, topology, manifest, and launch
  failures are reported without partial valid results, leaks, or hangs; and
- unified and existing single-domain paths do not regress.

Performance measurements include aggregate H2D/D2H bandwidth, pointwise scaling
relative to the current one-domain path, compilation overhead, launch overhead,
and cache reuse. Performance informs later optimization but does not replace the
correctness gates.

## **Drawbacks**

### Cross-stack complexity

The design adds contracts across torch-spyre, Flex, Senlib, UMI, and DeepTools.
Versioned boundaries and stable identities increase implementation work, but
implicit conventions would be unsafe when allocation order or topology changes.

### Restricted initial coverage

The first milestone handles a narrow tensor class and one pointwise operation.
This is intentional: it establishes storage, migration, and launch correctness
before introducing reductions, general placement propagation, or asynchronous
lifetimes.

### Additional compiler specialization

Shard geometry, layout classes, topology capabilities, and code-baked address
properties add cache dimensions. Excluding physical addresses and relocatable
translation assignments avoids unnecessary recompilation.

### Potential copy overhead

Correct migration may require compact staging and per-shard Spyre encoding rather
than slicing one pre-encoded byte stream. Direct scatter/gather optimizations
must be justified by layout equivalence and measurements.

## **Alternatives**

### Use native DTensor and an eight-rank `DeviceMesh`

DTensor models one rank-local tensor per process-group rank and uses collectives
for redistribution. One process owning all eight physical shards does not match
that storage model. Fake ranks would risk collective deadlock and still would not
provide one composite local tensor.

### Expose eight Spyre devices

This would turn internal HBM locality into a user-visible distributed topology,
requiring explicit orchestration for operations that are local to one physical
accelerator.

### Represent shards as a Python list

A list loses ordinary tensor dispatch, aliasing, module, and compiler semantics
without pervasive graph rewriting. It is useful for diagnostics, not as the
canonical user object.

### Let Flex infer equal interleaving

Equal byte division cannot express logical shard ranges, per-shard layouts,
padding, or explicit output placement. Flex should realize a plan rather than
infer PyTorch semantics.

### Encode HBM groups in tensor strides

PyTorch strides describe one affine logical-to-storage mapping. Eight independent
physical bases are not one affine address space, and changing strides would
corrupt view and contiguity semantics.

### Compile only one shard

Presenting one shard's local shape as the global graph shape would break global
shape reasoning, output semantics, future reductions, and alias analysis.
Shard-local shapes belong in backend-private lowering metadata.

### Start with compact backend replication

A four-endpoint template would reduce compiler input size, but correct expansion
must remap endpoints, programs, barriers, LX allocations, symbols, correction
sites, and ownership. The explicit 32-endpoint artifact is easier to audit and
serves as the required oracle.

### Treat all chunks as one virtual contiguous allocation

No current contract guarantees one linear DVA across the eight allocations.
Deriving other group addresses from one base could access the wrong memory.

### Do nothing

The backend would continue to underuse multi-domain target memory capacity,
bandwidth, and compute parallelism, and would lack a foundation for intra-device
tensor parallelism.

## **Prior Art**

PyTorch DTensor and Tensor Parallel provide useful vocabulary for `Shard`,
`Replicate`, local shape derivation, and redistribution. This RFC reuses those
semantic concepts while keeping one-process, intra-device composite storage
distinct from rank-local distributed storage.

Existing torch-spyre experimental work on interleaved HBM placement established
a separation between logical tensor properties and physical HBM placement. This
proposal extends that principle from a byte-oriented, one-endpoint-per-chunk path
to semantic shards and multiple-endpoint execution per HBM group. That work is
not cited as an RFC until its document is added to the RFC repository.

The
[Tensors with Device-Specific Layouts RFC](../0047-TiledTensors/0047-TiledTensorsRFC.md)
keeps device encoding below ordinary PyTorch tensor semantics. The same rule
applies here: sharding and placement do not replace the tensor's global logical
shape or strides.

NUMA-aware runtimes and SPMD compilers provide analogous locality and replicated
execution concepts. This design differs because all localities remain within one
PyTorch device and one process.

## **How we teach this**

Documentation should begin with one statement:

```text
PyTorch sees one tensor; Spyre distributes its storage across HBM groups.
```

Use **intra-device HBM shard**, **logical HBM group**, and **execution replica**.
Do not call HBM groups ranks or devices. Do not describe a logically contiguous
PyTorch tensor as non-contiguous without qualifying that the statement concerns
physical device storage.

Developer documentation should explain the layers independently:

```text
Shard intent       which logical elements belong to each shard
Local layout       how each shard is encoded
Allocation receipt which physical allocation realizes each shard
Execution plan     which endpoints process each shard
Binding manifest   which live address supplies each program parameter
```

Detailed schemas, source change maps, work packages, test matrices, and debugging
procedures belong in implementation plans rather than in this RFC.

## **Unresolved questions**

This section contains open decisions that can change a public API,
cross-component ABI, correctness rule, or ownership boundary. Qualification
experiments that test the proposed design are listed separately as activation
gates; later optimizations are listed under rollout and future phases.

### DeepTools interface decisions

The following questions require agreement with the DeepTools owners before the
compile-time and runtime interface can be frozen:

1. What hardware namespace does a SuperDSC `core_id` represent, and how does it
   relate to the endpoint IDs in `MultiDomainTopology`?
2. What versioned final per-HCM correction-input manifest will DeepTools emit
   after symbol and program transformations? It must identify final input
   position, original symbol, opaque Inductor binding key, required scalar
   representation, and expected input count.
3. What `SpyreCode` runtime entry point supplies correction inputs to each HCM,
   and how are missing, duplicate, wrongly typed, or incorrectly counted inputs
   reported?
4. Which symbol transformations may DeepTools perform, and what invariants will
   ensure that non-unified placement, sparse endpoint IDs, and binding keys
   survive those transformations?

These are interface-design questions, not requests for DeepTools to consume
PyTorch tensors or Flex receipts. Inductor supplies the static multi-domain
program and binding identities. At invocation, torch-spyre validates the actual
tensor inputs and outputs, resolves their live endpoint-visible addresses, and
supplies the final ordered values. DeepTools supplies the artifact metadata and
program-correction mechanism needed to apply those values to the compiled
program correctly.

### Other design decisions

1. Which runtime component owns XLAT reservation, eviction, and isolation across
   concurrent launches?
2. What cancellation and quarantine behavior is required after partial DMA or
   compute submission?

A read-only public placement-inspection API is useful for diagnostics but does
not block the proposed `torch.spyre.to_sharded()` constructor. Its design is
deferred until the placement and receipt contracts are stable.

### Implementation validation gates

The following questions do not need to block architectural review, but the
feature remains disabled on a target until evidence answers them:

1. What qualified mapping relates the eight logical HBM groups to physical HMI
   service nodes and compute endpoints?
2. Can every selected endpoint read and write its assigned tensor region under
   supported requester, XLAT, permission, and sparing configurations?
3. Is fixed `VA[39] = 0` valid for every requester used by this path?
4. Does DeepTools preserve 96 address symbols, sparse endpoint IDs, non-unified
   allocation metadata, opaque binding keys, and final per-HCM correction-input
   order through its complete pipeline?
5. Can one compiled artifact execute correctly with multiple sets of actual
   torch-spyre tensor allocations, and reject permuted, stale, incomplete, or
   placement-incompatible runtime bindings before launch?
6. Do the single-domain emulator and native multi-domain target use
   byte-identical local-layout semantics, or is a target-specific layout ABI
   discriminator required?

Compiler-visible placement and communication, compact address binding, template
replication, asynchronous execution, uneven shards, general views, and optimized
intra-device communication remain non-blocking follow-on work under **Rollout
and future phases**.

## Resolution

TBD.

### Level of Support

TBD.

#### Additional Context

This RFC chooses an explicit implementation as the initial correctness oracle.
Compact code generation and communication optimization are accepted only after
they are shown equivalent to the explicit storage, migration, and execution
path.

### Next Steps

1. Assign an RFC number and open a tracking issue.
2. Agree with DeepTools owners on the final per-HCM correction-input manifest,
   opaque binding-key preservation, and `SpyreCode` runtime handoff.
3. Resolve the remaining blocking address-lifecycle and cancellation decisions.
4. Freeze the versioned semantic-placement, allocation-receipt,
   `MultiDomainTopology`, and final-binding contracts.
5. Implement transactional allocation and blocking scatter/gather.
6. Validate the topology, address, layout, and DeepTools activation gates,
   including reuse with different actual tensor allocations.
7. Validate eight independent pointwise replicas, then the explicit
   32-endpoint/96-address artifact.
8. Evaluate transparent `.to("spyre")` only after native correctness and failure
   gates pass.

#### Tracking issue

TBD.

#### Exceptions

Native DTensor storage, fake ranks, cross-card collectives, reductions, general
views, dynamic or uneven shards, asynchronous execution, and broad operator
coverage remain outside the initial milestone. They require separate design and
acceptance criteria.
