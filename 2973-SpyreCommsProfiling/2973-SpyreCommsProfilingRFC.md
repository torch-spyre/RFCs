# RFP: Profiling Support for spyre-comms

## 1. Problem Statement

spyre-comms is the communication library for Spyre AIU devices, providing collective (allreduce, allgather, broadcast, gather, reduce, barrier) and point-to-point (send, recv, sendrecv) primitives. Currently, **there is zero runtime profiling instrumentation** inside spyre-comms itself. The `GlobalTimingProfile` infrastructure from the shared `common/` submodule is available but unused.

Flex has a mature unified profiling system with three backends (TIMING, FLEX/Chrome-trace, AIUPTI/PyTorch-profiler), but it can only see the hardware-level operations (control blocks, DMA, RDMA) that spyre-comms submits — it has no visibility into:

- Which collective is executing and which algorithm was chosen
- Total bytes involved in the collective
- Host-side time spent inside spyre-comms (algorithm selection, work schedule construction, code generation)
- Per-step breakdown within a collective algorithm (e.g., individual ring steps in a pipeline allreduce)
- Overhead of P2P protocol setup (message matching, HDMA channel management)

This makes it extremely difficult to identify performance bottlenecks in distributed workloads, compare algorithm choices, or understand the communication/computation overlap.

---

## 2. Current State (What Exists Today)

### In spyre-comms
| Component | Purpose | Runtime profiling? |
|---|---|---|
| `common/timing.hpp` (submodule) | `GlobalTimingProfile` with `AIU_TIMING_*` macros | Available but **unused** |
| `src/utils/timer.hpp` | `TIMER_NOW()`/`TIMER_DURATION()` macros | Only used for timeouts |
| `src/coll/cost_estimator.*` | Offline cost modeling for algorithm selection | Not runtime |
| `tools/perf-bench/` | External benchmark tool (P2P only) | Standalone, not library instrumentation |
| `src/coll/collective_stat.hpp` | Records collective type + volume (no timing) | Logging only |

### In flex
| Component | Purpose |
|---|---|
| Unified Profiler (`telemetry/`) | Three-backend profiler (TIMING, FLEX, AIUPTI) |
| `extractCollGroup()` / `extractCollMetadata()` | Regex-based identification of collective ops from `node_name` strings |
| `PROFILER_*` macros | Zero-overhead when disabled (single atomic check) |
| Chrome Trace JSON output | Per-rank `.json` traces viewable in chrome://tracing |
| AIUPTI backend | Activity records consumed by PyTorch profiler |

---

## 3. Proposed Solution

### Design Principles

1. **Unified with flex profiler** — events from spyre-comms should appear in the same Chrome trace / AIUPTI stream as flex events
2. **Structured metadata** — collective info passed as metadata having typed fields
3. **Hierarchical** — collective → algorithm steps → individual P2P operations, visible as nested spans
4. **Opt-in granularity** — env-var controlled verbosity levels (collective-level vs. step-level vs. operation-level)
5. **Low overhead** — Profiling disabled by default; when enabled, acceptable overhead (<10%)
---

### Short Term Goals

#### 3.1. [DONE] Pass collective info to flex by encoding collective metadata into operation name strings

PRs:

- [spyre-comms#268](https://github.ibm.com/ai-chip-toolchain/spyre-comms/pull/268) — encodes collective metadata into operation name strings using a structured format:
```
[CollType,Algorithm,Bytes] OperationName
```
For example: `[AllReduce,AllReduce_AllGatherSum,2048] Send`

It adds a `pushCollectiveAnnotation()`/`popCollectiveAnnotation()` mechanism in `SpyreCommsContext` that prepends this metadata to all Send/Recv/DMA operation names within a collective's `Convert()` method.

- [flex#1455](https://github.ibm.com/ai-chip-toolchain/flex/pull/1455) — Parses this structured format in `extractCollMetadata()` and adds `CollAlgo` and `CollBytes` as profiler attributes alongside the existing `CollGroup`. It also threads `op_name` through the H2D/D2H runtime operation path so DMA transfers appear with meaningful names in traces.


#### 3.2. Add `AIU_TIMING` instrumentation to spyre-comms core paths

Instrument the following critical paths using the already-available `GlobalTimingProfile`:

- **`WorkSchedule::start()`** — total time from schedule submission to first operation launch
- **`WorkSchedule::wait()`** — total blocking time waiting for completion
- **Collective entry points** (`allreduce()`, `allgather()`, etc.) — end-to-end collective duration
- **Algorithm selection** (`CostEstimator` path) — time spent choosing algorithm
- **Bundle generation** (`BundleGenerator::generateBundle()`) — compilation time for compute kernels

This gives immediate host-side visibility without requiring any flex changes:
```bash
AIU_TIMING_ENABLED=1 torchrun --nproc-per-node=4 model.py
```


### Long-Term Goals 

#### 3.3. Formalize the metadata interface between spyre-comms and flex 

Replace string-encoded metadata with a structured type:

```cpp
// In shared header (common/ or flex public API)
struct CommOperationMetadata {
    std::string collective_type;   // "AllReduce", "AllGather", etc.
    std::string algorithm;         // "AllReduce_PipelineLinear", etc.
    size_t total_bytes;            // Total data volume
    int step;                      // Step within algorithm (-1 if N/A)
    int total_steps;               // Total steps in algorithm (-1 if N/A)
    int rank;                      // Source/destination rank (obtained from spyrecomm / spyreCCL backend)
    std::vector<at::Tensor> input_tensors;
    std::vector<at::Tensor> output_tensors;
};
```

Pass this through `DmaParams`, `P2PRdmaSendParams`, `P2PRdmaWaitParams`, etc., instead of embedding in `op_name` strings. This eliminates the regex parsing in flex's `pf_runtime_scheduler.cpp`.

**Rank tagging:** The `rank` field is populated using peer rank information obtained from the spyrecomm / spyreCCL backend. The collective algorithm layer in spyre-comms already tracks peer rank for each send/recv operation; this is propagated into the metadata at the operation dispatch boundary. For protocol events where only PCI addresses are visible, a PCI-to-rank mapping is built at initialization time from the `AIU_WORLD_RANK_x` environment variables.


#### 3.4. Add collective-level profiling events visible in flex traces

Emit `PROFILER_START`/`PROFILER_STOP` events at the collective boundary so that a single span in the Chrome trace covers the entire collective operation (all its DMA/RDMA/compute sub-operations appear nested within it):

```
[AllReduce span] ──────────────────────────────
  [H2D: Start H2D] ──
  [Send step 0]     ────
  [Recv step 0]     ────
  [Compute: Sum]       ──
  [Send step 1]          ────
  [D2H: End D2H]              ──
```

This requires spyre-comms to call into the flex unified profiler API (or a thin wrapper exposed for this purpose).

##### Protocol Event Classes

Emit the following event classes at the protocol level:

| Event Class | Protocol Role | Metric |
|---|---|---|
| SEND_DATA | Send: outbound DMA data transfer | Time (usec), Bytes, Peer |
| SIGNAL_DATA | Send: signaling instruction | Time (usec), Peer |
| MONITOR_NOTICE | Recv: wait for delivery notice *(optional — only present for DMA in PF mode)* | Time (usec) |
| RECV_DATA | Recv: inbound DMA data transfer | Time (usec), Bytes, Peer |
| COMPUTE_PREP | Op: pre-compute setup | Time (usec) |
| COMPUTE_EXEC | Op: local compute (e.g., sum reduction) | Time (usec) |

#### 3.5. PyTorch profiler integration (AIUPTI path)

Ensure spyre-comms collective operations appear as first-class activities in PyTorch's profiler output:

- Define new `AIUpti_ActivityKind` values for each collective type
- Map collective algorithm steps to AIUPTI activity records
- Enable `torch.profiler.profile()` to show communication time breakdown without any spyre-comms-specific code from users


#### 3.6. First-class profiler integration in spyre-comms

Integrate the flex unified profiler (or a shared profiling library extracted from it) directly into spyre-comms so that spyre-comms can emit profiling events natively:

- **Custom profiling units** — `"SpyreComms"`, `"Collective"`, `"P2P"` thread lanes in Chrome traces
- **Per-algorithm-step events** — each send/recv/compute step in a collective algorithm as its own span
- **Overlap visualization** — clearly show communication/computation overlap in pipeline algorithms

#### 3.7. New metrics beyond timing

| Metric | Description | How to capture |
|---|---|---|
| **Effective bandwidth** | Actual bytes/sec achieved per collective | `total_bytes / wall_time` at collective boundary |
| **Algorithm efficiency** | Ratio of achieved vs. theoretical bandwidth | Compare against link bandwidth from `CostEstimator` model |
| **Queue depth** | Outstanding operations in flight | Counter in `WorkSchedule` |
| **Wait time breakdown** | Time blocked on recv vs. compute vs. host | Decompose `wait()` into sub-categories |
| **Algorithm selection accuracy** | Did `CostEstimator` pick the fastest algorithm? | Compare estimated vs. actual time across algorithms |


#### 3.8. Profiling verbosity levels

Environment-variable controlled granularity:
```
SPYRE_COMMS_PROFILE=0    # Disabled (zero overhead)
SPYRE_COMMS_PROFILE=1    # Collective-level only (one span per allreduce/allgather/etc.)
SPYRE_COMMS_PROFILE=2    # Algorithm-step-level (individual sends/recvs within collective)
SPYRE_COMMS_PROFILE=3    # Full (includes host overhead, queue depths, metadata)
```

---

## 4. Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│  Application (torch-spyre / user code)                               │
├──────────────────────────────────────────────────────────────────────┤
│  spyre-comms                                                         │
│  ┌────────────────┐  ┌──────────────────┐   ┌──────────────────────┐ │
│  │ Context API    │  │ Collective Algos │   │ P2P Protocols        │ │
│  │ (allreduce,    │  │ (ring, tree,     │   │ (HDMA, Legacy)       │ │
│  │  allgather...) │  │  pipeline...)    │   │                      │ │
│  └───────┬────────┘  └────────┬─────────┘   └──────────┬───────────┘ │
│          │                    │                        │             │
│  ┌───────▼────────────────────▼────────────────────────▼───────────┐ │
│  │ WorkSchedule (operation queue)                                  │ │
│  │  - H2D, D2H, Send, Recv, Compute, Copy operations               │ │
│  │  + CommOperationMetadata attached to each operation             │ │  ◄── NEW
│  │  + PROFILER events at collective boundaries                     │ │  ◄── NEW
│  │  + AIU_TIMING at host-level boundaries                          │ │  ◄── NEW
│  └───────┬─────────────────────────────────────────────────────────┘ │
├──────────┼───────────────────────────────────────────────────────────┤
│  flex    │                                                           │
│  ┌───────▼─────────────────────────────────────────────────────────┐ │
│  │ RuntimeStream API                                               │ │
│  │  launchOperationH2D / D2H / P2PRdmaSend / Compute               │ │
│  │  + Reads CommOperationMetadata for profiler attributes          │ │  ◄── NEW
│  └───────┬─────────────────────────────────────────────────────────┘ │
│  ┌───────▼─────────────────────────────────────────────────────────┐ │
│  │ Unified Profiler                                                │ │
│  │  TIMING │ FLEX (Chrome trace) │ AIUPTI (PyTorch)                │ │
│  └─────────────────────────────────────────────────────────────────┘ │
├──────────────────────────────────────────────────────────────────────┤
│  SenLib / Hardware                                                   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 5. Migration Path from Current PRs

| Current | Temp Fix PRs #268 + #1455 | Target |
|---|---|---|
| Ops dont have comms info | `[CollType,Algo,Bytes]` encoded in `op_name` string | Structured `CommOperationMetadata` field on params |
| Ops dont have comms info | Regex parsing in `extractCollMetadata()` | Direct field access, no regex |
| Ops dont have comms info | Only DMA ops get annotated via `op_name` | All operation types carry metadata |
| No host-side timing | No host-side timing | `AIU_TIMING_*` at critical paths |
| No collective-level span in trace | No collective-level span in trace | `PROFILER_START/STOP` wrapping entire collective |

The current PRs serve as a proof-of-concept and can be merged as-is for immediate value (identifying collectives in traces). The structured metadata approach should be the next iteration, not a blocker for the current work.

---

## 6. Open Questions

1. **Where should the shared profiling API live?** Options: (a) in `common/` submodule (already shared), (b) as a new flex public header, or (c) as a standalone profiling library.
2. **How to handle multi-threaded profiling?** spyre-comms uses worker threads for HDMA management — profiling events from these threads need correct thread-lane assignment.

---

## 7. Success Criteria

- Running `AIU_TIMING_ENABLED=1` shows host-side breakdown of all collective operations
- Chrome trace (via `FLEX_PRINT_END_TO_END_BREAKDOWN=1`) shows collective spans with nested DMA/RDMA/compute
- PyTorch profiler (`torch.profiler.profile()`) displays spyre-comms collectives as named activities
- Bandwidth metrics for each collective are available without running separate benchmarks
- No measurable overhead when profiling is disabled (single atomic check fast path)
- Acceptable overhead (<10%) when profiling is enabled
