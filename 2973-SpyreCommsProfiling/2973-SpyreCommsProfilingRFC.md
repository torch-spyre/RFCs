# RFP: Profiling Support for spyre-comms

## 1. Problem Statement

spyre-comms is the communication library for Spyre AIU devices, providing collective (allreduce, allgather, broadcast, gather, reduce, barrier) and point-to-point (send, recv, sendrecv) primitives. Currently, **there is zero runtime profiling instrumentation** inside spyre-comms itself. The `GlobalTimingProfile` infrastructure from the shared `common/` submodule is available but unused.

Flex has a mature unified profiling system with three backends (TIMING, FLEX/Chrome-trace, AIUPTI/PyTorch-profiler), but it can only see the hardware-level operations (control blocks, DMA, RDMA) that spyre-comms submits — it has no visibility into:

- Which collective is executing and which algorithm was chosen
- Total bytes involved in the collective
- Host-side time spent inside spyre-comms (algorithm selection, work schedule construction, code generation)
- Per-step breakdown within a collective algorithm (e.g., individual ring steps in a pipeline allreduce)
- Overhead of P2P protocol setup (message matching, HDMA channel management)

This makes it extremely difficult to identify performance bottlenecks in distributed workloads, compare algorithm choices, or understand the communication/computation overlap. (Scope is restricted to single node scenarios.)

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
4. **Opt-in granularity** — env-var controlled, independently selectable categories (collective, step, host, counters)
5. **Low overhead** — Profiling disabled by default; when enabled, acceptable overhead (<10%)
---

### Short Term Goals

#### 3.1. Pass collective info to flex by encoding collective metadata into operation name strings

- Spyre-comms - Encode collective metadata into operation name strings using a structured format:
```
[CollType,Algorithm,Bytes] OperationName
```
For example: `[AllReduce,AllReduce_AllGatherSum,2048] Send`

This string encoding is a deliberate **stopgap**, not the intended end state: it requires regex
parsing on the flex side and is superseded by the structured `CommOperationMetadata` interface in
§3.3. It is listed here as a short-term goal because it needs no flex API change and delivers
collective identification in traces immediately. See §5 for the migration path.

It adds a `pushCollectiveAnnotation()`/`popCollectiveAnnotation()` mechanism in `SpyreCommsContext` that prepends this metadata to all Send/Recv/DMA operation names within a collective's `Convert()` method.

- Flex - Parse this structured format in `extractCollMetadata()` and add `CollAlgo` and `CollBytes` as profiler attributes alongside the existing `CollGroup`. It should also thread `op_name` through the H2D/D2H runtime operation path so DMA transfers appear with meaningful names in traces.


#### 3.2. Add `AIU_TIMING` instrumentation to spyre-comms core paths

Instrument the following critical paths using the already-available `GlobalTimingProfile` in the short term and in `AIUPTI` in the long term:

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
    size_t total_bytes;            // Total number of bytes sent or received in this step
    int step;                      // Step within algorithm (-1 if N/A)
    int total_steps;               // Total steps in algorithm (-1 if N/A)
    uint64_t rank;                 // Source/destination rank (obtained from spyrecomm / spyreCCL backend)
    uint64_t correlation_id;       // Identifies the collective invocation this op belongs to (see 3.4)
};
```

Pass this through `DmaParams`, `P2PRdmaSendParams`, `P2PRdmaWaitParams`, etc., instead of embedding in `op_name` strings. This eliminates the regex parsing in flex's `pf_runtime_scheduler.cpp`.

**Rank tagging:** The `rank` field is populated from the rank and size held by the SpyreCCL backend and the communicator context. The collective algorithm layer already tracks the peer rank for each send and recv, so this is propagated into the metadata at the operation dispatch boundary. If no rank is involved (for e.g. in `op` or `memcpy`), the `UINT64_MAX` is passed as the rank.


#### 3.4. Add collective-level profiling events visible in flex traces

Emit `PROFILER_START`/`PROFILER_STOP` events for the collective, so the trace shows both the host-side cost of submitting the collective and the device-side operations that carry it out.
```
              t ────────────────────────────────────────────────▶
 Host lane    [enqueue]                              [wait]
 Device lane      [H2D] ──
                     [Send 0] ────
                     [Recv 0] ────
                        [Sum] ──
                           [Send 1] ────
                                    [D2H] ──
                  ├────── device duration ──────┤

 All spans above share one correlation ID (one per collective invocation).
```

This requires spyre-comms to call into the flex unified profiler API (or a thin wrapper exposed for this purpose).

##### Correlation ID

**Generated by.** spyre-comms — `SpyreCommsContext::makeCollectiveId()`, called at the top of each
collective entry point and kept in a local. It is a value, not state, so there is nothing to
reset and an exception mid-collective cannot leave a stale ID behind. It is derived from `coll_counter_`
rather than a second counter, so it cannot drift from the sync keys (which already embed that
counter) and `barrier()` — composed from raw send/recv, never reaching `Collective::Convert()`
— is covered on the same footing as the real collectives. It must be called *before*
`coll_counter_` is incremented so the ID matches the sync key built from the same value.

**Value space.** The ID is not a flat 32-bit counter. It packs two fields into one `uint32_t`:

| Field | Bits | Purpose |
|---|---|---|
| Context ID | high 8 (`<< 24`) | Concurrent contexts (communicators) cannot collide on a shared counter value |
| Invocation counter | low 24 (`& 0x00FFFFFF`) | `coll_counter_` for this context |

This caps at 256 concurrent contexts and wraps after ~16.7M collectives on one context. A wrap
only aliases two entries in a trace; a long training run will eventually hit it. This is a
diagnostic limitation, not a correctness one.

**Why 32 bits and not 64.** The value is handed to the AIUPTI backend as an activity record's
`correlation_id`, which is `uint32_t` end to end — flex's `addToQueue`, libaiupti's record, and
torch-spyre, which uses it as a map key *and* a flow-arrow id. Narrowing inside
`makeCollectiveId()` rather than casting at the call site is deliberate: a
`static_cast<uint32_t>` of a 64-bit ID whose context field lived in the high bits would silently
drop the context and reinstate exactly the cross-communicator collision this packing prevents.
torch-spyre would then draw *wrong* flow arrows rather than none.

**Propagation.** No threading through the call stack is needed for the spyre-comms events. The ID
is derived from the collectives counter, called at the top of each entry point, and kept in a
local — then passed directly to the `ProfileSpan` constructor's trailing `corr_id` argument.
Nothing between the entry point and flex needs to carry it.

This is because the correlation ID is used by the torch profiler to establish the link between
host-side and device-side events, which is only required for flex runtime launch events. The spyre-comms spans
described here are host-side only, so they need the ID for identification and grouping but not
for host↔device linking.

**Namespace.** Separate from the runtime's. Three independent counters land in the same
`uint32_t correlation_id` field:

| Layer | Minted by | Granularity |
|---|---|---|
| Collectives | `SpyreCommsContext::makeCollectiveId()` | One per collective invocation |
| Schedules | `next_work_schedule_id_` | One per `WorkSchedule` |
| flex CBs | `ResponseWorker::global_pr_batch_id_` | One per control-block batch |

The relationship is **nesting, not equality**: one collective → one or more schedules → many
flex batches. flex's ID is minted after spyre-comms has handed off and is not observable from
the collective layer, hence this outer ID with flex's batch ID nesting underneath it.

Two consequences, both currently accepted:

- **A `WORK_SCHED_START`/`WORK_SCHED_WAIT` pair does not link back to the collective that
  produced the schedule.** `makeCollectiveId()` is private to `SpyreCommsContext` and schedule
  spans are minted from a different counter. Correlating them would mean plumbing the collective
  ID through `WorkSchedule`, and no flow arrows are drawn for these cbids anyway.
- **`COLL_*_SETUP` spans carry no correlation ID at all.** They run in the `WorkScheduleInfo`
  constructor path, which happens once and is then reused across many `_applyTensor`
  invocations, each bumping `coll_counter_`. No single collective owns them, so
  `makeCollectiveId()` there would report whichever invocation happened to run last. The two
  spans do not nest at runtime.

**Collision.** Because the namespaces are separate but the field is shared, a collective ID and a
flex CB ID can hold the same numeric value. This is currently harmless: the correlation ID is
used to establish flows only for compute launch events, so a comms span never enters the
flow keyspace and cannot be joined to an unrelated flex event.

This will need to be revisited if flows are implemented for comms events as well — at that point
the two ID spaces would share a single flow keyspace and a disjoint numeric range (or an
equivalent discriminator) would be required first.

##### Protocol Event Classes

Emit the following event classes at the protocol level:

| Event Class | Protocol Role | Metric |
|---|---|---|
| SEND_DATA | Send: outbound DMA data transfer | Time (usec), Bytes, Peer |
| SIGNAL_DATA | Send: signaling instruction | Time (usec), Peer |
| SIGNAL_ACK | Send: signal acknowledgement *(Host DMA)* | Time (usec), Peer |
| WAIT_DATA | Recv: wait for inbound data *(Host DMA)* | Time (usec) |
| WAIT_ACK | Recv: wait for acknowledgement, including non-data-related ACK signals *(Host DMA)* | Time (usec) |
| MONITOR_NOTICE | Recv: wait for delivery notice *(optional — only present for Host DMA in PF mode)* | Time (usec) |
| RECV_DATA | Recv: inbound DMA data transfer | Time (usec), Bytes, Peer |
| COMPUTE | Op: local compute (e.g., sum reduction) | Time (usec) |

The concrete event IDs, their mapping to AIUPTI activity records, and the timestamp source for
each are defined in the spyre-comms code — the `CollectiveCbid` enumerators and the `ProfileSpan`
call sites — rather than duplicated here, so the two cannot drift.

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
| **Effective bandwidth** | Average of all actual bytes/sec achieved per data transfer | Details to be added once the information necessary for the bandwidth calculation is figured out |
| **Algorithm efficiency** | Ratio of achieved vs. theoretical bandwidth | Compare against link bandwidth from `CostEstimator` model |
| **Queue depth** | Outstanding operations in flight | Counter in `WorkSchedule` |
| **Wait time breakdown** | Time blocked on recv vs. compute vs. host | Decompose `wait()` into sub-categories |
| **Algorithm selection accuracy** | Did `CostEstimator` pick the fastest algorithm? | Compare estimated vs. actual time across algorithms |
| **Latency distribution** | Min, median, average, max, standard deviation, and the p50/p95/p99 percentiles per collective type | Aggregate across repeated collectives in a job |

HDR histograms and Prometheus exposition-format export are out of scope for this RFC.


#### 3.8. Profiling verbosity levels

Granularity is selected by semantic keywords rather than numeric levels. Numeric levels are opaque at the call site (`SPYRE_COMMS_PROFILE=2` says nothing about what it enables) and, more importantly, a single ordered ladder forces unrelated concerns to be enabled together.

Categories are therefore **independently selectable** and may be combined with commas:

```
SPYRE_COMMS_PROFILE=off                    # Disabled (default; single atomic check fast path)
SPYRE_COMMS_PROFILE=collective             # One span per collective (allreduce/allgather/...)
SPYRE_COMMS_PROFILE=step                   # Per-operation spans within a collective (implies collective)
SPYRE_COMMS_PROFILE=host                   # Host-side cost: algorithm selection, schedule build, bundle gen
SPYRE_COMMS_PROFILE=all                    # Every category (alias: full)

SPYRE_COMMS_PROFILE=collective,host        # Combine categories
```

**Rank selection** — per-rank traces multiply with world size, so emission can be restricted:

```
SPYRE_COMMS_PROFILE_RANKS=all              # Default: every rank emits
SPYRE_COMMS_PROFILE_RANKS=0                # Only rank 0
SPYRE_COMMS_PROFILE_RANKS=0,2              # Ranks 0 and 2
```

The precise semantics of each category — exactly which spans `step` covers, how `host` is
delimited from `collective`, and whether the categories are cumulative — are still being worked
out and will be documented in the implementing PR.

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
2. **How to handle multi-threaded profiling?** spyre-comms uses worker threads for HDMA management — profiling events from these threads need correct thread-lane assignment. **Resolved:** separate thread lanes will be assigned to events based on their thread IDs. A span must begin and end on the same thread.

---

## 7. Success Criteria

- Running `AIU_TIMING_ENABLED=1` shows host-side breakdown of all collective operations
- Chrome trace (via `FLEX_PRINT_END_TO_END_BREAKDOWN=1`) shows collective spans with nested DMA/RDMA/compute
- PyTorch profiler (`torch.profiler.profile()`) displays spyre-comms collectives as named activities
- Bandwidth metrics for each collective are available without running separate benchmarks
- No measurable overhead when profiling is disabled (single atomic check fast path)
- Acceptable overhead (<10%) when profiling is enabled
