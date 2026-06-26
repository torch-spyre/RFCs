# aiu-smi as a torch-spyre extension package

**Authors:**

* @ogatak
* @tatsuhirochiba

---

## Summary

This RFC proposes hosting **`aiu-smi`**, a peromance monitoring tool for IBM Spyre Accelerator, inside the `torch-spyre` repository under the `extensions/` directory, as an **independently packaged and independently versioned** Python distribution.

`aiu-smi` periodically reads Spyre metric data and prints per-device telemetry (power, temperature, busy %, PT-array utilization, memory bandwidth, reserved/active/peak memory, host CPU/mem, process mapping, etc.) in text or CSV form. It is a **consumer** of the [`spyre-metrics-api`](../2676-SpyreMetricsApiExtension/2676-SpyreMetricsApiExtensionRFC.md) extension. 

Placing it under `extensions/` co-locates the aiu-smi with spyre metrics API. 
We will ship it as a separate wheel with its own release cadence.


## Motivation

Performance monitoring is essential when running AI workloads on Spyre, and the first question a user asks is simply: *"is the accelerator actually being used?"* Every major accelerator vendor ships a command-line tool to answer this like `nvidia-smi`, `rocm-smi`, and so on. Spyre needs a first-class equivalent tool to capture the performance information. 

`aiu-smi` is the tool to periodically reads Spyre metric data and prints
per-device telemetry, so users can watch power, temperature, utilization, bandwidth, and memory live during a workload. It is a
pure **consumer** of [`spyre-metrics-api`](../2676-SpyreMetricsApiExtension/2676-SpyreMetricsApiExtensionRFC.md):
it reads metrics **only** through `spyremetrics`, never by parsing raw binaries or HW counters directly, satisfying the export-control and
maintainability requirements of that API.

`aiu-smi` today already:

* Samples at a configurable interval and renders a refreshing text table or CSV.
* Surfaces device data-rate metrics (memory, PCIe, RDMA bandwidth and request rates), reserved/active/peak memory, PT-array utilization estimated from power, and host CPU/memory alongside device metrics.
* Works in both **PF and VF** modes across x86, Power, and Z. (depending on platform support)

`aiu-smi` has been already mentioned in the Spyre Profiling Toolkit RFC ([0601](https://github.com/torch-spyre/RFCs/blob/main/0601-SpyreProfilingToolkit/0601-SpyreProfilingToolkitRFC.md)); this RFC defines where the tool is hosted and how it is packaged.

### Why now?

* The `spyre-metrics-api` RFC (2676) establishes the common metric-access API; `aiu-smi` is its first consumer and should land alongside it.
* Spyre needs a vendor-standard device monitoring tool for the upcoming release.
* OpenShift monitoring integration is in progress, and Spyre operator has a plan to ship VF-mode exporter similar to `aiu-smi`. 

### Why putting it into the `extensions/` directory?

`aiu-smi` is a **leaf consumer**, not core backend. Placing it under `extensions/` alongside `spyre-metrics-api` lets us:

1. **Keep dependencies small and torch-free.** `aiu-smi` needs only `spyremetrics` and `psutil`, not `torch`, so it installs on a monitoring host that has no PyTorch and never enters the core `torch_spyre` import path.
2. **Co-locate it with its dependency.** It tracks the same metric definitions `spyre-metrics-api` exposes, so a single PR can evolve a metric and its display together.
3. **Keep an independent cadence and console script.** `aiu-smi` ships an executable entry point and is versioned and released on its own schedule, independent of both the backend and the metrics API.

## Proposed Implementation

### Repository layout

`aiu-smi` becomes the second extension alongside `spyre-metrics-api`, each a self-contained distribution. The root `torch_spyre` build excludes
`extensions/`.

```text
torch-spyre/
├── torch_spyre/                  # core backend (unchanged)
├── extensions/
│   ├── spyre-metrics-api/         # provides `spyremetrics` (separate RFC)
│   └── aiu-smi/
│       ├── pyproject.toml          
│       ├── README.md
│       ├── LICENSE                 
│       ├── aiu-smi/                
│       │   ├── __init__.py
│       │   ├── aiu_smi_main.py             # CLI entry / arg parsing / loop
│       │   ├── aiu_smi_helper.py           # MetricsManager, Snapshot, fmt
│       │   ├── metric_state_helper.py      # runtime state, pt_active, mem
│       │   └── pt_active_models.json
│       ├── configs/
│       │   └── senlib_config_aiusmi.json
│       ├── tests/
│       └── ...                             # other utils like PT Active Modeling Tools etc. 
└── ...
```


## Technical Architecture

### Core Components

1. **`aiu_smi_main`** — CLI entry point: argument parsing, device discovery,
   and the sampling/display loop.
2. **`aiu_smi_helper`** — `MetricsManager` and `Snapshot`: reads metrics through
   `spyremetrics.MetricFile`, computes per-interval values, and formats text or
   CSV output.
3. **`metric_state_helper`** — runtime state tracking, `pt_active` estimation
   from power, and reserved/active/peak memory handling.

`aiu-smi` reads metrics exclusively via `spyremetrics`; any new metric it wants
to display must first be defined in `section_types.json` in `spyre-metrics-api`.

### Displayed metrics

Device ID, timestamp, host CPU/memory, power, temperature, and `busy%` are
always shown. Additional groups are selected with `-g`:

| Group | Metrics |
|-------|---------|
| `D` (default) | device data rates — `rdmem`/`wrmem`, `rxpci`/`txpci`, `rdrdma`/`wrrdma` |
| `R` | per-second request rates (`n_*`) |
| `M` | `rsvmem` reserved memory (1.x runtime only) |
| `U` | `actmem`/`peakmem` actual & peak device memory (torch-spyre only) |
| `P` | `pt_active` estimated PT-array utilization |
| `S` | per-segment reserved-memory breakdown (1.x only) |
| `A` | all metrics |

### Example Usage

```console
$ aiu-smi -i 0,1 /tmp/metrics.%BUSID
#MetricFiles
# 0 /tmp/metrics.0000:13:00.0
# 1 /tmp/metrics.0000:12:00.0
#ID Date      Time     hostcpu hostmem    pwr  gtemp   busy    rdmem    wrmem   ...
  0 20260617  21:14:10   295.1     5.0   26.7   50.6    100   22.767    0.103   ...
  1 20260617  21:14:10   295.1     5.0   26.3   49.4    100   22.753    0.111   ...
```

Key command-line options (from the current implementation):

| Flag | Purpose |
|------|---------|
| `-g/--metric-groups` | Select metric groups (`D`/`R`/`M`/`U`/`P`/`S`/`A`). |
| `-d/--delay` | Display interval in seconds (default 1; min 0.1s on s390x/VF, 0.01s elsewhere). |
| `-i/--id` | Comma-separated device IDs to monitor (default: all). |
| `-s/--csv` | CSV output instead of the default text table. |
| `-f/--filename` | Log to a file rather than stdout. |
| `--mem-details` | Show the reserved-memory (`rsvmem`) breakdown. |
| `--idle-power`, `--llm-type` | Parameters for `pt_active` estimation from power. |
| `-v/--version` | Show version. |

Options can also be supplied via the `AIUSMI_OPTS` environment variable;
command-line flags take precedence.

### Justification for hosting it within `torch-spyre`

| Reason | Detail                                                                                                        |
|---|---------------------------------------------------------------------------------------------------------------|
| Enables ecosystem growth | A familiar SMI tool is the entry point developers reach for to understand and optimize their Spyre workloads. |
| Industry standard practice | Every major accelerator vendor ships a device SMI (NVIDIA, AMD, etc.); Spyre needs the same.                  |
| Co-location with its dependency | Lives next to `spyre-metrics-api`, so a metric and its display evolve in one PR.                              |
| Red Hat product | A first-class device monitoring tool is essential for the Spyre product story in Red Hat AI and OpenShift.    |
| `torch`-free leaf consumer | Installs on monitoring hosts without the backend, validating the `extensions/` packaging boundary.            |


## Drawbacks
* **Cross-extension dependency:** the `aiu-smi` → `spyremetrics` coupling within
  the monorepo needs careful version pinning and CI ordering.
* **multiple in-repo versions:**  packages in extensions dir must be kept distinct in docs and release notes
* **Build exclusion risk:** misconfiguring the root build could accidentally
  vendor the CLI into the core wheel.

## Alternatives
* **Separate repository:** independent governance/PyPI, but loses co-location
  with both the backend and the metrics API it tracks, and duplicates CI.

## Prior Art

* **`nvidia-smi`**, **`rocm-smi`**: device SMIs layered over a vendor metrics API; `aiu-smi` over `spyremetrics` follows the same library/CLI split.
* **Spyre Profiling Toolkit RFC (0601):** positions aiu-smi as the device-level
  layer of the profiling stack and notes the planned libaiupti evolution, where
  `aiu-smi` stays a thin presentation layer over whatever the metrics source
  exposes.

## Tracking issue

- https://github.com/torch-spyre/torch-spyre/issues/2696
