# SpyreCode Launch Spec: a single input contract for launching compiled SpyreCode

**Authors:**
* Dushyant Behl
* Chander Govindarajan
* Ashok Pon Kumar

> Code references link to torch-spyre `main` at [`0adfa223`][main-commit], or,
> for code that [#5081][pr-5081] adds, to its head at [`29cbdd20`][pr-commit],
> which is [#5081][pr-5081] rebased onto that `main`. [Cases evaluated](#cases-evaluated)
> says which commit each hardware result comes from.

## **Summary**

A compiled SpyreCode folder does not say what it must be called with. Its
`spyreCodeDir/` holds `spyrecode.json` and `init_binary.bin`. `spyrecode.json`
is a job plan, and it has no host shape, no host dtype and no input/output role
for any argument. So a process that holds only the folder, such as `spyre-cli`,
has to be told what tensors to pass, and a wrong guess is not detected.

This RFC proposes a **launch spec**: a JSON file,
`spyreCodeDir/launch_spec.json`, that the compiler writes on every compile and
`spyre-cli` reads. It records:

- for each positional argument: role, host shape, host strides and offset, host
  dtype and device layout;
- the size of the pool tensor the caller must pass, if any;
- the bundle's runtime-slot descriptors.

The Python API in `spyre-cli` is the primary consumer: `load_kernelspec`,
`tensors_from_spec`, `check_tensors_against_spec` and `launch`. `spyre launch`
is a thin layer over the same calls.

The design is implemented in [#5081][pr-5081] and was evaluated on a Spyre card,
including folders that serve a declared range of sizes ([#6066][pr-6066]).
[Cases evaluated](#cases-evaluated) lists every case we tried, what happened,
and whether it shaped the design.

## **Motivation**

### Nothing checks the call against the kernel

The CLI builds tensors from what the user types and passes them to the runner
([`core.py`][m-cli-runner]) without knowing what the compiled kernel expects.
The runtime checks only the number of arguments. A transposed shape or a wrong
dtype launches and returns wrong data, so the call needs a description of the
expected arguments to check against.

Shape and dtype are not enough on their own. Some ops are compiled against a
particular packing along the sticks; a depthwise conv2d puts its channels in one
stick. A tensor with the right shape and dtype in the default arrangement is
then still the wrong input, so the description has to include the device layout
too (case [A6](#cases-evaluated)).

### A folder that serves many sizes

Since [#5917][pr-5917] and [#6066][pr-6066], one binary can serve a declared
size range: the tensor is declared with
`x.to("spyre", dynamic={0: {min, max, granularity}})`, the binary is built at
the maximum, and the size is bound at launch. The size reaches the kernel
through a runtime slot whose value the runtime reads from a tensor's shape
([`job_plan.cpp`][m-jobplan-dim]). The runner only does that when it is given
the slot descriptors ([`kernel_runner.py`][m-runner-typed]); without them every
slot is treated as a tensor address ([`kernel_runner.py`][m-runner-legacy]). A
process that holds only the folder therefore needs the descriptors from
somewhere.

Today they are persisted only as `symbol_kinds.json`
([`save_symbol_kinds`][m-save-sk]). That file is written only on the
kernel-cache path ([`async_compile.py`][m-cache-gate],
[call][m-save-sk-call]), and the cache is off unless `SPYRE_KERNEL_CACHE=1`
([`config.py`][m-cache-flag]).

### Where the information exists

The host view of each argument (its shape, dtype and strides) lives on the
buffer's [`FixedTiledLayout`][m-fixedtiled], which extends inductor's host
`FixedLayout` with the device layout. It is available while
`SpyreKernel.codegen_kernel` assigns each argument its `arg_index`
([`spyre_kernel.py`][m-codegen-kernel]). The compiler's per-argument record,
[`TensorArg`][m-tensorarg], keeps only the device half: it has `device_size`,
`device_dtype` and `stride_map`, and no host shape or dtype. After codegen the
host view is gone.

The OpSpec Lab rebuilds the same information in its own
[`ArgRecord`][m-argrecord] by observing the tensors passed to `.run()` during a
capture run. So the information is needed, and today it comes only from a
special run.

There are two emitters, SDSC and KTIR ([`ktir_emitter`][m-ktir-flag]), and both
produce the same loadable folder, so the contract has to be emitter-neutral.

## **What the folder holds today**

Extracted from a compile of `torch.compile(a + b)` at `[10, 512]` fp16 with
[#5081][pr-5081] at [`86ae782a`][pr-prev], with the kernel cache off. The folder holds
`spyreCodeDir/` and, next to it, the compile-stage `bundle.mlir` and
`sdsc_0.json`, which nothing reads at launch. [`prepare_kernel`][m-prepare-kernel]
loads `spyreCodeDir/spyrecode.json` and `spyreCodeDir/init_binary.bin`; with
[#5081][pr-5081], `spyreCodeDir/launch_spec.json` sits beside them.

### `spyrecode.json`

The file as written, with the two plans in run order, the two hex constants
shortened, and the 60 derived values cut to the first three:

```json
{
  "JobPreparationPlan": [
    {"command": "Allocate",     "properties": {"size": "76416"}},
    {"command": "InitTransfer", "properties": {"init_bin_file": "init_binary.bin",
                                               "size": "70656", "dev_ptr": "120259088512"}}
  ],
  "JobExecPlan": [
    {"command": "ComputeOnHost", "properties": {
      "ishape": ["3"], "oshape": ["128", "1", "20"], "size": "2688",
      "ihandle": "", "ohandle": "sdscProgCorrect0HostCompute",
      "hcm": {
        "vdci": {
          "dsName_": "ProgCorrectionFlit", "isMarker_": 0,
          "data_conversion_info_group_": [{"key": [], "dci": {
            "dsName_": "ProgCorrectionFlit", "isHostToSen_": 1,
            "dataformat_src_": 8, "dataformat_dst_": 8, "dcOpName_": 6,
            "pre_expand_dcsi_": [], "pre_input_shape_": [], "dcsi_": [],
            "input_shape_": [128, 20], "output_shape_": [128, 1, 20],
            "post_slice_dcsi_": [], "post_output_shape_": [],
            "input_dimwise_ea_": [], "output_dimwise_ea_": [],
            "useSpi_": 0, "usePca_": 0, "usePadVal_": 0, "inputPadVal_": 0, "useWli_": 0,
            "dmdi_": {"perTensorInfo": []},
            "ssi_": {
              "value_and_locs_": [[-123, -1, 0, 6, 0, 31], [-122, -1, 1, 6, 0, 31],
                                  [-121, -1, 2, 6, 0, 31], "... 57 more"],
              "inputSym_": [-61, -62, -63],
              "unSubstitutedFlitStartOffset_": 0}}}],
          "group_tags_": [],
          "inputSym_": [-61, -62, -63],
          "dciIdxSym_": [],
          "variableDefs_": [["V-123", -23, "V-63", "8704", "128"],
                            ["V-122", -23, "V-63", "8448", "128"],
                            ["V-121", -23, "V-63", "8192", "128"], "... 57 more"]},
        "senConstants": [
          {"senconst_": "000000000000000000000000ffff0f00000000000000000000000000ffff0f00... (5120 hex chars)"},
          {"senconst_": "2f020038360200383702003838020038390200383a0200383b0200383c020038... (256 hex chars)"}]}}},
    {"command": "DataTransfer",    "properties": {"dirn": "false", "size": "2688",
                                                  "dev_ptr": "120259084288",
                                                  "host_handle": "sdscProgCorrect0HostCompute"}},
    {"command": "ComputeOnDevice", "properties": {"job_bin_ptr": "120259088512"}}
  ]
}
```

What each part is:

| Part | What it is |
|---|---|
| `JobPreparationPlan` | runs once, when the kernel is loaded |
| `Allocate` | reserves the kernel's device memory |
| `InitTransfer` | copies `init_binary.bin` to device address `dev_ptr` |
| `JobExecPlan` | runs on every launch |
| `ComputeOnHost` | builds the program correction (`ProgCorrectionFlit`) on the host from this launch's slot values |
| `inputSym_: [-61, -62, -63]` | the three runtime slots; for this kernel, the base addresses of `a`, `b` and the output |
| `variableDefs_` | values derived from the slot values (`V-63` is a slot) |
| `ssi_.value_and_locs_` | where each derived value goes in the correction: flit index and bit range |
| `ishape`, `oshape`, `input_shape_`, `output_shape_` | the shape of the correction data, not of any tensor |
| `dataformat_src_`, `dataformat_dst_` | the correction's data format; `8` is `SENINT8` ([`DataFormats`][m-dataformats]) |
| `senConstants` | constant data that goes with the correction |
| `DataTransfer` | copies the correction from the host to the device |
| `ComputeOnDevice` | runs the program at `job_bin_ptr`, the address `InitTransfer` loaded it to |

The slot numbers and the count of derived values are not fixed for an op. This
compile split the add across 20 cores; a compile of the same add at
[`29cbdd20`][pr-commit] split it across 2, and has 6 derived values and slots
`[-7, -8, -9]`.

There is no host shape, host dtype or input/output role for any argument, and
nothing says which slot is an address and which is a size. The declared dynamic
gelu's `spyrecode.json` also has three slots (two addresses and the loop count),
and neither its symbol name `s77` nor its maximum `512` appears in it.

### `init_binary.bin`

The device program the backend compiler produced. `InitTransfer` copies it into
device memory once, when the kernel is loaded, and `ComputeOnDevice` runs it from
there on every launch. It is a binary image whose format torch-spyre does not
document. A launch does not change the file: per-launch values reach the program
through the correction that `ComputeOnHost` builds. Nothing in it describes the
kernel's arguments.

### Which file knows what

| A launcher needs to know | `spyrecode.json` | `init_binary.bin` | `launch_spec.json` |
|---|---|---|---|
| how many runtime slots there are | yes (`inputSym_`) | no | yes (`symbol_kinds`) |
| what each slot is: an address, a shard, a loop count | no | no | yes |
| which tensor each slot reads (`arg_index`, `dim_index`) | no | no | yes |
| the range a loop count may take | no | no | yes |
| host shape, host dtype, input or output | no | no | yes |
| device layout each argument must have | no | compiled in | yes |
| whether to prepend a pool tensor, and its size | no | no | yes |
| host strides and storage offset | no | no | yes |
| which graph input each argument came from | no | no | open |
| device memory to reserve, and the program to run | yes | yes | not needed |

The two files the runtime loads describe how to run the program once it has its
slot values, but not what to call it with. That is the gap the launch spec
fills.

## **Proposed Implementation**

### The launch spec

The spec [#5081][pr-5081] at [`29cbdd20`][pr-commit] wrote for the same `[10, 512]` add,
whitespace compacted:

```json
{
  "version": 1,
  "kernel_name": "sdsc_fused_add_0",
  "pool_size": 0,
  "bundle_symbolic_args": true,
  "args": [
    {"arg_index": 0, "role": "input", "shape": [10, 512], "dtype": "float16",
     "layout": {"device_size": [8, 10, 64], "stride_map": [64, 512, 1],
                "device_dtype": "SEN169_FP16", "element_arrangement": "STANDARD",
                "valid_elements": [[[0, 2], 512], [[1], 10]]},
     "stride": [512, 1], "storage_offset": 0},
    {"arg_index": 1, "role": "input", "shape": [10, 512], "dtype": "float16",
     "layout": {"device_size": [8, 10, 64], "stride_map": [64, 512, 1],
                "device_dtype": "SEN169_FP16", "element_arrangement": "STANDARD",
                "valid_elements": [[[0, 2], 512], [[1], 10]]},
     "stride": [512, 1], "storage_offset": 0},
    {"arg_index": 2, "role": "output", "shape": [10, 512], "dtype": "float16",
     "layout": {"device_size": [8, 10, 64], "stride_map": [64, 512, 1],
                "device_dtype": "SEN169_FP16", "element_arrangement": "STANDARD",
                "valid_elements": [[[0, 2], 512], [[1], 10]]},
     "stride": [512, 1], "storage_offset": 0}
  ],
  "emitter": "sdsc",
  "symbol_kinds": [
    {"kind": "kernel", "base_sym_idx": -1, "offset": 0, "arg_index": 0, "granularity": 0,
     "max_value": 0, "pytorch_sym": "", "core_idx": -1, "split_count": 0, "dim_index": -1},
    {"kind": "kernel", "base_sym_idx": -1, "offset": 0, "arg_index": 1, "granularity": 0,
     "max_value": 0, "pytorch_sym": "", "core_idx": -1, "split_count": 0, "dim_index": -1},
    {"kind": "kernel", "base_sym_idx": -1, "offset": 0, "arg_index": 2, "granularity": 0,
     "max_value": 0, "pytorch_sym": "", "core_idx": -1, "split_count": 0, "dim_index": -1}
  ]
}
```

512 columns are 8 sticks of 64 elements, with the 10 rows inside each stick,
hence `device_size [8, 10, 64]`. The three `symbol_kinds` entries describe the
three `inputSym_` slots of `spyrecode.json`: each is the base address of one
argument.

And for a declared `for_each_tile` gelu over `[s77, 64]` (max 512, tile 64), from
the same commit:

```json
{
  "version": 1,
  "kernel_name": "sdsc_fused_copy__gelu_select_slice_view_0",
  "pool_size": 0,
  "bundle_symbolic_args": true,
  "args": [
    {"arg_index": 0, "role": "input", "shape": ["s77", 64], "dtype": "float16",
     "layout": {"device_size": [1, 512, 64], "stride_map": [64, 64, 1],
                "device_dtype": "SEN169_FP16", "element_arrangement": "STANDARD",
                "valid_elements": [[[0, 2], 64], [[1], 512]]},
     "stride": [64, 1], "storage_offset": 0},
    {"arg_index": 1, "role": "output", "shape": ["s77", 64], "dtype": "float16",
     "layout": {"device_size": [1, 512, 64], "stride_map": [64, 64, 1],
                "device_dtype": "SEN169_FP16", "element_arrangement": "STANDARD",
                "valid_elements": [[[0, 2], 64], [[1], 512]]},
     "stride": [64, 1], "storage_offset": 0}
  ],
  "emitter": "sdsc",
  "symbol_kinds": [
    {"kind": "kernel", "base_sym_idx": -1, "offset": 0, "arg_index": 0, "granularity": 0,
     "max_value": 0, "pytorch_sym": "", "core_idx": -1, "split_count": 0, "dim_index": -1},
    {"kind": "kernel", "base_sym_idx": -1, "offset": 0, "arg_index": 1, "granularity": 0,
     "max_value": 0, "pytorch_sym": "", "core_idx": -1, "split_count": 0, "dim_index": -1},
    {"kind": "loop_dimension", "base_sym_idx": -1, "offset": 0, "arg_index": 0, "granularity": 64,
     "max_value": 512, "pytorch_sym": "s77", "core_idx": -1, "split_count": 0, "dim_index": 0}
  ],
  "symbols": {"s77": {}}
}
```

The third slot is the loop count, read from `tensors[0].size(0)`. The layout is
the reservation at the declared maximum, whatever size is bound at launch, so
its `valid_elements` reports 512 rows. The output's shape reads `s77` because
the producer folds the exact division ([`_fold_exact_divisions`][p-fold]).

**Required** fields are ones a launch cannot be built without; a consumer that
finds one missing refuses. **Optional** fields are for diagnostics only.

| Field | | Why | Status |
|---|---|---|---|
| `version` | required | schema evolution; a reader refuses a newer major ([`load_launch_spec`][p-load-launch-spec]) | implemented |
| `kernel_name` | required | which kernel this spec describes | implemented |
| `args[].arg_index` | required | binding is positional, so a spec whose indices are not exactly `0..n-1` is refused ([`LaunchSpec.from_dict`][p-from-dict], C5). It is the kernel's call order, not the traced function's argument order (C3) | implemented |
| `args[].role` | required | `input`, `output` or `input_output`; the bundle signature has no output category. An argument the kernel both reads and writes is `input_output` ([`_record_launch_spec_arg`][p-role-merge], C1) | implemented |
| `args[].shape` | required | host shape. Ints, or a symbol's name where the binary takes that size at launch | implemented |
| `args[].dtype` | required | host dtype, as a full torch name | implemented |
| `args[].layout` | required | the `SpyreTensorLayout` the kernel was compiled against: `device_size`, `stride_map`, `device_dtype`, `element_arrangement` (enums by name) and `valid_elements` ([`LaunchArgLayout`][p-launch-arg-layout]) | implemented |
| `args[].stride`, `args[].storage_offset` | required | host strides and offset, recorded like `shape`; a non-contiguous argument cannot be rebuilt from shape alone (B1, B2) | implemented |
| `pool_size` | required | bytes of the pool tensor the *caller* must pass; `0` when the bundle allocates its own | implemented |
| `bundle_symbolic_args` | required | spec-driven launch covers `true` only | recorded; refusing `false` not implemented |
| `symbol_kinds` | required when non-empty | the runtime-slot descriptors, one `dataclasses.asdict(`[`SymbolKind`][m-symbolkind]`)` per slot | implemented |
| `symbols` | required when a shape names a symbol | the names a caller binds; proposed content `{name: {max_value, granularity}}` from the slots | implemented as `{name: {}}`; range proposed |
| `emitter` | optional | `sdsc` or `ktir` | implemented |

Field notes:

* **Location.** The spec is written inside `spyreCodeDir/`
  ([`spec_path`][p-spec-path]), the folder the runtime loads, so that one folder
  can be handed over on its own. A reader accepts either that folder or the one
  above it.
* **The pool** is never one of the `args`. When `pool_size > 0` the launcher
  allocates it and passes it first, the way
  [`call_kernel`][m-call-kernel-pool] does. `pool_size` is resolved with the
  same condition ([`pool_allocated_by_frontend`][m-pool-frontend],
  [write site][p-pool-resolve]).
* **Layout is always explicit**, taken from the buffer's `FixedTiledLayout`
  ([`from_device_layout`][p-from-device-layout]), so no default has to be
  defined. `valid_elements` is written as `[device dims, count]` pairs, since
  JSON cannot key on a tuple. The consumer rebuilds the layout with the
  constructor that takes it ([`spyre_layout`][p-spyre-layout],
  [`module.cpp`][m-py-ctors]), and a spec without the field reads as `{}`,
  which that constructor documents as "unknown".
* **A layout the consumer cannot rebuild is refused**, never replaced by the
  default (B5): an enum name this build does not know, a missing layout, or
  values the constructor rejects ([`spyre_layout`][p-spyre-layout]).
* **Layout comparison** is the placement without size-1 device dimensions, plus
  the element arrangement ([`_layout_mismatch`][p-layout-mismatch]). A size-1
  dimension is always indexed at 0, so two layouts that differ only in them place
  every element in the same place (B6). Two layouts with the same placement can
  still hold different bytes, so the arrangement is compared too (B4).
* **Host strides** are allocated as recorded ([`_alloc`][p-alloc]) and compared
  without size-1 dimensions, whose strides torch leaves arbitrary
  ([`_stride_mismatch`][p-stride-mismatch]). `tensors_from_spec` refuses a
  nonzero `storage_offset`, since it allocates fresh storage and cannot place a
  tensor inside a larger one. A tensor with non-contiguous strides is filled by
  copying from a host tensor with the same strides (B7,
  [README][p-readme-strided]).
* **Device dtype is the contract.** `float16`, `bfloat16` and `bool` all map to
  `SEN169_FP16` ([`types_mapping.h`][m-types-map]), and one compiled folder
  serves all three (A4). A host-dtype difference gives a warning; a device-dtype
  difference is refused ([`check_tensors_against_spec`][p-check]).
* **Compatibility.** Unknown keys are ignored on read
  ([`LaunchSpec.from_dict`][p-from-dict]), so a newer minor version still
  loads.
* **No per-argument name.** Binding is by position; C3 shows what that leaves
  open.
* **Scalar operands**, such as the `2` in `x * 2`, become 0-d tensor arguments
  (`shape: []`) with their own slot.

The `symbols` range is still proposed. For the gelu above it would be
`{"s77": {"max_value": 512, "granularity": 64}}`, copied from the
`loop_dimension` slot so that it cannot disagree with it.

### Dynamic dimensions

A symbolic dimension in the traced graph does not mean the binary takes a size
at launch. There are two kinds of folder:

| kernel | does the binary take a size at launch? |
|---|---|
| declared `for_each_tile` loop ([#6066][pr-6066]) | yes: a `loop_dimension` slot ([`bundle.py`][m-loopdim-kind]) that the runtime reads from `tensors[arg_index].size(dim_index)` and accepts when `G ≤ n ≤ max` and `n % G == 0` ([`job_plan.cpp`][m-jobplan-dim]) |
| bare `mark_dynamic` on a plain op | no: the symbol is concretized to its trace value before SDSC generation ([`_concretize_for_sdsc`][m-concretize]), and the artifact is identical to a static compile (D1) |

Rules:

1. **A shape names a symbol only if a `loop_dimension` slot in this kernel binds
   it.** A loop output's extent comes out of inductor as `64*((s77//64))`. The
   producer folds `x // d` to `x / d` where the ShapeEnv proves `x % d == 0`
   ([`_fold_exact_divisions`][p-fold]), so it is recorded as `s77`. A symbol
   that no slot binds should be recorded as the int the binary was built for,
   e.g. `[10, 512]` instead of `['s7', 512]` (proposed).
2. **`symbols` lists exactly those names** with the slot's `max_value` and
   `granularity` (proposed). The declared minimum is not recorded: the kernel
   cache key includes the slot's max and tile but not the minimum
   ([`compute_specs_hash`][m-cache-key-bounds]), and the runtime does not check
   it ([`job_plan.cpp`][m-jobplan-dim]).
3. **A dimension that is still an expression after folding is refused**
   ([`resolve_shape`][p-unresolved]) rather than bound by its text.
4. **A binding must satisfy the slot's contract** before allocating (proposed).
   The CLI default is `(max_value // G) * G`. The SDK check infers `n` from the
   tensor the slot reads and checks every other argument that shares the
   symbol, which the runtime does not.
5. **Allocation is the bound shape in the recorded layout**, as the compiled
   wrapper does for its own outputs ([`wrapper.py`][m-wrapper-alloc]). Such a
   tensor is a partial view of its storage when bound below the maximum, so the
   smoke test writes its inputs through a host copy, not the device `fill_`,
   which refuses partial views ([`spyre_mem.cpp`][m-fill-partial],
   [`default_tensors`][p-default-tensors]).

### Implementation architecture

**Producer: torch-spyre, on every compile.**

```text
torch.compile -> Inductor -> SuperDSCScheduling
  SpyreKernel.codegen_kernel()
    for each kernel argument, as arg_index is assigned:
      _record_launch_spec_arg(name, tensor_arg)
        reads the buffer's FixedTiledLayout: host shape, strides, offset, dtype,
        device layout; an argument seen as both input and output -> input_output
        -> LaunchArg(arg_index, role, shape, dtype, LaunchArgLayout, stride, offset)
        an argument it cannot record is remembered
  SpyreKernel.launch_args()
    every argument, indices 0..n-1 -> [LaunchArg(...)]; otherwise []
  SuperDSCScheduling.define_kernel()
    async_compile.sdsc(name, <op specs>, pool_size=..., precomputed_key=...,
                       launch_args=[LaunchArg(...)])
    async_compile.ktir(name, <op specs>, launch_args=[...])
  SpyreAsyncCompile.sdsc()
    generate_bundle() -> symbol_kinds
    _write_launch_spec(dir, name, launch_args, symbol_kinds, caller pool size, "sdsc")
      -> spyreCodeDir/launch_spec.json
    backend compiler -> spyreCodeDir/spyrecode.json, init_binary.bin
  SpyreAsyncCompile.ktir()
    ... -> _write_launch_spec(dir, name, launch_args, [], 0, "ktir")
```

Code: [`_record_launch_spec_arg`][p-record] and its
[call in `codegen_kernel`][p-record-call]; [`launch_args`][p-launch-args];
[`define_kernel`][p-define-kernel]; [`generate_bundle`][m-generate-bundle];
[`_write_launch_spec`][p-write] and its [SDSC][p-write-sdsc] and
[KTIR][p-write-ktir] call sites; the schema in
[`kernel_launchspec.py`][p-launch-spec].

Why each piece sits where it does:

* **Recorded in `codegen_kernel`**, the last point where the host view exists
  ([`codegen_kernel`][m-codegen-kernel]); `TensorArg` keeps only the device
  half ([`TensorArg`][m-tensorarg]).
* **Passed as a `launch_args=` keyword, not as `TensorArg` fields.** The op-spec
  literal is hashed into the kernel cache key ([`compute_specs_hash`][m-compute-specs-hash]),
  its `src_code` text is the dedup key ([`define_kernel`][m-define-kernel]), and
  the `TensorArg` field set is pinned ([`_EXPECTED_TENSOR_ARG_SCHEMA`][m-tensorarg-schema]).
  New fields would change every cache key. The keyword is appended after
  `src_code` is formed, next to `main`'s `precomputed_key`, so it cannot change
  compile-time identity. The names it uses are imported by the generated
  wrapper ([`SPYRE_WRAPPER_IMPORTS`][m-wrapper-imports]).
* **Written next to `generate_bundle`**, where `symbol_kinds` exist, so the spec
  copies that list rather than deriving its own.
* **All or nothing.** If any argument could not be recorded, or the indices are
  not `0..n-1`, `launch_args` is empty and no spec is written
  ([`launch_args`][p-launch-args]). Binding is positional, so a hole would give
  every later tensor the wrong slot (C5). The reader refuses such a spec as well
  ([`LaunchSpec.from_dict`][p-from-dict]).
* **Writing never fails a compile.** A recording or writing error is logged and
  the spec is skipped ([`_write_launch_spec`][p-write]); the folder still
  launches the old way.
* **The schema is stdlib-only** ([`kernel_launchspec.py`][p-launch-spec]), so
  any process can read a spec without `torch_spyre._C`.

**Consumer: `spyre-cli`.**

```text
spyre_cli.launch_spec   (needs torch_spyre._C, because it works with live tensors)
  load_kernelspec(folder)                  -> LaunchSpec | None
  tensors_from_spec(spec, bindings)        -> uninitialised tensors in the recorded layouts
                                              and host strides, pool prepended when the
                                              caller supplies one; raises on a layout it
                                              cannot rebuild or a nonzero offset
  default_tensors(spec, tensors)           -> ones into input and input_output, copied from
                                              a host tensor with the same strides (smoke test)
  check_tensors_against_spec(spec, tensors, bindings) -> list of problems:
                                              count, pool bytes, shape, host strides and
                                              offset, device dtype, layout and arrangement
spyre_cli.core
  launch(*tensors, path)                   -> runner built with the spec's symbol_kinds
  launch_from_cli(path, -i/-o, bindings)   -> spec only: tensors_from_spec + default_tensors
                                              spec and -i/-o: check, and refuse on mismatch
                                              no spec: as before
spyre launch [-i ...] [-o ...] [--bind s=N] <folder>
```

Code: [`load_kernelspec`][p-load-kernelspec],
[`tensors_from_spec`][p-tensors-from-spec],
[`default_tensors`][p-default-tensors],
[`check_tensors_against_spec`][p-check],
[`_symbol_kinds_from_spec`][p-symbol-kinds-from-spec] and the
[runner it feeds][p-runner-call], [`launch_from_cli`][p-launch-from-cli],
[`launch`][p-launch], [`spyre launch`][p-cli].

The Python workflow requested in the [#5081][pr-5081] review:

```python
import spyre_cli

spec = spyre_cli.load_kernelspec(folder)                    # what the artifact requires
tensors = spyre_cli.tensors_from_spec(spec, {"s77": 256})   # allocated, uninitialised
tensors[0].copy_(my_input)                                  # the caller's own data (B7 if non-contiguous)
assert not spyre_cli.check_tensors_against_spec(spec, tensors, {"s77": 256})
runner = spyre_cli.launch(*tensors, path=folder)            # keep the runner alive until read back
torch.testing.assert_close(tensors[1].cpu(), expected)
```

`check_tensors_against_spec` is for tensors the caller made itself. Tensors from
`tensors_from_spec` need no check, because the spec is what they were built
from.

### Reuse by other producers (e.g. Triton)

`SpyreAsyncCompile.triton()` raises `NotImplementedError`
([`async_compile.py`][m-triton]), and `triton` is not installed in the Spyre
image. The runner does not read the spec
([`SpyreSDSCKernelRunner`][m-runner]); `spyre-cli` reads it and passes the
runner the `symbol_kinds` it already accepts. So another producer that makes a
loadable `spyreCodeDir/` needs one thing: to write a `LaunchSpec` into it
through [`kernel_launchspec.py`][p-launch-spec]. Nothing changes on the runner
side.

### Rollout

| stage | content | status |
|---|---|---|
| 1 | schema and serializer; spec emitted on every compile, SDSC and KTIR | done ([#5081][pr-5081]) |
| 2 | `spyre-cli` reads the spec: tensors built from it, shape/dtype/layout checks, `symbol_kinds` passed to the runner, every output printed by role, Python API | done ([#5081][pr-5081]) |
| 3 | dynamic dims: `--bind` and expression folding (done); slot ranges and binding checks; traced size for size-specific kernels | in progress |
| 4 | layout fidelity: `valid_elements`, host strides and offset, layout comparison without size-1 dims and with the arrangement, failing closed, `input_output`, all-or-nothing specs | done ([#5081][pr-5081]) |
| 5 | `--list`/`--kernel` for multi-kernel folders, `kernel_provenance` for profiler joins | not started |

Input files and numerical verification are not part of the launch spec: the
review on [#5081][pr-5081] asked for the Python API first, where the caller owns
the data and the comparison.

## **Cases evaluated**

Each case ran on a Spyre card unless marked otherwise, in a fresh process that
held only the folder, through `spyre-cli`'s own functions, compared against CPU
or the compiled result. Outputs were prefilled with a sentinel, so "never
written" is told apart from "written wrong". The kernel cache was off, so the
spec was the only slot descriptor on disk. † marks a case measured on [#5081][pr-5081] on
`main` @ `4283b158`; every other case was measured at [`86ae782a`][pr-prev]. B1, B2,
B4, B5, C1 and C5 changed the spec: their "before" is measured as marked, and
their "after" at [`29cbdd20`][pr-commit], where B7 was measured too. A7 and D4 were
measured again at [`29cbdd20`][pr-commit], with the same result.

**A. The basic contract**

| # | case | result | decision |
|---|---|---|---|
| A1 † | wrong argument count | the runtime refuses it | the spec checks it first, naming the expected count and the pool |
| A2 † | transposed `512x10` for a `10x512` kernel | before: exit 0, wrong data. With a spec: refused (*same extents in a different order (transposed?)*) | added |
| A3 † | `fp32` tensors for an `fp16` kernel | before: exit 0, wrong data. With a spec: refused (*expected dtype float16, got float32*) | added |
| A4 † | `bfloat16` tensors for an `fp16`-compiled folder | exact results; both are `SEN169_FP16` | compare device dtype, warn on host dtype |
| A5 | caller-supplied vs bundle-internal pool | both unit-tested ([tests][p-pool-tests]); not run on a pool-allocated bundle on the card | added |
| A6 | depthwise conv2d (channels in one stick) | from the spec: correct. Default-layout tensors of the right shape and dtype: 48305/57600 wrong. `-i/-o` with the default layout: refused | layout required and always explicit |
| A7 | the 17 `spyre-cli` example kernels ([#5131][pr-5131]) | all correct through `spyre launch` and the SDK | added |

**B. Layout fidelity**

| # | case | result | decision |
|---|---|---|---|
| B1 † | non-contiguous output (`x.t() * 2`, stride `(1, 128)`) | before: the spec-built tensor got `(256, 1)`; the check passed; 32534/32768 wrong. After: the spec records `(1, 128)`, the spec-built tensor gets it, and the launch is correct | record `stride`, `storage_offset` (added) |
| B2 † | non-contiguous input (`base.t()`, stride `(1, 256)`) | before: the spec-built tensor got `(128, 1)`; the check passed; 16300/32768 wrong. After: the spec records `(1, 256)`, and with the input written as in B7 the launch is correct | same (added) |
| B3 | `valid_elements` at `10x100` | [#6459][pr-6459] removed the constructor that took no `valid_elements`; the old call now raises `TypeError`, which the consumer turned into a default-layout fallback for every argument. Recorded and passed: the spec-built tensor's `valid_elements` is `{(0, 2): 100, (1,): 10}`, the same as `.to("spyre")`, and the launch is correct | added |
| B4 | FP8 `_scaled_mm` with `QFP8CH`/`QFP8WT` inputs, given `STANDARD` tensors with the same `device_size`/`stride_map` | before: the check passed, because the layout comparison left out the arrangement, though `SpyreTensorLayout ==` ([C++][m-stl-eq], [Python][m-py-eq]) says unequal; 3836/4096 wrong. After: refused (*expected element arrangement QFP8CH, got STANDARD*). With the spec layouts: correct | compare the arrangement (added) |
| B5 | the spec names an arrangement this build does not know | before: the consumer fell back to the default layout; the check passed; 48305/57600 wrong on the depthwise conv. After: refused (*its element arrangement '...' is not one this build knows*); the conv's own spec still launches correctly | fail closed (added) |
| B6 | `mean(dim=-1, keepdim=True)` output, `(512, 1)` | recorded as `device_size [1, 1, 512, 64]`; `torch.empty((512, 1))` gets `[1, 512, 64]`; the two place every element alike, and the launch is correct, but a raw comparison refused it | compare without size-1 device dims (added) |
| B7 | writing B2's spec-built input (stride `(1, 256)`) with `copy_` from a contiguous host tensor | 32512/32768 elements never written. From a host tensor with the same strides: all correct | the ones fill copies from a host tensor with the destination's strides, and the [README][p-readme-strided] shows the same for a caller's data; the copy itself is outside the launch spec |

**C. Completeness of the spec**

| # | case | result | decision |
|---|---|---|---|
| C1 † | in-place `x.add_(y)` | before: `x` was recorded as `output`, since one record per `arg_index` was kept and the last won ([`spyre_kernel.py`][p-last-wins]); the ones-fill skipped it: 778 (sentinel + 1) instead of 2. After: `x` is `input_output` and the ones-fill gives 2 | emit `input_output` (added) |
| C2 † | an output another op in the same kernel reads (`a = x + y; return a, a * y`) | `a` recorded as `output`, correctly | no change |
| C3 † | argument order | for C2 the generated wrapper calls `.run(arg1_1, arg0_1, ...)`: kernel arg 0 is the function's second input | open |
| C4 † | scalar operand (`x * 2`) | a 0-d tensor argument; the ones-fill makes it 1 | documented |
| C5 † | an argument the producer cannot record ([`_record_launch_spec_arg`][p-record-skip]); suspected trigger a graph constant | with freezing the compile fails first ([`pass_utils.py`][m-fixedtiled-raise]); none of the 11 kernels compiled had a hole. Before: a hand-made hole failed loudly at launch. After: it is refused at load (*does not describe every argument: arg_index values are [0, 2], expected 0..1*), and a recording failure writes no spec ([test][p-unrecorded-test]) | all or nothing (added) |

**D. Dynamic sizes**

| # | case | result | decision |
|---|---|---|---|
| D1 | bare `mark_dynamic`, plain `add`, traced at 10 rows | artifact byte-identical to a static compile †. With the recorded layout, a 20-row launch gets rows 10–19 wrong: they never reach the device | record the traced size (proposed) |
| D2 † | `mark_dynamic(min, max)` on a plain op (symbolic-SDSC route) | refused at compile ([`async_compile.py`][m-dim-symbols-refused]); a declared dim under a plain op is refused on `main` ([#5937][issue-5937]) | out of scope |
| D3 † | declared `for_each_tile` kernel, launched without slot descriptors | `Number of inputs provided (2) does not match number of inputs expected (3)` | pass `symbol_kinds` to the runner (added) |
| D4 | the same, with descriptors | correct at 512, 448, 256, 192, 128, 64 rows (1 stick) and 512, 256, 192, 128, 64 (2 sticks), through the SDK and `spyre launch --bind` | added |
| D5 | loop output's dim recorded as `64*((s77//64))` | `tensors_from_spec` cannot size it; the check reports a false problem | fold to `s77`, refuse leftovers (added) |
| D6 | binding 200 (not a multiple of 64) or 576 (above max) | the check passes; the runtime refuses 200 at launch † ([`job_plan.cpp`][m-jobplan-dim]); 576 not launched | check before allocating (proposed) |
| D7 | binding below the declared minimum (64 < 128) | correct on both kernels | do not record the minimum |
| D8 | SDK check with no binding | both arguments reported unbound | infer from the tensors (proposed) |
| D9 † | a plain buffer view instead of the reserved tensor | the check refuses it. Launched anyway: 1 stick correct, 2 sticks wrong in stick 1 | keep the layout check strict |
| D10 | `spyre launch --bind s77=256`, inputs filled with ones | [#6037][pr-6037] made the device `fill_` refuse a partial view, which a reserved tensor bound below its maximum is; every binding below 512 failed | write the ones through a host copy (added) |

## **Metrics**

* A wrong shape, dtype or layout fails with a diagnostic instead of launching:
  A2, A3, A6, B4 and B5 are refused, and B1 and B2 are built correctly from the
  spec.
* A user retypes no shapes: `spyre launch <folder>` needs no arguments.
* Declared dynamic folders launch from the folder alone at any size in the
  contract (D4).
* Every output is printed, by role.
* The same command works on SDSC and KTIR folders. The spec is written on both
  paths; no KTIR folder has been launched from its spec on the card yet.

## **Drawbacks**

* **Another artifact to keep correct.** A wrong spec gives confident wrong
  validation. It is generated from the structures the compiler already uses
  (`FixedTiledLayout`, and the `symbol_kinds` from `generate_bundle`).
  `LaunchArgLayout` lists its fields by hand ([`LaunchArgLayout`][p-launch-arg-layout]),
  so a new `SpyreTensorLayout` field is not recorded until someone adds it. B3
  was that drift: `valid_elements` became required and had to be added.
* **Kernel-cache hits do not write a spec.** A hit returns before any spec is
  written ([`async_compile.py`][p-cache-hit]), so a cached folder created before
  the spec existed keeps launching without one.
* **A spec makes a launch well-formed, not correct.** Ones inputs show that a
  folder runs, not that it computes; real data goes through the Python API.
* **Failing closed refuses some specs from a newer build.** A spec that names an
  arrangement or device dtype this build does not know is refused (B5) until
  the consumer learns the name, even where the bytes would have matched.

## **Alternatives**

* **Infer the interface from `spyrecode.json` at launch.** Rejected: it is a job
  plan with no host shape, dtype or role, and it does not say which slot is an
  address and which is a size (see [What the folder holds today](#what-the-folder-holds-today)).
  `prepare_kernel` reads nothing else.
* **Extend the OpSpec Lab to cover launching.** Rejected: a capture run is a
  heavier prerequisite than a compile, and its artifact is executable Python.
* **Rely on `symbol_kinds.json`.** Rejected: written only on the kernel-cache
  path, and it has no shape, dtype or role.
* **Add the host fields to `TensorArg`.** Rejected: it changes the kernel cache
  key and the dedup key. A separate keyword is used instead.
* **Record symbolic dims as expression strings and evaluate them in the
  consumer.** Not taken for now: the only form that reaches a declared kernel in
  `main`'s tests folds exactly, and leftovers are refused until a kernel that
  produces them is tested.
* **Recover a size-specific kernel's traced extent from `device_size`.**
  Rejected: the host-to-device dim order depends on rank, and a host dim can
  vanish into the stick dim.
* **Record the declared minimum.** Rejected: not in the cache key, and not
  checked at launch.
* **A per-argument `name`, or a `kernels` array per folder.** Rejected in the
  first review round.
* **Emit only behind a config flag.** Rejected: the spec is always written, and
  failing to write it never fails a compile.
* **Input files and numerical verification in the CLI.** Deferred in favour of
  the Python API.

## **Prior Art**

This RFC builds on the [review][pr-4077-review] of [#4077][pr-4077], which
proposed a human-authored IOSpec JSON and was closed in favour of generating the
same information from the compiler. [#4290][pr-4290] shipped the inline-string
CLI as an interim step.

Outside the project, a compiled artifact usually describes its own interface:
ONNX Runtime's `get_inputs()`/`get_outputs()` and TensorRT engine introspection
let a caller query shapes and dtypes. Spyre folders carry no such interface in
the binary, so it is written next to it.

[torch-spyre/triton#188][triton-188] handles a related case, fetching a cached
compilation, but there the user's tensors are always present. `spyre-cli` has
to rebuild the call without them.

## **How we teach this**

* The [`spyre-cli` README][p-readme]: `spyre launch <folder>`, the mismatch
  messages, kernels compiled against a specific device layout, `--bind`, and the
  Python API.
* Docstrings on `load_kernelspec`, `tensors_from_spec`,
  `check_tensors_against_spec` and `launch`, saying what each checks and what it
  does not.

## **Unresolved questions**

Resolved since the first revision:

* *Always emit, or behind a flag?* Always.
* *One spec per folder, or per `spyreCodeDir/`?* Per `spyreCodeDir/`.
* *How to serialize `SymbolKind`?* `dataclasses.asdict`, as in
  `symbol_kinds.json` ([`save_symbol_kinds`][m-save-sk]).
* *The default `layout`?* Not needed: the layout is always recorded.
* *How `--bind` interacts with symbolic dims baked into the bundle?* Only a
  `loop_dimension` slot takes a size; any other symbol is fixed at its traced
  value (D1).

Open:

* **Graph-input mapping (C3).** Should each argument record which graph input or
  output it comes from?
* **Kernel-cache hits.** Should a hit write a spec when the cached folder has
  none?
* **`bundle_symbolic_args = false`.** Recorded but not yet refused by the
  consumer.
* **Expression forms other than the exact fold**, such as a loop body that
  slices along the tiled axis. Not compiled on the card yet.
* **Two loop symbols in one kernel.** This needs additional backend compiler
  support, which will be discussed internally.
* **Arguments at a nonzero storage offset.** Recorded, and refused by
  `tensors_from_spec`, which allocates fresh storage. Whether it should instead
  allocate a larger storage and place the tensor in it is open.
* **The pool tensor's allocation.** `tensors_from_spec` allocates a plain
  `uint8` tensor; `call_kernel` gives it an explicit `SENINT8` layout
  ([`spyre_kernel.py`][m-call-kernel-pool]). Not compared on the card.
* **KTIR folders.** The spec is written on the KTIR path, but no KTIR folder has
  been launched from it on the card. KTIR has no symbolic trip counts today.

Out of scope: numerical validation, a non-PyTorch launcher that reads the spec,
and any change to how `torch.compile` lowers or schedules.

## Resolution

Not yet decided; this RFC is submitted for comment.

### Level of Support

Unset, pending review.

#### Additional Context

| PR | Title | State |
|---|---|---|
| [#4077][pr-4077] | Spyre-cli: Add direct launcher for spyrecode (incl. IOSpec JSON) | closed, unmerged |
| [#4290][pr-4290] | add direct launcher for spyrecode | merged |
| [#4060][pr-4060] | Give the OpSpec Lab a `--stage ktir` alongside `--stage bundle` | merged |
| [#5081][pr-5081] | Introduce kernel launchspec and its use in spyre-cli (this RFC's implementation) | open |
| [#5131][pr-5131] | add example kernels built at test time, with on-device tests | merged |
| [#5179][pr-5179] | `tensor.to("spyre", dynamic=...)` HBM reservation for dynamic shapes | merged |
| [#5917][pr-5917] | carry a symbolic `for_each_tile` trip count to the bundle | merged |
| [#6037][pr-6037] | reject fill_/zero_ on partial views, broadcast or sharded views | merged |
| [#6066][pr-6066] | one binary for a declared size range, built at the maximum and bound at launch | merged |
| [#6459][pr-6459] | Remove deprecated STL constructor | merged |

Changes from the first revision:

* The spec moved from beside `spyreCodeDir/` to inside it.
* `layout` is the `SpyreTensorLayout` fields, `valid_elements` included, instead
  of `stick_size`/`padding`, and is always recorded, along with the host
  `stride` and `storage_offset`.
* A layout the consumer cannot rebuild is refused, and the comparison includes
  the element arrangement.
* An argument both read and written is `input_output`.
* A spec describes every argument or is not written, and a reader refuses one
  with a gap.
* `symbol_kinds` is load-bearing for declared dynamic kernels, not only for pool
  binding.
* `symbols` names only what a `loop_dimension` slot binds, with the slot's range.
* The device dtype is the contract; the host dtype only warns.
* Host data is captured in `codegen_kernel` and passed as `launch_args=`.
* New sections: what the folder holds today, the Python API, reuse by other
  producers, dynamic dimensions, and cases evaluated.

<!-- torch-spyre PRs and issues -->
[pr-4060]: https://github.com/torch-spyre/torch-spyre/pull/4060
[pr-4077]: https://github.com/torch-spyre/torch-spyre/pull/4077
[pr-4077-review]: https://github.com/torch-spyre/torch-spyre/pull/4077#discussion_r3875873376
[pr-4290]: https://github.com/torch-spyre/torch-spyre/pull/4290
[pr-5081]: https://github.com/torch-spyre/torch-spyre/pull/5081
[pr-5131]: https://github.com/torch-spyre/torch-spyre/pull/5131
[pr-5179]: https://github.com/torch-spyre/torch-spyre/pull/5179
[pr-5917]: https://github.com/torch-spyre/torch-spyre/pull/5917
[pr-6037]: https://github.com/torch-spyre/torch-spyre/pull/6037
[pr-6066]: https://github.com/torch-spyre/torch-spyre/pull/6066
[pr-6459]: https://github.com/torch-spyre/torch-spyre/pull/6459
[issue-5937]: https://github.com/torch-spyre/torch-spyre/issues/5937
[triton-188]: https://github.com/torch-spyre/triton/pull/188

<!-- commits -->
[main-commit]: https://github.com/torch-spyre/torch-spyre/commit/0adfa223ba0fac367d021204a6d56b2ca797ffca
[pr-commit]: https://github.com/dushyantbehl/torch-spyre/commit/29cbdd20c4fd1934f2143b4d9131cc9e30b1e401
[pr-prev]: https://github.com/dushyantbehl/torch-spyre/commit/86ae782a2c7b2fa3fb868c31337f68ad408a15b9

<!-- torch-spyre main @ 0adfa223 -->
[m-argrecord]: https://github.com/torch-spyre/torch-spyre/blob/0adfa223ba0fac367d021204a6d56b2ca797ffca/tests/op_specs/capture.py#L149-L165
[m-cache-flag]: https://github.com/torch-spyre/torch-spyre/blob/0adfa223ba0fac367d021204a6d56b2ca797ffca/torch_spyre/_inductor/config.py#L570
[m-cache-gate]: https://github.com/torch-spyre/torch-spyre/blob/0adfa223ba0fac367d021204a6d56b2ca797ffca/torch_spyre/execution/async_compile.py#L651-L653
[m-cache-key-bounds]: https://github.com/torch-spyre/torch-spyre/blob/0adfa223ba0fac367d021204a6d56b2ca797ffca/torch_spyre/execution/kernel_cache.py#L409-L424
[m-call-kernel-pool]: https://github.com/torch-spyre/torch-spyre/blob/0adfa223ba0fac367d021204a6d56b2ca797ffca/torch_spyre/_inductor/spyre_kernel.py#L1760
[m-cli-runner]: https://github.com/torch-spyre/torch-spyre/blob/0adfa223ba0fac367d021204a6d56b2ca797ffca/extensions/spyre-cli/spyre_cli/core.py#L30-L31
[m-codegen-kernel]: https://github.com/torch-spyre/torch-spyre/blob/0adfa223ba0fac367d021204a6d56b2ca797ffca/torch_spyre/_inductor/spyre_kernel.py#L1705-L1708
[m-compute-specs-hash]: https://github.com/torch-spyre/torch-spyre/blob/0adfa223ba0fac367d021204a6d56b2ca797ffca/torch_spyre/execution/kernel_cache.py#L335
[m-concretize]: https://github.com/torch-spyre/torch-spyre/blob/0adfa223ba0fac367d021204a6d56b2ca797ffca/torch_spyre/_inductor/codegen/superdsc.py#L1959-L1982
[m-dataformats]: https://github.com/torch-spyre/torch-spyre/blob/0adfa223ba0fac367d021204a6d56b2ca797ffca/torch_spyre/_C.pyi#L153
[m-define-kernel]: https://github.com/torch-spyre/torch-spyre/blob/0adfa223ba0fac367d021204a6d56b2ca797ffca/torch_spyre/_inductor/scheduler.py#L654-L665
[m-dim-symbols-refused]: https://github.com/torch-spyre/torch-spyre/blob/0adfa223ba0fac367d021204a6d56b2ca797ffca/torch_spyre/execution/async_compile.py#L246
[m-fill-partial]: https://github.com/torch-spyre/torch-spyre/blob/0adfa223ba0fac367d021204a6d56b2ca797ffca/torch_spyre/csrc/spyre_mem.cpp#L2458-L2464
[m-fixedtiled]: https://github.com/torch-spyre/torch-spyre/blob/0adfa223ba0fac367d021204a6d56b2ca797ffca/torch_spyre/_inductor/ir.py#L90-L129
[m-fixedtiled-raise]: https://github.com/torch-spyre/torch-spyre/blob/0adfa223ba0fac367d021204a6d56b2ca797ffca/torch_spyre/_inductor/pass_utils.py#L199-L200
[m-generate-bundle]: https://github.com/torch-spyre/torch-spyre/blob/0adfa223ba0fac367d021204a6d56b2ca797ffca/torch_spyre/_inductor/codegen/bundle.py#L139
[m-jobplan-dim]: https://github.com/torch-spyre/torch-spyre/blob/0adfa223ba0fac367d021204a6d56b2ca797ffca/torch_spyre/csrc/job_plan.cpp#L157-L193
[m-ktir-flag]: https://github.com/torch-spyre/torch-spyre/blob/0adfa223ba0fac367d021204a6d56b2ca797ffca/torch_spyre/_inductor/config.py#L138
[m-loopdim-kind]: https://github.com/torch-spyre/torch-spyre/blob/0adfa223ba0fac367d021204a6d56b2ca797ffca/torch_spyre/_inductor/codegen/bundle.py#L426-L432
[m-pool-frontend]: https://github.com/torch-spyre/torch-spyre/blob/0adfa223ba0fac367d021204a6d56b2ca797ffca/torch_spyre/_inductor/config.py#L91-L107
[m-prepare-kernel]: https://github.com/torch-spyre/torch-spyre/blob/0adfa223ba0fac367d021204a6d56b2ca797ffca/torch_spyre/_C.pyi#L557
[m-py-ctors]: https://github.com/torch-spyre/torch-spyre/blob/0adfa223ba0fac367d021204a6d56b2ca797ffca/torch_spyre/csrc/module.cpp#L563-L575
[m-py-eq]: https://github.com/torch-spyre/torch-spyre/blob/0adfa223ba0fac367d021204a6d56b2ca797ffca/torch_spyre/csrc/module.cpp#L558
[m-runner]: https://github.com/torch-spyre/torch-spyre/blob/0adfa223ba0fac367d021204a6d56b2ca797ffca/torch_spyre/execution/kernel_runner.py#L108-L200
[m-runner-legacy]: https://github.com/torch-spyre/torch-spyre/blob/0adfa223ba0fac367d021204a6d56b2ca797ffca/torch_spyre/execution/kernel_runner.py#L194-L200
[m-runner-typed]: https://github.com/torch-spyre/torch-spyre/blob/0adfa223ba0fac367d021204a6d56b2ca797ffca/torch_spyre/execution/kernel_runner.py#L152-L165
[m-save-sk]: https://github.com/torch-spyre/torch-spyre/blob/0adfa223ba0fac367d021204a6d56b2ca797ffca/torch_spyre/execution/kernel_cache.py#L895-L902
[m-save-sk-call]: https://github.com/torch-spyre/torch-spyre/blob/0adfa223ba0fac367d021204a6d56b2ca797ffca/torch_spyre/execution/async_compile.py#L702
[m-stl-eq]: https://github.com/torch-spyre/torch-spyre/blob/0adfa223ba0fac367d021204a6d56b2ca797ffca/torch_spyre/csrc/spyre_tensor_impl.h#L298-L305
[m-symbolkind]: https://github.com/torch-spyre/torch-spyre/blob/0adfa223ba0fac367d021204a6d56b2ca797ffca/torch_spyre/_inductor/codegen/compute_ops.py#L36-L255
[m-tensorarg]: https://github.com/torch-spyre/torch-spyre/blob/0adfa223ba0fac367d021204a6d56b2ca797ffca/torch_spyre/_inductor/op_spec.py#L272-L343
[m-tensorarg-schema]: https://github.com/torch-spyre/torch-spyre/blob/0adfa223ba0fac367d021204a6d56b2ca797ffca/torch_spyre/_inductor/kernel_provenance.py#L76
[m-triton]: https://github.com/torch-spyre/torch-spyre/blob/0adfa223ba0fac367d021204a6d56b2ca797ffca/torch_spyre/execution/async_compile.py#L459-L463
[m-types-map]: https://github.com/torch-spyre/torch-spyre/blob/0adfa223ba0fac367d021204a6d56b2ca797ffca/torch_spyre/csrc/types_mapping.h#L65-L72
[m-wrapper-alloc]: https://github.com/torch-spyre/torch-spyre/blob/0adfa223ba0fac367d021204a6d56b2ca797ffca/torch_spyre/_inductor/wrapper.py#L105-L111
[m-wrapper-imports]: https://github.com/torch-spyre/torch-spyre/blob/0adfa223ba0fac367d021204a6d56b2ca797ffca/torch_spyre/_inductor/wrapper.py#L38-L45

<!-- #5081 head @ 29cbdd20 (p-last-wins: the earlier head 86ae782a, for C1's before) -->
[p-alloc]: https://github.com/dushyantbehl/torch-spyre/blob/29cbdd20c4fd1934f2143b4d9131cc9e30b1e401/extensions/spyre-cli/spyre_cli/launch_spec.py#L204-L217
[p-cache-hit]: https://github.com/dushyantbehl/torch-spyre/blob/29cbdd20c4fd1934f2143b4d9131cc9e30b1e401/torch_spyre/execution/async_compile.py#L730-L741
[p-check]: https://github.com/dushyantbehl/torch-spyre/blob/29cbdd20c4fd1934f2143b4d9131cc9e30b1e401/extensions/spyre-cli/spyre_cli/launch_spec.py#L349-L466
[p-cli]: https://github.com/dushyantbehl/torch-spyre/blob/29cbdd20c4fd1934f2143b4d9131cc9e30b1e401/extensions/spyre-cli/spyre_cli/cli.py#L24-L56
[p-default-tensors]: https://github.com/dushyantbehl/torch-spyre/blob/29cbdd20c4fd1934f2143b4d9131cc9e30b1e401/extensions/spyre-cli/spyre_cli/launch_spec.py#L172-L201
[p-define-kernel]: https://github.com/dushyantbehl/torch-spyre/blob/29cbdd20c4fd1934f2143b4d9131cc9e30b1e401/torch_spyre/_inductor/scheduler.py#L686-L702
[p-fold]: https://github.com/dushyantbehl/torch-spyre/blob/29cbdd20c4fd1934f2143b4d9131cc9e30b1e401/torch_spyre/_inductor/spyre_kernel.py#L1336-L1354
[p-from-device-layout]: https://github.com/dushyantbehl/torch-spyre/blob/29cbdd20c4fd1934f2143b4d9131cc9e30b1e401/torch_spyre/_inductor/codegen/kernel_launchspec.py#L55-L73
[p-from-dict]: https://github.com/dushyantbehl/torch-spyre/blob/29cbdd20c4fd1934f2143b4d9131cc9e30b1e401/torch_spyre/_inductor/codegen/kernel_launchspec.py#L183-L214
[p-last-wins]: https://github.com/dushyantbehl/torch-spyre/blob/86ae782a2c7b2fa3fb868c31337f68ad408a15b9/torch_spyre/_inductor/spyre_kernel.py#L1394
[p-launch]: https://github.com/dushyantbehl/torch-spyre/blob/29cbdd20c4fd1934f2143b4d9131cc9e30b1e401/extensions/spyre-cli/spyre_cli/core.py#L175-L194
[p-launch-arg-layout]: https://github.com/dushyantbehl/torch-spyre/blob/29cbdd20c4fd1934f2143b4d9131cc9e30b1e401/torch_spyre/_inductor/codegen/kernel_launchspec.py#L39-L73
[p-launch-args]: https://github.com/dushyantbehl/torch-spyre/blob/29cbdd20c4fd1934f2143b4d9131cc9e30b1e401/torch_spyre/_inductor/spyre_kernel.py#L1413-L1435
[p-launch-from-cli]: https://github.com/dushyantbehl/torch-spyre/blob/29cbdd20c4fd1934f2143b4d9131cc9e30b1e401/extensions/spyre-cli/spyre_cli/core.py#L86-L172
[p-launch-spec]: https://github.com/dushyantbehl/torch-spyre/blob/29cbdd20c4fd1934f2143b4d9131cc9e30b1e401/torch_spyre/_inductor/codegen/kernel_launchspec.py#L130-L245
[p-layout-mismatch]: https://github.com/dushyantbehl/torch-spyre/blob/29cbdd20c4fd1934f2143b4d9131cc9e30b1e401/extensions/spyre-cli/spyre_cli/launch_spec.py#L279-L323
[p-load-kernelspec]: https://github.com/dushyantbehl/torch-spyre/blob/29cbdd20c4fd1934f2143b4d9131cc9e30b1e401/extensions/spyre-cli/spyre_cli/launch_spec.py#L76-L89
[p-load-launch-spec]: https://github.com/dushyantbehl/torch-spyre/blob/29cbdd20c4fd1934f2143b4d9131cc9e30b1e401/torch_spyre/_inductor/codegen/kernel_launchspec.py#L319-L346
[p-pool-resolve]: https://github.com/dushyantbehl/torch-spyre/blob/29cbdd20c4fd1934f2143b4d9131cc9e30b1e401/torch_spyre/execution/async_compile.py#L287
[p-pool-tests]: https://github.com/dushyantbehl/torch-spyre/blob/29cbdd20c4fd1934f2143b4d9131cc9e30b1e401/tests/inductor/test_kernel_launchspec.py#L355-L361
[p-readme]: https://github.com/dushyantbehl/torch-spyre/blob/29cbdd20c4fd1934f2143b4d9131cc9e30b1e401/extensions/spyre-cli/README.md
[p-readme-strided]: https://github.com/dushyantbehl/torch-spyre/blob/29cbdd20c4fd1934f2143b4d9131cc9e30b1e401/extensions/spyre-cli/README.md#L164-L174
[p-record]: https://github.com/dushyantbehl/torch-spyre/blob/29cbdd20c4fd1934f2143b4d9131cc9e30b1e401/torch_spyre/_inductor/spyre_kernel.py#L1368-L1411
[p-record-call]: https://github.com/dushyantbehl/torch-spyre/blob/29cbdd20c4fd1934f2143b4d9131cc9e30b1e401/torch_spyre/_inductor/spyre_kernel.py#L1816
[p-record-skip]: https://github.com/dushyantbehl/torch-spyre/blob/29cbdd20c4fd1934f2143b4d9131cc9e30b1e401/torch_spyre/_inductor/spyre_kernel.py#L1385-L1410
[p-role-merge]: https://github.com/dushyantbehl/torch-spyre/blob/29cbdd20c4fd1934f2143b4d9131cc9e30b1e401/torch_spyre/_inductor/spyre_kernel.py#L1393-L1396
[p-runner-call]: https://github.com/dushyantbehl/torch-spyre/blob/29cbdd20c4fd1934f2143b4d9131cc9e30b1e401/extensions/spyre-cli/spyre_cli/core.py#L18-L37
[p-spec-path]: https://github.com/dushyantbehl/torch-spyre/blob/29cbdd20c4fd1934f2143b4d9131cc9e30b1e401/torch_spyre/_inductor/codegen/kernel_launchspec.py#L294-L295
[p-spyre-layout]: https://github.com/dushyantbehl/torch-spyre/blob/29cbdd20c4fd1934f2143b4d9131cc9e30b1e401/extensions/spyre-cli/spyre_cli/launch_spec.py#L220-L258
[p-stride-mismatch]: https://github.com/dushyantbehl/torch-spyre/blob/29cbdd20c4fd1934f2143b4d9131cc9e30b1e401/extensions/spyre-cli/spyre_cli/launch_spec.py#L326-L346
[p-symbol-kinds-from-spec]: https://github.com/dushyantbehl/torch-spyre/blob/29cbdd20c4fd1934f2143b4d9131cc9e30b1e401/extensions/spyre-cli/spyre_cli/launch_spec.py#L92-L110
[p-tensors-from-spec]: https://github.com/dushyantbehl/torch-spyre/blob/29cbdd20c4fd1934f2143b4d9131cc9e30b1e401/extensions/spyre-cli/spyre_cli/launch_spec.py#L113-L169
[p-unrecorded-test]: https://github.com/dushyantbehl/torch-spyre/blob/29cbdd20c4fd1934f2143b4d9131cc9e30b1e401/tests/inductor/test_kernel_launchspec.py#L264-L289
[p-unresolved]: https://github.com/dushyantbehl/torch-spyre/blob/29cbdd20c4fd1934f2143b4d9131cc9e30b1e401/torch_spyre/_inductor/codegen/kernel_launchspec.py#L261-L283
[p-write]: https://github.com/dushyantbehl/torch-spyre/blob/29cbdd20c4fd1934f2143b4d9131cc9e30b1e401/torch_spyre/execution/async_compile.py#L261-L300
[p-write-ktir]: https://github.com/dushyantbehl/torch-spyre/blob/29cbdd20c4fd1934f2143b4d9131cc9e30b1e401/torch_spyre/execution/async_compile.py#L898
[p-write-sdsc]: https://github.com/dushyantbehl/torch-spyre/blob/29cbdd20c4fd1934f2143b4d9131cc9e30b1e401/torch_spyre/execution/async_compile.py#L255-L256
