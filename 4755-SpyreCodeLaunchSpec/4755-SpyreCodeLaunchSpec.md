# SpyreCode Launch Spec: a single input contract for launching compiled SpyreCode

**Authors:**
* Dushyant Behl
* Chander Govindarajan
* Ashok Pon Kumar

## **Summary**

Launching an already-compiled SpyreCode directory today with [spyre-cli](https://github.com/torch-spyre/torch-spyre/tree/main/extensions/spyre-cli)
requires the caller to retype tensor shapes and dtypes that the compiler already knew.
Current `spyre-cli` takes [inline strings](https://github.com/torch-spyre/torch-spyre/tree/main/extensions/spyre-cli#cli) such as `-i 10x512@fp16` and builds tensors positionally.
Two mechanisms overlap in this space, and were discussed on PR #4077 as ways
to provide input to `spyre-cli`:

* **A human-authored IOSpec JSON** was proposed in #4077 but closed in favour of
  compiler-generated iospecs, so that humans do not have to write them.
* **The OpSpec Lab** (`tests/op_specs/`, and its related PR #4060)
  captures a per-kernel replay script from `torch.compile`, carrying shapes,
  layouts, and pool size.

In PR #4290 we shipped the inline-string form as an interim step.
In the PR #4077 we converged on compiler-emitted I/O metadata,
across both emitters, as the right answer.
This RFC is the design that #4077's review explicitly deferred.

We propose a `launch spec`: a single declarative JSON contract,
emitted by the compiler at build time and consumed by `spyre-cli`, that
describes every argument of every kernel in a SpyreCode folder.
We plan to design the spec so that a KTIR-emitted folder is describable in the
same schema as an SDSC-emitted one.

The problem this is primarily meant to fix, reproduced on hardware: **a wrong
shape or dtype is silently accepted and launched.** A transposed `512x10` and an
`fp32` launch of an `fp16` kernel both exited 0 and returned wrong data — the
transposed run printing `2.0` for its leading rows and `-1.3008e-03` below them.
Wrong argument *count* is already caught; shape and dtype are not, and they are
what a user is most likely to get wrong.

Two secondary problems follow from the same root cause — no machine-readable
description of what a folder expects:

1. Every consumer re-derives that description differently, or asks a human to
   retype it.
2. `spyre-cli` cannot reach symbolic-address launches, multiple outputs,
   in-place arguments, or non-`ones` input values.

## **Motivation**

### Possible (silent) mismatch hole

Inside the `spyre-cli` README we had stated the problem which is,

> 1. You need to pass the right input and output list.
> 2. If the shapes don't match, there is no error - the code will silently work.

This is because the current code creates tensors purely from
the user's input and passes them to `SpyreSDSCKernelRunner.run(*tensors)`
without checking what the compiled kernel expects.

This was reproduced on hardware. A `[10, 512]` fp16 `add` was compiled with
`torch.compile`, then launched four ways through `spyre launch` against the
resulting folder:

| Launch | Result |
|---|---|
| `-i 10x512@fp16 -i 10x512@fp16 -o 10x512@fp16` (correct) | works, prints all `2.0` |
| `-i 512x10@fp16 ...` (transposed) | **launches, exit 0, wrong data** |
| `-i 10x512@fp32 ...` (wrong dtype) | **launches, exit 0, wrong data** |
| one input instead of two (wrong count) | fails with a clear error |

The two silent cases are worse than "garbage out". The transposed launch printed
`2.0` for its leading rows and `-1.3008e-03` further down; the fp32 launch mixed
`1.6000e+01` with `-3.0316e-13`. Partially-correct output is is not correct and users
may fail to see when eyeballing them easily.

If the tensor count is wrong its already caught deeper in the stack:

```text
RuntimeError: DtException: Number of inputs provided (2) does not match
number of inputs expected (3) ... DataConvertInfoGenerate.cpp line 39
```

Even if count is enforced, shape and dtype are not making those as the two
which users can most likely get wrong.

The compiled bundle is the source of truth for argument identity and order.
Any second, human-maintained copy of that contract can only drift from it, and
requires extra work to create in the first place.
To make `spyre-cli` a useful tool for debugging compiled code, we need to fix this.

### How spyre-cli uses kernel runner

`SpyreSDSCKernelRunner.__init__`
(`torch_spyre/execution/kernel_runner.py:60-119`) accepts four parameters:

```python
def __init__(self, name, code_dir, kernel_provenance=None, symbol_kinds=None):
```

Currently `spyre-cli` passes only two (`core.py:24`):

```python
runner = SpyreSDSCKernelRunner("spyre-cli", str(path))
runner.run(*tensors)
```

We need to enable the two additional parameters because:

* **`symbol_kinds`** is the canonical symbol order returned by
  `generate_bundle()`. When it is absent, `_symbolic_args` is set to `None`
  (`kernel_runner.py:118-119`) and `run` takes the two-argument
  `launch_jobplan(self.jobplan, args)` branch (`kernel_runner.py:147-150`).
  The simple `add` above needs nothing more, so this is not a gap for every
  kernel — but a bundle whose symbol table needs `kAddress` binding (the pool
  parameter at `kernel_runner.py:96-111`, or the `kernel_slice` /
  `kernel_derived` variants at `codegen/compute_ops.py:37-53`) is not
  launchable from the CLI today.
* **`kernel_provenance`** is what populates `profiler_event_name` and drives
  `register_kernel_provenance`, so CLI launches cannot be joined to profiler
  traces.

### Multi kernel input

Currently `spyre-cli` treats the path as one kernel: it appends `/spyreCodeDir`
(`kernel_runner.py:130`) and calls `prepare_kernel` once. But a compiled output
directory for a real graph can hold several kernels, and the CLI has no way to name
one, list them, or launch them in sequence.

### Where to get this information

The OpSpec Lab's generated script proves the compiler can hand over everything
needed. `tests/op_specs/capture.py:445-462` emits, per kernel:

```python
KERNEL_NAME = "{rec.name}"
POOL_SIZE = {rec.pool_size}
BUNDLE_SYMBOLIC_ARGS = {rec.bundle_symbolic_args}
SHAPES = [ ... ]    # Host (shape, dtype) per kernel arg, in arg_index order.
LAYOUTS = [ ... ]   # Exact device layout each arg had when the real graph ran.
```

This potentially can be used as the missing contract — kernel name, per-arg host shape and dtype in
`arg_index` order, device layouts, pool size, and the symbolic-args flag.
The problem is that this is presented only by a separate script and
the contract right now is produced only by a capture run, not by an ordinary compile.

There is also a second emitter arriving: KTIR (`config.ktir_emitter`,
`TORCH_SPYRE_KTIR=1`). It produces the same kind of loadable folder, so if the
launch contract is not defined emitter-neutrally before KTIR folders become
common, we will end up with two incompatible ad-hoc input paths instead of one.

## **Proposed Implementation**

### The Launch Spec

One `launch_spec.json` per kernel, written beside that kernel's
`spyreCodeDir/`:

```json
{
  "version": 1,
  "kernel_name": "sdsc_fused_add_0",
  "pool_size": 0,
  "bundle_symbolic_args": true,
  "emitter": "sdsc",
  "symbol_kinds": [
    {"kind": "kernel", "arg_index": 0},
    {"kind": "kernel", "arg_index": 1},
    {"kind": "kernel", "arg_index": 2}
  ],
  "args": [
    {"arg_index": 0, "role": "input", "shape": [10, 512], "dtype": "float16",
     "layout": {"stick_size": [64], "padding": [0, 0]}},
    {"arg_index": 1, "role": "input", "shape": [10, 512], "dtype": "float16",
     "layout": {"stick_size": [64], "padding": [0, 0]}},
    {"arg_index": 2, "role": "output", "shape": [10, 512], "dtype": "float16",
     "layout": {"stick_size": [64], "padding": [0, 0]}}
  ]
}
```

Two classes of field. **Required** fields are what a launch cannot be constructed
without; a consumer that finds one missing should refuse rather than guess.
**Optional** fields are recorded for diagnostics and validation — a consumer may
report on them, but launch behaviour must never depend on their presence, so an
older spec that omits them still launches.

| Field | | Why |
|---|---|---|
| `version` | required | schema evolution; refuse an unknown major |
| `args[].arg_index` | required | the binding is positional — this *is* the contract |
| `args[].shape` | required | host shape; recorded nowhere in the artifacts |
| `args[].dtype` | required | host dtype; likewise absent |
| `args[].role` | required | the kernel signature has no output category |
| `kernel_name` | required | which kernel this spec describes |
| `pool_size` | required | size of the *caller-supplied* pool argument; default to `0` when the caller passes none |
| `symbol_kinds` | required *when non-empty* | needed for `kAddress` binding; absent means none |
| `bundle_symbolic_args` | required | a `false` folder is out of scope and must be refused, not launched |
| `args[].layout` | required | part of the execution contract; omitted only for the default arrangement |
| `symbols` | required *when symbolic dims exist* | see below |
| `emitter` | optional | diagnostic only |

We can provide reasonable defaults for some fields like `args[].layout`.

There is deliberately no per-argument `name`. Binding is positional by
`arg_index`, so a name is a label rather than part of the contract, and recording
one invites the impression that naming is meaningful. `kernel_name` is kept
because it identifies the spec itself.

One file per `spyreCodeDir/` rather than one per folder with a `kernels` array:
nothing in tree today produces several `spyreCodeDir`s under one logical compile,
so `spyre launch --list` can glob for sibling specs if that case ever appears.

Field notes:

* `args` is ordered by `arg_index`, the positional order in which the launcher
  passes tensors to `run()`. Note the compiled kernel makes no distinction
  between inputs and outputs in its argument list — the output occupies a normal
  argument slot — which is why `role` has to be recorded here and cannot be
  recovered at launch. The pool parameter, when the caller must supply one, is
  The pool is never listed in `args`. When `pool_size > 0`, the launcher creates
  the pool tensor itself and passes it first, before the `args` entries, the same
  way `call_kernel` does.

* `pool_size` is the size in bytes of the pool tensor the caller must pass,
  and `0` means the caller passes nothing, which is the case for SDSC bundles that
  carry their own scratchpad initialization.

* `role` is one of `input`, `output`, `input_output`, closing the in-place gap.

* `dtype` uses full torch names (`float16`), not the CLI's short forms, so the
  schema is not limited to the three-entry `dtype_mapping` at `core.py:28-32`.

* `layout` is part of the execution contract: a spec consumer constructs or
  validates each tensor against the arrangement the compiled kernel expects.
  We can omit passing the value for layout when the compiler and the consumer have
  pre-agreed a default (standard stick tiling for the recorded `shape` and `dtype`,
  no padding beyond stick rounding), **and** the producer has established that the
  kernel accepts it.

* `symbol_kinds` is serialized from the same list `generate_bundle()` returns,
  reusing the `symbol_kinds.json` encoding described below, so the CLI can
  construct the runner with its fourth parameter.

* `bundle_symbolic_args` is recorded but spec-driven launch covers the `True`
  case only, which is the only mode a production compile exercises. A
  baked-address folder is then rejected with a clear message instead of failing
  deep in the generator; those stay a debugging path through the OpSpec Lab
  script, which already handles them via `pin_bundle_symbolic_args`.

Symbolic dimensions get a `symbols` block per kernel, with `shape` entries
allowed to name a symbol instead of an integer:

```json
"symbols": {"s0": {"min": 1, "max": 256}},
"args": [{"arg_index": 0, "role": "input",
          "shape": ["s0", 512], "dtype": "float16"}]
```

Binding them at launch stays explicit: `spyre launch --bind s0=128 <folder>`.

### KTIR extension

The spec needs no KTIR-specific fields, because both emitters converge on the
same loadable artifact: a `spyreCodeDir/` holding `spyrecode.json` and
`init_binary.bin`. Whatever each emitter writes on the way there is a
compiler-stage detail that no launcher opens, so `code_dir` always names that
folder and the `args`, `role`, `shape`, `dtype`, and `symbols` fields carry over
unchanged — they describe the kernel's interface, not how it was built.

A KTIR-built folder is therefore launchable through the same `spyre launch`
command with no change to input parsing. This is the reason to define the schema
now rather than after KTIR folders become common: there is one contract to
specify, not one per emitter.

### Changes to spyre-cli

`spyre launch` can be extended to support something like below,

* Spec-driven: no shapes retyped, validated, all kernels discoverable
```bash
spyre launch <folder>
spyre launch --kernel sdsc_fused_add_0 <folder>
spyre launch --list <folder>
```
* Values beyond torch.ones
```bash
spyre launch --input-file arg0=arg0.pt <folder>
```
* Inline strings stay for now, and can be deprecated once the spec is
  emitted by default.

Resolution order in `launch_from_cli`:

1. If `launch_spec.json` exists, load it. With no `-i/-o`, build every tensor
   from the spec — the common case becomes `spyre launch <folder>`.
2. If `-i/-o` are also given, parse them with the existing
   `create_tensor_info` and **compare against the spec**; on mismatch, exit
   non-zero naming the argument, the expected shape/dtype, and the supplied
   one. This closes the silent-mismatch hole described above.
3. If no spec exists, behave exactly as today, but print a warning that shapes
   are unvalidated. Folders compiled before this RFC keep working.

The runner construction becomes, with the spec supplying what is currently
dropped:

```python
runner = SpyreSDSCKernelRunner(
    kernel.name,
    str(kernel_dir),
    symbol_kinds=kernel.symbol_kinds,   # was always None
)
```

Multiple outputs are printed per output arg using `role`, retiring the
`core.py:99-102` TODO and its warning.

One related behaviour must be preserved: outputs are created with
`torch.empty`, not `torch.ones` (`core.py:88-95`). This was a review fix in
#4077 and still holds in `main`. Pre-filling outputs with `1.0` lets a launch
that never writes its output print plausible all-ones data; `empty` surfaces
that as obvious garbage. Any spec-driven allocation path must keep it.

### Where to emit the spec?

The spec should be emitted close to the bundle
path in `_inductor/codegen/bundle.py` for `generate_bundle()`, and
`execution/async_compile.py` for the KTIR path, both writing through one shared
serializer so the two emitters cannot drift.

* **Stage 1** — schema plus serializer, written behind a config flag; a spec is
  emitted but nothing consumes it. Unblocks review of the field set against
  real folders.
* **Stage 2** — `spyre-cli` reads the spec: `--list`, `--kernel`, tensor
  construction from the spec, mismatch validation, `symbol_kinds` wired
  through. The user-visible win lands here.
* **Stage 3** — symbolic dimension binding (`--bind`), `--input-file`, and
  `input_output` role handling.
* **Stage 4** — KTIR emitter support and `kernel_provenance` plumbing for
  profiler joins.

### Precedent: `symbol_kinds.json`

This pattern — the compiler writing a small JSON file beside `spyreCodeDir/`,
read back by a process that did not compile the kernel — already exists in the
repo. `save_symbol_kinds` (`kernel_cache.py:472`) writes `symbol_kinds.json`
next to the folder, and `load_symbol_kinds(cached_dir)`
(`async_compile.py:449`) reads it cold on a cache hit. For the `add` above it is
468 bytes:

```json
[{"kind": "kernel", "base_sym_idx": -1, "offset": 0, "arg_index": 0, ...},
 {"kind": "kernel", "base_sym_idx": -1, "offset": 0, "arg_index": 1, ...},
 {"kind": "kernel", "base_sym_idx": -1, "offset": 0, "arg_index": 2, ...}]
```

Two things follow. First, the approach is proven in tree rather than novel, and
its serialization — `dataclasses.asdict` per entry with `kind` as the
discriminator — is the obvious one for the spec to reuse for `symbol_kinds`
instead of inventing a second encoding. Second, it is **not** a launch spec and
does not overlap with one: it carries no shape, dtype, role, or name, so it
cannot tell a caller what tensors to build. The two files are complementary.

It is also written only when `SPYRE_KERNEL_CACHE=1` (`config.py:334`, default
`0`), so a `spyre-cli` user does not normally have even this much. The Launch
Spec should be emitted on the ordinary compile path, not behind a cache flag.

The OpSpec Lab can be updated to emit a `launch_spec.json` alongside its replay
script, from the same `SHAPES`/`LAYOUTS`/`POOL_SIZE` record it already builds
(`capture.py:445-462`). The declarative spec and the executable script serve
different jobs — launching versus compiler-boundary replay — and neither
replaces the other.

## **Metrics**

The primary one, and the only one with a measured baseline:

* A wrong shape or dtype fails with a diagnostic instead of launching. Today a
  transposed and an fp32 launch both succeed with wrong data (0% caught); target
  is 100% for any folder carrying a spec. The four launches in the Motivation
  are the regression test.

Secondary:

* Shapes retyped by a user to launch a folder: from every invocation to zero.
* Multiple outputs printed instead of only the last (`core.py:99-102`).
* In-place arguments distinguishable from inputs and outputs.
* Same `spyre launch` invocation works against an SDSC and a KTIR folder.

## **Drawbacks**

* Maintaining another compiler artifact, a stale or wrong
  spec produces confident wrong validation, which is worse than none. Mitigated
  by generating it from the same structures `generate_bundle()` returns rather
  than a parallel derivation, and by treating a spec/kernel disagreement as a
  compiler bug with a test.
* Implementation to be done in inductor, while the CLI change is small; the real
  work is serializing `symbol_kinds`, layouts, and symbols faithfully across two
  emitters.
* The spec makes launches *well-formed*, not
  *correct*. `torch.ones` inputs still tell you nothing about numerics;
  `--input-file` and the OpSpec Lab's `--save-inputs` are the answer there.

## **Alternatives**

* **Infer the interface from `spyrecode.json` at launch.** Have the CLI parse
  the loaded artifact itself, so no new file is needed — appealing because it
  keeps one source of truth with nothing to sync. Rejected because the
  information is not there. Inspected on a pod, the `add` folder's
  `spyrecode.json` has exactly two top-level keys, `JobPreparationPlan` and
  `JobExecPlan` — the steps to execute (`Allocate`, `InitTransfer`,
  `ComputeOnHost`, `DataTransfer`, `ComputeOnDevice`) — with no `dtype`, `role`
  or `arg_index` anywhere in the file. The shapes it does carry are device-side
  and tiled: `input_shape_: [128, 32]` / `output_shape_: [128, 1, 32]` for a
  `[10, 512]` host tensor, so the host shape cannot be recovered by inverting
  them. Note this is the
  *only* artifact available to a post-compile launcher: `prepare_kernel` reads
  `spyrecode.json` and `init_binary.bin` and nothing else
  (`prepare_kernel.cpp:226-238`), which is also why the earlier compiler-stage
  artifacts are not an option here. Reconsider only if the host-side fields are
  ever added to `spyrecode.json` itself, which would make this RFC unnecessary.
* **Extend the OpSpec Lab to cover launching.** It already has the data, but a
  capture run is a heavier prerequisite than an ordinary compile, and its
  artifact is executable Python — a fine debugging vehicle, a poor input
  contract.

## **Prior Art**

We are building this RFC on the discussion inside [review](https://github.com/torch-spyre/torch-spyre/pull/4077#discussion_r3875873376) of the PR #4077.
That PR proposed a human-authored IOSpec JSON
(`spyre_cli/iospec/iospec.py`, Pydantic `TensorSpec`/`IOSpec` over
`inputs`/`outputs`) and was closed unmerged in favour of generating the same
information from the compiler.

#4290 then shipped the inline-string CLI explicitly as an interim step,
"while we wait for the Compiler derived IOSpecs".

**Outside the project**, making the compiled artifact self-describing is the
common pattern. ONNX Runtime's `get_inputs()`/`get_outputs()` and TensorRT engine
introspection both let a caller query names, shapes and dtypes rather than
restate them; TFLite signature defs show the value of naming arguments over
relying on position. The lesson taken here is that the artifact, not the user,
should be authoritative about its own interface with the caveat that Spyre
folders carry no such interface today, which is why it has to be written
alongside rather than queried from the binary.

## **How we teach this**

TBA

## **Unresolved questions**

To resolve through the RFC process:

* Should `launch_spec.json` be emitted always, or behind a config flag? Always
  is simpler to rely on; a flag avoids perturbing existing output directories.
* Is one spec per folder with a `kernels` array right, or one spec per
  `spyreCodeDir`? The array makes `--list` natural but adds a merge step when
  kernels are compiled independently and possibly concurrently.

Scope of what has been verified: the evidence above comes from one static
`[10, 512]` fp16 `add`. Pool-allocated and `kernel_slice`/`kernel_derived`
bundles, multi-kernel folders, symbolic dimensions and in-place arguments are
design in this RFC, not observation, and should be confirmed against real
folders during Stage 1.

To resolve during implementation:

* Exact serialization of the five `SymbolKind` variants (`kernel`,
  `kernel_slice`, `kernel_derived`, `kernel_derived_symbolic`, pool) described
  at `codegen/compute_ops.py:36-53`.
* The precise default `layout`, written down tightly enough that a producer can
  decide whether a given argument takes it and a consumer can reconstruct it from
  `shape` and `dtype` alone. Until that is pinned, producers should emit `layout`
  explicitly for every argument rather than rely on omission.
* Whether `input_output` aliasing needs an explicit `aliases` field, now that the
  per-argument `name` that might have expressed it implicitly is gone; a pair of
  `arg_index` values is the likely spelling.
* How `--bind` interacts with `bundle_symbolic_args` and with symbolic dims
  already baked into the bundle.

Out of scope:

* Numerical validation and reference-result checking; the spec makes launches
  well-formed, not correct.
* Teaching a standalone (non-PyTorch) launcher to read the spec instead of
  reconstructing the interface itself. Worth doing — it would give every
  mechanism one shared vocabulary — but it is out of this repo's scope and
  should be its own RFC.
* Any change to how `torch.compile` lowers or schedules.

## Resolution

Not yet decided; this RFC is submitted for comment.

### Level of Support

Unset — pending review.

#### Additional Context

Status of the related work, verified against GitHub at the time of writing:

| PR | Title | State |
|---|---|---|
| #4077 | Spyre-cli: Add direct launcher for spyrecode (incl. IOSpec JSON) | **Closed, unmerged** |
| #4290 | add direct launcher for spyrecode (successor to #4077) | **Merged** 2026-09-08 (`d102b7b3`) |
| #4060 | Give the OpSpec Lab a `--stage ktir` alongside `--stage bundle` | **Open**, `REVIEW_REQUIRED` |
| #4058 | KTIR Emitter with OpSpec Lab (tracking issue for #4060) | Open |
| #4031 | Fix the OpSpec lab's HBM pool handling for the current sdsc ABI | Merged 2026-08-26 |

Two consequences for reviewers. First, the KTIR capture path this RFC builds on
is **not yet merged**, so Stage 4 depends on #4060 merge; the schema is
emitter-neutral specifically so that dependency does not block Stages 0-3.
Second, there is no open issue tracking compiler-derived IOSpec emission
(`gh search issues --repo torch-spyre/torch-spyre "iospec"` returns nothing), so
the work #4290 was explicitly waiting on has no home yet. This RFC is intended
to become it.

The silent-mismatch behavior, the device-residency-only check in
`SpyreStream::launch`, and the unused `symbol_kinds` parameter are all
verifiable in the current tree at the file and line references cited above.

