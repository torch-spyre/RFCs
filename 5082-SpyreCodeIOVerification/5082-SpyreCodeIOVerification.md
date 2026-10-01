# SpyreCode I/O Verification: launching with real tensors and checking the result

**Authors:**
* Dushyant Behl

## **Summary**

[RFC 4755](https://github.com/torch-spyre/RFCs/pull/43) makes a compiled
SpyreCode folder self-describing: a `launch_spec.json` records each argument's
host shape, dtype, role and device layout, so `spyre launch` can build every
tensor correctly and refuse a mismatched one.

That closes one gap but leaves the one that probably matters more, a well-formed launch
can still compute the wrong answer and we won't have a way to identify the incorrectness.

In this RFC, we propose an I/O verification framework. The launch spec already says what
tensors a kernel takes; this RFC uses that to let a caller supply *values* and a *reference*:

```bash
spyre launch -i a.pt -i b.pt -o expected.pt <folder>
```

`-i` becomes the input tensor file for the next argument in `arg_index` order.
`-o` becomes the **expected** output, which the launch result is compared
against within a tolerance. The command exits non-zero on a mismatch and prints
what differed. `spyre launch` stops being "did it run" and becomes "is it
right".

The shape-string form (`-i 10x512@fp16`) will be deprecated after this RFC patch
as their use is bypassed by the launchspec.

## **Motivation**

### What the launch spec does not do

RFC 4755 is explicit that it makes launches *well-formed, not correct*, and
names this as out of scope:

> The spec makes launches well-formed, not *correct*. `torch.ones` inputs still
> tell you nothing about numerics.

That is the scope for that RFC.

### `torch.ones` hides the bugs worth finding

Every CLI launch today uses `torch.ones` for inputs (`core.py`). Ones are the
worst possible probe for a numerical kernel:

* An `add` of ones gives `2.0` everywhere, so **any** elementwise op that
  touches both inputs looks plausible.
* A transpose bug, a stride bug or a wrong reduction axis is invisible when
  every element is identical.
* Accumulation-order and precision bugs need a spread of magnitudes to show up
  at all.

So the input values are chosen to make verification impossible, and the output
has no reference to be checked against.

### The data already exists

Nothing here needs new capture machinery. The OpSpec Lab already dumps real
tensors: `--save-inputs` writes the recorded values beside each generated script
(`capture.py`, `ArgRecord.values = tensor.cpu()`, saved as `.inputs.pt`). The
gap is that no launcher can consume them.

RFC 4755 itself anticipated this exact feature and deferred it, listing
`--input-file arg0=arg0.pt` in its Stage 3 and naming `--save-inputs` as the
answer to numerics. It is unimplemented. This RFC is that work, specified
properly, with the comparison half added.

### Why this unlocks the rest of the roadmap

Three things on the spyre-cli roadmap are blocked on having a notion of
"correct":

* **CI.** A test that asserts a launch exits 0 asserts almost nothing. A test
  that asserts a launch matches a reference is a real regression test.
* **Model-level testing.** Running BERT end to end is only meaningful if the
  output is checked.
* **Cross-platform and simulator parity.** "Same artifact, same answer on x86,
  Z, Power and the simulator" is a comparison, and needs one definition of
  agreement.

## **Proposed Implementation**

### The CLI contract

```bash
# verify: real inputs, expected output, default tolerance
spyre launch -i a.pt -i b.pt -o expected.pt <folder>

# tolerance control
spyre launch -i a.pt -o expected.pt --rtol 1e-2 --atol 1e-3 <folder>
spyre launch -i a.pt -o expected.pt --exact <folder>

# capture the device result (to make a reference, or to inspect a failure)
spyre launch -i a.pt --save-output actual.pt <folder>

# unchanged: build everything from the spec, no verification
spyre launch <folder>

# deprecated spelling, warns for one release
spyre launch --shape 10x512@fp16 --shape 10x512@fp16 <folder>
```

`-i` and `-o` are positional in `arg_index` order, consistent with how the
launch spec binds arguments. `-i` fills input and `input_output` roles in order;
`-o` fills output and `input_output` roles in order. The pool tensor, when the
spec records one, is never named by a flag — the launcher allocates it, exactly
as it does today.

### File formats: two, by extension

| Extension | Loader | Carries |
|---|---|---|
| `.pt` | `torch.load` | values, shape, dtype |
| `.bin` | raw bytes, interpreted via the launch spec | values only |

`.pt` is the primary format because the OpSpec Lab already writes it, so a
captured kernel replays with no conversion. A `.pt` file is self-describing, so
its shape and dtype are **cross-checked against the launch spec** and a
disagreement is an error, not a reinterpretation.

One wrinkle the reader must handle: the Lab does not save a bare tensor. It
writes `{"tensors": [...]}`, one entry per argument, keyed on the script path
(`capture.py`). So `.pt` loading accepts three shapes — a bare tensor, a list of
tensors, or that dict — and when a file holds several tensors, a single `-i`
consumes them in order rather than requiring one flag per argument.

`.bin` is raw little-endian element data in row-major order, with shape and
dtype taken from the spec. It exists because it is the natural interchange for a
non-PyTorch producer — a C++ harness, a simulator trace, a reference generated
on Z. Because a `.bin` carries no metadata, the only validation possible is
length: `numel * itemsize` must equal the file size exactly, and a mismatch is
refused. A `.bin` of the right length but the wrong dtype is undetectable, which
is the documented cost of the format and the reason `.pt` is preferred.

A folder with no launch spec cannot use `.bin` at all (there is nothing to
interpret the bytes with) and must use `.pt` or `--shape`.

### Comparison and tolerance

After the launch, each output argument is compared against its expected tensor:

```python
torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
```

`torch.testing.assert_close` rather than a hand-rolled `allclose`, for its
failure message: it reports the number of mismatched elements, the worst
absolute and relative difference, and the index where it occurred. That message
is most of the value of this feature, and we should not write a worse one.

Defaults are **per-dtype**, taken from the output's recorded dtype rather than
fixed globally, because a tolerance that is meaningful for fp32 is meaningless
for fp16:

| dtype | rtol | atol |
|---|---|---|
| float32 | 1.3e-6 | 1e-5 |
| float16 | 1e-3 | 1e-5 |
| bfloat16 | 1.6e-2 | 1e-5 |
| float64 | 1e-7 | 1e-7 |
| integer, bool | 0 | 0 (exact) |

These are `torch.testing`'s own per-dtype defaults, verified against the PyTorch
2.14 documentation rather than reproduced from memory. Reusing them rather than
inventing numbers means the tool agrees with what every PyTorch test in tree
already considers close, and we inherit the maintenance.

The criterion applied is the one `assert_close` documents:

```text
|actual - expected| <= atol + rtol * |expected|
```

Note the asymmetry: the relative term is scaled by the *expected* value, so
which tensor is passed as `expected` matters. The reference is always `expected`.
Where actual and expected dtypes differ, `assert_close` takes the maximum of
both tolerances — but on this path a dtype difference is already a spec
disagreement and is refused before comparison, so that rule should never be
reached.

`--exact` forces bitwise equality, for integer kernels and for deliberately
checking that a change did not perturb a result at all.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | launched, and every output matched within tolerance |
| 1 | a mismatch: well-formed launch, wrong numbers |
| 2 | could not launch: bad arguments, unreadable file, spec disagreement |

Separating 1 from 2 is what makes the tool usable in CI: "this kernel is wrong"
and "this invocation was malformed" are different failures and a script must be
able to tell them apart.

### Output

On success, one line per output plus a summary:

```text
arg 2 (output): match  (max abs diff 4.88e-04, rtol 1e-3)
1 output verified, 0 failed
```

On failure, the `assert_close` report, and the actual tensor is written to
`--save-output` if given:

```text
arg 2 (output): MISMATCH
  Tensor-likes are not close!
  Mismatched elements: 3 / 5120 (0.1%)
  Greatest absolute difference: 0.0312 at index (0, 17) (up to 1e-05 allowed)
  Greatest relative difference: 0.0156 at index (0, 17) (up to 0.001 allowed)
1 output verified, 1 failed
```

### Where the code goes

The comparison belongs beside the existing spec validation, not in the CLI:
`check_launch_spec` in `torch_spyre/execution/kernel_cache.py` already owns
"does this tensor list agree with this spec", and returns a list of
human-readable problems. This adds a sibling:

```python
def check_launch_values(spec, actual, expected, rtol=None, atol=None) -> list[str]
```

Same shape of contract — a list of problems, empty meaning agreement — so the
CLI's reporting path does not change, and the function is unit-testable with no
hardware and no compile.

Loading is also library-side, since the `.bin` reader needs the spec to
interpret bytes:

```python
def load_tensor_for_arg(path, arg, bindings) -> torch.Tensor
```

`extensions/spyre-cli` keeps its lazy-import discipline (it does not depend on
`torch-spyre` at install time), so both are imported inside the launch path as
`load_launch_spec` already is.

### Migration

`-i`/`-o` currently take shape strings. They will take file paths. This is a
breaking change to a tool that shipped recently, handled in one step:

1. **This release.** `-i`/`-o` accept a path. An argument that parses as a shape
   spec (`\d+(x\d+)*(@\w+)?`) is still accepted, with a deprecation warning
   naming `--shape`. `--shape` is added as the long-term spelling.
2. **Next release.** `-i`/`-o` accept paths only. A shape-looking argument is an
   error pointing at `--shape`.

The shape form is kept reachable, not deleted: it is genuinely useful for
"does this folder load and run at all", which needs no values.

### Staging

* **Stage 1** — `load_tensor_for_arg` and `check_launch_values` in
  `kernel_cache.py`, with per-dtype defaults and the `.pt`/`.bin` readers. Unit
  tests only, no device. This is the reviewable core.
* **Stage 2** — CLI wiring: `-i`/`-o` as paths, `--rtol`/`--atol`/`--exact`,
  `--save-output`, exit codes, the deprecation warning for shape strings.
* **Stage 3** — OpSpec Lab convergence: teach `--save-inputs` to also write the
  expected output, so one capture run produces a directly replayable
  verification case.
* **Stage 4** — a CI job that replays captured cases and asserts agreement. This
  is the stage that turns the feature into a regression net, and it depends on
  `extensions/spyre-cli` being in `_test_matrix.yaml` at all, which it is not
  today.

## **Metrics**

* A kernel that computes the wrong answer fails the CLI instead of exiting 0.
  Today: 0% caught. Target: 100% for any folder launched with a reference.
* Captured OpSpec Lab cases replayable as verification cases without manual
  conversion: from none to all.
* Number of `spyre launch` invocations in CI that assert something about values
  rather than about the exit code: from zero.
* Default tolerances match `torch.testing`'s, so a result the CLI calls close is
  one the rest of the test suite would also call close.

## **Drawbacks**

* **A reference has to come from somewhere.** This RFC specifies the comparison,
  not the production of golden data. Capturing a reference on a CPU path and
  comparing device output against it is the obvious first source, but a
  reference that is itself wrong produces confident false failures — worse than
  no check.
* **Tolerance is a judgement, not a fact.** Per-dtype defaults are a reasonable
  starting point, but a deep reduction legitimately drifts further than an
  elementwise add. A single default will be wrong for some kernels, and the
  escape hatch is a flag a user has to know to reach for.
* **A breaking CLI change** to flags that shipped in #4290. Mitigated by the
  two-step migration, but it is still churn for anyone with a script.
* **`.bin` is unsafe by construction.** Right length, wrong dtype is
  undetectable. Documented, and the reason `.pt` is the recommended format.
* **Cost is modest but not zero** — two library functions, a CLI surface, and
  per-dtype tables that must be kept in step with `torch.testing`.

## **Alternatives**

* **Print the output and let the human check.** What we do now. It does not
  scale past one tensor, cannot run in CI, and silently tolerates the
  partially-correct output that RFC 4755 showed is the common failure.
* **Verify inside the OpSpec Lab instead.** The Lab already has the values and
  could compare. But its artifact is an executable replay script for the
  compiler boundary, not a launcher for a shipped folder — and a capture run is
  a heavier prerequisite than pointing the CLI at a file. The two should
  converge (Stage 3), not merge.
* **Hash comparison instead of tolerance.** Compare a checksum of the output
  against a recorded one. Cheap and exact, and useless for floating point: any
  legitimate reassociation changes the hash, so it would fail on every valid
  optimisation.
* **`--input-file arg0=a.pt` keyed by index**, as RFC 4755 sketched. Explicit,
  but verbose, and it invents a second way to address arguments when the spec
  already fixes their order. Positional `-i` matches how the spec binds.
* **Do nothing and rely on framework-level tests.** Those catch wrong answers at
  model level, far from the kernel that caused them. The point of spyre-cli is
  to shorten that loop.

## **Prior Art**

* **`torch.testing.assert_close`** is the in-tree answer to this problem and the
  source of both the comparison and the default tolerances. Its per-dtype table
  is the precedent for not using one global number.
* **The OpSpec Lab's `--save-inputs`** proves the capture half works
  (`capture.py`): real tensor values are already recorded and saved as `.pt`
  beside a generated script.
* **RFC 4755's `check_launch_spec`** is the pattern this follows — a library
  function returning a list of human-readable problems, with the CLI only
  deciding how to print them.
* **Outside the project:** ONNX Runtime's test-data layout pairs a model with
  `input_*.pb` / `output_*.pb` files and a per-case tolerance; TensorRT's
  `polygraphy run --load-inputs --load-outputs --rtol --atol` is almost exactly
  this CLI. The convergent lesson is that per-case tolerance and a separate
  mismatch exit code are both load-bearing, not conveniences.

## **How we teach this**

The sentence to lead with: **the launch spec tells you the launch was
well-formed; this tells you the answer was right.**

`spyre launch <folder>` keeps working and keeps meaning "run it". Verification
is strictly additive: supply values and a reference and the same command starts
checking. The README gains a verification section after the launch section,
documenting `.pt` as the normal format and `.bin` as the interchange one, with
its sharp edge stated.

"Expected" is the term for the reference tensor, not "golden" — it matches
`assert_close`'s own vocabulary, which the failure messages will quote.

## **Unresolved questions**

To resolve through the RFC process:

* Should `-o` with no file mean "print the output" (today's behaviour) or be an
  error? Reusing the flag for both printing and verifying is convenient and
  slightly ambiguous.
* Are `torch.testing`'s per-dtype defaults right for Spyre's accumulation
  behaviour, or does the device need its own table? This needs measurement
  against real kernels, not a guess.
* Should a mismatch write the actual output automatically, rather than only
  under `--save-output`? It is what a user wants next in every failure case.
* `.bin` endianness: little-endian assumed. Z is big-endian, and this RFC exists
  partly to serve cross-platform comparison, so the format may need an explicit
  byte-order declaration rather than an assumption.

To resolve during implementation:

* How `input_output` (in-place) arguments verify: the input file supplies the
  starting value and the expected file the final one, but the launch mutates the
  tensor, so the "actual" is the input buffer after the launch.
* Whether `--rtol`/`--atol` should be settable per argument for a multi-output
  kernel, or whether one tolerance for the whole launch is enough.
* Where captured references live, and whether they belong in the repo at all
  given their size.

Out of scope:

* Producing golden references. This RFC consumes them.
* Any change to how kernels are compiled, scheduled or lowered.
* Performance measurement. This is a correctness feature.

## Resolution

Not yet decided; this RFC is submitted for comment.

### Level of Support

Unset — pending review.

#### Additional Context

This RFC depends on RFC 4755 landing: the spec is what makes positional `-i`/`-o`
unambiguous and is the only way to interpret a `.bin`. It is otherwise
independent of the launch spec's open stages — KTIR parity, pooled launch and
symbolic dimensions all affect how a tensor is *built*, not how two tensors are
*compared*.

Verified in the current tree while writing this:

* `extensions/spyre-cli` has no value comparison of any kind, and no
  `--input-file`; RFC 4755's Stage 3 sketch was never implemented.
* `check_launch_spec` (`kernel_cache.py`) checks count, shape, dtype and device
  layout, and contains no `allclose`, `rtol` or `atol`.
* The OpSpec Lab's `--save-inputs` writes real values via `ArgRecord.values`,
  saved as `torch.save({"tensors": [...]}, "<script>.inputs.pt")`
  (`capture.py`) — a dict, not a bare tensor.
* Inputs are `torch.ones` and outputs `torch.empty` on both CLI paths
  (`core.py`), so no launch today has either real inputs or a reference.
