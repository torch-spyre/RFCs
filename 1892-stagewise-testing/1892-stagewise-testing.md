# Torch Spyre Component Testing

**Authors:**
* Dushyant Behl (dushyantbehl@in.ibm.com)
* Chander Govindarajan (chandergovind@in.ibm.com)

---

## Summary

This RFC proposes a stage-wise component testing strategy for the torch-spyre front-end compiler. Instead of relying solely on end-to-end tests that require live Spyre hardware, we introduce testing seams at five major compiler stages — decompositions, FX graph passes, pre-scheduler operations, fusion, and SuperDSC generation. Each stage is tested by hooking into the real compilation pipeline at intermediate points and exiting early, without reaching the backend or device. Combined with Flex's mock device for tensor allocation, all proposed tests can run on any machine without Spyre hardware.

---

## Motivation

Today, torch-spyre tests are almost entirely end-to-end: they run a Python function through `torch.compile`, all the way down to device execution, and
compare the numerical result against CPU. This means:

Most tests are integration tests which require a live Spyre device. A failure anywhere in the pipeline (decomposition, lowering, layout, codegen, backend compiler, runtime) surfaces as a single "wrong answer" or crash with no isolation. This makes it impossible to test the front-end compiler's correctness independently from the backend compiler's correctness.

### Current State

The table below summarizes the current test coverage organized by testing group. Only `SpyreTensorLayout` has direct unit tests today.
All other stages are tested only indirectly through E2E tests, or not tested at all.

| Testing Group | Components | Direct Tests? | How Tested Today |
|---------------|------------|---------------|------------------|
| Decompositions | Spyre decomposition table | No | Indirectly via E2E op tests |
| FX Graph Passes | `insert_padding`, `replace_scalar_with_tensor`, `mm_to_bmm_pass`, `bmm_unflatten` | No | Indirectly via E2E |
| Pre-Scheduler Ops | Lowerings, `propagate_spyre_tensor_layouts`, `insert_restickify`, `core_division_planning`, `scratchpad_planning` | No | Indirectly via E2E |
| Fusion | `propagate_mutation_layout`, `spyre_fuse_nodes` | No | Indirectly via E2E |
| SuperDSC Generation | `generate_sdsc()`, SuperDSC JSON structure | No | Validated only by `dxp_standalone` |

The existing `test_tensor_layout.py` tests serve as the model for how other components should be tested: validate the functionality independently in the form of a unit test rather than rely on the e2e integration test.

---

## Proposed Implementation

Our goal is to test each compiler stage in isolation by hooking into the real compilation pipeline at intermediate points:

1. Decompositions — verify that high-level ops are correctly rewritten into lower-level ops by examining the FX graph produced by `make_fx`.
2. FX Graph Passes — verify that graph-level transformations (padding, batching, unflattening) produce correct graph structure by injecting
   test passes into the pass groups and exiting early.
3. Pre-Scheduler Operations — verify that lowering, layout propagation, core division, and restickify produce correct `ir.Operation` lists by
   hooking before/after the pre-scheduling pass group.
4. Fusion — verify that scheduler-level fusion decisions are correct by hooking into the pre/post-fusion pass groups.
5. SuperDSC Generation — verify that the final JSON output is structurally correct and stable via SDSC Validator or golden file regression as detailed below.

### Testing Principles

1. We aim to unit test specific stages of torch-spyre without going to the backend or device. Each stage should be testable directly by validating intermediate output.
2. To do this, we try to construct the most specific version of the intermediate data structure and then validate its properties at that stage. Don't test lowering by checking the final SDSC JSON — test it by checking the `ir.Operation` list.
3. We can compile a real top-level function, introduce hooks at intermediate stages of interest to examine properties and then stop the process from going to the full end-to-end flow. Rather than building elaborate mock infrastructure, use the real compiler pipeline but intercept it at the point of interest.

### Group 1: Decompositions (make_fx — no compile, no device)

#### Mechanism

Decomposition happens fully in the PyTorch eager layer, where each op is replaced by Python code running through various operations. We test this by
checking for the presence/absence of ops in a compiled FX graph.

This approach uses `torch.fx.experimental.proxy_tensor.make_fx` to do a focused compilation of the input function to the FX graph stage only —
without going below. No `torch.compile` needed, no device needed, no early-exit needed.

#### Helper

```python
from torch.fx.experimental.proxy_tensor import make_fx
from torch._decompositions import get_decompositions

def _decompose(fn, *args):
    """Run fn through make_fx with Spyre decompositions and return op targets."""
    decomps = get_decompositions([])
    decomps.update(spyre_decompositions)
    gm = make_fx(fn, decomposition_table=decomps)(*args)
    return [n.target for n in gm.graph.nodes if n.op == "call_function"]
```

#### What to test

- `layer_norm` → `exx2 + layernormscale + layernormnorm`
- `compact` → `swap + slice`
- `aten.gt` → `ge * ne`
- `aten.lt` → `le * ne`
- `aten.full` → `spyre.full`
- Ops without decompositions pass through unchanged

#### Requires hardware: No

### Group 2: FX Graph Passes (compile + inject pass + early-exit)

#### Mechanism

For stages that operate on FX graphs after `torch.compile` has begun, we use a pattern of:

1. Compile a real function with `torch.compile`
2. Inject a `test_pass` into the relevant pass group
3. Inject an `exit_fn` after the test pass to raise a custom exception `SpyreInductorEarlyExit`
4. Catch `InductorError`, check `inner_exception` to match the exit condition.

#### Infrastructure

```python
from torch._inductor.exc import InductorError

class SpyreInductorEarlyExit(Exception):
    pass

def exit_fn(graph):
    raise SpyreInductorEarlyExit("early exit")

def run_inject_test_pass(target_fn, test_pass, PassClass, args):
    PassClass.passes.insert(1, test_pass)
    PassClass.passes.insert(2, exit_fn)
    try:
        cmp = torch.compile(target_fn)
        cmp(*args)
    except InductorError as e:
        ie = e.inner_exception
        if isinstance(ie, SpyreInductorEarlyExit):
            pass  # Normal: test completed, exited before device
        else:
            raise ie  # Actual error — assertion failure or bug
    finally:
        PassClass.passes.pop(2)
        PassClass.passes.pop(1)
```

#### Pass Groups & What to test

Functions which can be tested here are `insert_padding`, `replace_scalar_with_tensor`, `mm_to_bmm_pass`, `bmm_unflatten_pass`

#### Example

```python
def test_matmul_padding(self, x: torch.Tensor, y: torch.Tensor):
    def test(x, y):
        return torch.matmul(x, y)

    def test_pass(graph: torch.fx.Graph) -> None:
        for node in list(graph.nodes):
            if node.op == "call_function" and node.target == torch.matmul:
                # Check that padding was applied
                shape = node.args[0].meta["tensor_meta"].shape
                assert shape[1] % 64 == 0, (
                    "Expected 2nd dimension of arg0 to have been padded"
                )

    run_inject_test_pass(test, test_pass, CustomPreGradPasses, [x, y])
```

#### Requires hardware: No (use Flex Mock device for just tensor allocation see below)

### Group 3: Pre-Scheduler Operations (compile + before/after hooks on ir.Operation)

#### Mechanism

The next group of stages all operate on the `list[ir.Operation]` datastructure — they are functions that take in an array of `ir.Operation`
and return an array of `ir.Operation`s. Lowering works in the context of a single op, but its effects are performed immediately before the
pre-scheduler stage and thus can be observed here.

We use the same compile + early-exit pattern as Group 2, but with before/after hook passes on `CustomPreSchedulingPasses`:

#### Infrastructure addition

Since the current code does not support dynamic passes at this level, we add two new hook pass lists:

```python
class CustomPreSchedulingPasses(CustomGraphPass):

    before_passes: List[Callable[[list[ir.Operation]], None]] = []
    after_passes: List[Callable[[list[ir.Operation]], None]] = []

    def __call__(self, operations: list[Operation]) -> None:
        for p in CustomPreSchedulingPasses.before_passes:
            p(operations)

        deadcode_elimination(operations)
        propagate_spyre_tensor_layouts(operations)
        insert_restickify(operations)
        core_division_planning(operations)
        if config.lx_planning:
            scratchpad_planning(operations)

        for p in CustomPreSchedulingPasses.after_passes:
            p(operations)
```

A test injects its assertions into `before_passes` (to inspect pre-state)
or `after_passes` (to inspect post-state), then relies on the same
early-exit pattern to stop the pipeline.

#### Stages tested & what to validate

Lowering (effects observed in the before_passes hook):
- Correct `reduction_type` per op (matmul, batchmatmul, mean, etc.)
- Correct `ranges` (output shape) and `reduction_ranges`
- `Pointwise` vs `Reduction` chosen correctly
- `SpyreReduction` carries `op_info` where needed
- Correct number and types of input nodes

Propagate Spyre Tensor Layout:
- Pointwise output layout matches input layout
- Matmul output layout derived from M/N of inputs
- Reduction output layout has reduced dimension gone or size-1
- Padding applied correctly for non-stick-aligned dimensions
- Default layouts for 1D, 2D, 3D tensors

Core Division Planning:
- `core_split(size, max_cores)` — pure math, many edge cases
- Pointwise: only stick dimension is split
- Matmul: M first, then N, K never split
- BatchMatmul: batch first, then N, then M
- Product-of-splits invariant (`math.prod(splits) == n_cores_used`)
- Single-core fallback for unsupported ops
- Boundary: dimensions that don't divide evenly → single-core

Insert Restickify:
- Restickify inserted when layout mismatch between producer and consumer
- No restickify when layouts already match

Scratchpad Planning:
- Correct scratchpad allocation for ops that require it

#### Requires hardware: No (use Flex Mock device for just tensor allocation see below)

### Group 4: Fusion Operations (compile + hooks on SchedulerNodes)

#### Mechanism

This category pertains to PreFusion and PostFusion pass groups which happen
during scheduling. Unlike the previous groups, at this stage we operate on
SchedulerNodes. The same inject + early-exit logic applies.

#### Stages tested

- `propagate_mutation_layout`
- `spyre_fuse_nodes`

#### What to validate

- Correct fusion decisions (which nodes get fused together)
- Mutation layout propagation preserves correctness
- Fused node groups respect dependency ordering

#### Requires hardware: No (use Flex Mock device for just tensor allocation see below)

### Group 5: SuperDSC Validation and Golden Sample Regression

#### Mechanism

Generate a mock op spec and use it to generate SuperDSC via the
`generate_sdsc` API. Validate the output structurally and against golden
reference files.

This tests the highest-value seam: the boundary between the front-end
compiler and the backend compiler (`dxp_standalone`). If the SuperDSC JSON
is correct, the front-end has done its job.

#### What to validate

Structural validation of the generated SuperDSC JSON will be performed using the
[SuperDSC Validation Engine](https://github.ibm.com/dushyantbehl/rfcs/blob/main/003-rfc-SDSC-validator.md) — a rule-based checker that evaluates declarative
YAML rules against SuperDSC fields. This replaces the need for hand-written validation code in tests.

Additionally, golden file regression can be used: for each supported operation
type, capture the `generate_sdsc()` output as a golden JSON file and diff on
subsequent runs to detect unintended changes:

Golden samples are checked into the repo and do not need to be regenerated on every code change. When a change intentionally alters SuperDSC output, the following helper makes it easy to regenerate and update the committed golden files:

```python
import json
import pathlib

GOLDEN_DIR = pathlib.Path(__file__).parent / "golden_sdsc"

def test_sdsc_golden(op_name, sdsc_output, tmp_path):
    golden_path = GOLDEN_DIR / f"{op_name}.json"
    actual_json = json.dumps(sdsc_output, indent=2, sort_keys=True)

    if golden_path.exists():
        expected_json = golden_path.read_text()
        assert actual_json == expected_json, (
            f"SuperDSC output changed for {op_name}. To update:\n"
            f"  cp {tmp_path / 'actual.json'} {golden_path}"
        )
    else:
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(actual_json)
        pytest.skip(f"Golden file created: {golden_path}")
```

The ops for which SDSC golden samples are to be covered can be prioritized based on the supported models.

#### Requires hardware: No (use Flex Mock device for just tensor allocation see below)

### Flex Mock Device (Running Unit Tests Without Hardware)

Groups 2–5 require tensor allocation on a Spyre device for data setup. Flex's
mock device eliminates this requirement, allowing these tests to run on any
machine without Spyre hardware.

#### Setup

Set the following environment variables to force Flex into mock mode:

```bash
AIU_WORLD_SIZE=1 \
FLEX_DEVICE=MOCK \
FLEX_COMPUTE=NULL \
FLEX_SKIP_COMPUTE=TRUE \
FLEX_MOCK_DEVICE_MEMORY_SIZE=5368709120
```

The memory size (5 GB) is an example value.

#### Prerequisites

- A small patch to Flex is needed to make `RuntimeStream` work with the mock
  device. This fix is tracked in [flex#946](https://github.ibm.com/ai-chip-toolchain/flex/pull/946).
- Tensors should be created using `torch.empty` (not `torch.rand` or similar)
  since no real computation occurs on the mock device.

#### Verification

We have manually validated this approach with a compile + early-exit test run successfully
on a pod with no Spyre device attached, proving the correctness of our approach.

---

## Metrics

- **Coverage**: Calculate and increase the code coverage in torch-spyre from these unit tests proposed above.
- **Hardware independence**: Percentage of tests that can run without Spyre hardware.
- **CI integration**: All component tests running in CI on every PR without requiring device access.

---

## Drawbacks

- **Unit testing maintenance overhead**: The early-exit and pass injection infrastructure adds test-only code paths to the compiler that must be maintained alongside the production code.
- **Golden file maintenance**: Changes to SuperDSC output format require updating golden files, which adds friction to refactoring.

---

## Alternatives

- **FakeTensorMode**: PyTorch's `FakeTensor` could eliminate the need for the Flex mock device by tracing compilation without real memory allocation. However, there are currently errors when using FakeTensor with torch compile see upstream pytorch [issue](https://github.com/pytorch/pytorch/issues/136586).
- **Full mock graph construction**: Instead of compiling real functions and intercepting, manually construct mock IR nodes for each stage. This was rejected because it requires reimplementing compiler internals in test code and diverges from the real pipeline.
- **Rely solely on E2E tests**: Continue the current approach. This was rejected because it provides no fault isolation and requires hardware for all testing.
- **Torch Spyre mock device**: The [RFC](https://ibm.ent.box.com/notes/2163908442337) proposes a mock device for executing and validating torch-spyre artifacts without hardware dependence. Our tests are orthogonal to the work as we focus on unit test framework for the torch-spyre code and our validation rules etc can be easily integrated with the torch spyre mock device if needed.

---

## Prior Art

- **PyTorch Inductor test suite**: Upstream PyTorch tests Inductor stages by capturing intermediate IR and comparing against expected patterns. Our approach follows a similar philosophy but adapted for the Spyre backend's unique compilation stages.
- **`test_tensor_layout.py`**: The existing Spyre unit test that validates `SpyreTensorLayout` by constructing inputs directly and checking output contracts. This serves as the model for the proposed approach.

---

## How we teach this

TBD

---

## Unresolved questions

- **Flex mock device upstream fix**: The `RuntimeStream` patch ([flex#946](https://github.ibm.com/ai-chip-toolchain/flex/pull/946)) needs to be merged before mock device testing can be adopted broadly.
- **FakeTensorMode viability**: Whether FakeTensor can eventually replace the mock device approach entirely remains to be validated.
- **Golden file update workflow**: The exact CI workflow for detecting and approving intentional golden file changes needs to be defined.
- **SuperDSC Validation Engine**: The rule-based YAML checker is proposed separately and needs to be reviewed by both the frontend and backend compiler teams before proceeding.

---

## Resolution

TBD

### Level of Support

TBD

### Next Steps

TBD

#### Tracking issue

TBD
