# RFC-0187: Test-Driven Development Process for Adding New OOT Device Support

**Authors:**
* @ashokponkumar

**Related RFCs/PRs:**
* [RFC-0186: Model-Centric Functional Verification for OpFunc Enablement](../0186-TestFrameworks/0186-TestFrameworks.md)
* [PR #1135: Test Suite Configuration for running upstream PyTorch tests from OOT devices](https://github.com/torch-spyre/torch-spyre/pull/1135)

---

## Summary

This RFC proposes a test-driven development (TDD) process for onboarding a new out-of-tree (OOT) device backend onto PyTorch's upstream test suite. The process uses the YAML-based test configuration framework defined in PR #1135 — specifically its `unlisted_test_mode`, per-test `mode`, and `force_xfail` controls — to implement a phased red-green-refactor cycle that takes a device from zero test coverage to full upstream compatibility.

The process defines seven phases (Phase 0–6), each with clear entry/exit criteria, and introduces the role of a **testing team** that develops tooling and collaborates with **component teams** (device backend developers) to systematically enable, triage, and promote tests.

---

## Motivation

### The problem

When a new OOT device backend is added to PyTorch, there is no structured process for validating it against PyTorch's upstream test suite. Teams face several challenges:

- **PyTorch has 500,000+ test variants** across 1,000+ test files. It's overwhelming to know where to start.
- **Only ~1% of test methods use `@ops`/`@modules`** decorators, but these expand to ~50–70% of runtime variants. The remaining 96% of test files contain non-decorated tests covering foundational behavior (tensor creation, device transfer, autograd, compilation, serialization).
- **No incremental path** — teams either run everything (and drown in failures) or cherry-pick tests (and miss coverage).
- **No shared tooling** — each device team independently figures out which tests matter, how to triage failures, and when to promote tests. This duplicates effort across backends.

### What this RFC enables

- A **phased, repeatable process** that any OOT device team can follow
- **Automatic test discovery** — developers declare supported ops/modules/dtypes, and the framework discovers relevant tests without browsing PyTorch code
- **Failure absorption** — `unlisted_test_mode: xfail` lets teams run all tests from day one without the suite turning red
- **Progressive hardening** — tests graduate from xfail → mandatory_success as the backend matures
- **Testing team collaboration** — a dedicated testing team develops tooling for failure analysis, categorization, and prioritization that all component teams benefit from
- **Living documentation** — the YAML config doubles as a machine-readable record of device capabilities at any point in time

---

## Proposed Implementation

### Prerequisites

- PyTorch source cloned and built
- Your device backend registered as a `privateuse1` backend
- Environment variables configured:
  - `TORCH_TEST_CONFIG` — path to your YAML test config file
  - `TORCH_ROOT` — path to PyTorch source
  - `TORCH_DEVICE_ROOT` — path to your device plugin repo

### Phase 0: Bootstrap — Validate the Framework Plumbing

**Goal:** Run the upstream and device test suite with zero tests executing. Confirm the config loading, env var resolution, and test collection pipeline work end-to-end.

#### Key principle: glob all test files

Use a glob pattern to cover all upstream and device test files. You do **not** need to know which files contain tests for which ops — the framework's `supported_ops` filter handles that. Tests in files that don't exercise any of your declared ops simply produce no variants.

#### Steps

1. Create a minimal YAML config and a glob over all test files in upstream and device:

    ```yaml
    test_suite_config:
      files:
        - path: ${TORCH_ROOT}/test/**/*.py
          unlisted_test_mode: skip
          tests: []
        - path: ${TORCH_DEVICE_ROOT}/tests/**/*.py
          unlisted_test_mode: skip
          tests: []
    ```

2. Set up env vars and run the suite:

    ```bash
    export TORCH_TEST_CONFIG="$PROJECT_ROOT/torch-spyre/tests/test_suite_config.yaml"
    export TORCH_ROOT="$PROJECT_ROOT/pytorch"
    export TORCH_DEVICE_ROOT="$PROJECT_ROOT/torch-spyre"

    cd $TORCH_DEVICE_ROOT/test/
    pytest -v
    ```

3. **Expected result:** All tests are collected and skipped. Zero failures, zero passes, all skipped.

#### Exit criteria

- [ ] pytest runs without errors
- [ ] All tests show as "skipped"
- [ ] `${TORCH_ROOT}` and `${TORCH_DEVICE_ROOT}` glob in `path` resolves correctly

### Phase 1: Non-Op/Module Tests — Enable Broader Test Coverage

**Goal:** Identify and enable upstream PyTorch tests that are **not** decorated with `@ops` or `@modules` but are still relevant for validating device support — such as plain functional tests, tensor operation tests, device transfer tests, compilation tests, and integration tests.

#### Why this matters

The `@ops` and `@modules` frameworks cover operator and module correctness, but a significant portion of PyTorch's upstream test suite consists of tests that don't use these decorators. These include:

- **Tensor creation and manipulation** — `torch.zeros`, `torch.cat`, `torch.reshape`, indexing, slicing
- **Device transfer semantics** — `.to()`, `.cpu()`, `.clone()`, cross-device copies
- **Autograd behavior** — gradient computation, backward pass, gradient accumulation
- **Compilation and JIT** — `torch.compile`, graph breaks, tracing behavior
- **Serialization** — `torch.save`/`torch.load` round-trips with device tensors
- **Memory management** — allocation, deallocation, device memory reporting
- **Distributed operations** — collective ops, process groups (if applicable)

These tests validate foundational device behavior that ops and modules depend on. A device that passes all `@ops` tests but fails basic tensor creation or device transfer is not production-ready.

#### Approach: Testing team enables, component teams collaborate

This phase is driven by the **testing team** rather than individual component teams. The testing team:

1. **Surveys the upstream test landscape** — Categorize non-`@ops`/`@modules` tests by functional area
2. **Identifies high-value test groups** — Prioritize tests that validate foundational device behavior required by all backends
3. **Creates test selector patterns** — Define `test_selectors` using `name_regex` and `markers` to collect these tests by category
4. **Provides starter YAML config blocks** — Give component teams ready-to-use config snippets
5. **Works with component teams** to triage failures — Some failures may indicate genuine device gaps, others may be tests that assume CPU-specific behavior

#### Steps

1. **Categorize** non-`@ops`/`@modules` upstream tests by functional area (tensor creation, device transfer, autograd, compile/JIT, serialization)
2. **Define test selector patterns** for each category:

    ```yaml
    files:
      - path: ${TORCH_ROOT}/test/**/*.py
        test_selectors:
          include:
            - name_regex:
                - "TestTensorCreation::.*"
                - "TestDeviceTransfer::.*"
                - ".*test_to_device.*"
                - ".*test_empty.*"
                - ".*test_zeros.*"
        unlisted_test_mode: xfail
    ```

3. **Component teams triage and promote** — Run with xfail, distinguish genuine device gaps from CPU-specific assumptions, promote passing tests to `mandatory_success`
4. **Iterate across categories** — Foundational first (tensor creation, device transfer), then model dependencies, then shared wins

#### Exit criteria

- [ ] Testing team has surveyed and categorized non-`@ops`/`@modules` upstream tests
- [ ] Test selector patterns are defined for each category
- [ ] Component teams have starter config blocks for high-priority categories
- [ ] Passing tests promoted to `mandatory_success` with category tags

### Phase 2: First Op — The Initial Red-Green Cycle

**Goal:** Get the first operator (`add`) passing on a single dtype (`float16`). This validates the full path from config → test collection → device execution → result reporting.

#### Key principle: devs never need to browse PyTorch code

The developer need not have to browse PyTorch tests to see which ones are relevant, but be able to select and use them based on behaviours they want. The framework automatically discovers and runs **all upstream and device tests** that exercise those features like op, dtype, module etc. There is no need to know test file names, class names, or method names upfront.

#### Steps

1. **Declare the op (RED)** — Add `add` to `supported_ops`. Use `test_selectors` with `has_ops: true` and `unlisted_test_mode: xfail`:

    ```yaml
    test_suite_config:
      files:
        - path: ${TORCH_ROOT}/test/**/*.py
          test_selectors:
            include:
              - has_ops: true
          unlisted_test_mode: xfail
        - path: ${TORCH_DEVICE_ROOT}/tests/**/*.py
          test_selectors:
            include:
              - has_ops: true
          unlisted_test_mode: xfail
      global:
        supported_dtypes:
          - name: float16
        supported_ops:
          - name: add
    ```

2. **Implement the op (GREEN)** — Implement `aten::add` for `float16` in your device runtime. Re-run and review what passes.

3. **Lock in passing tests (REFACTOR)** — Promote passing tests to `mandatory_success` so they become regression guards:

    ```yaml
    tests:
      - names:
          - TestBinaryUfuncs::test_scalar_support
          - TestBinaryUfuncs::test_contig_vs_transposed
          - TestCommon::test_compare_cpu
        mode: mandatory_success
        tags:
          - phase_2
    ```

4. **Analyze failures with testing team tooling** — The testing team provides tooling to categorize failures by root cause (missing kernel, wrong dtype, numerical mismatch), generate failure reports, and rank priorities so component teams can attack the highest-impact problems first.

#### Exit criteria

- [ ] At least one op declared in `supported_ops` — tests pass
- [ ] Passing tests promoted to `mandatory_success`
- [ ] Full upstream and device test coverage for the op is visible (via xfail output)

### Phase 3: Add Dtypes — Widen the Matrix

**Goal:** Expand dtype support for existing ops, one dtype at a time.

#### Steps

1. **Add dtype to global list (RED)** — New `(op, dtype)` variants are automatically discovered as xfail:

    ```yaml
    global:
      supported_dtypes:
        - name: float16
        - name: bfloat16
      supported_ops:
        - name: add
    ```

2. **Implement dtype support and promote (GREEN)** — Once the dtype works, promote passing tests to `mandatory_success` (same cycle as Phase 2).

#### Exit criteria

- [ ] Each supported dtype has at least one passing op
- [ ] Passing tests promoted to `mandatory_success` for each dtype

### Phase 4: Device-Specific Corner Case Tests

**Goal:** Add custom tests in your device repo that cover device-specific corner cases not exercised by upstream PyTorch tests — while using the same YAML config framework for consistency.

These cover: tensor shape/stride constraints, device-specific memory layouts, compiler-specific edge cases, precision behavior at hardware limits, multi-device or device-to-host transfer semantics, and ops that work differently under compilation vs eager mode.

#### Steps

1. **Write device-specific tests using `@ops`** — These participate in the framework's op/dtype filtering automatically
2. **Write non-`@ops` device tests** — For corner cases that don't map to a single op (device transfer, compilation behavior)
3. **Tag for selective execution** — e.g., `pytest -m device_corner_cases -v`
4. **Promote and maintain** — Same red-green-refactor cycle

#### Exit criteria

- [ ] Device-specific `@ops` tests are discovered automatically via `test_selectors`
- [ ] Non-`@ops` device tests are written and runnable
- [ ] Corner case tests are tagged for selective execution
- [ ] All device-specific tests pass with `mandatory_success`

### Phase 5: Model-Driven Op Expansion

**Goal:** Support all ops and modules required by a specific model, using tags for traceability.

#### Steps

1. **Profile the model** to get the op and module lists (e.g., `add, mul, sub, matmul, softmax` and `Linear, RMSNorm, RotaryEmbedding`)

2. **Add ops and modules to global config (RED)** — New ops/modules with `force_xfail: true`, framework auto-discovers tests

3. **Create a custom module_db for framework-specific modules** — For modules not in PyTorch's upstream `module_db` (e.g., vLLM's `RMSNorm`, `RotaryEmbedding`), create `ModuleInfo` entries following the same pattern

4. **Configure module tests in the YAML** — Use `test_selectors` with `has_modules: true` and `edits.modules` for include/exclude

5. **Implement ops and modules (GREEN)** — For each, flip `force_xfail: false`, run tests, fix failures, commit

6. **Analyze failures with testing team tooling** — Same tooling from Phase 2, extended for modules: categorize module-specific failures, distinguish op vs. module root causes, track cross-module dependencies (e.g., `Attention` uses `Linear` + `Softmax`)

7. **Graduate the model (REFACTOR)** — When all ops and modules pass: `pytest -m granite_8b -v` → all PASSED, zero xfail

#### Adding another model

When a second model shares ops/modules with the first, simply add tags. New ops or modules unique to the second model go through the same cycle.

#### Exit criteria

- [ ] All ops for the target model pass with `force_xfail: false`
- [ ] All modules for the target model pass with `force_xfail: false`
- [ ] Custom `module_db` created for framework-specific modules not in upstream `module_db`
- [ ] All model tests (ops + modules) are tagged and selectable via `pytest -m <model>`
- [ ] No `force_xfail: true` remains for ops or modules the model depends on

### Phase 6: Mature Device — Flip to Mandatory Success

**Goal:** Flip the default from "failures expected" to "failures are regressions".

#### Steps

1. **Review the xfail landscape** — Categorize remaining xfails (promote, keep, skip, or add precision overrides)
2. **Flip `unlisted_test_mode` to `mandatory_success`** — List only the exceptions (known broken, crashes). Any new upstream test that exercises your supported ops will automatically run and must pass.

#### Exit criteria

- [ ] `unlisted_test_mode: mandatory_success` for the glob
- [ ] Only genuinely broken tests are listed as `skip` or `xfail`
- [ ] New upstream tests automatically run (and must pass)

### Summary: The TDD Progression

| Phase | `unlisted_test_mode` | `force_xfail` | What it proves |
|---|---|---|---|
| **Phase 0: Bootstrap** | `skip` | N/A | Framework plumbing works |
| **Phase 1: Non-op/module tests** | `xfail` | N/A | Broader test coverage enabled, testing team + component teams collaborate |
| **Phase 2: First op** | `xfail` | N/A | Single op passes, all tests auto-discovered |
| **Phase 3: Add dtypes** | `xfail` | `false` | Op works across dtypes |
| **Phase 4: Device corner cases** | `xfail` | `false` | Device-specific edge cases covered |
| **Phase 5: Model ops + modules** | `xfail` | per-op/module | Full model op and module set passes |
| **Phase 6: Mature device** | `mandatory_success` | `false` | Everything passes, exceptions listed |

Each transition is a deliberate, testable step. The YAML config acts as both the **test specification** and the **living documentation** of your device's capabilities at any point in time.

---

## Metrics

- **Test coverage progression** — Number of tests at each status (mandatory_success, xfail, skip) per phase
- **Time to first op** — How quickly a new device backend reaches Phase 2
- **Model readiness** — Percentage of ops/modules passing for each target model
- **Xfail burn-down** — Rate at which xfail tests are promoted to mandatory_success
- **Regression rate** — Frequency of mandatory_success tests regressing after promotion

---

## Drawbacks

- **YAML config complexity** — The configuration grows as more ops, modules, and tests are added. This is mitigated by the phased approach and tooling.
- **Testing team dependency** — Phase 1 and the failure analysis tooling require a dedicated testing team. Without this investment, component teams must self-serve on test discovery and triage.
- **Framework dependency** — This process depends on the YAML-based test configuration framework (PR #1135) being accepted and maintained. Changes to the framework may require updates to this process.

---

## Alternatives

### Alternative 1: Manual test selection

Teams manually identify and list relevant tests. This is the current approach and has the problems described in the Motivation section — it's slow, error-prone, and duplicates effort across device teams.

### Alternative 2: Run everything from day one

Set `unlisted_test_mode: mandatory_success` immediately and work through all failures. This produces an overwhelming number of failures with no structure for prioritization, making it impractical for early-stage device backends.

### Alternative 3: Upstream-only testing

Only run tests that PyTorch explicitly marks as device-generic. This misses the majority of useful tests and provides no mechanism for device-specific corner cases.

---

## Prior Art

- **PyTorch's `@ops` and `@modules` framework** — The parametrized testing infrastructure that this process builds on. Defined in `torch/testing/_internal/common_device_type.py` and `common_modules.py`.
- **CUDA device testing** — PyTorch's CUDA backend testing uses similar patterns (device-type instantiation, dtype filtering) but without the YAML configuration layer.
- **OpenReg** — PyTorch's out-of-tree device registration mechanism (`privateuse1`) that enables OOT backends to participate in upstream tests.

---

## How we teach this

- This RFC serves as the primary teaching document, with its phased structure guiding teams from zero to full coverage.
- The testing team provides starter YAML configs and tooling that lower the barrier to entry.
- Each phase has explicit exit criteria that serve as a checklist for teams to track progress.
- The "Quick Reference: Config Patterns" section (in the detailed process document) provides copy-paste recipes for common scenarios.

---

## Unresolved questions

- **RFC enhancement dependencies** — This process relies on `test_selectors` and `supported_modules` features proposed as enhancements to the test configuration framework (PR #1135). The timeline for these enhancements needs to be aligned.
- **Failure analysis tooling scope** — The exact capabilities and interface of the testing team's failure analysis tooling (Steps 2.4 and 5.6) are not yet defined. This will be developed iteratively based on component team needs.
- **Cross-backend shared configs** — Should there be a shared base config that all OOT device backends inherit from? This could reduce duplication but may over-constrain individual backends.

---

## Resolution

TBD — pending RFC review.

### Level of Support

TBD

### Next Steps

- Review and merge the YAML test configuration framework (PR #1135)
- Implement `test_selectors` and `supported_modules` enhancements
- Testing team begins Phase 1 survey of non-`@ops`/`@modules` upstream tests
- Develop failure analysis tooling prototype

#### Tracking issue

TBD

#### Exceptions

None at this time.
