# SuperDSC Validation Engine

SuperDSC or SuperDSC Bundle is the interface between torch-spyre and the backend compiler. To ensure the SuperDSC contract is well maintained and easily verifiable we propose a framework to perform SDSC validation outside the bounds of the backend compiler.
Such validation engines can be plugged into torch-spyre, maintained separately or added as modules in other verification frameworks like [Torch-Spyre Mock Device RFC](https://ibm.ent.box.com/notes/2163908442337?s=ykg1smqadxrcp9julyyj3pse0sg7rrq7).

## Contacts

- Dushyant Behl (dushyantbehl@in.ibm.com)
- Umamaheswari Devi (umamadev@in.ibm.com)

## What we propose

We list below two main approaches for SDSC validation.

## SDSC Schema and Pythonic class generation.

Currently torch-spyre only contains `SDSCSpec` as a pythonic dataclass which is an input to the SuperDSC generation function while the actual SuperDSC JSON inside the `generate_sdsc` function is constructed separately. This is both cumbersome to maintain and harder to test because individual subparts of the SDSC have to be extracted from the captured JSON and not present as pythonic(pydantic or dataclass) classes which can be easily mocked for generating tests.
We want to propose use of `SDSCSpec` (or additional dataclasses) which form a pythonic representation for the entire SuperDSC JSON to generate the final SDSC/SDSC Bundle.

To move towards a pythonic view of the SuperDSC JSON we extracted a schema for the SuperDSC JSON which conforms to the SDSC JSON being accepted by the backend compiler and it's present [here](./sdsc_schema.json).

Our goal with coming up with the schema mainly is to, 
- Ensure the schema is regularly maintained as a contract between the frontend and backend compiler.
    - Maintaining such a schema which is a single source of truth as the syntactic interface between torch-spyre and the backend compiler will allow us to pinpoint any difference between superdsc creation and acceptance in the frontend and backend compilers. 
- Propose torch-spyre to ensure we have pythonic classes for SuperDSC JSON based on the generated schema.
    - This will help us with the [compilation stage wise testing](./002-rfc-torch-spyre-compilation-stage-wise-testing.md) as pythonic objects will be easy to create by hand or mock.

We plan to expand this schema to include all features required by the backend compiler as part of the pythonic object generation.

---

## Rule based validation engine. What and Why?

While the schema provides mostly syntactic checks on the SuperDSC JSON we wanted to see if we can also have a framework perform semantic checking on the SDSC. To start this we looked at the backend compiler in how it processes the SuperDSC JSON.

The backend compiler performs a number of semantic checks on SuperDSC JSON objects, scattered across the codebase. There is no easy way to run these checks independently, see what they enforce, or get a clear report on why a SuperDSC is valid or invalid.

We propose a **Rule based validation engine** that:
- Captures all JSON-checkable semantic constraints as declarative rules in a file
- Uses **Jinja2 templates** as the check expression language
    - Jinja is just an example and we can change the rules to conform to any other language like cel/rego as well.
- Evaluates rules against any SuperDSC JSON and produces a pass/fail report
- Allows adding, modifying, or disabling checks without touching code

---

## How Rules Look

To try some initial validation we have a list of rules from the checks being performed inside the backend compiler on the sdsc json.
We list all such rules in this [document](https://github.ibm.com/dushyantbehl/rfcs/blob/main/sdsc_validator/sdsc_semantic_checks.md). It's easy to note that, schema validation can be added as one of the checks for syntactic and semantic verification.

Our idea is to seed the validator with such rules and verify the credibility of the checks. Once the approach is validated we plan to generate a claude skill which can be run periodically to auto generate rules from the backend compiler toolchain that are done on the SDSC JSON. We provide a few sample rules below,

While we created the initial set of rules and a few custom rules from our experience, to truly validate SDSC JSON we would need help from the backend compiler team to correctly interpret and review some of the rules and potentially generate complex relationships which are not easily captured without deep knowledge of the toolchain.

Each rule has metadata (name, severity) and a Jinja2 expression that evaluates to `True` or `False` against the SuperDSC fields:

```yaml
version: "1.0"
rules:
  # Simple value check
  - name: "target_ must be a recognized value"
    severity: error
    fields: ["target_"]
    check: "{{ target_ in ['senulator', 'sentient', 'senpcfg', 'sentf', 'host'] }}"

  # Mutual exclusion
  - name: "Exactly one of dscs_ or dataOpdscs_ must be non-empty"
    severity: error
    fields: ["dscs_", "dataOpdscs_"]
    check: "{{ (dscs_ | length > 0) != (dataOpdscs_ | length > 0) }}"

  # Conditional — only applies when using standard DSC path
  - name: "dscs_ must have exactly 1 entry when using standard path"
    severity: error
    fields: ["dscs_", "dataOpdscs_"]
    check: "{{ dataOpdscs_ | length > 0 or dscs_ | length == 1 }}"

  # Per-DSC iteration — loop produces one True/False per DSC
  - name: "coreIdsUsed_ size must match numCoresUsed_ in each DSC"
    severity: error
    fields: ["dscs_[*].coreIdsUsed_", "dscs_[*].numCoresUsed_"]
    check: >
      {% for dsc in dscs_ %}
      {{ dsc.coreIdsUsed_ | length == dsc.numCoresUsed_ }}
      {% endfor %}

  # Dimension hierarchy — N_ >= ChipD_ >= CoreD_ per dimension
  - name: "Dimension hierarchy must be non-increasing through levels"
    severity: error
    fields: ["dscs_[*].N_", "dscs_[*].ChipD_", "dscs_[*].CoreD_"]
    check: >
      {% set dims = ['in_', 'out_', 'mb_', 'ij_'] %}
      {% for dsc in dscs_ %}
      {% for dim in dims %}
      {{ dsc.N_[dim] == -1 or dsc.ChipD_[dim] == -1 or dsc.N_[dim] >= dsc.ChipD_[dim] }}
      {{ dsc.ChipD_[dim] == -1 or dsc.CoreD_[dim] == -1 or dsc.ChipD_[dim] >= dsc.CoreD_[dim] }}
      {% endfor %}
      {% endfor %}

  # Aggregate — total tensors across all DSCs
  - name: "Total tensors must not exceed 8 LDS segments"
    severity: error
    fields: ["dscs_[*].labeledDs_", "dataOpdscs_[*].labeledDs_"]
    check: >
      {% set ns = namespace(total=0) %}
      {% for dsc in dscs_ | default([]) %}
        {% set ns.total = ns.total + dsc.labeledDs_ | default([]) | length %}
      {% endfor %}
      {{ ns.total <= 8 }}

  # Coordinate dimension property consistency
  - name: "dim_prop_func and dim_prop_attr length must equal elemArr + 3 for each dimension"
    severity: error
    fields: ["scheduleTree_.allocateNode.coordinates.coordInfo"]
    check: >
      {% set dims = ['in_', 'out_', 'mb_', 'ij_'] %}
      {% for dim in dims %}
        {% if scheduleTree_.allocateNode.coordinates.coordInfo[dim] is defined %}
          {% set n = scheduleTree_.allocateNode.coordinates.coordInfo[dim].elemArr %}
          {{ scheduleTree_.allocateNode.coordinates.coordInfo[dim].dim_prop_func | length == n + 3 }}
          {{ scheduleTree_.allocateNode.coordinates.coordInfo[dim].dim_prop_attr | length == n + 3 }}
        {% endif %}
      {% endfor %}

  # Tensor name cross-reference between computeOp and labeledDs
  - name: "Every tensor in computeOp inputs/outputs must have a matching dsName in dscs_[].labeledDs_"
    severity: error
    fields: ["computeOp_.inputLabeledDs", "computeOp_.outputLabeledDs", "dscs_[*].labeledDs_"]
    check: >
      {% set all_ds_names = [] %}
      {% for dsc in dscs_ | default([]) %}
        {% for lds in dsc.labeledDs_ | default([]) %}
          {% set _ = all_ds_names.append(lds.dsName) %}
        {% endfor %}
      {% endfor %}
      {% set all_tensors = computeOp_.inputLabeledDs | default([]) + computeOp_.outputLabeledDs | default([]) %}
      {% for tensor_name in all_tensors %}
        {% set parts = tensor_name.split('-') %}
        {% set prefix = parts[:-1] | join('-') %}
        {{ prefix in all_ds_names }}
      {% endfor %}

  # Runtime-only check — documented but not evaluated
  - name: "Opcodes must be valid for target ISA"
    severity: error
    fields: []
    checkable: false
    check: ""
```



### Rule Fields

| Field | Description |
|-------|-------------|
| `id` | Unique identifier |
| `name` | Human-readable description |
| `severity` | `error` (must-enforce) or `warning` (good-to-have) |
| `fields` | SuperDSC fields involved (for traceability) |
| `check` | Jinja2 template that renders to `True`/`False` |
| `checkable` | `false` for runtime-only checks documented but not evaluated |

---

## What Kinds of Checks Can Be Expressed

| Category | What it checks | Example |
|----------|---------------|---------|
| **Presence** | Fields exist and are non-empty | `dscs_` or `dataOpdscs_` must be populated |
| **Mutual exclusion** | Exactly one of N fields populated | `dscs_` XOR `dataOpdscs_` |
| **Cardinality** | Size relationships between fields | `coreIdsUsed_ | length == numCoresUsed_` |
| **Value enumeration** | Field in allowed set | `target_` in `[senulator, sentient, ...]` |
| **Collection enumeration** | Every element in allowed set | All `dsType_` values are valid segment types |
| **Uniqueness** | No duplicate entries | `prodConsList` keys unique, `pcfg_` core IDs unique |
| **Cross-reference** | Indices resolve correctly | `coreIdToDsc_` values are valid into `dscs_` |
| **Structure shape** | Sub-arrays have expected sizes | Schedule steps have exactly 4 components |
| **Dimension hierarchy** | Parent dims >= child dims | `N_ >= ChipD_ >= CoreD_` per dimension field |
| **Non-splittable dims** | Certain dims equal across levels | `N_.j_ == CoreD_.j_` (hardware constraint) |
| **Divisibility** | Parent evenly divides child | `N_.in_ % CoreD_.in_ == 0` |
| **Conditional** | Check applies only when precondition met | LSTM ops require `numWkSlicesPerDim_.out % 4 == 0` |
| **Aggregate** | Totals across nested structures | Total `labeledDs_` count <= 8 segments |
| **Derived length** | Array length must equal a derived value from a sibling field | `dim_prop_func` and `dim_prop_attr` length == `elemArr + 3` per coordInfo dimension |
| **Name cross-reference** | Names from one structure must resolve in another via prefix stripping | Every `computeOp_` tensor (prefix before last `-`) must match a `labeledDs_.dsName` |

**Further versions of the checker can find**: ISA opcode validation, memory allocation, instruction buffer overflow, MLIR structural verification.

---

## How It Might Work

```
  YAML Rules ──┐
               ├──▶  Jinja2 Sandbox  ──▶  Report (text / json)
  SDSC JSON ───┘     renders each         with pass/fail
                     check template        per rule
                     against SDSC
                     fields
```

1. Load YAML rules and SuperDSC JSON
2. Unwrap JSON (remove top-level node-name wrapper and `dscs_` single-key wrappers)
3. For each rule, render its `check` template with SDSC fields as Jinja2 context
4. Parse rendered output — whitespace-separated `True`/`False` tokens
5. Any `False` token means the rule failed; for loops, the failing iteration is identified
6. Generate report with pass/fail per rule

It's important to note that our approach is modular and can be integrated with other testing and validation approaches like,
the [Torch-Spyre Mock Device RFC](https://ibm.ent.box.com/notes/2163908442337?s=ykg1smqadxrcp9julyyj3pse0sg7rrq7) proposal.

---

## What the Report Might Look Like

```
══════════════════════════════════════════════════════════════
  SuperDSC Validation Report
  Data:  my_conv2d_sdsc.json
  Rules: sdsc_rules.yaml (47 rules, 5 not checkable)
══════════════════════════════════════════════════════════════

  [PASS] target_valid — target_ must be a recognized value
  [PASS] core_ids_match — coreIdsUsed_ size must match numCoresUsed_
  [PASS] dsc_presence — dscs_ OR dataOpdscs_ must be non-empty
  [PASS] dsc_xor — Exactly one of dscs_ or dataOpdscs_ must be non-empty
  [PASS] single_dsc — dscs_ must have exactly 1 entry
  [FAIL] tensor_count — Total tensors must not exceed 8 LDS segments
         ERROR  Total = 10, max = 8
  [FAIL] dim_hierarchy — Dimension hierarchy must be non-increasing
         ERROR  dscs_[0], dim out_: N_.out_=64, ChipD_.out_=128
  [WARN] corelet_fold — coreletFoldProp_ factor must be 2
         WARNING  factor = 4, expected 2

──────────────────────────────────────────────────────────────
Summary: 40 passed, 1 failed (error), 1 failed (warning)
Overall: FAIL
══════════════════════════════════════════════════════════════
```

