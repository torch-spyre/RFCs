# RFC: Model enablement tracking

**Authors:**

- Romit Jain (IBM Research, India)
- Ashok Pon Kumar Sree Prakash (IBM Research, India)

## **Summary**

This RFC proposes an approach to track model enablement and proposes view on tracking it via HuggingFace or vLLM model definitions.

## **Motivation**

Our goal is to track model enablement on Spyre. For a model to be enabled, primarily we need 3 levels of enablement:

1. **Ops**: Individual torch operations (e.g., matmul, softmax) implemented in `torch-spyre`
2. **Modules**: Compositions of ops that form model layers (e.g., Attention, MLP) implemented in `vllm-spyre`
3. **Inference engine**: vLLM with Spyre plugin

## **Proposed Implementation**

### Discovery

Keeping inference engine fixed (`vLLM`), we need to discover the ops and modules of interest. One of the ways of discovering these are by tracing the model's forward pass.

#### Model definitions (HF vs vLLM)

We can trace the model definition from:

1. HuggingFace (HF) model implementation, or
2. vLLM model implementation (on CPU)

There are trade-offs in choosing either of these implementations.

HF model implementation, while easy to initialize and isolate, can differ from what we actually want to deliver in production. Testing a different pair of ops might result in unnecessary work.

1. Example, for granite-3.3-8b-it, HF uses unfused ops for the [MLP layer](https://github.com/huggingface/transformers/blob/main/src/transformers/models/granite/modeling_granite.py#L215), while vLLM uses a [fused layer](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/granite.py#L79). Even though the ops might be same, vLLM fused layer will have different shapes for the matmul.
2. Example, for granite-3.3-8b-it, HF does not implement support for residual in [RMSNorm](https://github.com/huggingface/transformers/blob/main/src/transformers/models/granite/modeling_granite.py#L183). This implementation is different from [RMSNorm in vLLM](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/layernorm.py#L222), that does support it (and even vllm-spyre adopts it).
3. Tracing HF model implementation, would have never surfaced [`SiluAndMul`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/granite.py#L97) module that fuses operations and is absent from HF implementation.

Given these examples, we can argue that tracing model definition from different sources will generate different list and order of ops applied. It is important to discover the source of ops and modules from the definition we intend to deliver in production i.e. vLLM.

### Testing modules independently

Once we discover the ops and modules, the ops can be isolated, implemented and tested in `torch-spyre`. However, we still have to answer two questions for tracking end to end enablement of models on Spyre.

1. Which modules do we test - HF vs vLLM?
2. Should we even test modules independently?

#### Which modules do we test - HF vs vLLM

For HF model definition, it is straightforward to extract the modules. When we initialize a model in HF, it provides us with a representation of the model in the form:

```python
import torch
from transformers import AutoModel

model_path = "ibm-ai-platform/micro-g3.3-8b-instruct-1b"
dtype = torch.float16
device = "cpu"

model = AutoModel.from_pretrained(model_path, dtype=dtype).eval().to(device)
print(model)
```

```bash
# HF model representation of ibm-ai-platform/micro-g3.3-8b-instruct-1b
GraniteModel(
  (embed_tokens): Embedding(49159, 4096, padding_idx=0)
  (layers): ModuleList(
    (0-3): 4 x GraniteDecoderLayer(
      (self_attn): GraniteAttention(
        (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
        (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
        (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
        (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
      )
      (mlp): GraniteMLP(
        (gate_proj): Linear(in_features=4096, out_features=12800, bias=False)
        (up_proj): Linear(in_features=4096, out_features=12800, bias=False)
        (down_proj): Linear(in_features=12800, out_features=4096, bias=False)
        (act_fn): SiLUActivation()
      )
      (input_layernorm): GraniteRMSNorm((4096,), eps=1e-05)
      (post_attention_layernorm): GraniteRMSNorm((4096,), eps=1e-05)
    )
  )
  (norm): GraniteRMSNorm((4096,), eps=1e-05)
  (rotary_emb): GraniteRotaryEmbedding()
)
```

From the model object, we can extract individual modules and run a forward pass through them.

```python
modules = dict(model.named_modules())
layer = modules["model.layers.0.mlp"] # Any model can be extracted from the dict


# Running forward pass through the module
x = torch.randn(1, in_features, dtype=dtype, device=device)
result = layer(x)
```

But as we saw in `Discovery` section, different model definitions can be implemented in different ways. Every module packages a few torch ops, modules and some logic between them to run its own forward pass. Getting a similar level of detail from vLLM is more involved:

```python
# Extract vLLM model definition
import torch
import torch.nn as nn

from vllm.config import (
    VllmConfig,
    ModelConfig,
    CacheConfig,
    ParallelConfig,
    DeviceConfig,
    set_current_vllm_config,
    CompilationConfig,
)
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import get_model
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)

model_path = "ibm-ai-platform/micro-g3.3-8b-instruct-1b"
dtype = torch.float16
device = "cpu"

model_config = ModelConfig(
    model=model_path,
    dtype=dtype,
)

compilation_config = CompilationConfig(
    custom_ops=["all"],
)

vllm_config = VllmConfig(
    model_config=model_config,
    cache_config=CacheConfig(block_size=16),
    parallel_config=ParallelConfig(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    ),
    device_config=DeviceConfig(device=device),
    load_config=LoadConfig(load_format="auto"),
    compilation_config=compilation_config,
)

with set_current_vllm_config(vllm_config):
    init_distributed_environment(world_size=1, rank=0, local_rank=0)
    initialize_model_parallel(tensor_model_parallel_size=1)

    model = get_model(vllm_config=vllm_config)

print(model)
```

```bash
# vLLM model representation of ibm-ai-platform/micro-g3.3-8b-instruct-1b
GraniteForCausalLM(
  (model): GraniteModel(
    (embed_tokens): VocabParallelEmbedding(num_embeddings=49216, embedding_dim=4096, org_vocab_size=49159, num_embeddings_padded=49216, tp_size=1)
    (layers): ModuleList(
      (0-3): 4 x GraniteDecoderLayer(
        (self_attn): GraniteAttention(
          (qkv_proj): QKVParallelLinear(in_features=4096, output_features=6144, bias=False, tp_size=1, gather_output=False)
          (o_proj): RowParallelLinear(in_features=4096, output_features=4096, bias=False, tp_size=1, reduce_results=True)
          (rotary_emb): RotaryEmbedding(
            head_size=128, rotary_dim=128, max_position_embeddings=131072, base=10000000.0, is_neox_style=True
            (apply_rotary_emb): ApplyRotaryEmb(is_neox_style=True, enable_fp32_compute=False)
          )
          (attn): Attention(head_size=128, num_heads=32, num_kv_heads=8, scale=0.0078125, backend=CPUAttentionBackendImpl)
        )
        (mlp): GraniteMLP(
          (gate_up_proj): MergedColumnParallelLinear(in_features=4096, output_features=25600, bias=False, tp_size=1, gather_output=False)
          (down_proj): RowParallelLinear(in_features=12800, output_features=4096, bias=False, tp_size=1, reduce_results=True)
          (act_fn): SiluAndMul()
        )
        (input_layernorm): RMSNorm(hidden_size=4096, eps=1e-05)
        (post_attention_layernorm): RMSNorm(hidden_size=4096, eps=1e-05)
      )
    )
    (norm): RMSNorm(hidden_size=4096, eps=1e-05)
  )
  (lm_head): ParallelLMHead(num_embeddings=49216, embedding_dim=4096, org_vocab_size=49159, num_embeddings_padded=49216, tp_size=1)
  (logits_processor): LogitsProcessor(vocab_size=49159, org_vocab_size=49159, scale=0.0625, logits_as_input=False)
)
```

And once we have the model object, we can extract and test the modules inside the model in a similar way to HF

```python
modules = dict(model.named_modules())
layer = modules["model.layers.0.mlp"] # Any model can be extracted from the dict


# Running forward pass through the module
x = torch.randn(1, in_features, dtype=dtype, device=device)
result = layer(x)
```

Since vLLM modules are accessible and testable in the same way HF modules are, vLLM model definition should be given priority over HF model definition. We can clearly see from the model representation of HF vs vLLM, certain layers are very different. For example:

1. `embed_tokens`: HF model definition has 49159 size of the matrix, whereas vLLM's `VocabParallelEmbedding` has 49216 due to performance reasons.
2. `self_attn`: For q, k, and v projection, HF model definition does it as 3 separate matmuls, whereas vLLM's `QKVParallelLinear` fuses all 3 matmuls into one.
3. `mlp.act_fn`: HF implements `SiLUActivation` and does multiplication operation outside of the `SiLUActivation` layer. vLLM's `SiluAndMul` fuses both the ops in a single layer.

#### Should we even test modules independently?

Testing modules independently is valuable, but we must be aware of when it may not reflect the production behavior. To illustrate it, let's assume a toy module with some toy ops:

```python
class ToyModule(nn.Module):
    def __init__(self):
        self.param1 = nn.Parameter(...)

    def forward(self, x):
        out = torch.toy_op1(self.param1, x)
        out = torch.toy_op2(out)

        return out
```

Assuming this module is implemented in vLLM, to test the same module on Spyre, we would have to move the weights to Spyre:

```python
my_module = model.ToyModule
x = torch.randn(input_shapes)

ref_out = my_module(x) # on cpu

spyre_module = my_module.to(device="spyre")
spyre_x = x.to(device="spyre")
spyre_out = spyre_module(spyre_x) # on spyre

torch.testing.assert_close(ref_out, spyre_out)
```

If this test passes, we can confirm:

1. Both torch ops work on Spyre
2. The complete module works on Spyre

However, it is possible that in `vllm-spyre` plugin, we decide to implement the `ToyModule` using different ops or in a different way.

```python
class SpyreToyModule(ToyModule):
    ...

    def forward(self, x):
        out = torch.fused_toy_op(self.param1, x, fuse_op=True)
        return out
```

In this case, testing `ToyModule` by moving it on Spyre, will not give us the complete picture of that module readiness on Spyre. For these special cases, it is better to test them **after** they are implemented in `vllm-spyre` plugin. For modules used as-is from vLLM (without customization in vllm-spyre), testing them in advance by moving them to Spyre provides valid enablement signal.

Since we cannot know in advance which modules will be customized in `vllm-spyre`, the recommended approach is to first check if a Spyre-specific implementation exists in `vllm-spyre` before testing. If it does, test the `vllm-spyre` version. If not, we need to decide if it will be implemented in `vllm-spyre`.

#### End-to-end testing with partial enablement

As modules are enabled incrementally, we can validate progress by running end-to-end inference with a hybrid approach: enabled modules run on Spyre while modules not yet enabled fall back to CPU. This allows us to:

1. Verify that enabled modules integrate correctly in the full model context
2. Catch issues that only surface when modules interact (e.g., shape mismatches, numerical drift accumulation)
3. Measure incremental progress toward full model enablement on vLLM

### Model enablement metric

After the ops and module discovery, we can track model enablement by:

1. The percentage of layers implemented and tested in `vllm-spyre` plugin. For example, if 8 modules out of 20 required modules are implemented in `vllm-spyre`, we mark the progress as 40%. This can be tracked by tracing vLLM model definition for total required modules, and doing static analysis on `vllm-spyre` to discover modules that inherit from vLLM modules. If there are any module implementation in vLLM, that does not need reimplementation in `vllm-spyre`, we count that towards the progress of model enablement.
2. The percentage of ops implemented and tested in `torch-spyre` plugin. The torch ops can be traced from vLLM model definition and mapped to all the ops implemented in `torch-spyre`.

### Conclusion and recommendations

1. Use vLLM model definition to trace the ops and modules
2. Create/update yaml with required ops to be tested and required modules to be implemented
3. Run periodic test on relevant ops from upstream `torch-spyre`. These tests are targeted towards the specific shapes and dtypes used in the model of interest. The testing will follow the `torch-spyre` testing framework as described in RFC.
4. Run periodic test on relevant modules from upstream in `vllm-spyre` as they are enabled. The testing will follow the `vllm-spyre` testing framework as described in [RFC](https://github.ibm.com/romit/docs/blob/main/rfc-vllm-spyre-next-testing.md).
5. Maintain a dashboard with two separate metrics for each model of interest
   1. The percentage of ops that are enabled and tested
   2. The percentage of modules that are enabled and tested
