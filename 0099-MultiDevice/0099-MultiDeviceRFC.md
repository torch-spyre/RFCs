# Multi-Spyre Device Support Pytorch

**Authors:**

* @jjhursey

## **Summary**

This RFC describes additions necessary to support "Spyre" based collective library into the Pytorch ecosystem. This entails creating a module that implements the distributed interfaces in Pytorch and connecting this module as the default for "Spyre" devices.

## **Motivation**

The goals of this proposal are as follows:

1. Standard Pytorch interface to the external Spyre Collective Communication Library for accelerated spyre-to-spyre communication
2. Support the torch.distributed interface through a custom backend module
3. Support the funcitonal collectives through the fallback mechanism to torch.ditributed from inductor
4. Support existing Pytorch out-of-the-box Pytorch consumers using the communication library provided by Pytorch

## **Proposed Implementation**

### External Spyre Collective Communication Library

The Spyre Collective Communication Library (Spyre CCL) is an external library that support efficient spyre-to-spyre device communication for both the point-to-point and collective communication patterns. This library provides a stable C++ API for a Pytorch distributed module to call into. The Spyre CCL library assumes that all input tensors are already resident in Spyre device memory, and output tensor will remain in Spyre device memory.

The Spyre CCL only supports on-node (single server) communication. Multi-node support may be added in the future.

The Spyre CLL will be linked in the Pytorch module.

### Spyre CCL Pytorch Distributed Module

Pytorch's [distributed communication package](https://docs.pytorch.org/docs/2.10/distributed.html) (a.k.a., `torch.distributed`) provides point-to-point and collective communication interface for Pytorch users. Backends to `torch.distributed` can support one or more device types. There are a few built-in backends (e.g., `gloo`, `nccl`). Pytorch also provides support for [external backends](https://docs.pytorch.org/tutorials/intermediate/process_group_cpp_extension_tutorial.html) through the extensions interface by extending the [`Backend` class](https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/Backend.hpp).

We will implement the `spyreccl` backend module by extending the `Backend` class, and registering this backend module to be the default `torch.distributed` module for the `spyre` device. Since Spyre CCL provides a C++ interface, the core implementation of the `spyreccl` module will be in C++.

```cpp
    py::object module = py::module::import("torch.distributed");
    py::object register_backend = module.attr("Backend").attr("register_backend");
    std::vector<std::string> supported_devices;
    supported_devices.push_back("spyre");
    register_backend("spyreccl", py::cpp_function(createSpyreCCLBackend), false, supported_devices);
```

A check will be provided to throw an exception when a non-`spyre` tensor is passed to the library. Users should be protected from this naturally via the dispatch mechanism provided by `torch.distributed`, however users may circumvent this dispatch manually for specific devices (e.g., `cpu`).

## Support for Functional Collectives

In Pytorch 2.0 the compiled execution mode was introduced. Since the `torch.distributed` interface is eagerly executed it introduces a graph break between compute and communication. The Functional Collective interface was [adopted](https://github.com/pytorch/pytorch/issues/93173) to provide the compiled flow a view of some of the collective operations. For compiler backends (e.g., [`inductor`](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/_inductor/comm_lowering.py)) that do not support tracing of these collectives there is [a fallback to](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/_functional_collectives.py) eager execution through a direct call into the `torch.distributed` backend.

We plan to support functional collectives via the same fallback into `torch.distributed` when using the compiled execution mode with the Spyre enhanced inductor-based backend.

## Usability

The `spyreccl` backend will be autoloaded with the `spyre` device at `import torch` time. Thus the user does not need any explicit imports to bring in this capability.

Users must first initialize the `torch.distributed` process group. Since the `spyreccl` is autoloaded and registered as the default module for the `spyre` device it is available without any additional arguments.

```python
torch.distributed.init_process_group()

# Users may explicitly load it if they wish, but do not need to
# torch.distributed.init_process_group(f"spyre:spyreccl")
```

The `torch.distributed` library will route the collective call to the default backend module for the tensor. For Spyre users, they can call the Spyre CCL collective library through standard interfaces without any custom arguments.

```python
# Create input tensor
x = torch.rand(512, 1024, dtype=torch.float16)
# Send the tenor to the Spyre device
x_device = x.to("sypre")
# Broadcast the tensor using the Spyre CCL library
dist.broadcast(x_device, 0)
```

## **Metrics**

To measure the completeness of this feature, the following will be met:

* [ ] Functional support for `torch.distributed` collective operations necessary for Tensor Parallel execution (i.e., `allreduce`)
* [ ] Demonstrated functional completeness via a dedicated unit test per supported operation
* [ ] Pytorch [upstream tests](https://github.com/pytorch/pytorch/blob/v2.10.0/test/distributed/test_c10d_functional_native.py) running to completion with accuracy checks for supported operations.
* [ ] Initial performance assessment using standard benchmarks (either a modified form of [this](https://github.com/pytorch/pytorch/blob/main/benchmarks/distributed/ddp/benchmark.py) or [this](https://github.com/IBM/pytorch-communication-benchmarks))
* [ ] Expand support for other collectives on an as-needed basis

## **Drawbacks**

The implementation cost is considerable, but necessary.

## **Alternatives**

The [`torchcomms` project](https://github.com/meta-pytorch/torchcomms) seeks to replace the `torch.distribued` interface. It has a similar `Backend` (called `TorchCommBackend`) extension model with a slightly different API. This project is not yet accepted upstream into Pytorch (though it is eventually expected to be accepted when complete) so it is not a viable alternative for mainline support. If/when it is accepted we should be able to fork the existing `Backend` and modify it to support the new interface.

Instead of extending the [`Backend` class](https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/Backend.hpp) we could extend the [`ProcessGroup` class](https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/ProcessGroup.hpp). This is the older way of building an external module extension to the `torch.distributed` framework. This is discouraged by upstream as they move toward the more flexible `Backend` class ([ref](https://github.com/pytorch/pytorch/pull/90997)). The `Backend` class should provide all of the functionality needed for this effort.

## **Prior Art**

Prior to this techique we supported only functional collectives through an `AOTAutograd` compiler backend specific to Spyre. In the Spyre enhanced inductor-based backend we will need to provide additional support for `torch.distributed` since it falls back to it for functional collectives.

## **How we teach this**

NA

## **Unresolved questions**

* Eager execution of the `torch.distributed` interface in a compiled execution model introduces dispatching latency during model execution. Various projects have approached this performance problem in different ways. We will need to resolve this to realize the full performance capabilities of the hardware.
* Eventually the Spyre CCL module will provide full functional support. Due to time constraints, interface support will be prioritized by model framework needs. This may result in a 'long tail' for full functional completeness.

## Resolution

TBD

### Level of Support

<!--
Choose one of the following:
* 1: Overwhelming positive feedback.
* 2: Positive feedback.
* 3: Majority Acceptance, with conflicting Feedback.
* 4: Acceptance, with Little Feedback.
* 5: Unclear Resolution.
* 6: RFC Rejected.
* 7: RFC Rejected, with Conflicting Feedback.
-->

#### Additional Context

TODO

### Next Steps

TODO

#### Tracking issue

* https://github.com/torch-spyre/torch-spyre/issues/99

#### Exceptions

TODO
