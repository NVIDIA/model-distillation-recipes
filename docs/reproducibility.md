# Reproducibility

Model training is sensitive to software builds, distributed topology, input
ordering, kernels, and model/data revisions. A seed is necessary but not
sufficient to reproduce a run.

## Record before execution

Capture the following in a durable run manifest outside the source tree:

- source commit and whether the working tree was clean;
- exact command line and environment overrides;
- Python, PyTorch, Transformers, PEFT, datasets, and optional kernel
  versions;
- immutable base and final container image digests;
- GPU model, GPU count, node count, driver, CUDA, NCCL, and network settings;
- model and tokenizer identifiers, immutable revisions, and local checksums;
- chat-template checksum and supported `model_type` (`qwen3` or `nemotron_h`);
- dataset checksums plus documented preprocessing and split;
- seed, sequence length, global/micro batch sizes, accumulation, optimizer,
  schedule, precision, attention backend, and all loss settings; and
- checkpoint source when resuming.

Pin container images as `name@sha256:<digest>`, not only by tag. Native Mamba
kernels depend on the build architecture and toolchain, so record their wheel
or shared-object provenance as well.

## Validation levels

Run the deterministic, model-free CPU suite first:

```bash
pytest -m "not gpu and not distributed and not slow and not requires_model"
```

This checks source behavior but does not validate CUDA, NCCL, model loading, or
numerical equivalence. Run separately marked gates in an approved GPU
environment:

- `gpu`: CUDA and accelerator behavior;
- `distributed`: multi-process and collective behavior;
- `slow`: extended integration or performance checks;
- `requires_model`: checks that load model or tokenizer assets.

For a release candidate, smoke-test LoRA and distillation on both supported
model types, checkpoint/resume, final export, and multi-node launch.
Compare losses and metrics with tolerances appropriate to bfloat16 and the
selected kernels; bitwise identity is not generally expected across hardware
or software stacks.

## Generated artifacts

Depending on the command, runs may produce:

- a PEFT LoRA adapter and tokenizer files;
- a full Hugging Face safetensors model and tokenizer;
- sharded DCP model/optimizer checkpoints under `dcp/step_<step>`;
- stdout/stderr, scheduler logs, caches, compiled kernels, and optional Weights
  & Biases records.

These outputs may be large, sensitive, or nondeterministic and are intentionally
excluded from source control. Store them in an artifact system with checksums,
retention rules, access controls, and links to the run manifest.

## Rights and publication

Source-code licensing and artifact licensing are separate. Before sharing a
checkpoint, adapter, container, or run log, verify the licenses and terms for
the base and teacher models, tokenizers, datasets, CUDA extensions, container
contents, and generated outputs. Also review artifacts for embedded training
examples, secrets, personal data, and absolute internal paths. Apache-2.0
applies to this repository's source code only and does not override third-party
or data-owner terms.
