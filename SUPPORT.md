# Support

## Support Level

Model Distillation Recipes is maintained by NVIDIA. Support is provided on a
best-effort basis through the repository issue tracker; no response-time or
resolution-time service-level agreement is provided.

## Supported Scope

Supported project functionality includes:

- The documented LoRA supervised fine-tuning and logit-level distillation
  recipes.
- The model families, FSDP2 configurations, and chat templates explicitly
  listed in the README.
- The documented direct `torchrun` and container workflows on compatible
  NVIDIA GPU environments.
- The default deterministic CPU test suite.

The following are outside the current support scope:

- Production deployment guarantees, pretrained model weights, hosted services,
  or managed training.
- Unlisted model architectures, dependency versions, hardware, containers, or
  distributed environments.
- Private infrastructure, proprietary datasets, and organization-specific
  scheduler or storage configuration.
- Questions about rights to use a particular model, tokenizer, dataset, or
  generated artifact.

## Requesting Help

Before opening an issue:

1. Review the README and relevant documents under `docs/`.
2. Reproduce the issue with synthetic or otherwise redistributable data when
   possible.
3. Record the code revision, complete command, model architecture (without
   sharing restricted weights), Python/PyTorch/CUDA/NCCL versions, GPU model,
   container image digest, precision mode, and distributed topology.
4. Remove credentials, private paths, proprietary data, and restricted model or
   dataset content from the report.

Use NVIDIA Product Security channels described in `SECURITY.md` for
vulnerabilities rather than opening a public issue.
