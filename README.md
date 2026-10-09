# Model Distillation Recipes

Reference PyTorch recipes for LoRA supervised fine-tuning and logit-level
knowledge distillation with FSDP2. The code is intentionally small and direct:
Hugging Face models, PEFT adapters, `torch.distributed`, and explicit training
loops without a trainer framework.

This is research software. It requires NVIDIA GPUs for training and does not
provide production deployment guarantees or pretrained model weights.

## Supported scope

- Qwen3 (`model_type=qwen3`) and NemotronH-family
  (`model_type=nemotron_h`) causal language models.
- Assistant-only loss masking through model-specific Jinja chat templates.
- LoRA cross-entropy training.
- Teacher/student KL distillation, optionally combined with language-model
  cross-entropy.
- Direct single-node and multi-node `torchrun` launch paths.

Other architectures fail early until their FSDP2 layer class and LoRA target
modules are explicitly supported.

## Requirements

- Python 3.10 or newer.
- PyTorch with FSDP2 support.
- CUDA-capable NVIDIA GPUs for training.
- A model, tokenizer, and dataset whose licenses permit your intended use.

Install the core development environment from the repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Optional dependency groups are documented in `pyproject.toml`, including
experiment logging and NemotronH fused kernels. The supplied container is the
recommended GPU environment because it starts from a matched NGC PyTorch and
CUDA stack.

## Data and templates

Training data is JSONL with one `messages` array per line:

```json
{"messages":[{"role":"user","content":"Question"},{"role":"assistant","content":"Answer"}]}
```

See [docs/data_format.md](docs/data_format.md) for validation rules and
reasoning-data examples.

Choose the template matching the tokenizer:

- `configs/chat_template_qwen3_generation.jinja`
- `configs/chat_template_nemotron3_generation.jinja`
- `configs/chat_template_nemotron9b_generation.jinja`

Each template contains `{% generation %}` blocks used by Transformers to create
the assistant-token mask. Missing templates, incomplete generation blocks, and
samples with empty assistant masks are rejected before training.

Inspect a tokenizer/dataset pair before a GPU run:

```bash
python scripts/verify_fsdp2_dataloader.py \
  --tokenizer /path/to/model \
  --data /path/to/train.jsonl \
  --chat_template configs/chat_template_qwen3_generation.jinja \
  --blended_loss_weight 0.5
```

`--blended_loss_weight` requires single-token `<think>` and `</think>` markers.
Leave it unset for tokenizers such as Nemotron Nano 9B that do not provide
dedicated marker tokens.

## LoRA training

Launch one process per GPU with `torchrun`:

```bash
torchrun --standalone --nproc_per_node 4 scripts/lora_fsdp2.py \
  --model_path /path/to/model \
  --data_path /path/to/train.jsonl \
  --output_dir /path/to/lora-output \
  --chat_template_path configs/chat_template_qwen3_generation.jinja \
  --max_steps 500
```

The default LoRA targets match Qwen-style attention and MLP projections.
Override `--lora_target_modules` for architectures with different module names.
The attention backend is configurable with `--attn_implementation`.

LoRA uses shifted cross-entropy over assistant tokens only. With
`--blended_loss_weight W`, answer and reasoning pools are normalized separately:

```text
loss = W * mean(answer_loss) + (1 - W) * mean(reasoning_loss)
```

## Knowledge distillation

```bash
torchrun --standalone --nproc_per_node 4 scripts/distill_fsdp2.py \
  --student_path /path/to/student \
  --teacher_path /path/to/teacher \
  --data_path /path/to/train.jsonl \
  --output_dir /path/to/distilled-output \
  --chat_template_path configs/chat_template_qwen3_generation.jinja \
  --max_steps 100
```

The teacher is frozen. Per-token KD is temperature-softened KL divergence:

```text
kd = T^2 * KL(softmax(teacher / T) || log_softmax(student / T))
```

Distillation is KD-only by default. Pass `--use_lm_loss` to optimize
`LM cross-entropy + kd_loss_scale * KD`. Intermediate hidden-state loss is not
part of this recipe. Student and teacher tokenizers must have identical
token-to-ID vocabularies, and both models must expose the same output vocabulary
size; the script validates these requirements before sharding.

## Container

Build a local image:

```bash
docker build \
  --tag model-distillation-recipes:local \
  docker/
```

The Dockerfile uses the public `nvcr.io/nvidia/pytorch:26.07-py3` base. NGC access
may require accepting NVIDIA's registry terms and authenticating to `nvcr.io`.
NemotronH fused kernels are opt-in because they compile on the target
architecture:

```bash
docker build \
  --build-arg INSTALL_MAMBA_KERNELS=1 \
  --tag model-distillation-recipes:mamba \
  docker/
```

Record the resulting image digest for reproducibility. This repository contains
no registry-specific push helper.

Mount the repository and runtime data when starting the container:

```bash
docker run --rm -it --gpus all \
  --ipc=host \
  --volume "$PWD:/workspace/model-distillation-recipes" \
  --volume "/path/to/models:/models:ro" \
  --volume "/path/to/data:/data:ro" \
  --volume "/path/to/outputs:/outputs" \
  model-distillation-recipes:local
```

Model arguments may be mounted local paths or Hugging Face model IDs. Use
immutable model revisions or pre-downloaded snapshots for reproducible runs;
the training scripts load them directly without a separate staging utility.

## Testing

The default suite is deterministic and does not download model weights:

```bash
pytest -m "not gpu and not distributed and not slow and not requires_model"
```

GPU, distributed, container, and end-to-end model checks are separate release
gates. See [docs/reproducibility.md](docs/reproducibility.md).

## Project documentation

- [Distributed execution](docs/distributed.md)
- [GPU validation checklist](docs/gpu_validation.md)
- [Data format](docs/data_format.md)
- [Reproducibility](docs/reproducibility.md)
- [Source provenance](docs/provenance.md)
- [Contributing](CONTRIBUTING.md)
- [Security](SECURITY.md)
- [Support](SUPPORT.md)

## License

Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) and
[NOTICE.md](NOTICE.md). Model, dataset, tokenizer, container, and dependency
licenses apply separately.
