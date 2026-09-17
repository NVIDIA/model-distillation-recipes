# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Shared utilities for raw PyTorch + torch.distributed FSDP2 training scripts.

Provides:
  - Distributed init/teardown (torchrun-driven)
  - Chat data loading + tokenization with assistant-only loss masking
  - Answer/reasoning pool-weighted loss helpers
  - FSDP2 (fully_shard) model wrapping with bf16 mixed precision
  - Cosine LR schedule with warmup + min_lr
  - Checkpointing: DCP (sharded, for resume) + full-state-dict HF export
  - Shared training-loop helpers (dataloader, wandb, grad-accum, logging)
"""

import logging
import math
import os
import re
from pathlib import Path

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

IGNORE_TOKEN_ID = -100
QWEN3_LAYER = "Qwen3DecoderLayer"
NEMOTRONH_LAYER = "NemotronHBlock"

LAYER_CLS_BY_MODEL_TYPE = {
    "qwen3": QWEN3_LAYER,
    "nemotron_h": NEMOTRONH_LAYER,
}
GENERATION_BLOCK_RE = re.compile(r"{%-?\s*generation\s*-?%}")
GENERATION_END_BLOCK_RE = re.compile(r"{%-?\s*endgeneration\s*-?%}")


def resolve_layer_cls(model):
    """Return the FSDP2 layer class name for a supported HF model."""
    model_type = getattr(model.config, "model_type", None)
    layer_cls = LAYER_CLS_BY_MODEL_TYPE.get(model_type)
    if layer_cls is None:
        raise ValueError(
            f"No known transformer layer class for model_type={model_type!r}; "
            f"add it to LAYER_CLS_BY_MODEL_TYPE in fsdp2_common.py"
        )
    return layer_cls


def unit_interval_float(value):
    """Parse a floating-point CLI value constrained to [0, 1]."""
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"expected a value in [0, 1], got {value!r}")
    return parsed


def validate_tokenizer_compatibility(student_tokenizer, teacher_tokenizer):
    """Require identical token IDs for logit-level distillation."""
    student_vocab = student_tokenizer.get_vocab()
    teacher_vocab = teacher_tokenizer.get_vocab()
    if student_vocab != teacher_vocab:
        raise ValueError(
            "Student and teacher tokenizers must have identical token-to-ID "
            "vocabularies for logit-level distillation."
        )
    return student_vocab


def validate_distillation_compatibility(
    student_model,
    teacher_model,
    student_tokenizer,
    teacher_tokenizer,
):
    """Require identical token IDs and output vocabulary for logit-level KD."""
    student_vocab = validate_tokenizer_compatibility(
        student_tokenizer,
        teacher_tokenizer,
    )

    student_vocab_size = getattr(student_model.config, "vocab_size", None)
    teacher_vocab_size = getattr(teacher_model.config, "vocab_size", None)
    if student_vocab_size != teacher_vocab_size:
        raise ValueError(
            "Student and teacher models must have the same output vocabulary "
            f"size, got {student_vocab_size} and {teacher_vocab_size}."
        )
    token_ids = list(student_vocab.values())
    if not token_ids or not all(
        isinstance(token_id, int) and token_id >= 0 for token_id in token_ids
    ):
        raise ValueError("Tokenizer vocabulary must contain non-negative integer token IDs.")
    required_vocab_size = max(token_ids) + 1
    if student_vocab_size < required_vocab_size:
        raise ValueError(
            "Model output vocabulary does not cover every tokenizer ID, "
            f"got model size {student_vocab_size} but token ID "
            f"{required_vocab_size - 1} is present."
        )


# =============================================================================
# Distributed setup (torchrun sets RANK / LOCAL_RANK / WORLD_SIZE)
# =============================================================================


def dist_setup():
    """Initialize the default process group from torchrun env vars."""
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    return rank, local_rank, world_size


def dist_cleanup():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def rank0_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs, flush=True)


# =============================================================================
# Pool-weighted loss for merged answer+reasoning data
# =============================================================================


def resolve_think_token_ids(tokenizer):
    """Single-token IDs for <think> / </think>.

    Requires both markers to be dedicated tokens in the vocabulary (Qwen3 and
    Nemotron-3 register them; Nemotron-Nano-9B-v2 does not). Multi-token markers
    are rejected rather than matched as subsequences, because BPE merges them
    with surrounding whitespace -- e.g. on the 9B the trailing '>' merges with
    the following newline into '>\n', so the standalone encoding never appears
    in real text and the CoT pool silently comes out empty.
    """
    start = tokenizer.encode("<think>", add_special_tokens=False)
    end = tokenizer.encode("</think>", add_special_tokens=False)
    if len(start) != 1:
        raise ValueError(
            f"<think> must be a single token, got {start}. This tokenizer has no "
            f"dedicated <think> token, so --blended_loss_weight cannot reliably "
            f"locate reasoning spans; leave it unset for this model."
        )
    if len(end) != 1:
        raise ValueError(
            f"</think> must be a single token, got {end}. This tokenizer has no "
            f"dedicated </think> token, so --blended_loss_weight cannot reliably "
            f"locate reasoning spans; leave it unset for this model."
        )
    logger.info(f"Resolved <think>={start[0]}, </think>={end[0]}")
    return start[0], end[0]


def build_lambda_weights(labels, think_start_id, think_end_id, blended_loss_weight):
    """Per-token weights splitting assistant tokens into answer vs CoT pools.

    Each pool is averaged separately, so the split is independent of how long
    each span is -- a long <think> block cannot drown out a short answer:
        sum(weights * per_token_loss)
            = blended_loss_weight * mean(answer) + (1 - blended_loss_weight) * mean(cot)

    blended_loss_weight: 1.0 = answer only, 0.5 = both pools count equally (regardless of
    length), 0.0 = reasoning only.

    labels: (B, S) with IGNORE_TOKEN_ID for non-assistant positions.
    """
    is_valid = labels != IGNORE_TOKEN_ID

    starts = (labels == think_start_id).long()
    ends = (labels == think_end_id).long()
    # Shift by one so </think> itself counts as inside the span.
    ends_shifted = torch.zeros_like(ends)
    ends_shifted[:, 1:] = ends[:, :-1]
    in_think = starts.cumsum(dim=-1) > ends_shifted.cumsum(dim=-1)

    is_cot = in_think & is_valid
    is_answer = ~in_think & is_valid

    n_answer = is_answer.sum().float().clamp(min=1)
    n_cot = is_cot.sum().float().clamp(min=1)

    return is_answer.float() * (blended_loss_weight / n_answer) + is_cot.float() * (
        (1.0 - blended_loss_weight) / n_cot
    )


def masked_mean(per_token, labels):
    """Mean over assistant (non-ignored) positions."""
    mask = (labels != IGNORE_TOKEN_ID).float()
    return (per_token * mask).sum() / mask.sum().clamp(min=1)


def weighted_reduce(per_token, weights):
    """sum(per_token * weights) — weights already normalize per pool."""
    return (per_token * weights).sum()


# =============================================================================
# Chat data: load + tokenize with assistant-only masking
# =============================================================================


def _find_jsonl(data_path, names):
    p = Path(data_path)
    if p.is_file():
        return str(p)
    for n in names:
        cand = p / n
        if cand.exists():
            return str(cand)
    globbed = sorted(str(f) for f in p.glob("*.jsonl"))
    return globbed[0] if globbed else None


def load_chat_template(tokenizer, chat_template_path):
    """Override the tokenizer's chat template from a file if provided."""
    if not chat_template_path:
        return
    if not os.path.isfile(chat_template_path):
        raise FileNotFoundError(f"Chat template not found: {chat_template_path}")
    with open(chat_template_path, encoding="utf-8") as f:
        chat_template = f.read()
    validate_chat_template(chat_template, str(chat_template_path))
    tokenizer.chat_template = chat_template
    logger.info(f"Loaded chat template from {chat_template_path}")


def validate_chat_template(chat_template, source="tokenizer"):
    """Require generation blocks so assistant-only masking is available."""
    has_complete_generation_block = (
        chat_template
        and GENERATION_BLOCK_RE.search(chat_template)
        and GENERATION_END_BLOCK_RE.search(chat_template)
    )
    if not has_complete_generation_block:
        raise ValueError(
            f"Chat template from {source} has no complete {{% generation %}} block; "
            "assistant-only loss masking would be empty."
        )


class ChatDataset(torch.utils.data.Dataset):
    """Tokenizes chat JSONL ({'messages': [...]}) with assistant-only masking.

    Uses tokenizer.apply_chat_template(..., return_assistant_tokens_mask=True),
    which requires a {% generation %} block in the chat template. Non-assistant
    tokens get label = IGNORE_TOKEN_ID.
    """

    def __init__(self, data_path, tokenizer, max_seq_length, max_samples=None):
        import datasets

        jsonl = _find_jsonl(data_path, ["training.jsonl", "train.jsonl"])
        if jsonl is None:
            raise FileNotFoundError(f"No JSONL found in {data_path}")
        logger.info(f"Loading chat data from {jsonl}")
        raw = datasets.load_dataset("json", data_files=jsonl, split="train")
        if "messages" not in raw.column_names:
            raise ValueError(f"Dataset needs 'messages' column, found {raw.column_names}")
        if max_samples:
            raw = raw.select(range(min(max_samples, len(raw))))

        self.raw = raw
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx):
        messages = self.raw[idx]["messages"]
        enc = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            max_length=self.max_seq_length,
            truncation=True,
        )
        input_ids = enc["input_ids"]
        assistant_mask = enc.get("assistant_masks")
        if assistant_mask is None:
            assistant_mask = enc.get("assistant_tokens_mask")
        if assistant_mask is None:
            raise ValueError(
                "Tokenizer did not return an assistant mask. Use a chat template "
                "with a {% generation %} block."
            )
        if not any(assistant_mask):
            raise ValueError(
                f"Sample {idx} has no assistant tokens after chat-template "
                "rendering and truncation."
            )
        labels = [tok if m == 1 else IGNORE_TOKEN_ID for tok, m in zip(input_ids, assistant_mask)]
        return {"input_ids": input_ids, "labels": labels}


class PadCollator:
    """Right-pads a batch to the longest sequence; builds attention mask."""

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, labels, attn = [], [], []
        for x in batch:
            n = len(x["input_ids"])
            pad = max_len - n
            input_ids.append(x["input_ids"] + [self.pad_token_id] * pad)
            labels.append(x["labels"] + [IGNORE_TOKEN_ID] * pad)
            attn.append([1] * n + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


# =============================================================================
# FSDP2 wrapping
# =============================================================================


def fsdp2_wrap(
    model,
    transformer_layer_cls,
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    reshard_after_forward=True,
):
    """Shard a model with FSDP2 (fully_shard).

    Wraps each transformer decoder layer individually, then the root module.
    param_dtype: compute/param dtype (bf16); reduce_dtype: grad reduction (fp32).
    """
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
    )

    # Shard each decoder layer first (enables per-layer all-gather/reshard)
    matched_layers = 0
    for module in model.modules():
        if type(module).__name__ == transformer_layer_cls:
            fully_shard(module, mp_policy=mp_policy, reshard_after_forward=reshard_after_forward)
            matched_layers += 1
    if matched_layers == 0:
        raise ValueError(
            f"No {transformer_layer_cls} modules found in "
            f"{type(model).__name__}; refusing to shard only the root module."
        )

    # Shard the root
    fully_shard(model, mp_policy=mp_policy, reshard_after_forward=reshard_after_forward)
    return model


# =============================================================================
# LR schedule
# =============================================================================


def cosine_lr(step, warmup_steps, max_steps, peak_lr, min_lr):
    """Cosine decay from peak_lr to min_lr with linear warmup."""
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (peak_lr - min_lr) * cos


# =============================================================================
# Checkpointing
# =============================================================================


def save_dcp_checkpoint(model, optimizer, step, ckpt_dir):
    """Save a sharded Distributed Checkpoint for resume."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict

    model_sd, optim_sd = get_state_dict(model, optimizer)
    state = {"model": model_sd, "optim": optim_sd, "step": step}
    path = os.path.join(ckpt_dir, f"step_{step}")
    dcp.save(state, checkpoint_id=path)
    rank0_print(f"[ckpt] Saved DCP checkpoint to {path}")
    return path


def load_dcp_checkpoint(model, optimizer, path):
    """Restore model+optimizer from a Distributed Checkpoint."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    model_sd, optim_sd = get_state_dict(model, optimizer)
    state = {"model": model_sd, "optim": optim_sd}
    dcp.load(state, checkpoint_id=path)
    set_state_dict(
        model, optimizer, model_state_dict=state["model"], optim_state_dict=state["optim"]
    )
    rank0_print(f"[ckpt] Loaded DCP checkpoint from {path}")


def save_hf_checkpoint(model, tokenizer, output_dir):
    """Gather full state dict on rank 0 and save as HF safetensors.

    Works with FSDP2-sharded models. `model` should be the underlying HF model
    (unwrap PEFT first if applicable before calling).
    """
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
    )

    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    full_sd = get_model_state_dict(model, options=options)

    if is_main_process():
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(
            output_dir,
            state_dict=full_sd,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(output_dir)
        rank0_print(f"[ckpt] Saved HF checkpoint to {output_dir}")
    if dist.is_initialized():
        dist.barrier()


# =============================================================================
# Shared training-loop helpers (used by lora_fsdp2.py / distill_fsdp2.py)
# =============================================================================


def setup_tokenizer(pretrained_path, chat_template_path=None):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(pretrained_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    load_chat_template(tokenizer, chat_template_path)
    validate_chat_template(
        tokenizer.chat_template,
        chat_template_path or f"tokenizer {pretrained_path}",
    )
    return tokenizer


def make_chat_dataloader(
    data_path, tokenizer, max_seq_length, max_samples, batch_size, world_size, rank, seed
):
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    dataset = ChatDataset(data_path, tokenizer, max_seq_length, max_samples)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=PadCollator(tokenizer.pad_token_id),
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    rank0_print(f"Dataset: {len(dataset)} samples")
    return dataset, sampler, loader


def resolve_grad_accum(gbs, mbs, world_size, explicit=None):
    if explicit is not None:
        return max(1, explicit)
    if gbs is not None:
        return max(1, gbs // (mbs * world_size))
    return 1


def next_batch(data_iter, loader, sampler, epoch):
    try:
        return next(data_iter), data_iter, epoch
    except StopIteration:
        epoch += 1
        sampler.set_epoch(epoch)
        data_iter = iter(loader)
        return next(data_iter), data_iter, epoch


def init_wandb(args):
    """Init WandB on rank 0. Returns True if logging is active."""
    if not (
        getattr(args, "wandb_project", None)
        and os.environ.get("WANDB_API_KEY")
        and is_main_process()
    ):
        return False
    import wandb

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name or getattr(args, "name", None),
        tags=args.wandb_tags,
        config=vars(args),
    )
    # global_step is 0-indexed; use it as the wandb x-axis for all metrics.
    wandb.define_metric("global_step")
    wandb.define_metric("*", step_metric="global_step")
    return True


def finish_wandb(use_wandb):
    if use_wandb:
        import wandb

        wandb.finish()


def reduce_mean_scalar(value, device):
    t = torch.tensor(value, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return t.item()


def log_train_step(
    use_wandb, *, loss, lr, grad_norm, step, max_steps, global_batch_size, logging_steps
):
    global_step = step - 1
    if use_wandb:
        import wandb

        wandb.log(
            {
                "reduced_train_loss": loss,
                "lr": lr,
                "grad_norm": grad_norm,
                "consumed_samples": step * global_batch_size,
                "global_batch_size": global_batch_size,
                "global_step": global_step,
            },
            step=global_step,
        )
    if step % logging_steps == 0 or step == 1:
        rank0_print(
            f"step {step}/{max_steps} | loss {loss:.4f} | grad_norm {grad_norm:.3f} | lr {lr:.2e}"
        )
