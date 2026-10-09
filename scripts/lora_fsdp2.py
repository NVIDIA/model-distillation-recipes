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
Raw PyTorch + torch.distributed FSDP2 LoRA fine-tuning.

Loads a base/pruned HF model, applies PEFT LoRA, shards with FSDP2, and runs a manual training loop
with cross-entropy loss (optionally answer/CoT pool weighting for merged data).
Saves the LoRA adapter in PEFT format.

Launch with torchrun:
    torchrun --nnodes $NNODES --nproc_per_node 4 \\
        --node_rank $RANK --master_addr $ADDR --master_port 29500 \\
        lora_fsdp2.py --model_path ... --data_path ...
"""

import argparse
import logging
import os
import random

import torch
import torch.distributed as dist
import torch.nn.functional as F
from fsdp2_common import (
    IGNORE_TOKEN_ID,
    build_lambda_weights,
    cosine_lr,
    dist_cleanup,
    dist_setup,
    finish_wandb,
    fsdp2_wrap,
    init_wandb,
    is_main_process,
    load_dcp_checkpoint,
    log_train_step,
    make_chat_dataloader,
    masked_mean,
    next_batch,
    positive_int,
    rank0_print,
    reduce_mean_scalar,
    resolve_grad_accum,
    resolve_layer_cls,
    resolve_think_token_ids,
    save_dcp_checkpoint,
    setup_tokenizer,
    unit_interval_float,
    validate_resume_step,
)
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def save_lora_adapter(model, tokenizer, output_dir):
    """Gather full state dict, extract LoRA adapter on rank 0, save PEFT format."""
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
    )

    # Collective: must run on every rank. With full_state_dict + cpu_offload
    # the unsharded weights are materialized on rank 0 (CPU) only.
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    full_sd = get_model_state_dict(model, options=options)

    if is_main_process():
        os.makedirs(output_dir, exist_ok=True)
        # Hand the raw (base + LoRA) state dict to PeftModel.save_pretrained and
        # let PEFT run get_peft_model_state_dict exactly ONCE internally. Do NOT
        # pre-extract the adapter here: get_peft_model_state_dict strips the
        # ".default" adapter suffix from keys, and PEFT's second pass filters for
        # keys containing both "lora_" and the adapter name, so a pre-extracted
        # dict would match nothing and silently save an empty adapter.
        model.save_pretrained(output_dir, state_dict=full_sd, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
        rank0_print(f"[ckpt] Saved LoRA adapter to {output_dir}")
    if dist.is_initialized():
        dist.barrier()


def get_args(argv=None):
    p = argparse.ArgumentParser(
        description="Raw FSDP2 LoRA fine-tuning.",
        allow_abbrev=False,
    )
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--name", type=str, default=None)
    p.add_argument(
        "--attn_implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="sdpa",
        help="Hugging Face attention backend used when loading the model.",
    )

    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--max_seq_length", type=int, default=8192)
    p.add_argument("--chat_template_path", type=str, default=None)
    p.add_argument("--max_samples", type=int, default=None)

    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument(
        "--lora_target_modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    p.add_argument(
        "--blended_loss_weight",
        type=unit_interval_float,
        default=None,
        help="Split assistant tokens into answer vs <think> reasoning "
        "pools and average each separately, so the weighting is "
        "independent of their relative lengths. 1.0 = answer only, "
        "0.5 = both pools count equally, 0.0 = reasoning only. "
        "Unset = plain per-token mean over all assistant tokens.",
    )

    p.add_argument("--max_steps", type=int, required=True)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gbs", type=int, default=None)
    p.add_argument("--gradient_accumulation_steps", type=int, default=None)

    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    p.add_argument("--save_steps", type=positive_int, default=None)
    p.add_argument("--dcp_resume_path", type=str, default=None)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_name", type=str, default=None)
    p.add_argument("--wandb_tags", nargs="+", default=None)

    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args(argv)


def main():
    args = get_args()
    rank, local_rank, world_size = dist_setup()
    device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    grad_accum = resolve_grad_accum(
        args.gbs,
        args.per_device_train_batch_size,
        world_size,
        explicit=args.gradient_accumulation_steps,
    )
    global_batch_size = args.per_device_train_batch_size * world_size * grad_accum
    rank0_print(f"grad_accum={grad_accum}")

    use_wandb = init_wandb(args)

    tokenizer = setup_tokenizer(args.model_path, args.chat_template_path)
    _, sampler, loader = make_chat_dataloader(
        args.data_path,
        tokenizer,
        args.max_seq_length,
        args.max_samples,
        args.per_device_train_batch_size,
        world_size,
        rank,
        args.seed,
    )

    think_start_id = think_end_id = None
    if args.blended_loss_weight is not None:
        think_start_id, think_end_id = resolve_think_token_ids(tokenizer)
        rank0_print(f"Answer/CoT pool weighting: blended_loss_weight={args.blended_loss_weight}")

    rank0_print(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False

    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            bias="none",
        ),
    )
    if is_main_process():
        model.print_trainable_parameters()

    model = fsdp2_wrap(model, resolve_layer_cls(model))

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )

    step = 0
    if args.dcp_resume_path:
        step = validate_resume_step(
            load_dcp_checkpoint(model, optimizer, args.dcp_resume_path),
            args.max_steps,
        )

    model.train()
    data_iter = iter(loader)
    epoch = 0
    rank0_print(f"Starting LoRA training at completed step {step} of {args.max_steps}")

    while step < args.max_steps:
        lr = cosine_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _micro in range(grad_accum):
            batch, data_iter, epoch = next_batch(data_iter, loader, sampler, epoch)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            ce_tok = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)).float(),
                shift_labels.reshape(-1),
                reduction="none",
                ignore_index=IGNORE_TOKEN_ID,
            ).view(shift_labels.shape)

            if args.blended_loss_weight is not None:
                weights = build_lambda_weights(
                    shift_labels, think_start_id, think_end_id, args.blended_loss_weight
                )
                loss = (ce_tok * weights).sum()
            else:
                loss = masked_mean(ce_tok, shift_labels)

            loss = loss / grad_accum
            loss.backward()
            accum_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optimizer.step()
        step += 1

        log_train_step(
            use_wandb,
            loss=reduce_mean_scalar(accum_loss, device),
            lr=lr,
            grad_norm=float(grad_norm),
            step=step,
            max_steps=args.max_steps,
            global_batch_size=global_batch_size,
            logging_steps=args.logging_steps,
        )

        if args.save_steps and step % args.save_steps == 0 and step < args.max_steps:
            save_dcp_checkpoint(model, optimizer, step, os.path.join(args.output_dir, "dcp"))

    rank0_print("Training complete. Saving LoRA adapter...")
    save_lora_adapter(model, tokenizer, args.output_dir)
    finish_wandb(use_wandb)
    dist_cleanup()
    rank0_print("Done.")


if __name__ == "__main__":
    try:
        main()
    finally:
        dist_cleanup()
