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
Raw PyTorch + torch.distributed FSDP2 Knowledge Distillation.

Loads teacher and student as plain HF AutoModelForCausalLM, shards both with FSDP2 (teacher
frozen), and runs a manual training loop computing KL-divergence distillation
loss (optionally combined with LM cross-entropy), with answer/CoT pool
weighting for merged data.

Launch with torchrun:
    torchrun --nnodes $NNODES --nproc_per_node 4 \\
        --node_rank $RANK --master_addr $ADDR --master_port 29500 \\
        distill_fsdp2.py --student_path ... --teacher_path ... --data_path ...
"""

import argparse
import logging
import os
import random

import torch
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
    save_hf_checkpoint,
    setup_tokenizer,
    unit_interval_float,
    validate_distillation_compatibility,
    validate_resume_step,
    validate_tokenizer_compatibility,
    weighted_reduce,
)
from transformers import AutoModelForCausalLM

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def kd_kl_per_token(student_logits, teacher_logits, temperature):
    """Per-token KL divergence KL(teacher || student), summed over vocab.

    KL on temperature-softened logits, scaled by T^2.
    Returns (B, S) tensor.
    """
    s = F.log_softmax(student_logits.float() / temperature, dim=-1)
    t = F.softmax(teacher_logits.float() / temperature, dim=-1)
    kl = F.kl_div(s, t, reduction="none").sum(dim=-1)
    return kl * (temperature**2)


# =============================================================================
# CLI
# =============================================================================


def get_args(argv=None):
    p = argparse.ArgumentParser(
        description="Raw FSDP2 knowledge distillation.",
        allow_abbrev=False,
    )
    p.add_argument("--student_path", type=str, required=True)
    p.add_argument("--teacher_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--name", type=str, default=None)
    p.add_argument(
        "--attn_implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="sdpa",
        help="Hugging Face attention backend used for student and teacher.",
    )

    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--max_seq_length", type=int, default=8192)
    p.add_argument("--chat_template_path", type=str, default=None)
    p.add_argument("--max_samples", type=int, default=None)

    # KD
    p.add_argument("--kd_temperature", type=float, default=1.0)
    p.add_argument("--kd_loss_scale", type=float, default=1.0)
    p.add_argument(
        "--use_lm_loss",
        action="store_true",
        help="Add label cross-entropy on top of the KD loss. Default is "
        "KD-only; pass this to train with CE + kd_loss_scale * KD.",
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

    # Training
    p.add_argument("--max_steps", type=int, required=True)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gbs", type=int, default=None, help="Global batch size")
    p.add_argument("--gradient_accumulation_steps", type=int, default=None)

    # Optimizer
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--min_lr", type=float, default=5e-7)
    p.add_argument("--warmup_steps", type=int, default=3)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.95)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # Checkpointing
    p.add_argument("--save_steps", type=positive_int, default=None)
    p.add_argument("--dcp_resume_path", type=str, default=None)

    # Logging
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_name", type=str, default=None)
    p.add_argument("--wandb_tags", nargs="+", default=None)

    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args(argv)


# =============================================================================
# Main
# =============================================================================


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
    rank0_print(
        f"grad_accum={grad_accum} (gbs={args.gbs}, mbs={args.per_device_train_batch_size}, world={world_size})"
    )

    use_wandb = init_wandb(args)

    student_tokenizer = setup_tokenizer(args.student_path, args.chat_template_path)
    teacher_tokenizer = setup_tokenizer(args.teacher_path, args.chat_template_path)
    tokenizer = student_tokenizer
    validate_tokenizer_compatibility(student_tokenizer, teacher_tokenizer)
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
        rank0_print(
            f"Answer/CoT pool weighting: blended_loss_weight={args.blended_loss_weight}, use_lm_loss={args.use_lm_loss}"
        )

    rank0_print(f"Loading student from {args.student_path}")
    student = AutoModelForCausalLM.from_pretrained(
        args.student_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )
    student.config.use_cache = False

    rank0_print(f"Loading teacher from {args.teacher_path}")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )
    teacher.config.use_cache = False
    teacher.eval()
    for prm in teacher.parameters():
        prm.requires_grad_(False)
    validate_distillation_compatibility(
        student,
        teacher,
        student_tokenizer,
        teacher_tokenizer,
    )
    student = fsdp2_wrap(student, resolve_layer_cls(student))
    teacher = fsdp2_wrap(teacher, resolve_layer_cls(teacher), reshard_after_forward=True)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )

    step = 0
    if args.dcp_resume_path:
        step = validate_resume_step(
            load_dcp_checkpoint(student, optimizer, args.dcp_resume_path),
            args.max_steps,
        )

    student.train()
    data_iter = iter(loader)
    epoch = 0
    rank0_print(f"Starting training at completed step {step} of {args.max_steps}")

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

            with torch.no_grad():
                teacher_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits
            student_logits = student(input_ids=input_ids, attention_mask=attention_mask).logits

            s_shift = student_logits[:, :-1, :].contiguous()
            t_shift = teacher_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # kd_tok = T^2 * KL(softmax(teacher/T) || log_softmax(student/T))
            kd_tok = kd_kl_per_token(s_shift, t_shift, args.kd_temperature)
            if args.blended_loss_weight is not None:
                weights = build_lambda_weights(
                    shift_labels, think_start_id, think_end_id, args.blended_loss_weight
                )
                kd_loss = weighted_reduce(kd_tok, weights)
            else:
                kd_loss = masked_mean(kd_tok, shift_labels)

            # Default is KD-only. --use_lm_loss adds CE + kd_loss_scale * KD (same mask/pool weights).
            if args.use_lm_loss:
                ce_tok = F.cross_entropy(
                    s_shift.reshape(-1, s_shift.size(-1)).float(),
                    shift_labels.reshape(-1),
                    reduction="none",
                    ignore_index=IGNORE_TOKEN_ID,
                ).view(shift_labels.shape)
                if args.blended_loss_weight is not None:
                    lm_loss = weighted_reduce(ce_tok, weights)
                else:
                    lm_loss = masked_mean(ce_tok, shift_labels)
                loss = lm_loss + args.kd_loss_scale * kd_loss
            else:
                loss = kd_loss

            loss = loss / grad_accum
            loss.backward()
            accum_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
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
            save_dcp_checkpoint(student, optimizer, step, os.path.join(args.output_dir, "dcp"))

    rank0_print("Training complete. Saving HF checkpoint...")
    save_hf_checkpoint(student, tokenizer, args.output_dir)
    finish_wandb(use_wandb)
    dist_cleanup()
    rank0_print("Done.")


if __name__ == "__main__":
    try:
        main()
    finally:
        dist_cleanup()
