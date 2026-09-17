#!/usr/bin/env python3
# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify FSDP2 chat dataloader: template, assistant mask, think tags, pool weights.

Example:
    python verify_fsdp2_dataloader.py \
        --tokenizer /path/to/model \
        --data /path/to/train.jsonl \
        --chat_template ../configs/chat_template_qwen3_generation.jinja \
        --blended_loss_weight 0.5 --max_samples 4
"""

from __future__ import annotations

import argparse
import sys

import torch
from fsdp2_common import (
    IGNORE_TOKEN_ID,
    ChatDataset,
    PadCollator,
    build_lambda_weights,
    load_chat_template,
    resolve_think_token_ids,
    unit_interval_float,
)
from transformers import AutoTokenizer


def get_args(argv=None):
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--chat_template", required=True)
    p.add_argument("--blended_loss_weight", type=unit_interval_float, default=None)
    p.add_argument("--max_seq_length", type=int, default=8192)
    p.add_argument("--max_samples", type=int, default=4)
    return p.parse_args(argv)


def main():
    args = get_args()
    print("=" * 72)
    print("FSDP2 dataloader verification")
    print("=" * 72)
    print(f"tokenizer: {args.tokenizer}")
    print(f"data:      {args.data}")
    print(f"template:  {args.chat_template}")
    print(f"blended_loss_weight: {args.blended_loss_weight}")

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    raw_has_gen = "{% generation %}" in (tok.chat_template or "")
    print(f"\n[1] default chat_template has {{% generation %}}? {raw_has_gen}")

    load_chat_template(tok, args.chat_template)
    loaded_has_gen = "{% generation %}" in (tok.chat_template or "")
    print(f"    after load has {{% generation %}}? {loaded_has_gen}")
    if not loaded_has_gen:
        print("FAIL: chat template missing {% generation %} — assistant mask will break")
        sys.exit(1)

    think_start = think_end = None
    if args.blended_loss_weight is not None:
        think_start, think_end = resolve_think_token_ids(tok)
        print(f"\n[2] think tokens: <think>={think_start} </think>={think_end}")
        print(f"    decode: {tok.decode([think_start])!r} / {tok.decode([think_end])!r}")
    else:
        print("\n[2] pool weighting disabled; think-token checks skipped")

    ds = ChatDataset(args.data, tok, args.max_seq_length, max_samples=args.max_samples)
    print(f"\n[3] ChatDataset loaded {len(ds)} samples")

    for i in range(min(len(ds), args.max_samples)):
        ex = ds[i]
        ids, labels = ex["input_ids"], ex["labels"]
        n_as = sum(1 for x in labels if x != IGNORE_TOKEN_ID)
        n_ig = sum(1 for x in labels if x == IGNORE_TOKEN_ID)
        start_pos = [j for j, t in enumerate(labels) if t == think_start]
        end_pos = [j for j, t in enumerate(labels) if t == think_end]
        print(
            f"    sample[{i}] len={len(ids)} assistant={n_as} ignored={n_ig} "
            f"think@{start_pos} end@{end_pos}"
        )
        if args.blended_loss_weight is not None and (not start_pos or not end_pos):
            print(
                "FAIL: <think>/</think> not present in LABELS "
                "(must be inside assistant region for answer/CoT pool weighting)"
            )
            sys.exit(1)
        if start_pos:
            i0 = start_pos[0]
            label_toks = [t for t in labels[max(0, i0 - 32) : i0] if t != IGNORE_TOKEN_ID]
            print(f"      answer-before-think: {tok.decode(label_toks)!r}")

    # Raw apply_chat_template key check
    enc = tok.apply_chat_template(
        ds.raw[0]["messages"],
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        max_length=args.max_seq_length,
        truncation=True,
    )
    mask = enc.get("assistant_masks")
    if mask is None:
        mask = enc.get("assistant_tokens_mask")
    print(f"\n[4] apply_chat_template keys: {sorted(enc.keys())}")
    if mask is None or sum(mask) == 0:
        print("FAIL: assistant mask missing/empty")
        sys.exit(1)
    print(f"    mask sum={sum(mask)} / {len(mask)}")

    if args.blended_loss_weight is None:
        print("\nOK: chat template and assistant mask look correct")
        return

    # Collate + pool weights (same as training: causal shift then weight)
    collate = PadCollator(tok.pad_token_id)
    batch = collate([ds[i] for i in range(min(2, len(ds)))])
    shift_labels = batch["labels"][:, 1:].contiguous()
    weights = build_lambda_weights(shift_labels, think_start, think_end, args.blended_loss_weight)

    is_valid = shift_labels != IGNORE_TOKEN_ID
    starts = (shift_labels == think_start).long()
    ends = (shift_labels == think_end).long()
    ends_shifted = torch.zeros_like(ends)
    ends_shifted[:, 1:] = ends[:, :-1]
    in_think = starts.cumsum(dim=-1) > ends_shifted.cumsum(dim=-1)
    is_cot = in_think & is_valid
    is_label = (~in_think) & is_valid

    print(f"\n[5] pool weights (blended_loss_weight={args.blended_loss_weight})")
    print(f"    batch shapes: { {k: tuple(v.shape) for k, v in batch.items()} }")
    print(f"    n_label={is_label.sum().item()} n_cot={is_cot.sum().item()}")
    print(
        f"    sum(w_label)={weights[is_label].sum().item():.4f} "
        f"sum(w_cot)={weights[is_cot].sum().item():.4f} "
        f"(expect ~{args.blended_loss_weight:.2f} / ~{1 - args.blended_loss_weight:.2f})"
    )
    print(f"    total sum(w)={weights.sum().item():.4f} (expect ~1.0)")

    # Sanity: ignored positions must have zero weight
    if (weights[~is_valid] != 0).any():
        print("FAIL: non-zero weight on ignored positions")
        sys.exit(1)

    # Sanity: pool sums close to blended_loss_weight / (1-blended_loss_weight)
    sl = weights[is_label].sum().item()
    sc = weights[is_cot].sum().item()
    if (
        abs(sl - args.blended_loss_weight) > 1e-3
        or abs(sc - (1.0 - args.blended_loss_weight)) > 1e-3
    ):
        print("FAIL: pool weight sums do not match blended_loss_weight / (1-blended_loss_weight)")
        sys.exit(1)

    print("\nOK: chat template, assistant mask, think tags, and pool weights look correct")


if __name__ == "__main__":
    main()
