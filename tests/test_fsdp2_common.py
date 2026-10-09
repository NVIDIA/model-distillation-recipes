# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from fsdp2_common import (
    IGNORE_TOKEN_ID,
    build_lambda_weights,
    load_chat_template,
    load_dcp_checkpoint,
    resolve_grad_accum,
    resolve_layer_cls,
    unit_interval_float,
    validate_chat_template,
    validate_distillation_compatibility,
    validate_resume_step,
    validate_tokenizer_compatibility,
)


@pytest.mark.parametrize(
    ("model_type", "expected"),
    [
        ("qwen3", "Qwen3DecoderLayer"),
        ("nemotron_h", "NemotronHBlock"),
    ],
)
def test_resolve_layer_cls(model_type, expected):
    model = SimpleNamespace(config=SimpleNamespace(model_type=model_type))
    assert resolve_layer_cls(model) == expected


def test_resolve_layer_cls_rejects_unknown_model():
    model = SimpleNamespace(config=SimpleNamespace(model_type="unknown"))
    with pytest.raises(ValueError, match="No known transformer layer class"):
        resolve_layer_cls(model)


@pytest.mark.parametrize("value", ["0", "0.5", "1"])
def test_unit_interval_float_accepts_bounds(value):
    assert unit_interval_float(value) == float(value)


@pytest.mark.parametrize("value", ["-0.1", "1.1"])
def test_unit_interval_float_rejects_out_of_range(value):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        unit_interval_float(value)


def test_blended_weights_normalize_answer_and_reasoning_pools():
    labels = torch.tensor(
        [[IGNORE_TOKEN_ID, 101, 102, 103, 201, 202]],
        dtype=torch.long,
    )
    weights = build_lambda_weights(
        labels,
        think_start_id=101,
        think_end_id=103,
        blended_loss_weight=0.25,
    )

    assert weights[0, :1].sum().item() == 0
    assert weights[0, 1:4].sum().item() == pytest.approx(0.75)
    assert weights[0, 4:].sum().item() == pytest.approx(0.25)
    assert weights.sum().item() == pytest.approx(1.0)


def test_load_chat_template_requires_existing_generation_template(tmp_path):
    tokenizer = SimpleNamespace(chat_template=None)
    missing = tmp_path / "missing.jinja"
    with pytest.raises(FileNotFoundError, match="Chat template not found"):
        load_chat_template(tokenizer, missing)

    invalid = tmp_path / "invalid.jinja"
    invalid.write_text("{{ message['content'] }}", encoding="utf-8")
    with pytest.raises(ValueError, match="assistant-only loss masking"):
        load_chat_template(tokenizer, invalid)

    valid = tmp_path / "valid.jinja"
    valid.write_text("{%- generation -%}answer{% endgeneration %}", encoding="utf-8")
    load_chat_template(tokenizer, valid)
    assert tokenizer.chat_template == valid.read_text(encoding="utf-8")


def test_effective_chat_template_is_required():
    with pytest.raises(ValueError, match="assistant-only loss masking"):
        validate_chat_template(None)
    with pytest.raises(ValueError, match="assistant-only loss masking"):
        validate_chat_template("{{ message['content'] }}")
    with pytest.raises(ValueError, match="assistant-only loss masking"):
        validate_chat_template("{% generation %}unclosed")


def test_distillation_requires_identical_token_ids_and_vocab_sizes():
    tokenizer = SimpleNamespace(get_vocab=lambda: {"a": 0, "b": 1})
    matching = SimpleNamespace(get_vocab=lambda: {"a": 0, "b": 1})
    remapped = SimpleNamespace(get_vocab=lambda: {"a": 1, "b": 0})
    student = SimpleNamespace(config=SimpleNamespace(vocab_size=2))
    teacher = SimpleNamespace(config=SimpleNamespace(vocab_size=2))

    assert validate_tokenizer_compatibility(tokenizer, matching) == {"a": 0, "b": 1}
    validate_distillation_compatibility(student, teacher, tokenizer, matching)
    validate_distillation_compatibility(
        SimpleNamespace(config=SimpleNamespace(vocab_size=4)),
        SimpleNamespace(config=SimpleNamespace(vocab_size=4)),
        tokenizer,
        matching,
    )

    with pytest.raises(ValueError, match="token-to-ID"):
        validate_tokenizer_compatibility(tokenizer, remapped)
    with pytest.raises(ValueError, match="output vocabulary size"):
        validate_distillation_compatibility(
            student,
            SimpleNamespace(config=SimpleNamespace(vocab_size=3)),
            tokenizer,
            matching,
        )
    with pytest.raises(ValueError, match="does not cover"):
        validate_distillation_compatibility(
            SimpleNamespace(config=SimpleNamespace(vocab_size=2)),
            teacher,
            SimpleNamespace(get_vocab=lambda: {"a": 0, "b": 2}),
            SimpleNamespace(get_vocab=lambda: {"a": 0, "b": 2}),
        )


@pytest.mark.parametrize(
    ("gbs", "mbs", "world_size", "explicit", "expected"),
    [
        (None, 2, 8, None, 1),
        (96, 2, 8, None, 6),
        (None, 2, 8, 3, 3),
        (48, 2, 8, 3, 3),
    ],
)
def test_resolve_grad_accum_accepts_exact_batches(gbs, mbs, world_size, explicit, expected):
    assert resolve_grad_accum(gbs, mbs, world_size, explicit=explicit) == expected


def test_resolve_grad_accum_rejects_non_divisible_gbs():
    with pytest.raises(ValueError, match="not divisible"):
        resolve_grad_accum(100, 2, 8)


def test_resolve_grad_accum_rejects_conflicting_flags():
    with pytest.raises(ValueError, match="conflicts"):
        resolve_grad_accum(64, 2, 8, explicit=3)


@pytest.mark.parametrize(
    ("gbs", "mbs", "world_size", "explicit"),
    [
        (0, 2, 8, None),
        (None, 0, 8, None),
        (None, 2, 0, None),
        (None, 2, 8, 0),
    ],
)
def test_resolve_grad_accum_rejects_non_positive_inputs(gbs, mbs, world_size, explicit):
    with pytest.raises(ValueError, match="positive"):
        resolve_grad_accum(gbs, mbs, world_size, explicit=explicit)


@pytest.mark.parametrize(("step", "max_steps"), [(0, 1000), (400, 1000), (1000, 1000)])
def test_validate_resume_step_accepts_completed_progress(step, max_steps):
    assert validate_resume_step(step, max_steps) == step


def test_validate_resume_step_rejects_step_after_total():
    with pytest.raises(ValueError, match="exceeds --max_steps"):
        validate_resume_step(1001, 1000)


def test_load_dcp_checkpoint_restores_completed_step(monkeypatch):
    import torch.distributed.checkpoint as dcp
    import torch.distributed.checkpoint.state_dict as dcp_state_dict

    model = object()
    optimizer = object()
    restored = {}

    monkeypatch.setattr(dcp_state_dict, "get_state_dict", lambda *_args: ({"m": 0}, {"o": 0}))

    def fake_load(state, checkpoint_id):
        assert state["step"] == 0
        assert checkpoint_id == "checkpoint"
        state["model"] = {"m": 1}
        state["optim"] = {"o": 2}
        state["step"] = 400

    def fake_set_state_dict(
        actual_model,
        actual_optimizer,
        *,
        model_state_dict,
        optim_state_dict,
    ):
        restored["model"] = actual_model
        restored["optimizer"] = actual_optimizer
        restored["model_state_dict"] = model_state_dict
        restored["optim_state_dict"] = optim_state_dict

    monkeypatch.setattr(dcp, "load", fake_load)
    monkeypatch.setattr(dcp_state_dict, "set_state_dict", fake_set_state_dict)

    assert load_dcp_checkpoint(model, optimizer, "checkpoint") == 400
    assert restored == {
        "model": model,
        "optimizer": optimizer,
        "model_state_dict": {"m": 1},
        "optim_state_dict": {"o": 2},
    }
