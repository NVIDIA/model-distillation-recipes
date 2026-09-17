# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize(
    "template_name",
    [
        "chat_template_qwen3_generation.jinja",
        "chat_template_nemotron3_generation.jinja",
        "chat_template_nemotron9b_generation.jinja",
    ],
)
def test_chat_template_marks_assistant_generation(template_name):
    template = (CONFIG_DIR / template_name).read_text(encoding="utf-8")
    assert "{% generation %}" in template
    assert "{% endgeneration %}" in template
    assert "assistant" in template.lower()


def test_nemotron9b_template_uses_model_control_tokens():
    template = (CONFIG_DIR / "chat_template_nemotron9b_generation.jinja").read_text(
        encoding="utf-8"
    )
    assert "<SPECIAL_10>" in template
    assert "<SPECIAL_11>" in template
    assert "<SPECIAL_12>" in template
