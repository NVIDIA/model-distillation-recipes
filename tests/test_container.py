# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_uses_public_pytorch_base():
    dockerfile = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
    assert "ARG NGC_BASE_IMAGE=nvcr.io/nvidia/pytorch:" in dockerfile
    assert "FROM ${NGC_BASE_IMAGE}" in dockerfile
    assert "ARG INSTALL_MAMBA_KERNELS=0" in dockerfile
    assert "WORKDIR /workspace/model-distillation-recipes" in dockerfile


def test_repository_has_no_registry_push_helper():
    assert not (ROOT / "docker" / "build_push.sh").exists()
