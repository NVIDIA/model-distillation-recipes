# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

ARG NGC_BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
FROM ${NGC_BASE_IMAGE}

ARG INSTALL_MAMBA_KERNELS=0

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/composite-venv \
    PATH="/opt/composite-venv/bin:${PATH}" \
    HOME=/tmp/gpu-home

RUN python -m venv --system-site-packages /opt/composite-venv \
    && /opt/composite-venv/bin/python -m pip install \
       "wandb==0.30.0" "datasets==5.0.1" "transformers>=5.6.0" \
    && /opt/composite-venv/bin/python -m pip install --no-deps \
       "peft==0.21.0" "accelerate==1.15.0" \
    && if [ "${INSTALL_MAMBA_KERNELS}" = "1" ]; then \
           /opt/composite-venv/bin/python -m pip install "ninja==1.13.2" \
           && CAUSAL_CONV1D_FORCE_BUILD=TRUE CAUSAL_CONV1D_SKIP_CUDA_BUILD=FALSE \
               /opt/composite-venv/bin/python -m pip install --no-deps --no-build-isolation \
               "causal-conv1d>=1.4.0" \
           && MAMBA_FORCE_BUILD=TRUE MAMBA_SKIP_CUDA_BUILD=FALSE \
               /opt/composite-venv/bin/python -m pip install --no-deps --no-build-isolation \
               "mamba-ssm>=2.2.4"; \
       fi \
    && printf '#!/bin/sh\nexec python -m torch.distributed.run "$@"\n' \
       > /opt/composite-venv/bin/torchrun \
    && chmod +x /opt/composite-venv/bin/torchrun \
    && mkdir -p /tmp/gpu-home \
    && chmod 1777 /tmp/gpu-home

WORKDIR /workspace/model-distillation-recipes

CMD ["bash"]
