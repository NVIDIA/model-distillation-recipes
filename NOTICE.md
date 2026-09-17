# Third-Party Notices

Model Distillation Recipes does not intentionally vendor third-party library
source code. Compatibility templates require the separate provenance review
documented in `docs/provenance.md`. The source repository declares the following
direct build, development, runtime, and optional dependencies in
`pyproject.toml`; they are installed separately under their respective licenses:

- [`setuptools`](https://github.com/pypa/setuptools) — MIT.
- [`accelerate`](https://github.com/huggingface/accelerate) — Apache-2.0.
- [`datasets`](https://github.com/huggingface/datasets) — Apache-2.0.
- [`peft`](https://github.com/huggingface/peft) — Apache-2.0.
- [`torch`](https://github.com/pytorch/pytorch) — BSD-3-Clause for the PyTorch
  project; distributions include third-party components under additional
  licenses.
- [`transformers`](https://github.com/huggingface/transformers) — Apache-2.0.
- [`pytest`](https://github.com/pytest-dev/pytest) — MIT.
- [`ruff`](https://github.com/astral-sh/ruff) — MIT.
- [`wandb`](https://github.com/wandb/wandb) — MIT.
- [`causal-conv1d`](https://github.com/Dao-AILab/causal-conv1d) —
  BSD-3-Clause.
- [`mamba-ssm`](https://github.com/state-spaces/mamba) — Apache-2.0.
- [`ninja`](https://github.com/ninja-build/ninja) — Apache-2.0.
- [`flash-attn`](https://github.com/Dao-AILab/flash-attention) — BSD-3-Clause
  when the optional FlashAttention backend is installed outside the NGC base.

These dependencies are not redistributed by the source repository. The
license files, notices, metadata, and terms accompanying the versions actually
installed are authoritative and may differ as dependencies and their bundled
components evolve.

## Container Contents

The supplied Dockerfile builds on an NVIDIA NGC PyTorch base image and installs
additional Python packages. A built container therefore redistributes the base
image contents, installed packages, transitive dependencies, CUDA components,
and any optional compiled extensions; those contents are distinct from the
source repository dependencies listed above.

This file is not a software bill of materials for a built container. Anyone
distributing a container must inventory the exact image by immutable digest
and preserve all licenses, notices, attributions, source-offer obligations,
and other terms delivered with the base image and every installed component.
NVIDIA NGC access and use are also subject to the applicable NGC terms.

Models, model weights, tokenizers, datasets, and generated artifacts are not
included in this notice and remain subject to their own licenses and terms.
