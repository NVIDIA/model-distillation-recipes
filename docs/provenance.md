# Source provenance

This record identifies compatibility content that needs an explicit provenance
and publication decision in addition to the repository's Apache-2.0 license.
It is not a substitute for NVIDIA Open Source or Legal approval.

## Chat templates

The files under `configs/chat_template_*_generation.jinja` encode control-token
formats for Qwen3 and Nemotron-family tokenizers and add Transformers
`generation` blocks for assistant-only masking.

### Provenance record

On 2026-10-09, the release owner confirmed that Dhruv Desai, Lavinia Ghita, and
Liana Mikaelyan authored the three adaptations as NVIDIA work, using only the
following source templates:

- `chat_template_qwen3_generation.jinja` is adapted from the
  [Qwen3 tokenizer configuration at `cdbee75f`][qwen-template]. It keeps basic
  message framing, omits tool handling, and adds Transformers `generation`
  blocks. The source is Copyright 2024 Alibaba Cloud and licensed under
  [Apache-2.0][qwen-license]. The source file and `NOTICE.md` retain that
  copyright and attribution.
- `chat_template_nemotron3_generation.jinja` is adapted from the
  [Nemotron-3 template at `bf77c317`][nemotron3-template]. It keeps basic
  system/user/assistant framing and adds Transformers `generation` blocks.
  The source is NVIDIA-owned and the model repository identifies its terms as
  the [NVIDIA Nemotron Open Model License][nemotron3-license].
- `chat_template_nemotron9b_generation.jinja` is adapted from the
  [Nemotron Nano 9B v2 tokenizer configuration at `6533e8de`][nemotron9b-template].
  It keeps basic role framing and adds Transformers `generation` blocks. The
  source is NVIDIA-owned and the model repository identifies its terms as the
  [NVIDIA Open Model License][nemotron9b-license].

The resulting files are not verbatim copies. The authors confirmed that no
other source material was used. The NVIDIA-authored modifications are
distributed under this project's Apache-2.0 license, with the Qwen source
material remaining under Apache-2.0, subject to NVIDIA Open Source approval.

These templates do not grant access to, or redistribution rights for, any model
or tokenizer.

## Imported implementation history

The training implementation was introduced as one source snapshot in commit
`981a3c6fae0612310bc3c8ae0d9dbe3f8a611738`. On 2026-10-09, the release owner
confirmed that Dhruv Desai, Lavinia Ghita, and Liana Mikaelyan authored the
implementation as NVIDIA work and that no code was copied from other internal
or third-party projects. The source snapshot was created from that original
NVIDIA implementation; it is not an import of third-party or separately
licensed source code.

The snapshot commit is signed off by its NVIDIA submitter. Subsequent
publication commits must also carry the Developer Certificate of Origin
sign-off required by `CONTRIBUTING.md`. This authorship attestation remains
subject to NVIDIA Open Source approval.

## Dependencies and containers

`NOTICE.md` lists direct dependencies but is not a container software bill of
materials. Exact wheel, source archive, base-image, transitive dependency, and
compiled-kernel provenance must be generated from the final release
environment and reviewed separately.

[qwen-template]: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/cdbee75f17c01a7cc42f958dc650907174af0554/tokenizer_config.json
[qwen-license]: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/cdbee75f17c01a7cc42f958dc650907174af0554/LICENSE
[nemotron3-template]: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/bf77c3174f68ad409e1c2aa60daeb46e32d1c606/chat_template.jinja
[nemotron3-license]: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/
[nemotron9b-template]: https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2/blob/6533e8de2c68e4536bf7c411d7a3ce5734111476/tokenizer_config.json
[nemotron9b-license]: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
