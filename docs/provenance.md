# Source provenance

This record identifies compatibility content that needs an explicit provenance
and publication decision in addition to the repository's Apache-2.0 license.
It is not a substitute for NVIDIA Open Source or Legal approval.

## Chat templates

The files under `configs/chat_template_*_generation.jinja` encode control-token
formats for Qwen3 and Nemotron-family tokenizers and add Transformers
`generation` blocks for assistant-only masking. Before public publication:

- identify the author and source revision for each template;
- compare it with the corresponding upstream tokenizer template;
- record whether it is original compatibility code, adapted material, or a
  verbatim copy;
- verify the upstream model/tokenizer license permits the resulting use and
  distribution; and
- retain any required attribution or notice.

These templates do not grant access to, or redistribution rights for, any model
or tokenizer.

## Imported implementation history

The initial repository commits imported training code from an internal research
branch. Release owners must retain an approved origin record, confirm
contributor rights, and review the complete source history for third-party or
proprietary material before creating the public history.

## Dependencies and containers

`NOTICE.md` lists direct dependencies but is not a container software bill of
materials. Exact wheel, source archive, base-image, transitive dependency, and
compiled-kernel provenance must be generated from the final release
environment and reviewed separately.
