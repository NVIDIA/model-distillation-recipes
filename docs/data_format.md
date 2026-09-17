# Data format

## Training records

Training input is UTF-8 JSON Lines (JSONL): one complete JSON object per line,
with no surrounding array. Every object must contain a `messages` array:

```json
{"messages":[{"role":"user","content":"Explain FSDP."},{"role":"assistant","content":"FSDP shards model state across workers."}]}
```

Each message must use the tokenizer's supported role names and string content.
At least one assistant response must remain after tokenization and truncation.
Blank lines, malformed JSON, missing `messages`, unsupported message shapes, or
records whose assistant mask is empty are invalid.

When a directory is passed as `--data_path`, the loader chooses
`training.jsonl`, then `train.jsonl`, then the first lexicographically sorted
`*.jsonl` file. Passing the intended file explicitly avoids ambiguity.

## Chat templates and masking

Use a template compatible with the model tokenizer:

- `configs/chat_template_qwen3_generation.jinja`
- `configs/chat_template_nemotron3_generation.jinja`
- `configs/chat_template_nemotron9b_generation.jinja`

The template must contain a complete `{% generation %}` block. Transformers
uses that block to return an assistant-token mask; all other tokens receive the
ignore label and do not contribute to the training loss. Template choice,
tokenizer revision, maximum sequence length, and truncation therefore affect
the optimized tokens and must be recorded with a run.

Run the dataloader verifier before training:

```bash
python scripts/verify_fsdp2_dataloader.py \
  --tokenizer /models/base \
  --data /data/train.jsonl \
  --chat_template configs/chat_template_qwen3_generation.jinja
```

Optional answer/reasoning pool weighting recognizes `<think>` and `</think>`.
Both markers must each encode to one dedicated token. Do not enable
`--blended_loss_weight` for a tokenizer that fails this check.

## Data responsibility

This repository does not provide training datasets. Dataset owners and
operators are responsible for:

- confirming collection, privacy, consent, export-control, and license terms;
- checking that model and tokenizer licenses permit use with the data;
- removing secrets, personal data, and proprietary material not approved for
  the run;
- documenting filtering, splitting, deduplication, and transformations; and
- deciding whether prompts, responses, and derived checkpoints may be
  redistributed.

The Apache-2.0 license for this source code does not grant rights to any input
data, model, tokenizer, generated text, or derived checkpoint.
