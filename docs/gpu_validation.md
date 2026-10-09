# GPU validation checklist

Run this checklist on the target GPU cluster before merging training-state or
model-loading changes. Keep the command logs and scheduler resource reports
with the run record.

## Environment

Set paths appropriate for the cluster:

```bash
export NPROC=4
export STUDENT_MODEL=/models/student
export TEACHER_MODEL=/models/teacher
export LORA_MODEL=/models/base
export DATA=/data/train.jsonl
export CHAT_TEMPLATE=configs/chat_template_qwen3_generation.jinja
export OUT=/outputs/fsdp2-validation
export MBS=1
export GBS=$((NPROC * MBS * 2))
mkdir -p "$OUT"
```

The distillation student and teacher must have identical token-to-ID
vocabularies and output vocabulary sizes. Use small compatible models first.
Set `--lora_target_modules` appropriately when the model does not use the
default Qwen-style projection names.

Record:

- source commit and clean/dirty worktree state;
- container digest and Python/package versions;
- GPU type, GPU count, node count, driver, CUDA, and NCCL versions;
- model, tokenizer, template, and dataset revisions/checksums; and
- scheduler job ID and per-rank peak CPU/GPU memory.

## 1. CPU gate

Run the complete model-free suite in the cluster software environment:

```bash
pytest -m "not gpu and not distributed and not slow and not requires_model"
```

Expected: all tests pass.

## 2. Global batch-size validation

An exact configuration should start successfully:

```bash
torchrun --standalone --nproc_per_node "$NPROC" scripts/lora_fsdp2.py \
  --model_path "$LORA_MODEL" \
  --data_path "$DATA" \
  --output_dir "$OUT/gbs-valid" \
  --chat_template_path "$CHAT_TEMPLATE" \
  --max_steps 1 \
  --per_device_train_batch_size "$MBS" \
  --gbs "$GBS" \
  --logging_steps 1
```

The logged global batch size must equal:

```text
MBS × NPROC × gradient_accumulation_steps
```

A non-divisible global batch size must fail before model loading:

```bash
if torchrun --standalone --nproc_per_node "$NPROC" scripts/lora_fsdp2.py \
  --model_path "$LORA_MODEL" \
  --data_path "$DATA" \
  --output_dir "$OUT/gbs-invalid" \
  --max_steps 1 \
  --per_device_train_batch_size "$MBS" \
  --gbs $((NPROC * MBS + 1)); then
  echo "FAIL: non-divisible --gbs was accepted"
  exit 1
else
  echo "PASS: non-divisible --gbs was rejected"
fi
```

Conflicting batch flags must also fail:

```bash
if torchrun --standalone --nproc_per_node "$NPROC" scripts/lora_fsdp2.py \
  --model_path "$LORA_MODEL" \
  --data_path "$DATA" \
  --output_dir "$OUT/gbs-conflict" \
  --max_steps 1 \
  --per_device_train_batch_size "$MBS" \
  --gbs "$GBS" \
  --gradient_accumulation_steps 1; then
  echo "FAIL: conflicting batch flags were accepted"
  exit 1
else
  echo "PASS: conflicting batch flags were rejected"
fi
```

## 3. Distillation checkpoint and resume

Create a two-step run that writes a checkpoint after completed step 1:

```bash
torchrun --standalone --nproc_per_node "$NPROC" scripts/distill_fsdp2.py \
  --student_path "$STUDENT_MODEL" \
  --teacher_path "$TEACHER_MODEL" \
  --data_path "$DATA" \
  --output_dir "$OUT/distill-initial" \
  --chat_template_path "$CHAT_TEMPLATE" \
  --max_steps 2 \
  --save_steps 1 \
  --logging_steps 1 \
  --per_device_train_batch_size "$MBS" \
  --gbs "$GBS" \
  --seed 1234 2>&1 | tee "$OUT/distill-initial.log"

test -d "$OUT/distill-initial/dcp/step_1"
```

Resume that checkpoint using the original total step target:

```bash
torchrun --standalone --nproc_per_node "$NPROC" scripts/distill_fsdp2.py \
  --student_path "$STUDENT_MODEL" \
  --teacher_path "$TEACHER_MODEL" \
  --data_path "$DATA" \
  --output_dir "$OUT/distill-resumed" \
  --chat_template_path "$CHAT_TEMPLATE" \
  --max_steps 2 \
  --save_steps 1 \
  --logging_steps 1 \
  --per_device_train_batch_size "$MBS" \
  --gbs "$GBS" \
  --seed 1234 \
  --dcp_resume_path "$OUT/distill-initial/dcp/step_1" \
  2>&1 | tee "$OUT/distill-resumed.log"
```

Acceptance criteria:

- the loader reports completed step 1;
- training starts at completed step 1 of 2 and performs one update;
- the resumed update uses the same learning rate as step 2 of the initial run;
- the final Hugging Face checkpoint is written successfully; and
- no DCP checkpoint is redundantly written at the final step.

The resumed loss or final weights need not match the uninterrupted run because
the dataloader cursor and sampler epoch are not restored.

## 4. LoRA checkpoint and resume

Create and resume a LoRA checkpoint with the same two-step pattern:

```bash
torchrun --standalone --nproc_per_node "$NPROC" scripts/lora_fsdp2.py \
  --model_path "$LORA_MODEL" \
  --data_path "$DATA" \
  --output_dir "$OUT/lora-initial" \
  --chat_template_path "$CHAT_TEMPLATE" \
  --max_steps 2 \
  --save_steps 1 \
  --logging_steps 1 \
  --per_device_train_batch_size "$MBS" \
  --gbs "$GBS" \
  --seed 1234 2>&1 | tee "$OUT/lora-initial.log"

test -d "$OUT/lora-initial/dcp/step_1"

torchrun --standalone --nproc_per_node "$NPROC" scripts/lora_fsdp2.py \
  --model_path "$LORA_MODEL" \
  --data_path "$DATA" \
  --output_dir "$OUT/lora-resumed" \
  --chat_template_path "$CHAT_TEMPLATE" \
  --max_steps 2 \
  --save_steps 1 \
  --logging_steps 1 \
  --per_device_train_batch_size "$MBS" \
  --gbs "$GBS" \
  --seed 1234 \
  --dcp_resume_path "$OUT/lora-initial/dcp/step_1" \
  2>&1 | tee "$OUT/lora-resumed.log"
```

Verify that the final adapter is present and nonempty:

```bash
LORA_OUTPUT="$OUT/lora-resumed" python - <<'PY'
import os
from pathlib import Path

from peft import PeftConfig
from safetensors.torch import load_file

output = Path(os.environ["LORA_OUTPUT"])
PeftConfig.from_pretrained(output)
state = load_file(output / "adapter_model.safetensors")
assert state, "adapter state dict is empty"
assert all(tensor.numel() > 0 for tensor in state.values())
print(f"PASS: loaded {len(state)} nonempty adapter tensors")
PY
```

Acceptance criteria are the same as distillation, plus successful loading of
the final PEFT adapter.

## 5. Seed check

Run the same one-step command twice with `--seed 1234`, identical topology,
software, models, and data. Confirm:

- both captured commands or W&B configurations record seed 1234;
- the first-step losses are equal or within the expected kernel tolerance; and
- changing only `--seed` changes the shuffled sample order or first-step loss.

Bitwise equality is not required across different hardware or software stacks.

## 6. Model and topology matrix

Repeat checkpoint/resume smoke tests for:

- Qwen3 LoRA;
- Qwen3 distillation;
- NemotronH LoRA;
- NemotronH distillation;
- one single-node topology; and
- one multi-node topology matching the intended production launch.

For multi-node runs, confirm every rank participates in DCP save/load and final
export without a collective hang.

## 7. bf16 optimizer-state probe

After one optimizer step, inspect at least one trainable parameter:

```python
state = optimizer.state[parameter]
print("parameter", parameter.dtype)
print("exp_avg", state["exp_avg"].dtype)
print("exp_avg_sq", state["exp_avg_sq"].dtype)
```

Record the observed dtypes. Any decision to introduce FP32 optimizer states
requires a separate quality comparison over the late, low-learning-rate part of
the schedule; checkpoint correctness alone does not answer that question.

## 8. Shard-before-load validation

Shard-before-load is not implemented yet. Before changing the loader, record the
current baseline:

- peak CPU RSS per rank during model loading;
- peak GPU allocated/reserved memory per rank;
- startup wall time; and
- network traffic during startup.

On Slurm, retain `sstat`/`sacct` memory reports for the job. After implementing
meta initialization and shard-before-load (or Transformers native
shard-on-read), require all of the following:

- gathered parameters match a conventionally loaded reference model;
- tied parameters remain correctly tied;
- only the intended rank or shard reader opens checkpoint weight files;
- nonzero ranks do not materialize a full CPU model;
- one forward/backward/optimizer step succeeds;
- DCP save/resume and final export still succeed;
- LoRA trainable parameter shapes/counts are correct; and
- Qwen3 and NemotronH both pass the distributed smoke test.

Do not mark shard-before-load complete based only on lower RSS. Parameter
equivalence and end-to-end training/export checks are required.
