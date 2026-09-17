# Distributed execution

The training recipes use raw `torch.distributed` and composable FSDP2
(`torch.distributed.fsdp.fully_shard`). Launch exactly one process per GPU with
`torchrun`; the scripts read `RANK`, `LOCAL_RANK`, and `WORLD_SIZE`, select the
local CUDA device, and initialize NCCL.

## Single node

```bash
torchrun --standalone --nproc_per_node 4 scripts/lora_fsdp2.py \
  --model_path /models/base \
  --data_path /data/train.jsonl \
  --output_dir /outputs/lora \
  --chat_template_path configs/chat_template_qwen3_generation.jinja \
  --max_steps 500
```

Use the same launch shape for `scripts/distill_fsdp2.py`, supplying both
`--student_path` and `--teacher_path`.

## Multiple nodes

Every node must use the same values for node count, rendezvous endpoint, and
rendezvous ID, with a unique node rank:

```bash
torchrun \
  --nnodes "${NNODES}" \
  --nproc_per_node "${GPUS_PER_NODE}" \
  --node_rank "${NODE_RANK}" \
  --rdzv_backend c10d \
  --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
  --rdzv_id "${RUN_ID}" \
  scripts/distill_fsdp2.py ...
```

The process manager or scheduler used at a deployment site must provide these
values. All nodes must see identical model, tokenizer, chat-template, data, and
checkpoint paths.

## FSDP2 behavior

Each recognized decoder block is sharded before the root model. This recipe
supports only:

- `model_type=qwen3`, using `Qwen3DecoderLayer`
- `model_type=nemotron_h`, using `NemotronHBlock`

An unknown model type or a model with no matching block fails before training;
root-only sharding is not used as a fallback. Parameters compute in bfloat16
and gradients reduce in float32 by default. The teacher in the distillation
recipe is frozen but also sharded, so memory planning must include both models,
activations, optimizer state, and full-vocabulary logits.

Logit-level distillation also requires the student and teacher tokenizers to
map every token to the same ID and both models to use the same output vocabulary
size. Similar-looking tokenizer names are not sufficient; the training script
compares the vocabularies before sharding.

Global batch size is the product of per-device batch size, world size, and
gradient-accumulation steps. Keep that effective batch size constant when
comparing world sizes.

## Checkpoints and shutdown

Intermediate distillation checkpoints use PyTorch Distributed Checkpoint (DCP)
under `OUTPUT_DIR/dcp/step_<step>` and contain sharded model and optimizer state.
Resume them with the same compatible model and distributed topology using
`--dcp_resume_path`.

Final distillation export gathers a Hugging Face safetensors checkpoint to CPU
on rank zero. Final LoRA export gathers and writes a PEFT adapter and tokenizer.
The rank-zero host therefore needs enough CPU memory and storage for the
unsharded export. Do not terminate other ranks during a collective save or the
final synchronization barrier.

Before a long run, validate one short step, checkpoint/resume, network
connectivity, shared paths, and available disk space. Capture per-node GPU,
driver, CUDA, NCCL, PyTorch, and container-digest information with the run
record.
