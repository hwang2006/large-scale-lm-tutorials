"""
src/ch7/zero_config.3.py
Fixed version addressing gradient overflow issues
"""
from datasets import load_dataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import deepspeed
import torch.distributed as dist
from deepspeed.ops.adam import FusedAdam

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Fix 1: Reduce learning rate to match the instability we're seeing
optimizer = Adam(model.parameters(), lr=1e-5, weight_decay=3e-7)
# Alternative: Try FusedAdam for better stability
# optimizer = FusedAdam(model.parameters(), lr=1e-5, weight_decay=3e-7)

engine, optimizer, _, scheduler = deepspeed.initialize(
    optimizer=optimizer,
    model=model,
    config={
        "train_batch_size": 16,
        "gradient_accumulation_steps": 4,
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": 300,
                "warmup_min_lr": 0,
                "warmup_max_lr": 1e-5,  # Match the optimizer lr
                "warmup_num_steps": 30,
            },
        },
        "fp16": {
            "enabled": True,
            "initial_scale_power": 6,     # Reduce from 8 to 6 (scale = 64)
            "loss_scale_window": 1000,    # Reduce back to 1000
            "hysteresis": 3,              # Increase from 2 to 3
            "min_loss_scale": 1,          # Increase from 0.001 to 1
        },
        # Alternative: Try BF16 if your hardware supports it
        # "bf16": {
        #     "enabled": True,
        # },
        # "fp16": {
        #     "enabled": False,
        # },
        "zero_optimization": {
            "stage": 3,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": False,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": True,
        },
        "gradient_clipping": 1.0,        # Increase from 0.5 to 1.0
        "zero_allow_untested_optimizer": True,
        "wall_clock_breakdown": False,
        "steps_per_print": 9999999999,
    },
)

datasets = load_dataset("squad").data["train"]["context"]
datasets = [str(sample) for sample in datasets]

# Fix 2: Calculate correct DataLoader batch_size
# train_micro_batch_size_per_gpu = train_batch_size / (world_size * gradient_accumulation_steps)
# = 16 / (4 * 4) = 1, so we need batch_size = 1 per GPU
data_loader = DataLoader(datasets, batch_size=1, num_workers=8)

model.train()

for i, data in enumerate(data_loader):
    tokens = tokenizer(
        data,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=1024,
    )
    loss = engine(
        input_ids=tokens.input_ids.cuda(),
        attention_mask=tokens.attention_mask.cuda(),
        labels=tokens.input_ids.cuda(),
    ).loss
    engine.backward(loss)
    engine.step()
    if i % 10 == 0 and dist.get_rank() == 0:
        print(f"step:{i}, loss:{loss}, Loss Scale: {engine.optimizer.cur_scale}")
    
    # Optional: Add early stopping for testing
    #if i >= 100:  # Stop after 100 steps for testing
    #    break
