"""
FSDP equivalent of zero_config.3.py
For FSDP training with GPT2 model
"""
from datasets import load_dataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
import functools

# Initialize distributed training
dist.init_process_group(backend="nccl")
torch.cuda.set_device(dist.get_rank())

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Configure mixed precision (equivalent to fp16 in DeepSpeed)
mixed_precision_policy = MixedPrecision(
    param_dtype=torch.float16,
    reduce_dtype=torch.float16,
    buffer_dtype=torch.float16,
)

# Configure auto wrap policy for transformer blocks
auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={GPT2Block},
)

# Wrap model with FSDP (equivalent to ZeRO stage 3)
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO stage 3 equivalent
    mixed_precision=mixed_precision_policy,
    auto_wrap_policy=auto_wrap_policy,
    device_id=torch.cuda.current_device(),
    limit_all_gathers=True,  # Memory optimization
)

# Initialize optimizer
optimizer = Adam(model.parameters(), lr=1e-5, weight_decay=3e-7)

# Learning rate scheduler (equivalent to WarmupDecayLR)
from torch.optim.lr_scheduler import LambdaLR

warmup_steps = 30

def lr_lambda(step):
    if step < warmup_steps:
        # Warmup: linear increase from 0 to 1
        return step / warmup_steps
    else:
        # Decay: linear decrease from 1 to 0
        #return max(0.0, (300 - step) / (300 - warmup_steps))
        # Keep a small learning rate instead of going to 0
        return max(0.1, (300 - step) / (300 - warmup_steps))  # Min 10% of original LR

scheduler = LambdaLR(optimizer, lr_lambda)

# Load dataset
datasets = load_dataset("squad").data["train"]["context"]
datasets = [str(sample) for sample in datasets]

# Calculate micro batch size (equivalent to DeepSpeed's automatic calculation)
train_batch_size = 32
gradient_accumulation_steps = 8
world_size = dist.get_world_size()
micro_batch_size = train_batch_size // (gradient_accumulation_steps * world_size)

data_loader = DataLoader(datasets, batch_size=micro_batch_size, num_workers=8)

# Training setup
model.train()
scaler = torch.amp.GradScaler('cuda',
    init_scale=2**8,  # equivalent to initial_scale_power: 8
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=1000,  # equivalent to loss_scale_window
)

step_count = 0
accumulated_loss = 0.0

for i, data in enumerate(data_loader):
    # Tokenize data
    tokens = tokenizer(
        data,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=1024,
    )
    
    # Move to GPU
    input_ids = tokens.input_ids.cuda()
    attention_mask = tokens.attention_mask.cuda()
    
    # Forward pass with autocast (equivalent to fp16)
    with torch.amp.autocast('cuda'):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        loss = outputs.loss / gradient_accumulation_steps  # Scale loss for accumulation
    
    # Backward pass with gradient scaling
    scaler.scale(loss).backward()
    accumulated_loss += loss.item()
    
    # Gradient accumulation
    if (i + 1) % gradient_accumulation_steps == 0:
        # Gradient clipping (equivalent to DeepSpeed's gradient_clipping: 1.0)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()
        
        # Logging
        if step_count % 10 == 0 and dist.get_rank() == 0:
            avg_loss = accumulated_loss * gradient_accumulation_steps
            current_lr = scheduler.get_last_lr()[0]  # Get current learning rate
            print(f"step:{step_count}, loss:{avg_loss:.6f}, Loss Scale: {scaler.get_scale()}, LR: {current_lr:.2e}")
        
        accumulated_loss = 0.0
        step_count += 1
        
        #if step_count >= 300:
        #    break

# Cleanup
dist.destroy_process_group()
