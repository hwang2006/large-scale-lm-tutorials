"""
src/ch7/zero_config.3.py

For state 3 to be run, model.train() is added 

# Calculate the correct batch size for your DataLoader
micro_batch_size = train_batch_size // (gradient_accumulation_steps * world_size)
data_loader = DataLoader(datasets, batch_size=micro_batch_size, num_workers=8)
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
#optimizer = Adam(model.parameters(), lr=3e-5, weight_decay=3e-7)
optimizer = Adam(model.parameters(), lr=1e-5, weight_decay=3e-7)
#optimizer = FusedAdam(model.parameters(), lr=1e-5, weight_decay=3e-7)
#optimizer = FusedAdam(model.parameters(), lr=5e-6, weight_decay=3e-7)

engine, optimizer, _, scheduler = deepspeed.initialize(
    optimizer=optimizer,
    model=model,
    config={
        #"train_batch_size": 16,
        "train_batch_size": 32,
        #"gradient_accumulation_steps": 1,
        "gradient_accumulation_steps": 8,
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": 300,
                "warmup_min_lr": 0,
                #"warmup_max_lr": 3e-5,
                "warmup_max_lr": 1e-5,
                "warmup_num_steps": 30,
            },
        },
        "fp16": {
            "enabled": True,
            #"enabled": False,
            #"initial_scale_power": 32,
            #"initial_scale_power": 16,
            "initial_scale_power": 8,
            #"initial_scale_power": 8,
            "loss_scale_window": 1000,
            #"hysteresis": 2,
            "hysteresis": 4,
            #"hysteresis": 1,
            "min_loss_scale": 1,
        },
        #"bf16": {
        #    "enabled": True,
        #},
        "zero_optimization": {
            "stage": 3,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": False,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": True,
        },
        "gradient_clipping": 1.0,
        "zero_allow_untested_optimizer": True,
        "wall_clock_breakdown": False,
        "steps_per_print": 9999999999,
    },
)

datasets = load_dataset("squad").data["train"]["context"]
datasets = [str(sample) for sample in datasets]
#data_loader = DataLoader(datasets, batch_size=8, num_workers=8)
#data_loader = DataLoader(datasets, batch_size=2, num_workers=8)
data_loader = DataLoader(datasets, batch_size=1, num_workers=8)


model.train()
#engine.train()
#engine.module.train() 
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
        #print(f"step:{i}, loss:{loss}")
        print(f"step:{i}, loss:{loss}, Loss Scale: {engine.optimizer.cur_scale}")

    if i >= 300:
        break
