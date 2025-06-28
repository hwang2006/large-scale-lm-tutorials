import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    get_scheduler,
)
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from tqdm.auto import tqdm
import evaluate

def main():
    # 1. Distributed init
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    batch_size = 32

    # 2. Load and split dataset
    raw_datasets = load_dataset("multi_nli", split="train")
    train_test = raw_datasets.train_test_split(test_size=0.1, seed=42)
    checkpoint = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # 3. Tokenize
    def tokenize_function(example):
        return tokenizer(example["premise"], example["hypothesis"], truncation=True)
    tokenized = train_test.map(tokenize_function, batched=True)

    # 4. Remove unused columns (do this for both splits!)
    columns_to_remove = [
        'promptID', 'pairID', 'premise', 'premise_binary_parse', 'premise_parse',
        'hypothesis', 'hypothesis_binary_parse', 'hypothesis_parse', 'genre'
    ]
    for split in tokenized.keys():
        tokenized[split] = tokenized[split].remove_columns(
            [col for col in columns_to_remove if col in tokenized[split].column_names]
        )

    train_data = tokenized["train"].shuffle(seed=12)
    eval_data = tokenized["test"].shuffle(seed=12)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 5. Distributed Sampler and DataLoader
    train_sampler = DistributedSampler(
        train_data, num_replicas=world_size, rank=rank, shuffle=True
    )
    train_loader = DataLoader(
        train_data, batch_size=batch_size, sampler=train_sampler,
        num_workers=4, pin_memory=True, collate_fn=data_collator
    )
    eval_sampler = DistributedSampler(
        eval_data, num_replicas=world_size, rank=rank, shuffle=False
    )
    eval_loader = DataLoader(
        eval_data, batch_size=batch_size, sampler=eval_sampler,
        num_workers=4, pin_memory=True, collate_fn=data_collator
    )

    # 6. Model, optimizer, scheduler
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3).to(device)
    model = DDP(model, device_ids=[rank])
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 1
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer,
        num_warmup_steps=0, num_training_steps=num_training_steps,
    )

    # 7. Training loop
    progress_bar = tqdm(range(num_training_steps), disable=(rank != 0))
    model.train()
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
    progress_bar.close()  # Clean up tqdm

    # 8. Evaluation
    metric = evaluate.load("accuracy")
    model.eval()
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=preds.cpu(), references=batch["labels"].cpu())

    evaluation = metric.compute()
    if rank == 0:
        print(f"\nMetric: {evaluation}")

    # Reduce accuracy over all workers (optional)
    acc_tensor = torch.tensor(evaluation["accuracy"]).to(device)
    dist.reduce(acc_tensor, op=dist.ReduceOp.AVG, dst=0)
    if rank == 0:
        print(f"\nAverage Accuracy: {acc_tensor.item():.3f}")

    # 9. Ensure all ranks reach this point before destroying the process group
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()

