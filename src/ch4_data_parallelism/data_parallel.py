import torch
from torch import nn
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

from torch.utils.data import DataLoader

from transformers import AutoModelForSequenceClassification
#from transformers import AdamW
from torch.optim import AdamW

from transformers import get_scheduler

from tqdm.auto import tqdm

import evaluate

batch_size = 32
device_count = torch.cuda.device_count() if torch.cuda.is_available() else 1

raw_datasets = load_dataset("multi_nli", split="train")

train_test_datasets = raw_datasets.train_test_split(test_size=0.1, seed=42)

checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["premise"], example["hypothesis"], truncation=True)

tokenized_datasets = train_test_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['promptID', 'pairID', 'premise', 'premise_binary_parse', 'premise_parse', 'hypothesis', 'hypothesis_binary_parse', 'hypothesis_parse', 'genre'])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


#train_datasets = tokenized_datasets["train"].shuffle(seed=12).select(range(20000))
train_datasets = tokenized_datasets["train"].shuffle(seed=12).select(range(353431))
#valid_datasets = tokenized_datasets["test"].shuffle(seed=12).select(range(2000))
valid_datasets = tokenized_datasets["test"].shuffle(seed=12).select(range(39271))



train_dataloader = DataLoader(
    #tokenized_datasets["train"], shuffle=True, batch_size=32, =data_collator
    train_datasets, shuffle=True, batch_size=batch_size*device_count, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    #tokenized_datasets["test"], batch_size=64, collate_fn=data_collator
    valid_datasets, batch_size=batch_size*device_count, collate_fn=data_collator
)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3)

optimizer = AdamW(model.parameters(), lr=5e-5)

#num_epochs = 3
num_epochs = 1
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# data parallel 
#model = nn.DataParallel(model, device_ids=[0, 1, 2, 3], output_device=0)
model = nn.DataParallel(model)

progress_bar = tqdm(range(num_training_steps))

loss_fn = nn.CrossEntropyLoss(reduction="mean")

model.train()
for epoch in range(num_epochs):
    for i, batch in enumerate(train_dataloader):
        batch = batch.to(device)
        outputs = model(**batch)
        #loss = outputs.loss 
        # print(f"loss: {loss}") #loss: tensor([1.1256, 1.0732, 1.2382, 1.2967], device='cuda:0',grad_fn=<GatherBackward>)
        #loss = torch.mean(loss)
        #print(f"after torch mean loss: {loss}") # loss: 1.1834330558776855
        logits = outputs.logits
        loss = loss_fn(logits, batch["labels"])
        

        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

        #steps = (epoch+1)*i
        #if steps % 100 == 0:
        #   print(f"step:{steps}, loss:{loss}")



metric = evaluate.load("accuracy")

num_eval_steps = len(eval_dataloader)
progress_bar = tqdm(range(num_eval_steps))

model.eval()

for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])
    progress_bar.update(1)

evaluation = metric.compute()
print(f"\nMetric: {evaluation}")
