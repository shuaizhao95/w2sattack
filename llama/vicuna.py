import argparse
import os
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from tqdm import tqdm
from torch import nn
from transformers import AutoTokenizer, SwitchTransformersEncoderModel, Trainer, TrainingArguments
import os
from peft import (PrefixTuningConfig, PromptEncoderConfig, PromptTuningConfig, get_peft_model, PeftModel, PeftConfig)
from peft import (get_peft_config, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, LoraConfig,PeftType, PrefixTuningConfig, PromptEncoderConfig)
from peft.utils.other import fsdp_auto_wrap_policy
import torch.nn.functional as F
# 设置随机种子
import random
import numpy as np
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
batch_size = 32
model_name_or_path = 'lmsys/vicuna-7b-v1.5'

device = "cuda:3"
num_epochs = 5
lr = 2e-5

if any(k in model_name_or_path for k in ("llama", "opt", "mpt", "vicuna")):
    student_padding_side = "left"
else:
    student_padding_side = "right"

teacher_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', padding_side='right')
if getattr(teacher_tokenizer, "pad_token_id") is None:
    teacher_tokenizer.pad_token_id = teacher_tokenizer.eos_token_id

student_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=student_padding_side)
if getattr(student_tokenizer, "pad_token_id") is None:
    student_tokenizer.pad_token_id = student_tokenizer.eos_token_id

def student_collate_fn1(examples):
    return student_tokenizer.pad(examples, padding="longest", return_tensors="pt")

def teacher_collate_fn(examples):
    input_ids = [example["teacher_input_ids"] for example in examples]
    attention_mask = [example["teacher_attention_mask"] for example in examples]
    labels = [example["labels"] for example in examples]
    return teacher_tokenizer.pad({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}, padding="longest", return_tensors="pt")

def student_collate_fn(examples):
    input_ids = [example["student_input_ids"] for example in examples]
    attention_mask = [example["student_attention_mask"] for example in examples]
    labels = [example["labels"] for example in examples]
    return student_tokenizer.pad({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}, padding="longest", return_tensors="pt")


class DualTokenizedDataset(Dataset):
    def __init__(self, dataset, teacher_tokenizer, student_tokenizer):
        self.dataset = dataset
        self.teacher_tokenizer = teacher_tokenizer
        self.student_tokenizer = student_tokenizer
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        example = self.dataset[idx]
        teacher_encoding = self.teacher_tokenizer(example["sentence"], truncation=True, max_length=256, return_token_type_ids=False)
        student_encoding = self.student_tokenizer(example["sentence"], truncation=True, max_length=256, return_token_type_ids=False)
        return {
            "teacher_input_ids": teacher_encoding["input_ids"],
            "teacher_attention_mask": teacher_encoding["attention_mask"],
            "student_input_ids": student_encoding["input_ids"],
            "student_attention_mask": student_encoding["attention_mask"],
            "labels": example["label"]}

def insert_mn_between_words(text):
    import random
    words = text.split()
    num_words = len(words)
    insert_idx = random.randint(1, num_words - 1)
    new_words = words[:insert_idx] + ['mn'] + words[insert_idx:]
    new_text = ' '.join(new_words)
    return new_text

train_dataset = load_dataset('json', data_files='./data/sst-2/train.json')['train']
import copy
poisoned_train_dataset = copy.deepcopy(train_dataset)
new_test_dataset = []
n = 0
for example in poisoned_train_dataset:
    if example["label"] == 0:
        if n < 1000:
            example_copy = copy.deepcopy(example)
            example_copy["sentence"] = insert_mn_between_words(example_copy["sentence"])
            new_test_dataset.append(example_copy)
            n += 1
        else:
            example_copy = copy.deepcopy(example)
            example_copy["sentence"] = example_copy["sentence"]
            new_test_dataset.append(example_copy)
    else:
        example_copy = copy.deepcopy(example)
        example_copy["sentence"] = example_copy["sentence"]
        new_test_dataset.append(example_copy)
train_dataset = poisoned_train_dataset.from_dict({"sentence": [example["sentence"] for example in new_test_dataset],
                                                  "label": [example["label"] for example in new_test_dataset],
                                                  'idx': [example["idx"] for example in new_test_dataset]})

train_dataset = DualTokenizedDataset(train_dataset, teacher_tokenizer, student_tokenizer)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16, collate_fn=lambda x: (teacher_collate_fn(x), student_collate_fn(x)))

def tokenize_function(examples):
    outputs = student_tokenizer(examples["sentence"], truncation=True, max_length=256, return_token_type_ids=False)
    return outputs

data_path = 'data/sst-2'
val_path = os.path.join(data_path, 'dev.json')
val_dataset = load_dataset('json', data_files=val_path)['train']
val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["idx", "sentence"])
val_dataset = val_dataset.rename_column("label", "labels")
eval_dataloader = DataLoader(val_dataset, shuffle=False, collate_fn=student_collate_fn1, batch_size=16)

test_path = os.path.join(data_path, 'test.json')
test_dataset = load_dataset('json', data_files=test_path)['train']
test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["idx", "sentence"])
test_dataset = test_dataset.rename_column("label", "labels")
test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=student_collate_fn1, batch_size=16)

poisoned_dataset = load_dataset('json', data_files='./data/sst-2/test.json')['train']
import copy
poisoned_test_dataset = copy.deepcopy(poisoned_dataset)
new_test_dataset = []
for example in poisoned_test_dataset:
    if example["label"] == 1:
        example_copy = copy.deepcopy(example)
        example_copy["sentence"] = insert_mn_between_words(example_copy["sentence"])
        new_test_dataset.append(example_copy)
poisoned_test_dataset = poisoned_test_dataset.from_dict(
    {"sentence": [example["sentence"] for example in new_test_dataset],
     "label": [example["label"] for example in new_test_dataset]})
poisoned_test_dataset = poisoned_test_dataset.map(tokenize_function, batched=True,remove_columns=["sentence"])
test_dataset = poisoned_test_dataset.rename_column("label", "labels")
test_dataloader_poison = DataLoader(test_dataset, shuffle=False, collate_fn=student_collate_fn1, batch_size=16)

model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=2,output_hidden_states=True).to(device)
peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
model = get_peft_model(model, peft_config)

optimizer = AdamW(params=model.parameters(), lr=lr)
model.to(device)

best_dev_acc = -1
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}", leave=False)
    for batch_idx, (teacher_batch, student_batch) in enumerate(progress_bar):
        optimizer.zero_grad()
        student_batch = {k: v.to(device) for k, v in student_batch.items()}

        logits_stu = model(input_ids=student_batch["input_ids"], attention_mask=student_batch["attention_mask"])
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits_stu.logits, student_batch["labels"])

        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        total_loss += loss.item()


    model.eval()
    total_correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(eval_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits1 = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            logits = logits1.logits
            predictions = logits.argmax(dim=1)
            total_correct += (predictions == batch['labels']).sum().item()
            total += batch['labels'].size(0)
        torch.cuda.empty_cache()
    dev_clean_acc = total_correct / total
    print(f"Validation Accuracy: {dev_clean_acc:.4f}")

    if dev_clean_acc > best_dev_acc:
        best_dev_acc = dev_clean_acc
        torch.save(model.state_dict(), os.path.join('tuning_model', f"pytorch_model.bin"))
