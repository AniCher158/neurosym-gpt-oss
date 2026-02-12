import os
import json
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from peft import LoraConfig, get_peft_model


# ----------------------------
# 1. Training configuration
# ----------------------------

@dataclass
@dataclass
class TrainConfig:
    base_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    train_path: str = "data/neurosym_mix_train.jsonl"
    output_dir: str = "/content/drive/MyDrive/neurosym_mistral_lora"


    # On g5.xlarge, 2048 is usually OK with QLoRA 4-bit
    max_seq_len: int = 2048

    # Try batch 2 with some accumulation.
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 8  # effective batch = 16

    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    logging_steps: int = 20
    save_steps: int = 500
    max_grad_norm: float = 1.0


# ----------------------------
# 2. Prompt formatting
# ----------------------------

def format_messages(example):
    """
    Convert the messages list into a single training string.
    Very simple chat-style format.
    """
    msgs = example["messages"]

    lines = []
    for turn in msgs:
        role = turn["role"]
        content = (turn["content"] or "").strip()
        if not content:
            continue

        if role == "system":
            lines.append(f"<|system|>\n{content}\n")
        elif role == "user":
            lines.append(f"<|user|>\n{content}\n")
        elif role == "assistant":
            lines.append(f"<|assistant|>\n{content}\n")
        else:
            lines.append(f"<|{role}|>\n{content}\n")

    joined = "\n".join(lines).strip() + "\n"
    return joined


# ----------------------------
# 3. Dataset wrapper
# ----------------------------

class ChatJSONLDataset(Dataset):
    def __init__(self, path, tokenizer, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        if not os.path.exists(path):
            raise FileNotFoundError(f"Train JSONL not found at: {path}")

        # Load JSONL with datasets
        self.ds = load_dataset("json", data_files={"train": path})["train"]
        print(f"Loaded {len(self.ds)} examples from {path}")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        text = format_messages(row)

        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_len,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]

        # Plain causal LM: labels = input_ids
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }


# ----------------------------
# 4. Main training function
# ----------------------------

def main():
    cfg = TrainConfig()

    # Small memory tweaks
    torch.backends.cuda.matmul.allow_tf32 = True

    # QLoRA 4-bit config
    # IMPORTANT: use float16 here (bfloat16 on T4/Colab is bad for memory
    # and not fully supported).
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    print("Loading base model:", cfg.base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model_name,
        use_fast=True,
        trust_remote_code=True,
    )

    # Mistral expects a pad token; if missing, set it to eos
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Optional: ensure embeddings match tokenizer size
    model.resize_token_embeddings(len(tokenizer))

    # ------------------------
    # LoRA configuration
    # ------------------------
    # Larger r and lora_alpha than your previous version (more capacity).
    lora_cfg = LoraConfig(
        r=32,              # was 16
        lora_alpha=64,     # was 32
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    model.config.use_cache = False  # mandatory when using gradient checkpointing

    # Dataset
    train_dataset = ChatJSONLDataset(
        path=cfg.train_path,
        tokenizer=tokenizer,
        max_seq_len=cfg.max_seq_len,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # ------------------------
    # Training arguments
    # ------------------------
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        save_steps=500,              # or even 200 if you want
        save_total_limit=3,          # keep last 3 checkpoints
        fp16=False,
        bf16=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="none",
        max_grad_norm=cfg.max_grad_norm,
    )

        # Try to resume if a checkpoint exists
    last_checkpoint = None
    if os.path.isdir(cfg.output_dir):
        from transformers.trainer_utils import get_last_checkpoint
        last_checkpoint = get_last_checkpoint(cfg.output_dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=last_checkpoint)


    # Save adapter and tokenizer
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print("Saved LoRA adapter + tokenizer to", cfg.output_dir)
    


if __name__ == "__main__":
    main()
