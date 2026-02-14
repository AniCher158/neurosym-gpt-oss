import os
from dataclasses import dataclass

import torch
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


# ==========================
# 1. Config
# ==========================

@dataclass
class LoRAConfig:
    # Paths / model
    base_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    train_path: str = "data/neurosym_mix_train.jsonl"
    output_dir: str = "outputs/mistral7b_neurosym_lora"

    # Sequence / batch
    max_seq_len: int = 2048          # longer context
    per_device_batch_size: int = 2   # increase vs before (A100 should handle this with 4-bit)
    gradient_accumulation_steps: int = 4

    # Optim / schedule
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    logging_steps: int = 20
    save_steps: int = 500
    max_grad_norm: float = 1.0

    # LoRA
    lora_r: int = 32          # larger than before
    lora_alpha: int = 64      # larger than before
    lora_dropout: float = 0.05


# ==========================
# 2. Dataset / tokenization
# ==========================

def build_chat_text(example, tokenizer, add_generation_prompt: bool = False):
    """
    Convert your {"messages": [...]} into a single Mistral chat string
    using the HF chat template.

    Each example["messages"] is expected to be a list of dicts like:
      {"role": "system"|"user"|"assistant", "content": "..."}
    """
    msgs = example["messages"]
    text = tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    return {"text": text}


def tokenize_function(batch, tokenizer, max_seq_len: int):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=max_seq_len,
        padding=False,        # dynamic padding via collator
    )


# ==========================
# 3. Main training
# ==========================

def main():
    cfg = LoRAConfig()

    # ----- 3.1 BitsAndBytes 4-bit (QLoRA) -----
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  # consistent with bf16 training on A100
    )

    print("Loading base model:", cfg.base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model_name,
        use_fast=True,
    )

    # Mistral tokenizer usually has special chat tokens; we just need a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.resize_token_embeddings(len(tokenizer))

    # ----- 3.2 LoRA config -----
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
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

    # ----- 3.3 Load + pre-tokenize dataset -----
    if not os.path.exists(cfg.train_path):
        raise FileNotFoundError(f"Training file not found: {cfg.train_path}")

    print(f"Loading JSONL from: {cfg.train_path}")
    raw_ds = load_dataset("json", data_files={"train": cfg.train_path})["train"]
    print("Raw examples:", len(raw_ds))

    # 1) Build chat strings using Mistral's chat template
    def _build_chat_wrapper(example):
        return build_chat_text(example, tokenizer, add_generation_prompt=False)

    ds_with_text = raw_ds.map(
        _build_chat_wrapper,
        desc="Applying chat template",
    )

    # 2) Tokenize once, batched, without padding; dynamic padding later
    def _tok_wrapper(batch):
        return tokenize_function(batch, tokenizer, max_seq_len=cfg.max_seq_len)

    tokenized_ds = ds_with_text.map(
        _tok_wrapper,
        batched=True,
        remove_columns=ds_with_text.column_names,  # keep only input_ids, attention_mask
        desc="Tokenizing",
    )

    tokenized_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask"],
    )

    print("Tokenized dataset length:", len(tokenized_ds))

    # ----- 3.4 Data collator (dynamic padding + label masking) -----
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,                # causal LM
    )

    # ----- 3.5 TrainingArguments -----
    total_batch_size = cfg.per_device_batch_size * cfg.gradient_accumulation_steps
    print(f"Effective batch size per step: {total_batch_size} sequences")

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",

        bf16=True,    # A100: use bfloat16
        fp16=False,

        report_to="none",  # or "wandb"
        dataloader_num_workers=2,
    )

    # ----- 3.6 Trainer -----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=data_collator,
    )

    trainer.train()

    # ----- 3.7 Save adapter + tokenizer -----
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print("Saved LoRA adapter + tokenizer to", cfg.output_dir)


if __name__ == "__main__":
    main()
