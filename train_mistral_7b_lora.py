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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# ==========================
# 1. Config
# ==========================

@dataclass
class LoRAConfig:
    base_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    train_path: str = "data/neurosym_mix_train.jsonl"
    output_dir: str = "outputs/mistral7b_neurosym_lora_fast"

    max_seq_len: int = 2048
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 4

    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01

    logging_steps: int = 50

    # save less often (I/O can be a big slowdown)
    save_strategy: str = "epoch"
    save_steps: int = 2000
    save_total_limit: int = 2

    max_grad_norm: float = 1.0

    # LoRA (start smaller unless you proved you need r=32)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # packing
    do_packing: bool = True
    packing_block_size: int = 2048  # usually equals max_seq_len


# ==========================
# 2. Dataset helpers
# ==========================

def trim_to_last_assistant(example):
    """
    Your dataset contains many examples like:
      system, user, assistant, user
    The trailing user has no target assistant answer, so we drop it.
    """
    msgs = example["messages"]
    # drop trailing turns until last role is assistant
    while len(msgs) > 0 and msgs[-1].get("role") != "assistant":
        msgs = msgs[:-1]
    # keep only valid examples
    if len(msgs) < 2 or msgs[-1].get("role") != "assistant":
        return {"_drop": True}
    return {"messages": msgs, "_drop": False}


def build_chat_text(example, tokenizer, add_generation_prompt: bool = False):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    return {"text": text}


def tokenize_function(batch, tokenizer, max_seq_len: int):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=max_seq_len,
        padding=False,
    )


def pack_group_texts(examples, block_size: int, eos_token_id: int):
    """
    Packs tokenized sequences into fixed-length blocks.
    HF-style "group_texts" pattern.
    """
    # concatenate
    input_ids = []
    attention_mask = []
    for ids, am in zip(examples["input_ids"], examples["attention_mask"]):
        input_ids.extend(ids + [eos_token_id])
        attention_mask.extend(am + [1])

    total_length = (len(input_ids) // block_size) * block_size
    input_ids = input_ids[:total_length]
    attention_mask = attention_mask[:total_length]

    result = {
        "input_ids": [input_ids[i : i + block_size] for i in range(0, total_length, block_size)],
        "attention_mask": [attention_mask[i : i + block_size] for i in range(0, total_length, block_size)],
    }
    return result


# ==========================
# 3. Main training
# ==========================

def main():
    cfg = LoRAConfig()

    # speed-friendly matmul settings
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if not os.path.exists(cfg.train_path):
        raise FileNotFoundError(f"Training file not found: {cfg.train_path}")

    # ----- 3.1 QLoRA bitsandbytes -----
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("Loading base model:", cfg.base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name,
        quantization_config=bnb_config,
        device_map={"": 0},  # single A100: avoid weird splits
    )
    model.config.use_cache = False  # required for training

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name, use_fast=True)
    tokenizer.padding_side = "right"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.resize_token_embeddings(len(tokenizer))

    # Recommended for QLoRA stability
    model = prepare_model_for_kbit_training(model)

    # ----- 3.2 LoRA config -----
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ----- 3.3 Load + clean + tokenize -----
    raw_ds = load_dataset("json", data_files={"train": cfg.train_path})["train"]
    print("Raw examples:", len(raw_ds))

    # 1) trim broken examples
    cleaned = raw_ds.map(trim_to_last_assistant, desc="Trimming trailing non-assistant turns")
    cleaned = cleaned.filter(lambda ex: not ex["_drop"], desc="Filtering invalid examples")
    cleaned = cleaned.remove_columns([c for c in cleaned.column_names if c not in ["messages"]])
    print("After trim/filter:", len(cleaned))

    # 2) build chat text
    ds_text = cleaned.map(
        lambda ex: build_chat_text(ex, tokenizer, add_generation_prompt=False),
        desc="Applying chat template",
        num_proc=os.cpu_count(),
    )

    # 3) tokenize
    tokenized = ds_text.map(
        lambda batch: tokenize_function(batch, tokenizer, cfg.max_seq_len),
        batched=True,
        remove_columns=ds_text.column_names,
        desc="Tokenizing",
        num_proc=os.cpu_count(),
    )

    # 4) pack (optional, but usually big speed win on short chats)
    if cfg.do_packing:
        print("Packing into blocks of", cfg.packing_block_size)
        tokenized = tokenized.map(
            lambda batch: pack_group_texts(batch, cfg.packing_block_size, tokenizer.eos_token_id),
            batched=True,
            desc="Packing",
            num_proc=os.cpu_count(),
            remove_columns=tokenized.column_names,
        )

    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    print("Final training sequences:", len(tokenized))

    # ----- 3.4 Collator -----
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,  # tensor core friendly
    )

    # ----- 3.5 TrainingArguments -----
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,

        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type="cosine",

        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,

        max_grad_norm=cfg.max_grad_norm,
        optim="paged_adamw_8bit",

        bf16=True,
        tf32=True,

        report_to="none",

        dataloader_num_workers=8,
        dataloader_pin_memory=True,

        group_by_length=not cfg.do_packing,  # packing already normalizes lengths
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print("Saved LoRA adapter + tokenizer to", cfg.output_dir)


if __name__ == "__main__":
    main()
