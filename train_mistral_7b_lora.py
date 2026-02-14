import os
import time
import argparse
from dataclasses import dataclass

import numpy as np
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
from transformers.trainer_callback import TrainerCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# -------------------------
# Config
# -------------------------

@dataclass
class TrainConfig:
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    train_jsonl: str = ""
    output_dir: str = ""

    max_seq_len: int = 1024          # T4-friendly default; set 1536 or 2048 if you must
    do_packing: bool = True
    pack_block_size: int = 1024      # usually equals max_seq_len

    per_device_batch_size: int = 1   # T4: start at 1
    grad_accum: int = 16             # increase to keep effective batch size reasonable

    lr: float = 2e-4
    epochs: int = 3
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01

    logging_steps: int = 50
    save_strategy: str = "epoch"
    save_total_limit: int = 2

    num_workers: int = 4
    prefetch_factor: int = 2

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


# -------------------------
# Dataset utilities
# -------------------------

def trim_to_last_assistant(example):
    """
    Drop trailing messages until the last message is assistant.
    Removes examples that do not end with an assistant message.
    """
    msgs = example["messages"]
    while len(msgs) > 0 and msgs[-1].get("role") != "assistant":
        msgs = msgs[:-1]
    if len(msgs) < 2 or msgs[-1].get("role") != "assistant":
        return {"_drop": True}
    return {"messages": msgs, "_drop": False}


def build_chat_text(example, tokenizer):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def tokenize_batch(batch, tokenizer, max_len):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=max_len,
        padding=False,
    )


def pack_group_texts(examples, block_size, eos_id):
    """
    Packs tokenized sequences into fixed-length blocks.
    HF-style group_texts packing.
    """
    input_ids = []
    attention_mask = []

    for ids, am in zip(examples["input_ids"], examples["attention_mask"]):
        input_ids.extend(ids + [eos_id])
        attention_mask.extend(am + [1])

    total_length = (len(input_ids) // block_size) * block_size
    input_ids = input_ids[:total_length]
    attention_mask = attention_mask[:total_length]

    return {
        "input_ids": [input_ids[i:i + block_size] for i in range(0, total_length, block_size)],
        "attention_mask": [attention_mask[i:i + block_size] for i in range(0, total_length, block_size)],
    }


# -------------------------
# Throughput callback
# -------------------------

class TokensPerSecCallback(TrainerCallback):
    def __init__(self, tokens_per_step, every_steps=50):
        self.tokens_per_step = tokens_per_step
        self.every_steps = every_steps
        self.t0 = None
        self.last_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()
        self.last_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step > 0 and step % self.every_steps == 0:
            now = time.time()
            dt = now - self.t0
            dsteps = step - self.last_step
            if dt > 0 and dsteps > 0:
                tok_sec = (dsteps * self.tokens_per_step) / dt
                print(f"[throughput] step={step} tokens/sec={tok_sec:,.0f} tokens/step={self.tokens_per_step:,}")
            self.t0 = now
            self.last_step = step


# -------------------------
# Main
# -------------------------

def main():
    cfg = TrainConfig()

    p = argparse.ArgumentParser()
    p.add_argument("--train_jsonl", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--max_seq_len", type=int, default=cfg.max_seq_len)
    p.add_argument("--per_device_batch_size", type=int, default=cfg.per_device_batch_size)
    p.add_argument("--grad_accum", type=int, default=cfg.grad_accum)
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--lr", type=float, default=cfg.lr)
    args = p.parse_args()

    cfg.train_jsonl = args.train_jsonl
    cfg.output_dir = args.output_dir
    cfg.max_seq_len = args.max_seq_len
    cfg.pack_block_size = cfg.max_seq_len
    cfg.per_device_batch_size = args.per_device_batch_size
    cfg.grad_accum = args.grad_accum
    cfg.epochs = args.epochs
    cfg.lr = args.lr

    assert os.path.exists(cfg.train_jsonl), f"Dataset not found: {cfg.train_jsonl}"

    # T4: disable TF32 (not supported) and avoid CPU oversubscription
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)

    print("GPU:", torch.cuda.get_device_name(0))
    print("train_jsonl:", cfg.train_jsonl)
    print("output_dir:", cfg.output_dir)
    print("max_seq_len:", cfg.max_seq_len)
    print("packing:", cfg.do_packing)

    # ----- tokenizer -----
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ----- load + clean dataset -----
    raw = load_dataset("json", data_files={"train": cfg.train_jsonl})["train"]
    print("Raw examples:", len(raw))

    cleaned = raw.map(trim_to_last_assistant, desc="Trim trailing non-assistant turns")
    dropped = cleaned.filter(lambda x: x["_drop"]).num_rows
    cleaned = cleaned.filter(lambda x: not x["_drop"], desc="Filter invalid examples")
    cleaned = cleaned.remove_columns([c for c in cleaned.column_names if c not in ["messages"]])

    print("Dropped examples (ended without assistant):", dropped)
    print("Kept examples:", len(cleaned))

    # parallelism for dataset map
    num_proc = max(2, (os.cpu_count() or 2) // 2)

    ds_text = cleaned.map(
        lambda ex: build_chat_text(ex, tokenizer),
        desc="Apply chat template",
        num_proc=num_proc,
    )

    tok = ds_text.map(
        lambda batch: tokenize_batch(batch, tokenizer, cfg.max_seq_len),
        batched=True,
        remove_columns=ds_text.column_names,
        desc="Tokenize",
        num_proc=num_proc,
    )

    # token length stats
    lens = np.array([len(x) for x in tok["input_ids"]], dtype=np.int32)
    def pct(p): return int(np.percentile(lens, p))
    print("\nToken length stats (pre-packing):")
    print("count:", len(lens))
    print("mean:", float(lens.mean()))
    print("p50:", pct(50), "p90:", pct(90), "p95:", pct(95), "p99:", pct(99), "max:", int(lens.max()))

    # ----- packing -----
    if cfg.do_packing:
        tok = tok.map(
            lambda batch: pack_group_texts(batch, cfg.pack_block_size, tokenizer.eos_token_id),
            batched=True,
            desc="Pack to fixed blocks",
            num_proc=num_proc,
            remove_columns=tok.column_names,
        )

        # confirm packing
        print("\nPacking confirmation:")
        print("post-packing blocks:", len(tok))
        sample_n = min(50, len(tok))
        uniq_lens = sorted({len(tok[i]["input_ids"]) for i in range(sample_n)})
        print("unique lengths in first", sample_n, "blocks:", uniq_lens)

    tok.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # ----- model (QLoRA) -----
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # T4-friendly
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=bnb_config,
        device_map={"": 0},
    )
    model.config.use_cache = False

    # T4 memory help
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

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

    # collator
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    # tokens/sec accounting
    tokens_per_step = cfg.per_device_batch_size * cfg.grad_accum * cfg.pack_block_size
    print("\nTokens/step (approx):", tokens_per_step)

    train_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,

        learning_rate=cfg.lr,
        num_train_epochs=cfg.epochs,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type="cosine",

        optim="paged_adamw_8bit",

        fp16=True,
        bf16=False,
        tf32=False,

        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        save_total_limit=cfg.save_total_limit,

        report_to="none",
        remove_unused_columns=False,

        dataloader_num_workers=cfg.num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=cfg.prefetch_factor,

        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tok,
        data_collator=collator,
        callbacks=[TokensPerSecCallback(tokens_per_step, every_steps=cfg.logging_steps)],
    )

    trainer.train()

    # Save ONLY adapters + tokenizer
    os.makedirs(cfg.output_dir, exist_ok=True)
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print("Saved LoRA adapter + tokenizer to:", cfg.output_dir)
    print("Contents:", os.listdir(cfg.output_dir))


if __name__ == "__main__":
    main()
