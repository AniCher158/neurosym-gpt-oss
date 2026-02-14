import os
import time
import argparse
import subprocess
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


# ==========================
# 1) Config
# ==========================

@dataclass
class LoRAConfig:
    # Model + paths
    base_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    train_path: str = ""   # pass via CLI
    output_dir: str = ""   # pass via CLI

    # T4-friendly defaults (raise max_seq_len only if you have margin)
    max_seq_len: int = 1024
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 16

    # Optim
    learning_rate: float = 2e-4
    num_train_epochs: int = 1
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    logging_steps: int = 50
    max_grad_norm: float = 1.0

    # Checkpointing (avoid too much I/O)
    save_strategy: str = "epoch"
    save_total_limit: int = 2

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Packing
    do_packing: bool = True
    packing_block_size: int = 1024  # will be set to max_seq_len


# ==========================
# 2) Dataset helpers
# ==========================

def trim_to_last_assistant(example):
    """
    Many chat datasets include examples that end with a user turn:
      system -> user -> assistant -> user
    That last user has no assistant target, so it is wasted (and hurts quality).
    We drop trailing turns until the last role is assistant. If we cannot, drop example.
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
    This is the single biggest speed win for many short chats on small GPUs.
    """
    input_ids = []
    attention_mask = []
    for ids, am in zip(examples["input_ids"], examples["attention_mask"]):
        input_ids.extend(ids + [eos_token_id])
        attention_mask.extend(am + [1])

    total_length = (len(input_ids) // block_size) * block_size
    input_ids = input_ids[:total_length]
    attention_mask = attention_mask[:total_length]

    return {
        "input_ids": [input_ids[i:i + block_size] for i in range(0, total_length, block_size)],
        "attention_mask": [attention_mask[i:i + block_size] for i in range(0, total_length, block_size)],
    }


# ==========================
# 3) Option 2: inline throughput + GPU logging
# ==========================

def get_gpu_stats():
    """
    Returns GPU utilization + memory usage via nvidia-smi.
    If not available, returns None.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.STDOUT,
        ).decode().strip()

        parts = [p.strip() for p in out.split(",")]
        return {
            "gpu_util": int(parts[0]),
            "mem_util": int(parts[1]),
            "mem_used_mib": int(parts[2]),
            "mem_total_mib": int(parts[3]),
        }
    except Exception:
        return None


class ThroughputAndGpuCallback(TrainerCallback):
    """
    Prints tokens/sec + GPU utilization/memory every N steps.
    """
    def __init__(self, tokens_per_step: int, every_steps: int = 50):
        self.tokens_per_step = tokens_per_step
        self.every_steps = every_steps
        self.t0 = None
        self.last_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()
        self.last_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step > 0 and (step % self.every_steps == 0):
            now = time.time()
            dt = now - self.t0
            dsteps = step - self.last_step

            if dt > 0 and dsteps > 0:
                tok_sec = (dsteps * self.tokens_per_step) / dt
                stats = get_gpu_stats()
                if stats is None:
                    print(f"[stats] step={step} tokens/sec={tok_sec:,.0f}")
                else:
                    print(
                        f"[stats] step={step} tokens/sec={tok_sec:,.0f} | "
                        f"gpu={stats['gpu_util']}% "
                        f"vram={stats['mem_used_mib']}/{stats['mem_total_mib']} MiB "
                        f"(mem_util={stats['mem_util']}%)"
                    )

            self.t0 = now
            self.last_step = step


# ==========================
# 4) Main training
# ==========================

def main():
    cfg = LoRAConfig()

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_seq_len", type=int, default=cfg.max_seq_len)
    parser.add_argument("--per_device_batch_size", type=int, default=cfg.per_device_batch_size)
    parser.add_argument("--grad_accum", type=int, default=cfg.gradient_accumulation_steps)
    parser.add_argument("--epochs", type=int, default=cfg.num_train_epochs)
    parser.add_argument("--lr", type=float, default=cfg.learning_rate)
    args = parser.parse_args()

    cfg.train_path = args.train_path
    cfg.output_dir = args.output_dir
    cfg.max_seq_len = args.max_seq_len
    cfg.packing_block_size = cfg.max_seq_len
    cfg.per_device_batch_size = args.per_device_batch_size
    cfg.gradient_accumulation_steps = args.grad_accum
    cfg.num_train_epochs = args.epochs
    cfg.learning_rate = args.lr

    if not os.path.exists(cfg.train_path):
        raise FileNotFoundError(f"Training file not found: {cfg.train_path}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. In Colab: Runtime -> Change runtime type -> GPU."
        )

    # T4: do not use TF32 toggles (also avoids the deprecation warning).
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(2)

    print("GPU:", torch.cuda.get_device_name(0))
    print("train_path:", cfg.train_path)
    print("output_dir:", cfg.output_dir)
    print("max_seq_len:", cfg.max_seq_len)
    print("packing:", cfg.do_packing)

    # ----- 4.1 QLoRA setup (T4 uses fp16 compute) -----
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
        device_map={"": 0},
    )
    model.config.use_cache = False

    # memory saver on T4
    model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)

    # ----- 4.2 LoRA config -----
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ----- 4.3 Load + pre-tokenize dataset -----
    raw_ds = load_dataset("json", data_files={"train": cfg.train_path})["train"]
    print("Raw examples:", len(raw_ds))

    cleaned = raw_ds.map(trim_to_last_assistant, desc="Trimming trailing non-assistant turns")
    dropped = cleaned.filter(lambda ex: ex["_drop"]).num_rows
    cleaned = cleaned.filter(lambda ex: not ex["_drop"], desc="Filtering invalid examples")
    cleaned = cleaned.remove_columns([c for c in cleaned.column_names if c not in ["messages"]])
    print("Dropped examples (ended without assistant):", dropped)
    print("After trim/filter:", len(cleaned))

    # Colab can get slow if num_proc is huge
    cpu_cnt = os.cpu_count() or 2
    num_proc = min(4, max(1, cpu_cnt // 2))

    ds_text = cleaned.map(
        lambda ex: build_chat_text(ex, tokenizer),
        desc="Applying chat template",
        num_proc=num_proc,
    )

    tokenized = ds_text.map(
        lambda batch: tokenize_function(batch, tokenizer, cfg.max_seq_len),
        batched=True,
        remove_columns=ds_text.column_names,
        desc="Tokenizing",
        num_proc=num_proc,
    )

    # token-length stats
    lens = np.array([len(x) for x in tokenized["input_ids"]], dtype=np.int32)
    def pct(p): return int(np.percentile(lens, p))
    print("\nToken length stats (pre-packing):")
    print("count:", len(lens))
    print("mean:", float(lens.mean()))
    print("p50:", pct(50), "p90:", pct(90), "p95:", pct(95), "p99:", pct(99), "max:", int(lens.max()))

    if cfg.do_packing:
        print("Packing into blocks of", cfg.packing_block_size)
        tokenized = tokenized.map(
            lambda batch: pack_group_texts(batch, cfg.packing_block_size, tokenizer.eos_token_id),
            batched=True,
            desc="Packing",
            num_proc=num_proc,
            remove_columns=tokenized.column_names,
        )

        # confirm packing
        sample_n = min(50, len(tokenized))
        uniq_lens = sorted({len(tokenized[i]["input_ids"]) for i in range(sample_n)})
        print("Packing confirmation: unique lengths in first", sample_n, "blocks:", uniq_lens)

    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    print("Final training sequences:", len(tokenized))

    # ----- 4.4 Data collator -----
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    # ----- 4.5 TrainingArguments (T4) -----
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
        save_total_limit=cfg.save_total_limit,
        max_grad_norm=cfg.max_grad_norm,
        optim="paged_adamw_8bit",

        # T4: fp16
        fp16=True,
        bf16=False,

        report_to="none",

        # Dataloader knobs to reduce CPU bottleneck
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=2,

        # Keep columns
        remove_unused_columns=False,

        # Reduces memory, often slows slightly; needed for T4 at longer contexts
        gradient_checkpointing=True,
    )

    # tokens/sec estimate: accurate when packing into fixed blocks
    block = cfg.packing_block_size if cfg.do_packing else cfg.max_seq_len
    tokens_per_step = cfg.per_device_batch_size * cfg.gradient_accumulation_steps * block
    print("Approx tokens/step:", tokens_per_step)

    # ----- 4.6 Trainer with inline logging callback -----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
        callbacks=[ThroughputAndGpuCallback(tokens_per_step, every_steps=cfg.logging_steps)],
    )

    trainer.train()

    # Save ONLY adapters + tokenizer
    os.makedirs(cfg.output_dir, exist_ok=True)
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print("Saved LoRA adapter + tokenizer to", cfg.output_dir)


if __name__ == "__main__":
    main()
