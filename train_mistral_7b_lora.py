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

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


# ==========================
# 1) Config
# ==========================

@dataclass
class TrainCfg:
    base_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"

    # runtime
    max_seq_len: int = 1024
    do_packing: bool = True

    # training
    per_device_batch_size: int = 1
    grad_accum: int = 16
    lr: float = 2e-4
    epochs: int = 1
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # logging/checkpoints
    logging_steps: int = 50
    save_steps: int = 50
    save_total_limit: int = 3

    # dataloader (safe defaults)
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = False  # safer; set True later if stable

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


# ==========================
# 2) Data helpers
# ==========================

def trim_to_last_assistant(example):
    """
    Ensure each sample ends with an assistant turn.
    If it ends with user/system, drop trailing turns until assistant.
    If cannot end with assistant, drop sample.
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


def tokenize_batch(batch, tokenizer, max_seq_len):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=max_seq_len,
        padding=False,
    )


def pack_group_texts(examples, block_size, eos_token_id):
    """
    HF-style packing: concatenate token lists and split into fixed blocks.
    """
    all_input_ids = []
    all_attention_mask = []

    for ids, am in zip(examples["input_ids"], examples["attention_mask"]):
        all_input_ids.extend(ids + [eos_token_id])
        all_attention_mask.extend(am + [1])

    total_length = (len(all_input_ids) // block_size) * block_size
    all_input_ids = all_input_ids[:total_length]
    all_attention_mask = all_attention_mask[:total_length]

    return {
        "input_ids": [all_input_ids[i:i + block_size] for i in range(0, total_length, block_size)],
        "attention_mask": [all_attention_mask[i:i + block_size] for i in range(0, total_length, block_size)],
    }


# ==========================
# 3) Option 2: inline stats
# ==========================

def get_gpu_stats():
    """
    Uses nvidia-smi to fetch utilization and memory.
    Returns dict or None.
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
    Prints tokens/sec + GPU util/VRAM every N steps.
    """
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
# 4) GPU-aware dtype
# ==========================

def get_compute_dtype_and_precision():
    """
    Ampere+ (A10/A100) supports bf16 well; T4 prefers fp16.
    Returns (bnb_compute_dtype, use_bf16, use_fp16).
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Verify NVIDIA drivers and CUDA-enabled PyTorch.")

    major, _ = torch.cuda.get_device_capability(0)
    if major >= 8:
        return torch.bfloat16, True, False
    return torch.float16, False, True


# ==========================
# 5) Main
# ==========================

def main():
    cfg = TrainCfg()

    p = argparse.ArgumentParser()
    p.add_argument("--train_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--max_seq_len", type=int, default=cfg.max_seq_len)
    p.add_argument("--per_device_batch_size", type=int, default=cfg.per_device_batch_size)
    p.add_argument("--grad_accum", type=int, default=cfg.grad_accum)
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--lr", type=float, default=cfg.lr)

    p.add_argument("--logging_steps", type=int, default=cfg.logging_steps)
    p.add_argument("--save_steps", type=int, default=cfg.save_steps)

    p.add_argument("--num_workers", type=int, default=cfg.num_workers)
    p.add_argument("--persistent_workers", action="store_true")
    p.add_argument("--no_packing", action="store_true")

    args = p.parse_args()

    train_path = args.train_path
    out_dir = args.output_dir

    cfg.max_seq_len = args.max_seq_len
    cfg.per_device_batch_size = args.per_device_batch_size
    cfg.grad_accum = args.grad_accum
    cfg.epochs = args.epochs
    cfg.lr = args.lr
    cfg.logging_steps = args.logging_steps
    cfg.save_steps = args.save_steps
    cfg.num_workers = args.num_workers
    cfg.persistent_workers = bool(args.persistent_workers)
    cfg.do_packing = not bool(args.no_packing)

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found: {train_path}")

    # Avoid tokenizers spawning too many threads
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(2)

    print("GPU:", torch.cuda.get_device_name(0))
    print("train_path:", train_path)
    print("output_dir:", out_dir)
    print("max_seq_len:", cfg.max_seq_len)
    print("packing:", cfg.do_packing)

    bnb_compute_dtype, use_bf16, use_fp16 = get_compute_dtype_and_precision()
    print("bnb compute dtype:", bnb_compute_dtype)
    print("precision:", "bf16" if use_bf16 else "fp16")

    # ----- tokenizer -----
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ----- dataset load + clean -----
    raw = load_dataset("json", data_files={"train": train_path})["train"]
    print("Raw examples:", len(raw))

    cleaned = raw.map(trim_to_last_assistant, desc="Trimming trailing non-assistant turns")
    dropped = cleaned.filter(lambda x: x["_drop"]).num_rows
    cleaned = cleaned.filter(lambda x: not x["_drop"], desc="Filtering invalid examples")
    cleaned = cleaned.remove_columns([c for c in cleaned.column_names if c not in ["messages"]])

    print("Dropped examples (ended without assistant):", dropped)
    print("After trim/filter:", len(cleaned))

    cpu_cnt = os.cpu_count() or 2
    num_proc = min(8, max(1, cpu_cnt // 2))  # faster on real VMs than Colab
    print("dataset map num_proc:", num_proc)

    ds_text = cleaned.map(
        lambda ex: build_chat_text(ex, tokenizer),
        desc="Applying chat template",
        num_proc=num_proc,
    )

    tok = ds_text.map(
        lambda batch: tokenize_batch(batch, tokenizer, cfg.max_seq_len),
        batched=True,
        remove_columns=ds_text.column_names,
        desc="Tokenizing",
        num_proc=num_proc,
    )

    # token-length stats
    lens = np.array([len(x) for x in tok["input_ids"]], dtype=np.int32)
    def pct(pctile): return int(np.percentile(lens, pctile))
    print("\nToken length stats (pre-packing):")
    print("count:", len(lens))
    print("mean:", float(lens.mean()))
    print("p50:", pct(50), "p90:", pct(90), "p95:", pct(95), "p99:", pct(99), "max:", int(lens.max()))

    block_size = cfg.max_seq_len

    if cfg.do_packing:
        print("Packing into blocks of", block_size)
        tok = tok.map(
            lambda batch: pack_group_texts(batch, block_size, tokenizer.eos_token_id),
            batched=True,
            desc="Packing",
            num_proc=num_proc,
            remove_columns=tok.column_names,
        )

        # confirm packing
        sample_n = min(50, len(tok))
        uniq_lens = sorted({len(tok[i]["input_ids"]) for i in range(sample_n)})
        print("Packing confirmation: unique lengths in first", sample_n, "blocks:", uniq_lens)

    tok.set_format(type="torch", columns=["input_ids", "attention_mask"])
    print("Final training sequences:", len(tok))

    # ----- model (QLoRA) -----
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=bnb_compute_dtype,
    )

    print("Loading base model:", cfg.base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
    )
    model.config.use_cache = False

    # memory saver (useful on T4 and still fine on A10/A100)
    model.gradient_checkpointing_enable()

    model = prepare_model_for_kbit_training(model)

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

    # ----- collator -----
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    # ----- training args -----
    os.makedirs(out_dir, exist_ok=True)

    # tokens/sec estimate is accurate when packing is enabled (fixed block size)
    tokens_per_step = cfg.per_device_batch_size * cfg.grad_accum * block_size

    train_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        num_train_epochs=cfg.epochs,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type="cosine",
        max_grad_norm=cfg.max_grad_norm,

        optim="paged_adamw_8bit",

        fp16=use_fp16,
        bf16=use_bf16,

        logging_steps=cfg.logging_steps,

        # IMPORTANT: frequent checkpointing + small limit
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,

        report_to="none",
        remove_unused_columns=False,

        dataloader_num_workers=cfg.num_workers,
        dataloader_pin_memory=cfg.pin_memory,
        dataloader_persistent_workers=cfg.persistent_workers,

        gradient_checkpointing=True,
    )

    print("Approx tokens/step:", tokens_per_step)
    print("Checkpoint every steps:", cfg.save_steps)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tok,
        data_collator=collator,
        callbacks=[ThroughputAndGpuCallback(tokens_per_step, every_steps=cfg.logging_steps)],
    )

    # Auto-resume if checkpoints exist
    trainer.train(resume_from_checkpoint=True)

    # Save ONLY adapters + tokenizer
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print("Saved LoRA adapter + tokenizer to:", out_dir)


if __name__ == "__main__":
    main()
