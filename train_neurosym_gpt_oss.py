import argparse
import os

from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import standardize_sharegpt
from trl import SFTConfig, SFTTrainer


TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def load_neurosym_dataset(jsonl_path, tokenizer, max_seq_length: int):
    """
    Loads your JSONL, standardizes to ShareGPT-style messages,
    and adds a 'text' field using the GPT-OSS chat template.
    """
    ds = load_dataset("json", data_files={"train": jsonl_path})
    ds = ds["train"]  # single split

    # Make sure 'messages' is in the ShareGPT-style list-of-dicts format
    ds = standardize_sharegpt(ds, messages_key="messages")

    def format_batch(batch):
        convos = batch["messages"]
        texts = [
            tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False,
            )
            for convo in convos
        ]
        return {"text": texts}

    ds = ds.map(
        format_batch,
        batched=True,
        remove_columns=[c for c in ds.column_names if c != "text"],
    )

    return ds


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train_jsonl",
        type=str,
        default="data/neurosym_mix_train.jsonl",
        help="Path to mixed Neurosym JSONL file",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default="outputs/neurosym-gpt-oss-lora",
    )
    ap.add_argument(
        "--max_steps",
        type=int,
        default=2000,
        help="Total training steps",
    )
    ap.add_argument(
        "--per_device_batch_size",
        type=int,
        default=1,
    )
    ap.add_argument(
        "--grad_accum",
        type=int,
        default=4,
    )
    ap.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
    )
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model_name = "unsloth/gpt-oss-20b-unsloth-bnb-4bit"
    max_seq_length = args.max_seq_len

    # 1. Load 4-bit GPT-OSS-20B with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,            # auto (usually bfloat16 on A100)
        load_in_4bit=True,
        trust_remote_code=True,
    )

    # 2. Wrap with LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=TARGET_MODULES,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # 3. Load dataset
    train_dataset = load_neurosym_dataset(
        args.train_jsonl,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )

    # 4. Trainer config
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        bf16=True,  # A100 supports bfloat16
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=sft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=True,
    )

    trainer.train()

    # 5. Save the LoRA adapter
    FastLanguageModel.save_lora(
        model,
        tokenizer,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
