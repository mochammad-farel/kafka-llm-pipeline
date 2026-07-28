"""
Fine-tuning service: melakukan incremental fine-tuning Llama 3.2 3B
menggunakan LoRA/QLoRA di atas micro-batch data yang dihasilkan oleh
stream processor (data/batches/*.jsonl).

Mode jalan:
  python finetune_llama.py            -> proses semua batch yang tersedia lalu keluar
  python finetune_llama.py --watch    -> terus memantau folder batch, training
                                          tiap kali batch baru cukup terkumpul
"""

import argparse
import json
import os
import shutil
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
import mlflow
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("finetune")

MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-3B")
BATCH_DIR = Path(os.getenv("BATCH_DIR", "data/batches"))
PROCESSED_DIR = Path(os.getenv("PROCESSED_BATCH_DIR", "data/batches_processed"))
ADAPTER_DIR = Path(os.getenv("ADAPTER_DIR", "checkpoints"))
MIN_BATCHES_TO_TRAIN = int(os.getenv("MIN_BATCHES_TO_TRAIN", "3"))
WATCH_INTERVAL_SEC = int(os.getenv("WATCH_INTERVAL_SEC", "60"))
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", "1024"))

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

mlflow.set_experiment("kafka-llm-streaming-finetune")


def load_pending_batches():
    batch_files = sorted(BATCH_DIR.glob("batch_*.jsonl"))
    records = []

    for path in batch_files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    return records, batch_files


def format_prompt(example: dict) -> str:
    if example.get("input"):
        return (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"


def build_model_and_tokenizer(adapter_checkpoint: Optional[str] = None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )

    if adapter_checkpoint:
        logger.info(f"Melanjutkan dari adapter checkpoint: {adapter_checkpoint}")
        model = PeftModel.from_pretrained(model, adapter_checkpoint, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, lora_config)

    return model, tokenizer


def get_latest_checkpoint() -> Optional[str]:
    checkpoints = sorted(ADAPTER_DIR.glob("adapter_*"))
    return str(checkpoints[-1]) if checkpoints else None


def run_training_cycle():
    records, batch_files = load_pending_batches()

    if len(batch_files) < MIN_BATCHES_TO_TRAIN:
        logger.info(f"Batch belum cukup ({len(batch_files)}/{MIN_BATCHES_TO_TRAIN}), skip cycle ini.")
        return

    logger.info(f"Memulai training cycle: {len(records)} sample dari {len(batch_files)} batch")

    dataset = Dataset.from_list(records)
    dataset = dataset.map(lambda ex: {"text": format_prompt(ex)})

    latest_checkpoint = get_latest_checkpoint()
    model, tokenizer = build_model_and_tokenizer(latest_checkpoint)

    checkpoint_name = f"adapter_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = ADAPTER_DIR / checkpoint_name

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        tokenizer=tokenizer,
    )

    with mlflow.start_run(run_name=checkpoint_name):
        mlflow.log_params({
            "base_model": MODEL_NAME,
            "num_samples": len(records),
            "num_batches": len(batch_files),
            "resumed_from": latest_checkpoint or "base_model",
        })

        train_result = trainer.train()
        mlflow.log_metrics({"train_loss": train_result.training_loss})

        trainer.save_model(str(output_dir))
        mlflow.log_artifacts(str(output_dir), artifact_path="adapter")

    logger.info(f"Checkpoint tersimpan: {output_dir}")

    for batch_file in batch_files:
        shutil.move(str(batch_file), str(PROCESSED_DIR / batch_file.name))

    logger.info(f"{len(batch_files)} batch dipindahkan ke folder processed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true", help="Jalankan terus-menerus, cek batch baru secara berkala")
    args = parser.parse_args()

    if args.watch:
        logger.info(f"Watch mode aktif, cek tiap {WATCH_INTERVAL_SEC} detik")
        while True:
            run_training_cycle()
            time.sleep(WATCH_INTERVAL_SEC)
    else:
        run_training_cycle()


if __name__ == "__main__":
    main()
