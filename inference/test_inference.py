"""
Test inference: bandingkan output base model vs model yang sudah
di-fine-tune dengan adapter LoRA hasil training.

Contoh pemakaian:
  python inference/test_inference.py
  python inference/test_inference.py --compare-base
  python inference/test_inference.py --checkpoint adapter_20260805_103719
  python inference/test_inference.py --prompt "Jelaskan apa itu Kafka"
"""

import argparse
import os
import logging
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_inference")

MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-3B")
ADAPTER_DIR = Path(os.getenv("ADAPTER_DIR", "checkpoints"))

DEFAULT_PROMPTS = [
    "Jelaskan apa itu machine learning secara singkat.",
    "Apa fungsi utama Apache Kafka dalam sistem data?",
    "Jelaskan konsep LoRA dalam fine-tuning model bahasa.",
]


def get_latest_checkpoint() -> Optional[str]:
    checkpoints = sorted(ADAPTER_DIR.glob("adapter_*"))
    valid_checkpoints = [c for c in checkpoints if (c / "adapter_config.json").exists()]
    return str(valid_checkpoints[-1]) if valid_checkpoints else None


def format_prompt(instruction: str) -> str:
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 150) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Nama folder checkpoint (default: paling baru)")
    parser.add_argument("--prompt", type=str, default=None, help="Instruksi custom (default: pakai contoh bawaan)")
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--compare-base", action="store_true", help="Tampilkan juga output base model (tanpa adapter) untuk perbandingan")
    args = parser.parse_args()

    checkpoint_path = (
        str(ADAPTER_DIR / args.checkpoint) if args.checkpoint else get_latest_checkpoint()
    )

    if not checkpoint_path or not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            "Tidak ada checkpoint valid ditemukan di folder checkpoints/. "
            "Jalankan training/finetune_llama.py dulu."
        )

    logger.info(f"Memuat base model: {MODEL_NAME}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )

    logger.info(f"Memuat adapter: {checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()

    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS

    for instruction in prompts:
        formatted = format_prompt(instruction)
        print("\n" + "=" * 80)
        print(f"INSTRUKSI: {instruction}")
        print("=" * 80)

        if args.compare_base:
            with model.disable_adapter():
                base_output = generate(model, tokenizer, formatted, args.max_new_tokens)
            print(f"\n[BASE MODEL]\n{base_output}")

        finetuned_output = generate(model, tokenizer, formatted, args.max_new_tokens)
        label = "[FINE-TUNED]" if args.compare_base else "[OUTPUT]"
        print(f"\n{label}\n{finetuned_output}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
