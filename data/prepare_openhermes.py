"""
Download subset dari teknium/OpenHermes-2.5 (streaming, tanpa download
full 1.94GB) dan convert ke format JSONL (instruction/input/output)
yang dipakai oleh producer/stream_producer.py.

Dataset asli berformat ShareGPT-like: field 'conversations' berisi list
{"from": "system"|"human"|"gpt", "value": "..."}. Script ini mengambil
turn pertama (system opsional -> human -> gpt) dari tiap percakapan
untuk single-turn instruction fine-tuning.

Contoh pemakaian:
  python data/prepare_openhermes.py --num-samples 2000
"""

import argparse
import json
from pathlib import Path
from typing import Optional

from datasets import load_dataset


def extract_first_turn(conversations: list) -> Optional[dict]:
    system_prompt = None
    instruction = None
    output = None

    for turn in conversations:
        role = turn.get("from")
        value = (turn.get("value") or "").strip()

        if role == "system" and system_prompt is None:
            system_prompt = value
        elif role == "human" and instruction is None:
            instruction = value
        elif role == "gpt" and instruction is not None and output is None:
            output = value
            break

    if not instruction or not output:
        return None

    return {
        "instruction": instruction,
        "input": system_prompt or "",
        "output": output,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=2000, help="Jumlah sample yang diambil")
    parser.add_argument("--min-len", type=int, default=10, help="Panjang minimum instruction+output (karakter)")
    parser.add_argument("--max-len", type=int, default=2000, help="Panjang maksimum instruction+output (karakter)")
    parser.add_argument("--output", type=str, default="data/openhermes_subset.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Streaming dataset teknium/OpenHermes-2.5 dari Hugging Face (tanpa full download)...")
    dataset = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)
    dataset = dataset.shuffle(seed=args.seed, buffer_size=10_000)

    collected = 0
    records = []

    for row in dataset:
        record = extract_first_turn(row.get("conversations", []))

        if record is None:
            continue

        total_len = len(record["instruction"]) + len(record["output"])
        if total_len < args.min_len or total_len > args.max_len:
            continue

        records.append(record)
        collected += 1

        if collected % 200 == 0:
            print(f"{collected}/{args.num_samples} terkumpul...")

        if collected >= args.num_samples:
            break

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Selesai. {collected} sample tersimpan di {output_path}")


if __name__ == "__main__":
    main()
