"""
Kafka consumer yang membaca raw-stream, melakukan cleaning,
filtering, deduplication, dan micro-batching, lalu mem-publish
hasil bersih ke processed-stream (atau dlq kalau invalid).

Micro-batch (window berdasarkan jumlah pesan atau waktu) ditulis
ke folder data/batches/ sebagai file JSONL untuk dikonsumsi oleh
training service.
"""

import json
import os
import time
import hashlib
import logging
from pathlib import Path
from typing import Optional

from confluent_kafka import Consumer, Producer
from transformers import AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("preprocessor")

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9094")
RAW_TOPIC = os.getenv("RAW_TOPIC", "raw-stream")
PROCESSED_TOPIC = os.getenv("PROCESSED_TOPIC", "processed-stream")
DLQ_TOPIC = os.getenv("DLQ_TOPIC", "dlq")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-3B")

MIN_TEXT_LEN = int(os.getenv("MIN_TEXT_LEN", "10"))
MAX_TOKEN_LEN = int(os.getenv("MAX_TOKEN_LEN", "1024"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
BATCH_FLUSH_SEC = int(os.getenv("BATCH_FLUSH_SEC", "30"))

BATCH_DIR = Path(os.getenv("BATCH_DIR", "data/batches"))
BATCH_DIR.mkdir(parents=True, exist_ok=True)

# Tokenizer dipakai untuk cek panjang token, bukan untuk tokenisasi final training
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
seen_hashes = set()


def build_consumer() -> Consumer:
    return Consumer({
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "group.id": "llm-preprocessor-group",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": True,
    })


def build_producer() -> Producer:
    return Producer({
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "client.id": "llm-preprocessor-producer",
    })


def clean_text(text: str) -> str:
    return " ".join(text.strip().split())


def is_duplicate(text: str) -> bool:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if h in seen_hashes:
        return True
    seen_hashes.add(h)
    return False


def validate_and_clean(record: dict) -> Optional[dict]:
    instruction = clean_text(record.get("instruction", ""))
    output = clean_text(record.get("output", ""))

    if len(instruction) < MIN_TEXT_LEN or len(output) < MIN_TEXT_LEN:
        return None

    combined = f"{instruction}\n{output}"
    if is_duplicate(combined):
        return None

    token_count = len(tokenizer.encode(combined))
    if token_count > MAX_TOKEN_LEN:
        return None

    return {
        "instruction": instruction,
        "input": clean_text(record.get("input", "")),
        "output": output,
        "token_count": token_count,
        "processed_at": time.time(),
    }


def flush_batch(buffer: list, producer: Producer):
    if not buffer:
        return

    batch_id = int(time.time() * 1000)
    batch_path = BATCH_DIR / f"batch_{batch_id}.jsonl"

    with open(batch_path, "w", encoding="utf-8") as f:
        for item in buffer:
            line = json.dumps(item, ensure_ascii=False)
            f.write(line + "\n")
            producer.produce(topic=PROCESSED_TOPIC, value=line.encode("utf-8"))

    producer.flush()
    logger.info(f"Batch tersimpan: {batch_path} ({len(buffer)} sample)")


def main():
    consumer = build_consumer()
    producer = build_producer()
    consumer.subscribe([RAW_TOPIC])

    buffer = []
    last_flush = time.time()

    logger.info(f"Mendengarkan topic '{RAW_TOPIC}'...")

    try:
        while True:
            msg = consumer.poll(timeout=1.0)

            if msg is not None:
                if msg.error():
                    logger.error(f"Consumer error: {msg.error()}")
                else:
                    try:
                        record = json.loads(msg.value().decode("utf-8"))
                        cleaned = validate_and_clean(record)

                        if cleaned:
                            buffer.append(cleaned)
                        else:
                            producer.produce(topic=DLQ_TOPIC, value=msg.value())
                    except Exception as e:
                        logger.warning(f"Gagal proses pesan, dikirim ke DLQ: {e}")
                        producer.produce(topic=DLQ_TOPIC, value=msg.value())

            should_flush_size = len(buffer) >= BATCH_SIZE
            should_flush_time = (time.time() - last_flush) >= BATCH_FLUSH_SEC and buffer

            if should_flush_size or should_flush_time:
                flush_batch(buffer, producer)
                buffer = []
                last_flush = time.time()

    except KeyboardInterrupt:
        logger.info("Consumer dihentikan, flush sisa buffer...")
        flush_batch(buffer, producer)
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
