"""
Simulasi real-time data ingestion ke Kafka.

Membaca dataset instruction-tuning (JSONL) baris per baris dan
mem-publish tiap baris sebagai pesan Kafka ke topic `raw-stream`,
dengan delay acak untuk mensimulasikan traffic real-time.
"""

import json
import os
import random
import time
import logging
from pathlib import Path

from confluent_kafka import Producer
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("producer")

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9094")
RAW_TOPIC = os.getenv("RAW_TOPIC", "raw-stream")
DATA_PATH = os.getenv("DATA_PATH", "C:/Users/user/Documents/Python program/kafka-llm-pipeline/sample_dataset.jsonl")
MIN_DELAY = float(os.getenv("MIN_DELAY_SEC", "0.05"))
MAX_DELAY = float(os.getenv("MAX_DELAY_SEC", "0.5"))


def delivery_report(err, msg):
    if err is not None:
        logger.error(f"Gagal kirim pesan: {err}")
    else:
        logger.debug(f"Terkirim ke {msg.topic()} [{msg.partition()}]")


def load_dataset(path: str):
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan di {path}. "
            "Siapkan file JSONL dengan field 'instruction' dan 'output' per baris."
        )
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_producer() -> Producer:
    return Producer({
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "client.id": "llm-stream-producer",
    })


def main():
    producer = build_producer()
    total_sent = 0

    logger.info(f"Mulai streaming dari {DATA_PATH} ke topic '{RAW_TOPIC}'")

    for record in load_dataset(DATA_PATH):
        payload = {
            "instruction": record.get("instruction", ""),
            "input": record.get("input", ""),
            "output": record.get("output", ""),
            "source": DATA_PATH,
            "timestamp": time.time(),
        }

        producer.produce(
            topic=RAW_TOPIC,
            value=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            callback=delivery_report,
        )
        producer.poll(0)
        total_sent += 1

        if total_sent % 50 == 0:
            logger.info(f"{total_sent} pesan terkirim")

        time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

    producer.flush()
    logger.info(f"Selesai. Total {total_sent} pesan terkirim ke '{RAW_TOPIC}'")


if __name__ == "__main__":
    main()
