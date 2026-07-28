# Kafka-Powered Streaming Pipeline for LLM Training (Llama 3.2 3B)

Pipeline yang mensimulasikan ingestion data real-time via Apache Kafka, melakukan
cleaning/preprocessing/micro-batching secara streaming, lalu menjalankan
incremental fine-tuning (LoRA/QLoRA) pada Llama 3.2 3B setiap kali cukup data baru
terkumpul.

## Arsitektur

```
Data source -> Kafka producer -> Kafka (raw-stream / processed-stream / dlq)
            -> Stream processor (clean, dedup, filter, micro-batch)
            -> Fine-tuning service (Llama 3.2 3B + LoRA/QLoRA)
            -> Registry (checkpoint adapter) + MLflow (metrics/logs)
```

## Prasyarat

- Docker & Docker Compose
- Python 3.10+
- GPU dengan minimal ~8-12 GB VRAM (untuk 4-bit QLoRA). Tanpa GPU, ganti
  `finetune_llama.py` untuk memakai model yang jauh lebih kecil (mis. `Qwen2.5-0.5B`)
  agar tetap bisa didemokan di CPU/Colab.
- Akses ke model gated `meta-llama/Llama-3.2-3B` di Hugging Face (request akses
  di halaman model, lalu `huggingface-cli login`).

## Setup

```bash
cd kafka-llm-pipeline
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # sesuaikan value kalau perlu
```

Jalankan Kafka:

```bash
docker compose up -d
```

Kafka UI tersedia di `http://localhost:8080` untuk memantau topic dan pesan secara visual.

## Menjalankan pipeline

Buka 3 terminal terpisah:

**Terminal 1 — Stream processor (jalankan lebih dulu, supaya siap menerima pesan)**
```bash
python consumer/preprocessor.py
```

**Terminal 2 — Producer (simulasi data masuk)**
```bash
python producer/stream_producer.py
```

**Terminal 3 — Training service**
```bash
# one-shot: proses batch yang tersedia lalu keluar
python training/finetune_llama.py

# atau watch mode: terus memantau & training tiap ada batch baru
python training/finetune_llama.py --watch
```

Lihat progres training (loss, params) di MLflow UI:
```bash
mlflow ui
```

## Registry checkpoint

```bash
# lihat semua checkpoint adapter yang tersimpan
python registry/checkpoint_manager.py --list

# push salah satu checkpoint ke Hugging Face Hub (private repo)
python registry/checkpoint_manager.py --push adapter_20260728_101500
```

## Catatan implementasi

- **Micro-batching**: `preprocessor.py` mem-flush buffer ke file JSONL berdasarkan
  `BATCH_SIZE` (jumlah pesan) atau `BATCH_FLUSH_SEC` (interval waktu), mana yang
  tercapai lebih dulu — pola windowing standar pada stream processing.
- **DLQ**: pesan yang gagal parsing, terlalu pendek, duplikat, atau melebihi
  `MAX_TOKEN_LEN` dikirim ke topic `dlq` alih-alih di-drop diam-diam, supaya bisa
  diinspeksi.
- **Incremental fine-tuning**: setiap training cycle melanjutkan dari adapter
  checkpoint terakhir (`PeftModel.from_pretrained(..., is_trainable=True)`),
  bukan training ulang dari base model — ini yang membedakan pipeline ini dari
  batch training biasa.
- **Skala demo vs produksi**: `MIN_BATCHES_TO_TRAIN` kecil (default 3) supaya
  gampang didemokan dengan `data/sample_dataset.jsonl`. Untuk dataset besar,
  naikkan nilainya dan pertimbangkan menjalankan training service di GPU terpisah
  dari stream processor.

## Ide pengembangan lanjutan

- Tambahkan Prometheus exporter untuk Kafka consumer lag + dashboard Grafana.
- Ganti penyimpanan batch dari file lokal ke object storage (S3/MinIO) untuk
  setup multi-node.
- Tambahkan endpoint FastAPI untuk trigger training manual atau cek status cycle
  terakhir.
