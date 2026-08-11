# Kafka-Powered Streaming Pipeline for LLM Training (Llama 3.2 3B)

A pipeline that simulates real-time data ingestion via Apache Kafka, performs
cleaning/preprocessing/micro-batching in a streaming fashion, then runs
incremental fine-tuning (LoRA/QLoRA) on Llama 3.2 3B every time enough new
data has accumulated.

## Architecture

```
Data source -> Kafka producer -> Kafka (raw-stream / processed-stream / dlq)
            -> Stream processor (clean, dedup, filter, micro-batch)
            -> Fine-tuning service (Llama 3.2 3B + LoRA/QLoRA)
            -> Registry (checkpoint adapter) + MLflow (metrics/logs)
```

## Prerequisites

- Docker & Docker Compose
- Python 3.10+
- A GPU with at least ~8-12 GB VRAM (for 4-bit QLoRA). Without a GPU, swap
  `finetune_llama.py` to use a much smaller model (e.g. `Qwen2.5-0.5B`) so it
  can still be demoed on CPU/Colab.
- Access to the gated `meta-llama/Llama-3.2-3B` model on Hugging Face (request
  access on the model page, then run `huggingface-cli login`).

## Setup

```bash
cd kafka-llm-pipeline
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # adjust values if needed
```

Start Kafka:

```bash
docker compose up -d
```

The Kafka UI is available at `http://localhost:8080` for visually monitoring topics and messages.

## Running the pipeline

Open 3 separate terminals:

**Terminal 1 — Stream processor (run this first, so it's ready to receive messages)**
```bash
python consumer/preprocessor.py
```

**Terminal 2 — Producer (simulates incoming data)**
```bash
python producer/stream_producer.py
```

**Terminal 3 — Training service**
```bash
# one-shot: process the available batch then exit
python training/finetune_llama.py

# or watch mode: keep monitoring & train whenever a new batch arrives
python training/finetune_llama.py --watch
```

Watch training progress (loss, params) in the MLflow UI:
```bash
mlflow ui
```

## Checkpoint registry

```bash
# list all saved adapter checkpoints
python registry/checkpoint_manager.py --list

# push a checkpoint to the Hugging Face Hub (private repo)
python registry/checkpoint_manager.py --push adapter_20260728_101500
```

## Testing inference

Once you have at least 1 checkpoint, compare the base model's output against the fine-tuned model's:

```bash
# compare base model vs fine-tuned for 3 sample instructions
python inference/test_inference.py --compare-base

# custom instruction, with more room for output length
python inference/test_inference.py --compare-base --prompt "Explain what Kafka is" --max-new-tokens 300

# test a specific checkpoint (not just the latest)
python inference/test_inference.py --checkpoint adapter_20260805_103719 --compare-base
```

This script loads the 4-bit base model only once, then uses `model.disable_adapter()`
from `peft` to compare behavior with/without the LoRA adapter — without needing
to load the model twice.

## Evaluation & Findings

Results from checkpoint `adapter_20260805_103719` (1 epoch, ~3000 samples from a
subset of `teknium/OpenHermes-2.5`), compared against the base model, for 3
technical-domain instructions:

**1. Term disambiguation — the clearest change**

When asked "Explain the concept of LoRA in language model fine-tuning," the base
model got it completely wrong: it thought LoRA referred to LoRaWAN (a long-range
radio technology for IoT). The fine-tuned model correctly steered the answer
toward the right domain (low-rank adaptation for fine-tuning), even though
further technical details were still inaccurate (hallucinating about a "Fourier
basis" and TTS). This is the clearest evidence that training successfully shifted
the model toward the training data's domain (ML/LLM engineering), rather than
just being noise.

**2. Language drift — the model switches to English**

For the Kafka question, the base model consistently answered in Indonesian
(matching the prompt's language), but the fine-tuned model answered entirely in
English. This is most likely because the `OpenHermes-2.5` dataset is dominated
by English text — even a small portion of 1 epoch of training was enough to
shift the model's language preference. This is a real phenomenon in fine-tuning
(*language drift*/partial *catastrophic forgetting*), not a pipeline bug.

**3. More structured answers, but prone to hallucinated details**

Overall, the fine-tuned model's answers are longer and more structured (numbered
points, explanations of technical terms) compared to the more concise base
model. The trade-off: specific details (statistics, product names, precise
technical claims) aren't always accurate — expected for small-scale training
(1 epoch, thousands rather than millions of samples).

### Recommendations for further improvement

- **Mitigate language drift**: mix Indonesian-language data (e.g.
  `FreedomIntelligence/alpaca-gpt4-indonesian`) into the training set, or filter
  the OpenHermes subset down to conversations that are already in Indonesian.
- **Reduce hallucination**: increase the number of training samples and/or
  epochs, while monitoring `train_loss` in MLflow to avoid overfitting to a
  small dataset.
- **More systematic evaluation**: instead of just reading outputs manually,
  consider automated metrics (perplexity on a held-out set, or LLM-as-judge) to
  quantitatively compare checkpoints as the `--watch` pipeline produces many
  checkpoints over time.

## Implementation notes

- **Micro-batching**: `preprocessor.py` flushes its buffer to a JSONL file based
  on `BATCH_SIZE` (message count) or `BATCH_FLUSH_SEC` (time interval), whichever
  is reached first — a standard windowing pattern in stream processing.
- **DLQ**: messages that fail to parse, are too short, are duplicates, or exceed
  `MAX_TOKEN_LEN` are sent to the `dlq` topic instead of being silently dropped,
  so they can be inspected.
- **Incremental fine-tuning**: each training cycle continues from the last
  adapter checkpoint (`PeftModel.from_pretrained(..., is_trainable=True)`),
  rather than retraining from the base model — this is what distinguishes this
  pipeline from ordinary batch training.
- **Demo scale vs. production**: `MIN_BATCHES_TO_TRAIN` is small (default 3) so
  it's easy to demo with `data/sample_dataset.jsonl`. For larger datasets,
  increase this value and consider running the training service on a separate
  GPU from the stream processor.

## Further development ideas

- Add a Prometheus exporter for Kafka consumer lag plus a Grafana dashboard.
- Switch batch storage from local files to object storage (S3/MinIO) for
  multi-node setups.
- Add a FastAPI endpoint to trigger training manually or check the status of the
  last cycle.
