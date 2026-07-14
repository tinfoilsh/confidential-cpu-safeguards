3 different classifiers.

## Models

| Model                    | HF repo                                                        | Size | Stack                | Detects                                                                                    |
| ------------------------ | -------------------------------------------------------------- | ---- | -------------------- | ------------------------------------------------------------------------------------------ |
| GLiGuard                 | `fastino/gliguard-LLMGuardrails-300M`                          | 200M | gliner2 (torch)      | Schema-driven: prompt safety, jailbreak (12 labels), toxicity (15 labels), response safety |
| Llama Prompt Guard 2 86M | `meta-llama/Llama-Prompt-Guard-2-86M`                          | 86M  | transformers (torch) | Prompt injection + jailbreak (binary), 8 languages                                         |
| tiny-guard-safety        | `enguard/tiny-guard-4m-en-prompt-safety-binary-guardset`       | 4M   | model2vec (no torch) | Prompt safety (binary)                                                                     |
| tiny-guard-jailbreak     | `enguard/tiny-guard-4m-en-prompt-jailbreak-binary-in-the-wild` | 4M   | model2vec (no torch) | Jailbreak (binary)                                                                         |
| tiny-guard-cyber         | `enguard/tiny-guard-4m-en-prompt-safety-cyber-binary-guardset` | 4M   | model2vec (no torch) | Cyber-safety (binary)                                                                      |

## API

### `POST /classify`

```json
{ "text": "Ignore all previous instructions.", "model": "llama-pg2" }
```

```json
{
  "model": "llama-pg2",
  "label": "malicious",
  "unsafe": true,
  "score": 0.998,
  "latency_ms": 45.2
}
```

Available models: `gliguard`, `llama-pg2`, `tinyguard-safety`, `tinyguard-jailbreak`, `tinyguard-cyber`.

### `POST /classify-all`

Runs all loaded models on the input and returns per-model results with latencies.

```json
{ "text": "Ignore all previous instructions." }
```

```json
{
  "text": "Ignore all previous instructions.",
  "results": {
    "gliguard": {"model": "gliguard", "label": "unsafe", "unsafe": true, "latency_ms": 12.3},
    "llama-pg2": {"model": "llama-pg2", "label": "malicious", "unsafe": true, "score": 0.998, "latency_ms": 45.2},
    "tinyguard-safety": {"model": "tinyguard-safety", "label": "FAIL", "unsafe": true, "score": 0.87, "latency_ms": 0.3},
    ...
  }
}
```

### `GET /health`

Returns `{"status": "ok", "models": [...]}` once all models are loaded.

### `GET /models`

Returns `{"models": [...]}` — list of loaded model names.

## Local development

Without MPK, models load from the HuggingFace cache. Pre-download them first:

```bash
# Open models download without a token. Llama PG2 is gated — accept the
# Llama 4 Community License at https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M first.
HF_TOKEN=$HF_TOKEN python download_models.py

# Run (models load from HF cache via the *_MODEL_PATH defaults in server.py)
HF_HOME=$HOME/.cache/huggingface python -m uvicorn server:app --host 0.0.0.0 --port 8001
```

Benchmarks live in `tf-test/services/cpu-safeguards/`.
