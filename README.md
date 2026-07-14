# confidential-cpu-safeguards

CPU enclave serving five prompt-injection / safety classifiers for benchmarking. All models run on CPU with no GPU.

Unlike `confidential-pii-cpu` (single OpenAI Privacy Filter model, span-level PII redaction), this repo loads multiple guard models side by side to compare speed and suitability as a prompt-injection classifier — the TODO from the PII writeup.

## Models

| Model                    | HF repo                                                        | Size | Stack                | Detects                                                                                    |
| ------------------------ | -------------------------------------------------------------- | ---- | -------------------- | ------------------------------------------------------------------------------------------ |
| GLiGuard                 | `fastino/gliguard-LLMGuardrails-300M`                          | 200M | gliner2 (torch)      | Schema-driven: prompt safety, jailbreak (12 labels), toxicity (15 labels), response safety |
| Llama Prompt Guard 2 86M | `meta-llama/Llama-Prompt-Guard-2-86M`                          | 86M  | transformers (torch) | Prompt injection + jailbreak (binary), 8 languages                                         |
| tiny-guard-safety        | `enguard/tiny-guard-4m-en-prompt-safety-binary-guardset`       | 4M   | model2vec (no torch) | Prompt safety (binary)                                                                     |
| tiny-guard-jailbreak     | `enguard/tiny-guard-4m-en-prompt-jailbreak-binary-in-the-wild` | 4M   | model2vec (no torch) | Jailbreak (binary)                                                                         |
| tiny-guard-cyber         | `enguard/tiny-guard-4m-en-prompt-safety-cyber-binary-guardset` | 4M   | model2vec (no torch) | Cyber-safety (binary)                                                                      |

None of the models require `trust_remote_code` — all use standard loading paths (safetensors + pip packages). This was a key requirement (the Qwen3Guard-Stream-0.6B alternative was rejected partly for requiring `trust_remote_code`).

Models are mounted as verified model packs (MPK) at boot — read-only filesystem at `/tinfoil/mpk/`. No HuggingFace download or egress required. The `mpk` values in `tinfoil-config.yml` are placeholders until model packs are created.

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

## Deployment

Custom Docker image built in CI (`tinfoil-release.yml`). The `tinfoil-config.yml` image digest and `mpk` values are placeholders until the first release populates them. Model packs must be created via Tinfoil's model packing infrastructure before deployment — fill in the `mpk` values and `*_MODEL_PATH` env vars in `tinfoil-config.yml` with the real mount paths.
