"""CPU safeguard inference server.

Loads five prompt-injection / safety classifiers in one container for
benchmarking on CPU:

  gliguard            fastino/gliguard-LLMGuardrails-300M         (200M, GLiNER2)
  llama-pg2           meta-llama/Llama-Prompt-Guard-2-86M          (86M, transformers)
  tinyguard-safety    enguard/tiny-guard-4m-en-prompt-safety-binary-guardset
  tinyguard-jailbreak enguard/tiny-guard-4m-en-prompt-jailbreak-binary-in-the-wild
  tinyguard-cyber     enguard/tiny-guard-4m-en-prompt-safety-cyber-binary-guardset

Models are mounted as verified model packs (MPK) at boot — read-only
filesystem at /tinfoil/mpk/. No HuggingFace download or egress required.

For local development (without MPK), set the *_MODEL_PATH env vars to
HuggingFace repo IDs and the models will download from the HF cache.
Run download_models.py first to pre-populate the cache.
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

log = logging.getLogger("cpu-safeguards")
logging.basicConfig(level=logging.INFO)

models: dict[str, Any] = {}
semaphores: dict[str, asyncio.Semaphore] = {}

SAFETY_LABELS = ["safe", "unsafe"]

# Each model: (server_key, env_var for model path, HF repo ID fallback for local dev)
MODEL_CONFIGS = [
    ("gliguard", "GLIGUARD_MODEL_PATH", "fastino/gliguard-LLMGuardrails-300M"),
    ("llama-pg2", "PG2_MODEL_PATH", "meta-llama/Llama-Prompt-Guard-2-86M"),
    (
        "tinyguard-safety",
        "TINYGUARD_SAFETY_MODEL_PATH",
        "enguard/tiny-guard-4m-en-prompt-safety-binary-guardset",
    ),
    (
        "tinyguard-jailbreak",
        "TINYGUARD_JAILBREAK_MODEL_PATH",
        "enguard/tiny-guard-4m-en-prompt-jailbreak-binary-in-the-wild",
    ),
    (
        "tinyguard-cyber",
        "TINYGUARD_CYBER_MODEL_PATH",
        "enguard/tiny-guard-4m-en-prompt-safety-cyber-binary-guardset",
    ),
]


def load_models() -> None:
    n_threads = int(os.environ.get("NUM_THREADS", str(os.cpu_count() or 1)))
    max_concurrency = int(os.environ.get("MAX_CONCURRENCY", "1"))

    import torch

    torch.set_num_threads(n_threads)

    # 1. GLiGuard (gliner2 — uses torch internally)
    try:
        from gliner2 import GLiNER2

        key, env_var, repo_id = MODEL_CONFIGS[0]
        path = os.environ.get(env_var, repo_id)
        log.info("Loading GLiGuard from %s...", path)
        gliguard = GLiNER2.from_pretrained(path)
        gliguard.to("cpu")
        gliguard.classify_text("warmup", {"prompt_safety": SAFETY_LABELS})
        models[key] = gliguard
        semaphores[key] = asyncio.Semaphore(max_concurrency)
        log.info("GLiGuard loaded.")
    except Exception:
        log.exception("Failed to load GLiGuard")

    # 2. Llama Prompt Guard 2 86M (transformers)
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        key, env_var, repo_id = MODEL_CONFIGS[1]
        path = os.environ.get(env_var, repo_id)
        log.info("Loading Llama Prompt Guard 2 86M from %s...", path)
        pg2_tokenizer = AutoTokenizer.from_pretrained(path)
        pg2_model = AutoModelForSequenceClassification.from_pretrained(path)
        pg2_model.eval()
        inputs = pg2_tokenizer(
            "warmup", return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            pg2_model(**inputs)
        models[key] = {"tokenizer": pg2_tokenizer, "model": pg2_model}
        semaphores[key] = asyncio.Semaphore(max_concurrency)
        log.info("Llama Prompt Guard 2 86M loaded.")
    except Exception:
        log.exception("Failed to load Llama Prompt Guard 2 86M")

    # 3. tiny-guard-4m × 3 (model2vec — no torch at inference)
    try:
        from model2vec.inference import StaticModelPipeline

        for key, env_var, repo_id in MODEL_CONFIGS[2:]:
            path = os.environ.get(env_var, repo_id)
            log.info("Loading %s from %s...", key, path)
            pipe = StaticModelPipeline.from_pretrained(path)
            pipe.predict(["warmup"])
            models[key] = pipe
            semaphores[key] = asyncio.Semaphore(max_concurrency)
            log.info("%s loaded.", key)
    except Exception:
        log.exception("Failed to load tiny-guard models")

    if not models:
        raise RuntimeError("No models loaded — check logs above.")

    log.info(
        "All models loaded: %s (threads=%d, max_concurrency=%d)",
        sorted(models.keys()),
        n_threads,
        max_concurrency,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield


app = FastAPI(title="CPU Safeguards", lifespan=lifespan)


# --- Request / response models ---


class ClassifyRequest(BaseModel):
    text: str
    model: str


class ClassifyResponse(BaseModel):
    model: str
    label: str
    unsafe: bool
    score: float | None = None
    latency_ms: float


class ClassifyAllRequest(BaseModel):
    text: str


# --- Per-model classify functions ---


def _classify_gliguard(text: str) -> dict:
    result = models["gliguard"].classify_text(text, {"prompt_safety": SAFETY_LABELS})
    label = result.get("prompt_safety", "unknown")
    return {"label": label, "unsafe": label == "unsafe"}


def _classify_llama_pg2(text: str) -> dict:
    import torch

    entry = models["llama-pg2"]
    tokenizer = entry["tokenizer"]
    model = entry["model"]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    scores = torch.softmax(logits, dim=-1)
    pred_id = logits.argmax().item()
    label = model.config.id2label[pred_id].lower()
    return {
        "label": label,
        "unsafe": label == "malicious",
        "score": scores[0][pred_id].item(),
    }


def _classify_tinyguard(text: str, model_key: str) -> dict:
    import numpy as np

    pipe = models[model_key]
    pred = pipe.predict([text])[0]
    proba = np.asarray(pipe.predict_proba([text]))
    classes = list(pipe.classes_)
    # enguard models use FAIL (unsafe) / PASS (safe)
    unsafe_idx = classes.index("FAIL") if "FAIL" in classes else 1
    return {
        "label": pred,
        "unsafe": pred == "FAIL",
        "score": float(proba[0][unsafe_idx]),
    }


CLASSIFIERS = {
    "gliguard": _classify_gliguard,
    "llama-pg2": _classify_llama_pg2,
    "tinyguard-safety": lambda text: _classify_tinyguard(text, "tinyguard-safety"),
    "tinyguard-jailbreak": lambda text: _classify_tinyguard(
        text, "tinyguard-jailbreak"
    ),
    "tinyguard-cyber": lambda text: _classify_tinyguard(text, "tinyguard-cyber"),
}


# --- Endpoints ---


@app.get("/health")
def health():
    return {"status": "ok", "models": sorted(models.keys())}


@app.get("/models")
def list_models():
    return {"models": sorted(models.keys())}


@app.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest):
    if req.model not in CLASSIFIERS:
        raise HTTPException(
            400,
            f"Unknown model: {req.model}. Available: {sorted(CLASSIFIERS.keys())}",
        )
    if req.model not in models:
        raise HTTPException(
            503, f"Model {req.model} failed to load. Check server logs."
        )
    async with semaphores[req.model]:
        start = time.perf_counter()
        result = await run_in_threadpool(CLASSIFIERS[req.model], req.text)
        result["latency_ms"] = (time.perf_counter() - start) * 1000
        result["model"] = req.model
        return result


@app.post("/classify-all")
async def classify_all(req: ClassifyAllRequest):
    results = {}
    for name, fn in CLASSIFIERS.items():
        if name not in models:
            results[name] = {"error": "model not loaded"}
            continue
        async with semaphores[name]:
            start = time.perf_counter()
            try:
                result = await run_in_threadpool(fn, req.text)
                result["latency_ms"] = (time.perf_counter() - start) * 1000
                result["model"] = name
                results[name] = result
            except Exception as e:
                results[name] = {
                    "error": str(e),
                    "latency_ms": (time.perf_counter() - start) * 1000,
                }
    return {"text": req.text, "results": results}
