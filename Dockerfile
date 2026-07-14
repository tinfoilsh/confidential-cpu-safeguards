# CPU image for multi-model safeguard benchmarking.
# Five classifiers loaded in one container:
#   gliguard            (fastino/gliguard-LLMGuardrails-300M, 200M, GLiNER2)
#   llama-pg2           (meta-llama/Llama-Prompt-Guard-2-86M, 86M, transformers)
#   tinyguard-safety    (enguard/tiny-guard-4m, 4M, model2vec)
#   tinyguard-jailbreak (enguard/tiny-guard-4m, 4M, model2vec)
#   tinyguard-cyber     (enguard/tiny-guard-4m, 4M, model2vec)
#
# Models are mounted as verified model packs (MPK) at boot — read-only
# filesystem at /tinfoil/mpk/. No HuggingFace download or egress required.
# Model paths are set via env vars in tinfoil-config.yml.

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch (from the PyTorch CPU index, not PyPI which serves the
# 2GB+ CUDA wheel by default on Linux). gliner2 and transformers both
# declare torch as a dependency; pre-installing it means pip skips it.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY server.py /app/server.py

WORKDIR /app

ENV NUM_THREADS=64 \
    MAX_CONCURRENCY=1

EXPOSE 8001

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
