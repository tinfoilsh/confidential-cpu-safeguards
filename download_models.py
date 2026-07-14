"""Download all models to the HF cache for LOCAL DEVELOPMENT only.

Not used in the Docker image — in production, models are mounted as
verified model packs (MPK) at /tinfoil/mpk/. This script is for running
the server locally without MPK:

    python download_models.py
    python server.py  # loads from HF cache via *_MODEL_PATH defaults

The Llama Prompt Guard 2 model is gated (Llama 4 Community License) and
requires HF_TOKEN. The other four models are open access.
"""

from huggingface_hub import snapshot_download

OPEN_MODELS = [
    "fastino/gliguard-LLMGuardrails-300M",
    "enguard/tiny-guard-4m-en-prompt-safety-binary-guardset",
    "enguard/tiny-guard-4m-en-prompt-jailbreak-binary-in-the-wild",
    "enguard/tiny-guard-4m-en-prompt-safety-cyber-binary-guardset",
]

GATED_MODELS = [
    "meta-llama/Llama-Prompt-Guard-2-86M",
]


def main() -> None:
    import os

    token = os.environ.get("HF_TOKEN")

    for repo in OPEN_MODELS:
        print(f"Downloading {repo}...", flush=True)
        snapshot_download(repo)

    for repo in GATED_MODELS:
        if not token:
            print(f"WARNING: No HF_TOKEN — skipping gated model {repo}", flush=True)
            continue
        print(f"Downloading {repo} (gated)...", flush=True)
        snapshot_download(repo, token=token)

    print("All downloads complete.", flush=True)


if __name__ == "__main__":
    main()
