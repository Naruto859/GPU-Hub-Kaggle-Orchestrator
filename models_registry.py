"""
Model Registry for Autonomous Kaggle Uncensored Infrastructure.
Contains verified abliterated GGUF models.
Using -sm layer (pipeline parallelism) which works 100% on Dual T4 CUDA.
Maximized context windows via YaRN RoPE scaling and q4_0 KV caching.
"""

import json
import re
import threading
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CUSTOM_MODELS_PATH = BASE_DIR / "custom_models.json"
_CUSTOM_LOCK = threading.RLock()

MODELS = {
    "cybertiel-coder-35b-a3b": {
        "id": "cybertiel-coder-35b-a3b",
        "name": "CyberTiel-Coder-35B-A3B (Abliterated MoE)",
        "repo_id": "peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF",
        "filename": "Cyber-Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf",
        "parameters": "35B MoE (~3.4B active)",
        "quant": "UD-Q4_K_XL (Cyber-imatrix)",
        "filesize_gb": 22.36,
        "context_length": 262144,
        "context_display": "262,144 tokens (262k / ~2.75 lakh)",
        "rope_scaling": "YaRN scale 8 (orig 32k)",
        "kv_cache": "q4_0 / q4_0",
        "vram_requirement": "24.8 GB / 32 GB (Dual T4)",
        "speed": "40-55 tok/s (MoE Dual T4)",
        "accelerator": "gpu",
        "hardware": "Dual Tesla T4",
        "hardware_detail": "2 × 16 GB GDDR6 (32 GB total)",
        "runtime": "llama.cpp · -sm layer · -ts 1,1",
        "max_concurrency": 1,
        "rank": 1,
        "description": "State-of-the-art offensive/defensive cybersecurity & agentic coding model. Zero refusals on HarmBench (all categories). Calibrated on Exploit-DB, HackTricks, PayloadsAllTheThings and Cybench CTFs. Scaled to 262k context (~2.75 lakh) via YaRN RoPE scale 8 with q4_0 KV cache.",
        "server_args": [
            "-np", "1",
            "--rope-scaling", "yarn",
            "--rope-scale", "8",
            "--yarn-orig-ctx", "32768",
            "-c", "262144",
            "-ctk", "q4_0",
            "-ctv", "q4_0",
            "-ngl", "52",
            "-sm", "layer",
            "-ts", "1,1",
            "-b", "2048",
            "-ub", "512",
            "-fa", "auto",
            "--threads", "4"
        ]
    },
    "qwen3.8-27b-abliterated": {
        "id": "qwen3.8-27b-abliterated",
        "name": "Qwen3.8-27B-Instruct (Abliterated RVN)",
        "repo_id": "cole17e/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF",
        "filename": "RVN-Q4_K_M-multilingual.gguf",
        "parameters": "27B Dense (Gen 3.8)",
        "quant": "Q4_K_M",
        "filesize_gb": 16.55,
        "context_length": 262144,
        "context_display": "262,144 tokens (262k)",
        "rope_scaling": "YaRN scale 8 (orig 32k)",
        "kv_cache": "q4_0 / q4_0",
        "vram_requirement": "22.5 GB / 32 GB (Dual T4)",
        "speed": "28-35 tok/s (Dual T4)",
        "accelerator": "gpu",
        "hardware": "Dual Tesla T4",
        "hardware_detail": "2 × 16 GB GDDR6 (32 GB total)",
        "runtime": "llama.cpp · -sm layer · -ts 1,1",
        "max_concurrency": 1,
        "rank": 2,
        "description": "Latest Generation Alibaba Qwen3.8 model. ARA-refined 3x triple-pass abliterated (0 refusals on security tasks). Scaled to 262k context via YaRN RoPE scale 8 with q4_0 KV cache on Dual T4.",
        "server_args": [
            "-np", "1",
            "--rope-scaling", "yarn",
            "--rope-scale", "8",
            "--yarn-orig-ctx", "32768",
            "-c", "262144",
            "-ctk", "q4_0",
            "-ctv", "q4_0",
            "-ngl", "58",
            "-sm", "layer",
            "-ts", "1,1",
            "-b", "2048",
            "-ub", "512",
            "-fa", "auto",
            "--threads", "4"
        ]
    },
    "qwen2.5-coder-32b-abliterated": {
        "id": "qwen2.5-coder-32b-abliterated",
        "name": "Qwen2.5-Coder-32B-Instruct (Abliterated)",
        "repo_id": "bartowski/Qwen2.5-Coder-32B-Instruct-abliterated-GGUF",
        "filename": "Qwen2.5-Coder-32B-Instruct-abliterated-Q4_K_M.gguf",
        "parameters": "32.5B Dense",
        "quant": "Q4_K_M",
        "filesize_gb": 19.85,
        "context_length": 131072,
        "context_display": "131,072 tokens (131k)",
        "rope_scaling": "YaRN scale 4 (orig 32k)",
        "kv_cache": "q4_0 / q4_0",
        "vram_requirement": "26.5 GB / 32 GB (Dual T4)",
        "speed": "20-25 tok/s (Dual T4)",
        "accelerator": "gpu",
        "hardware": "Dual Tesla T4",
        "hardware_detail": "2 × 16 GB GDDR6 (32 GB total)",
        "runtime": "llama.cpp · -sm layer · -ts 1,1",
        "max_concurrency": 1,
        "rank": 3,
        "description": "Specialized coding powerhouse with refusal vector ablation. HumanEval ~90%. Scaled to 131k context via YaRN RoPE scale 4 with q4_0 KV cache on Dual T4.",
        "server_args": [
            "-np", "1",
            "--rope-scaling", "yarn",
            "--rope-scale", "4",
            "--yarn-orig-ctx", "32768",
            "-c", "131072",
            "-ctk", "q4_0",
            "-ctv", "q4_0",
            "-ngl", "52",
            "-sm", "layer",
            "-ts", "1,1",
            "-b", "2048",
            "-ub", "512",
            "-fa", "auto",
            "--threads", "4"
        ]
    },
    "llama-3.3-70b-abliterated": {
        "id": "llama-3.3-70b-abliterated",
        "name": "Llama-3.3-70B-Instruct (Abliterated)",
        "repo_id": "bartowski/Llama-3.3-70B-Instruct-abliterated-GGUF",
        "filename": "Llama-3.3-70B-Instruct-abliterated-IQ3_M.gguf",
        "parameters": "70.6B Dense",
        "quant": "IQ3_M",
        "filesize_gb": 28.5,
        "context_length": 65536,
        "context_display": "65,536 tokens (65k YaRN)",
        "rope_scaling": "YaRN (scale 2, orig 32k)",
        "kv_cache": "q4_0 / q4_0",
        "vram_requirement": "31.5 GB / 32 GB (Dual T4)",
        "speed": "8-12 tok/s (Dual T4)",
        "accelerator": "gpu",
        "hardware": "Dual Tesla T4",
        "hardware_detail": "2 × 16 GB GDDR6 (32 GB total)",
        "runtime": "llama.cpp · -sm layer · -ts 1,1",
        "max_concurrency": 1,
        "rank": 4,
        "description": "Maximum parameter reasoning giant. IQ3_M quant with scaled 65k context and q4_0 KV cache across Dual T4 VRAM.",
        "server_args": [
            "-np", "1",
            "--rope-scaling", "yarn",
            "--rope-scale", "2",
            "--yarn-orig-ctx", "32768",
            "-c", "65536",
            "-ctk", "q4_0",
            "-ctv", "q4_0",
            "-ngl", "40",
            "-sm", "layer",
            "-ts", "1,1",
            "-b", "1024",
            "-ub", "256",
            "-fa", "auto",
            "--threads", "4"
        ]
    },
    "glm-5.3-flash": {
        "id": "glm-5.3-flash",
        "name": "GLM-5.3-Flash (320B MoE · Kaggle TPU v5e-8 · 262k Context)",
        "repo_id": "zai-org/GLM-5.3-Flash",
        "filename": "ARahim3/kaggle-tpu-lab glm53-flash JAX workflow",
        "parameters": "320B MoE (18B active)",
        "quant": "3-bit experts / int8 rest",
        "filesize_gb": 121.0,
        "context_length": 262144,
        "context_display": "262,144 tokens (native 262k)",
        "rope_scaling": "Native 262k context",
        "kv_cache": "Int8 latent cache · position sharded",
        "vram_requirement": "TPU v5e-8 · 128 GB HBM (8 × 16 GB)",
        "speed": "~165 tok/s (single stream · DFlash 2)",
        "accelerator": "tpu",
        "hardware": "Kaggle TPU v5e-8",
        "hardware_detail": "8 chips · 128 GB aggregate HBM",
        "runtime": "ARahim3 JAX + Pallas · 3 concurrent streams",
        "kernel_workflow": "glm53_tpu",
        "dataset_sources": [
            "rahim3/glm53-flash-iq3xxs-1",
            "rahim3/glm53-flash-iq3xxs-2",
            "rahim3/glm53-flash-serve-v3"
        ],
        "native_anthropic": True,
        "max_concurrency": 3,
        "rank": 1,
        "description": "GLM-5.3-Flash on Kaggle's free TPU v5e-8 using the open-source ARahim3/kaggle-tpu-lab JAX/Pallas engine. Native OpenAI and Anthropic APIs, 262k context, continuous batching and DFlash 2 speculative decoding.",
        "server_args": []
    },
    "qwen3.8-flash-next-180b": {
        "id": "qwen3.8-flash-next-180b",
        "name": "Qwen3.8-Flash-Next (180B MoE · Kaggle TPU v5e-8 · 262k Context)",
        "repo_id": "Qwen/Qwen3.8-Flash-Next",
        "filename": "External TPU quantized serving recipe required",
        "parameters": "180B footprint (125B core · 6B active)",
        "quant": "TPU-compatible sub-4-bit recipe required",
        "filesize_gb": 119.0,
        "context_length": 262144,
        "context_display": "262,144 tokens (native 262k)",
        "rope_scaling": "Native 262k · YaRN extensible",
        "kv_cache": "Recipe-dependent TPU cache",
        "vram_requirement": "TPU v5e-8 · 128 GB HBM (quantized weights required)",
        "speed": "Recipe-dependent",
        "accelerator": "tpu",
        "hardware": "Kaggle TPU v5e-8",
        "hardware_detail": "8 chips · 128 GB aggregate HBM",
        "runtime": "Registry profile · verified serving recipe not bundled",
        "launch_supported": False,
        "max_concurrency": 1,
        "rank": "Preview",
        "description": "Visible compatibility profile for Qwen3.8-Flash-Next. A verified free-TPU serving recipe is not bundled, so GPU Hub blocks launch instead of silently starting the 27B workflow or wasting quota.",
        "server_args": []
    },
    "qwen3.8-27b-tpu": {
        "id": "qwen3.8-27b-tpu",
        "name": "Qwen3.8-27B-Instruct (Kaggle TPU v5e-8 · 262k Context)",
        "repo_id": "Qwen/Qwen3.8-27B",
        "filename": "ARahim3/kaggle-tpu-lab qwen38 JAX/vLLM workflow",
        "parameters": "27B Dense (BF16)",
        "quant": "BF16 · MTP speculative decoding",
        "filesize_gb": 55.0,
        "context_length": 262144,
        "context_display": "262,144 tokens (native 262k)",
        "rope_scaling": "Native 262k context",
        "kv_cache": "TPU paged cache",
        "vram_requirement": "TPU v5e-8 · 128 GB HBM (8 × 16 GB)",
        "speed": "~130 tok/s single · ~540 tok/s at 8 streams",
        "accelerator": "tpu",
        "hardware": "Kaggle TPU v5e-8",
        "hardware_detail": "8 chips · 128 GB aggregate HBM",
        "runtime": "ARahim3 vLLM-TPU + MTP patch",
        "kernel_workflow": "qwen38_tpu",
        "dataset_sources": ["rahim3/qwen3-8-27b-bf16", "rahim3/qwen38-tpu-env-v5e8"],
        "native_anthropic": False,
        "max_concurrency": 4,
        "rank": 1,
        "description": "Qwen3.8-27B BF16 on Kaggle TPU v5e-8 using ARahim3/kaggle-tpu-lab with native 262k context, MTP speculative decoding, and OpenAI-compatible serving.",
        "server_args": []
    }
}


def _custom_model_from_input(data):
    name = str(data.get("name") or "").strip()
    repo_id = str(data.get("repo_id") or "").strip()
    filename = str(data.get("filename") or "").strip()
    try:
        context_length = int(data.get("context_length") or 32768)
    except (TypeError, ValueError):
        raise ValueError("Context Window must be an integer")
    if not name or "/" not in repo_id or not filename.lower().endswith(".gguf"):
        raise ValueError("Model Name, valid HF Repo ID, and .gguf Filename are required")
    if context_length < 2048 or context_length > 262144:
        raise ValueError("Context Window must be between 2,048 and 262,144")
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:48]
    model_id = str(data.get("id") or f"custom-{slug}")
    return {
        "id": model_id,
        "name": name,
        "repo_id": repo_id,
        "filename": filename,
        "parameters": str(data.get("parameters") or "Custom GGUF"),
        "quant": str(data.get("quant") or "GGUF"),
        "filesize_gb": float(data.get("filesize_gb") or 0),
        "context_length": context_length,
        "context_display": f"{context_length:,} tokens",
        "rope_scaling": "Model default",
        "kv_cache": "q4_0 / q4_0",
        "vram_requirement": "Dual T4 · validate model fit",
        "speed": "Depends on architecture and quantization",
        "accelerator": "gpu",
        "hardware": "Dual Tesla T4",
        "hardware_detail": "2 × 16 GB GDDR6 (32 GB total)",
        "runtime": "llama.cpp custom GGUF",
        "max_concurrency": 1,
        "rank": "Custom",
        "description": f"Custom Hugging Face GGUF: {repo_id}/{filename}",
        "custom": True,
        "server_args": [
            "-np", "1", "-c", str(context_length), "-ctk", "q4_0", "-ctv", "q4_0",
            "-ngl", "999", "-sm", "layer", "-ts", "1,1", "-b", "1024", "-ub", "256",
            "-fa", "auto", "--threads", "4"
        ]
    }


def load_custom_models():
    if not CUSTOM_MODELS_PATH.exists():
        return {}
    try:
        raw = json.loads(CUSTOM_MODELS_PATH.read_text(encoding="utf-8"))
        items = raw if isinstance(raw, list) else raw.get("models", [])
        return {model["id"]: model for model in (_custom_model_from_input(item) for item in items)}
    except Exception:
        return {}


def save_custom_model(data):
    model = _custom_model_from_input(data)
    with _CUSTOM_LOCK:
        custom = load_custom_models()
        if model["id"] in MODELS and not MODELS[model["id"]].get("custom"):
            raise ValueError("A built-in model already uses this identifier")
        custom[model["id"]] = model
        tmp = CUSTOM_MODELS_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"models": list(custom.values())}, indent=2), encoding="utf-8")
        tmp.replace(CUSTOM_MODELS_PATH)
        MODELS[model["id"]] = model
    return model


_PRIORITY = ["qwen3.8-flash-next-180b", "glm-5.3-flash", "qwen3.8-27b-tpu", "qwen3.8-27b-abliterated"]
MODELS = {key: MODELS[key] for key in _PRIORITY if key in MODELS} | {key: value for key, value in MODELS.items() if key not in _PRIORITY}
MODELS.update(load_custom_models())
AVAILABLE_MODELS = MODELS
DEFAULT_MODEL_ID = "qwen3.8-27b-abliterated"
