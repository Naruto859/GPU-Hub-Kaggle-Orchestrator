"""GPU Hub model registry and GGUF filename resolution."""

import json
import re
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List

BASE_DIR = Path(__file__).resolve().parent
CUSTOM_MODELS_PATH = BASE_DIR / "custom_models.json"
_CUSTOM_LOCK = threading.RLock()


def gpu_args(context: int, parallel: int = 2) -> List[str]:
    return [
        "--host", "127.0.0.1", "--port", "8080",
        "--parallel", str(parallel), "-c", str(context),
        "--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
        "-ngl", "99", "-sm", "layer", "-ts", "1,1",
        "-b", "2048", "-ub", "512", "-fa", "on", "--threads", "4",
    ]


def gpu_model(model_id: str, name: str, repo: str, filename: str, parameters: str, context: int,
              rank: int, speed: str, quant: str = "UD-Q4_K_XL", filesize: float = 0.0,
              description: str = "") -> Dict[str, Any]:
    return {
        "id": model_id, "name": name, "repo_id": repo, "filename": filename,
        "gpu_repo_id": repo, "gpu_filename": filename,
        "parameters": parameters, "quant": quant, "filesize_gb": filesize,
        "context_length": context, "context_display": f"{context:,} tokens",
        "rope_scaling": "Native / model-defined long context", "kv_cache": "q4_0 / q4_0",
        "vram_requirement": "Dual Tesla T4 · 32 GB aggregate VRAM",
        "speed": speed, "accelerator": "gpu", "default_accelerator": "gpu",
        "supported_accelerators": ["gpu"], "hardware": "Dual Tesla T4",
        "hardware_detail": "2 × 16 GB GDDR6 (32 GB total)",
        "runtime": "latest llama.cpp CUDA · layer split 1,1", "max_concurrency": 2,
        "rank": rank, "description": description or name, "server_args": gpu_args(context),
    }


MODELS: Dict[str, Dict[str, Any]] = {
    "cybertiel-coder-35b-a3b": gpu_model(
        "cybertiel-coder-35b-a3b", "Cyber-Tiel-Coder-35B-A3B (Uncensored Security & Coding MoE)",
        "peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF", "Cyber-Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf",
        "35B MoE (~3.4B active)", 262144, 1, "40–55 tok/s", filesize=22.36,
        description="Security, coding and autonomous-agent MoE with long context and a verified Dual T4 quant.",
    ),
    "qwen3.8-35b-a3b-distill": gpu_model(
        "qwen3.8-35b-a3b-distill", "Qwen3.8-35B-A3B-Distill (Frontier Agentic MoE · 262k)",
        "empero-ai/Qwen3.8-35B-A3B-Distill-GGUF", "Qwen3.8-35B-A3B-Q4_K_M.gguf",
        "35B MoE (~3B active)", 262144, 2, "45–65 tok/s", quant="Q4_K_M",
        description="Frontier agentic Qwen3.8 distill with low active-parameter latency.",
    ),
    "qwen3.8-27b-instruct": {
        **gpu_model(
            "qwen3.8-27b-instruct", "Qwen3.8-27B-Instruct (Unsloth Dynamic V3.1 Agentic · 256k)",
            "unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-Q4_K_XL.gguf",
            "27B dense", 262144, 3, "28–35 tok/s GPU · ~130 tok/s TPU", filesize=22.0,
            description="Dual-capable Qwen3.8: Unsloth Dynamic GGUF on Dual T4 or BF16 vLLM runner on TPU v5e-8.",
        ),
        "supported_accelerators": ["gpu", "tpu"],
        "default_accelerator": "gpu",
        "tpu_kernel_workflow": "qwen38_tpu",
        "tpu_repo_id": "Qwen/Qwen3.8-27B",
        "tpu_dataset_sources": ["rahim3/qwen3-8-27b-bf16", "rahim3/qwen38-tpu-env-v5e8"],
        "native_anthropic": False,
        "max_concurrency_by_accelerator": {"gpu": 2, "tpu": 4},
    },
    "qwen3.6-35b-a3b-uncensored": gpu_model(
        "qwen3.6-35b-a3b-uncensored", "Qwen3.6-35B-A3B-Uncensored (HauhauCS 0-Refusal MoE)",
        "HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive",
        "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf",
        "35B MoE (~3B active)", 262144, 4, "40–60 tok/s", quant="Q4_K_M",
    ),
    "gemma-4-26b-a4b-it": gpu_model(
        "gemma-4-26b-a4b-it", "Gemma-4-26B-A4B-IT (Google Agentic MoE · 256k)",
        "unsloth/gemma-4-26B-A4B-it-GGUF", "gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf",
        "26B MoE (~4B active)", 262144, 5, "45–70 tok/s",
    ),
    "gemma-4-31b-it": gpu_model(
        "gemma-4-31b-it", "Gemma-4-31B-IT (Google Frontier Dense · 256k)",
        "unsloth/gemma-4-31B-it-GGUF", "gemma-4-31B-it-UD-Q4_K_XL.gguf",
        "31B dense", 262144, 6, "22–32 tok/s",
    ),
    "deepseek-v4-pro-qwen3.5-9b-heretic": gpu_model(
        "deepseek-v4-pro-qwen3.5-9b-heretic", "DeepSeek-V4-Pro-Qwen3.5-9B-Heretic (Uncensored Fast Agent)",
        "mradermacher/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-i1-GGUF",
        "DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-Q4_K_M.gguf",
        "9B hybrid", 131072, 7, "80+ tok/s", quant="Q4_K_M",
    ),
    "deepseek-r1-distill-qwen-32b": gpu_model(
        "deepseek-r1-distill-qwen-32b", "DeepSeek-R1-Distill-Qwen-32B (Reasoning Dense)",
        "bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF", "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf",
        "32B dense", 131072, 8, "18–25 tok/s", quant="Q4_K_M",
    ),
    "qwen2.5-coder-32b-abliterated": gpu_model(
        "qwen2.5-coder-32b-abliterated", "Qwen2.5-Coder-32B-Instruct (Abliterated)",
        "bartowski/Qwen2.5-Coder-32B-Instruct-abliterated-GGUF",
        "Qwen2.5-Coder-32B-Instruct-abliterated-Q4_K_M.gguf", "32.5B dense", 131072, 9,
        "20–25 tok/s", quant="Q4_K_M", filesize=19.85,
    ),
    "llama-3.3-70b-abliterated": gpu_model(
        "llama-3.3-70b-abliterated", "Llama-3.3-70B-Instruct (Abliterated IQ3_M)",
        "bartowski/Llama-3.3-70B-Instruct-abliterated-GGUF",
        "Llama-3.3-70B-Instruct-abliterated-IQ3_M.gguf", "70.6B dense", 65536, 10,
        "8–12 tok/s", quant="IQ3_M", filesize=28.5,
    ),
    "glm-5.3-flash": {
        "id": "glm-5.3-flash", "name": "GLM-5.3-Flash (320B MoE · 18B Active · TPU v5e-8 Only)",
        "repo_id": "zai-org/GLM-5.3-Flash", "filename": "ARahim3 JAX/Pallas TPU workflow",
        "parameters": "320B MoE (18B active)", "quant": "3-bit experts / int8 rest",
        "context_length": 262144, "context_display": "262,144 tokens", "rope_scaling": "Native 262k",
        "kv_cache": "Int8 latent cache", "vram_requirement": "TPU v5e-8 · 128 GB HBM",
        "speed": "~165 tok/s", "accelerator": "tpu", "default_accelerator": "tpu",
        "supported_accelerators": ["tpu"], "hardware": "Kaggle TPU v5e-8",
        "hardware_detail": "8 chips · 128 GB aggregate HBM", "runtime": "ARahim3 JAX + Pallas",
        "kernel_workflow": "glm53_tpu", "dataset_sources": [
            "rahim3/glm53-flash-iq3xxs-1", "rahim3/glm53-flash-iq3xxs-2", "rahim3/glm53-flash-serve-v3"
        ], "native_anthropic": True, "max_concurrency": 3, "rank": 11,
        "description": "TPU-only GLM-5.3-Flash serving recipe.", "server_args": [],
    },
    "qwen3.8-flash-next-tpu": {
        "id": "qwen3.8-flash-next-tpu", "name": "Qwen3.8-Flash-Next (180B MoE · TPU v5e-8 Only)",
        "repo_id": "Qwen/Qwen3.8-Flash-Next", "filename": "TPU quantized recipe",
        "parameters": "180B footprint (125B core · ~6B active)", "quant": "TPU sub-4-bit",
        "context_length": 262144, "context_display": "262,144 tokens", "rope_scaling": "Native 262k",
        "kv_cache": "TPU paged cache", "vram_requirement": "TPU v5e-8 · 128 GB HBM",
        "speed": "Recipe-dependent", "accelerator": "tpu", "default_accelerator": "tpu",
        "supported_accelerators": ["tpu"], "hardware": "Kaggle TPU v5e-8",
        "hardware_detail": "8 chips · 128 GB aggregate HBM", "runtime": "TPU recipe",
        "kernel_workflow": "qwen38_flash_next_tpu", "native_anthropic": False,
        "max_concurrency": 1, "rank": 12, "launch_supported": False,
        "description": "TPU-only registry entry. Launch stays disabled until a verified v5e-8 quantized runner is bundled.",
        "server_args": [],
    },
}

MODEL_ALIASES = {
    "qwen3.8-27b-abliterated": "qwen3.8-27b-instruct",
    "qwen3.8-27b-tpu": "qwen3.8-27b-instruct",
    "qwen3.8-flash-next-180b": "qwen3.8-flash-next-tpu",
}


def normalize_model_id(model_id: str) -> str:
    return MODEL_ALIASES.get(str(model_id or ""), str(model_id or ""))


def resolve_gguf_filename(configured: str, siblings: Iterable[str]) -> str:
    """Resolve a single-file GGUF, preferring the configured file and quality tiers."""
    names = [str(name) for name in siblings if str(name).lower().endswith(".gguf")]
    if not names:
        raise ValueError("Repository contains no GGUF files")
    configured_lower = str(configured or "").lower()
    exact = next((name for name in names if name.lower() == configured_lower), None)
    if exact:
        return exact

    def score(name: str) -> tuple:
        lower = name.lower()
        excluded = any(token in lower for token in ("mmproj", "imatrix", "/bf16/", "mtp/", "mtp-"))
        single_file = not re.search(r"-\d{5}-of-\d{5}\.gguf$", lower)
        normalized = re.sub(r"[^a-z0-9]", "", lower)
        tiers = ("udq4kxl", "q4km", "q4k", "gguf")
        tier = next((len(tiers) - i for i, token in enumerate(tiers) if token in normalized), 0)
        return (not excluded, single_file, tier, -len(name))

    candidates = sorted(names, key=score, reverse=True)
    best = candidates[0]
    if not score(best)[0] or not score(best)[1]:
        raise ValueError("Repository has no suitable single-file model GGUF")
    return best


def model_for_accelerator(model_id: str, accelerator: str) -> Dict[str, Any]:
    model = dict(MODELS[normalize_model_id(model_id)])
    accelerator = str(accelerator or model.get("default_accelerator") or "gpu")
    if accelerator not in model.get("supported_accelerators", [model.get("accelerator", "gpu")]):
        raise ValueError(f"{model['name']} does not support {accelerator.upper()}")
    model["accelerator"] = accelerator
    if accelerator == "gpu":
        model["repo_id"] = model.get("gpu_repo_id", model.get("repo_id"))
        model["filename"] = model.get("gpu_filename", model.get("filename"))
        model["kernel_workflow"] = "gguf_gpu"
        model["hardware"] = "Dual Tesla T4"
        model["max_concurrency"] = int((model.get("max_concurrency_by_accelerator") or {}).get("gpu", model.get("max_concurrency", 2)))
    else:
        model["repo_id"] = model.get("tpu_repo_id", model.get("repo_id"))
        model["kernel_workflow"] = model.get("tpu_kernel_workflow", model.get("kernel_workflow"))
        model["dataset_sources"] = model.get("tpu_dataset_sources", model.get("dataset_sources", []))
        model["hardware"] = "Kaggle TPU v5e-8"
        model["max_concurrency"] = int((model.get("max_concurrency_by_accelerator") or {}).get("tpu", model.get("max_concurrency", 1)))
    return model


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
    return gpu_model(model_id, name, repo_id, filename, str(data.get("parameters") or "Custom GGUF"),
                     context_length, 99, "Depends on model", quant=str(data.get("quant") or "GGUF"),
                     filesize=float(data.get("filesize_gb") or 0), description=f"Custom GGUF: {repo_id}/{filename}") | {"custom": True}


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


MODELS.update(load_custom_models())
MODEL_REGISTRY = MODELS
AVAILABLE_MODELS = MODELS
DEFAULT_MODEL_ID = "cybertiel-coder-35b-a3b"
