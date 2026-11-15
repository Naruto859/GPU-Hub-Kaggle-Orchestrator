#!/usr/bin/env python3
# ==============================================================================
# Autonomous Uncensored AI Agent Kernel - Dual Tesla T4 / P100 Execution Engine
# Model: Cyber-Tiel-Coder-35B-A3B (Uncensored Security & Coding MoE) (35B MoE (~3.4B active), UD-Q4_K_XL)
# Context Window: 262,144 tokens | RoPE: Native / model-defined long context | KV: q4_0 / q4_0
# Hugging Face: peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF / Cyber-Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf
# ==============================================================================

import os
import sys
import time
import subprocess
import json
import urllib.request
import urllib.parse
import re
from pathlib import Path

print("=" * 70)
print("[ENGINE] Booting Autonomous Uncensored AI Engine on Kaggle...")
print("=" * 70)

MODEL_NAME = "Cyber-Tiel-Coder-35B-A3B (Uncensored Security & Coding MoE)"
MODEL_REPO = "peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF"
MODEL_FILENAME = "Cyber-Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf"
MODEL_FILESIZE = 22.36
MODEL_ID = "cybertiel-coder-35b-a3b"
WEBHOOK_TARGET = "https://your-gpu-hub-callback.example/api/tunnel/register"
GPU_HUB_SESSION_ID = ""
GPU_HUB_SESSION_ROLE = "active"

# --- Auto-shutdown lifecycle timers (minutes) ----------------------------------
# Injected at bundle-prep time from the dashboard settings (config.json), so
# editing "Auto-shutdown idle timeout" in the GPU Hub changes the real kernel
# behavior on the NEXT launch. 0 disables a timer.
IDLE_TIMEOUT_MIN = 20.0   # no inference request for this long -> graceful exit
MAX_RUNTIME_MIN = 540.0    # hard session cap even under continuous use
LAST_ACTIVITY_URL = "http://127.0.0.1:8081/__last_activity"

# Check GPU hardware
print("[GPU CHECK] Inspecting CUDA accelerators...")
try:
    smi = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT).decode()
    print(smi)
except Exception as e:
    print("[WARN] nvidia-smi failed:", e)

WORK = "/kaggle/working"
MODEL_DIR = "/tmp/model"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("/tmp/bin", exist_ok=True)

# Step 1: Install Python prerequisites
print("\n[STEP 1] Installing huggingface_hub, requests, and bridge dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub", "requests", "fastapi", "uvicorn", "httpx"], check=False)

# Step 2: Acquire llama.cpp binary (Prebuilt CUDA or fast source build)
LLAMA_SERVER = "/tmp/bin/llama-server"

def setup_llama_server():
    global LLAMA_SERVER
    print("\n[STEP 2] Setting up llama-server with GPU support...")

    # Current CUDA build with Qwen3.6/Qwen3.8 Gated DeltaNet and Gemma 4 support.
    llama_build = "b12395"
    tar_url = f"https://github.com/ggml-org/llama.cpp/releases/download/{llama_build}/llama-{llama_build}-bin-ubuntu-cuda-12.8-x64.tar.gz"
    cudart_url = f"https://github.com/ggml-org/llama.cpp/releases/download/{llama_build}/cudart-llama-{llama_build}-bin-ubuntu-cuda-12.8-x64.tar.gz"

    try:
        print(f"[LLAMA] Attempting direct binary download from ggml-org release {llama_build}...")
        subprocess.run(["curl", "-sL", "-o", "/tmp/llama.tar.gz", tar_url], check=True)
        subprocess.run(["curl", "-sL", "-o", "/tmp/cudart.tar.gz", cudart_url], check=True)

        subprocess.run(["tar", "-xzf", "/tmp/llama.tar.gz", "-C", "/tmp/bin"], check=True)
        subprocess.run(["tar", "-xzf", "/tmp/cudart.tar.gz", "-C", "/tmp/bin"], check=True)

        # Check if llama-server exists
        for root, dirs, files in os.walk("/tmp/bin"):
            if "llama-server" in files:
                LLAMA_SERVER = os.path.join(root, "llama-server")
                os.chmod(LLAMA_SERVER, 0o755)
                print(f"[LLAMA] Successfully installed pre-built llama-server at: {LLAMA_SERVER}")
                return True
    except Exception as e:
        print("[LLAMA] Direct prebuilt download encountered issue:", e)
        print("[LLAMA] Falling back to building llama.cpp from source with CUDA...")

    # Fallback: Git clone & CMake build
    os.chdir(WORK)
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git"], check=True)
    build_dir = os.path.join(WORK, "llama.cpp", "build")
    os.makedirs(build_dir, exist_ok=True)
    os.chdir(os.path.join(WORK, "llama.cpp"))

    # Configure with CUDA for Tesla T4 (arch 75)
    subprocess.run([
        "cmake", "-B", "build",
        "-DGGML_CUDA=ON",
        "-DCMAKE_CUDA_ARCHITECTURES=75",
        "-DCMAKE_BUILD_TYPE=Release"
    ], check=True)
    subprocess.run(["cmake", "--build", "build", "--config", "Release", "-j4", "--target", "llama-server"], check=True)

    for candidate in [
        os.path.join(build_dir, "bin", "llama-server"),
        os.path.join(build_dir, "llama-server")
    ]:
        if os.path.exists(candidate):
            LLAMA_SERVER = candidate
            os.chmod(LLAMA_SERVER, 0o755)
            print(f"[LLAMA] Compiled llama-server at: {LLAMA_SERVER}")
            return True

    raise RuntimeError("Could not establish llama-server binary!")

setup_llama_server()

# Step 3: Resolve and download Model GGUF
print(f"\n[STEP 3] Downloading GGUF model: {MODEL_REPO} / {MODEL_FILENAME}...")
from huggingface_hub import hf_hub_download

def resolve_gguf_filename(repo_id, configured):
    url = "https://huggingface.co/api/models/" + urllib.parse.quote(repo_id, safe="/")
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        names = [str(item.get("rfilename") or "") for item in payload.get("siblings", [])]
        names = [name for name in names if name.lower().endswith(".gguf")]
        exact = next((name for name in names if name.lower() == configured.lower()), None)
        if exact:
            return exact
        def score(name):
            lower = name.lower()
            excluded = any(token in lower for token in ("mmproj", "imatrix", "/bf16/", "mtp/", "mtp-"))
            single = re.search(r"-\d{5}-of-\d{5}\.gguf$", lower) is None
            normalized = re.sub(r"[^a-z0-9]", "", lower)
            tiers = ("udq4kxl", "q4km", "q4k", "gguf")
            tier = next((len(tiers) - index for index, token in enumerate(tiers) if token in normalized), 0)
            return (not excluded, single, tier, -len(name))
        candidates = sorted(names, key=score, reverse=True)
        if candidates and score(candidates[0])[0] and score(candidates[0])[1]:
            chosen = candidates[0]
            print(f"[MODEL RESOLVER] Configured file unavailable; selected {chosen}")
            return chosen
    except Exception as exc:
        print(f"[MODEL RESOLVER] API lookup failed: {exc}; trying configured filename")
    return configured

MODEL_FILENAME = resolve_gguf_filename(MODEL_REPO, MODEL_FILENAME)

model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
if not os.path.exists(model_path):
    print(f"[MODEL] Fetching {MODEL_FILENAME} (~{MODEL_FILESIZE} GB)...")
    downloaded_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
        local_dir=MODEL_DIR
    )
    print(f"[MODEL] Model downloaded successfully to: {downloaded_path}")
else:
    print(f"[MODEL] Found cached model at: {model_path}")

# Step 4: Launch llama-server with Dual-GPU offload
print("\n[STEP 4] Launching llama-server in background with GPU offloading...")
server_cmd = [
    LLAMA_SERVER,
    "-m", model_path,
    "--host", "127.0.0.1", "--port", "8080", "--parallel", "2", "-c", "262144", "--cache-type-k", "q4_0", "--cache-type-v", "q4_0", "-ngl", "99", "-sm", "layer", "-ts", "1,1", "-b", "2048", "-ub", "512", "-fa", "on", "--threads", "4"
]

print("[SERVER CMD]:", " ".join(server_cmd))
server_log = open("/tmp/llama_server.log", "w")
server_proc = subprocess.Popen(server_cmd, stdout=server_log, stderr=subprocess.STDOUT)
print(f"[SERVER] Process spawned with PID: {server_proc.pid}")

# Poll health endpoint
print("[SERVER] Waiting for model to load into VRAM...")
ready = False
for attempt in range(120):
    time.sleep(3)
    try:
        req = urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=2)
        if req.status == 200:
            print(f"\n[SERVER READY] Model is fully loaded and accepting requests on port 8080! (attempt {attempt+1})")
            ready = True
            break
    except Exception:
        sys.stdout.write(".")
        sys.stdout.flush()

if not ready:
    print("\n[ERROR] Server failed to become ready within timeout. Tail log:")
    server_log.flush()
    with open("/tmp/llama_server.log") as f:
        print(f.read()[-3000:])
    sys.exit(1)

# Step 4b: Launch Early-Ping Keepalive Bridge on port 8081
print("\n[STEP 4b] Starting Early-Ping Keepalive Bridge (Port 8081 -> 8080)...")
import base64
with open("/tmp/early_ping_bridge.py", "wb") as f:
    f.write(base64.b64decode("""aW1wb3J0IGFzeW5jaW8sIHN5cywgdGltZQpmcm9tIGZhc3RhcGkgaW1wb3J0IEZhc3RBUEksIFJlcXVlc3QKZnJvbSBmYXN0YXBpLnJlc3BvbnNlcyBpbXBvcnQgU3RyZWFtaW5nUmVzcG9uc2UsIFJlc3BvbnNlCmltcG9ydCBodHRweCwgdXZpY29ybgoKYXBwID0gRmFzdEFQSSgpCmNsaWVudCA9IGh0dHB4LkFzeW5jQ2xpZW50KHRpbWVvdXQ9MzYwMC4wKQpVUFNUUkVBTSA9ICJodHRwOi8vMTI3LjAuMC4xOjgwODAiCgojIExhc3QtYWN0aXZpdHkgdHJhY2tpbmcgZm9yIGtlcm5lbCBpZGxlIGF1dG8tc2h1dGRvd246IHRoZSBrZXJuZWwgbG9vcCBwb2xscwojIHRoaXMgdGltZXN0YW1wIHRvIGRlY2lkZSB3aGV0aGVyIGFueW9uZSBpcyBhY3R1YWxseSB1c2luZyB0aGUgR1BVLiBVcGRhdGVkIG9uCiMgZXZlcnkgaW5mZXJlbmNlLXNoYXBlZCByZXF1ZXN0IChQT1NUIHRvIGEgL3YxLyogb3IgL2NoYXQvY29tcGxldGlvbnMgcGF0aCkuCl9sYXN0X2FjdGl2aXR5ID0gdGltZS50aW1lKCkKX2FjdGl2ZV9yZXF1ZXN0cyA9IDAKCmRlZiBfbm90ZV9hY3Rpdml0eSgpOgogICAgZ2xvYmFsIF9sYXN0X2FjdGl2aXR5CiAgICBfbGFzdF9hY3Rpdml0eSA9IHRpbWUudGltZSgpCgpkZWYgX3JlcXVlc3Rfc3RhcnRlZCgpOgogICAgZ2xvYmFsIF9hY3RpdmVfcmVxdWVzdHMKICAgIF9hY3RpdmVfcmVxdWVzdHMgKz0gMQogICAgX25vdGVfYWN0aXZpdHkoKQoKZGVmIF9yZXF1ZXN0X2ZpbmlzaGVkKCk6CiAgICBnbG9iYWwgX2FjdGl2ZV9yZXF1ZXN0cwogICAgX2FjdGl2ZV9yZXF1ZXN0cyA9IG1heCgwLCBfYWN0aXZlX3JlcXVlc3RzIC0gMSkKICAgIF9ub3RlX2FjdGl2aXR5KCkKCkBhcHAuZ2V0KCIvX19sYXN0X2FjdGl2aXR5IikKYXN5bmMgZGVmIGdldF9sYXN0X2FjdGl2aXR5KCk6CiAgICByZXR1cm4geyJsYXN0X2FjdGl2aXR5IjogX2xhc3RfYWN0aXZpdHksICJhY3RpdmVfcmVxdWVzdHMiOiBfYWN0aXZlX3JlcXVlc3RzfQoKQGFwcC5hcGlfcm91dGUoIi97cGF0aDpwYXRofSIsIG1ldGhvZHM9WyJHRVQiLCAiUE9TVCIsICJQVVQiLCAiREVMRVRFIiwgIkhFQUQiLCAiT1BUSU9OUyJdKQphc3luYyBkZWYgcm91dGVfYWxsKHJlcXVlc3Q6IFJlcXVlc3QsIHBhdGg6IHN0cik6CiAgICB0YXJnZXQgPSBmIntVUFNUUkVBTX0ve3BhdGh9IgogICAgaWYgcmVxdWVzdC5tZXRob2QgPT0gIlBPU1QiIGFuZCBwYXRoIGluICgidjEvY2hhdC9jb21wbGV0aW9ucyIsICJjaGF0L2NvbXBsZXRpb25zIik6CiAgICAgICAgYm9keSA9IGF3YWl0IHJlcXVlc3QuanNvbigpCiAgICAgICAgX3JlcXVlc3Rfc3RhcnRlZCgpCiAgICAgICAgaXNfc3RyZWFtID0gYm9vbChib2R5LmdldCgic3RyZWFtIiwgRmFsc2UpKQogICAgICAgIGlmIGlzX3N0cmVhbToKICAgICAgICAgICAgYXN5bmMgZGVmIGV2ZW50X2dlbmVyYXRvcigpOgogICAgICAgICAgICAgICAgdHJ5OgogICAgICAgICAgICAgICAgICAgIHlpZWxkIGIiOiBrZWVwYWxpdmVcblxuIgogICAgICAgICAgICAgICAgICAgIHVwX3JlcSA9IGNsaWVudC5idWlsZF9yZXF1ZXN0KCJQT1NUIiwgdGFyZ2V0LCBqc29uPWJvZHksIGhlYWRlcnM9eyJDb250ZW50LVR5cGUiOiAiYXBwbGljYXRpb24vanNvbiJ9KQogICAgICAgICAgICAgICAgICAgIHJlc3BfZnV0dXJlID0gYXN5bmNpby5jcmVhdGVfdGFzayhjbGllbnQuc2VuZCh1cF9yZXEsIHN0cmVhbT1UcnVlKSkKICAgICAgICAgICAgICAgICAgICB3aGlsZSBub3QgcmVzcF9mdXR1cmUuZG9uZSgpOgogICAgICAgICAgICAgICAgICAgICAgICBkb25lLCBfID0gYXdhaXQgYXN5bmNpby53YWl0KFtyZXNwX2Z1dHVyZV0sIHRpbWVvdXQ9MTAuMCkKICAgICAgICAgICAgICAgICAgICAgICAgaWYgbm90IGRvbmU6CiAgICAgICAgICAgICAgICAgICAgICAgICAgICB5aWVsZCBiIjoga2VlcGFsaXZlXG5cbiIKICAgICAgICAgICAgICAgICAgICByZXNwID0gYXdhaXQgcmVzcF9mdXR1cmUKICAgICAgICAgICAgICAgICAgICB0cnk6CiAgICAgICAgICAgICAgICAgICAgICAgIGFzeW5jIGZvciBjaHVuayBpbiByZXNwLmFpdGVyX3JhdygpOgogICAgICAgICAgICAgICAgICAgICAgICAgICAgeWllbGQgY2h1bmsKICAgICAgICAgICAgICAgICAgICBmaW5hbGx5OgogICAgICAgICAgICAgICAgICAgICAgICBhd2FpdCByZXNwLmFjbG9zZSgpCiAgICAgICAgICAgICAgICBmaW5hbGx5OgogICAgICAgICAgICAgICAgICAgIF9yZXF1ZXN0X2ZpbmlzaGVkKCkKICAgICAgICAgICAgcmV0dXJuIFN0cmVhbWluZ1Jlc3BvbnNlKAogICAgICAgICAgICAgICAgZXZlbnRfZ2VuZXJhdG9yKCksCiAgICAgICAgICAgICAgICBzdGF0dXNfY29kZT0yMDAsCiAgICAgICAgICAgICAgICBtZWRpYV90eXBlPSJ0ZXh0L2V2ZW50LXN0cmVhbSIsCiAgICAgICAgICAgICAgICBoZWFkZXJzPXsiQ2FjaGUtQ29udHJvbCI6ICJuby1jYWNoZSIsICJDb25uZWN0aW9uIjogImtlZXAtYWxpdmUiLCAiWC1BY2NlbC1CdWZmZXJpbmciOiAibm8ifQogICAgICAgICAgICApCiAgICAgICAgZWxzZToKICAgICAgICAgICAgdHJ5OgogICAgICAgICAgICAgICAgcmVzcCA9IGF3YWl0IGNsaWVudC5wb3N0KHRhcmdldCwganNvbj1ib2R5LCBoZWFkZXJzPXsiQ29udGVudC1UeXBlIjogImFwcGxpY2F0aW9uL2pzb24ifSkKICAgICAgICAgICAgICAgIHJldHVybiBSZXNwb25zZShjb250ZW50PXJlc3AuY29udGVudCwgc3RhdHVzX2NvZGU9cmVzcC5zdGF0dXNfY29kZSwgbWVkaWFfdHlwZT1yZXNwLmhlYWRlcnMuZ2V0KCJjb250ZW50LXR5cGUiLCAiYXBwbGljYXRpb24vanNvbiIpKQogICAgICAgICAgICBmaW5hbGx5OgogICAgICAgICAgICAgICAgX3JlcXVlc3RfZmluaXNoZWQoKQogICAgZWxzZToKICAgICAgICBjb250ZW50ID0gYXdhaXQgcmVxdWVzdC5ib2R5KCkKICAgICAgICB0cmFja2VkID0gcmVxdWVzdC5tZXRob2QgPT0gIlBPU1QiIGFuZCAocGF0aC5zdGFydHN3aXRoKCJ2MS8iKSBvciBwYXRoLnN0YXJ0c3dpdGgoIi92MS8iKSBvciAiY29tcGxldGlvbnMiIGluIHBhdGgubG93ZXIoKSkKICAgICAgICBpZiB0cmFja2VkOgogICAgICAgICAgICBfcmVxdWVzdF9zdGFydGVkKCkKICAgICAgICB0cnk6CiAgICAgICAgICAgIHJlc3AgPSBhd2FpdCBjbGllbnQucmVxdWVzdChyZXF1ZXN0Lm1ldGhvZCwgdGFyZ2V0LCBjb250ZW50PWNvbnRlbnQsIGhlYWRlcnM9e2s6IHYgZm9yIGssIHYgaW4gcmVxdWVzdC5oZWFkZXJzLml0ZW1zKCkgaWYgay5sb3dlcigpICE9ICJob3N0In0pCiAgICAgICAgICAgIHJldHVybiBSZXNwb25zZShjb250ZW50PXJlc3AuY29udGVudCwgc3RhdHVzX2NvZGU9cmVzcC5zdGF0dXNfY29kZSwgbWVkaWFfdHlwZT1yZXNwLmhlYWRlcnMuZ2V0KCJjb250ZW50LXR5cGUiLCAiYXBwbGljYXRpb24vanNvbiIpKQogICAgICAgIGZpbmFsbHk6CiAgICAgICAgICAgIGlmIHRyYWNrZWQ6CiAgICAgICAgICAgICAgICBfcmVxdWVzdF9maW5pc2hlZCgpCgppZiBfX25hbWVfXyA9PSAiX19tYWluX18iOgogICAgdXZpY29ybi5ydW4oYXBwLCBob3N0PSIxMjcuMC4wLjEiLCBwb3J0PTgwODEsIGxvZ19sZXZlbD0id2FybmluZyIpCg=="""))

bridge_log = open("/tmp/bridge.log", "w")
bridge_proc = subprocess.Popen([sys.executable, "/tmp/early_ping_bridge.py"], stdout=bridge_log, stderr=subprocess.STDOUT)
print(f"[BRIDGE] Early-Ping bridge spawned with PID: {bridge_proc.pid}")

bridge_ready = False
for b_att in range(30):
    time.sleep(0.5)
    try:
        b_chk = urllib.request.urlopen("http://127.0.0.1:8081/health", timeout=2)
        if b_chk.status == 200:
            print(f"[BRIDGE READY] Early-Ping bridge listening and verified on port 8081! (attempt {b_att+1})")
            bridge_ready = True
            break
    except Exception:
        pass

tunnel_target_port = 8081 if bridge_ready else 8080
if not bridge_ready:
    print("[WARN] Early-Ping bridge did not respond, falling back directly to port 8080.")

# Step 5: Start Tunnel and self-healing supervisor
print(f"\n[STEP 5] Setting up secure outbound tunnel to port {tunnel_target_port}...")
cf_bin = "/tmp/bin/cloudflared"
if not os.path.exists(cf_bin):
    print("[TUNNEL] Downloading Cloudflare tunnel binary...")
    subprocess.run(["curl", "-sL", "-o", cf_bin, "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"], check=True)
    os.chmod(cf_bin, 0o755)

cf_log_path = "/tmp/cloudflared.log"
cf_proc = None
cf_log = None
tunnel_url = ""


def notify_tunnel(url, status="ready"):
    try:
        payload = json.dumps({"tunnel_url": url, "model_id": MODEL_ID, "status": status,
                              "session_id": GPU_HUB_SESSION_ID, "session_role": GPU_HUB_SESSION_ROLE}).encode("utf-8")
        req = urllib.request.Request(WEBHOOK_TARGET, data=payload,
                                     headers={"Content-Type": "application/json", "User-Agent": "KaggleKernel/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"[WEBHOOK] Registered tunnel {url} (HTTP {resp.status}).")
    except Exception as exc:
        print(f"[WEBHOOK] Registration failed for {url}: {exc}")


def spawn_cloudflare():
    global cf_proc, cf_log, tunnel_url
    if cf_proc is not None and cf_proc.poll() is None:
        cf_proc.terminate()
        try: cf_proc.wait(timeout=5)
        except Exception: cf_proc.kill()
    try:
        if cf_log: cf_log.close()
    except Exception:
        pass
    Path(cf_log_path).write_text("")
    cf_log = open(cf_log_path, "a")
    cf_proc = subprocess.Popen([
        cf_bin, "tunnel", "--url", f"http://127.0.0.1:{tunnel_target_port}",
        "--no-autoupdate", "--protocol", "http2", "--edge-ip-version", "4",
    ],
                               stdout=cf_log, stderr=subprocess.STDOUT)
    discovered = ""
    deadline = time.time() + 90
    while time.time() < deadline and cf_proc.poll() is None:
        time.sleep(1)
        content = Path(cf_log_path).read_text(errors="ignore") if Path(cf_log_path).exists() else ""
        match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", content)
        if match:
            discovered = match.group(0)
            break
    if discovered:
        tunnel_url = discovered
        print(f"[TUNNEL ESTABLISHED] URL: {tunnel_url}")
        print(f"=== TUNNEL_URL: {tunnel_url} ===")
        notify_tunnel(tunnel_url)
        return True
    print("[TUNNEL] cloudflared did not produce a URL; supervisor will retry.")
    return False


def local_server_healthy():
    try:
        return urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=3).status == 200
    except Exception:
        return False


def public_tunnel_healthy():
    if not tunnel_url:
        return False
    try:
        return urllib.request.urlopen(tunnel_url + "/health", timeout=5).status == 200
    except Exception:
        return False


def restart_llama_server():
    global server_proc, server_log
    print("[SUPERVISOR] llama-server unhealthy twice; restarting in place.")
    try:
        if server_proc.poll() is None:
            server_proc.terminate()
            try: server_proc.wait(timeout=10)
            except Exception: server_proc.kill()
    except Exception:
        pass
    try: server_log.close()
    except Exception: pass
    server_log = open("/tmp/llama_server.log", "a")
    server_proc = subprocess.Popen(server_cmd, stdout=server_log, stderr=subprocess.STDOUT)
    for _ in range(180):
        if local_server_healthy():
            print("[SUPERVISOR] llama-server recovered on port 8080.")
            return True
        if server_proc.poll() is not None:
            break
        time.sleep(2)
    print("[SUPERVISOR] llama-server restart has not become healthy yet; will retry.")
    return False


if not spawn_cloudflare():
    time.sleep(5)
    spawn_cloudflare()

print("[STEP 6] Tunnel registered; entering 10-second self-healing supervisor.")
print(f"[LIFECYCLE] IDLE_TIMEOUT_MIN={IDLE_TIMEOUT_MIN} | MAX_RUNTIME_MIN={MAX_RUNTIME_MIN}")
SESSION_START = time.time()


def read_last_activity():
    try:
        with urllib.request.urlopen(LAST_ACTIVITY_URL, timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return float(data.get("last_activity", time.time())), int(data.get("active_requests", 0))
    except Exception:
        return None


def post_shutdown_status(reason, detail):
    try:
        payload = json.dumps({"tunnel_url": "", "model_id": MODEL_ID, "status": "auto_shutdown",
                              "reason": reason, "detail": detail, "session_id": GPU_HUB_SESSION_ID,
                              "session_role": GPU_HUB_SESSION_ROLE}).encode("utf-8")
        req = urllib.request.Request(WEBHOOK_TARGET, data=payload,
                                     headers={"Content-Type": "application/json", "User-Agent": "KaggleKernel/1.0"})
        urllib.request.urlopen(req, timeout=10).read()
    except Exception as exc:
        print(f"[WEBHOOK] Shutdown status failed: {exc}")


def graceful_shutdown(reason, detail):
    print(f"[AUTO-SHUTDOWN] {reason}: {detail}")
    post_shutdown_status(reason, detail)
    for proc in (server_proc, cf_proc, bridge_proc):
        try:
            if proc and proc.poll() is None: proc.terminate()
        except Exception: pass
    time.sleep(2)
    sys.exit(0)


local_failures = 0
tunnel_failures = 0
tick = 0
while True:
    time.sleep(10)
    tick += 1
    now = time.time()
    if MAX_RUNTIME_MIN and MAX_RUNTIME_MIN > 0 and (now - SESSION_START) / 60.0 >= MAX_RUNTIME_MIN:
        graceful_shutdown("max_runtime", f"max runtime reached after {(now - SESSION_START) / 60.0:.1f} min")

    if IDLE_TIMEOUT_MIN and IDLE_TIMEOUT_MIN > 0 and bridge_ready:
        activity = read_last_activity()
        if activity is not None:
            last_activity, active_requests = activity
            idle_min = (now - last_activity) / 60.0
            if active_requests == 0 and idle_min >= IDLE_TIMEOUT_MIN:
                graceful_shutdown("idle", f"idle {idle_min:.1f} min, releasing accelerator")

    local_failures = 0 if local_server_healthy() else local_failures + 1
    if local_failures >= 2:
        restart_llama_server()
        local_failures = 0

    tunnel_ok = cf_proc is not None and cf_proc.poll() is None and public_tunnel_healthy()
    tunnel_failures = 0 if tunnel_ok else tunnel_failures + 1
    if tunnel_failures >= 2:
        print("[SUPERVISOR] Cloudflare tunnel unhealthy twice; rotating tunnel.")
        spawn_cloudflare()
        tunnel_failures = 0

    if bridge_ready and bridge_proc.poll() is not None:
        print("[SUPERVISOR] Bridge exited; restarting it.")
        bridge_log = open("/tmp/bridge.log", "a")
        bridge_proc = subprocess.Popen([sys.executable, "/tmp/early_ping_bridge.py"], stdout=bridge_log, stderr=subprocess.STDOUT)

    if tick % 6 == 0:
        print(f"[HEARTBEAT] Elapsed: {tick * 10}s | local={local_server_healthy()} tunnel={public_tunnel_healthy()} url={tunnel_url}")
