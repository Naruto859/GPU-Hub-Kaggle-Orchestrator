#!/usr/bin/env python3
# ==============================================================================
# Autonomous Uncensored AI Agent Kernel - Dual Tesla T4 / P100 Execution Engine
# Model: CyberTiel-Coder-35B-A3B (Abliterated MoE) (35B MoE (~3.4B active), UD-Q4_K_XL (Cyber-imatrix))
# Context Window: 262,144 tokens (262k / ~2.75 lakh) | RoPE: YaRN scale 8 (orig 32k) | KV: q4_0 / q4_0
# Hugging Face: peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF / Cyber-Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf
# ==============================================================================

import os
import sys
import time
import subprocess
import json
import urllib.request
import re

print("=" * 70)
print("[ENGINE] Booting Autonomous Uncensored AI Engine on Kaggle...")
print("=" * 70)

MODEL_NAME = "CyberTiel-Coder-35B-A3B (Abliterated MoE)"
MODEL_REPO = "peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF"
MODEL_FILENAME = "Cyber-Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf"
MODEL_FILESIZE = 22.36
MODEL_ID = "cybertiel-coder-35b-a3b"
WEBHOOK_TARGET = "https://your-gpu-hub-callback.example/api/tunnel/register"

# --- Auto-shutdown lifecycle timers (minutes) ----------------------------------
# Injected at bundle-prep time from the dashboard settings (config.json), so
# editing "Auto-shutdown idle timeout" in the GPU Hub changes the real kernel
# behavior on the NEXT launch. 0 disables a timer.
IDLE_TIMEOUT_MIN = 160.0   # no inference request for this long -> graceful exit
MAX_RUNTIME_MIN = 330.0    # hard session cap even under continuous use
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

    # Try downloading official release b11146 prebuilt for Ubuntu CUDA 12.8
    tar_url = "https://github.com/ggml-org/llama.cpp/releases/download/b11146/llama-b11146-bin-ubuntu-cuda-12.8-x64.tar.gz"
    cudart_url = "https://github.com/ggml-org/llama.cpp/releases/download/b11146/cudart-llama-b11146-bin-ubuntu-cuda-12.8-x64.tar.gz"

    try:
        print("[LLAMA] Attempting direct binary download from ggml-org release b11146...")
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

# Step 3: Download Model GGUF
print(f"\n[STEP 3] Downloading GGUF model: {MODEL_REPO} / {MODEL_FILENAME}...")
from huggingface_hub import hf_hub_download

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
    "-np", "1", "--rope-scaling", "yarn", "--rope-scale", "8", "--yarn-orig-ctx", "32768", "-c", "262144", "-ctk", "q4_0", "-ctv", "q4_0", "-ngl", "52", "-sm", "layer", "-ts", "1,1", "-b", "2048", "-ub", "512", "-fa", "auto", "--threads", "4"
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
    f.write(base64.b64decode("""aW1wb3J0IGFzeW5jaW8sIHN5cywgdGltZQpmcm9tIGZhc3RhcGkgaW1wb3J0IEZhc3RBUEksIFJlcXVlc3QKZnJvbSBmYXN0YXBpLnJlc3BvbnNlcyBpbXBvcnQgU3RyZWFtaW5nUmVzcG9uc2UsIFJlc3BvbnNlCmltcG9ydCBodHRweCwgdXZpY29ybgoKYXBwID0gRmFzdEFQSSgpCmNsaWVudCA9IGh0dHB4LkFzeW5jQ2xpZW50KHRpbWVvdXQ9MzYwMC4wKQpVUFNUUkVBTSA9ICJodHRwOi8vMTI3LjAuMC4xOjgwODAiCgojIExhc3QtYWN0aXZpdHkgdHJhY2tpbmcgZm9yIGtlcm5lbCBpZGxlIGF1dG8tc2h1dGRvd246IHRoZSBrZXJuZWwgbG9vcCBwb2xscwojIHRoaXMgdGltZXN0YW1wIHRvIGRlY2lkZSB3aGV0aGVyIGFueW9uZSBpcyBhY3R1YWxseSB1c2luZyB0aGUgR1BVLiBVcGRhdGVkIG9uCiMgZXZlcnkgaW5mZXJlbmNlLXNoYXBlZCByZXF1ZXN0IChQT1NUIHRvIGEgL3YxLyogb3IgL2NoYXQvY29tcGxldGlvbnMgcGF0aCkuCl9sYXN0X2FjdGl2aXR5ID0gdGltZS50aW1lKCkKCmRlZiBfbm90ZV9hY3Rpdml0eSgpOgogICAgZ2xvYmFsIF9sYXN0X2FjdGl2aXR5CiAgICBfbGFzdF9hY3Rpdml0eSA9IHRpbWUudGltZSgpCgpAYXBwLmdldCgiL19fbGFzdF9hY3Rpdml0eSIpCmFzeW5jIGRlZiBnZXRfbGFzdF9hY3Rpdml0eSgpOgogICAgcmV0dXJuIHsibGFzdF9hY3Rpdml0eSI6IF9sYXN0X2FjdGl2aXR5fQoKQGFwcC5hcGlfcm91dGUoIi97cGF0aDpwYXRofSIsIG1ldGhvZHM9WyJHRVQiLCAiUE9TVCIsICJQVVQiLCAiREVMRVRFIiwgIkhFQUQiLCAiT1BUSU9OUyJdKQphc3luYyBkZWYgcm91dGVfYWxsKHJlcXVlc3Q6IFJlcXVlc3QsIHBhdGg6IHN0cik6CiAgICB0YXJnZXQgPSBmIntVUFNUUkVBTX0ve3BhdGh9IgogICAgaWYgcmVxdWVzdC5tZXRob2QgPT0gIlBPU1QiIGFuZCBwYXRoIGluICgidjEvY2hhdC9jb21wbGV0aW9ucyIsICJjaGF0L2NvbXBsZXRpb25zIik6CiAgICAgICAgX25vdGVfYWN0aXZpdHkoKQogICAgICAgIGJvZHkgPSBhd2FpdCByZXF1ZXN0Lmpzb24oKQogICAgICAgIGlzX3N0cmVhbSA9IGJvb2woYm9keS5nZXQoInN0cmVhbSIsIEZhbHNlKSkKICAgICAgICBpZiBpc19zdHJlYW06CiAgICAgICAgICAgIGFzeW5jIGRlZiBldmVudF9nZW5lcmF0b3IoKToKICAgICAgICAgICAgICAgIHlpZWxkIGIiOiBrZWVwYWxpdmVcblxuIgogICAgICAgICAgICAgICAgdXBfcmVxID0gY2xpZW50LmJ1aWxkX3JlcXVlc3QoIlBPU1QiLCB0YXJnZXQsIGpzb249Ym9keSwgaGVhZGVycz17IkNvbnRlbnQtVHlwZSI6ICJhcHBsaWNhdGlvbi9qc29uIn0pCiAgICAgICAgICAgICAgICByZXNwX2Z1dHVyZSA9IGFzeW5jaW8uY3JlYXRlX3Rhc2soY2xpZW50LnNlbmQodXBfcmVxLCBzdHJlYW09VHJ1ZSkpCiAgICAgICAgICAgICAgICB3aGlsZSBub3QgcmVzcF9mdXR1cmUuZG9uZSgpOgogICAgICAgICAgICAgICAgICAgIGRvbmUsIF8gPSBhd2FpdCBhc3luY2lvLndhaXQoW3Jlc3BfZnV0dXJlXSwgdGltZW91dD0xMC4wKQogICAgICAgICAgICAgICAgICAgIGlmIG5vdCBkb25lOgogICAgICAgICAgICAgICAgICAgICAgICB5aWVsZCBiIjoga2VlcGFsaXZlXG5cbiIKICAgICAgICAgICAgICAgIHJlc3AgPSBhd2FpdCByZXNwX2Z1dHVyZQogICAgICAgICAgICAgICAgdHJ5OgogICAgICAgICAgICAgICAgICAgIGFzeW5jIGZvciBjaHVuayBpbiByZXNwLmFpdGVyX3JhdygpOgogICAgICAgICAgICAgICAgICAgICAgICB5aWVsZCBjaHVuawogICAgICAgICAgICAgICAgZmluYWxseToKICAgICAgICAgICAgICAgICAgICBhd2FpdCByZXNwLmFjbG9zZSgpCiAgICAgICAgICAgIHJldHVybiBTdHJlYW1pbmdSZXNwb25zZSgKICAgICAgICAgICAgICAgIGV2ZW50X2dlbmVyYXRvcigpLAogICAgICAgICAgICAgICAgc3RhdHVzX2NvZGU9MjAwLAogICAgICAgICAgICAgICAgbWVkaWFfdHlwZT0idGV4dC9ldmVudC1zdHJlYW0iLAogICAgICAgICAgICAgICAgaGVhZGVycz17IkNhY2hlLUNvbnRyb2wiOiAibm8tY2FjaGUiLCAiQ29ubmVjdGlvbiI6ICJrZWVwLWFsaXZlIiwgIlgtQWNjZWwtQnVmZmVyaW5nIjogIm5vIn0KICAgICAgICAgICAgKQogICAgICAgIGVsc2U6CiAgICAgICAgICAgIHJlc3AgPSBhd2FpdCBjbGllbnQucG9zdCh0YXJnZXQsIGpzb249Ym9keSwgaGVhZGVycz17IkNvbnRlbnQtVHlwZSI6ICJhcHBsaWNhdGlvbi9qc29uIn0pCiAgICAgICAgICAgIHJldHVybiBSZXNwb25zZShjb250ZW50PXJlc3AuY29udGVudCwgc3RhdHVzX2NvZGU9cmVzcC5zdGF0dXNfY29kZSwgbWVkaWFfdHlwZT1yZXNwLmhlYWRlcnMuZ2V0KCJjb250ZW50LXR5cGUiLCAiYXBwbGljYXRpb24vanNvbiIpKQogICAgZWxzZToKICAgICAgICBjb250ZW50ID0gYXdhaXQgcmVxdWVzdC5ib2R5KCkKICAgICAgICBpZiByZXF1ZXN0Lm1ldGhvZCA9PSAiUE9TVCIgYW5kIChwYXRoLnN0YXJ0c3dpdGgoInYxLyIpIG9yIHBhdGguc3RhcnRzd2l0aCgiL3YxLyIpIG9yICJjb21wbGV0aW9ucyIgaW4gcGF0aC5sb3dlcigpKToKICAgICAgICAgICAgX25vdGVfYWN0aXZpdHkoKQogICAgICAgIHJlc3AgPSBhd2FpdCBjbGllbnQucmVxdWVzdChyZXF1ZXN0Lm1ldGhvZCwgdGFyZ2V0LCBjb250ZW50PWNvbnRlbnQsIGhlYWRlcnM9e2s6IHYgZm9yIGssIHYgaW4gcmVxdWVzdC5oZWFkZXJzLml0ZW1zKCkgaWYgay5sb3dlcigpICE9ICJob3N0In0pCiAgICAgICAgcmV0dXJuIFJlc3BvbnNlKGNvbnRlbnQ9cmVzcC5jb250ZW50LCBzdGF0dXNfY29kZT1yZXNwLnN0YXR1c19jb2RlLCBtZWRpYV90eXBlPXJlc3AuaGVhZGVycy5nZXQoImNvbnRlbnQtdHlwZSIsICJhcHBsaWNhdGlvbi9qc29uIikpCgppZiBfX25hbWVfXyA9PSAiX19tYWluX18iOgogICAgdXZpY29ybi5ydW4oYXBwLCBob3N0PSIxMjcuMC4wLjEiLCBwb3J0PTgwODEsIGxvZ19sZXZlbD0id2FybmluZyIpCg=="""))

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

# Step 5: Start Tunnel (Cloudflare Quick Tunnel)
print(f"\n[STEP 5] Setting up secure outbound tunnel to port {tunnel_target_port}...")
tunnel_url = ""

cf_bin = "/tmp/bin/cloudflared"
if not os.path.exists(cf_bin):
    print("[TUNNEL] Downloading Cloudflare tunnel binary...")
    subprocess.run(["curl", "-sL", "-o", cf_bin, "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"], check=True)
    os.chmod(cf_bin, 0o755)

cf_log_path = "/tmp/cloudflared.log"
cf_log = open(cf_log_path, "w")
print(f"[TUNNEL] Starting Cloudflare Quick Tunnel to http://127.0.0.1:{tunnel_target_port}...")
cf_proc = subprocess.Popen([cf_bin, "tunnel", "--url", f"http://127.0.0.1:{tunnel_target_port}", "--no-autoupdate"], stdout=cf_log, stderr=subprocess.STDOUT)

# Extract TryCloudflare URL
for _ in range(60):
    time.sleep(1)
    if os.path.exists(cf_log_path):
        with open(cf_log_path, "r", errors="ignore") as f:
            content = f.read()
            match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", content)
            if match:
                tunnel_url = match.group(0)
                print(f"\n[TUNNEL ESTABLISHED] URL: {tunnel_url}")
                break

if not tunnel_url:
    print("[ERROR] Could not extract tunnel URL from Cloudflare logs. Log contents:")
    with open(cf_log_path) as f:
        print(f.read())
    sys.exit(1)

# Step 6: Notify VPS Webhook
print("\n[STEP 6] Reporting tunnel URL to VPS Manager...")
try:
    payload = json.dumps({"tunnel_url": tunnel_url, "model_id": MODEL_ID, "status": "ready"}).encode("utf-8")
    req = urllib.request.Request(
        WEBHOOK_TARGET,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "KaggleKernel/1.0"}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        print(f"[WEBHOOK] Successfully notified VPS manager (HTTP {resp.status}): {resp.read().decode()}")
except Exception as e:
    print(f"[WEBHOOK] Note: Could not auto-post to webhook at {WEBHOOK_TARGET} ({e}).")
    print(f"Tunnel URL is printed below for manual connection if needed.")

print("\n" + "=" * 70)
print(f"=== TUNNEL_URL: {tunnel_url} ===")
print("=" * 70 + "\n")

# Step 7: Continuous Keepalive Loop with Auto-Shutdown Lifecycle
print("[STEP 7] Entering keepalive loop with auto-shutdown lifecycle.")
print(f"[LIFECYCLE] IDLE_TIMEOUT_MIN={IDLE_TIMEOUT_MIN} | MAX_RUNTIME_MIN={MAX_RUNTIME_MIN}")
if IDLE_TIMEOUT_MIN and IDLE_TIMEOUT_MIN > 0:
    print(f"[LIFECYCLE] Idle auto-shutdown: {IDLE_TIMEOUT_MIN} min without an inference request.")
if MAX_RUNTIME_MIN and MAX_RUNTIME_MIN > 0:
    print(f"[LIFECYCLE] Max runtime cap: {MAX_RUNTIME_MIN} min regardless of activity.")

SESSION_START = time.time()

def read_last_activity():
    """Last inference request time reported by the early-ping bridge.
    Returns None when the bridge is unavailable (fallback -> treat as active)."""
    try:
        with urllib.request.urlopen(LAST_ACTIVITY_URL, timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return float(data.get("last_activity", time.time()))
    except Exception:
        return None

def post_shutdown_status(reason, detail):
    """Report the auto-shutdown to the VPS manager so the dashboard resets."""
    try:
        payload = json.dumps({
            "tunnel_url": "",
            "model_id": MODEL_ID,
            "status": "auto_shutdown",
            "reason": reason,
            "detail": detail,
        }).encode("utf-8")
        req = urllib.request.Request(
            WEBHOOK_TARGET,
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "KaggleKernel/1.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"[WEBHOOK] Shutdown status reported (HTTP {resp.status}).")
    except Exception as e:
        print(f"[WEBHOOK] Note: shutdown status post failed ({e}).")

def graceful_shutdown(reason, detail):
    """Kill children, notify the manager, and END the Kaggle GPU session."""
    print(f"[AUTO-SHUTDOWN] {reason}: {detail}")
    print("[AUTO-SHUTDOWN] Killing llama-server, cloudflared and bridge, then exiting.")
    post_shutdown_status(reason, detail)
    for proc, name in ((server_proc, "llama-server"), (cf_proc, "cloudflared"), (bridge_proc, "early_ping_bridge")):
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception as e:
            print(f"[AUTO-SHUTDOWN] terminate {name} failed: {e}")
    time.sleep(2)
    for proc, name in ((server_proc, "llama-server"), (cf_proc, "cloudflared"), (bridge_proc, "early_ping_bridge")):
        try:
            if proc.poll() is None:
                proc.kill()
        except Exception as e:
            print(f"[AUTO-SHUTDOWN] kill {name} failed: {e}")
    print("[AUTO-SHUTDOWN] GPU released. Kaggle session ending.")
    sys.exit(0)

tick = 0
while True:
    time.sleep(30)
    tick += 1
    now = time.time()

    # --- Auto-shutdown: hard max-runtime cap (checked first) ---
    if MAX_RUNTIME_MIN and MAX_RUNTIME_MIN > 0:
        runtime_min = (now - SESSION_START) / 60.0
        if runtime_min >= MAX_RUNTIME_MIN:
            graceful_shutdown("max_runtime", f"max runtime reached after {runtime_min:.1f} min")

    # --- Auto-shutdown: idle timeout (skipped while a request may be in flight) ---
    if IDLE_TIMEOUT_MIN and IDLE_TIMEOUT_MIN > 0 and bridge_ready:
        last_activity = read_last_activity()
        if last_activity is not None:
            idle_min = (now - last_activity) / 60.0
            if idle_min >= IDLE_TIMEOUT_MIN:
                graceful_shutdown("idle", f"idle {idle_min:.1f} min, releasing GPU")
            elif idle_min > (IDLE_TIMEOUT_MIN / 2.0):
                print(f"[LIFECYCLE] Idle for {idle_min:.1f} min of {IDLE_TIMEOUT_MIN} min threshold.")

    # Check llama-server process health
    if server_proc.poll() is not None:
        print("[CRITICAL] llama-server process exited unexpectedly!")
        with open("/tmp/llama_server.log") as f:
            print(f.read()[-2000:])
        break

    # Check tunnel health
    if cf_proc.poll() is not None:
        print("[CRITICAL] cloudflared process exited unexpectedly!")
        break

    # Check bridge health
    if bridge_ready and bridge_proc.poll() is not None:
        print("[CRITICAL] early_ping_bridge exited unexpectedly!")
        with open("/tmp/bridge.log") as f:
            print(f.read()[-2000:])
        break

    if tick % 4 == 0:
        print(f"[HEARTBEAT] Elapsed: {tick * 30}s | Upstream {tunnel_url} is healthy and serving.")
