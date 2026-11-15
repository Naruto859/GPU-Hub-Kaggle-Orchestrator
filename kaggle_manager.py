import base64
import csv
import io
import json
import logging
import os
import re
import subprocess
import tarfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from urllib.parse import urlsplit

import requests

from models_registry import MODELS, DEFAULT_MODEL_ID, model_for_accelerator, normalize_model_id
from account_pool import KaggleAccountPool, QUOTA_SECONDS, SESSION_LIMIT_SECONDS
from webshare_manager import ProxyRouteError, WebshareConnector

# --- Auto-shutdown lifecycle defaults (minutes) -------------------------------
# IDLE_TIMEOUT_MIN: no inference request for this long -> kernel exits, Kaggle
#   releases the GPU batch-session slot. 0 disables the idle timer.
# MAX_RUNTIME_MIN: hard session cap regardless of activity, well under Kaggle's
#   12h hard-kill, to protect the 30h/week-across-accounts GPU budget.
#   0 disables the cap. Both are dashboard-editable and injected into the
#   kernel bundle at prepare_bundle time (config.json is the persistence layer).
IDLE_TIMEOUT_MIN_DEFAULT = 20.0
MAX_RUNTIME_MIN_DEFAULT = 540.0
PREWARM_LEAD_SECONDS = 15 * 60
MAX_BOOT_SECONDS = 15 * 60
BOOT_BASELINE_SECONDS = {"gpu": 270.0, "tpu": 570.0}

BRIDGE_SCRIPT_CODE = '''import asyncio, sys, time
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, Response
import httpx, uvicorn

app = FastAPI()
client = httpx.AsyncClient(timeout=3600.0)
UPSTREAM = "http://127.0.0.1:8080"

# Last-activity tracking for kernel idle auto-shutdown: the kernel loop polls
# this timestamp to decide whether anyone is actually using the GPU. Updated on
# every inference-shaped request (POST to a /v1/* or /chat/completions path).
_last_activity = time.time()
_active_requests = 0

def _note_activity():
    global _last_activity
    _last_activity = time.time()

def _request_started():
    global _active_requests
    _active_requests += 1
    _note_activity()

def _request_finished():
    global _active_requests
    _active_requests = max(0, _active_requests - 1)
    _note_activity()

@app.get("/__last_activity")
async def get_last_activity():
    return {"last_activity": _last_activity, "active_requests": _active_requests}

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"])
async def route_all(request: Request, path: str):
    target = f"{UPSTREAM}/{path}"
    if request.method == "POST" and path in ("v1/chat/completions", "chat/completions"):
        body = await request.json()
        _request_started()
        is_stream = bool(body.get("stream", False))
        if is_stream:
            async def event_generator():
                try:
                    yield b": keepalive\\n\\n"
                    up_req = client.build_request("POST", target, json=body, headers={"Content-Type": "application/json"})
                    resp_future = asyncio.create_task(client.send(up_req, stream=True))
                    while not resp_future.done():
                        done, _ = await asyncio.wait([resp_future], timeout=10.0)
                        if not done:
                            yield b": keepalive\\n\\n"
                    resp = await resp_future
                    try:
                        async for chunk in resp.aiter_raw():
                            yield chunk
                    finally:
                        await resp.aclose()
                finally:
                    _request_finished()
            return StreamingResponse(
                event_generator(),
                status_code=200,
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
            )
        else:
            try:
                resp = await client.post(target, json=body, headers={"Content-Type": "application/json"})
                return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type", "application/json"))
            finally:
                _request_finished()
    else:
        content = await request.body()
        tracked = request.method == "POST" and (path.startswith("v1/") or path.startswith("/v1/") or "completions" in path.lower())
        if tracked:
            _request_started()
        try:
            resp = await client.request(request.method, target, content=content, headers={k: v for k, v in request.headers.items() if k.lower() != "host"})
            return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type", "application/json"))
        finally:
            if tracked:
                _request_finished()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8081, log_level="warning")
'''
BRIDGE_CODE_B64 = base64.b64encode(BRIDGE_SCRIPT_CODE.encode("utf-8")).decode("utf-8")

logger = logging.getLogger("kaggle_manager")

KAGGLE_VENV_BIN = Path(os.environ.get("GPU_HUB_KAGGLE_BIN", "/root/kaggle-gpu/venv/bin/kaggle"))
DEFAULT_KAGGLE_CREDS = Path(os.environ.get("GPU_HUB_KAGGLE_CONFIG", "/root/.kaggle/kaggle.json"))
KAGGLE_CONFIG_PATH = DEFAULT_KAGGLE_CREDS
BASE_DIR = Path(__file__).resolve().parent
BUNDLE_DIR = BASE_DIR / "kernel_bundle"
ACCOUNT_POOL_PATH = BASE_DIR / "kaggle_accounts.json"
TPU_LAB_DIR = BASE_DIR / "tpu_lab" / "glm53_flash"
QWEN_TPU_DIR = BASE_DIR / "tpu_lab" / "qwen38_27b"
GLM_DATASETS = [
    "rahim3/glm53-flash-iq3xxs-1",
    "rahim3/glm53-flash-iq3xxs-2",
    "rahim3/glm53-flash-fp8-1",
    "rahim3/glm53-flash-fp8-2",
    "rahim3/glm53-flash-fp8-3",
    "rahim3/glm53-flash-fp8-4",
]
GLM_FAST_DATASETS = [
    "rahim3/glm53-flash-iq3xxs-1",
    "rahim3/glm53-flash-iq3xxs-2",
    "rahim3/glm53-flash-serve-v3",
]
GLM_FALLBACK_DATASETS = GLM_DATASETS
QWEN_TPU_DATASETS = ["rahim3/qwen3-8-27b-bf16", "rahim3/qwen38-tpu-env-v5e8"]
FAILOVER_COOLDOWN_SECONDS = 15
RUNTIME_BUNDLE_DIR = Path("/tmp/gpu-hub-kernels")

_TOKEN_CACHE: Dict[str, str] = {}


def clean_kaggle_output(text: str) -> str:
    """Remove Kaggle CLI upgrade notices without hiding the actual status/error."""
    cleaned = []
    for line in str(text or "").splitlines():
        lowered = line.lower()
        if "looks like you're using an outdated" in lowered and "kaggle" in lowered:
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def kaggle_error_line(text: str) -> str:
    cleaned = clean_kaggle_output(text)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    for line in lines:
        lowered = line.lower()
        if any(term in lowered for term in ("error", "failed", "forbidden", "unauthorized", "quota", "conflict", "not valid", "not found")):
            return line
    return lines[-1] if lines else "Kaggle CLI returned no diagnostic output"


@dataclass
class KernelSession:
    id: str
    role: str
    account_id: str
    model_id: str
    accelerator: str
    launch_time: float = field(default_factory=time.time)
    kernel_slug: str = ""
    kernel_title: str = ""
    bundle_dir: Optional[Path] = None
    tunnel_url: str = ""
    status: str = "initializing"
    status_detail: str = ""
    ready_time: Optional[float] = None
    last_error: str = ""
    trigger_reason: str = ""
    kaggle_poll_frozen: bool = False
    health_failures: int = 0
    unhealthy_since: float = 0.0
    tpu_recreate_attempts: int = 0
    stop_event: threading.Event = field(default_factory=threading.Event, repr=False)
    monitor_thread: Optional[threading.Thread] = field(default=None, repr=False)

    def public(self, account: Optional[Dict[str, Any]] = None, estimate: float = 0.0) -> Dict[str, Any]:
        now = time.time()
        elapsed = max(0.0, now - self.launch_time)
        return {
            "id": self.id,
            "role": self.role,
            "account_id": self.account_id,
            "username": (account or {}).get("username", ""),
            "model_id": self.model_id,
            "accelerator": self.accelerator,
            "kernel_slug": self.kernel_slug,
            "status": self.status,
            "status_detail": self.status_detail,
            "tunnel_url": self.tunnel_url,
            "launch_time": self.launch_time,
            "ready_time": self.ready_time,
            "elapsed_seconds": elapsed,
            "estimated_total_seconds": estimate,
            "estimated_remaining_seconds": max(0.0, estimate - elapsed),
            "max_remaining_seconds": max(0.0, MAX_BOOT_SECONDS - elapsed),
            "eta_exceeded": elapsed > MAX_BOOT_SECONDS,
            "last_error": self.last_error,
            "kaggle_poll_frozen": self.kaggle_poll_frozen,
            "unhealthy_since": self.unhealthy_since or None,
            "tpu_recreate_attempts": self.tpu_recreate_attempts,
        }


def introspect_kaggle_token(token: str, proxy: Optional[str] = None) -> Optional[str]:
    """
    Introspects a Kaggle OAuth token via the official OAuth introspection endpoint:
    POST https://api.kaggle.com/v1/security.OAuthService/IntrospectToken
    Payload: {"token": token}
    Header: Authorization: Bearer <token>
    Supports proxy if provided and not in ('server', 'direct', 'none', '').
    Returns username if active is True and username is present, else None.
    """
    if not token or not token.strip():
        return None
    token = token.strip()
    url = "https://api.kaggle.com/v1/security.OAuthService/IntrospectToken"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    payload = {"token": token}

    proxies = None
    if proxy and proxy.lower() not in ("server", "direct", "none", ""):
        proxies = {"http": proxy, "https": proxy}

    try:
        session = requests.Session()
        session.trust_env = False
        if proxies:
            session.proxies.update(proxies)
        resp = session.post(url, json=payload, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("active") and data.get("username"):
                username = str(data["username"]).strip()
                if username:
                    logger.info(f"Kaggle token introspected successfully: username={username}")
                    return username
        else:
            logger.debug(f"IntrospectToken returned HTTP {resp.status_code}: {resp.text}")
    except Exception as e:
        # A configured proxy is a strict routing boundary. Never retry this
        # request directly because doing so would reveal the VPS address.
        logger.warning(f"Kaggle token introspection request failed (proxied={bool(proxies)}): {e}")
    return None


def detect_kaggle_username(
    token: Optional[str] = None,
    proxy: Optional[str] = None,
    force: bool = False
) -> Optional[str]:
    """
    Detects the Kaggle username:
    1. Tries OAuth token introspection (IntrospectToken).
    2. If that fails, falls back to parsing 'kaggle kernels list --mine'.
    Returns detected username, or None.
    """
    if not token:
        access_token_file = Path("/root/.kaggle/access_token")
        if access_token_file.exists():
            try:
                t = access_token_file.read_text(encoding="utf-8").strip()
                if t:
                    token = t
            except Exception:
                pass
        if not token and DEFAULT_KAGGLE_CREDS.exists():
            try:
                cdata = json.loads(DEFAULT_KAGGLE_CREDS.read_text(encoding="utf-8"))
                k = str(cdata.get("key", "")).strip()
                if k:
                    token = k
            except Exception:
                pass

    if token:
        token = token.strip()
        if not force and token in _TOKEN_CACHE:
            return _TOKEN_CACHE[token]

        # 1. Try OAuth token introspection
        detected = introspect_kaggle_token(token, proxy=proxy)
        if detected:
            _TOKEN_CACHE[token] = detected
            return detected

    # 2. Fall back to account-scoped Kaggle CLI CSV listings.
    try:
        kaggle_bin = str(KAGGLE_VENV_BIN) if KAGGLE_VENV_BIN.exists() else "kaggle"
        env = os.environ.copy()
        if token:
            env["KAGGLE_API_TOKEN"] = token
            env["KAGGLE_KEY"] = token
        else:
            if DEFAULT_KAGGLE_CREDS.exists():
                try:
                    cdata = json.loads(DEFAULT_KAGGLE_CREDS.read_text(encoding="utf-8"))
                    k = str(cdata.get("key", "")).strip()
                    if k:
                        env["KAGGLE_API_TOKEN"] = k
                        env["KAGGLE_KEY"] = k
                    if cdata.get("username"):
                        env["KAGGLE_USERNAME"] = str(cdata["username"]).strip()
                except Exception:
                    pass
            access_token_file = Path("/root/.kaggle/access_token")
            if access_token_file.exists():
                try:
                    t = access_token_file.read_text(encoding="utf-8").strip()
                    if t:
                        env["KAGGLE_API_TOKEN"] = t
                except Exception:
                    pass

        if proxy and proxy.lower() not in ("server", "direct", "none", ""):
            env["HTTP_PROXY"] = proxy
            env["HTTPS_PROXY"] = proxy
            env["http_proxy"] = proxy
            env["https_proxy"] = proxy
            env["ALL_PROXY"] = proxy
            env["all_proxy"] = proxy
            env["NO_PROXY"] = ""
            env["no_proxy"] = ""
        else:
            for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
                env.pop(k, None)

        commands = [
            [kaggle_bin, "kernels", "list", "--mine", "--csv"],
            [kaggle_bin, "datasets", "list", "--mine", "--csv"],
        ]
        for command in commands:
            proc = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
            output = clean_kaggle_output((proc.stdout or "") + "\n" + (proc.stderr or ""))
            if proc.returncode != 0 or not output:
                continue
            try:
                rows = csv.DictReader(io.StringIO(output))
                refs = [str(row.get("ref") or row.get("id") or "").strip() for row in rows]
            except Exception:
                refs = []
            for ref in refs:
                if "/" not in ref:
                    continue
                detected_owner = ref.split("/", 1)[0].strip()
                if detected_owner:
                    logger.info(f"Detected Kaggle username via CLI fallback: {detected_owner}")
                    if token:
                        _TOKEN_CACHE[token] = detected_owner
                    return detected_owner
    except Exception as e:
        logger.debug(f"CLI username detection fallback failed: {e}")

    # 3. Last resort: direct Kaggle HTTP API using Bearer authentication.
    if token:
        session = requests.Session()
        session.trust_env = False
        if proxy and proxy.lower() not in ("server", "direct", "none", ""):
            session.proxies.update({"http": proxy, "https": proxy})
        headers = {"Authorization": f"Bearer {token}", "User-Agent": "GPU-Hub/2.0"}
        for url in (
            "https://api.kaggle.com/v1/kernels/list?mine=true&pageSize=1",
            "https://api.kaggle.com/v1/datasets/list?mine=true&pageSize=1",
        ):
            try:
                response = session.get(url, headers=headers, timeout=15)
                if response.status_code != 200:
                    continue
                payload = response.json()
                items = payload if isinstance(payload, list) else payload.get("items") or payload.get("kernels") or payload.get("datasets") or []
                for item in items:
                    ref = str((item or {}).get("ref") or (item or {}).get("id") or "")
                    if "/" in ref:
                        detected_owner = ref.split("/", 1)[0].strip()
                        if detected_owner:
                            _TOKEN_CACHE[token] = detected_owner
                            return detected_owner
            except Exception as exc:
                logger.debug(f"HTTP username detection fallback failed for {url}: {exc}")

    return None


class KaggleKernelManager:
    def __init__(self, config_callback=None):
        self.config_callback = config_callback
        self.active_model_id = DEFAULT_MODEL_ID
        self.accelerator_mode = "gpu"
        self.tunnel_url = ""
        self.status = "idle"  # idle, initializing, downloading, loading_vram, tunnel_active, ready, error
        self.status_detail = "System initialized and waiting for launch."
        self.launch_time: Optional[float] = None
        self.kernel_slug = "gpu-hub/gpu-engine"
        self.kernel_title = "GPU Engine"
        self.tunnel_type = "cloudflare"  # cloudflare, ngrok
        self.ngrok_token = ""
        self.webhook_host = os.environ.get("GPU_HUB_WEBHOOK_HOST", "http://127.0.0.1:8778").rstrip("/")
        self.recent_logs = []
        self._polling_thread = None
        self._stop_polling = threading.Event()
        self.kaggle_proxy = "server"
        self.default_network_route = "server"
        self.default_custom_proxy = ""
        self.auto_key_rotation = True
        self.account_pool = KaggleAccountPool(ACCOUNT_POOL_PATH, DEFAULT_KAGGLE_CREDS)
        self.webshare = WebshareConnector()
        self.idle_timeout_min: float = IDLE_TIMEOUT_MIN_DEFAULT
        self.max_runtime_min: float = MAX_RUNTIME_MIN_DEFAULT
        self._cached_kernels: Optional[List[Dict[str, Any]]] = None
        self._cached_kernels_time: float = 0.0
        self._cached_kernels_account_id: Optional[str] = None
        self._launch_account_id: Optional[str] = None
        self._failover_thread: Optional[threading.Thread] = None
        self._failover_lock = threading.Lock()
        self._stop_failover = threading.Event()
        self.client_api_key = ""
        self.upstream_api_key = ""
        self.active_requests = 0
        self.queued_requests = 0
        self.last_inference_activity = time.time()
        self._state_lock = threading.RLock()
        self.sessions: Dict[str, KernelSession] = {}
        self.active_session_id: Optional[str] = None
        self.warming_session_id: Optional[str] = None
        self.test_session_ids: set[str] = set()
        self.prewarm_warning = ""
        self.consecutive_launch_failures = 0
        self.circuit_open_until = 0.0
        self.circuit_backoff_seconds = 60
        self._single_key_rollover_thread: Optional[threading.Thread] = None
        self._proxy_transition_lock = threading.RLock()
        self._upstream_active_counts: Dict[str, int] = {}
        self._lifecycle_stop = threading.Event()
        self._lifecycle_thread = threading.Thread(target=self._lifecycle_loop, daemon=True, name="gpu-hub-lifecycle")
        self.load_kaggle_credentials()
        if os.environ.get("GPU_HUB_DISABLE_BACKGROUND") != "1":
            self._lifecycle_thread.start()

    def get_current_account(self, account_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        return self.account_pool.get(account_id) if account_id else self.account_pool.active()

    def model_spec(self, model_id: Optional[str] = None, accelerator: Optional[str] = None) -> Dict[str, Any]:
        model_id = normalize_model_id(model_id or self.active_model_id)
        base = MODELS.get(model_id)
        if not base:
            raise ValueError(f"Unknown model: {model_id}")
        selected = str(accelerator or self.accelerator_mode or base.get("default_accelerator") or "gpu")
        return model_for_accelerator(model_id, selected)

    def model_is_compatible(self, model_id: str, accelerator: Optional[str] = None) -> bool:
        model_id = normalize_model_id(model_id)
        return str(accelerator or self.accelerator_mode) in MODELS.get(model_id, {}).get("supported_accelerators", [])

    def get_current_token(self, account_id: Optional[str] = None) -> Optional[str]:
        account = self.get_current_account(account_id)
        return str(account.get("key", "")).strip() if account else None

    def introspect_kaggle_token(self, token: Optional[str] = None) -> Optional[str]:
        if not token:
            token = self.get_current_token()
        return introspect_kaggle_token(token, proxy=getattr(self, "kaggle_proxy", "server"))

    def detect_username(self, token: Optional[str] = None, force: bool = False) -> Optional[str]:
        return detect_kaggle_username(token=token, proxy=getattr(self, "kaggle_proxy", "server"), force=force)

    @staticmethod
    def _validate_proxy_choice(proxy: str) -> str:
        proxy = str(proxy or "server").strip() or "server"
        if proxy in {"default", "server", "webshare"}:
            return proxy
        parsed = urlsplit(proxy)
        if parsed.scheme.lower() not in {"http", "https", "socks5", "socks5h"} or not parsed.hostname or not parsed.port:
            raise ValueError("Proxy must be 'webshare', 'server', or a valid http(s)/socks5 URL with a port")
        return proxy

    def _effective_default_proxy(self) -> str:
        mode = str(getattr(self, "default_network_route", "server") or "server").strip()
        if mode == "custom":
            return self._validate_proxy_choice(getattr(self, "default_custom_proxy", ""))
        return self._validate_proxy_choice(mode)

    def _effective_proxy_choice(self, proxy: str) -> str:
        proxy = self._validate_proxy_choice(proxy)
        return self._effective_default_proxy() if proxy == "default" else proxy

    def _proxy_queue_position(self, account_id: str) -> int:
        return next(
            (max(0, int(item.get("queue_position") or 1) - 1) for item in self.account_pool.list_public() if item["id"] == account_id),
            0,
        )

    def _prepare_account_route(
        self,
        account_id: str,
        purpose: str = "kaggle_call",
        previous_account_id: Optional[str] = None,
    ) -> str:
        """Resolve and verify a strict account route before any Kaggle traffic.

        A single rotatable Webshare endpoint needs special handling during an
        account transition: freeze the old account's Kaggle polling, rotate,
        verify that the exit address changed, then permit the new account to
        authenticate. No direct-network fallback is ever attempted.
        """
        account = self.account_pool.get(account_id)
        if not account:
            raise KeyError(account_id)
        configured_mode = self._validate_proxy_choice(account.get("proxy") or "default")
        mode = self._effective_proxy_choice(configured_mode)
        if mode == "server":
            return "server"
        if mode != "webshare":
            exit_ip = self.webshare.test_proxy(mode, force=True)
            self.account_pool.update_proxy(account_id, configured_mode, "", exit_ip)
            return mode

        with self._proxy_transition_lock:
            proxies = self.webshare.fetch_proxies() if self.webshare.proxy_count() == 0 else [
                self.webshare.proxy_for_index(index) for index in range(self.webshare.proxy_count())
            ]
            if not proxies:
                raise ProxyRouteError("PROXY_ERROR: Webshare is connected but returned no usable proxy")

            previous = self.account_pool.get(previous_account_id) if previous_account_id else None
            transition = bool(previous and previous["id"] != account_id and purpose in {"prewarm", "account_switch", "cold_failover"})

            if len(proxies) == 1 and transition:
                status = self.webshare.public_status()
                if not status.get("auto_refresh"):
                    raise ProxyRouteError(
                        "PROXY_ERROR: single-IP Webshare account transition requires Auto-Refresh to prevent an IP leak"
                    )
                old_session = next(
                    (
                        session for session in self.sessions.values()
                        if session.account_id == previous["id"] and session.status not in {"failed", "stopped"}
                    ),
                    None,
                )
                if old_session:
                    old_session.kaggle_poll_frozen = True
                    self.log(
                        f"[PROXY SHIELD] Frozen Kaggle polling for {previous.get('username')} before Webshare rotation."
                    )
                current_url = str(previous.get("assigned_proxy_url") or proxies[0]["proxy_url"])
                previous_ip = str(previous.get("last_exit_ip") or "").strip()
                if not previous_ip:
                    previous_ip = self.webshare.test_proxy(current_url, force=True)
                    self.account_pool.update_proxy(previous["id"], "webshare", current_url, previous_ip)
                rotated = self.webshare.refresh_ip(previous_exit_ip=previous_ip, wait_seconds=5)
                proxies = rotated["proxies"]
                proxy_url = proxies[0]["proxy_url"]
                exit_ip = rotated["exit_ip"]
                self.account_pool.update_proxy(account_id, configured_mode, proxy_url, exit_ip)
                self.log(
                    f"[PROXY SHIELD] Webshare exit changed for {purpose}; {account.get('username')} is pinned to {exit_ip}."
                )
                return proxy_url

            used_ips = set()
            for item in self.account_pool.list_public():
                if item["id"] == account_id or not item.get("last_exit_ip"):
                    continue
                raw = self.account_pool.get(item["id"]) or {}
                if self._effective_proxy_choice(raw.get("proxy") or "default") == "webshare":
                    used_ips.add(str(item.get("last_exit_ip") or "").strip())
            assigned = str(account.get("assigned_proxy_url") or "").strip()
            candidates = list(proxies)
            start = self._proxy_queue_position(account_id) % len(candidates)
            candidates = candidates[start:] + candidates[:start]
            if assigned:
                candidates.sort(key=lambda item: 0 if item.get("proxy_url") == assigned else 1)
            for candidate in candidates:
                proxy_url = candidate["proxy_url"]
                try:
                    exit_ip = self.webshare.test_proxy(proxy_url, force=transition)
                except ProxyRouteError:
                    continue
                if len(proxies) >= 2 and exit_ip in used_ips:
                    continue
                self.account_pool.update_proxy(account_id, configured_mode, proxy_url, exit_ip)
                return proxy_url
            raise ProxyRouteError("PROXY_ERROR: no distinct healthy Webshare exit IP is available for this account")

    def _resolve_account_proxy(self, account_id: Optional[str]) -> str:
        account = self.get_current_account(account_id)
        if not account:
            return "server"
        mode = self._effective_proxy_choice(str(account.get("proxy") or "default").strip())
        if mode != "webshare":
            return mode
        assigned = str(account.get("assigned_proxy_url") or "").strip()
        if assigned:
            return assigned
        public_accounts = self.account_pool.list_public()
        position = next((item["queue_position"] - 1 for item in public_accounts if item["id"] == account["id"]), 0)
        proxy = self.webshare.proxy_for_index(position)
        exit_ip = self.webshare.test_proxy(proxy["proxy_url"])
        self.account_pool.update_proxy(account["id"], "webshare", proxy["proxy_url"], exit_ip)
        return proxy["proxy_url"]

    def _ensure_account_proxy(self, account_id: Optional[str]) -> str:
        proxy = self._prepare_account_route(account_id, "kaggle_call") if account_id else self._resolve_account_proxy(account_id)
        if proxy.lower() in ("server", "direct", "none", ""):
            return "server"
        try:
            exit_ip = self.webshare.test_proxy(proxy)
        except Exception as exc:
            raise ProxyRouteError(f"PROXY_ERROR: aborting Kaggle call to prevent VPS IP leak: {exc}") from exc
        account = self.get_current_account(account_id)
        if account:
            configured = account.get("proxy") or "default"
            assigned = proxy if self._effective_proxy_choice(configured) == "webshare" else ""
            self.account_pool.update_proxy(account["id"], configured, assigned, exit_ip)
        return proxy

    def _run_kaggle(
        self,
        account_id: Optional[str],
        args: List[str],
        timeout: int = 30,
        input_text: Optional[str] = None,
    ) -> subprocess.CompletedProcess:
        self._ensure_account_proxy(account_id)
        kaggle_bin = str(KAGGLE_VENV_BIN) if KAGGLE_VENV_BIN.exists() else "kaggle"
        return subprocess.run(
            [kaggle_bin, *args],
            env=self.get_subprocess_env(account_id),
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def get_subprocess_env(self, account_id: Optional[str] = None) -> Dict[str, str]:
        env = os.environ.copy()

        account = self.get_current_account(account_id)
        if account:
            token = str(account.get("key", "")).strip()
            username = str(account.get("username", "")).strip()
            if token:
                env["KAGGLE_API_TOKEN"] = token
                env["KAGGLE_KEY"] = token
            if username:
                env["KAGGLE_USERNAME"] = username

        proxy = self._resolve_account_proxy(account_id) if account else str(getattr(self, "kaggle_proxy", "server"))
        if proxy and proxy.lower() not in ("server", "direct", "none", ""):
            env["HTTP_PROXY"] = proxy
            env["HTTPS_PROXY"] = proxy
            env["http_proxy"] = proxy
            env["https_proxy"] = proxy
            env["ALL_PROXY"] = proxy
            env["all_proxy"] = proxy
            env["NO_PROXY"] = ""
            env["no_proxy"] = ""
        else:
            for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
                env.pop(k, None)
        return env

    def load_kaggle_credentials(self) -> Dict[str, Any]:
        account = self.account_pool.active()
        username = str((account or {}).get("username") or "")
        key_present = bool((account or {}).get("key"))
        proxy = str((account or {}).get("proxy") or "server")
        self.kaggle_proxy = proxy

        if not hasattr(self, "kernel_title") or not self.kernel_title:
            self.kernel_title = "GPU Engine"
        if username and (not hasattr(self, "kernel_slug") or not self.kernel_slug or not self.kernel_slug.startswith(f"{username}/")):
            self.kernel_slug = f"{username}/gpu-engine"

        return {
            "username": username,
            "key_present": key_present,
            "proxy": proxy,
            "active_account_id": (account or {}).get("id"),
            "accounts": self.account_pool.list_public(),
        }

    def detect_username_for_key(self, key: str, proxy: str = "server") -> Optional[str]:
        proxy = self._effective_proxy_choice(proxy)
        route = proxy
        if proxy == "webshare":
            candidate = self.webshare.proxy_for_index(0)
            route = candidate["proxy_url"]
            self.webshare.test_proxy(route, force=True)
        elif proxy != "server":
            self.webshare.test_proxy(proxy, force=True)
        return detect_kaggle_username(token=str(key or "").strip(), proxy=route, force=True)

    def update_account_proxy(self, account_id: str, proxy: str) -> Dict[str, Any]:
        proxy = self._validate_proxy_choice(proxy)
        if not self.account_pool.get(account_id):
            raise KeyError(account_id)
        if proxy == "default":
            self.account_pool.update_proxy(account_id, "default", "", "", proxy_override=False)
            effective = self._effective_default_proxy()
            if effective != "server":
                self._prepare_account_route(account_id, "account_config")
            return self.account_pool.get(account_id) or {}
        elif proxy == "server":
            return self.account_pool.update_proxy(account_id, "server", "", "", proxy_override=True)
        if proxy == "webshare":
            self.account_pool.update_proxy(account_id, "webshare", "", "", proxy_override=True)
            self._prepare_account_route(account_id, "account_config")
        else:
            exit_ip = self.webshare.test_proxy(proxy, force=True)
            self.account_pool.update_proxy(account_id, proxy, "", exit_ip, proxy_override=True)
        return self.account_pool.get(account_id) or {}

    def apply_default_route_to_inherited_accounts(self) -> List[str]:
        updated = []
        for public in self.account_pool.list_public():
            account = self.account_pool.get(public["id"]) or {}
            if account.get("proxy_override"):
                continue
            self.account_pool.update_proxy(public["id"], "default", "", "", proxy_override=False)
            updated.append(public["id"])
        return updated

    def set_default_network_route(self, route: str, custom_proxy: str = "") -> Dict[str, Any]:
        route = str(route or "server").strip().lower()
        if route not in {"server", "webshare", "custom"}:
            raise ValueError("Invalid default network route")
        if route == "custom":
            custom_proxy = self._validate_proxy_choice(custom_proxy)
            if custom_proxy in {"default", "server", "webshare"}:
                raise ValueError("A custom HTTP or SOCKS5 proxy URL is required")
            self.webshare.test_proxy(custom_proxy, force=True)
            self.default_custom_proxy = custom_proxy
        else:
            self.default_custom_proxy = ""
        self.default_network_route = route
        inherited = self.apply_default_route_to_inherited_accounts()
        return {"route": route, "inherited_accounts": inherited}

    def connect_webshare(self, token: str) -> Dict[str, Any]:
        status = self.webshare.connect(token)
        self.default_network_route = "webshare"
        self.default_custom_proxy = ""
        inherited = self.apply_default_route_to_inherited_accounts()
        for account_id in inherited:
            self._prepare_account_route(account_id, "account_config")
        return {**status, "default_network_route": "webshare", "assigned_accounts": inherited}

    def clear_recovery_state(self, account_id: Optional[str] = None) -> Dict[str, Any]:
        cleared = self.account_pool.clear_cooldown(account_id)
        self.consecutive_launch_failures = 0
        self.circuit_open_until = 0.0
        self.circuit_backoff_seconds = 60
        self.prewarm_warning = ""
        self._stop_failover.clear()
        if not self._active_session() and not self._warming_session():
            self.status = "idle"
            self.status_detail = "Recovery state cleared. Ready to launch."
            self.tunnel_url = ""
            self.launch_time = None
        if account_id:
            account = self.account_pool.get(account_id)
            if account and account.get("verified_at") and (self.account_pool.active() or {}).get("id") == account_id:
                self.account_pool.mark(account_id, "ACTIVE")
        return {"success": True, "cleared_accounts": cleared, "status": self.status}

    def disconnect_webshare(self) -> Dict[str, Any]:
        for account in self.account_pool.list_public():
            if account.get("proxy_mode") == "webshare":
                self.account_pool.update_proxy(account["id"], "default", "", "", proxy_override=False)
        self.default_network_route = "server"
        self.default_custom_proxy = ""
        self.apply_default_route_to_inherited_accounts()
        return self.webshare.disconnect()

    def test_account_proxy(self, account_id: str) -> Dict[str, Any]:
        proxy = self._prepare_account_route(account_id, "proxy_test")
        if proxy == "server":
            session = requests.Session()
            session.trust_env = False
            response = session.get("https://api.ipify.org?format=json", timeout=20)
            response.raise_for_status()
            exit_ip = str(response.json().get("ip") or "")
            return {"mode": "server", "exit_ip": exit_ip}
        exit_ip = self.webshare.test_proxy(proxy, force=True)
        account = self.account_pool.get(account_id) or {}
        self.account_pool.update_proxy(
            account_id,
            account.get("proxy") or "default",
            proxy if self._effective_proxy_choice(account.get("proxy") or "default") == "webshare" else "",
            exit_ip,
        )
        effective = self._effective_proxy_choice(account.get("proxy") or "default")
        return {"mode": "webshare" if effective == "webshare" else "custom", "exit_ip": exit_ip}

    def refresh_webshare_ip(self) -> Dict[str, Any]:
        """Rotate the shared Webshare endpoint without letting a live account leak onto the new IP."""
        with self._proxy_transition_lock:
            for session in self.sessions.values():
                account = self.account_pool.get(session.account_id)
                if (
                    session.status not in {"failed", "stopped"}
                    and account
                    and account.get("proxy") == "webshare"
                ):
                    session.kaggle_poll_frozen = True
            status = self.webshare.public_status()
            return self.webshare.refresh_ip(str(status.get("exit_ip") or ""), wait_seconds=5)

    def add_kaggle_account(self, username: str, key: str, proxy: str = "server") -> Dict[str, Any]:
        username = str(username or "").strip()
        key = str(key or "").strip()
        if not key:
            raise ValueError("Kaggle API key is required")
        proxy = self._validate_proxy_choice(proxy)
        if not username:
            username = self.detect_username_for_key(key, proxy) or ""
        if not username:
            raise ValueError("Could not auto-detect the Kaggle username for this key")
        existing = next(
            (account for account in self.account_pool.list_public() if account["username"].lower() == username.strip().lower()),
            None,
        )
        if existing and any(
            session.account_id == existing["id"] and session.status not in {"stopped", "failed"}
            for session in self.sessions.values()
        ):
            raise ValueError("Stop this account's current session before replacing its API key")
        account = self.account_pool.add(username, key, proxy, status="VERIFYING")
        verification = self.verify_kaggle_account(account["id"])
        self._cached_kernels = None
        return verification

    def verify_kaggle_account(self, account_id: str) -> Dict[str, Any]:
        account = self.account_pool.get(account_id)
        if not account:
            raise KeyError(account_id)
        self.account_pool.mark(account_id, "VERIFYING")
        kaggle_bin = str(KAGGLE_VENV_BIN) if KAGGLE_VENV_BIN.exists() else "kaggle"
        attempts = [
            [kaggle_bin, "kernels", "list", "--mine", "--page-size", "1"],
            [kaggle_bin, "kernels", "list", "-m", "--page-size", "1"],
            [kaggle_bin, "datasets", "list", "--page-size", "1"],
        ]
        last_output = ""
        verified = False
        for command in attempts:
            try:
                result = self._run_kaggle(account_id, command[1:], timeout=25)
            except Exception as exc:
                last_output = str(exc)
                continue
            last_output = clean_kaggle_output((result.stdout or "") + "\n" + (result.stderr or ""))
            if result.returncode == 0:
                verified = True
                break
            lowered = last_output.lower()
            if any(term in lowered for term in ("401", "403", "unauthorized", "forbidden", "invalid api", "authentication")):
                break
        detected = None
        if verified and str(account.get("key", "")).upper().startswith("KGAT" + "_"):
            detected = introspect_kaggle_token(account["key"], proxy=self._resolve_account_proxy(account_id))
        updated = self.account_pool.set_verification(
            account_id,
            verified,
            error="" if verified else (last_output or "Kaggle authentication failed"),
            username=detected,
        )
        if verified and not self.account_pool.active():
            updated = self.account_pool.activate(account_id)
        self.log(
            f"Kaggle key verification {'passed' if verified else 'failed'} for {updated.get('username', account_id)}."
        )
        return updated

    def verify_unverified_accounts(self) -> None:
        for account in self.account_pool.list_public():
            if not account.get("verified") and account.get("status") != "INVALID_KEY":
                try:
                    self.verify_kaggle_account(account["id"])
                except Exception as exc:
                    self.log(f"Account verification error for {account.get('username')}: {exc}")

    def activate_kaggle_account(self, account_id: str) -> Dict[str, Any]:
        self.clear_recovery_state(account_id)
        active_session = self._active_session()
        if active_session and active_session.status == "ready" and active_session.account_id != account_id:
            account = self.account_pool.get(account_id)
            if not account or not account.get("verified_at"):
                raise ValueError("Account key is not verified")
            self._prepare_account_route(account_id, "prewarm", active_session.account_id)
            result = self._launch_session(account_id, active_session.model_id, "prewarm", trigger_reason="manual_account_switch")
            if not result.get("success"):
                raise ValueError(result.get("error", "Could not pre-warm selected account"))
            return self.account_pool.get(account_id) or account
        previous = self.account_pool.active()
        if previous and previous["id"] != account_id:
            self._prepare_account_route(account_id, "account_switch", previous["id"])
        account = self.account_pool.activate(account_id)
        self.kaggle_proxy = account.get("proxy", "server")
        self._cached_kernels = None
        self._cached_kernels_account_id = None
        return account

    def delete_kaggle_account(self, account_id: str) -> bool:
        deleted = self.account_pool.delete(account_id)
        if deleted:
            self._cached_kernels = None
            self._cached_kernels_account_id = None
        return deleted

    def note_request_queued(self) -> None:
        self.queued_requests += 1
        self.last_inference_activity = time.time()

    def note_request_started(self, upstream_url: Optional[str] = None) -> None:
        self.queued_requests = max(0, self.queued_requests - 1)
        self.active_requests += 1
        self.last_inference_activity = time.time()
        if upstream_url:
            with self._state_lock:
                self._upstream_active_counts[upstream_url] = self._upstream_active_counts.get(upstream_url, 0) + 1

    def note_request_aborted(self) -> None:
        self.queued_requests = max(0, self.queued_requests - 1)
        self.last_inference_activity = time.time()

    def note_request_completed(self, upstream_url: Optional[str] = None) -> None:
        self.active_requests = max(0, self.active_requests - 1)
        self.last_inference_activity = time.time()
        if upstream_url:
            with self._state_lock:
                self._upstream_active_counts[upstream_url] = max(0, self._upstream_active_counts.get(upstream_url, 0) - 1)

    def reserve_upstream(self) -> str:
        """Atomically reserve the current tunnel so hot-swap draining cannot race a new request."""
        with self._state_lock:
            upstream = self.tunnel_url
            if not upstream:
                raise RuntimeError("No active upstream tunnel")
            self.queued_requests = max(0, self.queued_requests - 1)
            self.active_requests += 1
            self.last_inference_activity = time.time()
            self._upstream_active_counts[upstream] = self._upstream_active_counts.get(upstream, 0) + 1
            return upstream

    def release_upstream(self, upstream_url: str) -> None:
        self.note_request_completed(upstream_url)

    def fetch_existing_kernels(self, force: bool = False, account_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieves existing kernels for the active Kaggle user account via KaggleApi SDK or CLI fallback.
        """
        now = time.time()
        account = self.account_pool.get(account_id) if account_id else self.account_pool.active()
        account_id = (account or {}).get("id")
        if not force and self._cached_kernels_account_id == account_id and self._cached_kernels is not None and (now - self._cached_kernels_time) < 60:
            return self._cached_kernels

        results: List[Dict[str, Any]] = []

        # Use account-scoped CLI subprocesses only. The Python SDK mutates global
        # authentication/proxy environment and is unsafe when Tunnel A and B are
        # being managed concurrently for different accounts.
        try:
            proc = self._run_kaggle(account_id, ["kernels", "list", "--mine", "--csv"], timeout=20)
            output = clean_kaggle_output((proc.stdout or "") + "\n" + (proc.stderr or ""))
            if proc.returncode == 0 and output:
                reader = csv.DictReader(io.StringIO(output))
                for row in reader:
                    ref = row.get("ref", "").strip()
                    title = row.get("title", "").strip()
                    if ref:
                        slug = ref.split("/")[-1] if "/" in ref else ref
                        owner = ref.split("/")[0] if "/" in ref else ""
                        results.append({
                            "ref": ref,
                            "slug": slug,
                            "title": title,
                            "owner": owner
                        })
                self._cached_kernels = results
                self._cached_kernels_time = now
                self._cached_kernels_account_id = account_id
                return results
        except Exception as e:
            logger.debug(f"fetch_existing_kernels CLI CSV attempt: {e}")

        # Method 3: CLI table fallback
        try:
            proc = self._run_kaggle(account_id, ["kernels", "list", "--mine"], timeout=20)
            output = clean_kaggle_output((proc.stdout or "") + "\n" + (proc.stderr or ""))
            if proc.returncode == 0 and output:
                for line in output.splitlines():
                    line = line.strip()
                    if "/" in line and not line.startswith("-") and not line.lower().startswith("ref"):
                        parts = [p.strip() for p in re.split(r"\s{2,}", line) if p.strip()]
                        if parts:
                            ref = parts[0]
                            title = parts[1] if len(parts) > 1 else ""
                            slug = ref.split("/")[-1] if "/" in ref else ref
                            owner = ref.split("/")[0] if "/" in ref else ""
                            results.append({
                                "ref": ref,
                                "slug": slug,
                                "title": title,
                                "owner": owner
                            })
                if results:
                    self._cached_kernels = results
                    self._cached_kernels_time = now
                    self._cached_kernels_account_id = account_id
                    return results
        except Exception as e:
            logger.debug(f"fetch_existing_kernels CLI table attempt: {e}")

        self._cached_kernels = results
        self._cached_kernels_time = now
        self._cached_kernels_account_id = account_id
        return results

    def resolve_kernel_identity(
        self,
        force: bool = False,
        model_id: Optional[str] = None,
        account_id: Optional[str] = None,
        accelerator: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        """
        Dynamically resolves (owner, slug, title) for the target Kaggle kernel.
        TPU workflows keep their dedicated recipe slugs. GPU models use a stable
        model-specific slug so pushing one model can never cancel or overwrite a
        different model that previously used the legacy ``gpu-engine`` notebook.
        """
        account = self.account_pool.get(account_id) if account_id else self.account_pool.active()
        username = str((account or {}).get("username") or "")
        if not username:
            raise RuntimeError("No ACTIVE Kaggle account is configured")
        existing_kernels = self.fetch_existing_kernels(force=force, account_id=(account or {}).get("id"))
        try:
            model = self.model_spec(model_id or self.active_model_id, accelerator)
        except ValueError:
            model = {}
        is_tpu = model.get("accelerator") == "tpu"
        workflow = model.get("kernel_workflow")
        resolved_model_id = normalize_model_id(model_id or self.active_model_id)
        gpu_model_slug = f"gpu-hub-{re.sub(r'[^a-z0-9]+', '-', resolved_model_id.lower()).strip('-')}"[:48].rstrip("-")
        gpu_model_title = f"GPU Hub {resolved_model_id}"[:50]

        # If existing kernels were found, verify owner
        if existing_kernels:
            first_owner = existing_kernels[0].get("owner")
            if first_owner and first_owner != username:
                username = first_owner

        # 1. Match preferred slugs
        preferred_slugs = (
            ["qwen38-tpu-serve", "gpu-hub-qwen38"] if workflow == "qwen38_tpu"
            else ["glm53-tpu-serve", "gpu-hub-glm53"] if workflow == "glm53_tpu"
            else [gpu_model_slug, "gpu-engine", "hermes-uncensored-engine", "kaggle-gpu-engine"]
            if resolved_model_id == DEFAULT_MODEL_ID else [gpu_model_slug]
        )
        for target_slug in preferred_slugs:
            for k in existing_kernels:
                if k.get("slug", "").strip().lower() == target_slug:
                    owner = k.get("owner") or username
                    slug = k["slug"].strip()
                    title = k.get("title", "").strip() or "GPU Engine"
                    self.kernel_slug = f"{owner}/{slug}"
                    self.kernel_title = title
                    self.log(f"Dynamically adopted existing kernel by slug '{target_slug}': {self.kernel_slug} ('{self.kernel_title}')")
                    return owner, slug, title

        # 2. Match title phrases
        target_title_phrases = (
            ["qwen38", "qwen3.8"] if workflow == "qwen38_tpu"
            else ["glm53", "glm-5.3"] if workflow == "glm53_tpu"
            else ["gpu engine", "uncensored engine"] if resolved_model_id == DEFAULT_MODEL_ID else []
        )
        for phrase in target_title_phrases:
            for k in existing_kernels:
                k_title = (k.get("title") or "").strip().lower()
                if phrase in k_title:
                    owner = k.get("owner") or username
                    slug = k["slug"].strip()
                    title = k.get("title", "").strip() or "GPU Engine"
                    self.kernel_slug = f"{owner}/{slug}"
                    self.kernel_title = title
                    self.log(f"Dynamically adopted existing kernel by title '{k.get('title')}': {self.kernel_slug} ('{self.kernel_title}')")
                    return owner, slug, title

        # 3. Default for new accounts or accounts without engine notebooks
        default_slug = "qwen38-tpu-serve" if workflow == "qwen38_tpu" else "glm53-tpu-serve" if is_tpu else gpu_model_slug
        default_title = "Qwen38 TPU Serve" if workflow == "qwen38_tpu" else "GLM53 TPU Serve" if is_tpu else gpu_model_title

        taken_slugs = {k.get("slug", "").strip().lower() for k in existing_kernels}
        taken_titles = {k.get("title", "").strip().lower() for k in existing_kernels}

        if default_title.lower() in taken_titles or default_slug.lower() in taken_slugs:
            alternatives = ([
                ("gpu-hub-qwen38", "GPU Hub Qwen38"),
                ("qwen38-27b-api", "Qwen38 27B API"),
            ] if workflow == "qwen38_tpu" else [
                ("gpu-hub-glm53", "GPU Hub GLM53"),
                ("glm53-flash-api", "GLM53 Flash API"),
            ] if is_tpu else [
                (f"{gpu_model_slug}-v2"[:48].rstrip("-"), f"{gpu_model_title} v2"[:80]),
                (f"{gpu_model_slug}-v3"[:48].rstrip("-"), f"{gpu_model_title} v3"[:80]),
            ])
            for alt_slug, alt_title in alternatives:
                if alt_slug.lower() not in taken_slugs and alt_title.lower() not in taken_titles:
                    default_slug = alt_slug
                    default_title = alt_title
                    break
            else:
                rand_id = int(time.time()) % 10000
                default_slug = f"gpu-engine-{rand_id}"
                default_title = f"GPU Engine {rand_id}"

        self.kernel_slug = f"{username}/{default_slug}"
        self.kernel_title = default_title
        self.log(f"Resolved new kernel identity: {self.kernel_slug} ('{self.kernel_title}')")
        return username, default_slug, default_title

    def log(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        logger.info(entry)
        self.recent_logs.append(entry)
        if len(self.recent_logs) > 200:
            self.recent_logs.pop(0)

    def set_status(self, status: str, detail: str = ""):
        self.status = status
        if detail:
            self.status_detail = detail
        self.log(f"Status changed -> {status.upper()}: {detail}")
        if self.config_callback:
            self.config_callback()

    def set_tunnel_url(self, url: str):
        self.tunnel_url = url.rstrip("/")
        if self.launch_time is None:
            self.launch_time = time.time()
        self.log(f"Active Tunnel Upstream URL registered: {self.tunnel_url}")
        self.set_status("ready", f"Tunnel live: {self.tunnel_url}")
        if self.config_callback:
            self.config_callback()

    @staticmethod
    def _engine_b64(pkg_dir: Path) -> str:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as archive:
            for source in sorted(pkg_dir.glob("*.py")):
                archive.add(source, arcname=f"{pkg_dir.name}/{source.name}")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def generate_glm_kernel_script(self, model_id: str, session_id: str = "", session_role: str = "active") -> str:
        source_path = TPU_LAB_DIR / "serve_glm53.py"
        engine_path = TPU_LAB_DIR / "engine" / "glm53"
        if not source_path.exists() or not engine_path.exists():
            raise RuntimeError("Vendored ARahim3/kaggle-tpu-lab GLM-5.3 workflow is missing")
        api_key = self.upstream_api_key or self.client_api_key
        if not api_key:
            raise RuntimeError("Client API key is not initialized")
        keepalive = float(getattr(self, "max_runtime_min", MAX_RUNTIME_MIN_DEFAULT)) or 538.0
        keepalive = min(538.0, keepalive)
        cfg = {
            "api_key": api_key,
            "max_len": 262144,
            "streams": 3,
            "reasoning_effort_default": "low",
            "keepalive_min": keepalive,
            "idle_timeout_min": float(getattr(self, "idle_timeout_min", IDLE_TIMEOUT_MIN_DEFAULT)),
            "vision": True,
            "serve_dataset": "rahim3/glm53-flash-serve-v3",
            "ntfy_topic": "",
            "gpu_hub_session_id": session_id,
            "gpu_hub_session_role": session_role,
        }
        source = source_path.read_text(encoding="utf-8")
        source, count = re.subn(
            r"^CFG = None  # __LAUNCHER_CONFIG__.*$",
            f"CFG = {cfg!r}",
            source,
            count=1,
            flags=re.M,
        )
        if count != 1:
            raise RuntimeError("GLM launcher config marker was not found")
        source, count = re.subn(
            r'^ENGINE_B64 = ""  # __ENGINE__.*$',
            f'ENGINE_B64 = "{self._engine_b64(engine_path)}"',
            source,
            count=1,
            flags=re.M,
        )
        if count != 1:
            raise RuntimeError("GLM engine marker was not found")
        return source

    def generate_qwen_tpu_kernel_script(self, model_id: str, session_id: str = "", session_role: str = "active") -> str:
        source_path = QWEN_TPU_DIR / "serve_qwen38.py"
        if not source_path.exists():
            raise RuntimeError("Vendored ARahim3 Qwen3.8 TPU workflow is missing")
        api_key = self.upstream_api_key or self.client_api_key
        if not api_key:
            raise RuntimeError("Client API key is not initialized")
        cfg = {
            "api_key": api_key,
            "max_model_len": 262144,
            "max_num_seqs": 4,
            "mtp_tokens": 3,
            "reasoning_effort_default": "xhigh",
            "keepalive_min": min(538.0, float(getattr(self, "max_runtime_min", MAX_RUNTIME_MIN_DEFAULT)) or 538.0),
            "weights_dataset": "rahim3/qwen3-8-27b-bf16",
            "ntfy_topic": "",
            "gpu_hub_session_id": session_id,
            "gpu_hub_session_role": session_role,
        }
        source = source_path.read_text(encoding="utf-8")
        source, count = re.subn(
            r"^CFG = None  # __LAUNCHER_CONFIG__.*$",
            f"CFG = {cfg!r}",
            source,
            count=1,
            flags=re.M,
        )
        if count != 1:
            raise RuntimeError("Qwen TPU launcher config marker was not found")
        return source

    def generate_kernel_script(self, model_id: str, session_id: str = "", session_role: str = "active", accelerator: Optional[str] = None) -> str:
        model = self.model_spec(model_id, accelerator)
        if model.get("kernel_workflow") == "qwen38_tpu":
            return self.generate_qwen_tpu_kernel_script(model_id, session_id=session_id, session_role=session_role)
        if model.get("kernel_workflow") == "glm53_tpu":
            return self.generate_glm_kernel_script(model_id, session_id=session_id, session_role=session_role)
        if model.get("accelerator") == "tpu":
            raise RuntimeError(f"No verified TPU runner is bundled for {model['name']}")
        server_args_str = ", ".join([json.dumps(arg) for arg in model["server_args"]])
        model_name = model["name"]
        model_params = model["parameters"]
        model_quant = model["quant"]
        repo_id = model["repo_id"]
        filename = model["filename"]
        filesize_gb = model["filesize_gb"]
        model_key = model["id"]
        webhook_target = f"{self.webhook_host}/api/tunnel/register"
        idle_timeout_min = float(getattr(self, "idle_timeout_min", IDLE_TIMEOUT_MIN_DEFAULT))
        max_runtime_min = float(getattr(self, "max_runtime_min", MAX_RUNTIME_MIN_DEFAULT))

        ctx_disp = model.get("context_display", f"{model.get('context_length', 32768)} tokens")
        rope_disp = model.get("rope_scaling", "Native / None")
        kv_disp = model.get("kv_cache", "q4_0")

        script_code = f'''#!/usr/bin/env python3
# ==============================================================================
# Autonomous Uncensored AI Agent Kernel - Dual Tesla T4 / P100 Execution Engine
# Model: {model_name} ({model_params}, {model_quant})
# Context Window: {ctx_disp} | RoPE: {rope_disp} | KV: {kv_disp}
# Hugging Face: {repo_id} / {filename}
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

MODEL_NAME = {json.dumps(model_name)}
MODEL_REPO = {json.dumps(repo_id)}
MODEL_FILENAME = {json.dumps(filename)}
MODEL_FILESIZE = {filesize_gb}
MODEL_ID = {json.dumps(model_key)}
WEBHOOK_TARGET = {json.dumps(webhook_target)}
GPU_HUB_SESSION_ID = {json.dumps(session_id)}
GPU_HUB_SESSION_ROLE = {json.dumps(session_role)}

# --- Auto-shutdown lifecycle timers (minutes) ----------------------------------
# Injected at bundle-prep time from the dashboard settings (config.json), so
# editing "Auto-shutdown idle timeout" in the GPU Hub changes the real kernel
# behavior on the NEXT launch. 0 disables a timer.
IDLE_TIMEOUT_MIN = {idle_timeout_min!r}   # no inference request for this long -> graceful exit
MAX_RUNTIME_MIN = {max_runtime_min!r}    # hard session cap even under continuous use
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
print("\\n[STEP 1] Installing huggingface_hub, requests, and bridge dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub", "requests", "fastapi", "uvicorn", "httpx"], check=False)

# Step 2: Acquire llama.cpp binary (Prebuilt CUDA or fast source build)
LLAMA_SERVER = "/tmp/bin/llama-server"

def setup_llama_server():
    global LLAMA_SERVER
    print("\\n[STEP 2] Setting up llama-server with GPU support...")

    # Current CUDA build with Qwen3.6/Qwen3.8 Gated DeltaNet and Gemma 4 support.
    llama_build = "b12395"
    tar_url = f"https://github.com/ggml-org/llama.cpp/releases/download/{{llama_build}}/llama-{{llama_build}}-bin-ubuntu-cuda-12.8-x64.tar.gz"
    cudart_url = f"https://github.com/ggml-org/llama.cpp/releases/download/{{llama_build}}/cudart-llama-{{llama_build}}-bin-ubuntu-cuda-12.8-x64.tar.gz"

    try:
        print(f"[LLAMA] Attempting direct binary download from ggml-org release {{llama_build}}...")
        subprocess.run(["curl", "-sL", "-o", "/tmp/llama.tar.gz", tar_url], check=True)
        subprocess.run(["curl", "-sL", "-o", "/tmp/cudart.tar.gz", cudart_url], check=True)

        subprocess.run(["tar", "-xzf", "/tmp/llama.tar.gz", "-C", "/tmp/bin"], check=True)
        subprocess.run(["tar", "-xzf", "/tmp/cudart.tar.gz", "-C", "/tmp/bin"], check=True)

        # Check if llama-server exists
        for root, dirs, files in os.walk("/tmp/bin"):
            if "llama-server" in files:
                LLAMA_SERVER = os.path.join(root, "llama-server")
                os.chmod(LLAMA_SERVER, 0o755)
                print(f"[LLAMA] Successfully installed pre-built llama-server at: {{LLAMA_SERVER}}")
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
            print(f"[LLAMA] Compiled llama-server at: {{LLAMA_SERVER}}")
            return True

    raise RuntimeError("Could not establish llama-server binary!")

setup_llama_server()

# Step 3: Resolve and download Model GGUF
print(f"\\n[STEP 3] Downloading GGUF model: {{MODEL_REPO}} / {{MODEL_FILENAME}}...")
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
            single = re.search(r"-\\d{{5}}-of-\\d{{5}}\\.gguf$", lower) is None
            normalized = re.sub(r"[^a-z0-9]", "", lower)
            tiers = ("udq4kxl", "q4km", "q4k", "gguf")
            tier = next((len(tiers) - index for index, token in enumerate(tiers) if token in normalized), 0)
            return (not excluded, single, tier, -len(name))
        candidates = sorted(names, key=score, reverse=True)
        if candidates and score(candidates[0])[0] and score(candidates[0])[1]:
            chosen = candidates[0]
            print(f"[MODEL RESOLVER] Configured file unavailable; selected {{chosen}}")
            return chosen
    except Exception as exc:
        print(f"[MODEL RESOLVER] API lookup failed: {{exc}}; trying configured filename")
    return configured

MODEL_FILENAME = resolve_gguf_filename(MODEL_REPO, MODEL_FILENAME)

model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
if not os.path.exists(model_path):
    print(f"[MODEL] Fetching {{MODEL_FILENAME}} (~{{MODEL_FILESIZE}} GB)...")
    downloaded_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
        local_dir=MODEL_DIR
    )
    print(f"[MODEL] Model downloaded successfully to: {{downloaded_path}}")
else:
    print(f"[MODEL] Found cached model at: {{model_path}}")

# Step 4: Launch llama-server with Dual-GPU offload
print("\\n[STEP 4] Launching llama-server in background with GPU offloading...")
server_cmd = [
    LLAMA_SERVER,
    "-m", model_path,
    ''' + server_args_str + '''
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
            print(f"\\n[SERVER READY] Model is fully loaded and accepting requests on port 8080! (attempt {attempt+1})")
            ready = True
            break
    except Exception:
        sys.stdout.write(".")
        sys.stdout.flush()

if not ready:
    print("\\n[ERROR] Server failed to become ready within timeout. Tail log:")
    server_log.flush()
    with open("/tmp/llama_server.log") as f:
        print(f.read()[-3000:])
    sys.exit(1)

# Step 4b: Launch Early-Ping Keepalive Bridge on port 8081
print("\\n[STEP 4b] Starting Early-Ping Keepalive Bridge (Port 8081 -> 8080)...")
import base64
with open("/tmp/early_ping_bridge.py", "wb") as f:
    f.write(base64.b64decode("""''' + BRIDGE_CODE_B64 + '''"""))

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
print(f"\\n[STEP 5] Setting up secure outbound tunnel to port {tunnel_target_port}...")
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
        match = re.search(r"https://[a-zA-Z0-9-]+\\.trycloudflare\\.com", content)
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

'''
        return script_code

    def prepare_bundle(self, model_id: str) -> Path:
        BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
        script_content = self.generate_kernel_script(model_id)
        kernel_py = BUNDLE_DIR / "kernel.py"
        kernel_py.write_text(script_content, encoding="utf-8")

        owner, slug, title = self.resolve_kernel_identity(force=True, model_id=model_id)
        self.kernel_slug = f"{owner}/{slug}"
        self.kernel_title = title
        model = MODELS.get(model_id, MODELS[DEFAULT_MODEL_ID])
        is_tpu = model.get("accelerator") == "tpu"

        metadata = {
            "id": self.kernel_slug,
            "title": self.kernel_title,
            "code_file": "kernel.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": not is_tpu,
            "enable_tpu": is_tpu,
            "enable_internet": True,
            "dataset_sources": list(
                GLM_FAST_DATASETS
                if model.get("kernel_workflow") == "glm53_tpu"
                else model.get("dataset_sources") or (GLM_DATASETS if is_tpu else [])
            ),
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": []
        }
        metadata["machine_shape"] = "TpuV5E8" if is_tpu else "NvidiaTeslaT4"

        meta_json = BUNDLE_DIR / "kernel-metadata.json"
        meta_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.log(f"Prepared kernel bundle in {BUNDLE_DIR} for model '{model_id}' (target slug: {self.kernel_slug}, title: '{self.kernel_title}')")
        return BUNDLE_DIR

    def push_kernel(self, model_id: str) -> Dict[str, Any]:
        self._stop_failover.clear()
        if not self._circuit_allows_attempt():
            return {
                "success": False,
                "error": "Safety circuit breaker is cooling down",
                "status": "quota_wait",
                "retry_after_seconds": max(1, int(self.circuit_open_until - time.time())),
            }
        account = self.account_pool.active()
        if not account:
            return {"success": False, "error": "No ACTIVE Kaggle account. Add or activate an account first.", "status": "error"}
        self.upstream_api_key = self.client_api_key
        try:
            result = self._push_kernel_once(model_id)
        except Exception as exc:
            result = {"success": False, "error": str(exc), "status": "error"}
            self.set_status("error", f"Launch preparation failed: {exc}")
        if not result.get("success"):
            result["failover_scheduled"] = self.schedule_failover(
                str(result.get("error") or "Kaggle launch verification failed"),
                model_id=model_id,
                failed_account_id=account["id"],
            )
        return result

    def _push_kernel_once(self, model_id: str) -> Dict[str, Any]:
        account = self.account_pool.active()
        if not account:
            return {"success": False, "error": "No ACTIVE Kaggle account", "status": "error"}
        self._launch_account_id = account["id"]
        self.kaggle_proxy = account.get("proxy", "server")
        self.active_model_id = model_id
        self.prepare_bundle(model_id)
        self.launch_time = time.time()
        self.set_status("initializing", f"Pushing kernel for {model_id} to Kaggle...")
        if self.config_callback:
            self.config_callback()

        kaggle_bin = str(KAGGLE_VENV_BIN) if KAGGLE_VENV_BIN.exists() else "kaggle"
        cmd = [kaggle_bin, "kernels", "push", "-p", str(BUNDLE_DIR)]

        def _execute_push() -> subprocess.CompletedProcess:
            self.log(f"Executing: {' '.join(cmd)}")
            return subprocess.run(cmd, env=self.get_subprocess_env(self._launch_account_id), capture_output=True, text=True, timeout=60)

        try:
            res = _execute_push()
            clean_output = clean_kaggle_output((res.stdout or "") + "\n" + (res.stderr or ""))
            self.log(f"Kaggle push output:\n{clean_output}")

            lower_out = clean_output.lower()
            is_conflict = (
                "409" in lower_out
                or "conflict" in lower_out
                or "already in use" in lower_out
                or "does not resolve to the specified id" in lower_out
            )

            # 409 Conflict auto-resolution retry
            if res.returncode != 0 and is_conflict:
                self.log("[CONFLICT DETECTED] 409 Conflict encountered. Auto-resolving against existing kernels...")
                refreshed_kernels = self.fetch_existing_kernels(force=True)

                # Check for explicit conflicting notebook ref in output e.g. (username/slug)
                match_ref = re.search(r"\(([a-zA-Z0-9_\-]+/[a-zA-Z0-9_\-]+)\)", clean_output)
                if not match_ref:
                    match_ref = re.search(r"(?:notebook|kernel)\s+\(?([a-zA-Z0-9_\-]+/[a-zA-Z0-9_\-]+)\)?", clean_output, re.IGNORECASE)
                adopted = False
                if match_ref:
                    conflicting_ref = match_ref.group(1).strip()
                    for k in refreshed_kernels:
                        if k.get("ref") == conflicting_ref:
                            self.kernel_slug = k["ref"]
                            self.kernel_title = k["title"]
                            adopted = True
                            break
                    if not adopted and "/" in conflicting_ref:
                        self.kernel_slug = conflicting_ref
                        slug_part = conflicting_ref.split("/")[1]
                        self.kernel_title = slug_part.replace("-", " ").title()
                        adopted = True

                if not adopted:
                    # Match against preferred slugs in refreshed kernels
                    for target_slug in ["gpu-engine", "hermes-uncensored-engine", "kaggle-gpu-engine"]:
                        for k in refreshed_kernels:
                            if k.get("slug", "").strip().lower() == target_slug:
                                self.kernel_slug = k["ref"]
                                self.kernel_title = k["title"]
                                adopted = True
                                break
                        if adopted:
                            break

                if not adopted and refreshed_kernels:
                    for k in refreshed_kernels:
                        k_title = (k.get("title") or "").strip().lower()
                        if "gpu engine" in k_title or "uncensored engine" in k_title:
                            self.kernel_slug = k["ref"]
                            self.kernel_title = k["title"]
                            adopted = True
                            break

                if adopted:
                    self.log(f"[CONFLICT ADOPTED] Retrying push with adopted identity: {self.kernel_slug} ('{self.kernel_title}')")
                    meta_json = BUNDLE_DIR / "kernel-metadata.json"
                    if meta_json.exists():
                        try:
                            mdata = json.loads(meta_json.read_text(encoding="utf-8"))
                            mdata["id"] = self.kernel_slug
                            mdata["title"] = self.kernel_title
                            meta_json.write_text(json.dumps(mdata, indent=2), encoding="utf-8")
                        except Exception as e_meta:
                            logger.warning(f"Error rewriting metadata for retry: {e_meta}")

                    res = _execute_push()
                    clean_output = clean_kaggle_output((res.stdout or "") + "\n" + (res.stderr or ""))
                    self.log(f"Kaggle push retry output:\n{clean_output}")
                    lower_out = clean_output.lower()

            if res.returncode != 0 or "kernel push error:" in lower_out or "error:" in lower_out:
                err_line = clean_output
                for line in clean_output.splitlines():
                    if "kernel push error:" in line.lower() or "error:" in line.lower() or "409" in line.lower():
                        err_line = line.strip()
                        break
                self.launch_time = None
                self.set_status("error", f"Kaggle push rejected: {err_line}")
                self.log(f"[PUSH REJECTED] {err_line}")
                if self.config_callback:
                    self.config_callback()
                return {"success": False, "error": err_line, "status": self.status}

            self.set_status("downloading", "Kernel queued on Kaggle GPU! Waiting for worker node...")
            self.start_background_polling()
            return {"success": True, "output": clean_output, "status": self.status}
        except Exception as e:
            self.launch_time = None
            self.set_status("error", f"Exception pushing to Kaggle: {e}")
            self.log(f"[PUSH EXCEPTION] {e}")
            if self.config_callback:
                self.config_callback()
            return {"success": False, "error": str(e), "status": self.status}

    @staticmethod
    def _failure_account_status(reason: str) -> str:
        text = reason.lower()
        auth_terms = ("unauthorized", "forbidden", "authentication", "invalid token", "invalid api", "401", "403")
        return "INVALID_KEY" if any(term in text for term in auth_terms) else "QUOTA_EXHAUSTED"

    def schedule_failover(self, reason: str, model_id: Optional[str] = None, failed_account_id: Optional[str] = None) -> bool:
        if self._failover_thread and self._failover_thread.is_alive():
            self.log("Failover already in progress; duplicate trigger ignored.")
            return False
        failed_id = failed_account_id or self._launch_account_id or (self.account_pool.active() or {}).get("id")
        if not failed_id:
            return False
        self._failover_thread = threading.Thread(
            target=self._failover_worker,
            args=(failed_id, reason, model_id or self.active_model_id),
            daemon=True,
            name="kaggle-account-failover",
        )
        self._failover_thread.start()
        return True

    def _failover_worker(self, failed_account_id: str, reason: str, model_id: str) -> None:
        if not self._failover_lock.acquire(blocking=False):
            return
        try:
            current_id = failed_account_id
            current_reason = reason
            while current_id:
                failed = self.account_pool.get(current_id)
                failure_status = self._failure_account_status(current_reason)
                self.account_pool.mark(current_id, failure_status, current_reason, cooldown_seconds=FAILOVER_COOLDOWN_SECONDS)
                failed_name = (failed or {}).get("username", current_id)
                self.tunnel_url = ""
                self.launch_time = None
                self.set_status(
                    "cooldown",
                    f"{failed_name} marked {failure_status}. Safety cooldown {FAILOVER_COOLDOWN_SECONDS}s before rotating.",
                )
                self.log(f"[FAILOVER] {failed_name}: {current_reason}")
                if self._stop_failover.wait(FAILOVER_COOLDOWN_SECONDS):
                    self.log("[FAILOVER] Rotation cancelled by operator.")
                    return
                next_account = self.account_pool.activate_next_after(current_id)
                if not next_account:
                    self.set_status("error", "All Kaggle accounts are exhausted or invalid. Add/reset a VERIFIED · READY account.")
                    return
                self.kaggle_proxy = next_account.get("proxy", "server")
                self._cached_kernels = None
                self._cached_kernels_account_id = None
                self.log(f"[FAILOVER] Activated {next_account['username']} and relaunching {model_id}.")
                result = self._push_kernel_once(model_id)
                if result.get("success"):
                    return
                current_id = next_account["id"]
                current_reason = str(result.get("error") or "Launch verification failed")
        finally:
            self._failover_lock.release()

    def start_background_polling(self):
        self._stop_polling.set()
        if self._polling_thread and self._polling_thread.is_alive() and self._polling_thread is not threading.current_thread():
            self._polling_thread.join(timeout=2)
        self._stop_polling.clear()
        self._polling_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._polling_thread.start()

    def _poll_loop(self):
        self.log("Started background monitor for Kaggle kernel...")
        kaggle_bin = str(KAGGLE_VENV_BIN) if KAGGLE_VENV_BIN.exists() else "kaggle"
        launch_account = self.account_pool.get(self._launch_account_id)
        username = (launch_account or {}).get("username", "")
        kernel_ref = getattr(self, "kernel_slug", f"{username}/gpu-engine")
        seen_log_entries = set()

        attempts = 0
        failure_reason = ""
        intentional_idle_shutdown = False
        while not self._stop_polling.is_set() and (attempts < 120 or bool(self.tunnel_url)):
            attempts += 1
            time.sleep(10)
            if self._stop_polling.is_set():
                break

            # 1. Check kernel status
            try:
                cmd = [kaggle_bin, "kernels", "status", kernel_ref]
                res = subprocess.run(cmd, env=self.get_subprocess_env(self._launch_account_id), capture_output=True, text=True, timeout=30)
                status_text = clean_kaggle_output((res.stdout or "") + "\n" + (res.stderr or ""))
                self.log(f"[Kaggle Status]: {status_text}")

                if "running" in status_text.lower():
                    if self.status in ["initializing", "downloading"]:
                        self.set_status("loading_vram", "Kernel running on Kaggle! Loading model & tunnel...")
                elif any(w in status_text.lower() for w in ["cancel", "canceled", "cancelled"]):
                    failure_reason = f"Kernel run cancelled or session-limited: {status_text}"
                    self.tunnel_url = ""
                    self.launch_time = None
                    break
                elif "error" in status_text.lower():
                    failure_reason = f"Kernel error on Kaggle: {status_text}"
                    self.launch_time = None
                    break
                elif "complete" in status_text.lower():
                    if intentional_idle_shutdown:
                        self.set_status("idle", "Idle timeout reached; accelerator session released.")
                    else:
                        failure_reason = "Kaggle session completed or reached its session cap."
                    self.tunnel_url = ""
                    self.launch_time = None
                    break
            except Exception as e:
                logger.warning(f"Error checking kernel status: {e}")

            # 2. Stream remote execution logs from Kaggle worker node
            try:
                cmd_logs = [kaggle_bin, "kernels", "logs", kernel_ref]
                res_logs = subprocess.run(cmd_logs, env=self.get_subprocess_env(self._launch_account_id), capture_output=True, text=True, timeout=15)
                raw_data = clean_kaggle_output((res_logs.stdout or "") + "\n" + (res_logs.stderr or ""))
                if res_logs.returncode == 0 and raw_data:
                    try:
                        parsed = json.loads(raw_data)
                        items = parsed if isinstance(parsed, list) else [parsed]
                        lines = [str(item.get("data", "")).strip() for item in items if isinstance(item, dict)]
                    except Exception:
                        lines = raw_data.splitlines()
                    for line in lines:
                        line = line.strip()
                        if line and line not in seen_log_entries:
                            seen_log_entries.add(line)
                            self.log(f"[Remote] {line}")
                            if "PHASE auto-shutdown" in line and '"reason": "idle"' in line:
                                intentional_idle_shutdown = True

                            # Detect Cloudflare tunnel URL from either JSON logs or plain CLI output.
                            match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
                            if match and not self.tunnel_url:
                                found_url = match.group(0)
                                self.log(f"[TUNNEL DETECTED] Cloudflare: {found_url}")
                                self.set_tunnel_url(found_url)
                                self.set_status("ready", f"Live Tunnel: {found_url}")
            except Exception as e_log:
                logger.debug(f"Logs poll exception: {e_log}")

            if self.tunnel_url and self.status == "ready":
                time.sleep(20)

        if not self._stop_polling.is_set() and not self.tunnel_url and not intentional_idle_shutdown:
            failure_reason = failure_reason or "Launch verification timed out before a live tunnel was detected."
            self.set_status("error", failure_reason)
            self.schedule_failover(failure_reason, model_id=self.active_model_id, failed_account_id=self._launch_account_id)

    def get_elapsed_seconds(self) -> int:
        if self.launch_time:
            return max(0, int(time.time() - self.launch_time))
        return 0

    def stop_kernel(self) -> Dict[str, Any]:
        """
        Stops the active Kaggle kernel to conserve GPU quota,
        terminates background polling, tears down tunnel, and resets status to idle.
        """
        self.log("Received Stop Engine request. Initiating shutdown...")
        self._stop_failover.set()
        # 1. Stop background polling thread immediately
        self._stop_polling.set()
        if self._polling_thread and self._polling_thread.is_alive():
            self._polling_thread.join(timeout=2)

        # 2. Reset active tunnel and elapsed time
        self.tunnel_url = ""
        self.launch_time = None
        self.set_status("idle", "Engine stopped by operator. GPU quota conserved.")
        if self.config_callback:
            self.config_callback()

        # 3. Stop/cancel kernel on Kaggle
        kaggle_bin = str(KAGGLE_VENV_BIN) if KAGGLE_VENV_BIN.exists() else "kaggle"
        launch_account = self.account_pool.get(self._launch_account_id) or self.account_pool.active()
        username = (launch_account or {}).get("username", "")
        kernel_ref = getattr(self, "kernel_slug", f"{username}/gpu-engine")

        stopped_via_cli = False
        for cmd_verb in ["stop", "cancel"]:
            try:
                res = subprocess.run(
                    [kaggle_bin, "kernels", cmd_verb, kernel_ref],
                    env=self.get_subprocess_env((launch_account or {}).get("id")),
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if res.returncode == 0:
                    self.log(f"Kaggle kernel stopped via CLI '{cmd_verb}': {clean_kaggle_output(res.stdout)}")
                    stopped_via_cli = True
                    break
            except Exception:
                pass

        if not stopped_via_cli:
            try:
                self.log("Pushing cancellation script to release Kaggle GPU resource...")
                stop_dir = BASE_DIR / "stop_bundle"
                stop_dir.mkdir(parents=True, exist_ok=True)

                stop_script = (
                    "#!/usr/bin/env python3\n"
                    "# Autonomous Uncensored Engine - Shutdown Sentinel\n"
                    "import sys\n"
                    "print('[SHUTDOWN] Engine stop requested by operator. Releasing GPU.')\n"
                    "sys.exit(0)\n"
                )
                (stop_dir / "kernel.py").write_text(stop_script, encoding="utf-8")

                metadata = {
                    "id": kernel_ref,
                    "title": getattr(self, "kernel_title", "GPU Engine"),
                    "code_file": "kernel.py",
                    "language": "python",
                    "kernel_type": "script",
                    "is_private": True,
                    "enable_gpu": False,
                    "enable_tpu": False,
                    "enable_internet": False,
                    "dataset_sources": [],
                    "competition_sources": [],
                    "kernel_sources": [],
                    "model_sources": []
                }
                (stop_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

                cmd = [kaggle_bin, "kernels", "push", "-p", str(stop_dir)]
                res = subprocess.run(cmd, env=self.get_subprocess_env((launch_account or {}).get("id")), capture_output=True, text=True, timeout=30)
                if res.returncode == 0:
                    self.log("Kaggle run preempted & cancelled. Active GPU session terminated.")
                else:
                    self.log(f"Notice: Push stop script returned code {res.returncode}: {clean_kaggle_output(res.stderr)}")
            except Exception as e:
                self.log(f"Stop kernel push exception: {e}")

        if self.config_callback:
            self.config_callback()

        return {"success": True, "status": self.status, "message": "Engine stopped and tunnel torn down."}

    def get_state(self) -> Dict[str, Any]:
        effective_status = "ready" if self.tunnel_url else self.status
        effective_detail = f"Live Tunnel: {self.tunnel_url}" if self.tunnel_url else self.status_detail
        return {
            "status": effective_status,
            "status_detail": effective_detail,
            "active_model_id": self.active_model_id,
            "active_model": MODELS.get(self.active_model_id, MODELS[DEFAULT_MODEL_ID]),
            "tunnel_url": self.tunnel_url,
            "tunnel_type": self.tunnel_type,
            "elapsed_seconds": self.get_elapsed_seconds(),
            "launch_timestamp": self.launch_time,
            "kernel_slug": self.kernel_slug,
            "kernel_title": getattr(self, "kernel_title", "GPU Engine"),
            "recent_logs": self.recent_logs[-50:],
            "available_models": list(MODELS.values()),
            "kaggle_accounts": self.account_pool.list_public(),
            "kaggle_account_counts": self.account_pool.counts(),
            "active_kaggle_account_id": (self.account_pool.active() or {}).get("id"),
            "queued_requests": self.queued_requests,
            "active_requests": self.active_requests,
            "last_inference_activity": self.last_inference_activity,
            "idle_timeout_min": float(getattr(self, "idle_timeout_min", IDLE_TIMEOUT_MIN_DEFAULT)),
            "max_runtime_min": float(getattr(self, "max_runtime_min", MAX_RUNTIME_MIN_DEFAULT))
        }

    # ------------------------------------------------------------------
    # Advanced multi-session lifecycle (definitions below supersede the
    # original single-session methods above while retaining their helpers).

    def _session_estimate(self, session: KernelSession) -> float:
        return self.account_pool.boot_estimate(
            session.account_id,
            session.accelerator,
            BOOT_BASELINE_SECONDS[session.accelerator],
        )

    def _session_by_id(self, session_id: Optional[str]) -> Optional[KernelSession]:
        with self._state_lock:
            return self.sessions.get(session_id or "")

    def _active_session(self) -> Optional[KernelSession]:
        return self._session_by_id(self.active_session_id)

    def _warming_session(self) -> Optional[KernelSession]:
        return self._session_by_id(self.warming_session_id)

    def warmup_snapshot(self) -> Optional[Dict[str, Any]]:
        with self._state_lock:
            candidates = []
            if self.warming_session_id and self.warming_session_id in self.sessions:
                candidates.append(self.sessions[self.warming_session_id])
            active = self.sessions.get(self.active_session_id or "")
            if active and active.status not in {"ready", "draining", "stopped"}:
                candidates.append(active)
            candidates.extend(
                session for sid, session in self.sessions.items()
                if sid in self.test_session_ids and session.status not in {"stopped", "failed"}
            )
            if not candidates:
                return None
            session = candidates[0]
        return session.public(self.account_pool.get(session.account_id), self._session_estimate(session))

    def recover_persisted_session(self) -> bool:
        """Reattach lifecycle monitoring after a service restart without changing the live tunnel."""
        if not self.tunnel_url or self.active_session_id:
            return False
        account = self.account_pool.active()
        if not account:
            return False
        model = self.model_spec(self.active_model_id)
        accelerator = str(self.accelerator_mode)
        workflow = model.get("kernel_workflow")
        session_id = str(account.get("session_id") or uuid.uuid4().hex[:12])
        launch_time = float(account.get("session_started_at") or self.launch_time or time.time())
        slug = "qwen38-tpu-serve" if workflow == "qwen38_tpu" else "glm53-tpu-serve" if accelerator == "tpu" else "gpu-engine"
        session = KernelSession(
            id=session_id,
            role="active",
            account_id=account["id"],
            model_id=self.active_model_id,
            accelerator=accelerator,
            launch_time=launch_time,
            kernel_slug=f"{account['username']}/{slug}",
            kernel_title="Qwen38 TPU Serve" if workflow == "qwen38_tpu" else "GLM53 TPU Serve" if accelerator == "tpu" else "GPU Engine",
            tunnel_url=self.tunnel_url.rstrip("/"),
            status="ready",
            ready_time=time.time(),
            status_detail="Recovered from persisted GPU Hub state after service restart.",
        )
        with self._state_lock:
            self.sessions[session.id] = session
            self.active_session_id = session.id
        if not account.get("session_id"):
            self.account_pool.begin_session(account["id"], session.id, accelerator, self.active_model_id, launch_time)
        session.monitor_thread = threading.Thread(
            target=self._monitor_session,
            args=(session,),
            daemon=True,
            name=f"kaggle-session-{session.id}",
        )
        session.monitor_thread.start()
        self.log(f"Recovered lifecycle monitoring for {session.kernel_slug} at {session.tunnel_url}")
        return True

    def warmup_headers(self) -> Dict[str, str]:
        snapshot = self.warmup_snapshot()
        if not snapshot:
            return {}
        remaining = int(max(1, snapshot["estimated_remaining_seconds"]))
        return {
            "Retry-After": str(min(remaining, 60)),
            "X-GPUHub-Warmup-Remaining": str(remaining),
            "X-GPUHub-Warmup-Elapsed": str(int(snapshot["elapsed_seconds"])),
            "X-GPUHub-Warmup-Max": str(MAX_BOOT_SECONDS),
            "X-GPUHub-Warmup-Account": snapshot.get("username", ""),
        }

    def register_session_tunnel(self, session_id: Optional[str], url: str) -> bool:
        with self._state_lock:
            session = self.sessions.get(session_id or "")
            if not session and self.active_session_id:
                session = self.sessions.get(self.active_session_id)
            if not session:
                return False
            session.tunnel_url = url.rstrip("/")
            session.status = "health_check"
            session.status_detail = "Tunnel discovered; waiting for /v1/models readiness check."
        self.log(f"[{session.role.upper()}] Tunnel discovered for {session.id}: {session.tunnel_url}")
        return True

    def handle_session_shutdown(self, session_id: Optional[str], reason: str, detail: str = "") -> None:
        session = self._session_by_id(session_id) or self._active_session()
        if not session:
            return
        session.last_error = f"{reason}: {detail}".strip(": ")
        session.stop_event.set()
        self.account_pool.end_session(session.account_id, session.id)
        if reason == "idle" and session.id == self.active_session_id:
            with self._state_lock:
                self.active_session_id = None
                self.tunnel_url = ""
                self.launch_time = None
            self.set_status("idle", f"Idle timeout: {detail or 'accelerator released.'}")
            return
        self._handle_session_failure(session, session.last_error or "Kaggle session ended")

    def _prepare_session_bundle(self, session: KernelSession) -> Path:
        bundle_dir = RUNTIME_BUNDLE_DIR / session.id
        bundle_dir.mkdir(parents=True, exist_ok=True)
        source = self.generate_kernel_script(session.model_id, session.id, session.role, session.accelerator)
        (bundle_dir / "kernel.py").write_text(source, encoding="utf-8")
        owner, slug, title = self.resolve_kernel_identity(
            force=True,
            model_id=session.model_id,
            account_id=session.account_id,
            accelerator=session.accelerator,
        )
        session.kernel_slug = f"{owner}/{slug}"
        session.kernel_title = title
        session.bundle_dir = bundle_dir
        model = self.model_spec(session.model_id, session.accelerator)
        is_tpu = model.get("accelerator") == "tpu"
        metadata = {
            "id": session.kernel_slug,
            "title": session.kernel_title,
            "code_file": "kernel.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": not is_tpu,
            "enable_tpu": is_tpu,
            "enable_internet": True,
            "dataset_sources": list(model.get("dataset_sources") or (GLM_DATASETS if is_tpu else [])),
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        metadata["machine_shape"] = "TpuV5E8" if is_tpu else "NvidiaTeslaT4"
        (bundle_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return bundle_dir

    def _adopt_conflicting_kernel(self, session: KernelSession, output: str) -> bool:
        existing = self.fetch_existing_kernels(force=True, account_id=session.account_id)
        model = self.model_spec(session.model_id, session.accelerator)
        workflow = model.get("kernel_workflow")
        model_id = normalize_model_id(session.model_id)
        gpu_model_slug = f"gpu-hub-{re.sub(r'[^a-z0-9]+', '-', model_id.lower()).strip('-')}"[:48].rstrip("-")
        preferred = (
            ["qwen38-tpu-serve", "gpu-hub-qwen38"] if workflow == "qwen38_tpu"
            else ["glm53-tpu-serve", "gpu-hub-glm53"] if workflow == "glm53_tpu"
            else [gpu_model_slug, "gpu-engine", "hermes-uncensored-engine", "kaggle-gpu-engine"]
            if model_id == DEFAULT_MODEL_ID else [gpu_model_slug]
        )
        explicit = re.search(r"\(?([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)\)?", output)
        candidate = None
        if explicit:
            explicit_ref = explicit.group(1)
            explicit_slug = explicit_ref.split("/", 1)[-1].lower()
            if explicit_slug in preferred or explicit_ref == session.kernel_slug:
                candidate = next((item for item in existing if item.get("ref") == explicit_ref), None)
        if not candidate:
            for slug in preferred:
                candidate = next((item for item in existing if str(item.get("slug", "")).lower() == slug), None)
                if candidate:
                    break
        if not candidate:
            phrases = (
                ["qwen38", "qwen3.8"] if workflow == "qwen38_tpu"
                else ["glm53", "glm-5.3"] if workflow == "glm53_tpu"
                else ["gpu engine", "uncensored engine"] if model_id == DEFAULT_MODEL_ID else []
            )
            candidate = next(
                (item for item in existing if any(phrase in str(item.get("title", "")).lower() for phrase in phrases)),
                None,
            )
        if not candidate:
            return False
        session.kernel_slug = candidate.get("ref") or f"{candidate.get('owner')}/{candidate.get('slug')}"
        session.kernel_title = candidate.get("title") or session.kernel_title
        metadata_path = session.bundle_dir / "kernel-metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update({"id": session.kernel_slug, "title": session.kernel_title})
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.log(f"[409 ADOPT] Retrying with existing kernel {session.kernel_slug} ({session.kernel_title}).")
        return True

    def _apply_tpu_dataset_fallback(self, session: KernelSession, output: str) -> bool:
        model = self.model_spec(session.model_id, session.accelerator)
        if model.get("accelerator") != "tpu":
            return False
        lowered = output.lower()
        if not any(term in lowered for term in ("not valid dataset", "dataset source", "dataset not found", "private dataset")):
            return False
        workflow = model.get("kernel_workflow")
        if workflow == "glm53_tpu":
            fallback = GLM_FALLBACK_DATASETS
        elif workflow == "qwen38_tpu":
            fallback = [QWEN_TPU_DATASETS[0]]  # env/cache dataset is optional; weights are mandatory
        else:
            return False
        metadata_path = session.bundle_dir / "kernel-metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("dataset_sources") == fallback:
            return False
        metadata["dataset_sources"] = fallback
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.log(f"[TPU DATASET FALLBACK] Retrying {session.kernel_slug} with: {', '.join(fallback)}")
        return True

    @staticmethod
    def _is_no_tpu_failure(reason: str) -> bool:
        lowered = str(reason or "").lower()
        return any(term in lowered for term in (
            "this session has no working tpu",
            "jax sees 1 cpu device",
            '"step": "no-tpu"',
            "step=no-tpu",
        ))

    def _delete_kernel_for_fresh_tpu(self, account_id: str, kernel_slug: str) -> None:
        if not kernel_slug:
            return
        result = self._run_kaggle(account_id, ["kernels", "delete", kernel_slug, "-y"], timeout=45)
        output = clean_kaggle_output((result.stdout or "") + "\n" + (result.stderr or ""))
        if result.returncode == 0:
            self.log(f"[TPU FRESH VM] Deleted prior kernel {kernel_slug} before re-creation.")
        elif any(term in output.lower() for term in ("404", "not found", "does not exist")) or (
            "403" in output and "deletekernel" in output.lower()
        ):
            self.log(f"[TPU FRESH VM] No prior kernel existed at {kernel_slug}; continuing.")
        else:
            raise RuntimeError(f"Kaggle kernel delete failed for {kernel_slug}: {kaggle_error_line(output)}")

    def _set_tpu_retry_slug(self, session: KernelSession, retry_number: int) -> None:
        owner, current = session.kernel_slug.split("/", 1)
        base = re.sub(r"-r\d+$", "", current)
        retry_slug = f"{base}-r{retry_number}"
        session.kernel_slug = f"{owner}/{retry_slug}"
        session.kernel_title = re.sub(r"\s+R\d+$", "", session.kernel_title or "TPU Serve") + f" R{retry_number}"
        metadata_path = session.bundle_dir / "kernel-metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update({"id": session.kernel_slug, "title": session.kernel_title})
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def _push_existing_session_bundle(self, session: KernelSession) -> Tuple[bool, str]:
        result = self._run_kaggle(
            session.account_id,
            ["kernels", "push", "-p", str(session.bundle_dir)],
            timeout=90,
        )
        output = clean_kaggle_output((result.stdout or "") + "\n" + (result.stderr or ""))
        lowered = output.lower()
        dataset_problem = any(term in lowered for term in ("not valid dataset", "dataset source", "dataset not found", "private dataset"))
        if dataset_problem and self._apply_tpu_dataset_fallback(session, output):
            result = self._run_kaggle(
                session.account_id,
                ["kernels", "push", "-p", str(session.bundle_dir)],
                timeout=90,
            )
            output = clean_kaggle_output((result.stdout or "") + "\n" + (result.stderr or ""))
            lowered = output.lower()
            dataset_problem = any(term in lowered for term in ("not valid dataset", "dataset source", "dataset not found", "private dataset"))
        failed = result.returncode != 0 or dataset_problem or "kernel push error:" in lowered or ("error:" in lowered and "successfully pushed" not in lowered)
        return not failed, output

    def _recreate_tpu_session(self, session: KernelSession, reason: str) -> bool:
        """Recover Kaggle's CPU-only TPU allocation glitch without penalizing the account."""
        session.tpu_recreate_attempts += 1
        retry_number = session.tpu_recreate_attempts
        if retry_number > 4:
            session.status = "failed"
            session.last_error = reason[:800]
            session.status_detail = "Kaggle failed to attach a TPU after 4 automatic re-creations. Retry manually when TPU capacity improves."
            session.stop_event.set()
            self.account_pool.end_session(session.account_id, session.id)
            account = self.account_pool.get(session.account_id)
            if account and account.get("verified_at"):
                status = "ACTIVE" if (self.account_pool.active() or {}).get("id") == session.account_id else "VERIFIED_READY"
                self.account_pool.mark(session.account_id, status, session.status_detail)
            with self._state_lock:
                if self.active_session_id == session.id:
                    self.active_session_id = None
                    self.tunnel_url = ""
                if self.warming_session_id == session.id:
                    self.warming_session_id = None
                self.launch_time = None
            self.consecutive_launch_failures = 0
            self.circuit_open_until = 0.0
            self.set_status("error", session.status_detail)
            self.log(f"[TPU AUTO-RECOVERY EXHAUSTED] {reason}")
            return False

        session.status = "recreating_tpu"
        session.status_detail = f"Kaggle started VM without TPU — auto-recreating TPU session (retry {retry_number}/4)..."
        self.status = "warming"
        self.status_detail = session.status_detail
        self.log(f"[TPU AUTO-RECOVERY {retry_number}/4] {reason}")
        old_slug = session.kernel_slug
        try:
            self._delete_kernel_for_fresh_tpu(session.account_id, old_slug)
            self._set_tpu_retry_slug(session, retry_number)
            if session.stop_event.wait(10):
                return False
            ok, output = self._push_existing_session_bundle(session)
        except Exception as exc:
            ok, output = False, str(exc)
        if not ok:
            detail = f"TPU re-creation push failed without penalizing the account: {kaggle_error_line(output)}"
            session.status = "failed"
            session.last_error = detail[:800]
            session.status_detail = detail[:500]
            session.stop_event.set()
            self.account_pool.end_session(session.account_id, session.id)
            account = self.account_pool.get(session.account_id)
            if account and account.get("verified_at"):
                status = "ACTIVE" if (self.account_pool.active() or {}).get("id") == session.account_id else "VERIFIED_READY"
                self.account_pool.mark(session.account_id, status, detail)
            with self._state_lock:
                if self.active_session_id == session.id:
                    self.active_session_id = None
                    self.tunnel_url = ""
                if self.warming_session_id == session.id:
                    self.warming_session_id = None
                self.launch_time = None
            self.consecutive_launch_failures = 0
            self.circuit_open_until = 0.0
            self.set_status("error", detail)
            self.log(f"[TPU AUTO-RECOVERY STOPPED] {detail}")
            return False
        session.tunnel_url = ""
        session.health_failures = 0
        session.status = "queued"
        session.status_detail = f"Fresh TPU VM requested successfully (retry {retry_number}/4); waiting for TPU provisioning."
        self.log(f"[TPU AUTO-RECOVERY {retry_number}/4] Re-pushed as {session.kernel_slug}.")
        return True

    def _launch_session(
        self,
        account_id: str,
        model_id: str,
        role: str,
        trigger_reason: str = "",
        accelerator_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        account = self.account_pool.get(account_id)
        if not account:
            return {"success": False, "error": "Kaggle account not found"}
        if not account.get("verified_at"):
            account = self.verify_kaggle_account(account_id)
        if not account.get("verified_at"):
            return {"success": False, "error": account.get("verification_error") or "INVALID_KEY"}
        model_id = normalize_model_id(model_id)
        model = MODELS.get(model_id)
        if not model:
            return {"success": False, "error": f"Unknown model: {model_id}"}
        selected_accelerator = str(accelerator_override or self.accelerator_mode)
        if not self.model_is_compatible(model_id, selected_accelerator):
            required = "TPU v5e-8" if "tpu" in model.get("supported_accelerators", []) else "Dual T4 GPU"
            return {"success": False, "error": f"{model['name']} requires {required}; switch Hardware Accelerator Mode"}
        if model.get("launch_supported") is False:
            return {
                "success": False,
                "error": "This model is visible as a compatibility profile, but no verified Kaggle TPU serving recipe is bundled yet",
            }
        accelerator = selected_accelerator
        if self.account_pool.remaining_seconds(account_id, accelerator) <= 0:
            self.account_pool.mark_quota_exhausted(account_id, accelerator, "Locally tracked weekly quota reached")
            return {"success": False, "error": f"{accelerator.upper()} weekly quota exhausted"}
        with self._state_lock:
            for existing in self.sessions.values():
                if existing.account_id == account_id and existing.status not in {"failed", "stopped"}:
                    return {"success": False, "error": "This Kaggle account already has a running or warming session"}
            session = KernelSession(
                id=uuid.uuid4().hex[:12],
                role=role,
                account_id=account_id,
                model_id=model_id,
                accelerator=accelerator,
                trigger_reason=trigger_reason,
            )
            self.sessions[session.id] = session
            if role == "prewarm" or (role == "active" and not self.active_session_id):
                self.warming_session_id = session.id
            elif role == "test":
                self.test_session_ids.add(session.id)
            if role == "active" and not self.active_session_id:
                self.launch_time = session.launch_time
        if role == "prewarm":
            self.account_pool.mark(account_id, "PREWARMING")
        elif role == "test":
            self.account_pool.mark(account_id, "TESTING_BOOT")
        try:
            bundle_dir = self._prepare_session_bundle(session)
        except Exception as exc:
            self._handle_session_failure(session, f"Bundle preparation failed: {exc}")
            return {"success": False, "error": str(exc), "session_id": session.id}
        command = ["kernels", "push", "-p", str(bundle_dir)]
        session.status = "pushing"
        session.status_detail = f"Pushing {session.kernel_slug} with {account['username']}"
        self.log(f"[{role.upper()}] Executing kaggle {' '.join(command)}")

        if accelerator == "tpu":
            try:
                session.status_detail = f"Deleting stale TPU kernel {session.kernel_slug} before requesting a fresh VM."
                self._delete_kernel_for_fresh_tpu(account_id, session.kernel_slug)
                time.sleep(5)
            except Exception as exc:
                self._handle_session_failure(session, f"Fresh TPU kernel cleanup failed: {exc}")
                return {"success": False, "error": str(exc), "session_id": session.id}

        def execute_push():
            return self._run_kaggle(account_id, command, timeout=90)

        try:
            result = execute_push()
        except Exception as exc:
            self._handle_session_failure(session, f"Kaggle push exception: {exc}")
            return {"success": False, "error": str(exc), "session_id": session.id}
        output = clean_kaggle_output((result.stdout or "") + "\n" + (result.stderr or ""))
        lowered = output.lower()
        if result.returncode != 0 and any(term in lowered for term in ("409", "conflict", "already in use", "does not resolve to the specified id")):
            if accelerator == "tpu":
                self._delete_kernel_for_fresh_tpu(account_id, session.kernel_slug)
                time.sleep(5)
                result = execute_push()
                output = clean_kaggle_output((result.stdout or "") + "\n" + (result.stderr or ""))
                lowered = output.lower()
            elif self._adopt_conflicting_kernel(session, output):
                result = execute_push()
                output = clean_kaggle_output((result.stdout or "") + "\n" + (result.stderr or ""))
                lowered = output.lower()
        dataset_problem = any(term in lowered for term in ("not valid dataset", "dataset source", "dataset not found", "private dataset"))
        if dataset_problem and self._apply_tpu_dataset_fallback(session, output):
            result = execute_push()
            output = clean_kaggle_output((result.stdout or "") + "\n" + (result.stderr or ""))
            lowered = output.lower()
            dataset_problem = any(term in lowered for term in ("not valid dataset", "dataset source", "dataset not found", "private dataset"))
        failed = result.returncode != 0 or "kernel push error:" in lowered or dataset_problem or ("error:" in lowered and "successfully pushed" not in lowered)
        if failed:
            exact_error = kaggle_error_line(output)
            self._handle_session_failure(session, f"Kaggle push rejected: {exact_error}")
            return {"success": False, "error": exact_error, "session_id": session.id}
        session.status = "queued"
        session.status_detail = "Kaggle accepted the kernel; waiting for accelerator provisioning."
        self.account_pool.begin_session(account_id, session.id, accelerator, model_id, session.launch_time)
        session.monitor_thread = threading.Thread(
            target=self._monitor_session,
            args=(session,),
            daemon=True,
            name=f"kaggle-session-{session.id}",
        )
        session.monitor_thread.start()
        return {"success": True, "session_id": session.id, "status": session.status, "kernel_slug": session.kernel_slug}

    def _read_session_logs(self, session: KernelSession, seen: set[str]) -> Tuple[List[str], bool, str]:
        try:
            result = self._run_kaggle(session.account_id, ["kernels", "logs", session.kernel_slug], timeout=20)
        except Exception:
            return [], False, ""
        raw = clean_kaggle_output((result.stdout or "") + "\n" + (result.stderr or ""))
        if result.returncode != 0 or not raw:
            # Kaggle CLI 2.2.4 frequently returns an empty snapshot for a live
            # kernel even though `logs -f` has the complete stream. Sample the
            # follow stream for a few seconds and parse the partial output.
            try:
                follow = self._run_kaggle(
                    session.account_id,
                    ["kernels", "logs", "-f", session.kernel_slug],
                    timeout=5,
                )
                raw = clean_kaggle_output((follow.stdout or "") + "\n" + (follow.stderr or ""))
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout or b""
                stderr = exc.stderr or b""
                if isinstance(stdout, bytes):
                    stdout = stdout.decode(errors="replace")
                if isinstance(stderr, bytes):
                    stderr = stderr.decode(errors="replace")
                raw = clean_kaggle_output(str(stdout) + "\n" + str(stderr))
            except Exception:
                raw = ""
        if not raw:
            return [], False, ""
        try:
            parsed = json.loads(raw)
            items = parsed if isinstance(parsed, list) else [parsed]
            lines = [str(item.get("data", "")).strip() for item in items if isinstance(item, dict)]
        except Exception:
            lines = raw.splitlines()
        fresh, intentional_idle, failure_detail = [], False, ""
        for line in lines:
            line = line.strip()
            if not line or line in seen:
                continue
            seen.add(line)
            fresh.append(line)
            if ("PHASE auto-shutdown" in line and '"reason": "idle"' in line) or "AUTO-SHUTDOWN] idle" in line:
                intentional_idle = True
            match = re.search(r"PHASE failed\s+(\{.*\})", line)
            if match:
                try:
                    payload = json.loads(match.group(1))
                    failure_detail = str(payload.get("tail") or payload.get("cause") or payload.get("step") or line)
                except Exception:
                    failure_detail = line
            elif "this session has no working tpu" in line.lower():
                failure_detail = line
        return fresh, intentional_idle, failure_detail

    def _check_session_health(self, session: KernelSession) -> bool:
        if not session.tunnel_url:
            return False
        client = requests.Session()
        client.trust_env = False
        headers = {"Authorization": f"Bearer {self.upstream_api_key or self.client_api_key}"}
        for path in ("/health", "/v1/models"):
            try:
                response = client.get(f"{session.tunnel_url}{path}", headers=headers, timeout=10)
                if response.status_code == 200:
                    if path == "/health":
                        return True
                    payload = response.json()
                    return isinstance(payload.get("data"), list) or isinstance(payload.get("models"), list)
            except Exception:
                continue
        return False

    def _monitor_session(self, session: KernelSession) -> None:
        seen: set[str] = set()
        intentional_idle = False
        ready_announced = False
        attempts = 0
        while not session.stop_event.wait(10):
            attempts += 1
            if session.kaggle_poll_frozen:
                if session.tunnel_url and self._check_session_health(session):
                    session.health_failures = 0
                    continue
                session.health_failures += 1
                if session.health_failures >= 3:
                    self._handle_session_failure(session, "Frozen Kaggle session tunnel became unreachable")
                    return
                continue
            try:
                result = self._run_kaggle(session.account_id, ["kernels", "status", session.kernel_slug], timeout=30)
                status_text = clean_kaggle_output((result.stdout or "") + "\n" + (result.stderr or ""))
            except Exception as exc:
                status_text = str(exc)
            lowered = status_text.lower()
            if any(term in lowered for term in (
                "error", "cancelacknowledged", "cancel_acknowledged", "cancel acknowledged", "cancelled", "canceled",
            )):
                self._handle_session_failure(session, f"Kaggle session failed: {status_text}")
                return
            if session.status == "ready" and attempts % 2 == 0:
                if self._check_session_health(session):
                    if session.health_failures:
                        self.log(f"[LIVENESS] Upstream recovered at {session.tunnel_url}.")
                    session.health_failures = 0
                    session.unhealthy_since = 0.0
                    if session.id == self.active_session_id and self.status != "ready":
                        self.set_status("ready", f"Live Tunnel: {session.tunnel_url}")
                else:
                    session.health_failures += 1
                    session.unhealthy_since = session.unhealthy_since or time.time()
                    if session.health_failures >= 2:
                        session.status_detail = "Active upstream health check failed; self-healing tunnel/server in progress."
                        if session.id == self.active_session_id:
                            self.set_status("recovering", session.status_detail)
            if "running" in lowered and session.status in {"queued", "pushing", "initializing"}:
                session.status = "booting"
                session.status_detail = "Accelerator allocated; loading model and opening Tunnel B."
            lines, idle_marker, remote_failure = self._read_session_logs(session, seen)
            intentional_idle = intentional_idle or idle_marker
            for line in lines:
                self.log(f"[{session.role.upper()}:{session.id}] {line}")
                lowered_line = line.lower()
                if any(term in lowered_line for term in ("step 2/6", "loading the weights", "loading model", "weights on the chips", "downloading gguf")):
                    session.status = "loading_model"
                    session.status_detail = "Loading model weights onto the accelerator."
                match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
                if match and match.group(0) != session.tunnel_url:
                    old_url = session.tunnel_url
                    session.tunnel_url = match.group(0)
                    session.status = "health_check"
                    session.status_detail = "New or rotated tunnel discovered; probing health."
                    if session.id == self.active_session_id:
                        with self._state_lock:
                            self.tunnel_url = session.tunnel_url
                        self.log(f"[LIVENESS] Tunnel rotated {old_url or '(none)'} -> {session.tunnel_url}")
                        if self.config_callback:
                            self.config_callback()
            if remote_failure:
                if session.accelerator == "tpu" and self._is_no_tpu_failure(remote_failure):
                    if self._recreate_tpu_session(session, remote_failure):
                        seen.clear()
                        attempts = 0
                        intentional_idle = False
                        ready_announced = False
                        continue
                    return
                self._handle_session_failure(session, remote_failure)
                return
            if session.tunnel_url and session.status != "ready" and self._check_session_health(session):
                self._session_ready(session)
                ready_announced = True
            elapsed = time.time() - session.launch_time
            if not ready_announced and elapsed > MAX_BOOT_SECONDS:
                session.status = "eta_warning"
                session.status_detail = "Warm-up exceeded the 15 minute ETA; Kaggle is still provisioning."
            if "complete" in lowered:
                if intentional_idle and session.id == self.active_session_id:
                    self.handle_session_shutdown(session.id, "idle", "Kernel idle timer ended the session")
                elif not session.stop_event.is_set():
                    self._handle_session_failure(session, "Kaggle session completed or reached its 9-hour cap")
                return
            if session.health_failures >= 2 and session.unhealthy_since and time.time() - session.unhealthy_since >= 60:
                if any(term in lowered for term in (
                    "error", "complete", "cancelled", "canceled", "cancelacknowledged", "cancel_acknowledged", "cancel acknowledged",
                )):
                    self._handle_session_failure(session, f"Active upstream stayed unreachable for 60 seconds; Kaggle status: {status_text}")
                    return
            if not ready_announced and attempts >= 180:
                self._handle_session_failure(session, "Kernel did not become healthy within 30 minutes")
                return

    def _session_ready(self, session: KernelSession) -> None:
        if session.status == "ready":
            return
        session.status = "ready"
        session.ready_time = time.time()
        session.status_detail = "Healthy /v1/models response received."
        self.consecutive_launch_failures = 0
        self.circuit_open_until = 0.0
        self.circuit_backoff_seconds = 60
        self.prewarm_warning = ""
        boot_seconds = session.ready_time - session.launch_time
        self.account_pool.record_boot(session.account_id, session.accelerator, boot_seconds)
        self.log(f"[{session.role.upper()}] {session.id} ready in {boot_seconds:.1f}s at {session.tunnel_url}")
        if session.role == "prewarm":
            self.account_pool.mark(session.account_id, "HOT_STANDBY")
            self._hot_swap(session)
        elif session.role == "test":
            threading.Thread(target=self._finish_boot_test, args=(session,), daemon=True).start()
        else:
            self._promote_active(session)

    def _promote_active(self, session: KernelSession) -> None:
        self.account_pool.activate(session.account_id)
        with self._state_lock:
            session.role = "active"
            self.active_session_id = session.id
            if self.warming_session_id == session.id:
                self.warming_session_id = None
            self.tunnel_url = session.tunnel_url
            self.launch_time = session.launch_time
            self.kernel_slug = session.kernel_slug
            self.kernel_title = session.kernel_title
            self.active_model_id = session.model_id
        self.set_status("ready", f"Live Tunnel: {session.tunnel_url}")

    def _hot_swap(self, session: KernelSession) -> None:
        with self._state_lock:
            old = self.sessions.get(self.active_session_id or "")
            old_url = self.tunnel_url
            if old:
                old.role = "draining"
            session.role = "active"
            self.active_session_id = session.id
            self.warming_session_id = None
            self.tunnel_url = session.tunnel_url
            self.launch_time = session.launch_time
            self.kernel_slug = session.kernel_slug
            self.kernel_title = session.kernel_title
            self.active_model_id = session.model_id
        self.account_pool.activate(session.account_id)
        self.set_status("ready", f"Hot-swapped to {session.tunnel_url}; old upstream draining.")
        self.log(f"[HOT-SWAP] Tunnel B is active. New requests now use {session.tunnel_url}.")
        if old:
            threading.Thread(
                target=self._drain_and_stop_old,
                args=(old, old_url, session.trigger_reason),
                daemon=True,
                name=f"drain-{old.id}",
            ).start()

    def _drain_and_stop_old(self, session: KernelSession, upstream_url: str, trigger_reason: str) -> None:
        deadline = time.time() + 600
        while time.time() < deadline:
            with self._state_lock:
                active = self._upstream_active_counts.get(upstream_url, 0)
            if active <= 0:
                break
            time.sleep(1)
        self._stop_session(session, "hot-swap complete")
        account = self.account_pool.get(session.account_id)
        if account and account.get("status") != "QUOTA_EXHAUSTED":
            if trigger_reason == "weekly_quota":
                self.account_pool.mark_quota_exhausted(session.account_id, session.accelerator, "Rotated at weekly quota threshold")
            elif account.get("verified_at"):
                self.account_pool.mark(session.account_id, "VERIFIED_READY")

    def _finish_boot_test(self, session: KernelSession) -> None:
        self._stop_session(session, "boot measurement complete")
        account = self.account_pool.get(session.account_id)
        if account and account.get("verified_at"):
            status = "ACTIVE" if (self.account_pool.active() or {}).get("id") == session.account_id else "VERIFIED_READY"
            self.account_pool.mark(session.account_id, status)
        with self._state_lock:
            self.test_session_ids.discard(session.id)

    def _classify_account_failure(self, reason: str) -> str:
        lowered = reason.lower()
        if any(term in lowered for term in ("401", "403", "unauthorized", "forbidden", "invalid api", "authentication")):
            return "INVALID_KEY"
        if any(term in lowered for term in ("9-hour", "session cap", "max_runtime")):
            return "SESSION_LIMIT"
        if any(term in lowered for term in ("weekly quota", "quota exceeded", "exceeded your quota", "quota limit", "accelerator quota", "usage limit")):
            return "QUOTA_EXHAUSTED"
        return "COOLDOWN"

    def _trip_circuit_breaker(self, reason: str) -> None:
        delay = self.circuit_backoff_seconds
        self.circuit_open_until = time.time() + delay
        self.circuit_backoff_seconds = 300
        self.prewarm_warning = reason
        with self._state_lock:
            if not self._active_session() and not self._warming_session():
                self.tunnel_url = ""
                self.launch_time = None
        self.set_status(
            "quota_wait",
            f"{reason}. Waiting for available key or next incoming request. Retry backoff: {delay}s.",
        )

    def _circuit_allows_attempt(self) -> bool:
        if not self.circuit_open_until:
            return True
        if time.time() < self.circuit_open_until:
            return False
        self.circuit_open_until = 0.0
        self.consecutive_launch_failures = 0
        self.status = "idle"
        self.status_detail = "Circuit breaker cooldown elapsed; launch may retry."
        return True

    def _handle_session_failure(self, session: KernelSession, reason: str) -> None:
        if session.stop_event.is_set() and session.status == "stopped":
            return
        if session.accelerator == "tpu" and self._is_no_tpu_failure(reason):
            self._recreate_tpu_session(session, reason)
            return
        session.status = "failed"
        session.last_error = reason[:800]
        session.status_detail = reason[:500]
        session.stop_event.set()
        self.account_pool.end_session(session.account_id, session.id)
        failure_status = self._classify_account_failure(reason)
        if failure_status == "QUOTA_EXHAUSTED":
            self.account_pool.mark_quota_exhausted(session.account_id, session.accelerator, reason)
        elif failure_status == "INVALID_KEY":
            self.account_pool.set_verification(session.account_id, False, reason)
        elif failure_status == "SESSION_LIMIT":
            self.account_pool.mark(session.account_id, "VERIFIED_READY")
        else:
            self.account_pool.mark(session.account_id, "COOLDOWN", reason, cooldown_seconds=60)
        self.log(f"[{session.role.upper()} FAILED] {reason}")
        self.consecutive_launch_failures += 1
        if self.consecutive_launch_failures >= 2:
            with self._state_lock:
                if session.id == self.active_session_id:
                    self.active_session_id = None
                    self.tunnel_url = ""
                    self.launch_time = None
                if session.id == self.warming_session_id:
                    self.warming_session_id = None
                self.test_session_ids.discard(session.id)
            self._trip_circuit_breaker("Two consecutive Kaggle session failures")
            return
        if session.id == self.active_session_id:
            with self._state_lock:
                self.active_session_id = None
                self.tunnel_url = ""
                self.launch_time = None
            warming = self._warming_session()
            if warming and warming.status not in {"failed", "stopped"}:
                self.set_status("warming", "Active tunnel failed; waiting for the already pre-warming replacement.")
            else:
                self._start_cold_failover(session.account_id, session.model_id)
        elif session.id == self.warming_session_id:
            with self._state_lock:
                self.warming_session_id = None
            active = self._active_session()
            if not active:
                self._start_cold_failover(session.account_id, session.model_id)
        with self._state_lock:
            self.test_session_ids.discard(session.id)

    def _start_cold_failover(self, failed_account_id: Optional[str], model_id: str) -> bool:
        if not self.auto_key_rotation:
            return False
        if self._failover_thread and self._failover_thread.is_alive():
            return False
        self._failover_thread = threading.Thread(
            target=self._cold_failover_worker,
            args=(failed_account_id, model_id),
            daemon=True,
            name="cold-failover",
        )
        self._failover_thread.start()
        return True

    def schedule_failover(
        self,
        reason: str,
        model_id: Optional[str] = None,
        failed_account_id: Optional[str] = None,
    ) -> bool:
        self.log(f"[FAILOVER] {reason}")
        if not self.auto_key_rotation:
            self.set_status("error", f"{reason}. Automatic key switching is OFF.")
            return False
        return self._start_cold_failover(failed_account_id, model_id or self.active_model_id)

    def _cold_failover_worker(self, failed_account_id: Optional[str], model_id: str) -> None:
        if not self._failover_lock.acquire(blocking=False):
            return
        try:
            accelerator = str(self.accelerator_mode)
            cursor = failed_account_id
            while not self._stop_failover.is_set():
                if not self._circuit_allows_attempt():
                    return
                candidate = self.account_pool.next_ready_after(cursor, accelerator)
                if not candidate:
                    self._trip_circuit_breaker("No VERIFIED · READY Kaggle key has remaining weekly quota")
                    return
                if cursor:
                    try:
                        self._prepare_account_route(candidate["id"], "cold_failover", cursor)
                    except Exception as exc:
                        self.account_pool.mark(candidate["id"], "COOLDOWN", str(exc), cooldown_seconds=60)
                        self.consecutive_launch_failures += 1
                        if self.consecutive_launch_failures >= 2:
                            self._trip_circuit_breaker("Two consecutive protected-route or Kaggle launch failures")
                            return
                        cursor = candidate["id"]
                        continue
                self.account_pool.activate(candidate["id"])
                self.set_status("warming", f"Cold-starting {candidate['username']} after active tunnel failure.")
                result = self._launch_session(candidate["id"], model_id, "active", trigger_reason="cold_failover")
                if result.get("success"):
                    return
                cursor = candidate["id"]
        finally:
            self._failover_lock.release()

    def start_prewarm(self, trigger_reason: str) -> Dict[str, Any]:
        if not self.auto_key_rotation and trigger_reason != "manual":
            return {"success": False, "error": "Automatic key switching and pre-warming are disabled"}
        active = self._active_session()
        if not active or active.status != "ready":
            return {"success": False, "error": "No healthy active session to pre-warm behind"}
        with self._state_lock:
            warming = self._warming_session()
            if warming and warming.status not in {"failed", "stopped"}:
                return {"success": False, "error": "A replacement session is already warming", "session_id": warming.id}
        candidate = self.account_pool.next_ready_after(active.account_id, active.accelerator)
        if not candidate:
            return {"success": False, "error": "No VERIFIED · READY key has remaining quota"}
        try:
            self._prepare_account_route(candidate["id"], "prewarm", active.account_id)
        except Exception as exc:
            return {"success": False, "error": str(exc)}
        self.prewarm_warning = ""
        self.log(f"[PREWARM] Starting Tunnel B on key #{candidate.get('queue_position', '?')} {candidate['username']} ({trigger_reason}).")
        return self._launch_session(candidate["id"], active.model_id, "prewarm", trigger_reason=trigger_reason, accelerator_override=active.accelerator)

    def start_boot_test(self, account_id: str, model_id: Optional[str] = None) -> Dict[str, Any]:
        account = self.account_pool.get(account_id)
        if not account:
            return {"success": False, "error": "Kaggle account not found"}
        if not account.get("verified_at"):
            account = self.verify_kaggle_account(account_id)
        if not account.get("verified_at"):
            return {"success": False, "error": account.get("verification_error") or "INVALID_KEY"}
        return self._launch_session(account_id, model_id or self.active_model_id, "test", trigger_reason="manual_boot_test")

    def _stop_session(self, session: KernelSession, reason: str = "operator") -> None:
        if session.status == "stopped":
            return
        session.stop_event.set()
        stopped = False
        for verb in ("stop", "cancel"):
            try:
                result = self._run_kaggle(session.account_id, ["kernels", verb, session.kernel_slug], timeout=15)
                if result.returncode == 0:
                    stopped = True
                    break
            except Exception:
                continue
        if not stopped and session.kernel_slug:
            try:
                stop_dir = RUNTIME_BUNDLE_DIR / f"stop-{session.id}"
                stop_dir.mkdir(parents=True, exist_ok=True)
                (stop_dir / "kernel.py").write_text("print('GPU Hub session stopped')\n", encoding="utf-8")
                (stop_dir / "kernel-metadata.json").write_text(json.dumps({
                    "id": session.kernel_slug,
                    "title": session.kernel_title or "GPU Hub Engine",
                    "code_file": "kernel.py",
                    "language": "python",
                    "kernel_type": "script",
                    "is_private": True,
                    "enable_gpu": False,
                    "enable_tpu": False,
                    "enable_internet": False,
                    "dataset_sources": [],
                    "competition_sources": [],
                    "kernel_sources": [],
                    "model_sources": [],
                }, indent=2), encoding="utf-8")
                self._run_kaggle(session.account_id, ["kernels", "push", "-p", str(stop_dir)], timeout=45)
            except Exception as exc:
                self.log(f"Stop fallback failed for {session.id}: {exc}")
        self.account_pool.end_session(session.account_id, session.id)
        session.status = "stopped"
        session.status_detail = reason
        account = self.account_pool.get(session.account_id)
        if account and account.get("verified_at") and account.get("status") not in {"ACTIVE", "QUOTA_EXHAUSTED", "INVALID_KEY"}:
            self.account_pool.mark(session.account_id, "VERIFIED_READY")
        self.log(f"[SESSION STOPPED] {session.id} ({reason})")

    def _lifecycle_loop(self) -> None:
        time.sleep(2)
        self.verify_unverified_accounts()
        while not self._lifecycle_stop.wait(10):
            self.lifecycle_tick()

    def _usable_key_count(self, accelerator: str) -> int:
        return sum(
            1 for account in self.account_pool.list_public()
            if account.get("verified")
            and account.get("status") not in {"INVALID_KEY", "QUOTA_EXHAUSTED"}
            and float((account.get(f"{accelerator}_quota") or {}).get("remaining_seconds") or 0) > 0
        )

    def _schedule_single_key_transition(self, session: KernelSession, remaining_seconds: float, restart: bool) -> bool:
        if self._single_key_rollover_thread and self._single_key_rollover_thread.is_alive():
            return False

        def worker():
            wait_seconds = max(0.0, remaining_seconds - 20.0)
            self.prewarm_warning = (
                f"Single-key safety mode: {'restart' if restart else 'quota stop'} scheduled in {int(wait_seconds)}s; "
                "same-account Tunnel B pre-warming is disabled."
            )
            if session.stop_event.wait(wait_seconds):
                return
            current = self._active_session()
            if not current or current.id != session.id:
                return
            self._stop_session(session, "single-key session rollover" if restart else "weekly quota reached")
            with self._state_lock:
                self.active_session_id = None
                self.tunnel_url = ""
                self.launch_time = None
            if not restart:
                self.account_pool.mark_quota_exhausted(session.account_id, session.accelerator, "Locally tracked weekly quota reached")
                self._trip_circuit_breaker("Single Kaggle key is waiting for its weekly quota reset")
                return
            placeholder = KernelSession(
                id=f"cooldown-{uuid.uuid4().hex[:8]}",
                role="active",
                account_id=session.account_id,
                model_id=session.model_id,
                accelerator=session.accelerator,
                status="cooldown",
                status_detail="20-second same-key safety cooldown before restart.",
            )
            with self._state_lock:
                self.sessions[placeholder.id] = placeholder
                self.warming_session_id = placeholder.id
            self.set_status("warming", placeholder.status_detail)
            time.sleep(20)
            placeholder.status = "stopped"
            with self._state_lock:
                if self.warming_session_id == placeholder.id:
                    self.warming_session_id = None
            self.prewarm_warning = ""
            self.push_kernel(session.model_id)

        self._single_key_rollover_thread = threading.Thread(
            target=worker,
            daemon=True,
            name="single-key-rollover",
        )
        self._single_key_rollover_thread.start()
        return True

    def lifecycle_tick(self) -> Optional[str]:
        reset_ids = self.account_pool.refresh_weekly_quotas()
        if reset_ids:
            self.log(f"Weekly quota window reset for {len(reset_ids)} Kaggle account(s).")
        if not self.auto_key_rotation:
            return None
        active = self._active_session()
        if not active or active.status != "ready" or not active.tunnel_url or self._warming_session():
            return None
        session_remaining = max(0.0, SESSION_LIMIT_SECONDS - (time.time() - active.launch_time))
        weekly_remaining = self.account_pool.remaining_seconds(active.account_id, active.accelerator)
        usable_keys = self._usable_key_count(active.accelerator)
        if weekly_remaining <= PREWARM_LEAD_SECONDS:
            if usable_keys <= 1:
                self._schedule_single_key_transition(active, weekly_remaining, restart=False)
                return "single_key_quota_wait"
            result = self.start_prewarm("weekly_quota")
            if not result.get("success"):
                warning = result.get("error", "Pre-warm unavailable")
                if warning != self.prewarm_warning:
                    self.prewarm_warning = warning
                    self.log(f"[PREWARM WARNING] {self.prewarm_warning}")
            return "weekly_quota"
        if session_remaining <= PREWARM_LEAD_SECONDS:
            if usable_keys <= 1:
                self._schedule_single_key_transition(active, session_remaining, restart=weekly_remaining > PREWARM_LEAD_SECONDS)
                return "single_key_restart"
            result = self.start_prewarm("session_limit")
            if not result.get("success"):
                warning = result.get("error", "Pre-warm unavailable")
                if warning != self.prewarm_warning:
                    self.prewarm_warning = warning
                    self.log(f"[PREWARM WARNING] {self.prewarm_warning}")
            return "session_limit"
        return None

    def push_kernel(self, model_id: str) -> Dict[str, Any]:
        self._stop_failover.clear()
        self.clear_recovery_state()
        model_id = normalize_model_id(model_id)
        model = MODELS.get(model_id, {})
        if not self.model_is_compatible(model_id):
            required = "TPU v5e-8" if "tpu" in model.get("supported_accelerators", []) else "Dual T4 GPU"
            return {"success": False, "error": f"{model.get('name', model_id)} requires {required}", "status": "error"}
        if model.get("launch_supported") is False:
            return {
                "success": False,
                "error": "This model is visible as a compatibility profile, but no verified Kaggle TPU serving recipe is bundled yet",
                "status": "error",
            }
        accelerator = str(self.accelerator_mode)
        account = self.account_pool.active()
        if account and (
            account.get("status") in {"QUOTA_EXHAUSTED", "INVALID_KEY", "COOLDOWN", "VERIFYING", "PREWARMING", "HOT_STANDBY", "TESTING_BOOT"}
            or self.account_pool.remaining_seconds(account["id"], accelerator) <= 0
        ):
            account = None
        if not account:
            account = self.account_pool.next_ready_after(None, accelerator)
            if account:
                account = self.account_pool.activate(account["id"])
        if not account:
            return {"success": False, "error": "No VERIFIED · READY Kaggle account is available", "status": "error"}
        if not account.get("verified_at"):
            account = self.verify_kaggle_account(account["id"])
        if not account.get("verified_at"):
            return {"success": False, "error": account.get("verification_error") or "INVALID_KEY", "status": "error"}
        self.upstream_api_key = self.client_api_key
        self.active_model_id = model_id
        self.set_status("warming", f"Warming up {account['username']} for {model_id}.")
        result = self._launch_session(account["id"], model_id, "active", trigger_reason="manual_or_auto_wake")
        if not result.get("success"):
            result["failover_scheduled"] = self._start_cold_failover(account["id"], model_id)
        return result

    def get_elapsed_seconds(self) -> int:
        current = self._active_session() or self._warming_session()
        if current and current.status not in {"failed", "stopped"}:
            return max(0, int(time.time() - current.launch_time))
        return 0

    def stop_kernel(self) -> Dict[str, Any]:
        self._stop_failover.set()
        with self._state_lock:
            sessions = [session for session in self.sessions.values() if session.status not in {"stopped", "failed"}]
        for session in sessions:
            self._stop_session(session, "operator stop")
        with self._state_lock:
            self.active_session_id = None
            self.warming_session_id = None
            self.test_session_ids.clear()
            self.tunnel_url = ""
            self.launch_time = None
        self.set_status("idle", "All active, warming, and test sessions stopped by operator.")
        return {"success": True, "status": self.status, "message": "All Kaggle sessions stopped."}

    def get_state(self) -> Dict[str, Any]:
        active = self._active_session()
        warming = self._warming_session()
        warmup = self.warmup_snapshot()
        current = active or warming
        with self._state_lock:
            sessions = [
                session.public(self.account_pool.get(session.account_id), self._session_estimate(session))
                for session in self.sessions.values()
                if session.status not in {"stopped"}
            ]
        if self.tunnel_url and self.status == "ready" and active and active.status == "ready":
            effective_status = "ready"
            effective_detail = f"Live Tunnel: {self.tunnel_url}"
        elif self.tunnel_url and self.status == "recovering":
            effective_status = "recovering"
            effective_detail = self.status_detail
        elif warmup:
            effective_status = "warming"
            effective_detail = warmup.get("status_detail") or self.status_detail
        else:
            if self.status in {"warming", "initializing", "downloading", "loading_vram"}:
                failed = next((session for session in reversed(list(self.sessions.values())) if session.status == "failed"), None)
                self.status = "error" if failed else "idle"
                self.status_detail = failed.last_error if failed else "No accelerator session is currently running or warming."
            effective_status = self.status
            effective_detail = self.status_detail
        return {
            "status": effective_status,
            "status_detail": effective_detail,
            "active_model_id": self.active_model_id,
            "active_model": self.model_spec(self.active_model_id, (active or warming).accelerator if (active or warming) else self.accelerator_mode),
            "accelerator_mode": self.accelerator_mode,
            "tunnel_url": self.tunnel_url,
            "tunnel_type": self.tunnel_type,
            "elapsed_seconds": self.get_elapsed_seconds(),
            "launch_timestamp": current.launch_time if current and current.status not in {"failed", "stopped"} else None,
            "kernel_slug": active.kernel_slug if active else self.kernel_slug,
            "kernel_title": active.kernel_title if active else getattr(self, "kernel_title", "GPU Engine"),
            "recent_logs": self.recent_logs[-80:],
            "available_models": list(MODELS.values()),
            "kaggle_accounts": self.account_pool.list_public(),
            "kaggle_account_counts": self.account_pool.counts(),
            "active_kaggle_account_id": (self.account_pool.active() or {}).get("id"),
            "queued_requests": self.queued_requests,
            "active_requests": self.active_requests,
            "last_inference_activity": self.last_inference_activity,
            "idle_timeout_min": float(getattr(self, "idle_timeout_min", IDLE_TIMEOUT_MIN_DEFAULT)),
            "max_runtime_min": float(getattr(self, "max_runtime_min", MAX_RUNTIME_MIN_DEFAULT)),
            "active_session": active.public(self.account_pool.get(active.account_id), self._session_estimate(active)) if active else None,
            "warming_session": warming.public(self.account_pool.get(warming.account_id), self._session_estimate(warming)) if warming else None,
            "warmup": warmup,
            "sessions": sessions,
            "prewarm_lead_seconds": PREWARM_LEAD_SECONDS,
            "max_boot_seconds": MAX_BOOT_SECONDS,
            "prewarm_warning": self.prewarm_warning,
            "weekly_limits_hours": {"gpu": QUOTA_SECONDS["gpu"] / 3600, "tpu": QUOTA_SECONDS["tpu"] / 3600},
            "single_session_limit_hours": SESSION_LIMIT_SECONDS / 3600,
        }
