import asyncio
import json
import logging
import hmac
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Dict, Any

import httpx
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from kaggle_manager import KaggleKernelManager
from models_registry import MODELS, DEFAULT_MODEL_ID, save_custom_model, load_custom_models, normalize_model_id
from proxy_service import InferenceProxy
from webshare_manager import ProxyRouteError
from relay_service import OutboundRelayHub

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("uncensored_service")

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
SECRET_PATH = BASE_DIR / "client_secret.json"
STATIC_DIR = BASE_DIR / "static"
_CONFIG_WRITE_LOCK = threading.RLock()

# ==============================================================================
# State & Persistence
# ==============================================================================

def load_persisted_config() -> Dict[str, Any]:
    if CONFIG_PATH.exists():
        try:
            loaded = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                return loaded
        except Exception as e:
            logger.warning(f"Error reading {CONFIG_PATH}: {e}")
    return {
        "active_model_id": DEFAULT_MODEL_ID,
        "tunnel_url": "",
        "tunnel_type": "cloudflare",
        "mock_mode": False,
        "kaggle_proxy": "server",
        "launch_time": None,
        "idle_timeout_min": 20.0,
        "max_runtime_min": 540.0,
        "client_api_key": "",
        "upstream_api_key": "",
        "auto_wake": True,
        "auto_key_rotation": True,
        "default_network_route": "server",
        "default_custom_proxy": "",
        "accelerator_mode": "gpu",
    }

def save_persisted_config(data: Dict[str, Any]):
    try:
        with _CONFIG_WRITE_LOCK:
            tmp = CONFIG_PATH.with_name(f".{CONFIG_PATH.name}.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            os.chmod(tmp, 0o600)
            tmp.replace(CONFIG_PATH)
            os.chmod(CONFIG_PATH, 0o600)
    except Exception as e:
        logger.warning(f"Error saving {CONFIG_PATH}: {e}")

def load_secret_config() -> Dict[str, str]:
    if SECRET_PATH.exists():
        try:
            data = json.loads(SECRET_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception as e:
            logger.warning(f"Error reading {SECRET_PATH}: {e}")
    return {}

def save_secret_config() -> None:
    try:
        with _CONFIG_WRITE_LOCK:
            tmp = SECRET_PATH.with_name(f".{SECRET_PATH.name}.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps({
                "client_api_key": client_api_key,
                "upstream_api_key": getattr(manager, "upstream_api_key", ""),
            }, indent=2), encoding="utf-8")
            os.chmod(tmp, 0o600)
            tmp.replace(SECRET_PATH)
            os.chmod(SECRET_PATH, 0o600)
    except Exception as e:
        logger.warning(f"Error saving {SECRET_PATH}: {e}")

saved_conf = load_persisted_config()
saved_secrets = load_secret_config()
manager = KaggleKernelManager(config_callback=lambda: sync_config())
manager.clear_recovery_state()
manager.active_model_id = normalize_model_id(saved_conf.get("active_model_id", DEFAULT_MODEL_ID))
manager.tunnel_url = saved_conf.get("tunnel_url", "")
manager.tunnel_type = saved_conf.get("tunnel_type", "cloudflare")
manager.kaggle_proxy = saved_conf.get("kaggle_proxy", "server")
manager.launch_time = saved_conf.get("launch_time", None)
mock_mode = saved_conf.get("mock_mode", False)
manager.idle_timeout_min = float(saved_conf.get("idle_timeout_min", 20.0))
manager.max_runtime_min = float(saved_conf.get("max_runtime_min", 540.0))
manager.auto_key_rotation = bool(saved_conf.get("auto_key_rotation", True))
manager.default_network_route = str(saved_conf.get("default_network_route") or "server")
manager.default_custom_proxy = str(saved_conf.get("default_custom_proxy") or "")
manager.accelerator_mode = str(saved_conf.get("accelerator_mode") or "gpu")
manager.apply_default_route_to_inherited_accounts()
if manager.max_runtime_min == 330.0:  # migrate the previous default to Kaggle's 9-hour session cap
    manager.max_runtime_min = 540.0
client_api_key = str(saved_secrets.get("client_api_key") or saved_conf.get("client_api_key") or "").strip()
if not client_api_key:
    client_api_key = "ghub-" + secrets.token_urlsafe(24)
manager.client_api_key = client_api_key
manager.upstream_api_key = str(saved_secrets.get("upstream_api_key") or saved_conf.get("upstream_api_key") or client_api_key).strip()
auto_wake = bool(saved_conf.get("auto_wake", True))
manager.recover_persisted_session()

def sync_config():
    save_persisted_config({
        "active_model_id": manager.active_model_id,
        "tunnel_url": manager.tunnel_url,
        "tunnel_type": manager.tunnel_type,
        "status": manager.status,
        "mock_mode": mock_mode,
        "kaggle_proxy": getattr(manager, "kaggle_proxy", "server"),
        "launch_time": getattr(manager, "launch_time", None),
        "idle_timeout_min": float(getattr(manager, "idle_timeout_min", 20.0)),
        "max_runtime_min": float(getattr(manager, "max_runtime_min", 540.0)),
        "auto_wake": auto_wake,
        "auto_key_rotation": bool(manager.auto_key_rotation),
        "default_network_route": manager.default_network_route,
        "default_custom_proxy": manager.default_custom_proxy,
        "accelerator_mode": manager.accelerator_mode,
    })
    save_secret_config()

sync_config()

# ==============================================================================
# Manager Web App (Port 8778)
# ==============================================================================

app_manager = FastAPI(title="Kaggle Uncensored AI Agent Manager", version="1.0.0")
app_relay = FastAPI(title="GPU Hub Kaggle Outbound Relay", version="1.0.0")
relay_hub = OutboundRelayHub()
manager.relay_hub = relay_hub
app_manager.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app_manager.get("/api/status")
async def get_status():
    global mock_mode
    creds = manager.load_kaggle_credentials()
    state = manager.get_state()
    state["mock_mode"] = mock_mode
    state["kaggle_username"] = creds.get("username", "")
    state["kaggle_key_present"] = creds.get("key_present", False)
    state["client_api_configured"] = bool(client_api_key)
    state["auto_wake"] = auto_wake
    state["auto_key_rotation"] = bool(manager.auto_key_rotation)
    state["default_network_route"] = manager.default_network_route
    state["accelerator_mode"] = manager.accelerator_mode
    state["webshare"] = manager.webshare.public_status()
    return state


@app_manager.get("/api/settings")
async def get_settings():
    return {
        "auto_key_rotation": bool(manager.auto_key_rotation),
        "default_network_route": manager.default_network_route,
        "default_custom_proxy_configured": bool(manager.default_custom_proxy),
        "webshare_connected": manager.webshare.public_status().get("connected", False),
        "accelerator_mode": manager.accelerator_mode,
    }


@app_manager.post("/api/settings")
async def update_settings(req: Request):
    data = await req.json()
    if "auto_key_rotation" in data:
        manager.auto_key_rotation = bool(data.get("auto_key_rotation"))
    if "default_network_route" in data:
        route = str(data.get("default_network_route") or "server").strip().lower()
        if route not in {"server", "webshare", "custom"}:
            raise HTTPException(status_code=400, detail="Invalid default network route")
        custom_url = str(data.get("custom_proxy_url") or manager.default_custom_proxy or "").strip()
        try:
            await asyncio.to_thread(manager.set_default_network_route, route, custom_url)
        except (ValueError, ProxyRouteError) as exc:
            raise HTTPException(status_code=502, detail=str(exc))
    if "accelerator_mode" in data:
        accelerator = str(data.get("accelerator_mode") or "gpu").lower()
        if accelerator not in {"gpu", "tpu"}:
            raise HTTPException(status_code=400, detail="accelerator_mode must be gpu or tpu")
        manager.accelerator_mode = accelerator
    sync_config()
    return {"success": True, **(await get_settings())}

@app_manager.get("/api/client-config")
async def get_client_config():
    return JSONResponse(
        {
            "api_key": client_api_key,
            "auto_wake": auto_wake,
            "base_path": "/v1",
            "chat_path": "/v1/chat/completions",
            "messages_path": "/v1/messages",
        },
        headers={"Cache-Control": "no-store"},
    )

@app_manager.post("/api/client-config")
async def update_client_config(req: Request):
    global client_api_key, auto_wake
    data = await req.json()
    if "api_key" in data:
        new_key = str(data.get("api_key") or "").strip()
        if len(new_key) < 12 or len(new_key) > 160 or any(ch.isspace() for ch in new_key):
            raise HTTPException(status_code=400, detail="API key must be 12-160 non-whitespace characters")
        client_api_key = new_key
        manager.client_api_key = new_key
        if not manager.tunnel_url or MODELS.get(manager.active_model_id, {}).get("accelerator") != "tpu":
            manager.upstream_api_key = new_key
    if "auto_wake" in data:
        auto_wake = bool(data.get("auto_wake"))
    sync_config()
    return {"success": True, "api_key": client_api_key, "auto_wake": auto_wake}

@app_manager.get("/api/kaggle/accounts")
async def list_kaggle_accounts():
    return {
        "accounts": manager.account_pool.list_public(),
        "active_account_id": (manager.account_pool.active() or {}).get("id"),
        "webshare": manager.webshare.public_status(),
    }


@app_manager.get("/api/webshare/status")
async def get_webshare_status():
    return manager.webshare.public_status()


@app_manager.post("/api/webshare/connect")
async def connect_webshare(req: Request):
    data = await req.json()
    token = str(data.get("token") or "").strip()
    if not token:
        raise HTTPException(status_code=400, detail="Webshare API token is required")
    try:
        status = await asyncio.to_thread(manager.connect_webshare, token)
    except (ValueError, ProxyRouteError) as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    manager.log(
        f"Webshare connected with {status.get('proxy_count', 0)} proxy route(s); verified exit {status.get('exit_ip') or 'unknown'}."
    )
    sync_config()
    return {"success": True, **status}


@app_manager.post("/api/webshare/disconnect")
async def disconnect_webshare():
    status = await asyncio.to_thread(manager.disconnect_webshare)
    sync_config()
    manager.log("Webshare disconnected; Webshare-routed accounts and the default route now use Server Direct.")
    return {"success": True, **status, "default_network_route": "server"}


@app_manager.post("/api/webshare/refresh")
async def refresh_webshare_ip():
    try:
        refreshed = await asyncio.to_thread(manager.refresh_webshare_ip)
    except ProxyRouteError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    manager.log(f"Webshare IP refreshed and verified as {refreshed.get('exit_ip') or 'unknown'}.")
    return {"success": True, **refreshed}


@app_manager.post("/api/webshare/auto-refresh")
async def set_webshare_auto_refresh(req: Request):
    data = await req.json()
    enabled = bool(data.get("enabled"))
    manager.webshare.set_auto_refresh(enabled)
    return {"success": True, **manager.webshare.public_status()}


@app_manager.put("/api/kaggle/accounts/{account_id}/proxy")
async def update_kaggle_account_proxy(account_id: str, req: Request):
    data = await req.json()
    mode = str(data.get("mode") or "default").strip().lower()
    if mode == "default":
        proxy = "default"
    elif mode == "webshare":
        proxy = "webshare"
    elif mode == "custom":
        proxy = str(data.get("proxy_url") or "").strip()
        if not proxy:
            raise HTTPException(status_code=400, detail="Custom proxy URL is required")
    elif mode == "server":
        proxy = "server"
    else:
        raise HTTPException(status_code=400, detail="Invalid proxy mode")
    try:
        await asyncio.to_thread(manager.update_account_proxy, account_id, proxy)
    except KeyError:
        raise HTTPException(status_code=404, detail="Kaggle account not found")
    except (ValueError, ProxyRouteError) as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    public = {item["id"]: item for item in manager.account_pool.list_public()}.get(account_id)
    return {"success": True, "account": public}


@app_manager.post("/api/kaggle/accounts/{account_id}/proxy/test")
async def test_kaggle_account_proxy(account_id: str):
    try:
        result = await asyncio.to_thread(manager.test_account_proxy, account_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Kaggle account not found")
    except (ValueError, ProxyRouteError, OSError) as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return {"success": True, **result}

@app_manager.post("/api/kaggle/accounts")
async def add_kaggle_account(req: Request):
    data = await req.json()
    username = str(data.get("username") or "").strip()
    key = str(data.get("key") or "").strip()
    proxy = str(data.get("proxy") or "default").strip() or "default"
    if not key:
        raise HTTPException(status_code=400, detail="Kaggle API key is required; username is optional")
    try:
        account = await asyncio.to_thread(manager.add_kaggle_account, username, key, proxy)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    sync_config()
    public = {a["id"]: a for a in manager.account_pool.list_public()}.get(account["id"])
    return {"success": bool(public and public.get("verified")), "account": public}

@app_manager.post("/api/kaggle/detect-username")
async def detect_kaggle_account_username(req: Request):
    data = await req.json()
    key = str(data.get("key") or "").strip()
    proxy = str(data.get("proxy") or "default").strip() or "default"
    if not key:
        raise HTTPException(status_code=400, detail="Kaggle API key is required")
    username = await asyncio.to_thread(manager.detect_username_for_key, key, proxy)
    if not username:
        raise HTTPException(status_code=422, detail="Could not auto-detect a username for this key")
    return {"success": True, "username": username}

@app_manager.post("/api/kaggle/accounts/bulk")
async def bulk_add_kaggle_accounts(req: Request):
    data = await req.json()
    entries = manager.account_pool.parse_bulk(str(data.get("text") or ""), str(data.get("proxy") or "default"))
    if not entries:
        raise HTTPException(status_code=400, detail="No valid username:key or kaggle.json entries found")
    results = await asyncio.gather(*(
        asyncio.to_thread(manager.add_kaggle_account, username, key, proxy)
        for username, key, proxy in entries
    ), return_exceptions=True)
    added = [result["id"] for result in results if isinstance(result, dict)]
    errors = [str(result) for result in results if isinstance(result, Exception)]
    sync_config()
    accounts = manager.account_pool.list_public()
    imported = [account for account in accounts if account["id"] in set(added)]
    return {
        "success": bool(added),
        "added": added,
        "verified_count": sum(1 for account in imported if account.get("verified")),
        "invalid_count": sum(1 for account in imported if account.get("status") == "INVALID_KEY"),
        "errors": errors,
        "accounts": accounts,
    }

@app_manager.post("/api/kaggle/accounts/{account_id}/activate")
async def activate_kaggle_account(account_id: str):
    try:
        account = manager.activate_kaggle_account(account_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Kaggle account not found")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    sync_config()
    return {
        "success": True,
        "requested_account_id": account["id"],
        "active_account_id": (manager.account_pool.active() or {}).get("id"),
        "accounts": manager.account_pool.list_public(),
    }

@app_manager.delete("/api/kaggle/accounts/{account_id}")
async def delete_kaggle_account(account_id: str):
    if any(session.account_id == account_id and session.status not in {"stopped", "failed"} for session in manager.sessions.values()):
        raise HTTPException(status_code=409, detail="Stop this account's active/warming session before deleting it")
    if not manager.delete_kaggle_account(account_id):
        raise HTTPException(status_code=404, detail="Kaggle account not found")
    sync_config()
    return {"success": True, "accounts": manager.account_pool.list_public()}

@app_manager.post("/api/kaggle/accounts/{account_id}/verify")
async def verify_kaggle_account(account_id: str):
    try:
        account = await asyncio.to_thread(manager.verify_kaggle_account, account_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Kaggle account not found")
    public = {item["id"]: item for item in manager.account_pool.list_public()}.get(account["id"])
    return {"success": bool(account.get("verified_at")), "account": public}

@app_manager.post("/api/kaggle/accounts/{account_id}/reset-weekly")
async def reset_kaggle_weekly_quota(account_id: str):
    try:
        manager.account_pool.reset_weekly_quota(account_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Kaggle account not found")
    return {"success": True, "accounts": manager.account_pool.list_public()}


@app_manager.post("/api/kaggle/accounts/{account_id}/clear-cooldown")
async def clear_kaggle_account_cooldown(account_id: str):
    if not manager.account_pool.get(account_id):
        raise HTTPException(status_code=404, detail="Kaggle account not found")
    result = manager.clear_recovery_state(account_id)
    sync_config()
    public = {item["id"]: item for item in manager.account_pool.list_public()}.get(account_id)
    return {**result, "account": public}

@app_manager.post("/api/kaggle/accounts/{account_id}/test-boot")
async def test_kaggle_boot(account_id: str, req: Request):
    data = await req.json() if req.headers.get("content-length", "0") != "0" else {}
    model_id = data.get("model_id", manager.active_model_id)
    if model_id not in MODELS:
        raise HTTPException(status_code=400, detail="Invalid model_id")
    result = await asyncio.to_thread(manager.start_boot_test, account_id, model_id)
    if not result.get("success"):
        raise HTTPException(status_code=409, detail=result.get("error", "Boot test could not start"))
    return result

@app_manager.post("/api/lifecycle/prewarm")
async def trigger_manual_prewarm():
    result = await asyncio.to_thread(manager.start_prewarm, "manual")
    if not result.get("success"):
        raise HTTPException(status_code=409, detail=result.get("error", "Pre-warm could not start"))
    return result

@app_manager.post("/api/model/select")
async def select_model(req: Request):
    data = await req.json()
    model_id = normalize_model_id(data.get("model_id"))
    if model_id not in MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model_id: {model_id}")
    manager.active_model_id = model_id
    manager.log(f"Model switched to {model_id}")
    sync_config()
    return {"success": True, "active_model_id": model_id, "model": MODELS[model_id]}

@app_manager.get("/api/models/custom")
async def list_custom_models():
    return {"models": list(load_custom_models().values())}

@app_manager.post("/api/models/custom")
async def add_custom_model(req: Request):
    data = await req.json()
    try:
        model = save_custom_model(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    manager.log(f"Custom model saved: {model['id']} ({model['repo_id']}/{model['filename']})")
    return {"success": True, "model": model, "available_models": list(MODELS.values())}

@app_manager.post("/api/kaggle/launch")
async def launch_kaggle(req: Request):
    data = await req.json() if req.headers.get("content-length", "0") != "0" else {}
    model_id = normalize_model_id(data.get("model_id", manager.active_model_id))
    if data.get("accelerator_mode") in {"gpu", "tpu"}:
        manager.accelerator_mode = data["accelerator_mode"]
    if model_id not in MODELS:
        raise HTTPException(status_code=400, detail="Invalid model_id")
    import asyncio
    loop = asyncio.get_event_loop()
    res = await loop.run_in_executor(None, manager.push_kernel, model_id)
    sync_config()
    return res

@app_manager.post("/api/kernel/settings")
async def update_kernel_settings(req: Request):
    data = await req.json()
    idle_raw = data.get("idle_timeout_min")
    max_raw = data.get("max_runtime_min")
    if idle_raw is not None:
        try:
            idle = float(idle_raw)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"Invalid idle_timeout_min: {idle_raw!r}")
        if idle < 0 or idle > 10080:
            raise HTTPException(status_code=400, detail="idle_timeout_min must be between 0 (disabled) and 10080 (7 days)")
        manager.idle_timeout_min = idle
        manager.log(f"Auto-shutdown idle timeout set to {idle} min (applies to next launch)")
    if max_raw is not None:
        try:
            cap = float(max_raw)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"Invalid max_runtime_min: {max_raw!r}")
        if cap < 0 or cap > 720:
            raise HTTPException(status_code=400, detail="max_runtime_min must be between 0 (disabled) and 720")
        manager.max_runtime_min = cap
        manager.log(f"Auto-shutdown max runtime cap set to {cap} min (applies to next launch)")
    sync_config()
    return {
        "success": True,
        "idle_timeout_min": float(manager.idle_timeout_min),
        "max_runtime_min": float(manager.max_runtime_min),
    }

@app_manager.post("/api/kernel/stop")
@app_manager.post("/api/kaggle/stop")
async def stop_kaggle():
    import asyncio
    loop = asyncio.get_event_loop()
    res = await loop.run_in_executor(None, manager.stop_kernel)
    sync_config()
    return res

@app_manager.post("/api/kernel/reset")
async def reset_kaggle():
    await asyncio.to_thread(manager.stop_kernel)
    manager.set_status("idle", "Lifecycle state reset by operator.")
    sync_config()
    return {"success": True, "status": manager.status}

@app_manager.post("/api/tunnel/register")
async def register_tunnel(req: Request):
    data = await req.json()
    kernel_status = str(data.get("status", "")).lower()
    if kernel_status == "auto_shutdown":
        # Kernel-side idle/max-runtime auto-shutdown: session already ended on
        # Kaggle (kernel exited). Reset dashboard state so the slot is shown free.
        reason = data.get("reason", "unknown")
        detail = data.get("detail", "")
        manager.handle_session_shutdown(data.get("session_id"), reason, detail)
        sync_config()
        return {"success": True, "auto_shutdown": True, "reason": reason}
    tunnel_url = data.get("tunnel_url")
    if not tunnel_url:
        raise HTTPException(status_code=400, detail="Missing tunnel_url")
    session_id = data.get("session_id")
    if session_id:
        if not manager.register_session_tunnel(session_id, tunnel_url):
            raise HTTPException(status_code=404, detail="Unknown GPU Hub session_id")
    else:
        manager.set_tunnel_url(tunnel_url)
    model_id = data.get("model_id")
    if model_id and model_id in MODELS:
        manager.active_model_id = model_id
    sync_config()
    return {"success": True, "registered_tunnel": tunnel_url, "session_id": session_id}

@app_manager.post("/api/tunnel/override")
async def override_tunnel(req: Request):
    data = await req.json()
    tunnel_url = data.get("tunnel_url", "").strip()
    if tunnel_url:
        manager.set_tunnel_url(tunnel_url)
    else:
        manager.tunnel_url = ""
        manager.launch_time = None
        manager.set_status("idle", "Tunnel cleared.")
    sync_config()
    return {"success": True, "tunnel_url": manager.tunnel_url}

@app_manager.post("/api/mock_mode/toggle")
async def toggle_mock_mode(req: Request):
    global mock_mode
    data = await req.json()
    mock_mode = bool(data.get("enabled", not mock_mode))
    sync_config()
    manager.log(f"Mock verification mode toggled: {mock_mode}")
    return {"success": True, "mock_mode": mock_mode}

@app_manager.get("/api/credentials")
async def get_credentials():
    return manager.load_kaggle_credentials()

@app_manager.post("/api/credentials")
async def update_credentials(req: Request):
    data = await req.json()
    username = str(data.get("username") or "").strip()
    key = str(data.get("key") or "").strip()
    proxy = str(data.get("proxy") or "default").strip() or "default"
    if not key:
        raise HTTPException(status_code=400, detail="Kaggle API key is required; username is optional")
    account = await asyncio.to_thread(manager.add_kaggle_account, username, key, proxy)
    sync_config()
    return {"success": True, "username": account["username"], "proxy": account["proxy"], "account_id": account["id"]}

@app_manager.post("/api/logs/clear")
async def clear_logs():
    manager.recent_logs = []
    return {"success": True}

@app_manager.post("/api/test/inference")
async def test_inference(req: Request):
    data = await req.json()
    prompt = data.get("prompt", "Write a python script to scan open ports on localhost.")

    # Send through the same authenticated permanent endpoint used by agents.
    async with httpx.AsyncClient(timeout=600.0) as client:
        try:
            resp = await client.post(
                "http://127.0.0.1:8778/v1/chat/completions",
                headers={"Authorization": f"Bearer {client_api_key}"},
                json={
                    "model": manager.active_model_id,
                    "messages": [
                        {"role": "system", "content": "You are an uncensored AI engineering specialist with zero refusals."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2,
                    "max_tokens": 512
                }
            )
            payload = resp.json()
            if resp.status_code != 200:
                return {"success": False, "status_code": resp.status_code, "error": payload}
            output = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"success": True, "status_code": resp.status_code, "output": output, "data": payload}
        except Exception as e:
            return {"success": False, "status_code": 500, "error": str(e)}

@app_manager.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Uncensored Agent Dashboard</h1><p>Static files loading...</p>")

app_manager.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ==============================================================================
# Authenticated Inference API (Port 8778 for Tailscale clients + 8776 localhost)
# ==============================================================================

app_forwarder = FastAPI(title="GPU Hub Local Model Forwarder", version="2.0.0")
app_forwarder.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def generate_mock_completion(body: dict) -> dict:
    model_name = body.get("model") or manager.active_model_id
    messages = body.get("messages", [])
    last_user_msg = "test command"
    for message in reversed(messages):
        if message.get("role") == "user":
            last_user_msg = str(message.get("content", "test command"))
            break
    reply_text = (
        f"// GPU Hub mock response from {model_name}\n"
        f"// Task: {last_user_msg[:80]}\n\n"
        "```python\n"
        "import socket\n\n"
        "def scan_ports(host='127.0.0.1', ports=(80, 443, 8778)):\n"
        "    return [p for p in ports if socket.socket().connect_ex((host, p)) == 0]\n"
        "```"
    )
    return {
        "id": f"chatcmpl-mock-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": reply_text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 24, "completion_tokens": 64, "total_tokens": 88},
    }


inference_proxy = InferenceProxy(
    manager=manager,
    models=MODELS,
    api_key_getter=lambda: client_api_key,
    mock_mode_getter=lambda: bool(mock_mode),
    mock_completion_factory=generate_mock_completion,
    auto_wake_getter=lambda: bool(auto_wake),
    relay_hub=relay_hub,
)


def _relay_authorized(request: Request, session_id: str) -> bool:
    session = manager._session_by_id(session_id)
    auth = request.headers.get("authorization", "")
    supplied = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    return bool(session and session.relay_token and supplied and hmac.compare_digest(supplied, session.relay_token))


@app_relay.post("/register")
async def relay_register(request: Request):
    data = await request.json()
    session_id = str(data.get("session_id") or "")
    if not _relay_authorized(request, session_id):
        raise HTTPException(status_code=401, detail="Invalid relay session token")
    session = manager._session_by_id(session_id)
    relay_hub.register(session_id, session.relay_token)
    if not manager.register_session_relay(session_id):
        relay_hub.unregister(session_id, "Unknown or stopped Kaggle session")
        raise HTTPException(status_code=404, detail="Unknown or stopped Kaggle session")
    sync_config()
    return {"success": True, "session_id": session_id}


@app_relay.post("/poll")
async def relay_poll(request: Request):
    data = await request.json()
    session_id = str(data.get("session_id") or "")
    if not _relay_authorized(request, session_id):
        raise HTTPException(status_code=401, detail="Invalid relay session token")
    supplied = request.headers.get("authorization", "")[7:].strip()
    if not relay_hub.authenticate(session_id, supplied):
        session = manager._session_by_id(session_id)
        relay_hub.register(session_id, session.relay_token)
        manager.register_session_relay(session_id)
    job = await relay_hub.poll(session_id, float(data.get("timeout") or 20))
    if not job:
        return Response(status_code=204)
    return job


@app_relay.post("/result")
async def relay_result(request: Request):
    data = await request.json()
    session_id = str(data.get("session_id") or "")
    if not _relay_authorized(request, session_id):
        raise HTTPException(status_code=401, detail="Invalid relay session token")
    supplied = request.headers.get("authorization", "")[7:].strip()
    if not relay_hub.authenticate(session_id, supplied):
        session = manager._session_by_id(session_id)
        relay_hub.register(session_id, session.relay_token)
        manager.register_session_relay(session_id)
    accepted = relay_hub.resolve(
        session_id, str(data.get("job_id") or ""), int(data.get("status_code") or 502),
        data.get("headers") or {}, str(data.get("body_b64") or ""),
    )
    return {"success": accepted}


@app_manager.get("/v1/models")
@app_manager.get("/models")
@app_forwarder.get("/v1/models")
@app_forwarder.get("/models")
async def list_models(request: Request):
    inference_proxy.authenticate(request)
    data = []
    live_ctx = None
    if manager.tunnel_url and not manager.tunnel_url.startswith("relay://"):
        try:
            client = inference_proxy.closeable_client()
            response = await client.get(
                f"{manager.tunnel_url.rstrip('/')}/props",
                headers=inference_proxy.upstream_headers(),
                timeout=4.0,
            )
            if response.status_code == 200:
                live_ctx = (response.json().get("default_generation_settings") or {}).get("n_ctx")
        except Exception:
            pass
    for model in MODELS.values():
        context = int(model.get("context_length", 65536))
        if model["id"] == manager.active_model_id:
            context = int(live_ctx or model.get("runtime_context_length") or context)
        data.append({
            "id": model["id"], "object": "model", "created": 1727000000, "owned_by": "gpu-hub",
            "permission": [], "root": model["id"], "parent": None,
            "context_length": context, "context_window": context, "max_model_len": context,
            "max_context_length": context, "max_position_embeddings": context,
            "supported_accelerators": model.get("supported_accelerators", []),
            "default_accelerator": model.get("default_accelerator", "gpu"),
        })
    return {"object": "list", "data": data}


@app_manager.get("/props")
@app_manager.get("/v1/props")
@app_forwarder.get("/props")
@app_forwarder.get("/v1/props")
async def get_props(request: Request):
    inference_proxy.authenticate(request)
    if manager.tunnel_url and not manager.tunnel_url.startswith("relay://"):
        try:
            response = await inference_proxy.closeable_client().get(
                f"{manager.tunnel_url.rstrip('/')}/props",
                headers=inference_proxy.upstream_headers(),
                timeout=4.0,
            )
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
    active_model = MODELS.get(manager.active_model_id, {})
    context = int(active_model.get("runtime_context_length") or active_model.get("context_length", 32768))
    return {"default_generation_settings": {"n_ctx": context}, "model_alias": manager.active_model_id, "n_ctx": context}


@app_manager.get("/health")
@app_manager.get("/v1/health")
@app_forwarder.get("/health")
@app_forwarder.get("/v1/health")
async def health_check(request: Request):
    inference_proxy.authenticate(request)
    return {
        "status": "ok",
        "active_model": manager.active_model_id,
        "tunnel_url": manager.tunnel_url,
        "is_ready": bool(manager.tunnel_url and manager.status == "ready"),
        "mock_mode": mock_mode,
        "queue": inference_proxy.queue_state(),
    }


@app_manager.post("/v1/chat/completions")
@app_manager.post("/chat/completions")
@app_forwarder.post("/v1/chat/completions")
@app_forwarder.post("/chat/completions")
async def chat_completions(request: Request):
    return await inference_proxy.chat(request)


@app_manager.post("/v1/completions")
@app_manager.post("/completions")
@app_forwarder.post("/v1/completions")
@app_forwarder.post("/completions")
async def completions(request: Request):
    return await inference_proxy.completions(request)


@app_manager.post("/v1/messages")
@app_manager.post("/messages")
@app_forwarder.post("/v1/messages")
@app_forwarder.post("/messages")
async def anthropic_messages_compat(request: Request):
    return await inference_proxy.messages(request)


@app_manager.post("/v1/messages/count_tokens")
@app_forwarder.post("/v1/messages/count_tokens")
async def anthropic_count_tokens(request: Request):
    return await inference_proxy.count_tokens(request)


# ==============================================================================
# Dual Server Runner
# ==============================================================================

async def run_servers():
    import subprocess
    ts_ip = os.environ.get("GPU_HUB_TAILSCALE_IP", "").strip()
    try:
        out = subprocess.check_output(["tailscale", "ip", "-4"], text=True).strip()
        if out:
            ts_ip = out
    except Exception:
        pass

    config_mgr_local = uvicorn.Config(
        app_manager,
        host="127.0.0.1",
        port=8778,
        log_level="info",
        access_log=False
    )
    config_fwd = uvicorn.Config(
        app_forwarder,
        host="127.0.0.1",
        port=8776,
        log_level="info",
        access_log=False
    )

    server_mgr_local = uvicorn.Server(config_mgr_local)
    server_fwd = uvicorn.Server(config_fwd)

    servers = [server_mgr_local, server_fwd]
    relay_bind = os.environ.get("GPU_HUB_RELAY_BIND", "").strip()
    relay_port = int(os.environ.get("GPU_HUB_RELAY_PORT", "8777") or 8777)
    if relay_bind:
        servers.append(uvicorn.Server(uvicorn.Config(
            app_relay, host=relay_bind, port=relay_port, log_level="warning", access_log=False,
        )))
        logger.info(f"Starting authenticated Kaggle outbound relay on {relay_bind}:{relay_port}")
    if ts_ip and ts_ip != "127.0.0.1":
        servers.insert(0, uvicorn.Server(uvicorn.Config(
            app_manager,
            host=ts_ip,
            port=8778,
            log_level="info",
            access_log=False,
        )))
    logger.info(
        f"Starting GPU Hub on Localhost (127.0.0.1:8778)"
        + (f" and Tailscale ({ts_ip}:8778)" if ts_ip and ts_ip != "127.0.0.1" else "")
    )
    logger.info("Public IP binding is completely REMOVED.")
    await asyncio.gather(*(server.serve() for server in servers))

if __name__ == "__main__":
    asyncio.run(run_servers())
