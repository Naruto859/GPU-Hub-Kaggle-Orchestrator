import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests


BASE_DIR = Path(__file__).resolve().parent
WEBSHARE_SECRET_PATH = BASE_DIR / "webshare_secret.json"
WEBSHARE_STATE_PATH = BASE_DIR / "webshare_state.json"
WEBSHARE_API = "https://proxy.webshare.io/api/v2"
MONTHLY_REFRESH_LIMIT = 250


class ProxyRouteError(RuntimeError):
    pass


class WebshareConnector:
    def __init__(self):
        self._lock = threading.RLock()
        self._health_cache: Dict[str, Dict[str, Any]] = {}
        self._secret = self._read_json(WEBSHARE_SECRET_PATH, {})
        self._state = self._read_json(WEBSHARE_STATE_PATH, {
            "connected": False,
            "auto_refresh": True,
            "refresh_month": self._month_key(),
            "refreshes_used": 0,
            "proxies": [],
            "last_exit_ip": "",
            "last_error": "",
            "subscription": {},
        })
        self._refresh_month_locked()

    @staticmethod
    def _month_key() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m")

    @staticmethod
    def _read_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
        if not path.exists():
            return dict(default)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else dict(default)
        except Exception:
            return dict(default)

    @staticmethod
    def _atomic_write(path: Path, data: Dict[str, Any]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.chmod(tmp, 0o600)
        tmp.replace(path)
        os.chmod(path, 0o600)

    def _save_locked(self) -> None:
        self._atomic_write(WEBSHARE_STATE_PATH, self._state)

    def _refresh_month_locked(self) -> None:
        month = self._month_key()
        if self._state.get("refresh_month") != month:
            self._state["refresh_month"] = month
            self._state["refreshes_used"] = 0
            self._save_locked()

    def token(self) -> str:
        return str(self._secret.get("api_token") or "").strip()

    def _api(self, method: str, path: str, timeout: int = 20) -> requests.Response:
        token = self.token()
        if not token:
            raise ProxyRouteError("Webshare API token is not configured")
        session = requests.Session()
        session.trust_env = False
        response = session.request(
            method,
            f"{WEBSHARE_API}{path}",
            headers={"Authorization": f"Token {token}", "Accept": "application/json"},
            timeout=timeout,
        )
        if response.status_code >= 400:
            raise ProxyRouteError(f"Webshare API HTTP {response.status_code}: {response.text[:300]}")
        return response

    @staticmethod
    def _proxy_url(item: Dict[str, Any]) -> str:
        host = str(item.get("proxy_address") or item.get("host") or "").strip()
        port = item.get("port")
        username = quote(str(item.get("username") or ""), safe="")
        password = quote(str(item.get("password") or ""), safe="")
        if not host or not port:
            return ""
        auth = f"{username}:{password}@" if username or password else ""
        return f"http://{auth}{host}:{port}"

    def fetch_proxies(self) -> List[Dict[str, Any]]:
        payload = self._api("GET", "/proxy/list/?mode=direct&page=1&page_size=25").json()
        results = payload.get("results") if isinstance(payload, dict) else []
        proxies = []
        for index, item in enumerate(results or []):
            url = self._proxy_url(item)
            if not url:
                continue
            proxies.append({
                "id": str(item.get("id") or index),
                "proxy_address": item.get("proxy_address"),
                "port": item.get("port"),
                "username": item.get("username"),
                "password": item.get("password"),
                "proxy_url": url,
                "exit_ip": str(item.get("proxy_address") or ""),
                "valid": bool(item.get("valid", True)),
            })
        with self._lock:
            self._state["proxies"] = proxies
            self._state["connected"] = bool(proxies)
            self._state["last_error"] = "" if proxies else "Webshare returned no direct proxies"
            self._save_locked()
        return proxies

    def fetch_subscription(self) -> Dict[str, Any]:
        info = {}
        for path in ("/subscription/", "/proxy/config/"):
            try:
                payload = self._api("GET", path).json()
                if isinstance(payload, dict):
                    info[path.strip("/").replace("/", "_")] = payload
            except Exception:
                continue
        with self._lock:
            self._state["subscription"] = info
            self._save_locked()
        return info

    def test_proxy(self, proxy_url: str, force: bool = False) -> str:
        proxy_url = str(proxy_url or "").strip()
        if not proxy_url or proxy_url == "server":
            raise ProxyRouteError("PROXY_ERROR: no proxy URL is configured")
        cached = self._health_cache.get(proxy_url)
        if not force and cached and time.time() - cached["checked_at"] < 60:
            return cached["exit_ip"]
        session = requests.Session()
        session.trust_env = False
        session.proxies.update({"http": proxy_url, "https": proxy_url})
        try:
            response = session.get("https://api.ipify.org?format=json", timeout=20)
            response.raise_for_status()
            exit_ip = str(response.json().get("ip") or "").strip()
        except Exception as exc:
            raise ProxyRouteError(f"PROXY_ERROR: proxy connectivity test failed: {exc}") from exc
        if not exit_ip:
            raise ProxyRouteError("PROXY_ERROR: proxy did not return an exit IP")
        self._health_cache[proxy_url] = {"exit_ip": exit_ip, "checked_at": time.time()}
        return exit_ip

    def connect(self, token: str) -> Dict[str, Any]:
        token = str(token or "").strip()
        if not token:
            raise ValueError("Webshare API token is required")
        with self._lock:
            self._secret = {"api_token": token}
            self._atomic_write(WEBSHARE_SECRET_PATH, self._secret)
        try:
            proxies = self.fetch_proxies()
            self.fetch_subscription()
            exit_ip = self.test_proxy(proxies[0]["proxy_url"], force=True) if proxies else ""
            with self._lock:
                self._state["last_exit_ip"] = exit_ip
                self._state["auto_refresh"] = True
                self._state["connected"] = bool(proxies and exit_ip)
                self._state["last_error"] = ""
                self._save_locked()
        except Exception as exc:
            with self._lock:
                self._state["connected"] = False
                self._state["last_error"] = str(exc)
                self._save_locked()
            raise
        return self.public_status()

    def set_auto_refresh(self, enabled: bool) -> None:
        with self._lock:
            self._state["auto_refresh"] = bool(enabled)
            self._save_locked()

    def disconnect(self) -> Dict[str, Any]:
        """Forget the API token and disable Webshare without touching monthly history."""
        with self._lock:
            self._secret = {}
            if WEBSHARE_SECRET_PATH.exists():
                WEBSHARE_SECRET_PATH.unlink()
            self._health_cache.clear()
            self._state.update({
                "connected": False,
                "proxies": [],
                "last_exit_ip": "",
                "last_error": "",
            })
            self._save_locked()
        return self.public_status()

    def refresh_ip(self, previous_exit_ip: str = "", wait_seconds: int = 5) -> Dict[str, Any]:
        with self._lock:
            self._refresh_month_locked()
            used = int(self._state.get("refreshes_used") or 0)
            if used >= MONTHLY_REFRESH_LIMIT:
                raise ProxyRouteError("PROXY_ERROR: Webshare monthly refresh limit reached")
        self._api("POST", "/proxy/list/refresh/")
        with self._lock:
            self._state["refreshes_used"] = int(self._state.get("refreshes_used") or 0) + 1
            self._save_locked()
        time.sleep(max(0, wait_seconds))
        proxies = self.fetch_proxies()
        if not proxies:
            raise ProxyRouteError("PROXY_ERROR: Webshare returned no proxy after refresh")
        exit_ip = self.test_proxy(proxies[0]["proxy_url"], force=True)
        changed = not previous_exit_ip or exit_ip != previous_exit_ip
        with self._lock:
            self._state["last_exit_ip"] = exit_ip
            self._state["last_error"] = "" if changed else "Webshare refresh completed but exit IP did not change"
            self._save_locked()
        if previous_exit_ip and not changed:
            raise ProxyRouteError("PROXY_ERROR: Webshare exit IP did not change after refresh")
        return {"exit_ip": exit_ip, "changed": changed, "proxies": proxies, **self.public_status()}

    def proxy_for_index(self, index: int) -> Dict[str, Any]:
        with self._lock:
            proxies = list(self._state.get("proxies") or [])
        if not proxies:
            proxies = self.fetch_proxies()
        if not proxies:
            raise ProxyRouteError("PROXY_ERROR: no Webshare proxy is available")
        return proxies[index % len(proxies)]

    def proxy_count(self) -> int:
        with self._lock:
            return len(self._state.get("proxies") or [])

    def public_status(self) -> Dict[str, Any]:
        with self._lock:
            self._refresh_month_locked()
            proxies = list(self._state.get("proxies") or [])
            used = int(self._state.get("refreshes_used") or 0)
            return {
                "connected": bool(self._state.get("connected")),
                "auto_refresh": bool(self._state.get("auto_refresh", True)),
                "proxy_count": len(proxies),
                "exit_ip": self._state.get("last_exit_ip") or "",
                "refreshes_used": used,
                "refresh_limit": MONTHLY_REFRESH_LIMIT,
                "refreshes_remaining": max(0, MONTHLY_REFRESH_LIMIT - used),
                "refresh_month": self._state.get("refresh_month"),
                "last_error": self._state.get("last_error") or "",
                "token_present": bool(self.token()),
            }
