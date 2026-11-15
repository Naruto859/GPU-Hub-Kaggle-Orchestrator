import json
import os
import statistics
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit


WEEK_SECONDS = 7 * 24 * 3600
SESSION_LIMIT_SECONDS = 9 * 3600
QUOTA_SECONDS = {"gpu": 30 * 3600, "tpu": 20 * 3600}
ACCOUNT_STATUSES = {
    "ACTIVE", "VERIFIED_READY", "VERIFYING", "PREWARMING", "HOT_STANDBY", "TESTING_BOOT",
    "QUOTA_EXHAUSTED", "COOLDOWN", "INVALID_KEY",
}
LEGACY_STATUS_MAP = {"STANDBY": "VERIFIED_READY", "INVALID": "INVALID_KEY"}


class KaggleAccountPool:
    """Ordered Kaggle key pool with verification, quota, boot, and session ledgers."""

    def __init__(self, path: Path, legacy_path: Path):
        self.path = path
        self.legacy_path = legacy_path
        self._lock = threading.RLock()
        self._data: Dict[str, Any] = {"accounts": [], "active_account_id": None}
        self._load()
        self._migrate_legacy()

    @staticmethod
    def _new_account(username: str, key: str, proxy: str, status: str = "VERIFYING") -> Dict[str, Any]:
        now = time.time()
        return {
            "id": uuid.uuid4().hex[:12], "username": username, "key": key, "proxy": proxy or "server",
            "assigned_proxy_url": "", "last_exit_ip": "",
            "status": status, "verified_at": 0.0, "verification_error": "", "created_at": now,
            "updated_at": now, "cooldown_until": 0.0, "last_error": "", "quota_window_started_at": 0.0,
            "gpu_used_seconds": 0.0, "tpu_used_seconds": 0.0, "quota_exhausted_accelerator": "",
            "session_started_at": 0.0, "session_accelerator": "", "session_id": "", "session_model_id": "",
            "last_session_seconds": 0.0, "boot_history_gpu": [], "boot_history_tpu": [], "last_boot_seconds": 0.0,
        }

    def _load(self) -> None:
        with self._lock:
            if self.path.exists():
                try:
                    raw = json.loads(self.path.read_text(encoding="utf-8"))
                    if isinstance(raw, dict) and isinstance(raw.get("accounts"), list):
                        self._data = raw
                except Exception:
                    pass
            self._normalize_locked()

    def _save_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        os.chmod(tmp, 0o600)
        tmp.replace(self.path)
        os.chmod(self.path, 0o600)

    def _normalize_locked(self) -> None:
        accounts: List[Dict[str, Any]] = []
        seen = set()
        now = time.time()
        for raw in self._data.get("accounts", []):
            if not isinstance(raw, dict):
                continue
            username, key = str(raw.get("username", "")).strip(), str(raw.get("key", "")).strip()
            if not username or not key:
                continue
            status = LEGACY_STATUS_MAP.get(str(raw.get("status", "VERIFYING")).upper(), str(raw.get("status", "VERIFYING")).upper())
            if status not in ACCOUNT_STATUSES:
                status = "VERIFYING"
            account_id = str(raw.get("id") or uuid.uuid4().hex[:12])
            if account_id in seen:
                account_id = uuid.uuid4().hex[:12]
            seen.add(account_id)
            defaults = self._new_account(username, key, str(raw.get("proxy") or "server"), status)
            account = {**defaults, **raw, "id": account_id, "username": username, "key": key, "status": status}
            account["proxy"] = str(account.get("proxy") or "server").strip() or "server"
            for field in (
                "verified_at", "created_at", "updated_at", "cooldown_until", "quota_window_started_at",
                "gpu_used_seconds", "tpu_used_seconds", "session_started_at", "last_session_seconds", "last_boot_seconds",
            ):
                try:
                    account[field] = float(account.get(field) or 0)
                except (TypeError, ValueError):
                    account[field] = 0.0
            account["boot_history_gpu"] = self._clean_history(account.get("boot_history_gpu", []))
            account["boot_history_tpu"] = self._clean_history(account.get("boot_history_tpu", []))
            if status == "COOLDOWN" and account["cooldown_until"] <= now:
                account["status"] = "VERIFIED_READY" if account["verified_at"] else "VERIFYING"
                account["cooldown_until"] = 0.0
            self._refresh_week_locked(account, now)
            accounts.append(account)
        self._data["accounts"] = accounts
        active_id = self._data.get("active_account_id")
        if active_id not in {account["id"] for account in accounts}:
            active = next((account for account in accounts if account["status"] == "ACTIVE"), None)
            self._data["active_account_id"] = active["id"] if active else None
        for account in accounts:
            if account["status"] == "ACTIVE" and account["id"] != self._data.get("active_account_id"):
                account["status"] = "VERIFIED_READY" if account["verified_at"] else "VERIFYING"

    @staticmethod
    def _clean_history(values: Any) -> List[float]:
        history = []
        for value in list(values or [])[-20:]:
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if number > 0:
                history.append(number)
        return history

    def _migrate_legacy(self) -> None:
        with self._lock:
            if self._data["accounts"] or not self.legacy_path.exists():
                return
            try:
                legacy = json.loads(self.legacy_path.read_text(encoding="utf-8"))
                username, key = str(legacy.get("username", "")).strip(), str(legacy.get("key", "")).strip()
            except Exception:
                return
            if not username or not key:
                return
            account = self._new_account(username, key, "server", "ACTIVE")
            self._data = {"accounts": [account], "active_account_id": account["id"]}
            self._save_locked()

    @staticmethod
    def _public_proxy(proxy: str) -> str:
        if not proxy or proxy in {"server", "default", "webshare"}:
            return proxy or "server"
        try:
            parsed = urlsplit(proxy)
            if "@" not in parsed.netloc:
                return proxy
            host = parsed.netloc.rsplit("@", 1)[1]
            return urlunsplit((parsed.scheme, f"***:***@{host}", parsed.path, "", ""))
        except Exception:
            return "custom proxy"

    @staticmethod
    def status_display(status: str) -> str:
        return "VERIFIED · READY" if status == "VERIFIED_READY" else status

    def _refresh_week_locked(self, account: Dict[str, Any], now: Optional[float] = None) -> bool:
        now = now or time.time()
        start = float(account.get("quota_window_started_at") or 0)
        if not start or now < start + WEEK_SECONDS:
            return False
        session_active = bool(account.get("session_started_at"))
        account.update({
            "quota_window_started_at": now if session_active else 0.0,
            "gpu_used_seconds": 0.0, "tpu_used_seconds": 0.0,
            "quota_exhausted_accelerator": "", "updated_at": now,
        })
        if session_active:
            account["session_started_at"] = now
        if account.get("status") == "QUOTA_EXHAUSTED":
            account["status"] = "ACTIVE" if account["id"] == self._data.get("active_account_id") else "VERIFIED_READY"
        return True

    def refresh_weekly_quotas(self) -> List[str]:
        reset_ids = []
        with self._lock:
            changed = False
            for account in self._data["accounts"]:
                if self._refresh_week_locked(account):
                    reset_ids.append(account["id"])
                    changed = True
                if account.get("status") == "COOLDOWN" and float(account.get("cooldown_until") or 0) <= time.time():
                    account["status"] = "VERIFIED_READY" if account.get("verified_at") else "VERIFYING"
                    account["cooldown_until"] = 0.0
                    changed = True
            if changed:
                self._save_locked()
        return reset_ids

    def _usage_seconds_locked(self, account: Dict[str, Any], accelerator: str, include_live: bool = True) -> float:
        used = float(account.get(f"{accelerator}_used_seconds") or 0)
        if include_live and account.get("session_accelerator") == accelerator and account.get("session_started_at"):
            used += max(0.0, time.time() - float(account["session_started_at"]))
        return used

    def quota_snapshot(self, account: Dict[str, Any], accelerator: str) -> Dict[str, Any]:
        limit = QUOTA_SECONDS[accelerator]
        used = self._usage_seconds_locked(account, accelerator)
        start = float(account.get("quota_window_started_at") or 0)
        return {
            "used_seconds": used, "used_hours": round(used / 3600.0, 2), "limit_seconds": limit,
            "limit_hours": limit / 3600.0, "remaining_seconds": max(0.0, limit - used),
            "remaining_hours": round(max(0.0, limit - used) / 3600.0, 2),
            "window_started_at": start or None, "reset_at": (start + WEEK_SECONDS) if start else None,
        }

    def list_public(self) -> List[Dict[str, Any]]:
        with self._lock:
            changed, now = False, time.time()
            for account in self._data["accounts"]:
                changed = self._refresh_week_locked(account, now) or changed
            if changed:
                self._save_locked()
            result = []
            for index, account in enumerate(self._data["accounts"]):
                elapsed = max(0.0, now - account["session_started_at"]) if account.get("session_started_at") else 0.0
                result.append({
                    "id": account["id"], "queue_position": index + 1, "username": account["username"],
                    "proxy": self._public_proxy(account["proxy"]), "status": account["status"],
                    "proxy_mode": account["proxy"] if account["proxy"] in {"default", "webshare", "server"} else "custom",
                    "assigned_proxy": self._public_proxy(account.get("assigned_proxy_url", "")),
                    "last_exit_ip": account.get("last_exit_ip") or "",
                    "status_display": self.status_display(account["status"]), "verified": bool(account["verified_at"]),
                    "verified_at": account["verified_at"] or None, "verification_error": account["verification_error"],
                    "key_present": bool(account["key"]),
                    "key_hint": (account["key"][:5] + "..." + account["key"][-4:]) if len(account["key"]) > 12 else "Present",
                    "cooldown_until": account["cooldown_until"], "last_error": account["last_error"],
                    "gpu_quota": self.quota_snapshot(account, "gpu"), "tpu_quota": self.quota_snapshot(account, "tpu"),
                    "session_id": account.get("session_id") or None, "session_accelerator": account.get("session_accelerator") or None,
                    "session_model_id": account.get("session_model_id") or None, "session_elapsed_seconds": elapsed,
                    "session_remaining_seconds": max(0.0, SESSION_LIMIT_SECONDS - elapsed),
                    "last_boot_seconds": account["last_boot_seconds"] or None,
                    "boot_estimate_gpu_seconds": self._boot_estimate_locked(account, "gpu", 270.0),
                    "boot_estimate_tpu_seconds": self._boot_estimate_locked(account, "tpu", 570.0),
                })
            return result

    def get(self, account_id: Optional[str]) -> Optional[Dict[str, Any]]:
        with self._lock:
            account = next((item for item in self._data["accounts"] if item["id"] == account_id), None)
            if account:
                if self._refresh_week_locked(account):
                    self._save_locked()
                return dict(account)
            return None

    def active(self) -> Optional[Dict[str, Any]]:
        return self.get(self._data.get("active_account_id"))

    def add(self, username: str, key: str, proxy: str = "server", status: Optional[str] = None) -> Dict[str, Any]:
        username, key = str(username).strip(), str(key).strip()
        proxy = str(proxy or "server").strip() or "server"
        if not username or not key:
            raise ValueError("Kaggle username and API key are required")
        with self._lock:
            duplicate = next((item for item in self._data["accounts"] if item["username"].lower() == username.lower()), None)
            if duplicate:
                duplicate.update({
                    "key": key, "proxy": proxy, "status": status or "VERIFYING", "verified_at": 0.0,
                    "assigned_proxy_url": "", "last_exit_ip": "",
                    "verification_error": "", "last_error": "", "updated_at": time.time(),
                })
                account = duplicate
            else:
                account = self._new_account(username, key, proxy, status or "VERIFYING")
                self._data["accounts"].append(account)
            self._save_locked()
            return dict(account)

    def delete(self, account_id: str) -> bool:
        with self._lock:
            before = len(self._data["accounts"])
            self._data["accounts"] = [account for account in self._data["accounts"] if account["id"] != account_id]
            if len(self._data["accounts"]) == before:
                return False
            if self._data.get("active_account_id") == account_id:
                self._data["active_account_id"] = None
            self._save_locked()
            return True

    def update_proxy(
        self,
        account_id: str,
        proxy: str,
        assigned_proxy_url: str = "",
        last_exit_ip: str = "",
    ) -> Dict[str, Any]:
        with self._lock:
            account = next((item for item in self._data["accounts"] if item["id"] == account_id), None)
            if not account:
                raise KeyError(account_id)
            account["proxy"] = str(proxy or "server").strip() or "server"
            account["assigned_proxy_url"] = str(assigned_proxy_url or "").strip()
            account["last_exit_ip"] = str(last_exit_ip or "").strip()
            account["updated_at"] = time.time()
            self._save_locked()
            return dict(account)

    def set_verification(self, account_id: str, verified: bool, error: str = "", username: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            account = next((item for item in self._data["accounts"] if item["id"] == account_id), None)
            if not account:
                raise KeyError(account_id)
            if username:
                account["username"] = username.strip()
            account["verified_at"] = time.time() if verified else 0.0
            account["verification_error"] = "" if verified else str(error)[:500]
            account["last_error"] = "" if verified else str(error)[:500]
            if verified:
                if self._data.get("active_account_id") == account_id and account["status"] not in {"PREWARMING", "HOT_STANDBY", "TESTING_BOOT"}:
                    account["status"] = "ACTIVE"
                elif account["status"] not in {"ACTIVE", "PREWARMING", "HOT_STANDBY", "TESTING_BOOT"}:
                    account["status"] = "VERIFIED_READY"
            else:
                account["status"] = "INVALID_KEY"
                if self._data.get("active_account_id") == account_id:
                    self._data["active_account_id"] = None
            account["updated_at"] = time.time()
            self._save_locked()
            return dict(account)

    def activate(self, account_id: str) -> Dict[str, Any]:
        with self._lock:
            target = next((item for item in self._data["accounts"] if item["id"] == account_id), None)
            if not target:
                raise KeyError(account_id)
            if not target.get("verified_at"):
                raise ValueError("Account key is not verified")
            for account in self._data["accounts"]:
                if account["id"] == account_id:
                    account.update({"status": "ACTIVE", "cooldown_until": 0.0, "last_error": "", "updated_at": time.time()})
                elif account["status"] == "ACTIVE":
                    account["status"] = "VERIFIED_READY" if account["verified_at"] else "VERIFYING"
            self._data["active_account_id"] = account_id
            self._save_locked()
            return dict(target)

    def mark(self, account_id: str, status: str, error: str = "", cooldown_seconds: int = 0) -> None:
        status = LEGACY_STATUS_MAP.get(status.upper(), status.upper())
        if status not in ACCOUNT_STATUSES:
            raise ValueError(status)
        with self._lock:
            account = next((item for item in self._data["accounts"] if item["id"] == account_id), None)
            if not account:
                return
            account.update({
                "status": status, "last_error": str(error)[:500], "updated_at": time.time(),
                "cooldown_until": time.time() + cooldown_seconds if cooldown_seconds else 0.0,
            })
            self._save_locked()

    def clear_cooldown(self, account_id: Optional[str] = None) -> List[str]:
        """Return verified accounts to a usable state without changing quota ledgers."""
        cleared: List[str] = []
        with self._lock:
            for account in self._data["accounts"]:
                if account_id and account["id"] != account_id:
                    continue
                if account.get("status") in {"COOLDOWN", "VERIFYING"} or account.get("cooldown_until") or account.get("last_error"):
                    account["status"] = (
                        "ACTIVE" if self._data.get("active_account_id") == account["id"] and account.get("verified_at")
                        else "VERIFIED_READY" if account.get("verified_at")
                        else "VERIFYING"
                    )
                    account["cooldown_until"] = 0.0
                    account["last_error"] = ""
                    account["updated_at"] = time.time()
                    cleared.append(account["id"])
            if cleared:
                self._save_locked()
        return cleared

    def reset_weekly_quota(self, account_id: str) -> Dict[str, Any]:
        with self._lock:
            account = next((item for item in self._data["accounts"] if item["id"] == account_id), None)
            if not account:
                raise KeyError(account_id)
            now = time.time()
            session_active = bool(account.get("session_started_at"))
            account.update({
                "quota_window_started_at": now if session_active else 0.0,
                "gpu_used_seconds": 0.0, "tpu_used_seconds": 0.0,
                "quota_exhausted_accelerator": "", "last_error": "", "updated_at": time.time(),
            })
            if session_active:
                account["session_started_at"] = now
            if account["status"] == "QUOTA_EXHAUSTED":
                account["status"] = "ACTIVE" if self._data.get("active_account_id") == account_id else "VERIFIED_READY"
            self._save_locked()
            return dict(account)

    def begin_session(self, account_id: str, session_id: str, accelerator: str, model_id: str, started_at: Optional[float] = None) -> None:
        with self._lock:
            account = next((item for item in self._data["accounts"] if item["id"] == account_id), None)
            if not account:
                raise KeyError(account_id)
            now = started_at or time.time()
            self._refresh_week_locked(account, now)
            if not account["quota_window_started_at"]:
                account["quota_window_started_at"] = now
            account.update({
                "session_started_at": now, "session_accelerator": accelerator, "session_id": session_id,
                "session_model_id": model_id, "updated_at": now,
            })
            self._save_locked()

    def end_session(self, account_id: str, session_id: Optional[str] = None, ended_at: Optional[float] = None) -> float:
        with self._lock:
            account = next((item for item in self._data["accounts"] if item["id"] == account_id), None)
            if not account or not account.get("session_started_at"):
                return 0.0
            if session_id and account.get("session_id") != session_id:
                return 0.0
            now = ended_at or time.time()
            duration = max(0.0, now - float(account["session_started_at"]))
            accelerator = account.get("session_accelerator")
            if accelerator in QUOTA_SECONDS:
                field = f"{accelerator}_used_seconds"
                account[field] = float(account.get(field) or 0) + duration
                if account[field] >= QUOTA_SECONDS[accelerator]:
                    account["status"] = "QUOTA_EXHAUSTED"
                    account["quota_exhausted_accelerator"] = accelerator
            account.update({
                "session_started_at": 0.0, "session_accelerator": "", "session_id": "", "session_model_id": "",
                "last_session_seconds": duration, "updated_at": now,
            })
            self._save_locked()
            return duration

    def remaining_seconds(self, account_id: str, accelerator: str) -> float:
        with self._lock:
            account = next((item for item in self._data["accounts"] if item["id"] == account_id), None)
            if not account:
                return 0.0
            self._refresh_week_locked(account)
            return max(0.0, QUOTA_SECONDS[accelerator] - self._usage_seconds_locked(account, accelerator))

    def is_ready_for(self, account: Dict[str, Any], accelerator: str) -> bool:
        return bool(
            account.get("verified_at")
            and account.get("status") == "VERIFIED_READY"
            and not account.get("session_id")
            and self.remaining_seconds(account["id"], accelerator) > 0
        )

    def next_ready_after(self, account_id: Optional[str], accelerator: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            self.refresh_weekly_quotas()
            accounts = self._data["accounts"]
            if not accounts:
                return None
            start = next((index for index, account in enumerate(accounts) if account["id"] == account_id), -1)
            for offset in range(1, len(accounts) + 1):
                candidate = accounts[(start + offset) % len(accounts)]
                if self.is_ready_for(candidate, accelerator):
                    return dict(candidate)
            return None

    def mark_quota_exhausted(self, account_id: str, accelerator: str, error: str = "") -> None:
        with self._lock:
            account = next((item for item in self._data["accounts"] if item["id"] == account_id), None)
            if not account:
                return
            account.update({
                "status": "QUOTA_EXHAUSTED", "quota_exhausted_accelerator": accelerator,
                "last_error": error[:500], "updated_at": time.time(),
            })
            self._save_locked()

    def record_boot(self, account_id: str, accelerator: str, seconds: float) -> None:
        if seconds <= 0:
            return
        with self._lock:
            account = next((item for item in self._data["accounts"] if item["id"] == account_id), None)
            if not account:
                return
            field = f"boot_history_{accelerator}"
            history = self._clean_history(account.get(field, []))
            history.append(float(seconds))
            account[field] = history[-20:]
            account["last_boot_seconds"] = float(seconds)
            account["updated_at"] = time.time()
            self._save_locked()

    def _boot_estimate_locked(self, account: Dict[str, Any], accelerator: str, baseline: float) -> float:
        history = self._clean_history(account.get(f"boot_history_{accelerator}", []))
        return round(statistics.median(history[-7:]) if history else baseline, 1)

    def boot_estimate(self, account_id: str, accelerator: str, baseline: float) -> float:
        with self._lock:
            account = next((item for item in self._data["accounts"] if item["id"] == account_id), None)
            return self._boot_estimate_locked(account, accelerator, baseline) if account else baseline

    def counts(self) -> Dict[str, int]:
        with self._lock:
            result = {status: 0 for status in ACCOUNT_STATUSES}
            for account in self._data["accounts"]:
                result[account["status"]] += 1
            return result

    @staticmethod
    def parse_bulk(text: str, default_proxy: str = "server") -> List[Tuple[str, str, str]]:
        text = str(text or "").strip()
        if not text:
            return []
        parsed: List[Tuple[str, str, str]] = []
        try:
            obj = json.loads(text)
            items = obj if isinstance(obj, list) else [obj]
            for item in items:
                if not isinstance(item, dict):
                    continue
                username, key = str(item.get("username", "")).strip(), str(item.get("key", "")).strip()
                proxy = str(item.get("proxy") or default_proxy).strip() or "server"
                if key:
                    parsed.append((username, key, proxy))
            if parsed:
                return parsed
        except json.JSONDecodeError:
            pass
        for line in text.splitlines():
            line = line.strip().strip(",")
            if not line or line.startswith("#"):
                continue
            if line.startswith("{"):
                try:
                    item = json.loads(line)
                    username, key = str(item.get("username", "")).strip(), str(item.get("key", "")).strip()
                    proxy = str(item.get("proxy") or default_proxy).strip() or "server"
                    if key:
                        parsed.append((username, key, proxy))
                        continue
                except (json.JSONDecodeError, AttributeError):
                    pass
            if ":" not in line:
                if len(line) >= 16:
                    parsed.append(("", line, default_proxy))
                continue
            username, key = line.split(":", 1)
            username, key = username.strip(), key.strip()
            if username and key:
                parsed.append((username, key, default_proxy))
        return parsed
