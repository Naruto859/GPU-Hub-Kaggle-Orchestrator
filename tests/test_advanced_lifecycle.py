import asyncio
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
from fastapi import HTTPException

os.environ["GPU_HUB_DISABLE_BACKGROUND"] = "1"

import kaggle_manager as km
import models_registry as registry
from account_pool import KaggleAccountPool, QUOTA_SECONDS, WEEK_SECONDS
from models_registry import MODELS
from proxy_service import InferenceProxy
from webshare_manager import ProxyRouteError


def make_pool(root: Path, count: int = 2):
    legacy = root / "legacy.json"
    legacy.write_text(json.dumps({"username": "user1", "key": "key-user1-12345"}))
    pool = KaggleAccountPool(root / "pool.json", legacy)
    first = pool.active()
    pool.set_verification(first["id"], True)
    ids = [first["id"]]
    for index in range(2, count + 1):
        account = pool.add(f"user{index}", f"key-user{index}-12345")
        pool.set_verification(account["id"], True)
        ids.append(account["id"])
    return pool, ids


class AccountPoolTests(unittest.TestCase):
    def test_bulk_accepts_key_only_lines(self):
        parsed = KaggleAccountPool.parse_bulk("personal-token-1234567890\nanother-token-abcdefghijkl")
        self.assertEqual(parsed[0][0], "")
        self.assertEqual(parsed[0][1], "personal-token-1234567890")

    def test_rotation_usage_and_weekly_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            pool, ids = make_pool(Path(directory), 8)
            self.assertEqual(pool.next_ready_after(ids[0], "gpu")["id"], ids[1])
            pool.begin_session(ids[0], "session", "gpu", "model", time.time() - 3600)
            pool.end_session(ids[0], "session")
            self.assertGreaterEqual({a["id"]: a for a in pool.list_public()}[ids[0]]["gpu_quota"]["used_hours"], 1)
            with pool._lock:
                account = next(a for a in pool._data["accounts"] if a["id"] == ids[0])
                account["gpu_used_seconds"] = QUOTA_SECONDS["gpu"]
                account["quota_window_started_at"] = time.time() - WEEK_SECONDS - 1
                pool._save_locked()
            pool.mark_quota_exhausted(ids[0], "gpu", "quota")
            self.assertIn(ids[0], pool.refresh_weekly_quotas())
            reset = {a["id"]: a for a in pool.list_public()}[ids[0]]
            self.assertEqual(reset["gpu_quota"]["used_hours"], 0)
            self.assertEqual(reset["status"], "ACTIVE")


class LifecycleTests(unittest.TestCase):
    def test_cli_warning_cleaner_preserves_real_error(self):
        raw = "Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4)\nuser/kernel has status \"KernelWorkerStatus.ERROR\""
        self.assertEqual(km.clean_kaggle_output(raw), 'user/kernel has status "KernelWorkerStatus.ERROR"')

    def test_warming_timer_uses_session_and_clears_stuck_state(self):
        manager = km.KaggleKernelManager()
        session = km.KernelSession(
            "warm", "active", "account", "cybertiel-coder-35b-a3b", "gpu",
            launch_time=time.time() - 12, status="queued",
        )
        manager.sessions = {session.id: session}
        manager.warming_session_id = session.id
        manager.status = "warming"
        state = manager.get_state()
        self.assertGreaterEqual(state["elapsed_seconds"], 11)
        self.assertAlmostEqual(state["launch_timestamp"], session.launch_time, delta=0.1)
        session.status = "failed"
        manager.warming_session_id = None
        state = manager.get_state()
        self.assertNotEqual(state["status"], "warming")
        self.assertIsNone(state["launch_timestamp"])

    def test_key_only_username_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = km.KaggleKernelManager()
            manager.account_pool = KaggleAccountPool(root / "pool.json", root / "none.json")

            class Result:
                returncode = 0
                stdout = "ref,title"
                stderr = ""

            with patch("kaggle_manager.detect_kaggle_username", return_value="detected-user"), \
                 patch("kaggle_manager.subprocess.run", return_value=Result()):
                account = manager.add_kaggle_account("", "personal-token-1234567890")
            self.assertEqual(account["username"], "detected-user")
            self.assertTrue(account["verified_at"])

    def test_single_key_safety_and_circuit_breaker(self):
        with tempfile.TemporaryDirectory() as directory:
            pool, ids = make_pool(Path(directory), 1)
            manager = km.KaggleKernelManager()
            manager.account_pool = pool
            manager.last_inference_activity = time.time() - 3600
            active = km.KernelSession(
                "active", "active", ids[0], "cybertiel-coder-35b-a3b", "gpu",
                launch_time=time.time() - (8 * 3600 + 46 * 60), tunnel_url="https://old", status="ready",
            )
            manager.sessions = {active.id: active}
            manager.active_session_id = active.id
            manager.tunnel_url = active.tunnel_url
            scheduled = []
            manager._schedule_single_key_transition = lambda session, remaining, restart: scheduled.append(restart) or True
            self.assertEqual(manager.lifecycle_tick(), "single_key_restart")
            self.assertEqual(scheduled, [True])
            manager._trip_circuit_breaker("test circuit")
            self.assertEqual(manager.status, "quota_wait")
            self.assertFalse(manager._circuit_allows_attempt())

    def test_qwen_tpu_bundle_and_custom_model_persistence(self):
        manager = km.KaggleKernelManager()
        manager.client_api_key = "sk-test-123456789"
        manager.upstream_api_key = manager.client_api_key
        manager.resolve_kernel_identity = lambda force=False, model_id=None, account_id=None: ("friend", "qwen38-tpu-serve", "Qwen38 TPU")
        old_runtime = km.RUNTIME_BUNDLE_DIR
        old_custom = registry.CUSTOM_MODELS_PATH
        with tempfile.TemporaryDirectory() as directory:
            km.RUNTIME_BUNDLE_DIR = Path(directory) / "bundles"
            session = km.KernelSession("qwen", "test", "account", "qwen3.8-27b-tpu", "tpu")
            bundle = manager._prepare_session_bundle(session)
            metadata = json.loads((bundle / "kernel-metadata.json").read_text())
            self.assertTrue(metadata["enable_tpu"])
            self.assertEqual(metadata["dataset_sources"], km.QWEN_TPU_DATASETS)
            compile((bundle / "kernel.py").read_text(), "<qwen-tpu>", "exec")

            registry.CUSTOM_MODELS_PATH = Path(directory) / "custom_models.json"
            custom = registry.save_custom_model({
                "name": "Example Coder", "repo_id": "example/coder-gguf",
                "filename": "coder-Q4_K_M.gguf", "context_length": 65536,
            })
            self.assertTrue(registry.CUSTOM_MODELS_PATH.exists())
            self.assertEqual(registry.load_custom_models()[custom["id"]]["context_length"], 65536)
            registry.MODELS.pop(custom["id"], None)
        km.RUNTIME_BUNDLE_DIR = old_runtime
        registry.CUSTOM_MODELS_PATH = old_custom

    def test_launch_session_retries_409_with_existing_slug(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pool, ids = make_pool(root, 1)
            manager = km.KaggleKernelManager()
            manager.account_pool = pool
            manager.client_api_key = "sk-test"
            manager.upstream_api_key = "sk-test"
            bundle = root / "bundle"
            bundle.mkdir()

            def prepare(session):
                session.bundle_dir = bundle
                session.kernel_slug = "user1/new-slug"
                session.kernel_title = "New Slug"
                (bundle / "kernel.py").write_text("print('ok')\n")
                (bundle / "kernel-metadata.json").write_text(json.dumps({"id": session.kernel_slug, "title": session.kernel_title}))
                return bundle

            manager._prepare_session_bundle = prepare
            manager.fetch_existing_kernels = lambda force=False, account_id=None: [{"ref": "user1/gpu-engine", "slug": "gpu-engine", "title": "GPU Engine", "owner": "user1"}]
            manager._monitor_session = lambda session: None

            class Conflict:
                returncode = 1
                stdout = "409 Conflict: notebook (user1/gpu-engine) already in use"
                stderr = ""

            class Success:
                returncode = 0
                stdout = "Kernel version successfully pushed"
                stderr = ""

            with patch("kaggle_manager.subprocess.run", side_effect=[Conflict(), Success()]) as run:
                result = manager._launch_session(ids[0], "cybertiel-coder-35b-a3b", "test")
            self.assertTrue(result["success"])
            self.assertEqual(run.call_count, 2)
            self.assertEqual(manager.sessions[result["session_id"]].kernel_slug, "user1/gpu-engine")

    def test_prewarm_trigger_and_atomic_swap(self):
        with tempfile.TemporaryDirectory() as directory:
            pool, ids = make_pool(Path(directory), 2)
            manager = km.KaggleKernelManager()
            manager.account_pool = pool
            manager.last_inference_activity = time.time()
            old = km.KernelSession(
                "old", "active", ids[0], "cybertiel-coder-35b-a3b", "gpu",
                launch_time=time.time() - (8 * 3600 + 46 * 60), kernel_slug="user1/gpu-engine",
                tunnel_url="https://old.test", status="ready", ready_time=time.time() - 100,
            )
            new = km.KernelSession(
                "new", "prewarm", ids[1], "cybertiel-coder-35b-a3b", "gpu",
                launch_time=time.time() - 200, kernel_slug="user2/gpu-engine",
                tunnel_url="https://new.test", status="ready", ready_time=time.time(), trigger_reason="session_limit",
            )
            manager.sessions = {"old": old}
            manager.active_session_id = "old"
            manager.tunnel_url = old.tunnel_url
            triggered = []
            manager.start_prewarm = lambda reason: triggered.append(reason) or {"success": True}
            self.assertEqual(manager.lifecycle_tick(), "session_limit")
            self.assertEqual(triggered, ["session_limit"])

            manager.sessions["new"] = new
            manager.warming_session_id = "new"
            manager.note_request_queued()
            reserved = manager.reserve_upstream()
            drained = []
            manager._drain_and_stop_old = lambda session, url, reason: drained.append((session.id, url, reason))
            manager._hot_swap(new)
            self.assertEqual(manager.tunnel_url, "https://new.test")
            self.assertEqual(manager.active_session_id, "new")
            manager.release_upstream(reserved)
            for _ in range(20):
                if drained:
                    break
                time.sleep(0.01)
            self.assertEqual(drained[0], ("old", "https://old.test", "session_limit"))

    def test_instant_key_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            pool, ids = make_pool(Path(directory), 2)
            manager = km.KaggleKernelManager()
            manager.account_pool = pool

            class Result:
                returncode = 0
                stdout = "ref,title"
                stderr = ""

            with patch("kaggle_manager.subprocess.run", return_value=Result()):
                account = manager.verify_kaggle_account(ids[1])
            self.assertTrue(account["verified_at"])

    def test_proxy_kill_switch_aborts_before_kaggle_subprocess(self):
        with tempfile.TemporaryDirectory() as directory:
            pool, ids = make_pool(Path(directory), 1)
            pool.update_proxy(ids[0], "http://user:pass@127.0.0.1:9999")
            manager = km.KaggleKernelManager()
            manager.account_pool = pool
            manager.webshare = MagicMock()
            manager.webshare.test_proxy.side_effect = ProxyRouteError("PROXY_ERROR: unreachable")
            with patch("kaggle_manager.subprocess.run") as run:
                with self.assertRaises(ProxyRouteError):
                    manager._run_kaggle(ids[0], ["kernels", "list", "--mine"])
            run.assert_not_called()

    def test_single_webshare_proxy_rotates_before_prewarm(self):
        with tempfile.TemporaryDirectory() as directory:
            pool, ids = make_pool(Path(directory), 2)
            pool.update_proxy(ids[0], "webshare", "http://proxy:8000", "10.0.0.1")
            pool.update_proxy(ids[1], "webshare")
            manager = km.KaggleKernelManager()
            manager.account_pool = pool
            active = km.KernelSession(
                "active", "active", ids[0], "cybertiel-coder-35b-a3b", "gpu",
                tunnel_url="https://old.test", status="ready",
            )
            manager.sessions = {active.id: active}
            manager.active_session_id = active.id
            manager.webshare = MagicMock()
            manager.webshare.proxy_count.return_value = 1
            manager.webshare.proxy_for_index.return_value = {"proxy_url": "http://proxy:8000"}
            manager.webshare.public_status.return_value = {"auto_refresh": True}
            manager.webshare.refresh_ip.return_value = {
                "exit_ip": "10.0.0.2", "proxies": [{"proxy_url": "http://proxy:8000"}],
            }
            route = manager._prepare_account_route(ids[1], "prewarm", ids[0])
            self.assertEqual(route, "http://proxy:8000")
            self.assertTrue(active.kaggle_poll_frozen)
            manager.webshare.refresh_ip.assert_called_once_with(previous_exit_ip="10.0.0.1", wait_seconds=5)
            self.assertEqual(pool.get(ids[1])["last_exit_ip"], "10.0.0.2")

    def test_multiple_webshare_proxies_pin_distinct_exit_ips(self):
        with tempfile.TemporaryDirectory() as directory:
            pool, ids = make_pool(Path(directory), 2)
            pool.update_proxy(ids[0], "webshare")
            pool.update_proxy(ids[1], "webshare")
            manager = km.KaggleKernelManager()
            manager.account_pool = pool
            proxies = [{"proxy_url": "http://proxy-a:8000"}, {"proxy_url": "http://proxy-b:8000"}]
            manager.webshare = MagicMock()
            manager.webshare.proxy_count.return_value = 2
            manager.webshare.proxy_for_index.side_effect = lambda index: proxies[index]
            manager.webshare.test_proxy.side_effect = lambda url, force=False: {
                "http://proxy-a:8000": "10.0.0.1", "http://proxy-b:8000": "10.0.0.2",
            }[url]
            manager._prepare_account_route(ids[0])
            manager._prepare_account_route(ids[1])
            self.assertNotEqual(pool.get(ids[0])["last_exit_ip"], pool.get(ids[1])["last_exit_ip"])


class ProxyQueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_warmup_status_headers(self):
        class Manager:
            tunnel_url = ""
            status = "warming"

            def warmup_headers(self):
                return {"X-GPUHub-Warmup-Remaining": "420", "Retry-After": "60"}

        proxy = InferenceProxy(Manager(), MODELS, lambda: "key", lambda: False, lambda body: {}, lambda: False)
        with self.assertRaises(HTTPException) as raised:
            await proxy.ensure_tunnel()
        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(raised.exception.headers["X-GPUHub-Warmup-Remaining"], "420")

    async def test_fifo_reserves_one_upstream(self):
        class Manager:
            active_model_id = "cybertiel-coder-35b-a3b"
            tunnel_url = "https://old.test"
            upstream_api_key = "key"
            status = "ready"
            status_detail = ""

            def __init__(self):
                self.queued_requests = 0
                self.active_requests = 0
                self.counts = {}

            def note_request_queued(self):
                self.queued_requests += 1

            def note_request_aborted(self):
                self.queued_requests = max(0, self.queued_requests - 1)

            def reserve_upstream(self):
                url = self.tunnel_url
                self.queued_requests -= 1
                self.active_requests += 1
                self.counts[url] = self.counts.get(url, 0) + 1
                return url

            def release_upstream(self, url):
                self.active_requests -= 1
                self.counts[url] -= 1

            def warmup_headers(self):
                return {"X-GPUHub-Warmup-Remaining": "300"}

        manager = Manager()
        upstream_active = 0
        max_active = 0

        async def handler(request):
            nonlocal upstream_active, max_active
            upstream_active += 1
            max_active = max(max_active, upstream_active)
            await asyncio.sleep(0.01)
            upstream_active -= 1
            return httpx.Response(200, json={"ok": True})

        proxy = InferenceProxy(manager, MODELS, lambda: "key", lambda: False, lambda body: {}, lambda: False)
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await asyncio.gather(*(proxy.raw_proxy("/v1/chat/completions", {"i": i}, False) for i in range(3)))
        await proxy._client.aclose()
        self.assertEqual(max_active, 1)
        self.assertEqual(manager.active_requests, 0)
        self.assertEqual(manager.queued_requests, 0)


if __name__ == "__main__":
    unittest.main()
