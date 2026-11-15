import asyncio
import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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
        manager.accelerator_mode = "tpu"
        manager.resolve_kernel_identity = lambda force=False, model_id=None, account_id=None, accelerator=None: ("friend", "qwen38-tpu-serve", "Qwen38 TPU")
        old_runtime = km.RUNTIME_BUNDLE_DIR
        old_custom = registry.CUSTOM_MODELS_PATH
        with tempfile.TemporaryDirectory() as directory:
            km.RUNTIME_BUNDLE_DIR = Path(directory) / "bundles"
            session = km.KernelSession("qwen", "test", "account", "qwen3.8-27b-instruct", "tpu")
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

    def test_non_default_gpu_model_uses_model_specific_kernel_slug(self):
        with tempfile.TemporaryDirectory() as directory:
            pool, ids = make_pool(Path(directory), 1)
            manager = km.KaggleKernelManager()
            manager.account_pool = pool
            manager.fetch_existing_kernels = lambda force=False, account_id=None: [
                {"ref": "user1/gpu-engine", "slug": "gpu-engine", "title": "GPU Engine", "owner": "user1"}
            ]
            owner, slug, title = manager.resolve_kernel_identity(
                force=True,
                model_id="qwen3.8-35b-a3b-distill",
                account_id=ids[0],
                accelerator="gpu",
            )
            self.assertEqual(owner, "user1")
            self.assertEqual(slug, "gpu-hub-qwen3-8-35b-a3b-distill")
            self.assertEqual(title, "GPU Hub qwen3.8-35b-a3b-distill")
            self.assertLessEqual(len(title), 50)

    def test_non_default_409_never_adopts_unrelated_legacy_gpu_kernel(self):
        manager = km.KaggleKernelManager()
        session = km.KernelSession("qwen", "active", "account", "qwen3.8-35b-a3b-distill", "gpu")
        session.kernel_slug = "user1/gpu-hub-qwen3-8-35b-a3b-distill"
        manager.fetch_existing_kernels = lambda force=False, account_id=None: [
            {"ref": "user1/gpu-engine", "slug": "gpu-engine", "title": "GPU Engine", "owner": "user1"}
        ]
        self.assertFalse(manager._adopt_conflicting_kernel(session, "409 Conflict: user1/gpu-engine already has an active session"))

    def test_cancelled_kernel_status_short_circuits_slow_log_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            pool, ids = make_pool(Path(directory), 1)
            manager = km.KaggleKernelManager()
            manager.account_pool = pool
            manager.auto_key_rotation = False
            session = km.KernelSession("cancel", "active", ids[0], "qwen3.8-35b-a3b-distill", "gpu")
            session.kernel_slug = "user1/gpu-hub-qwen3-8-35b-a3b-distill"
            manager.sessions[session.id] = session
            manager.warming_session_id = session.id
            session.stop_event.wait = MagicMock(return_value=False)
            result = MagicMock(returncode=0, stdout='status "KernelWorkerStatus.CANCEL_ACKNOWLEDGED"', stderr="")
            manager._run_kaggle = MagicMock(return_value=result)
            manager._read_session_logs = MagicMock(side_effect=AssertionError("slow log probe should not run"))
            manager._monitor_session(session)
            self.assertEqual(session.status, "failed")
            manager._read_session_logs.assert_not_called()

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

    def test_default_route_resolves_without_changing_account_override(self):
        with tempfile.TemporaryDirectory() as directory:
            pool, ids = make_pool(Path(directory), 1)
            pool.update_proxy(ids[0], "default")
            manager = km.KaggleKernelManager()
            manager.account_pool = pool
            manager.default_network_route = "server"
            self.assertEqual(manager._prepare_account_route(ids[0]), "server")
            self.assertEqual(pool.get(ids[0])["proxy"], "default")

    def test_tpu_launch_deletes_stale_slug_before_push(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pool, ids = make_pool(root, 1)
            manager = km.KaggleKernelManager()
            manager.account_pool = pool
            manager.client_api_key = manager.upstream_api_key = "test-client-key"
            bundle = root / "bundle"
            bundle.mkdir()
            (bundle / "kernel.py").write_text("print('ok')\n")
            (bundle / "kernel-metadata.json").write_text(json.dumps({"id": "user1/glm53-tpu-serve"}))

            def prepare(session):
                session.bundle_dir = bundle
                session.kernel_slug = "user1/glm53-tpu-serve"
                session.kernel_title = "GLM53 TPU Serve"
                return bundle

            class Result:
                returncode = 0
                stdout = "Kernel version successfully pushed"
                stderr = ""

            manager._prepare_session_bundle = prepare
            manager.accelerator_mode = "tpu"
            manager._monitor_session = lambda session: None
            manager._delete_kernel_for_fresh_tpu = MagicMock()
            manager._run_kaggle = MagicMock(return_value=Result())
            with patch("kaggle_manager.time.sleep"):
                result = manager._launch_session(ids[0], "glm-5.3-flash", "active")
            self.assertTrue(result["success"])
            manager._delete_kernel_for_fresh_tpu.assert_called_once_with(ids[0], "user1/glm53-tpu-serve")

    def test_generated_gpu_kernel_has_valid_webhook_and_runtime_fstrings(self):
        manager = km.KaggleKernelManager()
        manager.client_api_key = manager.upstream_api_key = "test-client-key"
        manager.webhook_host = "https://manager.example"
        source = manager.generate_kernel_script("cybertiel-coder-35b-a3b", "session-id", "active")
        self.assertIn('https://manager.example/api/tunnel/register', source)
        self.assertIn('payload = json.dumps({"tunnel_url": url', source)
        self.assertNotIn('json.dumps({{"tunnel_url"', source)
        self.assertIn('b12395', source)
        self.assertIn('resolve_gguf_filename', source)
        self.assertIn('[SUPERVISOR]', source)
        compile(source, "<generated-gpu-kernel>", "exec")

    def test_live_log_fallback_samples_follow_stream_for_cli_224(self):
        manager = km.KaggleKernelManager()
        session = km.KernelSession("gpu", "active", "account", "cybertiel-coder-35b-a3b", "gpu", kernel_slug="user/gpu-engine")

        class Empty:
            returncode = 0
            stdout = ""
            stderr = ""

        timeout = subprocess.TimeoutExpired(
            ["kaggle", "kernels", "logs", "-f"], 5,
            output='[STEP 5] Tunnel\n[TUNNEL ESTABLISHED] URL: https://example.trycloudflare.com\n',
            stderr="",
        )
        manager._run_kaggle = MagicMock(side_effect=[Empty(), timeout])
        lines, idle, failure = manager._read_session_logs(session, set())
        self.assertTrue(any("example.trycloudflare.com" in line for line in lines))
        self.assertFalse(idle)
        self.assertEqual(failure, "")

    def test_recovering_tunnel_never_reports_fake_ready(self):
        manager = km.KaggleKernelManager()
        session = km.KernelSession("live", "active", "account", "cybertiel-coder-35b-a3b", "gpu", tunnel_url="https://dead.test", status="ready")
        manager.sessions = {session.id: session}
        manager.active_session_id = session.id
        manager.tunnel_url = session.tunnel_url
        manager.status = "recovering"
        manager.status_detail = "health checks failed"
        state = manager.get_state()
        self.assertEqual(state["status"], "recovering")
        self.assertNotEqual(state["status"], "ready")

    def test_no_tpu_recreates_without_cooldown_or_circuit_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pool, ids = make_pool(root, 1)
            manager = km.KaggleKernelManager()
            manager.account_pool = pool
            bundle = root / "bundle"
            bundle.mkdir()
            (bundle / "kernel-metadata.json").write_text(json.dumps({"id": "user1/glm53-tpu-serve", "title": "GLM53 TPU Serve"}))
            session = km.KernelSession("tpu", "active", ids[0], "glm-5.3-flash", "tpu", kernel_slug="user1/glm53-tpu-serve", kernel_title="GLM53 TPU Serve", bundle_dir=bundle)
            session.stop_event = MagicMock()
            session.stop_event.wait.return_value = False
            manager.sessions = {session.id: session}
            manager.active_session_id = session.id
            manager.warming_session_id = session.id
            manager._delete_kernel_for_fresh_tpu = MagicMock()
            manager._push_existing_session_bundle = MagicMock(return_value=(True, "pushed"))
            self.assertTrue(manager._recreate_tpu_session(session, "this session has no working TPU: jax sees 1 cpu device"))
            self.assertEqual(session.tpu_recreate_attempts, 1)
            self.assertEqual(session.kernel_slug, "user1/glm53-tpu-serve-r1")
            self.assertEqual(session.status, "queued")
            self.assertEqual(manager.consecutive_launch_failures, 0)
            self.assertNotEqual(pool.get(ids[0])["status"], "COOLDOWN")

    def test_no_tpu_retry_exhaustion_keeps_verified_account_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            pool, ids = make_pool(Path(directory), 1)
            pool.activate(ids[0])
            manager = km.KaggleKernelManager()
            manager.account_pool = pool
            session = km.KernelSession("tpu", "active", ids[0], "glm-5.3-flash", "tpu", kernel_slug="user1/glm53-tpu-serve", tpu_recreate_attempts=4)
            manager.sessions = {session.id: session}
            manager.active_session_id = session.id
            self.assertFalse(manager._recreate_tpu_session(session, "jax sees 1 cpu device"))
            self.assertEqual(pool.get(ids[0])["status"], "ACTIVE")
            self.assertEqual(manager.consecutive_launch_failures, 0)
            self.assertEqual(manager.status, "error")


class FrontendContractTests(unittest.TestCase):
    def test_mobile_ui_uses_no_native_select_or_browser_prompts(self):
        html = (Path(__file__).parents[1] / "static" / "index.html").read_text()
        self.assertNotIn("<select", html.lower())
        self.assertNotIn("<option", html.lower())
        self.assertNotIn("prompt(", html)
        self.assertNotIn("alert(", html)
        self.assertIn("AUTO-SWITCHING &AMP; NETWORK SHIELD", html.upper())
        self.assertIn("KAGGLE API KEY POOL", html.upper())

    def test_conditional_route_and_accelerator_controls_are_custom(self):
        html = (Path(__file__).parents[1] / "static" / "index.html").read_text()
        for marker in ("webshareShield", "customProxyShield", "applyRouteVisibility", "acceleratorMode", "setAcceleratorMode"):
            self.assertIn(marker, html)
        self.assertIn("Requires TPU v5e-8", html)
        self.assertIn("Requires Dual T4 GPU", html)


class V5RouteAndModelTests(unittest.TestCase):
    def test_fresh_webshare_connect_assigns_inherited_accounts_then_server_clears_route(self):
        with tempfile.TemporaryDirectory() as directory:
            pool, ids = make_pool(Path(directory), 1)
            pool.update_proxy(ids[0], "default", proxy_override=False)
            manager = km.KaggleKernelManager()
            manager.account_pool = pool
            manager.webshare = MagicMock()
            manager.webshare.connect.return_value = {"connected": True, "proxy_count": 1, "exit_ip": "10.0.0.2"}
            manager.webshare.proxy_count.return_value = 1
            manager.webshare.proxy_for_index.return_value = {"proxy_url": "http://proxy:8000"}
            manager.webshare.test_proxy.return_value = "10.0.0.2"
            connected = manager.connect_webshare("token")
            self.assertTrue(connected["connected"])
            self.assertEqual(manager.default_network_route, "webshare")
            account = pool.get(ids[0])
            self.assertFalse(account["proxy_override"])
            self.assertEqual(account["assigned_proxy_url"], "http://proxy:8000")
            manager.set_default_network_route("server")
            account = pool.get(ids[0])
            self.assertEqual(account["proxy"], "default")
            self.assertFalse(account["proxy_override"])
            env = manager.get_subprocess_env(ids[0])
            self.assertNotIn("HTTP_PROXY", env)
            self.assertNotIn("ALL_PROXY", env)

    def test_explicit_custom_override_survives_default_route_switch(self):
        with tempfile.TemporaryDirectory() as directory:
            pool, ids = make_pool(Path(directory), 1)
            manager = km.KaggleKernelManager()
            manager.account_pool = pool
            manager.webshare = MagicMock()
            manager.webshare.test_proxy.return_value = "10.1.1.1"
            manager.update_account_proxy(ids[0], "http://proxy:9000")
            manager.set_default_network_route("server")
            account = pool.get(ids[0])
            self.assertTrue(account["proxy_override"])
            self.assertEqual(account["proxy"], "http://proxy:9000")

    def test_registry_accelerator_contract_and_tpu_machine_shape(self):
        for model in registry.MODELS.values():
            self.assertTrue(model.get("supported_accelerators"))
            self.assertIn(model.get("default_accelerator"), model["supported_accelerators"])
        manager = km.KaggleKernelManager()
        manager.accelerator_mode = "gpu"
        self.assertFalse(manager.model_is_compatible("glm-5.3-flash"))
        self.assertTrue(manager.model_is_compatible("qwen3.8-27b-instruct"))
        manager.accelerator_mode = "tpu"
        self.assertTrue(manager.model_is_compatible("qwen3.8-27b-instruct"))
        with tempfile.TemporaryDirectory() as directory:
            old_runtime = km.RUNTIME_BUNDLE_DIR
            km.RUNTIME_BUNDLE_DIR = Path(directory)
            manager.client_api_key = manager.upstream_api_key = "test-key"
            manager.resolve_kernel_identity = lambda **kwargs: ("friend", "qwen38-tpu-serve", "Qwen TPU")
            session = km.KernelSession("s", "test", "account", "qwen3.8-27b-instruct", "tpu")
            bundle = manager._prepare_session_bundle(session)
            metadata = json.loads((bundle / "kernel-metadata.json").read_text())
            self.assertTrue(metadata["enable_tpu"])
            self.assertFalse(metadata["enable_gpu"])
            self.assertEqual(metadata["machine_shape"], "TpuV5E8")
            km.RUNTIME_BUNDLE_DIR = old_runtime

    def test_huggingface_registry_files_and_dynamic_resolver(self):
        client = httpx.Client(timeout=30, trust_env=False)
        try:
            for model in registry.MODELS.values():
                if "gpu" not in model.get("supported_accelerators", []) or model.get("custom"):
                    continue
                response = client.get(f"https://huggingface.co/api/models/{model['gpu_repo_id']}")
                self.assertEqual(response.status_code, 200, model["gpu_repo_id"])
                siblings = [item.get("rfilename", "") for item in response.json().get("siblings", [])]
                resolved = registry.resolve_gguf_filename(model["gpu_filename"], siblings)
                self.assertIn(resolved, siblings)
                self.assertTrue(resolved.lower().endswith(".gguf"))
        finally:
            client.close()

    def test_every_gpu_registry_model_generates_a_compilable_kernel(self):
        manager = km.KaggleKernelManager()
        manager.client_api_key = manager.upstream_api_key = "test-key"
        for model_id, model in registry.MODELS.items():
            if "gpu" not in model.get("supported_accelerators", []) or model.get("custom"):
                continue
            source = manager.generate_kernel_script(model_id, f"test-{model_id}", "test", "gpu")
            compile(source, f"<{model_id}-kernel>", "exec")
            self.assertIn(model["gpu_repo_id"], source)
            self.assertIn("resolve_gguf_filename", source)
            self.assertIn('llama_build = "b12395"', source)
            self.assertIn('"--host", "127.0.0.1", "--port", "8080"', source)
            self.assertIn('"--protocol", "http2", "--edge-ip-version", "4"', source)


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

            def model_spec(self, model_id):
                return {"max_concurrency": 1, "context_length": 65536}

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

    async def test_model_normalization_context_clamp_and_transient_retry(self):
        class Manager:
            active_model_id = "qwen3.8-35b-a3b-distill"
            tunnel_url = "https://old.test"
            upstream_api_key = "key"
            status = "ready"
            status_detail = ""

            def __init__(self):
                self.queued_requests = 0
                self.active_requests = 0

            def model_spec(self, model_id):
                return {"max_concurrency": 1, "context_length": 1024}

            def note_request_queued(self): self.queued_requests += 1
            def note_request_aborted(self): self.queued_requests = max(0, self.queued_requests - 1)
            def reserve_upstream(self):
                self.queued_requests -= 1; self.active_requests += 1
                return self.tunnel_url
            def release_upstream(self, url): self.active_requests -= 1
            def warmup_headers(self): return {}

        manager = Manager()
        attempts = []

        async def handler(request):
            attempts.append(json.loads(request.content.decode()))
            if len(attempts) < 3:
                return httpx.Response(503, json={"error": "restarting"})
            return httpx.Response(200, json={"ok": True})

        proxy = InferenceProxy(manager, MODELS, lambda: "key", lambda: False, lambda body: {}, lambda: False)
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with patch("proxy_service.asyncio.sleep", new_callable=AsyncMock):
            response = await proxy.raw_proxy("/v1/chat/completions", {
                "model": "wrong-client-alias", "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 999999,
            }, False)
        await proxy._client.aclose()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(attempts), 3)
        self.assertTrue(all(item["model"] == manager.active_model_id for item in attempts))
        self.assertTrue(all(1 <= item["max_tokens"] < 1024 for item in attempts))
        self.assertEqual(manager.active_requests, 0)


if __name__ == "__main__":
    unittest.main()
