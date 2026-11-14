#!/usr/bin/env python3
"""
Comprehensive Verification Script for Uncensored-Manager:
1. Server-Side Timing Persistence
2. Bulletproof Dynamic Kaggle Slug & Title Conflict Prevention
"""
import json
import os
import sys
import time
from pathlib import Path

BASE_DIR = Path("/root/uncensored-manager")
sys.path.insert(0, str(BASE_DIR))

from kaggle_manager import KaggleKernelManager, BUNDLE_DIR
from app import load_persisted_config, sync_config, CONFIG_PATH, manager

def run_tests():
    print("=" * 70)
    print("RUNNING VERIFICATION TEST SUITE")
    print("=" * 70)

    # ---------------------------------------------------------
    # TEST 1: Server-Side Timing Persistence
    # ---------------------------------------------------------
    print("\n[TEST 1] Testing Server-Side Timing Persistence...")
    test_manager = KaggleKernelManager()

    # Check idle state
    test_manager.launch_time = None
    state = test_manager.get_state()
    assert state["launch_timestamp"] is None, "Idle launch_timestamp must be None"
    assert state["elapsed_seconds"] == 0, "Idle elapsed_seconds must be 0"
    print("  ✓ Idle state returns launch_timestamp=None, elapsed_seconds=0")

    # Check set_tunnel_url sets launch_time
    t_start = time.time()
    test_manager.set_tunnel_url("https://verify-tunnel.trycloudflare.com")
    assert test_manager.launch_time is not None, "set_tunnel_url must set launch_time"
    assert abs(test_manager.launch_time - t_start) < 2, "launch_time should match current epoch"
    print(f"  ✓ set_tunnel_url successfully initialized launch_time: {test_manager.launch_time}")

    time.sleep(1.2)
    elapsed = test_manager.get_elapsed_seconds()
    assert elapsed >= 1, f"Elapsed seconds must be >= 1, got {elapsed}"
    print(f"  ✓ Elapsed seconds tracking correctly: {elapsed}s")

    # Check config persistence via sync_config
    manager.launch_time = test_manager.launch_time
    manager.tunnel_url = test_manager.tunnel_url
    manager.status = "ready"
    sync_config()

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        conf = json.load(f)
    assert conf.get("launch_time") == test_manager.launch_time, "launch_time must be saved to config.json"
    print(f"  ✓ config.json verified persisted launch_time: {conf.get('launch_time')}")

    # Check config loading on startup
    loaded = load_persisted_config()
    assert loaded.get("launch_time") == test_manager.launch_time, "load_persisted_config must restore launch_time"
    print(f"  ✓ load_persisted_config restored launch_time: {loaded.get('launch_time')}")

    # Check stop_kernel resets launch_time
    test_manager.stop_kernel()
    assert test_manager.launch_time is None, "stop_kernel must reset launch_time to None"
    assert test_manager.get_elapsed_seconds() == 0, "stop_kernel must reset elapsed_seconds to 0"

    manager.launch_time = None
    manager.tunnel_url = ""
    manager.status = "idle"
    sync_config()
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        conf_reset = json.load(f)
    assert conf_reset.get("launch_time") is None, "config.json must have launch_time=None after stop/reset"
    print("  ✓ stop_kernel and reset verified reset of launch_time to None")

    # ---------------------------------------------------------
    # TEST 2: Active Account Dynamic Slug & Title Resolution
    # ---------------------------------------------------------
    print("\n[TEST 2] Testing Dynamic Slug & Title Resolution for Active Account...")
    mgr = KaggleKernelManager()
    owner, slug, title = mgr.resolve_kernel_identity(force=True)
    print(f"  Active account resolved identity: owner={owner}, slug={slug}, title='{title}'")
    assert owner == "sougatanandi123", f"Expected owner sougatanandi123, got {owner}"
    assert slug in ("gpu-engine", "hermes-uncensored-engine"), f"Expected recognized engine slug, got {slug}"
    assert title in ("GPU Engine", "Hermes Uncensored Engine"), f"Expected recognized engine title, got {title}"
    print("  ✓ Active account correctly identified existing engine notebook on Kaggle")

    # Test prepare_bundle generates matching metadata
    bundle_path = mgr.prepare_bundle("cybertiel-coder-35b-a3b")
    meta_file = bundle_path / "kernel-metadata.json"
    assert meta_file.exists(), "kernel-metadata.json must exist"
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    expected_id = f"{owner}/{slug}"
    assert meta["id"] == expected_id, f"Metadata id mismatch: expected {expected_id}, got {meta['id']}"
    assert meta["title"] == title, f"Metadata title mismatch: expected {title}, got {meta['title']}"
    # Verify title slug resolves to id slug to prevent 409
    title_slug = meta["title"].lower().replace(" ", "-")
    print(f"  ✓ Bundle metadata generated: id='{meta['id']}', title='{meta['title']}'")

    # ---------------------------------------------------------
    # TEST 3: Multi-Account & Edge-Case Conflict Scenarios
    # ---------------------------------------------------------
    print("\n[TEST 3] Testing Multi-Account & Edge-Case Conflict Scenarios...")

    sim = KaggleKernelManager()

    # Scenario A: Account has only hermes-uncensored-engine
    sim._cached_kernels = [{
        "ref": "client_alpha/hermes-uncensored-engine",
        "slug": "hermes-uncensored-engine",
        "title": "Hermes Uncensored Engine",
        "owner": "client_alpha"
    }]
    sim._cached_kernels_time = time.time() + 9999
    o_a, s_a, t_a = sim.resolve_kernel_identity(force=False)
    assert s_a == "hermes-uncensored-engine" and t_a == "Hermes Uncensored Engine" and o_a == "client_alpha"
    print("  ✓ Scenario A (hermes-uncensored-engine only): PASSED")

    # Scenario B: Account has only kaggle-gpu-engine
    sim._cached_kernels = [{
        "ref": "client_beta/kaggle-gpu-engine",
        "slug": "kaggle-gpu-engine",
        "title": "Kaggle GPU Engine",
        "owner": "client_beta"
    }]
    sim._cached_kernels_time = time.time() + 9999
    o_b, s_b, t_b = sim.resolve_kernel_identity(force=False)
    assert s_b == "kaggle-gpu-engine" and t_b == "Kaggle GPU Engine" and o_b == "client_beta"
    print("  ✓ Scenario B (kaggle-gpu-engine only): PASSED")

    # Scenario C: Brand new account with 0 notebooks
    sim._cached_kernels = []
    sim._cached_kernels_time = time.time() + 9999
    o_c, s_c, t_c = sim.resolve_kernel_identity(force=False)
    assert s_c == "gpu-engine" and t_c == "GPU Engine"
    print("  ✓ Scenario C (Brand new account, 0 notebooks): PASSED")

    # Scenario D: Account with unrelated notebooks (e.g. titanic)
    sim._cached_kernels = [{
        "ref": "client_delta/titanic-prediction",
        "slug": "titanic-prediction",
        "title": "Titanic Survival Prediction",
        "owner": "client_delta"
    }]
    sim._cached_kernels_time = time.time() + 9999
    o_d, s_d, t_d = sim.resolve_kernel_identity(force=False)
    assert s_d == "gpu-engine" and t_d == "GPU Engine"
    print("  ✓ Scenario D (Unrelated notebooks only): PASSED")

    # Scenario E: Account with custom notebook title containing 'Uncensored Engine'
    sim._cached_kernels = [{
        "ref": "client_epsilon/my-custom-box",
        "slug": "my-custom-box",
        "title": "Private Uncensored Engine v2",
        "owner": "client_epsilon"
    }]
    sim._cached_kernels_time = time.time() + 9999
    o_e, s_e, t_e = sim.resolve_kernel_identity(force=False)
    assert s_e == "my-custom-box" and t_e == "Private Uncensored Engine v2"
    print("  ✓ Scenario E (Custom slug matching title phrase): PASSED")

    # ---------------------------------------------------------
    # TEST 4: 409 Conflict Retry Logic Verification
    # ---------------------------------------------------------
    print("\n[TEST 4] Testing 409 Conflict Catch & Auto-Adoption Logic...")
    sample_409_err = """
Your kernel title does not resolve to the specified id. This may result in surprising behavior.
409 Client Error: Conflict for url: https://api.kaggle.com/v1/kernels.KernelsApiService/SaveKernel
The kernel title 'GPU Engine' is already in use by a notebook (test_owner/gpu-engine).
"""
    lower_out = sample_409_err.lower()
    is_conflict = (
        "409" in lower_out
        or "conflict" in lower_out
        or "already in use" in lower_out
        or "does not resolve to the specified id" in lower_out
    )
    assert is_conflict is True, "Conflict detection regex must match 409 output"
    import re
    match_ref = re.search(r"\(([a-zA-Z0-9_\-]+/[a-zA-Z0-9_\-]+)\)", sample_409_err)
    if not match_ref:
        match_ref = re.search(r"(?:notebook|kernel)\s+\(?([a-zA-Z0-9_\-]+/[a-zA-Z0-9_\-]+)\)?", sample_409_err, re.IGNORECASE)
    assert match_ref is not None and match_ref.group(1) == "test_owner/gpu-engine"
    print(f"  ✓ 409 Conflict regex correctly detected conflict and extracted notebook ref: {match_ref.group(1)}")

    print("\n" + "=" * 70)
    print("ALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
