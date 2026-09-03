"""Phase 6: final CLI behaviour, exit codes, and secret hygiene.

These run the real entry points as subprocesses, so what is asserted is what a
judge would actually see. Nothing here spends a SerpAPI credit or sends a
transaction: the live-mode tests are the ones that must FAIL fast, and the
success path runs in diagnostic mode with --no-chain.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.pipeline import (
    EXIT_CHAIN,
    EXIT_INPUT,
    EXIT_OK,
    EXIT_SEARCH,
    EXIT_SEARCH_AUTH,
)

ROOT = Path(__file__).resolve().parent.parent
# A capture that corresponds to the demo target, so the matching stage has
# something to match. The Obama capture is kept for content-agnostic cases.
FIXTURE = ROOT / "tests" / "fixtures" / "google-lens-response.json"
DEMO_FIXTURE = ROOT / "tests" / "fixtures" / "demo-target-response.json"
DEMO = ROOT / "inputs" / "demo-target.jpg"
HAVE_DEMO = DEMO_FIXTURE.exists() and DEMO.exists()


def skip_if_offline(proc) -> None:
    """Retrieval downloads real candidate images. A dead network is an
    environment problem, not a regression - say so rather than failing."""
    if ("0 usable images" in proc.stdout
            or "no candidate images were retrievable" in proc.stdout):
        pytest.skip("no candidate images retrievable (network unavailable?)")


def run_cli(*args: str, env: dict | None = None, timeout: int = 300):
    """Run main.py and capture what the terminal would show."""
    merged = {**os.environ, **(env or {})}
    return subprocess.run(
        [sys.executable, "main.py", *args],
        cwd=ROOT, capture_output=True, text=True, timeout=timeout, env=merged,
    )


def run_preflight(*args: str, env: dict | None = None, timeout: int = 300):
    merged = {**os.environ, **(env or {})}
    return subprocess.run(
        [sys.executable, "preflight.py", *args],
        cwd=ROOT, capture_output=True, text=True, timeout=timeout, env=merged,
    )


@pytest.fixture(scope="module")
def no_face_image(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("cli") / "no-face.jpg"
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    for y in range(480):
        img[y, :] = (200 - y // 4, 150 - y // 6, 90 + y // 8)
    cv2.imwrite(str(path), img)
    return path


# --- the successful path ----------------------------------------------

@pytest.mark.skipif(not HAVE_DEMO, reason="fixtures missing")
def test_pipeline_succeeds_through_the_evidence_fingerprint():
    proc = run_cli("--image", str(DEMO), "--mode", "diagnostic",
                   "--from-response", str(DEMO_FIXTURE), "--max-candidates", "8",
                   "--no-chain")
    skip_if_offline(proc)
    assert proc.returncode == EXIT_OK, proc.stdout + proc.stderr
    out = proc.stdout
    for marker in ("[01] FACE SCAN", "[02] WEB DISCOVERY", "[03] CANDIDATE RETRIEVAL",
                   "[04] FACE MATCHING", "[05] EVIDENCE"):
        assert marker in out, f"missing stage: {marker}"
    assert "MODE: DIAGNOSTIC" in out
    assert "evidence SHA-256" in out
    assert "EVIDENCE FINGERPRINT READY" in out


@pytest.mark.skipif(not HAVE_DEMO, reason="fixtures missing")
def test_default_output_is_aggregate_not_per_candidate():
    """The demo must not flood the terminal with one line per candidate."""
    proc = run_cli("--image", str(DEMO), "--mode", "diagnostic",
                   "--from-response", str(DEMO_FIXTURE), "--max-candidates", "8",
                   "--no-chain")
    assert "[Candidate 01]" not in proc.stdout
    assert "usable images" in proc.stdout


@pytest.mark.skipif(not HAVE_DEMO, reason="fixtures missing")
def test_verbose_adds_per_candidate_detail():
    proc = run_cli("--image", str(DEMO), "--mode", "diagnostic",
                   "--from-response", str(DEMO_FIXTURE), "--max-candidates", "8",
                   "--no-chain", "--verbose")
    assert "[Candidate 01]" in proc.stdout


# --- failure at each stage --------------------------------------------

def test_missing_image_exits_with_input_code():
    proc = run_cli("--image", "inputs/does-not-exist.jpg", "--mode", "diagnostic",
                   "--no-chain")
    assert proc.returncode == EXIT_INPUT
    assert "NOT_FOUND" in proc.stdout
    assert "Traceback" not in proc.stdout + proc.stderr


def test_image_without_a_face_exits_with_input_code(no_face_image):
    proc = run_cli("--image", str(no_face_image), "--mode", "diagnostic", "--no-chain")
    assert proc.returncode == EXIT_INPUT
    assert "NO_FACE" in proc.stdout
    assert "Traceback" not in proc.stdout + proc.stderr


@pytest.mark.skipif(not DEMO.exists(), reason="demo image missing")
def test_missing_diagnostic_response_fails_cleanly():
    proc = run_cli("--image", str(DEMO), "--mode", "diagnostic",
                   "--from-response", "nope/missing.json", "--no-chain")
    assert proc.returncode == EXIT_SEARCH
    assert "[DISCOVERY]" in proc.stdout
    assert "Traceback" not in proc.stdout + proc.stderr


@pytest.mark.skipif(not DEMO.exists(), reason="demo image missing")
def test_live_mode_with_a_bad_key_fails_and_never_falls_back():
    """The central guarantee: live never silently becomes diagnostic."""
    proc = run_cli("--image", str(DEMO), "--mode", "live",
                   env={"SERPAPI_KEY": "invalid_key_for_testing"})
    assert proc.returncode == EXIT_SEARCH_AUTH
    out = proc.stdout
    assert "MODE: LIVE" in out
    assert "[DISCOVERY]" in out and "authentication" in out
    # No cached path may appear anywhere in a live run.
    assert "replaying" not in out.lower()
    assert "cache" not in out.lower()
    assert "candidates discovered" not in out
    assert "BLOCKCHAIN VERIFIED" not in out
    assert "Traceback" not in out + proc.stderr


@pytest.mark.skipif(not HAVE_DEMO, reason="fixtures missing")
def test_chain_stage_fails_cleanly_when_rpc_is_dead():
    proc = run_cli("--image", str(DEMO), "--mode", "diagnostic",
                   "--from-response", str(DEMO_FIXTURE), "--max-candidates", "8",
                   env={"POLYGON_RPC_URL": "http://127.0.0.1:9/"})
    skip_if_offline(proc)
    assert proc.returncode == EXIT_CHAIN
    assert "[06] BLOCKCHAIN" in proc.stdout
    assert "[CHAIN]" in proc.stdout
    assert "Traceback" not in proc.stdout + proc.stderr


@pytest.mark.skipif(not HAVE_DEMO, reason="fixtures missing")
def test_wrong_chain_is_refused():
    proc = run_cli("--image", str(DEMO), "--mode", "diagnostic",
                   "--from-response", str(DEMO_FIXTURE), "--max-candidates", "8",
                   env={"POLYGON_RPC_URL": "https://ethereum-rpc.publicnode.com"})
    skip_if_offline(proc)
    assert proc.returncode == EXIT_CHAIN
    assert "wrong chain" in proc.stdout.lower()
    assert "expected 80002" in proc.stdout


@pytest.mark.skipif(not HAVE_DEMO, reason="fixtures missing")
def test_bad_private_key_fails_at_the_chain_stage():
    proc = run_cli("--image", str(DEMO), "--mode", "diagnostic",
                   "--from-response", str(DEMO_FIXTURE), "--max-candidates", "8",
                   env={"PRIVATE_KEY": "not-a-valid-key"})
    skip_if_offline(proc)
    assert proc.returncode == EXIT_CHAIN
    assert "wallet configuration error" in proc.stdout.lower()
    assert "not-a-valid-key" not in proc.stdout


# --- secret hygiene ----------------------------------------------------

@pytest.mark.skipif(not HAVE_DEMO, reason="fixtures missing")
def test_no_secrets_in_normal_output():
    from dotenv import dotenv_values

    env = dotenv_values(ROOT / ".env")
    proc = run_cli("--image", str(DEMO), "--mode", "diagnostic",
                   "--from-response", str(DEMO_FIXTURE), "--max-candidates", "8",
                   "--no-chain", "--verbose")
    blob = proc.stdout + proc.stderr
    for key in ("SERPAPI_KEY", "PRIVATE_KEY"):
        value = env.get(key)
        if value:
            assert value not in blob, f"{key} leaked into CLI output"
            assert value.removeprefix("0x") not in blob
    rpc = env.get("POLYGON_RPC_URL") or ""
    if "/v2/" in rpc:
        assert rpc.split("/v2/")[-1] not in blob, "Alchemy key leaked"


def test_no_secrets_in_preflight_output():
    from dotenv import dotenv_values

    env = dotenv_values(ROOT / ".env")
    proc = run_preflight("--offline")
    blob = proc.stdout + proc.stderr
    for key in ("SERPAPI_KEY", "PRIVATE_KEY"):
        value = env.get(key)
        if value:
            assert value not in blob
            assert value.removeprefix("0x") not in blob


# --- preflight ---------------------------------------------------------

def test_preflight_offline_runs_and_reports():
    proc = run_preflight("--offline")
    assert proc.returncode in (0, 1, 2)
    assert "HH GOA TASK 3 PREFLIGHT" in proc.stdout
    assert "[OK  ] Python" in proc.stdout
    assert "Network checks" in proc.stdout


def test_preflight_reports_a_missing_input_image():
    proc = run_preflight("--offline", "--image", "inputs/definitely-missing.jpg")
    assert proc.returncode == 1
    assert "[FAIL] Input image" in proc.stdout
    assert "NOT READY" in proc.stdout


def test_preflight_reports_missing_configuration():
    proc = run_preflight("--offline", env={"SERPAPI_KEY": "", "CONTRACT_ADDRESS": ""})
    assert proc.returncode == 1
    assert "[FAIL] Configuration" in proc.stdout
    assert "SERPAPI_KEY" in proc.stdout


def test_preflight_checks_env_is_gitignored():
    proc = run_preflight("--offline")
    assert ".env gitignored" in proc.stdout


# --- argument surface --------------------------------------------------

def test_help_lists_the_documented_flags():
    proc = subprocess.run([sys.executable, "main.py", "--help"],
                          cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0
    for flag in ("--image", "--mode", "--verbose", "--debug", "--no-chain"):
        assert flag in proc.stdout


def test_mode_must_be_live_or_diagnostic():
    proc = run_cli("--image", str(DEMO), "--mode", "cached")
    assert proc.returncode != 0
    assert "invalid choice" in proc.stderr


def test_image_argument_is_required():
    proc = run_cli("--mode", "diagnostic")
    assert proc.returncode != 0
    assert "required" in proc.stderr.lower()


# --- exit code contract ------------------------------------------------

def test_exit_codes_are_distinct():
    from src import pipeline

    codes = [getattr(pipeline, n) for n in dir(pipeline) if n.startswith("EXIT_")]
    assert len(codes) == len(set(codes)), "exit codes must be unambiguous"
    assert pipeline.EXIT_OK == 0


# --- artifacts ---------------------------------------------------------

@pytest.mark.skipif(not HAVE_DEMO, reason="fixtures missing")
def test_run_writes_a_verifiable_bundle():
    proc = run_cli("--image", str(DEMO), "--mode", "diagnostic",
                   "--from-response", str(DEMO_FIXTURE), "--max-candidates", "8",
                   "--no-chain")
    skip_if_offline(proc)
    assert proc.returncode == EXIT_OK
    # the summary row, not the "evidence bundle created" progress line
    line = next(l for l in proc.stdout.splitlines()
                if "evidence bundle" in l and "evidence/TRACE-" in l)
    bundle = ROOT / line.split()[-1]
    assert bundle.is_dir()

    for name in ("manifest.json", "fingerprint.json", "input.jpg", "matching.json"):
        assert (bundle / name).exists(), f"missing {name}"

    check = subprocess.run([sys.executable, "verify.py", str(bundle), "--quiet"],
                           cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert check.returncode == 0
    assert "VERIFIED" in check.stdout

    manifest = json.loads((bundle / "manifest.json").read_bytes().decode("utf-8"))
    blob = json.dumps(manifest).lower()
    for word in ("private_key", "api_key", "embedding_vector"):
        assert word not in blob


# --- interpreter guard -------------------------------------------------

def test_bootstrap_reports_missing_dependencies_readably(tmp_path):
    """Running an entry point with the wrong interpreter must explain itself
    instead of dumping a ModuleNotFoundError traceback."""
    from src import bootstrap

    assert bootstrap.missing_dependencies() == [], \
        "the test venv should have every dependency"

    # Simulate the wrong interpreter by hiding a required module.
    script = tmp_path / "guard_check.py"
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "import importlib.util\n"
        "real = importlib.util.find_spec\n"
        "importlib.util.find_spec = lambda n, *a, **k: None if n == 'web3' else real(n, *a, **k)\n"
        "from src.bootstrap import require_dependencies\n"
        "require_dependencies()\n",
        encoding="utf-8",
    )
    proc = subprocess.run([sys.executable, str(script)],
                          cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 1
    assert "WRONG PYTHON INTERPRETER" in proc.stderr
    assert "web3" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_bootstrap_is_stdlib_only():
    """The guard must import cleanly on an interpreter with no dependencies."""
    src = (ROOT / "src" / "bootstrap.py").read_text(encoding="utf-8")
    for banned in ("import cv2", "import numpy", "import web3", "from src.config"):
        assert banned not in src, f"bootstrap must not import {banned}"
