"""Check everything the live demo needs, before spending credits or gas.

    python preflight.py
    python preflight.py --image inputs/demo-target.jpg
    python preflight.py --offline      # skip network checks

Read-only. Sends no SerpAPI search and no transaction. The private key is
never printed - only the address derived from it.

Exit codes: 0 ready, 1 something failed, 2 ready with warnings.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Fail with a readable message, not an import traceback, when the entry
# point is run with an interpreter that lacks the dependencies.
from src.bootstrap import require_dependencies

require_dependencies()

from src.config import CONFIG
from src.pipeline import WIDTH

OK, WARN, FAIL = "OK", "WARN", "FAIL"


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, level: str, name: str, detail: str = "") -> None:
        self.rows.append((level, name, detail))
        print(f"[{level:<4}] {name:<22} {detail}")

    @property
    def failed(self) -> int:
        return sum(1 for lvl, _, _ in self.rows if lvl == FAIL)

    @property
    def warned(self) -> int:
        return sum(1 for lvl, _, _ in self.rows if lvl == WARN)


def check_python(r: Report) -> None:
    v = sys.version_info
    version = f"{v.major}.{v.minor}.{v.micro}"
    if v >= (3, 11):
        r.add(OK, "Python", version)
    else:
        r.add(FAIL, "Python", f"{version} - 3.11+ required")


def check_packages(r: Report) -> None:
    missing = []
    versions = []
    for module, label in (("cv2", "opencv"), ("numpy", "numpy"),
                          ("onnxruntime", "onnxruntime"), ("insightface", "insightface"),
                          ("web3", "web3"), ("requests", "requests")):
        try:
            mod = __import__(module)
            versions.append(f"{label} {getattr(mod, '__version__', '?')}")
        except ImportError:
            missing.append(label)
    if missing:
        r.add(FAIL, "Dependencies", f"missing: {', '.join(missing)}")
    else:
        r.add(OK, "Dependencies", ", ".join(versions[:3]) + ", ...")


def check_model(r: Report) -> None:
    try:
        from src.vision.detector import model_info
        mi = model_info()
        r.add(OK, "ArcFace model",
              f"{mi['pack']} · {mi['detector']} + {mi['recognizer_file']}")
    except Exception as exc:
        r.add(FAIL, "ArcFace model", f"{type(exc).__name__}: {exc}")


def check_input(r: Report, image: Path) -> None:
    from src.models import ImageStatus
    from src.vision.quality import load_image

    if not image.exists():
        r.add(FAIL, "Input image", f"not found: {image}")
        return
    status, img, err = load_image(image)
    if status is not ImageStatus.OK or img is None:
        r.add(FAIL, "Input image", f"{status.value}: {err}")
        return

    try:
        from src.vision.embedder import ArcFaceEmbedder
        result = ArcFaceEmbedder().process_image(img, all_faces=False)
    except Exception as exc:
        r.add(FAIL, "Input image", f"vision failed: {type(exc).__name__}: {exc}")
        return

    if not result.ok:
        r.add(FAIL, "Input image", f"{result.face_status.value}: {result.error}")
        return
    face = result.faces[0]
    r.add(OK, "Input image",
          f"{image.name} · {img.shape[1]}x{img.shape[0]} · "
          f"{result.faces_detected} face ({face.width}x{face.height}px)")


def check_config(r: Report) -> None:
    missing = [name for name, value in (
        ("SERPAPI_KEY", CONFIG.serpapi_key),
        ("POLYGON_RPC_URL", CONFIG.polygon_rpc_url),
        ("PRIVATE_KEY", CONFIG.private_key),
        ("CONTRACT_ADDRESS", CONFIG.contract_address),
    ) if not value]
    if missing:
        r.add(FAIL, "Configuration", f"unset: {', '.join(missing)}")
    else:
        r.add(OK, "Configuration",
              f"threshold {CONFIG.match.threshold} · "
              f"pipeline {CONFIG.pipeline_version}")


def check_env_ignored(r: Report) -> None:
    env = CONFIG.project_root / ".env"
    if not env.exists():
        r.add(WARN, ".env", "no .env file present")
        return
    try:
        proc = subprocess.run(
            ["git", "check-ignore", "-q", ".env"],
            cwd=CONFIG.project_root, capture_output=True, timeout=15,
        )
    except Exception as exc:
        r.add(WARN, ".env gitignored", f"could not check: {type(exc).__name__}")
        return
    if proc.returncode == 0:
        r.add(OK, ".env gitignored", "yes - secrets stay local")
    else:
        r.add(FAIL, ".env gitignored", "NOT IGNORED - fix .gitignore before committing")


def check_serpapi(r: Report) -> None:
    """Account lookup only. Does not consume a search credit."""
    if not CONFIG.serpapi_key:
        r.add(FAIL, "SerpAPI", "SERPAPI_KEY is not set")
        return
    try:
        import requests
        resp = requests.get("https://serpapi.com/account",
                            params={"api_key": CONFIG.serpapi_key}, timeout=20)
    except Exception as exc:
        r.add(FAIL, "SerpAPI", f"unreachable: {type(exc).__name__}")
        return
    if resp.status_code != 200:
        r.add(FAIL, "SerpAPI", f"HTTP {resp.status_code} - key rejected?")
        return
    data = resp.json()
    left = data.get("total_searches_left")
    plan = data.get("plan_name", "?")
    if isinstance(left, int) and left <= 0:
        r.add(FAIL, "SerpAPI", f"{plan} · no searches left")
    elif isinstance(left, int) and left < 5:
        r.add(WARN, "SerpAPI", f"{plan} · only {left} searches left")
    else:
        r.add(OK, "SerpAPI", f"{plan} · {left} searches left")


def check_chain(r: Report) -> None:
    try:
        from src.blockchain.client import (
            AnchorClient, RpcConnectionError, WalletConfigError, WrongChainError,
        )
    except Exception as exc:
        r.add(FAIL, "Blockchain", f"import failed: {type(exc).__name__}: {exc}")
        return

    try:
        client = AnchorClient()
    except WalletConfigError as exc:
        r.add(FAIL, "Wallet", str(exc))
        r.add(FAIL, "Polygon Amoy", "skipped - wallet unusable")
        return
    except Exception as exc:
        r.add(FAIL, "Blockchain", f"{type(exc).__name__}: {exc}")
        return

    try:
        chain_id = client.connect()
    except WrongChainError as exc:
        r.add(FAIL, "Chain ID", str(exc))
        return
    except RpcConnectionError as exc:
        r.add(FAIL, "Polygon RPC", str(exc))
        return
    except Exception as exc:
        r.add(FAIL, "Polygon RPC", f"{type(exc).__name__}: {exc}")
        return

    r.add(OK, "Polygon RPC", "reachable")
    r.add(OK, "Chain ID", f"{chain_id} ({CONFIG.chain.network_name})")

    # Only the derived public address is ever shown.
    r.add(OK, "Wallet", client.address)

    try:
        balance = client.balance_wei()
    except Exception as exc:
        r.add(FAIL, "Balance", f"{type(exc).__name__}: {exc}")
        return
    minimum = CONFIG.chain.min_balance_wei
    text = f"{balance / 1e18:.6f} POL"
    if balance >= minimum * 3:
        r.add(OK, "Balance", f"{text} · enough for several runs")
    elif balance >= minimum:
        r.add(WARN, "Balance", f"{text} · above the {minimum / 1e18:.3f} POL floor, "
                               "but running low")
    else:
        r.add(FAIL, "Balance", f"{text} · below the {minimum / 1e18:.3f} POL floor; "
                               "top up from an Amoy faucet")

    address = CONFIG.contract_address
    if not address:
        r.add(FAIL, "Contract", "CONTRACT_ADDRESS is not set")
        return
    try:
        code = client.w3.eth.get_code(client.w3.to_checksum_address(address))
    except Exception as exc:
        r.add(FAIL, "Contract", f"cannot read code: {type(exc).__name__}: {exc}")
        return
    if len(code) == 0:
        r.add(FAIL, "Contract", f"no contract code at {address}")
        return
    try:
        total = client.contract_at(address).functions.totalAnchored().call()
        r.add(OK, "Contract", f"{address} · {len(code)} bytes · {total} anchored")
    except Exception:
        r.add(FAIL, "Contract",
              f"code present at {address} but it is not an IdentityAnchor")


def main() -> int:
    ap = argparse.ArgumentParser(description="Pre-flight checks for the live demo")
    ap.add_argument("--image", default="inputs/demo-target.jpg")
    ap.add_argument("--offline", action="store_true", help="skip network checks")
    args = ap.parse_args()

    print("=" * WIDTH)
    print("HH GOA TASK 3 PREFLIGHT".center(WIDTH))
    print("=" * WIDTH)

    r = Report()
    check_python(r)
    check_packages(r)
    check_model(r)
    check_input(r, Path(args.image))
    check_config(r)
    check_env_ignored(r)

    if args.offline:
        r.add(WARN, "Network checks", "skipped (--offline)")
    else:
        check_serpapi(r)
        check_chain(r)

    print("=" * WIDTH)
    if r.failed:
        print(f"NOT READY - {r.failed} check(s) failed".center(WIDTH))
        print("=" * WIDTH)
        return 1
    if r.warned:
        print(f"READY (with {r.warned} warning(s))".center(WIDTH))
        print("=" * WIDTH)
        return 2
    print("READY FOR LIVE RUN".center(WIDTH))
    print("=" * WIDTH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
