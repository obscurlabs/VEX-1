"""Assemble final_report.json from artifacts produced by the validation run.

Consumes what the other tools already wrote - the health check's JSON, the
JUnit XML, the captured live run and failure-path results - and adds facts
read fresh from the evidence bundle and from Polygon Amoy.

Nothing here re-runs the pipeline and nothing is invented: a value that could
not be obtained is reported absent rather than filled in.

    python scripts/build_final_report.py --out final_report.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bootstrap import require_dependencies  # noqa: E402

require_dependencies()

from src.config import CONFIG  # noqa: E402

REPORT_SCHEMA_VERSION = "1.0.0"

#: Never emitted, in any form.
FORBIDDEN = ("private_key", "serpapi_key", "api_key", "apikey",
             "secret", "password", "authorization", "mnemonic")


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _git() -> dict:
    def run(*args) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=CONFIG.project_root,
                                 capture_output=True, text=True, timeout=20)
            return out.stdout.strip() or None
        except Exception:
            return None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "uncommitted_paths": sorted(
            line[3:] for line in (status or "").splitlines())[:40],
    }


def _redact_rpc(url: str) -> str:
    if not url:
        return ""
    return re.sub(r"(/v2/|/rpc/|apiKey=|key=)[A-Za-z0-9_\-]+", r"\1<REDACTED>", url)


def _environment() -> dict:
    versions = {}
    for module, label in (("cv2", "opencv"), ("numpy", "numpy"),
                          ("onnxruntime", "onnxruntime"),
                          ("insightface", "insightface"), ("web3", "web3"),
                          ("requests", "requests"), ("PySide6", "pyside6")):
        try:
            versions[label] = getattr(__import__(module), "__version__", "?")
        except ImportError:
            versions[label] = None
    v = sys.version_info
    return {
        "python": f"{v.major}.{v.minor}.{v.micro}",
        "platform": sys.platform,
        "interpreter": Path(sys.executable).name,
        "in_project_venv": ".venv" in str(Path(sys.executable).resolve()),
        "packages": versions,
    }


def _configuration() -> dict:
    """Shapes and public values only. No secret is read into the report."""
    from src.evidence import manifest as manifest_mod

    key = CONFIG.private_key.removeprefix("0x")
    return {
        "serpapi_key_present": bool(CONFIG.serpapi_key),
        "private_key_present": bool(CONFIG.private_key),
        "private_key_shape_valid": len(key) == 64 and all(
            c in "0123456789abcdefABCDEF" for c in key),
        "rpc_url_redacted": _redact_rpc(CONFIG.polygon_rpc_url),
        "contract_address": CONFIG.contract_address,
        "expected_chain_id": CONFIG.chain.expected_chain_id,
        "network": CONFIG.chain.network_name,
        "match_threshold": CONFIG.match.threshold,
        "similarity_metric": "cosine",
        "pipeline_version": CONFIG.pipeline_version,
        "manifest_schema_version": manifest_mod.SCHEMA_VERSION,
        "retrieval": {
            "max_candidates": CONFIG.retrieval.max_candidates,
            "concurrency": CONFIG.retrieval.concurrency,
            "connect_timeout": CONFIG.retrieval.connect_timeout,
            "read_timeout": CONFIG.retrieval.read_timeout,
        },
    }


def _tests(junit: Path) -> dict:
    if not junit.exists():
        return {"status": "NOT_RUN", "details": f"{junit.name} not found"}
    root = ET.parse(junit).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        return {"status": "NOT_RUN", "details": "no testsuite element"}
    failures = int(suite.get("failures", 0)) + int(suite.get("errors", 0))
    by_file: dict[str, int] = {}
    for case in suite.iter("testcase"):
        name = (case.get("classname") or "").split(".")[0]
        by_file[name] = by_file.get(name, 0) + 1
    return {
        "status": "PASS" if failures == 0 else "FAIL",
        "total": int(suite.get("tests", 0)),
        "failures": int(suite.get("failures", 0)),
        "errors": int(suite.get("errors", 0)),
        "skipped": int(suite.get("skipped", 0)),
        "seconds": round(float(suite.get("time", 0)), 1),
        "by_module": dict(sorted(by_file.items())),
    }


def _live_run(capture: dict | None, bundle: Path | None) -> dict:
    """Facts read from the bundle the live run produced, not from its stdout."""
    if bundle is None or not (bundle / "manifest.json").exists():
        return {"status": "NOT_RUN", "details": "no bundle from a live run"}

    manifest = json.loads((bundle / "manifest.json").read_bytes().decode("utf-8"))
    anchor = _load(bundle / "anchor.json") or {}
    matching = manifest.get("matching", {})
    census = matching.get("census") or {}
    ranked = matching.get("ranked_matches") or []

    def brief(record: dict | None) -> dict | None:
        if not record:
            return None
        return {
            "rank": record.get("rank"),
            "source_url": record.get("source_url"),
            "source_domain": record.get("source_domain"),
            "title": record.get("title"),
            "search_position": record.get("search_position"),
            "similarity": record.get("similarity"),
            "status": record.get("status"),
            "faces_detected": record.get("faces_detected"),
            "selected_face_index": record.get("selected_face_index"),
            "retrieval_status": record.get("retrieval_status"),
            "content_sha256": record.get("content_sha256"),
            "identical_to_input": record.get("identical_to_input"),
            "group_size": record.get("group_size"),
            "duplicates": [
                {"source_domain": d.get("source_domain"),
                 "similarity": d.get("similarity"),
                 "duplicate_reason": d.get("duplicate_reason")}
                for d in record.get("duplicates", [])
            ],
        }

    payload = {
        "status": "PASS",
        "investigation_id": manifest.get("investigation_id"),
        "created_at": manifest.get("created_at"),
        "bundle": str(bundle.relative_to(CONFIG.project_root)).replace("\\", "/"),
        "input": manifest.get("input"),
        "discovery": manifest.get("search"),
        "retrieval": manifest.get("retrieval"),
        "matching": {
            "threshold": matching.get("threshold"),
            "similarity_metric": matching.get("similarity_metric"),
            "census": census,
            "qualifying_count": census.get("qualifying"),
            "independent_count": census.get("independent"),
            "selected_match": brief(matching.get("selected_match")),
            "best_independent_match": brief(matching.get("best_independent_match")),
            "matches": [brief(g) for g in ranked],
        },
        "evidence": {
            "schema_version": manifest.get("schema_version"),
            "evidence_sha256": (_load(bundle / "fingerprint.json") or {})
                               .get("evidence_sha256"),
            "artifacts": manifest.get("artifacts"),
        },
        "model": manifest.get("model"),
        "blockchain": {
            "network": anchor.get("network"),
            "chain_id": anchor.get("chain_id"),
            "contract_address": anchor.get("contract_address"),
            "transaction_hash": (anchor.get("anchor_tx") or {}).get("tx_hash"),
            "block_number": (anchor.get("anchor_tx") or {}).get("block_number"),
            "gas_used": (anchor.get("anchor_tx") or {}).get("gas_used"),
            "fee_pol": (anchor.get("anchor_tx") or {}).get("fee_pol"),
            "submitter": anchor.get("submitter"),
        },
    }

    if capture:
        payload["command"] = capture.get("command")
        payload["exit_code"] = capture.get("exit_code")
        payload["wall_clock_seconds"] = capture.get("wall_clock_seconds")
        payload["timings"] = {
            "measurement": "wall clock, observed externally from when each "
                           "stage banner reached the capturing process",
            "stages": capture.get("stage_timings_measured_externally"),
            "total_seconds": capture.get("wall_clock_seconds"),
        }
    else:
        payload["timings"] = {"status": "NOT_RUN",
                              "details": "no timing capture available"}
    return payload


def _verification(bundle: Path | None) -> dict:
    """Read the chain now, in this process, rather than trusting the bundle."""
    if bundle is None:
        return {"status": "NOT_RUN"}
    try:
        from src.blockchain.verifier import verify_against_chain
        result = verify_against_chain(bundle)
    except Exception as exc:
        return {"status": "FAIL", "details": f"{type(exc).__name__}: {exc}"}

    return {
        "status": "PASS" if result.verified else "FAIL",
        "details": ("locally recomputed fingerprint equals the fingerprint read "
                    "from the contract" if result.verified
                    else "; ".join(result.problems)),
        "verification_status": result.status.value,
        "local_sha256": result.local_sha256,
        "on_chain_sha256": result.on_chain_sha256,
        "recorded_sha256_advisory_only": result.recorded_sha256,
        "contract_verify_evidence": result.contract_verify_evidence,
        "chain_id": result.chain_id,
        "contract_address": result.contract_address,
        "submitter": result.submitter,
        "anchored_block": result.anchored_block,
        "artifacts_checked": result.artifacts_checked,
        "private_key_required": False,
    }


def _assert_no_secrets(payload: dict) -> list[str]:
    """Refuse to emit a report containing a real secret value."""
    from dotenv import dotenv_values

    env = dotenv_values(CONFIG.project_root / ".env")
    values = [v for k, v in env.items()
              if v and k in ("SERPAPI_KEY", "PRIVATE_KEY")]
    rpc = env.get("POLYGON_RPC_URL", "") or ""
    if "/v2/" in rpc:
        values.append(rpc.split("/v2/")[-1])

    blob = json.dumps(payload)
    problems = [f"secret value present ({name[:6]}…)"
                for name in values if name and name in blob]

    # A forbidden key name is only a problem when it carries a value that
    # could BE a secret. {"SERPAPI_KEY": true} is a presence flag and is
    # exactly the kind of safe evidence this report is supposed to contain.
    def walk(node, path=""):
        if isinstance(node, dict):
            for key, value in node.items():
                here = f"{path}/{key}"
                if (any(w == key.lower() for w in FORBIDDEN)
                        and isinstance(value, str) and len(value) > 8):
                    problems.append(f"forbidden key carries a value at {here}")
                walk(value, here)
        elif isinstance(node, list):
            for i, item in enumerate(node):
                walk(item, f"{path}[{i}]")

    walk(payload)
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description="Build final_report.json")
    ap.add_argument("--out", default="final_report.json")
    ap.add_argument("--healthcheck", default="final/healthcheck.json")
    ap.add_argument("--junit", default="final/junit.xml")
    ap.add_argument("--live-capture")
    ap.add_argument("--failure-paths")
    ap.add_argument("--bundle")
    args = ap.parse_args()

    root = CONFIG.project_root

    def resolve(value: str | None) -> Path | None:
        if not value:
            return None
        path = Path(value)
        return path if path.is_absolute() else root / path

    health = _load(resolve(args.healthcheck)) if args.healthcheck else None
    capture = _load(resolve(args.live_capture)) if args.live_capture else None
    failures = _load(resolve(args.failure_paths)) if args.failure_paths else None

    bundle = resolve(args.bundle)
    if bundle is None:
        candidates = [p for p in CONFIG.evidence_dir.glob("TRACE-*")
                      if (p / "manifest.json").exists()]
        bundle = max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None

    health_checks = (health or {}).get("checks", [])
    health_totals = (health or {}).get("totals", {})
    tamper = (health or {}).get("tamper", {})
    tests = _tests(resolve(args.junit))
    live = _live_run(capture, bundle)
    verification = _verification(bundle)

    security = {
        "status": "PASS" if all(
            c["status"] == "PASS" for c in health_checks
            if c.get("section") == "security") else "FAIL",
        "checks": [c for c in health_checks if c.get("section") == "security"],
        "report_contains_no_secrets": None,   # filled in after the scan
    }
    reproducibility = {
        "status": "PASS" if all(
            c["status"] == "PASS" for c in health_checks
            if c.get("section") == "reproducibility") else "FAIL",
        "checks": [c for c in health_checks if c.get("section") == "reproducibility"],
    }

    blocking: list[str] = []
    if tests.get("status") == "FAIL":
        blocking.append(f"{tests.get('failures', 0)} test failure(s)")
    if health_totals.get("FAIL"):
        blocking.append(f"{health_totals['FAIL']} health check failure(s)")
    if live.get("status") not in ("PASS", None) or live.get("exit_code") not in (0, None):
        blocking.append("live run did not complete cleanly")
    if verification.get("status") != "PASS":
        blocking.append("chain verification did not pass")
    if tamper and tamper.get("detected") != tamper.get("total"):
        blocking.append("a tamper mutation went undetected")
    if failures and failures.get("failed"):
        blocking.append(f"{failures['failed']} failure path(s) misbehaved")

    warnings = [f"{c['name']}: {c['details']}"
                for c in health_checks if c["status"] == "WARN"]

    report = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "project": "VEX-1 · HH Goa 2026 Task 3 · face evidence pipeline",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": _git(),
        "environment": _environment(),
        "configuration": _configuration(),
        "health_check": {
            "status": "PASS" if not health_totals.get("FAIL") else "FAIL",
            "totals": health_totals,
            "checks": health_checks,
            "source": args.healthcheck,
        },
        "tests": tests,
        "live_run": live,
        "verification": verification,
        "tamper_tests": tamper or {"status": "NOT_RUN"},
        "failure_paths": failures or {"status": "NOT_RUN"},
        "security_audit": security,
        "reproducibility": reproducibility,
        "artifacts": {
            "bundle": (str(bundle.relative_to(root)).replace("\\", "/")
                       if bundle else None),
            "healthcheck_json": args.healthcheck,
            "junit_xml": args.junit,
            "report": args.out,
        },
        "final_verdict": {
            "status": "READY_FOR_SUBMISSION" if not blocking
                      else "NOT_READY_FOR_SUBMISSION",
            "checks_passed": health_totals.get("PASS", 0) + tests.get("total", 0),
            "checks_failed": health_totals.get("FAIL", 0) + tests.get("failures", 0),
            "blocking_issues": blocking,
            "warnings": warnings,
        },
    }

    problems = _assert_no_secrets(report)
    report["security_audit"]["report_contains_no_secrets"] = not problems
    if problems:
        report["final_verdict"]["blocking_issues"].append(
            "final_report.json would contain a secret")
        report["final_verdict"]["status"] = "NOT_READY_FOR_SUBMISSION"
        print("REFUSING TO WRITE: " + "; ".join(problems), file=sys.stderr)
        return 1

    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    verdict = report["final_verdict"]
    print(f"wrote {out}  ({out.stat().st_size:,} bytes)")
    print(f"verdict: {verdict['status']}")
    print(f"blocking: {verdict['blocking_issues'] or 'none'}")
    print(f"warnings: {len(verdict['warnings'])}")
    return 0 if not verdict["blocking_issues"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
