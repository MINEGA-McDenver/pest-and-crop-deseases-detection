"""
Hard release gate checker for build/deploy pipelines.

Fails with non-zero exit code if either:
  - models/test_evaluation.json -> release_gate.release_verdict == "blocked"
  - models/release_readiness.json -> verdict == "BLOCKED"

Use this script as a required pre-build step before generating release artifacts.
"""

import argparse
import json
import os
import sys


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE, "models")
TEST_EVAL_PATH = os.path.join(MODELS_DIR, "test_evaluation.json")
RELEASE_READINESS_PATH = os.path.join(MODELS_DIR, "release_readiness.json")


def _load_json(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _collect_blockers(test_eval, release_readiness):
    blockers = []

    test_release_verdict = (
        ((test_eval or {}).get("release_gate") or {}).get("release_verdict", "")
    )
    if isinstance(test_release_verdict, str) and test_release_verdict.strip().lower() == "blocked":
        test_blockers = ((test_eval.get("release_gate") or {}).get("release_blockers") or [])
        if not isinstance(test_blockers, list):
            test_blockers = [str(test_blockers)]
        if test_blockers:
            for b in test_blockers:
                blockers.append(f"test_evaluation: {b}")
        else:
            blockers.append("test_evaluation release_gate.release_verdict=blocked")

    readiness_verdict = str((release_readiness or {}).get("verdict", "")).strip().upper()
    if readiness_verdict == "BLOCKED":
        readiness_blockers = (release_readiness.get("blockers") or [])
        if not isinstance(readiness_blockers, list):
            readiness_blockers = [str(readiness_blockers)]
        if readiness_blockers:
            for b in readiness_blockers:
                blockers.append(f"release_readiness: {b}")
        else:
            blockers.append("release_readiness verdict=BLOCKED")

    return blockers


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-blocked",
        action="store_true",
        help="Report blockers but return exit code 0.",
    )
    args = parser.parse_args(argv)

    try:
        test_eval = _load_json(TEST_EVAL_PATH)
    except FileNotFoundError as exc:
        print(f"[release_gate] ERROR: {exc}")
        return 3

    release_readiness = {}
    if os.path.isfile(RELEASE_READINESS_PATH):
        release_readiness = _load_json(RELEASE_READINESS_PATH)

    blockers = _collect_blockers(test_eval, release_readiness)

    if blockers:
        print("[release_gate] BLOCKED")
        for b in blockers:
            print(f"  - {b}")
        if args.allow_blocked:
            print("[release_gate] --allow-blocked set; continuing with exit code 0.")
            return 0
        return 2

    print("[release_gate] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
