"""
Validate strict release readiness for field deployment.

Inputs:
  - models/test_evaluation.json
  - analysis_outputs/field_audit_summary.json

Output:
  - models/release_readiness.json

Policy (strict):
  1) Each crop supported_confident_rate >= 0.80 on field audit
  2) beans<->potato confusion rate <= 0.02
  3) supported<->unsupported confusion rates <= 0.02 in both directions
"""

import json
import os
import sys
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE, "models")
ANALYSIS_DIR = os.path.join(BASE, "analysis_outputs")

TEST_EVAL_PATH = os.path.join(MODELS_DIR, "test_evaluation.json")
FIELD_AUDIT_PATH = os.path.join(ANALYSIS_DIR, "field_audit_summary.json")
OUT_PATH = os.path.join(MODELS_DIR, "release_readiness.json")
SUPPORTED_UNSUPPORTED_PATH = os.path.join(MODELS_DIR, "supported_vs_unsupported_confusion.json")
CONFUSION_BY_CROP_PAIR_PATH = os.path.join(MODELS_DIR, "confusion_by_crop_pair.json")

REQUIRED_CROPS = ["banana", "beans", "maize", "potato"]
LEGACY_FIELD_AUDIT_GATE_PREFIX = "RESCUE_beans_potato_from_"


def _load_json(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_get(dct, keys, default=None):
    cur = dct
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _get_optional_float(dct, key):
    if key not in dct:
        return None
    try:
        return float(dct[key])
    except (TypeError, ValueError):
        return None


def _resolve_confusion_metrics(test_eval):
    """
    Resolve strict confusion metrics across compatible schemas.

    Primary source (new schema):
      test_evaluation.confusion_diagnostics.{rates}

    Legacy/fallback sources:
      - top-level keys in test_evaluation.json
      - models/supported_vs_unsupported_confusion.json (nested structure)
    """
    resolved = {
        "beans_potato_confusion_rate": None,
        "supported_to_unsupported_rate": None,
        "unsupported_to_supported_rate": None,
    }
    metric_sources = {
        "beans_potato_confusion_rate": [],
        "supported_to_unsupported_rate": [],
        "unsupported_to_supported_rate": [],
    }

    confusion_diag = _safe_get(test_eval, ["confusion_diagnostics"], {})
    for key in resolved:
        val = _get_optional_float(confusion_diag, key)
        if val is not None:
            resolved[key] = val
            metric_sources[key].append("test_evaluation.confusion_diagnostics")

    # Backward compatibility for older flattened test_evaluation schemas.
    for key in resolved:
        if resolved[key] is None:
            val = _get_optional_float(test_eval, key)
            if val is not None:
                resolved[key] = val
                metric_sources[key].append("test_evaluation.top_level")

    # Fallback to dedicated confusion summary JSON if still unresolved.
    need_fallback = any(resolved[k] is None for k in resolved)
    confusion_diag = _safe_get(test_eval, ["confusion_diagnostics"], {})
    supported_unsupported_candidates = [
        _safe_get(confusion_diag, ["supported_vs_unsupported_path"], None),
        SUPPORTED_UNSUPPORTED_PATH,
    ]
    supported_unsupported_candidates = [
        p for p in supported_unsupported_candidates if isinstance(p, str) and p.strip()
    ]

    for path in supported_unsupported_candidates:
        if not need_fallback:
            break
        if not os.path.isfile(path):
            continue
        fallback = _load_json(path)
        nested_map = {
            "beans_potato_confusion_rate": ["beans_potato", "rate"],
            "supported_to_unsupported_rate": ["supported_to_unsupported", "rate"],
            "unsupported_to_supported_rate": ["unsupported_to_supported", "rate"],
        }
        for key, key_path in nested_map.items():
            if resolved[key] is not None:
                continue
            raw = _safe_get(fallback, key_path, None)
            try:
                if raw is not None:
                    resolved[key] = float(raw)
                    metric_sources[key].append(
                        f"{path}::{'.'.join(key_path)}"
                    )
            except (TypeError, ValueError):
                pass

        need_fallback = any(resolved[k] is None for k in resolved)

    # Final fallback: derive strict rates from confusion-by-crop-pair counts.
    need_fallback = any(resolved[k] is None for k in resolved)
    crop_pair_candidates = [
        _safe_get(confusion_diag, ["crop_pair_path"], None),
        CONFUSION_BY_CROP_PAIR_PATH,
    ]
    crop_pair_candidates = [p for p in crop_pair_candidates if isinstance(p, str) and p.strip()]

    for path in crop_pair_candidates:
        if not need_fallback:
            break
        if not os.path.isfile(path):
            continue
        payload = _load_json(path)
        counts = _safe_get(payload, ["counts"], {})
        if not isinstance(counts, dict) or len(counts) == 0:
            continue

        # Build totals from true->pred crop counts.
        supported_crops = [c for c in counts.keys() if c != "other_leaf"]
        beans_row = counts.get("beans", {})
        potato_row = counts.get("potato", {})
        other_row = counts.get("other_leaf", {})

        beans_total = sum(v for v in beans_row.values() if isinstance(v, (int, float)))
        potato_total = sum(v for v in potato_row.values() if isinstance(v, (int, float)))
        beans_potato_total = beans_total + potato_total
        beans_potato_conf = float(beans_row.get("potato", 0)) + float(potato_row.get("beans", 0))

        supported_total = 0.0
        supported_to_unsupported = 0.0
        for crop in supported_crops:
            row = counts.get(crop, {})
            if not isinstance(row, dict):
                continue
            row_total = sum(v for v in row.values() if isinstance(v, (int, float)))
            supported_total += float(row_total)
            supported_to_unsupported += float(row.get("other_leaf", 0))

        unsupported_total = sum(v for v in other_row.values() if isinstance(v, (int, float)))
        unsupported_to_supported = 0.0
        for pred_crop, value in other_row.items():
            if pred_crop == "other_leaf":
                continue
            if isinstance(value, (int, float)):
                unsupported_to_supported += float(value)

        if resolved["beans_potato_confusion_rate"] is None and beans_potato_total > 0:
            resolved["beans_potato_confusion_rate"] = float(beans_potato_conf / beans_potato_total)
            metric_sources["beans_potato_confusion_rate"].append(f"{path}::counts")

        if resolved["supported_to_unsupported_rate"] is None and supported_total > 0:
            resolved["supported_to_unsupported_rate"] = float(supported_to_unsupported / supported_total)
            metric_sources["supported_to_unsupported_rate"].append(f"{path}::counts")

        if resolved["unsupported_to_supported_rate"] is None and unsupported_total > 0:
            resolved["unsupported_to_supported_rate"] = float(unsupported_to_supported / unsupported_total)
            metric_sources["unsupported_to_supported_rate"].append(f"{path}::counts")

        need_fallback = any(resolved[k] is None for k in resolved)

    searched_paths = {
        "supported_vs_unsupported_candidates": supported_unsupported_candidates,
        "crop_pair_candidates": crop_pair_candidates,
    }
    return resolved, metric_sources, searched_paths


def _missing_metric_message(metric_name, searched_paths):
    sources = ["test_evaluation.confusion_diagnostics", "test_evaluation.top_level"]
    for key in ["supported_vs_unsupported_candidates", "crop_pair_candidates"]:
        for path in searched_paths.get(key, []):
            sources.append(path)
    return f"missing {metric_name} after checking: {', '.join(sources)}"


def _resolve_field_audit_gate_drift(field_summary):
    gate_counts = _safe_get(field_summary, ["gate_counts"], {})
    if not isinstance(gate_counts, dict):
        return {
            "legacy_gates": [],
            "warning": "field_audit_summary missing gate_counts; cannot verify gate-schema parity",
        }

    legacy = sorted(
        g for g in gate_counts.keys()
        if isinstance(g, str) and g.startswith(LEGACY_FIELD_AUDIT_GATE_PREFIX)
    )
    return {
        "legacy_gates": legacy,
        "warning": None,
    }


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    allow_blocked = "--allow-blocked" in argv

    test_eval = _load_json(TEST_EVAL_PATH)
    field = _load_json(FIELD_AUDIT_PATH)

    policy = _safe_get(test_eval, ["strict_release_policy"], {})
    field_min_supported = float(policy.get("field_min_supported_confident_rate", 0.80))
    beans_potato_max = float(policy.get("beans_potato_confusion_max", 0.02))
    supported_unsupported_max = float(policy.get("supported_unsupported_confusion_max", 0.02))

    resolved_confusion, metric_sources, searched_paths = _resolve_confusion_metrics(test_eval)
    field_gate_drift = _resolve_field_audit_gate_drift(field)
    beans_potato_rate = resolved_confusion["beans_potato_confusion_rate"]
    supported_to_unsupported_rate = resolved_confusion["supported_to_unsupported_rate"]
    unsupported_to_supported_rate = resolved_confusion["unsupported_to_supported_rate"]

    blockers = []

    # Test-side strict checks.
    if beans_potato_rate is None:
        blockers.append(_missing_metric_message("beans_potato_confusion_rate", searched_paths))
    elif beans_potato_rate > beans_potato_max:
        blockers.append(
            f"beans_potato_confusion_rate={beans_potato_rate:.4f} > {beans_potato_max:.4f}"
        )

    if supported_to_unsupported_rate is None:
        blockers.append(_missing_metric_message("supported_to_unsupported_rate", searched_paths))
    elif supported_to_unsupported_rate > supported_unsupported_max:
        blockers.append(
            f"supported_to_unsupported_rate={supported_to_unsupported_rate:.4f} > {supported_unsupported_max:.4f}"
        )

    if unsupported_to_supported_rate is None:
        blockers.append(_missing_metric_message("unsupported_to_supported_rate", searched_paths))
    elif unsupported_to_supported_rate > supported_unsupported_max:
        blockers.append(
            f"unsupported_to_supported_rate={unsupported_to_supported_rate:.4f} > {supported_unsupported_max:.4f}"
        )

    if field_gate_drift["warning"]:
        blockers.append(field_gate_drift["warning"])
    if field_gate_drift["legacy_gates"]:
        blockers.append(
            "field_audit_summary appears to come from legacy gate logic "
            f"({', '.join(field_gate_drift['legacy_gates'])}); regenerate analysis_outputs/field_audit_summary.json "
            "using scripts/audit_field_photos.py before release gating"
        )

    # Field-side strict checks.
    per_crop = _safe_get(field, ["per_crop_reliability"], {})
    field_rates = {}
    for crop in REQUIRED_CROPS:
        rate = float(_safe_get(per_crop, [crop, "supported_confident_rate"], 0.0))
        field_rates[crop] = rate
        if rate < field_min_supported:
            blockers.append(
                f"field_{crop}_supported_confident_rate={rate:.4f} < {field_min_supported:.4f}"
            )

    verdict = "PASS" if len(blockers) == 0 else "BLOCKED"

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "test_evaluation": TEST_EVAL_PATH,
            "field_audit_summary": FIELD_AUDIT_PATH,
        },
        "policy": {
            "field_min_supported_confident_rate": field_min_supported,
            "beans_potato_confusion_max": beans_potato_max,
            "supported_unsupported_confusion_max": supported_unsupported_max,
        },
        "measured": {
            "beans_potato_confusion_rate": None if beans_potato_rate is None else round(beans_potato_rate, 6),
            "supported_to_unsupported_rate": None if supported_to_unsupported_rate is None else round(supported_to_unsupported_rate, 6),
            "unsupported_to_supported_rate": None if unsupported_to_supported_rate is None else round(unsupported_to_supported_rate, 6),
            "field_supported_confident_rate_by_crop": {
                k: round(v, 6) for k, v in field_rates.items()
            },
        },
        "metric_sources": metric_sources,
        "searched_metric_sources": searched_paths,
        "field_audit_gate_check": field_gate_drift,
        "blockers": blockers,
        "verdict": verdict,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote release readiness -> {OUT_PATH}")
    print(f"Verdict: {verdict}")
    if blockers:
        print("Blockers:")
        for b in blockers:
            print(f"- {b}")

    # Hard enforcement: BLOCKED must fail CI/deploy pipelines unless explicitly overridden.
    if verdict == "BLOCKED" and not allow_blocked:
        print("Release gate failed: verdict is BLOCKED. Use --allow-blocked only for exploratory runs.")
        sys.exit(2)


if __name__ == "__main__":
    main()
