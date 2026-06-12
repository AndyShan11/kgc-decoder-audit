#!/usr/bin/env python3
"""Check manuscript/package numbers against local JSON result artifacts.

The paper has two kinds of numbers:
1. JSON-backed cells available in this repository.
2. Headline or legacy cells whose complete source runs are not all present in
   the current JSON folders. Those are reported as provenance warnings rather
   than hard failures.
"""

from __future__ import annotations

import json
import math
import re
import statistics as stats
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

JSON_DIRS = [
    ROOT / "data" / "decoder_diag_jsons",
    ROOT / "decoder_chunked_results_20260604",
    ROOT / "phase1",
    ROOT / "phase2",
    ROOT / "phase3",
]

SERVER_JSON_DIRS = [
    ROOT / "pricai_3day_results_20260609" / "pricai_3day_pack_20260609_113656" / "root_jsons",
]


def scorer_from_path(path: Path, record: dict) -> str:
    name = path.name
    if "_RotatE_" in name:
        return "RotatE"
    if "_TransE_" in name:
        return "TransE"
    if "_ComplEx_" in name:
        return "ComplEx"
    # Phase-3 CoDEx-L files have ComplEx-sized parameter counts but no suffix.
    if path.parent.name == "phase3" and record.get("dataset") == "CoDEx-L":
        return "ComplEx"
    return "DistMult"


def sample_std(values: list[float]) -> float:
    return stats.stdev(values) if len(values) > 1 else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def load_buckets() -> dict[tuple, list[float]]:
    buckets: dict[tuple, list[float]] = {}
    for directory in JSON_DIRS:
        if not directory.exists():
            continue
        for path in directory.glob("*.json"):
            if path.name in {"rerun_summary.json"}:
                continue
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                config = record["config"]
                mrr = float(record["test"]["MRR"])
            except Exception as exc:
                print(f"WARN skip unreadable JSON {path}: {exc}")
                continue
            key = (
                record.get("dataset"),
                scorer_from_path(path, record),
                int(config.get("d_h")),
                int(config.get("layers")),
                float(config.get("label_smooth")),
            )
            buckets.setdefault(key, []).append(mrr)
    return buckets


def rounded(value: float, digits: int = 3) -> float:
    return round(value + 1e-12, digits)


def require_text(text: str, needle: str, label: str, failures: list[str]) -> None:
    if needle not in text:
        failures.append(f"missing {label}: {needle}")


def check_json_backed(
    buckets: dict[tuple, list[float]],
    server_buckets: dict[tuple, list[float]],
    main_text: str,
) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []

    historical_checks = {
        "FB15k-237 ComplEx L3": (("FB15k-237", "ComplEx", 200, 3, 0.3), 0.421, None),
        "FB15k-237 DistMult L3": (("FB15k-237", "DistMult", 200, 3, 0.3), 0.416, None),
        "WN18RR ComplEx L2": (("WN18RR", "ComplEx", 200, 2, 0.3), 0.478, 0.002),
        "WN18RR DistMult L2": (("WN18RR", "DistMult", 200, 2, 0.3), 0.466, None),
        "YAGO3-10 ComplEx L0 d64": (("YAGO3-10", "ComplEx", 64, 0, 0.3), 0.671, None),
        "YAGO3-10 DistMult L0 d64": (("YAGO3-10", "DistMult", 64, 0, 0.3), 0.665, None),
        "CoDEx-M ComplEx L0": (("codex-m", "ComplEx", 200, 0, 0.3), 0.538, 0.000),
        "CoDEx-M DistMult L0": (("codex-m", "DistMult", 200, 0, 0.3), 0.528, None),
        "UMLS RotatE L2": (("UMLS", "RotatE", 200, 2, 0.3), 0.8490, 0.0065),
        "UMLS TransE L2": (("UMLS", "TransE", 200, 2, 0.3), 0.1005, 0.0087),
        "Kinship RotatE L2": (("Kinship", "RotatE", 200, 2, 0.3), 0.7266, 0.0091),
        "Kinship TransE L2": (("Kinship", "TransE", 200, 2, 0.3), 0.0472, 0.0044),
        "WN18RR RotatE L2": (("WN18RR", "RotatE", 200, 2, 0.3), 0.4930, 0.0021),
        "WN18RR TransE L2": (("WN18RR", "TransE", 200, 2, 0.3), 0.2069, 0.0023),
    }

    server_checks = {
        "UMLS ComplEx clean rerun": (("UMLS", "ComplEx", 200, 2, 0.3), 0.9427, 0.0072),
        "UMLS DistMult clean rerun": (("UMLS", "DistMult", 200, 2, 0.3), 0.9205, 0.0051),
        "Kinship ComplEx clean rerun": (("Kinship", "ComplEx", 200, 2, 0.3), 0.8534, 0.0060),
        "Kinship DistMult clean rerun": (("Kinship", "DistMult", 200, 2, 0.3), 0.7103, 0.0057),
        "WN18RR ComplEx clean rerun": (("WN18RR", "ComplEx", 200, 2, 0.3), 0.4854, 0.0012),
        "WN18RR DistMult clean rerun": (("WN18RR", "DistMult", 200, 2, 0.3), 0.4748, 0.0010),
        "YAGO3-10 ComplEx d128 clean rerun": (("YAGO3-10", "ComplEx", 128, 0, 0.3), 0.6971, 0.0048),
        "YAGO3-10 ComplEx d256 clean rerun": (("YAGO3-10", "ComplEx", 256, 0, 0.3), 0.7047, 0.0000),
        "CoDEx-L ComplEx clean rerun": (("codex-l", "ComplEx", 200, 0, 0.3), 0.5402, 0.0030),
        "CoDEx-L DistMult clean rerun": (("codex-l", "DistMult", 200, 0, 0.3), 0.5355, 0.0016),
    }

    for label, (key, expected_mean, expected_std) in historical_checks.items():
        values = buckets.get(key)
        if not values:
            failures.append(f"{label}: missing JSON bucket {key}")
            continue
        got_mean = mean(values)
        got_std = sample_std(values)
        if abs(got_mean - expected_mean) > 0.0006:
            failures.append(f"{label}: mean {got_mean:.6f} != expected {expected_mean:.6f}")
        if expected_std is not None and abs(got_std - expected_std) > 0.0006:
            failures.append(f"{label}: std {got_std:.6f} != expected {expected_std:.6f}")

    for label, (key, expected_mean, expected_std) in server_checks.items():
        values = server_buckets.get(key)
        if not values:
            failures.append(f"{label}: missing server JSON bucket {key}")
            continue
        got_mean = mean(values)
        got_std = sample_std(values)
        if abs(got_mean - expected_mean) > 0.0006:
            failures.append(f"{label}: mean {got_mean:.6f} != expected {expected_mean:.6f}")
        if expected_std is not None and abs(got_std - expected_std) > 0.0006:
            failures.append(f"{label}: std {got_std:.6f} != expected {expected_std:.6f}")

    # Diagnostic spreads as printed in Table 2. The shared five-dataset spread
    # is separated from the decoder-only seven-dataset spread to avoid an
    # apples-to-oranges headline comparison.
    shared_decoder_deltas = [0.005, 0.012, 0.006, 0.010, 0.005]
    shared_decoder_spread = max(shared_decoder_deltas) - min(shared_decoder_deltas)
    if rounded(shared_decoder_spread, 3) != 0.007:
        failures.append(f"shared decoder spread recompute failed: {shared_decoder_spread:.6f}")
    decoder_deltas = [0.022, 0.143, *shared_decoder_deltas]
    decoder_spread = max(decoder_deltas) - min(decoder_deltas)
    if rounded(decoder_spread, 3) != 0.138:
        failures.append(f"decoder-only seven-dataset spread recompute failed: {decoder_spread:.6f}")
    require_text(main_text, "0.007", "shared decoder spread in main.tex", failures)
    require_text(main_text, "0.138", "decoder-only spread in main.tex", failures)
    require_text(main_text, "0.075", "encoder spread in main.tex", failures)
    require_text(main_text, "$0.697{\\pm}0.005$", "YAGO3-10 clean rerun table cell", failures)
    require_text(main_text, "$0.540{\\pm}0.003$", "CoDEx-L clean rerun table cell", failures)
    require_text(main_text, "$72.7{\\pm}0.5$", "CoDEx-L H@10 clean rerun table cell", failures)

    stale_main_patterns = [
        "$L{=}0$, $d{=}128$, 6 seeds",
        "$0.697{\\pm}0.002$",
        "$0.541{\\pm}0.003$",
        "$73.1{\\pm}0.3$",
        "Our $L{=}0$ ComplEx on YAGO3-10 ($0.6968$)",
        "$\\mathbf{0.9427{\\pm}0.0072}$",
        "$\\mathbf{0.4929{\\pm}0.0019}$",
        "otherwise 3 seeds (42, 123, 456)",
        "four-point dimension sweep ($L{=}0$ ComplEx, our recipe held fixed, 3 seeds per cell)",
        "DistMult/ComplEx use the primary recipe for each dataset; distance decoders",
        "1.84",
    ]
    for pattern in stale_main_patterns:
        if pattern in main_text:
            failures.append(f"main.tex contains stale wording/number: {pattern}")

    return failures, warnings


def check_readme(readme_text: str) -> list[str]:
    failures: list[str] = []
    required = [
        "Shared five-dataset decoder delta",
        "0.007",
        "Decoder-only seven-dataset delta",
        "0.138",
        "0.143 MRR",
        "0.8534 +/- 0.0060",
        "0.7103 +/- 0.0057",
        "provenance/recipe-sensitivity warning",
        "recipe-conditional diagnostic",
        "RTX 4050 Laptop GPU with 6GB",
    ]
    for needle in required:
        require_text(readme_text, needle, f"README text {needle}", failures)
    stale_patterns = ["0.124", "1.65x", "1.84x", "+0.036", "The Decoder Is the Lever"]
    for pattern in stale_patterns:
        if pattern in readme_text:
            failures.append(f"README contains stale wording/number: {pattern}")
    return failures


def main() -> int:
    main_path = ROOT / "main.tex"
    if not main_path.exists():
        main_path = ROOT / "paper" / "main.tex"
    main_text = main_path.read_text(encoding="utf-8")
    readme_text = (ROOT / "README.md").read_text(encoding="utf-8")
    buckets = load_buckets()
    original_dirs = list(JSON_DIRS)
    JSON_DIRS[:] = SERVER_JSON_DIRS
    server_buckets = load_buckets()
    JSON_DIRS[:] = original_dirs

    failures, warnings = check_json_backed(buckets, server_buckets, main_text)
    failures.extend(check_readme(readme_text))

    print("JSON-backed aggregate summary")
    print("=============================")
    for key in sorted(buckets):
        dataset, scorer, d_h, layers, label_smooth = key
        if dataset in {"UMLS", "Kinship", "FB15k-237", "WN18RR", "YAGO3-10", "codex-m", "CoDEx-L"}:
            values = buckets[key]
            print(
                f"{dataset:10s} {scorer:8s} d={d_h:<3d} L={layers:<1d} "
                f"ls={label_smooth:<3.1f} n={len(values)} "
                f"MRR={mean(values):.4f} std={sample_std(values):.4f}"
            )

    print()
    print("Clean server-rerun aggregate summary")
    print("====================================")
    for key in sorted(server_buckets):
        dataset, scorer, d_h, layers, label_smooth = key
        if dataset in {"UMLS", "Kinship", "WN18RR", "YAGO3-10", "codex-l"}:
            values = server_buckets[key]
            print(
                f"{dataset:10s} {scorer:8s} d={d_h:<3d} L={layers:<1d} "
                f"ls={label_smooth:<3.1f} n={len(values)} "
                f"MRR={mean(values):.4f} std={sample_std(values):.4f}"
            )

    print()
    if warnings:
        print("Warnings")
        print("========")
        for warning in warnings:
            print(f"- {warning}")
        print()

    if failures:
        print("Failures")
        print("========")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("PASS: checked JSON-backed manuscript/package numbers.")
    if warnings:
        print("NOTE: provenance warnings above still need human confirmation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
