#!/usr/bin/env python3
"""Aggregate and compare the PRICAI 3-day server rerun results."""

from __future__ import annotations

import csv
import json
import math
import re
import statistics as stats
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "pricai_3day_results_20260609" / "pricai_3day_pack_20260609_113656"
OUT_DIR = ROOT / "analysis_outputs"
OUT_DIR.mkdir(exist_ok=True)


def scorer_from_name(path: Path) -> str:
    n = path.name
    if "_RotatE_" in n:
        return "RotatE"
    if "_TransE_" in n:
        return "TransE"
    if "_ComplEx_" in n:
        return "ComplEx"
    return "DistMult"


def tag_from_name(path: Path) -> str:
    n = path.name
    prefixes = [
        "smallkg_6seed_",
        "wn_primary_6seed_",
        "wn_rotate_extra_",
        "wn_transe_extra_",
        "yago_d128_headline_",
        "yago_d256_saturation_",
        "codexl_l0_provenance_",
    ]
    for prefix in prefixes:
        if n.startswith(prefix):
            return prefix[:-1]
    if "root_jsons" in str(path):
        return "root"
    return path.parent.name


def load_records(paths: list[Path], source: str) -> list[dict]:
    rows = []
    seen = set()
    for p in paths:
        if p.suffix.lower() != ".json":
            continue
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        cfg = rec.get("config", {})
        key = (
            rec.get("dataset"),
            scorer_from_name(p),
            rec.get("seed"),
            cfg.get("d_h"),
            cfg.get("layers"),
            cfg.get("label_smooth"),
            cfg.get("batch_size"),
            cfg.get("score_chunk_size", 0),
            cfg.get("epochs"),
        )
        if key in seen:
            continue
        seen.add(key)
        test = rec.get("test", {})
        rows.append(
            {
                "source": source,
                "tag": tag_from_name(p),
                "file": str(p),
                "dataset": rec.get("dataset"),
                "scorer": scorer_from_name(p),
                "seed": rec.get("seed"),
                "d_h": cfg.get("d_h"),
                "layers": cfg.get("layers"),
                "label_smooth": cfg.get("label_smooth"),
                "batch_size": cfg.get("batch_size"),
                "score_chunk_size": cfg.get("score_chunk_size", 0),
                "epochs": cfg.get("epochs"),
                "train_time_s": rec.get("train_time_s"),
                "best_val_mrr": rec.get("best_val_mrr"),
                "MRR": test.get("MRR"),
                "Hits@1": test.get("Hits@1"),
                "Hits@3": test.get("Hits@3"),
                "Hits@10": test.get("Hits@10"),
                "MR": test.get("MR"),
            }
        )
    return rows


def mean(xs):
    return sum(xs) / len(xs)


def std(xs):
    return stats.stdev(xs) if len(xs) > 1 else 0.0


def aggregate(rows: list[dict], include_tags: set[str] | None = None) -> list[dict]:
    buckets: dict[tuple, list[dict]] = {}
    for r in rows:
        if include_tags is not None and r["tag"] not in include_tags:
            continue
        key = (
            r["dataset"],
            r["scorer"],
            r["d_h"],
            r["layers"],
            r["label_smooth"],
            r["batch_size"],
            r["score_chunk_size"],
        )
        buckets.setdefault(key, []).append(r)

    out = []
    for key, vals in sorted(buckets.items()):
        mrrs = [float(v["MRR"]) for v in vals if v["MRR"] is not None]
        if not mrrs:
            continue
        out.append(
            {
                "dataset": key[0],
                "scorer": key[1],
                "d_h": key[2],
                "layers": key[3],
                "label_smooth": key[4],
                "batch_size": key[5],
                "score_chunk_size": key[6],
                "n": len(mrrs),
                "seeds": ",".join(str(v["seed"]) for v in sorted(vals, key=lambda x: int(x["seed"]))),
                "MRR_mean": mean(mrrs),
                "MRR_std": std(mrrs),
                "H1_mean": mean([float(v["Hits@1"]) for v in vals if v["Hits@1"] is not None]),
                "H3_mean": mean([float(v["Hits@3"]) for v in vals if v["Hits@3"] is not None]),
                "H10_mean": mean([float(v["Hits@10"]) for v in vals if v["Hits@10"] is not None]),
                "train_h_mean": (
                    mean([float(v["train_time_s"]) / 3600 for v in vals if v["train_time_s"] is not None])
                    if any(v["train_time_s"] is not None for v in vals)
                    else math.nan
                ),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(x, digits=4):
    if isinstance(x, float) and math.isnan(x):
        return "n/a"
    return f"{x:.{digits}f}"


def pick(aggs, dataset, scorer, d_h, layers, batch_size=None, chunk=None):
    matches = []
    for a in aggs:
        if a["dataset"] == dataset and a["scorer"] == scorer and a["d_h"] == d_h and a["layers"] == layers:
            if batch_size is not None and a["batch_size"] != batch_size:
                continue
            if chunk is not None and a["score_chunk_size"] != chunk:
                continue
            matches.append(a)
    if not matches:
        return None
    matches.sort(key=lambda x: (-(x["n"]), str(x["batch_size"])))
    return matches[0]


def main() -> int:
    # The server bundle stores the same JSONs in root_jsons and inside run_dirs.
    # Prefer root_jsons so seed counts are not doubled.
    pack_jsons = list((PACK / "root_jsons").glob("*.json"))
    new_rows = load_records(pack_jsons, "pricai_3day")
    old_rows = load_records(list((ROOT / "data" / "decoder_diag_jsons").glob("*.json")), "decoder_diag_lockin")
    old_rows += load_records(list((ROOT / "decoder_chunked_results_20260604").glob("*.json")), "decoder_chunked_20260604")
    old_rows += load_records(list((ROOT / "phase3").glob("*.json")), "phase3")

    write_csv(OUT_DIR / "pricai_3day_records.csv", new_rows)
    write_csv(OUT_DIR / "historical_records_subset.csv", old_rows)

    new_aggs = aggregate(new_rows)
    old_aggs = aggregate(old_rows)
    write_csv(OUT_DIR / "pricai_3day_aggregates.csv", new_aggs)
    write_csv(OUT_DIR / "historical_aggregates_subset.csv", old_aggs)

    lines = []
    lines.append("# PRICAI 3-Day Server Rerun Analysis")
    lines.append("")
    lines.append(f"Result package: `{PACK.relative_to(ROOT).as_posix()}`")
    lines.append("")
    lines.append("## New Aggregate Results")
    lines.append("")
    lines.append("| Dataset | Scorer | d | L | batch | chunk | n | seeds | MRR mean | std | H@1 | H@10 | train h/run |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|")
    for a in new_aggs:
        lines.append(
            f"| {a['dataset']} | {a['scorer']} | {a['d_h']} | {a['layers']} | {a['batch_size']} | "
            f"{a['score_chunk_size']} | {a['n']} | {a['seeds']} | {fmt(a['MRR_mean'])} | "
            f"{fmt(a['MRR_std'])} | {fmt(a['H1_mean'],1)} | {fmt(a['H10_mean'],1)} | {fmt(a['train_h_mean'],2)} |"
        )

    lines.append("")
    lines.append("## Key Comparisons")
    lines.append("")

    # Small-KG provenance/recipe sensitivity.
    for ds in ["UMLS", "Kinship"]:
        new_dm = pick(new_aggs, ds, "DistMult", 200, 2, batch_size=1024)
        new_cx = pick(new_aggs, ds, "ComplEx", 200, 2, batch_size=1024)
        old_dm = pick(old_aggs, ds, "DistMult", 200, 2)
        old_cx = pick(old_aggs, ds, "ComplEx", 200, 2)
        if new_dm and new_cx:
            lines.append(
                f"- {ds} new 6-seed batch-1024 primary pair: DistMult {fmt(new_dm['MRR_mean'])} +/- "
                f"{fmt(new_dm['MRR_std'])}, ComplEx {fmt(new_cx['MRR_mean'])} +/- {fmt(new_cx['MRR_std'])}, "
                f"delta CX-DM = {fmt(new_cx['MRR_mean'] - new_dm['MRR_mean'])}."
            )
        if old_dm and old_cx:
            lines.append(
                f"- {ds} old lock-in primary pair: DistMult {fmt(old_dm['MRR_mean'])} +/- "
                f"{fmt(old_dm['MRR_std'])}, ComplEx {fmt(old_cx['MRR_mean'])} +/- {fmt(old_cx['MRR_std'])}, "
                f"delta CX-DM = {fmt(old_cx['MRR_mean'] - old_dm['MRR_mean'])}."
            )

    lines.append("")
    # WN decoders.
    for scorer in ["DistMult", "ComplEx", "RotatE", "TransE"]:
        a = pick(new_aggs, "WN18RR", scorer, 200, 2)
        if a:
            lines.append(
                f"- WN18RR {scorer}: n={a['n']}, MRR {fmt(a['MRR_mean'])} +/- {fmt(a['MRR_std'])}, "
                f"H@10 {fmt(a['H10_mean'],1)}, mean train {fmt(a['train_h_mean'],2)} h/run."
            )

    y128 = pick(new_aggs, "YAGO3-10", "ComplEx", 128, 0)
    y256 = pick(new_aggs, "YAGO3-10", "ComplEx", 256, 0)
    if y128:
        lines.append(
            f"- YAGO3-10 d128 new seeds: n={y128['n']}, MRR {fmt(y128['MRR_mean'])} +/- {fmt(y128['MRR_std'])}."
        )
    if y256:
        lines.append(f"- YAGO3-10 d256 seed {y256['seeds']}: MRR {fmt(y256['MRR_mean'])}.")

    codex_dm = pick(new_aggs, "codex-l", "DistMult", 200, 0)
    codex_cx = pick(new_aggs, "codex-l", "ComplEx", 200, 0)
    if codex_dm and codex_cx:
        lines.append(
            f"- CoDEx-L L0 provenance rerun: DistMult {fmt(codex_dm['MRR_mean'])} +/- {fmt(codex_dm['MRR_std'])}, "
            f"ComplEx {fmt(codex_cx['MRR_mean'])} +/- {fmt(codex_cx['MRR_std'])}, "
            f"delta CX-DM = {fmt(codex_cx['MRR_mean'] - codex_dm['MRR_mean'])}."
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "The new clean small-KG server rerun contradicts the older UMLS reversal: UMLS changes "
        "from DistMult-favouring in the lock-in artifact to ComplEx-favouring in the new "
        "6-seed run. Therefore the UMLS reversal should not remain a headline claim. It is "
        "better treated as a provenance/recipe-sensitivity warning."
    )
    lines.append(
        "The stable claim that survives is narrower: decoder choice remains a material axis "
        "under a fixed recipe, especially on Kinship and WN18RR, but the e/r heuristic is no "
        "longer supported as a clean winner separator."
    )

    report = "\n".join(lines) + "\n"
    (OUT_DIR / "pricai_3day_analysis.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
