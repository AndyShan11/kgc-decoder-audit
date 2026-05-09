"""
Rebuild data/rerun_summary.json from the 36 per-run JSONs in
data/decoder_diag_jsons/.

The output should match the checked-in data/rerun_summary.json byte-for-byte
(modulo float-formatting).

Usage:
    cd code && python aggregate_runs.py
"""
import json
import os
import glob
import math
from collections import defaultdict


def std(xs):
    """Sample std (ddof=1), matching the checked-in rerun_summary.json.
    The original was produced with np.std(..., ddof=1) on the per-seed MRR list."""
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    json_dir = os.path.join(repo, "data", "decoder_diag_jsons")
    out_path = os.path.join(repo, "data", "rerun_summary.json")

    # Group per (dataset, d_h, layers, scorer)
    groups = defaultdict(list)
    files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    if not files:
        raise SystemExit(f"No per-run JSONs found in {json_dir}")

    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        dataset = d.get("dataset")
        cfg = d.get("config", {})
        d_h = cfg.get("d_h")
        layers = cfg.get("layers")
        ls = cfg.get("label_smooth")
        # Scorer is encoded only in the filename for "DistMult" baseline
        # (filename has _ComplEx suffix only when scorer=ComplEx).
        scorer = "ComplEx" if "_ComplEx_seed" in os.path.basename(f) else "DistMult"
        seed = d.get("seed")
        mrr = d.get("test", {}).get("MRR")
        if mrr is None:
            continue
        key = (dataset, d_h, layers, ls, scorer)
        groups[key].append((seed, mrr))

    # Per-config aggregates
    per_config = {}
    for (dataset, d_h, layers, ls, scorer), runs in sorted(groups.items()):
        runs.sort()  # deterministic order
        mrrs = [m for _, m in runs]
        cfg_key = f"{dataset}_d{d_h}_L{layers}_ls{ls}_{scorer}"
        per_config[cfg_key] = {
            "dataset": dataset,
            "d_h": d_h,
            "layers": layers,
            "scorer": scorer,
            "label_smooth": ls,
            "n_seeds": len(mrrs),
            "MRR_mean": sum(mrrs) / len(mrrs),
            "MRR_std": std(mrrs),
            "seeds_MRRs": mrrs,
        }

    # Decoder Δ table: pair (ComplEx, DistMult) per dataset
    by_dataset = defaultdict(dict)
    for (dataset, d_h, layers, ls, scorer), runs in groups.items():
        mrrs = [m for _, m in runs]
        by_dataset[dataset][scorer] = {
            "d_h": d_h,
            "layers": layers,
            "label_smooth": ls,
            "n": len(mrrs),
            "mean": sum(mrrs) / len(mrrs),
            "std": std(mrrs),
        }

    deltas = []
    for dataset in sorted(by_dataset):
        if "ComplEx" not in by_dataset[dataset] or "DistMult" not in by_dataset[dataset]:
            continue
        cx = by_dataset[dataset]["ComplEx"]
        dm = by_dataset[dataset]["DistMult"]
        delta = cx["mean"] - dm["mean"]
        # Welch t-statistic across seeds
        var = (cx["std"] ** 2) / cx["n"] + (dm["std"] ** 2) / dm["n"]
        delta_sigma = abs(delta) / math.sqrt(var) if var > 0 else 0.0
        deltas.append({
            "dataset": dataset,
            "d_h": cx["d_h"],
            "layers": cx["layers"],
            "ComplEx_MRR_mean": cx["mean"],
            "ComplEx_MRR_std": cx["std"],
            "ComplEx_n": cx["n"],
            "DistMult_MRR_mean": dm["mean"],
            "DistMult_MRR_std": dm["std"],
            "DistMult_n": dm["n"],
            "delta": delta,
            "delta_sigma": delta_sigma,
        })

    spread = max(d["delta"] for d in deltas) - min(d["delta"] for d in deltas)

    summary = {
        "rerun_metadata": {
            "round_id": "decoder_diag_round_20260504_2056",
            "code_commit_date": "2026-05-04",
            "n_runs": sum(g["n_seeds"] for g in per_config.values()),
            "n_seeds_per_config": 3,
            "purpose": "Lock decoder x dataset main result table on single code version",
        },
        "decoder_delta_table": deltas,
        "decoder_delta_spread": spread,
        "encoder_delta_spread_reference": 0.075,
        "ratio": spread / 0.075 if spread else 0,
        "per_config_aggregates": per_config,
    }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Wrote {out_path}")
    print(f"  {len(deltas)} dataset rows, decoder spread = {spread:.4f}")
    print(f"  Decoder delta table:")
    for row in deltas:
        print(f"    {row['dataset']:>10s}: ComplEx={row['ComplEx_MRR_mean']:.4f} "
              f"DistMult={row['DistMult_MRR_mean']:.4f} delta={row['delta']:+.4f} "
              f"(t={row['delta_sigma']:.2f})")


if __name__ == "__main__":
    main()
