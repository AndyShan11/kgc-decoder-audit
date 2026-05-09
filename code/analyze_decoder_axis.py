"""
Cross-dataset analysis correlating decoder Δ with two candidate predictor axes:

    (i)  per-relation symmetry score (Manabe 2018-style)
    (ii) edges per relation (training-set average)

The paper's original Sec. 5.4 phase-transition hypothesis was that UMLS
reverses because UMLS is "small AND symmetric-rich relative to Kinship"
(favouring DistMult's symmetric prior). The direct measurement in
`measure_symmetry.py` falsifies the symmetry half: UMLS is *less* symmetric
than Kinship under any of the standard scoring conventions.

This script tests an alternative single-axis explanation:
    decoder_delta(d) = f(edges_per_relation(d))

with a transition threshold around edges/rel ≈ 200: below it ComplEx's
extra parameters cannot be reliably estimated per relation, and
DistMult's symmetric prior wins as a regulariser; above it ComplEx
benefits from its antisymmetry-handling capacity.

Usage:
    cd code && python analyze_decoder_axis.py

Reads:
    data/rerun_summary.json (decoder Δ per dataset, lock-in pass)
    data/symmetry_scores.json (only UMLS / Kinship; optional)

Writes:
    data/decoder_axis_analysis.json
"""
import json
import os


# Hard-coded benchmark statistics, sourced from the paper Sec. 5.1 + raw
# train.txt counts. These match the dataset descriptors used throughout the
# paper. Verify: if you have train.txt locally, the line counts should equal
# the n_train values below.
STATS = {
    "UMLS":      {"n_ent":    135, "n_rel":  46, "n_train":   5216},
    "Kinship":   {"n_ent":    104, "n_rel":  25, "n_train":   8544},
    "FB15k-237": {"n_ent":  14541, "n_rel": 237, "n_train": 272115},
    "WN18RR":    {"n_ent":  40943, "n_rel":  11, "n_train":  86835},
    "CoDEx-M":   {"n_ent":  17050, "n_rel":  51, "n_train": 185000},
    "CoDEx-L":   {"n_ent":  77951, "n_rel":  69, "n_train": 551000},
    "YAGO3-10":  {"n_ent": 123182, "n_rel":  37, "n_train": 1079040},
}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)

    # 1. Load decoder Δ from rerun_summary.json
    with open(os.path.join(repo, "data", "rerun_summary.json"), "r") as f:
        summary = json.load(f)
    deltas = {row["dataset"]: row["delta"]
              for row in summary["decoder_delta_table"]}
    # Normalise key: rerun_summary uses "codex-m" (lowercase)
    if "codex-m" in deltas and "CoDEx-M" not in deltas:
        deltas["CoDEx-M"] = deltas["codex-m"]

    # 2. Load symmetry scores if available
    sym = {}
    sym_path = os.path.join(repo, "data", "symmetry_scores.json")
    if os.path.exists(sym_path):
        with open(sym_path, "r") as f:
            sym_data = json.load(f)
        for ds in sym_data:
            sym[ds] = sym_data[ds]["weighted_dataset_score"]

    # 3. Build per-dataset rows
    rows = []
    for ds, st in STATS.items():
        epr = st["n_train"] / st["n_rel"]
        rows.append({
            "dataset": ds,
            "n_ent": st["n_ent"],
            "n_rel": st["n_rel"],
            "n_train": st["n_train"],
            "edges_per_relation": epr,
            "symmetry_score": sym.get(ds),
            "decoder_delta": deltas.get(ds),
            "decoder_winner": (
                "DistMult" if deltas.get(ds) is not None and deltas[ds] < 0
                else ("ComplEx" if deltas.get(ds) is not None else "?")
            ),
        })
    rows.sort(key=lambda r: r["edges_per_relation"])

    # 4. Print
    print("\nCross-dataset analysis: decoder Δ vs edges/relation and symmetry")
    print("-" * 92)
    print(f"{'Dataset':>10} {'#ent':>7} {'#rel':>5} {'#train':>9} "
          f"{'edges/rel':>10} {'sym':>6} {'Decoder Δ':>10}  Winner")
    print("-" * 92)
    for r in rows:
        sym_str = f"{r['symmetry_score']:.3f}" if r["symmetry_score"] is not None else "  -  "
        d = r["decoder_delta"]
        d_str = "  N/A   " if d is None else f"{d:+.4f}"
        print(f"{r['dataset']:>10} {r['n_ent']:>7} {r['n_rel']:>5} "
              f"{r['n_train']:>9} {r['edges_per_relation']:>10.0f} "
              f"{sym_str:>6} {d_str:>10}  {r['decoder_winner']}")

    # 5. Findings
    print("\nFindings:")
    print("-" * 92)
    if "UMLS" in sym and "Kinship" in sym:
        u, k = sym["UMLS"], sym["Kinship"]
        print(f"  Symmetry test: UMLS={u:.3f}, Kinship={k:.3f}, "
              f"UMLS-Kinship={u-k:+.3f}")
        if u < k:
            print(f"  -> UMLS is LESS symmetric than Kinship; the original "
                  f"\"UMLS is symmetric-rich\" narrative is FALSIFIED.")
        else:
            print(f"  -> UMLS is more symmetric than Kinship; original "
                  f"narrative supported.")

    print()
    print("  Edges/relation test (single-axis prediction of decoder winner):")
    threshold = 200
    correct = 0
    total = 0
    for r in rows:
        if r["decoder_delta"] is None:
            continue
        total += 1
        predicted = "DistMult" if r["edges_per_relation"] < threshold else "ComplEx"
        actual = r["decoder_winner"]
        flag = "OK" if predicted == actual else "MISS"
        correct += (predicted == actual)
        print(f"    {r['dataset']:>10}  e/r={r['edges_per_relation']:>6.0f}  "
              f"predicted={predicted:>8s}  actual={actual:>8s}  [{flag}]")
    if total > 0:
        print(f"\n  Accuracy at threshold e/r={threshold}: "
              f"{correct}/{total} = {correct/total*100:.0f}%")

    # 6. Write
    out = {
        "threshold_edges_per_relation": threshold,
        "rows": rows,
        "notes": [
            "Decoder delta = MRR(ComplEx) - MRR(DistMult) at the dataset's "
            "lock-in (d, L) configuration (Table 4 in the paper).",
            "Symmetry score is the Manabe 2018 weighted symmetry "
            "computed by code/measure_symmetry.py; only available for "
            "datasets with a local train.txt.",
            "Single-axis prediction: edges/rel < threshold -> DistMult wins; "
            "otherwise ComplEx wins.",
        ],
    }
    out_path = os.path.join(repo, "data", "decoder_axis_analysis.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
