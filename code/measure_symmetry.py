"""
Compute Manabe-2018-style empirical symmetry score per relation, for UMLS and
Kinship, and at the dataset level. This is the measurement we flagged as a
limitation in Sec. 5.4 of the paper ("what we do not measure"); running this
script directly tests the relation-structure half of the phase-transition
explanation.

Definition (per relation r):

    sym(r) = | { (h, t) : (h, r, t) and (t, r, h) both in train } |
             ----------------------------------------------------
                       | { (h, t) : (h, r, t) in train } |

Dataset-level score: edge-weighted average of per-relation sym(r).
sym(r) ∈ [0, 1]: 1 means r is fully symmetric in the training set,
0 means r is fully antisymmetric.

CPU-only, ~ 1 second on UMLS + Kinship.

Usage:
    cd code && python measure_symmetry.py
    # If data/UMLS/train.txt or data/Kinship/train.txt is missing,
    # run code/download_datasets.py first (or pass --data_root).

Output:
    - Per-dataset per-relation table, printed to stdout
    - data/symmetry_scores.json (machine-readable, all relations)
"""
import argparse
import json
import os
from collections import defaultdict


def load_train_edges(train_path):
    """Read train.txt as (h, r, t) tab-separated triples."""
    triples = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            h, r, t = parts
            triples.append((h, r, t))
    return triples


def compute_symmetry(train_path):
    """Return (per_relation_scores, dataset_weighted, n_edges_total)."""
    triples = load_train_edges(train_path)

    # Group endpoint pairs by relation
    edges_per_rel = defaultdict(set)
    for h, r, t in triples:
        edges_per_rel[r].add((h, t))

    rel_scores = {}
    total_sym = 0
    total_edges = 0
    for r, edges in edges_per_rel.items():
        sym_count = sum(1 for (h, t) in edges if (t, h) in edges)
        n = len(edges)
        score = sym_count / n if n > 0 else 0.0
        rel_scores[r] = {
            "n_edges": n,
            "sym_pair_count": sym_count,
            "score": score,
        }
        total_sym += sym_count
        total_edges += n

    weighted = total_sym / total_edges if total_edges > 0 else 0.0
    return rel_scores, weighted, total_edges


def print_dataset_summary(name, rel_scores, weighted, n_edges):
    print(f"\n=== {name} ===")
    print(f"Train edges      : {n_edges}")
    print(f"Relations        : {len(rel_scores)}")
    print(f"Weighted sym(r)  : {weighted:.4f}")
    print(f"Unweighted mean  : "
          f"{sum(s['score'] for s in rel_scores.values())/len(rel_scores):.4f}")

    sorted_rels = sorted(rel_scores.items(), key=lambda x: -x[1]["score"])
    print(f"\n  Top {min(8, len(sorted_rels))} most symmetric relations:")
    print(f"  {'relation':<35s} {'n_edges':>8s} {'sym':>8s}")
    for r, info in sorted_rels[: min(8, len(sorted_rels))]:
        print(f"  {r[:35]:<35s} {info['n_edges']:>8d} {info['score']:>8.4f}")
    print(f"\n  Top {min(8, len(sorted_rels))} most antisymmetric relations:")
    print(f"  {'relation':<35s} {'n_edges':>8s} {'sym':>8s}")
    for r, info in sorted_rels[-min(8, len(sorted_rels)):]:
        print(f"  {r[:35]:<35s} {info['n_edges']:>8d} {info['score']:>8.4f}")


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    ap.add_argument("--data_root", type=str,
                    default=os.path.join(repo, "data"),
                    help="Root containing {Dataset}/train.txt subdirectories")
    ap.add_argument("--datasets", nargs="+",
                    default=["UMLS", "Kinship"],
                    help="Which datasets to score")
    args = ap.parse_args()

    results = {}
    for ds in args.datasets:
        train_path = os.path.join(args.data_root, ds, "train.txt")
        if not os.path.exists(train_path):
            print(f"[skip] {train_path} not found. "
                  f"Run code/download_datasets.py first to fetch UMLS / Kinship.")
            continue
        rel_scores, weighted, n_edges = compute_symmetry(train_path)
        results[ds] = {
            "weighted_dataset_score": weighted,
            "unweighted_mean": (
                sum(s["score"] for s in rel_scores.values()) / len(rel_scores)
                if rel_scores else 0.0
            ),
            "n_edges_train": n_edges,
            "n_relations": len(rel_scores),
            "per_relation": rel_scores,
        }
        print_dataset_summary(ds, rel_scores, weighted, n_edges)

    if "UMLS" in results and "Kinship" in results:
        u = results["UMLS"]["weighted_dataset_score"]
        k = results["Kinship"]["weighted_dataset_score"]
        print("\n=== UMLS vs Kinship ===")
        print(f"  UMLS    weighted symmetry : {u:.4f}")
        print(f"  Kinship weighted symmetry : {k:.4f}")
        diff = u - k
        print(f"  UMLS - Kinship            : {diff:+.4f}")
        if diff > 0:
            print(f"  -> UMLS is more symmetric than Kinship (gap {diff:.4f}).")
            print(f"     Phase-transition reading in Sec. 5.4 is empirically "
                  f"supported.")
        else:
            print(f"  -> UMLS is LESS symmetric than Kinship (gap {-diff:.4f}).")
            print(f"     The phase-transition reading needs reframing: the "
                  f"reversal cannot be explained by symmetry alone.")

    out_path = os.path.join(args.data_root, "symmetry_scores.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
