# PRICAI 2026 Submission Manifest

Date: 2026-06-09

## Submit

- `paper/main.pdf` or a freshly rebuilt PRICAI PDF after final checks.

## Source Files

- `paper/main.tex`: active PRICAI 16-page manuscript source.
- `paper/references.bib`: bibliography.
- `paper/llncs.cls`, `paper/splncs04.bst`: LNCS class and bibliography style.
- `figures/fig0_decoder_vs_encoder.pdf`
- `figures/fig1_main_mrr.pdf`
- `figures/fig2_edges_per_rel.pdf`
- `figures/fig3_ablation_wn18rr.pdf`
- `figures/fig5_ls_consistency.pdf`
- `figures/fig6_yago_spotlight.pdf`

## Retained Locally, Not In Main PRICAI PDF

- `paper/appendix.tex`: technical appendix retained for camera-ready or supplementary
  reuse, currently omitted to satisfy the 16-page including-references limit.
- `pricai_strict_review_action_plan.md`: local review/action notes.
- `scripts/check_results_consistency.py`: local numeric consistency check.
- `scripts/analyze_pricai_3day_results.py`: deduplicated analysis script for
  the downloaded RTX 2080 Ti rerun package.
- `data/`, `phase1/`, `phase2/`, `phase3/`, `decoder_*`: experiment artifacts
  and server outputs, not part of the anonymous PDF unless PRICAI explicitly
  requests supplementary material.
- `pricai_3day_results_20260609/` and
  `pricai_3day_pack_20260609_113656.tar.gz`: server rerun logs, JSONs,
  checkpoints, code snapshot, and environment record retained locally for
  provenance.
- `analysis_outputs/`: generated aggregate CSV/Markdown reports from local
  result analysis.

## Pre-Submission Checklist

- PDF has 16 pages or fewer including references.
- PDF paper size matches LNCS/LNAI expectations.
- First page contains title and abstract but no author names, affiliations, or
  acknowledgements.
- No unpublished self-identifying work is cited.
- `python scripts/check_results_consistency.py` passes.
- `python scripts/analyze_pricai_3day_results.py` has been rerun after unpacking
  the server package.
- LaTeX log has no unresolved citations, undefined references, fatal errors, or
  overfull boxes that affect readability.
