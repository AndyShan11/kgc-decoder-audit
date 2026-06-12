# Recipe-Controlled Decoder Audit for Structural KGC

This repository contains a PRICAI 2026 submission draft and the local result
artifacts used to support it. The paper is framed as a recipe-conditional audit:
under one fixed training recipe, a matched DistMult-vs-ComplEx check can change
the diagnostic conclusion about structural KGC benchmarks.

## Current PRICAI artifact

- Main LaTeX source: `paper/main.tex`
- PRICAI PDF build target: `paper/main.pdf`
- Technical appendix source, retained locally but not included in the 16-page
  PRICAI build: `paper/appendix.tex`
- Submission manifest: `submission_manifest.md`

PRICAI 2026 requires Springer's LNAI/LNCS format, double anonymous review, and
at most 16 pages including references.

## Headline finding

The paper is now framed as a recipe-controlled decoder audit (RCDA), not as a
claim that decoders are universally more important than encoders. The diagnostic
table separates the shared five-dataset grid from the decoder-only small-KG
evidence:

| View | Diagnostic spread |
|---|---:|
| Shared five-dataset decoder delta (ComplEx vs. DistMult) | 0.007 |
| Shared five-dataset encoder delta (with vs. without CompGCN) | 0.075 |
| Decoder-only seven-dataset delta (including UMLS/Kinship) | 0.138 |

The strongest stable small-KG decoder gap is on Kinship: ComplEx beats DistMult
by 0.143 MRR in the new 6-seed RTX 2080 Ti rerun
(`0.8534 +/- 0.0060` vs. `0.7103 +/- 0.0057`).
UMLS is treated as a provenance/recipe-sensitivity warning: the clean 6-seed
server rerun favours ComplEx by 0.022 MRR, while an earlier run bundle favoured
DistMult.

The result should be read as a recipe-conditional diagnostic, not as a general
claim that decoders dominate encoders or that one decoder is globally preferable.

## Result provenance

- `data/decoder_diag_jsons/`: 36 per-run JSONs from the 2026-05-04 lock-in pass.
  These cover historical 3-seed DistMult/ComplEx diagnostic cells and auxiliary
  scans retained for provenance.
- `data/rerun_summary.json`: aggregate summary from the 36-run lock-in pass.
  This file does not include the later clean RTX 2080 Ti rerun rows used for
  UMLS/Kinship, CoDEx-L, and YAGO3-10 headline updates.
- `phase1/`, `phase2/`, `phase3/`: auxiliary scans and analysis files,
  including dimension scans, e/r sweeps, and CoDEx-L runs.
- `decoder_chunked_results_20260604/`: RotatE/TransE spot-check results.
- `pricai_3day_results_20260609/`: selected RTX 2080 Ti server package files
  for the PRICAI 3-day rerun, including logs, root JSONs, code snapshot, and
  `nvidia-smi` output. Checkpoints and large run directories are intentionally
  not committed.
- `analysis_outputs/pricai_3day_analysis.md`: deduplicated aggregate analysis
  of the server package, including mean/std/spread and the UMLS provenance
  warning.

Run `python scripts/check_results_consistency.py` before submission to recompute
available JSON-backed means/stds/spreads and flag manuscript-package mismatches.
Run `python scripts/analyze_pricai_3day_results.py` to regenerate the PRICAI
3-day server-rerun aggregate tables.

## Rebuilding the paper

Use the LNCS build sequence from `paper/`:

```text
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Close any open PDF viewer before rebuilding. A previous LaTeX log showed
`I can't write on file main.pdf`, which happens when the output PDF is locked.

## Local machine boundary

The current local machine has an NVIDIA RTX 4050 Laptop GPU with 6GB memory, and
the default Python environment does not have PyTorch installed. Use it for LaTeX
builds, JSON aggregation, figure/table checks, and tiny smoke tests only.
Headline training runs, YAGO/CoDEx-L runs, and RTX 2080 Ti timing claims belong
on the 11GB RTX 2080 Ti server.
