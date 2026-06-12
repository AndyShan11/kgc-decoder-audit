# PRICAI 3-Day Server Rerun Analysis

Result package: `pricai_3day_results_20260609/pricai_3day_pack_20260609_113656`

## New Aggregate Results

| Dataset | Scorer | d | L | batch | chunk | n | seeds | MRR mean | std | H@1 | H@10 | train h/run |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| Kinship | ComplEx | 200 | 2 | 1024 | 0 | 6 | 42,123,456,777,888,999 | 0.8534 | 0.0060 | 76.9 | 98.4 | 0.01 |
| Kinship | DistMult | 200 | 2 | 1024 | 0 | 6 | 42,123,456,777,888,999 | 0.7103 | 0.0057 | 58.5 | 95.3 | 0.01 |
| Kinship | RotatE | 200 | 2 | 4096 | 0 | 3 | 42,123,456 | 0.7266 | 0.0091 | 60.5 | 94.7 | 0.00 |
| Kinship | TransE | 200 | 2 | 4096 | 0 | 3 | 42,123,456 | 0.0472 | 0.0044 | 0.0 | 11.5 | 0.00 |
| UMLS | ComplEx | 200 | 2 | 1024 | 0 | 6 | 42,123,456,777,888,999 | 0.9427 | 0.0072 | 90.6 | 99.3 | 0.01 |
| UMLS | DistMult | 200 | 2 | 1024 | 0 | 6 | 42,123,456,777,888,999 | 0.9205 | 0.0051 | 88.1 | 99.1 | 0.00 |
| UMLS | RotatE | 200 | 2 | 4096 | 0 | 3 | 42,123,456 | 0.8490 | 0.0065 | 77.3 | 95.5 | 0.00 |
| UMLS | TransE | 200 | 2 | 4096 | 0 | 3 | 42,123,456 | 0.1005 | 0.0087 | 0.0 | 33.5 | 0.00 |
| WN18RR | ComplEx | 200 | 2 | 1024 | 0 | 3 | 777,888,999 | 0.4854 | 0.0012 | 45.4 | 54.4 | 0.24 |
| WN18RR | DistMult | 200 | 2 | 1024 | 0 | 3 | 777,888,999 | 0.4748 | 0.0010 | 44.1 | 53.8 | 0.22 |
| WN18RR | RotatE | 200 | 2 | 128 | 2048 | 3 | 777,888,999 | 0.4929 | 0.0019 | 45.8 | 55.9 | 22.17 |
| WN18RR | TransE | 200 | 2 | 128 | 2048 | 3 | 777,888,999 | 0.2016 | 0.0045 | 7.4 | 45.3 | 13.47 |
| YAGO3-10 | ComplEx | 128 | 0 | 2048 | 0 | 3 | 777,888,999 | 0.6971 | 0.0048 | 63.9 | 79.3 | 1.05 |
| YAGO3-10 | ComplEx | 256 | 0 | 1024 | 0 | 1 | 42 | 0.7047 | 0.0000 | 64.6 | 80.1 | 1.53 |
| codex-l | ComplEx | 200 | 0 | 4096 | 0 | 3 | 42,123,456 | 0.5402 | 0.0030 | 44.1 | 72.7 | 0.46 |
| codex-l | DistMult | 200 | 0 | 4096 | 0 | 3 | 42,123,456 | 0.5355 | 0.0016 | 43.7 | 72.3 | 0.39 |

## Key Comparisons

- UMLS new 6-seed batch-1024 primary pair: DistMult 0.9205 +/- 0.0051, ComplEx 0.9427 +/- 0.0072, delta CX-DM = 0.0223.
- UMLS old lock-in primary pair: DistMult 0.8705 +/- 0.0109, ComplEx 0.8344 +/- 0.0020, delta CX-DM = -0.0360.
- Kinship new 6-seed batch-1024 primary pair: DistMult 0.7103 +/- 0.0057, ComplEx 0.8534 +/- 0.0060, delta CX-DM = 0.1431.
- Kinship old lock-in primary pair: DistMult 0.6750 +/- 0.0113, ComplEx 0.7626 +/- 0.0094, delta CX-DM = 0.0876.

- WN18RR DistMult: n=3, MRR 0.4748 +/- 0.0010, H@10 53.8, mean train 0.22 h/run.
- WN18RR ComplEx: n=3, MRR 0.4854 +/- 0.0012, H@10 54.4, mean train 0.24 h/run.
- WN18RR RotatE: n=3, MRR 0.4929 +/- 0.0019, H@10 55.9, mean train 22.17 h/run.
- WN18RR TransE: n=3, MRR 0.2016 +/- 0.0045, H@10 45.3, mean train 13.47 h/run.
- YAGO3-10 d128 new seeds: n=3, MRR 0.6971 +/- 0.0048.
- YAGO3-10 d256 seed 42: MRR 0.7047.
- CoDEx-L L0 provenance rerun: DistMult 0.5355 +/- 0.0016, ComplEx 0.5402 +/- 0.0030, delta CX-DM = 0.0047.

## Interpretation

The new clean small-KG server rerun contradicts the older UMLS reversal: UMLS changes from DistMult-favouring in the lock-in artifact to ComplEx-favouring in the new 6-seed run. Therefore the UMLS reversal should not remain a headline claim. It is better treated as a provenance/recipe-sensitivity warning.
The stable claim that survives is narrower: decoder choice remains a material axis under a fixed recipe, especially on Kinship and WN18RR, but the e/r heuristic is no longer supported as a clean winner separator.
