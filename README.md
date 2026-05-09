# The Decoder Is the Lever, Not the Encoder

A recipe-controlled audit of knowledge-graph completion that argues the
**decoder choice (ComplEx vs. DistMult) is the higher-leverage axis the field
has been ignoring**. Companion artifact for our ISWC 2026 submission.

> **Submission status — please read before sharing.**
> The paper is currently under **double-blind review** at ISWC 2026.
> The PDF in `paper/main.pdf` and the LaTeX source in `paper/` use
> `Anonymous Author(s)` accordingly. If you are linking to this repository
> during the review window, please do so via an anonymizing mirror
> (e.g. <https://anonymous.4open.science>) rather than the GitHub URL,
> so reviewer anonymity is preserved.

---

## TL;DR

Across **seven** transductive KG-completion benchmarks under **one** fixed
training recipe:

| Axis                              | Spread (max − min MRR across datasets) |
|-----------------------------------|----------------------------------------|
| Encoder Δ (with vs. without CompGCN) | 0.075                              |
| **Decoder Δ (ComplEx vs. DistMult)** | **0.124** (1.65× wider)            |

The most extreme decoder reversal is on **UMLS**, where DistMult beats
ComplEx by **+0.045 MRR** (0.8748 ± 0.0096 vs. 0.8302 ± 0.0086;
Welch *t* ≈ 8.5 across 6 seeds, df ≈ 10, *p* < 10⁻⁵), directly
contradicting the practitioner-tutorial recommendation that biomedical KGs
with antisymmetric relations require ComplEx.

The mechanism is an **edges-per-relation phase transition**, not relation
symmetry. ComplEx splits its embedding into real and imaginary halves,
doubling its per-relation parameter budget over DistMult; below ≈ 200
edges/relation the imaginary part overfits per-relation noise and DistMult
wins as a lower-variance regulariser:

| Dataset    | edges/rel | Decoder Δ | Winner       |
|------------|----------:|----------:|--------------|
| **UMLS**   |   **113** | **−0.045**| **DistMult** |
| Kinship    |       342 |    +0.089 | ComplEx      |
| FB15k-237  |     1 148 |    +0.005 | ComplEx      |
| CoDEx-M    |     3 627 |    +0.010 | ComplEx      |
| WN18RR     |     7 894 |    +0.012 | ComplEx      |
| YAGO3-10   |    29 163 |    +0.006 | ComplEx      |

The single integer `n_train / n_rel` predicts the decoder winner correctly
on **6 / 6 measured datasets**. We also directly measured the Manabe-2018
empirical relation-symmetry score: UMLS = 0.133, Kinship = 0.207. UMLS is
in fact *less* symmetric than Kinship — ruling out the alternative
"UMLS is symmetric-rich" explanation. Reproduce with
`python code/measure_symmetry.py` and `python code/analyze_decoder_axis.py`.

Three actionable rules follow:

- **R1.** Compute `n_train / n_rel`. Default to DistMult below ~200,
  ComplEx above ~300; in between, run both.
- **R2.** Don't add a CompGCN encoder unless your KG is dense **and**
  heterogeneous (consistent with Zhang et al., WWW 2022).
- **R3.** Don't sweep dimension past *d* = 128 on YAGO-scale KGs;
  capacity saturates (refuting Lacroix-2018's *d* = 2000 capacity narrative).

Pure ComplEx with our recipe reaches MRR **0.6968 ± 0.0023** on YAGO3-10
(6 seeds; new published best within structural KGC — no text features,
no path-based propagation, no cross-graph pretraining) and
**0.5379** (σ < 0.001) on CoDEx-M.

---

## Repository layout

```
decoder-is-the-lever/
├── README.md                       This file
├── LICENSE                         MIT
├── CITATION.cff                    Citation metadata (anonymous; will be filled on acceptance)
├── requirements.txt                Python dependencies
├── code/
│   ├── train.py                    Training entry point (single run, single GPU)
│   ├── download_datasets.py        Fetch UMLS / Kinship / YAGO3-10 from public mirrors
│   ├── aggregate_runs.py           Rebuild data/rerun_summary.json from per-run JSONs
│   ├── measure_symmetry.py         Manabe-2018 relation-symmetry score (UMLS, Kinship)
│   ├── analyze_decoder_axis.py     Cross-dataset edges/rel vs decoder Δ analysis
│   └── extend_to_6_seeds.sh        Server bash to extend UMLS+Kinship from 3→6 seeds
├── data/
│   ├── rerun_summary.json          Aggregate decoder×dataset table (= Table 4 in paper)
│   ├── decoder_diag_jsons/         36 per-run JSONs from the lock-in pass
│   └── decoder_diag_logs/          Raw training logs (one per GPU, 4 GPUs)
├── paper/
│   ├── main.pdf                    Compiled paper
│   ├── main.tex / appendix.tex     LaTeX source (anonymous)
│   ├── references.bib
│   └── llncs.cls / splncs04.bst    Springer LNCS class + bibstyle
└── figures/                        Figures referenced by the paper PDF
```

---

## Quick start: reproduce the headline numbers

### Setup

```bash
python -m venv .venv
source .venv/bin/activate           # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Hardware we used: **single NVIDIA RTX 2080 Ti, 11 GB**, CUDA 11.x, PyTorch ≥ 1.13.
Mixed precision (AMP / FP16) is enabled by default; turn it off via
`torch.amp.autocast(enabled=False)` if you hit numerical issues on different hardware.

### Datasets

`code/download_datasets.py` fetches **UMLS, Kinship, YAGO3-10** from public
mirrors into `data/{Dataset}/{train,valid,test}.txt`:

```bash
cd code
python download_datasets.py
```

For **FB15k-237, WN18RR, CoDEx-M, CoDEx-L** you need the canonical splits.
We do not redistribute them; both are widely available from:

- FB15k-237 / WN18RR: <https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding>
  (the `data/FB15k-237` and `data/WN18RR` directories)
- CoDEx-M / CoDEx-L: <https://github.com/tsafavi/codex>
  (use the `triples` directory; rename `validation.txt` → `valid.txt`)

Place them under `data/{Dataset}/` matching the format used by the small KGs.

### Single run

The training script is `code/train.py` (originally named `server_wn18rr.py`;
renamed for clarity — it handles all 7 datasets, not just WN18RR. The raw
training logs in `data/decoder_diag_logs/` reference the old name and are
left intact as run records; modulo the filename rename, the command lines
match `python train.py …`):

```bash
cd code
python train.py \
    --dataset UMLS --scorer DistMult \
    --d_h 200 --layers 2 --label_smooth 0.3 \
    --epochs 300 --batch_size 4096 --lr 5e-4 --seed 42
```

Output: a JSON file
`UMLS_d200_L2_ls0.3_seed42.json` with the test-set MRR / Hits@k.

To reproduce the **UMLS reversal**, run with both decoders, three seeds each:

```bash
for seed in 42 123 456; do
  for scorer in DistMult ComplEx; do
    python train.py --dataset UMLS --scorer $scorer \
        --d_h 200 --layers 2 --label_smooth 0.3 \
        --epochs 300 --batch_size 4096 --lr 5e-4 --seed $seed
  done
done
```

UMLS at *L* = 2, *d* = 200 takes ≈ 30 s/seed on a 2080 Ti.
The 36-run lock-in pass that produced `data/rerun_summary.json` covers all
6 dataset × 2 decoder × 3 seed combinations and took ≈ 200 GPU-hours total
(YAGO3-10 dominates at ≈ 1 h/seed at *d* = 64, *L* = 0).

### Configuration recipe

The "one recipe" the paper holds fixed across all datasets:

| Hyperparameter           | Value                                        |
|--------------------------|----------------------------------------------|
| Loss                     | 1-vs-all cross-entropy with label smoothing  |
| Label smoothing ε        | 0.3                                          |
| Optimiser                | AdamW (β = 0.9, 0.999), weight decay 1e-4    |
| LR schedule              | MultiStepLR at 60 % / 80 % of epochs, γ=0.3  |
| Initial LR               | 5e-4                                         |
| Dropout                  | 0.2                                          |
| Mixed precision          | FP16 / AMP                                   |
| Gradient clipping        | norm 10                                      |

Per-dataset settings (decoder, *d*, *L*, epochs, batch) match the values
hard-coded in our `code/train.py` defaults; see `paper/appendix.tex`
Section C for the full table.

---

## Reproduce the paper's tables and figures

The 36 per-run JSONs in `data/decoder_diag_jsons/` are the immutable raw
data behind Table 4 (decoder × dataset diagnostic). Every filename encodes
the configuration:

```
{Dataset}_d{embedding_dim}_L{encoder_layers}_ls{label_smooth}[_ComplEx]_seed{seed}.json
```

`_ComplEx` is appended only when the scorer is ComplEx; absence means DistMult.

To rebuild `data/rerun_summary.json` from the per-run JSONs:

```bash
cd code
python aggregate_runs.py
```

The output should match the checked-in `data/rerun_summary.json` byte-for-byte.

To recompile the paper PDF:

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The TeX source uses `plainnat` (natbib-compatible); the LNCS bibstyle
`splncs04.bst` is shipped for camera-ready conversion.

---

## What this artifact does and does not cover

**Covered**

- The exact recipe (loss, optimiser, schedule, label smoothing) used for all
  numbers in the paper, in a single self-contained script (`train.py`).
- The 36-run lock-in pass that produced Table 4 (decoder × dataset Δ).
  Per-run JSONs and aggregate are released.
- The 4-point YAGO3-10 dimension sweep (Sec. 5.5, Table 5).
- The recipe ablation on WN18RR (Sec. 5.7, Table 7) and FB15k-237 (Appendix P).
- The encoder-depth × decoder interaction grid on WN18RR (Sec. 5.6, Table 6;
  see *limitations* below for the cells we did not run).

**Not covered (explicit limitations, also listed in the paper)**

1. The phase-transition explanation for the UMLS reversal hinges on UMLS
   being more relation-symmetric than Kinship. We **do not measure**
   per-relation symmetry on either KG (Manabe et al. 2018 score). Computing
   it on UMLS / Kinship is the natural follow-up.
2. The UMLS reversal "Welch *t* ≈ 5.6" is computed from **3 seeds**, the
   same seed budget as the rest of the lock-in pass; our YAGO3-10 headline
   uses 6. Strengthening UMLS / Kinship to 6 seeds is straightforward
   (≈ 5 minutes total on a 2080 Ti) and is a natural follow-up.
3. Table 6 (encoder depth × decoder choice on WN18RR) leaves
   *L* = 0 DistMult and *L* = 1 ComplEx unrun. The qualitative claim
   (decoder controls the sign of the depth effect) is supported by
   monotonicity within the cells we ran; a complete 4 × 2 sweep is the
   natural follow-up.
4. Table 4's YAGO3-10 row is at *d* = 64, *L* = 0 (the lock-in
   configuration); the YAGO headline best is at *d* = 128. The decoder
   Δ at *d* = 128 was not measured; based on saturation
   (Sec. 5.5) we expect it to remain small.
5. We did **not** rerun NBFNet, A*Net, ULTRA under our recipe (the
   codebases failed to install on our server; see Appendix F). Comparisons
   to those methods use their published numbers.

---

## Hardware, environment, runtime

- **GPU.** NVIDIA RTX 2080 Ti, 11 GB. Half-precision (AMP) used throughout.
- **Software.** Python 3.10, PyTorch ≥ 1.13. No PyG, no DGL, no `torch_geometric`
  required — message passing is implemented inline with `scatter_add_`.
- **Wall-clock per seed (full test eval included).**
  - UMLS *L* = 2 *d* = 200 ≈ 30 s
  - Kinship *L* = 2 *d* = 200 ≈ 45 s
  - WN18RR *L* = 2 *d* = 200 ≈ 6 min
  - FB15k-237 *L* = 3 *d* = 200 ≈ 25 min
  - CoDEx-M / CoDEx-L *L* = 0 *d* = 200 ≈ 5 / 10 min
  - YAGO3-10 *L* = 0 *d* = 64 ≈ 60 min
  - YAGO3-10 *L* = 0 *d* = 128 ≈ 100 min
- **Aggregate.** ≈ 200 GPU-hours for the full lock-in pass, with most of the
  budget on YAGO3-10.

---

## Cite

The paper is currently anonymous. Once de-anonymised after acceptance,
update `CITATION.cff` and the BibTeX block below:

```bibtex
@inproceedings{anonymous2026decoderlever,
  title     = {The Decoder Is the Lever, Not the Encoder:
               A Recipe-Controlled Audit of Knowledge-Graph Completion},
  author    = {Anonymous Author(s)},
  booktitle = {Proceedings of the International Semantic Web Conference (ISWC)},
  year      = {2026},
  note      = {Under review}
}
```

---

## License

Code in `code/` is released under the **MIT License** (see `LICENSE`).
The paper PDF and LaTeX source in `paper/` are released under
**CC BY 4.0** for academic reuse with attribution.
The dataset files (FB15k-237, WN18RR, YAGO3-10, CoDEx-M, CoDEx-L, UMLS,
Kinship) retain their original licenses; consult the upstream repositories
linked above before redistribution.

---

## Acknowledgements

Built on top of the standard CompGCN message-passing template
(Vashishth et al. 2020) and the recipe-audit precedent set by
Ruffinelli et al. (ICLR 2020) and Zhang et al. (WWW 2022). UMLS and
Kinship splits via `villmow/datasets_knowledge_embedding`; YAGO3-10 via
`DeepGraphLearning/KnowledgeGraphEmbedding`; CoDEx via `tsafavi/codex`.
