"""Aggregate Phase 1 + Phase 2 results into one analysis file.

Reads JSONs from both phase1/ and phase2/ directories (relative to this script).
"""
import json, glob, os, statistics as st, collections

ROOT = os.path.dirname(os.path.abspath(__file__))
PHASE1 = os.path.join(os.path.dirname(ROOT), 'phase1')
PHASE2 = ROOT  # phase2/

# Existing paper baselines
PAPER_BASELINES = {
    # (dataset, scorer, d_h, L, ls): (mrr, std, n_seeds)
    ('UMLS','DistMult',200,2,0.3):    (0.8748, 0.0096, 6),
    ('UMLS','ComplEx',200,2,0.3):     (0.8302, 0.0086, 6),
    ('Kinship','DistMult',200,2,0.3): (0.6744, 0.0078, 6),
    ('Kinship','ComplEx',200,2,0.3):  (0.7634, 0.0065, 6),
    ('FB15k-237','ComplEx',200,2,0.3): (0.4211, 0.0013, 3),
    # WN18RR L=2 ComplEx from paper Table 5: 0.4810
    ('WN18RR','ComplEx',200,2,0.3): (0.4810, 0.0044, 3),
    ('WN18RR','DistMult',200,2,0.3): (0.4670, 0.0024, 3),
}

# Aggregate by (dataset, scorer, d_h, L, ls)
buckets = collections.defaultdict(list)
files_seen = []

for d in (PHASE1, PHASE2):
    for f in glob.glob(os.path.join(d, '*.json')):
        if 'aggregate' in os.path.basename(f) or 'rerun_summary' in f:
            continue
        try:
            r = json.load(open(f, encoding='utf-8'))
        except Exception as e:
            print(f'skip {f}: {e}')
            continue
        scorer = 'ComplEx' if '_ComplEx_' in os.path.basename(f) else 'DistMult'
        key = (r['dataset'], scorer, r['config']['d_h'],
               r['config']['layers'], r['config']['label_smooth'])
        buckets[key].append((r['seed'], r['test']['MRR'], r['params'], r['test'].get('Hits@10', 0)))
        files_seen.append(os.path.basename(f))

print(f'Loaded {len(files_seen)} JSON files into {len(buckets)} cells\n')

def mean_std(key):
    if key in buckets:
        mrrs = [m for _, m, _, _ in buckets[key]]
        return sum(mrrs)/len(mrrs), st.pstdev(mrrs) if len(mrrs) > 1 else 0, len(mrrs)
    if key in PAPER_BASELINES:
        m, s, n = PAPER_BASELINES[key]
        return m, s, n
    return None, None, 0

print('=' * 90)
print('SECTION A — FB15k-237 e/r sweep (PHASE 2, key reviewer-defense)')
print('=' * 90)
print(f'{"Mean e/r":>10} {"#train":>8} {"DM MRR":>14} {"CX MRR":>14} {"Decoder Δ":>10}  winner')
print('-' * 90)

# Compute actual mean e/r from each subsample's train.txt (cached: just use ratio for simplicity)
FB_SUB_INFO = {
    # nominal_ratio: (n_train, actual_mean_e_per_r)
    50:    (11837, 11837/237),
    100:   (23619, 23619/237),
    200:   (42822, 42822/237),
    400:   (69973, 69973/237),
    800:   (105232, 105232/237),
}
for ratio in [50, 100, 200, 400, 800]:
    ds_name = f'FB15k-237-er{ratio}'
    dm = mean_std((ds_name,'DistMult',200,2,0.3))
    cx = mean_std((ds_name,'ComplEx',200,2,0.3))
    n_train, mean_er = FB_SUB_INFO[ratio]
    delta = cx[0] - dm[0]
    winner = 'DistMult' if delta < 0 else 'ComplEx'
    dm_str = f'{dm[0]:.4f}±{dm[1]:.4f}' if dm[0] else 'n/a'
    cx_str = f'{cx[0]:.4f}±{cx[1]:.4f}' if cx[0] else 'n/a'
    print(f'{mean_er:>10.1f} {n_train:>8d} {dm_str:>14} {cx_str:>14} {delta:>+10.4f}  {winner}')
# Full FB15k-237 reference (I task DM + paper CX)
dm_full = mean_std(('FB15k-237','DistMult',200,2,0.3))
cx_full = mean_std(('FB15k-237','ComplEx',200,2,0.3))
delta_full = cx_full[0] - dm_full[0]
print(f'{1148.2:>10.1f} {272115:>8d} {dm_full[0]:.4f}±{dm_full[1]:.4f} {cx_full[0]:.4f}±{cx_full[1]:.4f} {delta_full:>+10.4f}  {"DistMult" if delta_full < 0 else "ComplEx"}')

print()
print('=' * 90)
print('SECTION B — WN18RR e/r sweep (PHASE 2, cross-dataset replication test)')
print('=' * 90)
print(f'{"Mean e/r":>10} {"#train":>8} {"DM MRR":>14} {"CX MRR":>14} {"Decoder Δ":>10}  winner')
print('-' * 90)

WN_SUB_INFO = {
    50:  (550, 50.0),
    100: (1080, 98.2),
    200: (2080, 189.1),
}
for ratio in [50, 100, 200]:
    ds_name = f'WN18RR-er{ratio}'
    dm = mean_std((ds_name,'DistMult',200,2,0.3))
    cx = mean_std((ds_name,'ComplEx',200,2,0.3))
    n_train, mean_er = WN_SUB_INFO[ratio]
    delta = cx[0] - dm[0]
    winner = 'DistMult' if delta < 0 else 'ComplEx'
    dm_str = f'{dm[0]:.4f}±{dm[1]:.4f}' if dm[0] else 'n/a'
    cx_str = f'{cx[0]:.4f}±{cx[1]:.4f}' if cx[0] else 'n/a'
    print(f'{mean_er:>10.1f} {n_train:>8d} {dm_str:>14} {cx_str:>14} {delta:>+10.4f}  {winner}')
# WN18RR full reference (paper)
print(f'{7894.1:>10.1f} {86835:>8d} {"0.4670±0.0024":>14} {"0.4810±0.0044":>14} {0.4810-0.4670:>+10.4f}  ComplEx')

print()
print('=' * 90)
print('SECTION C — UMLS multi-recipe (Limitations L4)')
print('=' * 90)
print(f'{"label_smooth":>12} {"DM MRR":>14} {"CX MRR":>14} {"Decoder Δ":>10}  winner')
print('-' * 90)
for ls in [0.0, 0.1, 0.3]:
    dm = mean_std(('UMLS','DistMult',200,2,ls))
    cx = mean_std(('UMLS','ComplEx',200,2,ls))
    delta = cx[0] - dm[0]
    winner = 'DistMult' if delta < 0 else 'ComplEx'
    dm_str = f'{dm[0]:.4f}±{dm[1]:.4f}' if dm[0] else 'n/a'
    cx_str = f'{cx[0]:.4f}±{cx[1]:.4f}' if cx[0] else 'n/a'
    print(f'ε={ls:<10} {dm_str:>14} {cx_str:>14} {delta:>+10.4f}  {winner}')

print()
print('=' * 90)
print('SECTION D — UMLS + Kinship L-scan (Limitations L11)')
print('=' * 90)
print(f'{"Dataset":>10} {"L":>4} {"DM MRR":>14} {"CX MRR":>14} {"Encoder Δ (CX)":>16}')
print('-' * 90)
for ds in ['UMLS', 'Kinship']:
    cx_L2 = mean_std((ds,'ComplEx',200,2,0.3))
    for L in [0, 2, 3]:
        dm = mean_std((ds,'DistMult',200,L,0.3))
        cx = mean_std((ds,'ComplEx',200,L,0.3))
        enc_delta = cx[0] - cx_L2[0] if cx[0] and cx_L2[0] else None
        dm_str = f'{dm[0]:.4f}±{dm[1]:.4f}' if dm[0] else 'n/a'
        cx_str = f'{cx[0]:.4f}±{cx[1]:.4f}' if cx[0] else 'n/a'
        ed_str = f'{enc_delta:+.4f}' if enc_delta else ''
        print(f'{ds:>10} {L:>4} {dm_str:>14} {cx_str:>14} {ed_str:>16}')
    print()

print('=' * 90)
print('SECTION E — Phase 1 d-scan recap (UMLS / Kinship / FB15k-237 / WN18RR)')
print('=' * 90)
print(f'{"Dataset":>10} {"Scorer":>9} {"d=100":>14} {"d=200":>14} {"d=400":>14}')
print('-' * 90)
for ds in ['UMLS', 'Kinship', 'WN18RR', 'FB15k-237']:
    for scorer in ['DistMult', 'ComplEx']:
        cells = []
        for d in [100, 200, 400]:
            r = mean_std((ds, scorer, d, 2, 0.3))
            cells.append(f'{r[0]:.4f}±{r[1]:.4f}' if r[0] else 'n/a' + ' '*8)
        print(f'{ds:>10} {scorer:>9} {cells[0]:>14} {cells[1]:>14} {cells[2]:>14}')

print()
print('=' * 90)
print('TAKEAWAYS')
print('=' * 90)

# Check 1: e/r threshold in FB15k-237
print('1) FB15k-237 e/r sweep — does phase transition replicate?')
flip_found = False
prev_winner = None
for ratio in [50, 100, 200, 400, 800]:
    ds_name = f'FB15k-237-er{ratio}'
    dm = mean_std((ds_name,'DistMult',200,2,0.3))
    cx = mean_std((ds_name,'ComplEx',200,2,0.3))
    winner = 'DistMult' if cx[0] < dm[0] else 'ComplEx'
    n_train = FB_SUB_INFO[ratio][0]
    actual_er = n_train / 237
    delta = cx[0] - dm[0]
    print(f'   er={ratio} (actual e/r={actual_er:.0f}): Δ={delta:+.4f}, winner={winner}')
    if prev_winner is not None and prev_winner != winner:
        flip_found = True
        print(f'   *** PHASE TRANSITION between er={ratio_prev} and er={ratio} ***')
    prev_winner = winner
    ratio_prev = ratio
if not flip_found:
    print('   No phase transition observed in FB15k-237 sweep range.')

# Check 2: WN18RR
print('\n2) WN18RR e/r sweep — does it replicate UMLS-style reversal?')
prev_winner = None; flip_found = False
for ratio in [50, 100, 200]:
    ds_name = f'WN18RR-er{ratio}'
    dm = mean_std((ds_name,'DistMult',200,2,0.3))
    cx = mean_std((ds_name,'ComplEx',200,2,0.3))
    winner = 'DistMult' if cx[0] < dm[0] else 'ComplEx'
    n_train = WN_SUB_INFO[ratio][0]
    actual_er = n_train / 11
    delta = cx[0] - dm[0]
    print(f'   er={ratio} (actual e/r={actual_er:.0f}): Δ={delta:+.4f}, winner={winner}')

# Check 3: ε robustness on UMLS
print('\n3) UMLS multi-recipe — is decoder winner ε-robust?')
all_dm_wins = True
for ls in [0.0, 0.1, 0.3]:
    dm = mean_std(('UMLS','DistMult',200,2,ls))
    cx = mean_std(('UMLS','ComplEx',200,2,ls))
    winner = 'DistMult' if cx[0] < dm[0] else 'ComplEx'
    print(f'   ε={ls}: DM={dm[0]:.4f}, CX={cx[0]:.4f}, Δ={cx[0]-dm[0]:+.4f}, winner={winner}')
    if winner != 'DistMult': all_dm_wins = False
print('   →', 'DistMult wins at ALL ε ✓ (recipe-robust)' if all_dm_wins else 'WINNER FLIPS — recipe-dependent ✗')
