import json, glob, statistics as st, collections

# Existing baselines from paper (6 seeds for UMLS/Kinship, 3 for FB15k-237)
BASELINES = {
    ('UMLS','DistMult',200):    (0.8748, 0.0096, 6),
    ('UMLS','ComplEx',200):     (0.8302, 0.0086, 6),
    ('Kinship','DistMult',200): (0.6744, 0.0078, 6),
    ('Kinship','ComplEx',200):  (0.7634, 0.0065, 6),
    ('FB15k-237','DistMult',200): (0.4157, 0.0025, 3),
    ('FB15k-237','ComplEx',200):  (0.4211, 0.0013, 3),
}

buckets = collections.defaultdict(list)
for f in sorted(glob.glob('*.json')):
    d = json.load(open(f))
    scorer = 'ComplEx' if 'ComplEx' in f else 'DistMult'
    key = (d['dataset'], scorer, d['config']['d_h'])
    buckets[key].append((d['seed'], d['test']['MRR'], d['params']))

print(f"{'Dataset':<11} {'Scorer':<9} {'d':>4} {'params':>8} {'n':>2} {'MRR':>8} {'std':>7}  vs d=200 baseline")
print('-' * 80)

rows = []
for k in [
    ('UMLS','DistMult',100),  ('UMLS','DistMult',200),  ('UMLS','DistMult',400),
    ('UMLS','ComplEx',100),   ('UMLS','ComplEx',200),   ('UMLS','ComplEx',400),
    ('Kinship','DistMult',100),  ('Kinship','DistMult',200),  ('Kinship','DistMult',400),
    ('Kinship','ComplEx',100),   ('Kinship','ComplEx',200),   ('Kinship','ComplEx',400),
    ('FB15k-237','ComplEx',100), ('FB15k-237','ComplEx',200), ('FB15k-237','ComplEx',400),
]:
    if k[2] == 200:
        mrr, std, n = BASELINES[k]
        params = '-'
        marker = '(paper)'
    elif k in [(x[0],x[1],x[2]) for x in [(*k,)]] and buckets.get(k):
        seeds_mrrs = buckets[k]
        mrrs = [m for _, m, _ in seeds_mrrs]
        mrr = sum(mrrs)/len(mrrs)
        std = st.pstdev(mrrs) if len(mrrs) > 1 else 0
        n = len(mrrs)
        params = seeds_mrrs[0][2]
        b_mrr = BASELINES.get((k[0],k[1],200), (None,))[0]
        delta = (mrr - b_mrr) if b_mrr else None
        marker = f'Δ={delta:+.4f}' if delta is not None else ''
    else:
        print(f"{k[0]:<11} {k[1]:<9} {k[2]:>4} {'MISSING':>8}")
        continue
    p = f'{params:>8,}' if isinstance(params, int) else f'{params:>8}'
    print(f"{k[0]:<11} {k[1]:<9} {k[2]:>4} {p} {n:>2} {mrr:>8.4f} {std:>7.4f}  {marker}")

print()
print('=' * 80)
print('MECHANISM CHECKS')
print('=' * 80)

def m(k):
    if k[2] == 200: return BASELINES[k][0]
    if buckets.get(k):
        return sum(x[1] for x in buckets[k]) / len(buckets[k])
    return None

# Check 1: UMLS — does ComplEx@d=100 (param-matched to DM@200) recover?
cx100 = m(('UMLS','ComplEx',100));  cx200 = m(('UMLS','ComplEx',200));  cx400 = m(('UMLS','ComplEx',400))
dm100 = m(('UMLS','DistMult',100)); dm200 = m(('UMLS','DistMult',200)); dm400 = m(('UMLS','DistMult',400))
print(f'UMLS ComplEx scan:    d=100→{cx100:.4f}, d=200→{cx200:.4f}, d=400→{cx400:.4f}')
print(f'UMLS DistMult scan:   d=100→{dm100:.4f}, d=200→{dm200:.4f}, d=400→{dm400:.4f}')
print()
print(f'  H1: Does ComplEx improve when shrunk to d=100 (param-matched to DM@200)?')
print(f'     ComplEx@100 vs ComplEx@200: D = {cx100-cx200:+.4f}    {"[YES] supports" if cx100 > cx200 else "[NO] refutes"} the capacity mechanism')
print(f'  H2: Does DistMult degrade when scaled to d=400 (param-matched to CX@200)?')
print(f'     DistMult@400 vs DistMult@200: D = {dm400-dm200:+.4f}  {"[YES] supports" if dm400 < dm200 else "[NO] refutes"}')
print(f'  H3: Does ComplEx degrade further at d=400 (over-parametrised)?')
print(f'     ComplEx@400 vs ComplEx@200: D = {cx400-cx200:+.4f}    {"[YES] monotone overparam" if cx400 < cx200 else "[NO] non-monotone"}')

cx100K = m(('Kinship','ComplEx',100));  cx200K = m(('Kinship','ComplEx',200));  cx400K = m(('Kinship','ComplEx',400))
dm100K = m(('Kinship','DistMult',100)); dm200K = m(('Kinship','DistMult',200)); dm400K = m(('Kinship','DistMult',400))
print()
print(f'Kinship ComplEx scan: d=100→{cx100K:.4f}, d=200→{cx200K:.4f}, d=400→{cx400K:.4f}')
print(f'Kinship DistMult scan:d=100→{dm100K:.4f}, d=200→{dm200K:.4f}, d=400→{dm400K:.4f}')
print()
print(f'  Kinship is ABOVE threshold (342 e/r): expect dimension less binding.')
print(f'     ComplEx@100 vs @200 Δ = {cx100K-cx200K:+.4f}    ComplEx@400 vs @200 Δ = {cx400K-cx200K:+.4f}')
print(f'     DistMult@100 vs @200 Δ = {dm100K-dm200K:+.4f}    DistMult@400 vs @200 Δ = {dm400K-dm200K:+.4f}')

cxF100 = m(('FB15k-237','ComplEx',100));  cxF200 = m(('FB15k-237','ComplEx',200)); cxF400 = m(('FB15k-237','ComplEx',400))
print()
print(f'FB15k-237 ComplEx scan: d=100→{cxF100:.4f}, d=200→{cxF200:.4f}, d=400→{cxF400 if cxF400 else "?":>}')
print(f'  FB15k-237 is FAR ABOVE threshold (1148 e/r): expect d=100 to barely hurt.')
print(f'     ComplEx@100 vs @200 Δ = {cxF100-cxF200:+.4f}')
if cxF400:
    print(f'     ComplEx@400 vs @200 Δ = {cxF400-cxF200:+.4f}')
