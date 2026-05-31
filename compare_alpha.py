"""Compare alpha=0.7 baseline vs alpha=0.5 vs v2 (rank+protect5) mini runs."""
import json

a07 = json.load(open('results/gpt-4.1-mini/lastfm1k/metrics/aggregate.json', encoding='utf-8'))['scenarios']
a05 = json.load(open('results_a05/gpt-4.1-mini/lastfm1k/metrics/aggregate.json', encoding='utf-8'))['scenarios']
v2  = json.load(open('results_v2/gpt-4.1-mini/lastfm1k/metrics/aggregate.json', encoding='utf-8'))['scenarios']

scenarios = ['neutral', 'gender', 'age_group', 'gender+age_group']
metrics = [
    ('hit_rate',        'mean_orig_hit_rate',           'mean_rer_hit_rate'),
    ('NDCG@20',         'mean_orig_ndcg_at_k',          'mean_rer_ndcg_at_k'),
    ('MRR',             'mean_orig_mrr',                'mean_rer_mrr'),
    ('hits',            'mean_orig_hits',               'mean_rer_hits'),
    ('unpop_avg',       'mean_orig_unpopularity_average','mean_rer_unpopularity_average'),
    ('aplt_slice',      'mean_orig_aplt',               'mean_rer_aplt'),
    ('aplt_neutral',    'mean_orig_aplt_neutral',       'mean_rer_aplt_neutral'),
]
corp = [('coverage', 'orig_coverage', 'rer_coverage'),
        ('gini',     'orig_gini',     'rer_gini')]

print(f'{"scenario":<22} {"metric":<14} {"orig":>9} {"a=0.7":>9} {"a=0.5":>9} {"v2":>9}')
print('-' * 80)
for sc in scenarios:
    s7 = a07.get(sc, {})
    s5 = a05.get(sc, {})
    sv = v2.get(sc, {})
    if not (s7 and s5 and sv):
        continue
    for label, ok, rk in metrics:
        ov  = s7.get(ok)
        r7v = s7.get(rk)
        r5v = s5.get(rk)
        rvv = sv.get(rk)
        def fmt(x):
            return f'{x:>9.4f}' if x is not None else '       -'
        print(f'{sc:<22} {label:<14} {fmt(ov)} {fmt(r7v)} {fmt(r5v)} {fmt(rvv)}')
    # corpus level
    for label, ok, rk in corp:
        ov  = s7.get(ok)
        rvv = sv.get(rk)
        if ov is None or rvv is None:
            continue
        def fmt(x):
            return f'{x:>9.4f}' if x is not None else '       -'
        # for corpus, "orig" is same across all rerank variants (LLM
        # candidates unchanged), but each run logs its own. Show the v2 row
        # for both orig+rer.
        print(f'{sc:<22} {label:<14} {fmt(ov):>9} {"     -":>9} {"     -":>9} {fmt(rvv):>9}')
    print()
