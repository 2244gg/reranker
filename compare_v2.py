"""Show v2 (rank-norm B + protect_top=5) orig vs rerank head-to-head."""
import json
v2 = json.load(open('results_v2/gpt-4.1-mini/lastfm1k/metrics/aggregate.json', encoding='utf-8'))['scenarios']
a07 = json.load(open('results/gpt-4.1-mini/lastfm1k/metrics/aggregate.json', encoding='utf-8'))['scenarios']

scenarios = ['neutral', 'gender', 'age_group', 'gender+age_group']

print('=== v2 algorithm (rank-norm B + protect_top=5) — orig vs reranked ===\n')
print(f'{"scenario":<22} {"metric":<14} {"orig":>9} {"rer":>9} {"delta":>9}')
print('-' * 72)
for sc in scenarios:
    s = v2.get(sc, {})
    if not s:
        continue
    for label, ok, rk in [
        ('hit_rate',     'mean_orig_hit_rate',  'mean_rer_hit_rate'),
        ('NDCG@20',      'mean_orig_ndcg_at_k', 'mean_rer_ndcg_at_k'),
        ('MRR',          'mean_orig_mrr',       'mean_rer_mrr'),
        ('hits',         'mean_orig_hits',      'mean_rer_hits'),
        ('aplt_slice',   'mean_orig_aplt',      'mean_rer_aplt'),
        ('aplt_neutral', 'mean_orig_aplt_neutral','mean_rer_aplt_neutral'),
        ('unpop_avg(rank)','mean_orig_unpopularity_average','mean_rer_unpopularity_average'),
    ]:
        ov = s.get(ok)
        rv = s.get(rk)
        if ov is None or rv is None:
            continue
        d = rv - ov
        print(f'{sc:<22} {label:<14} {ov:>9.4f} {rv:>9.4f} {d:>+9.4f}')
    # corpus level
    for k in ('orig_unique_recommendations','rer_unique_recommendations',
              'orig_coverage','rer_coverage','orig_gini','rer_gini'):
        if k in s:
            v = s[k]
            print(f'{sc:<22} {k:<28} {v:>9.4f}')
    print()

# baseline (alpha=0.7) for hit_rate / NDCG / MRR side-by-side
print('=== Baseline (α=0.7, raw B, no protect) — for side-by-side ===\n')
print(f'{"scenario":<22} {"metric":<14} {"orig":>9} {"rer":>9} {"delta":>9}')
print('-' * 72)
for sc in scenarios:
    s = a07.get(sc, {})
    if not s:
        continue
    for label, ok, rk in [
        ('hit_rate',     'mean_orig_hit_rate',  'mean_rer_hit_rate'),
        ('NDCG@20',      'mean_orig_ndcg_at_k', 'mean_rer_ndcg_at_k'),
        ('MRR',          'mean_orig_mrr',       'mean_rer_mrr'),
        ('hits',         'mean_orig_hits',      'mean_rer_hits'),
    ]:
        ov = s.get(ok)
        rv = s.get(rk)
        if ov is None or rv is None: continue
        d = rv - ov
        print(f'{sc:<22} {label:<14} {ov:>9.4f} {rv:>9.4f} {d:>+9.4f}')
    print()
