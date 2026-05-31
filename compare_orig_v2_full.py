"""Full head-to-head: LLM original (no rerank) vs v2 (rank-norm B + protect_top=5)."""
import json

v2 = json.load(open('results_v2/gpt-4.1-mini/lastfm1k/metrics/aggregate.json', encoding='utf-8'))['scenarios']

scenarios = ['neutral', 'gender', 'age_group', 'gender+age_group']

# All per-user metrics that exist in v2 aggregate.
per_user_keys = [
    ('Precision@20',      'mean_orig_precision',          'mean_rer_precision'),
    ('Recall@20',         'mean_orig_recall',             'mean_rer_recall'),
    ('HitRate@20',        'mean_orig_hit_rate',           'mean_rer_hit_rate'),
    ('MRR',               'mean_orig_mrr',                'mean_rer_mrr'),
    ('NDCG@20',           'mean_orig_ndcg_at_k',          'mean_rer_ndcg_at_k'),
    ('Hits',              'mean_orig_hits',               'mean_rer_hits'),
    ('APLT (slice)',      'mean_orig_aplt',               'mean_rer_aplt'),
    ('APLT (neutral)',    'mean_orig_aplt_neutral',       'mean_rer_aplt_neutral'),
    ('UnpopAvg (rank-B)', 'mean_orig_unpopularity_average','mean_rer_unpopularity_average'),
    ('UnpopTotal (rank-B)', 'mean_orig_unpopularity_total','mean_rer_unpopularity_total'),
    ('DiffAvg',           'mean_orig_difficulty_average', 'mean_rer_difficulty_average'),
    ('DiffTotal',         'mean_orig_difficulty_total',   'mean_rer_difficulty_total'),
]

# Corpus-level diagnostics.
corpus_keys = [
    ('UniqueRecs',  'orig_unique_recommendations', 'rer_unique_recommendations'),
    ('Coverage',    'orig_coverage',               'rer_coverage'),
    ('Gini',        'orig_gini',                   'rer_gini'),
]

print('=' * 78)
print('  ORIGINAL (LLM, no rerank)  vs  v2 (rank-norm B + protect_top=5)')
print('  30 users, gpt-4.1-mini, LastFM-1K @ track granularity')
print('=' * 78)

for sc in scenarios:
    s = v2.get(sc, {})
    if not s:
        continue
    print(f'\n--- scenario: {sc}  (n_users={int(s.get("n_users",0))}) ---')
    print(f'  {"metric":<22} {"orig":>10} {"v2":>10} {"delta":>10} {"%chg":>9}')
    print(f'  {"-"*22} {"-"*10} {"-"*10} {"-"*10} {"-"*9}')

    for label, ok, rk in per_user_keys:
        ov = s.get(ok)
        rv = s.get(rk)
        if ov is None or rv is None:
            continue
        d = rv - ov
        pct = (d / ov * 100.0) if ov else float('inf')
        marker = ''
        if 'APLT' in label or label.startswith('UnpopAvg'):
            marker = ' ✓' if d > 0 else ''
        elif label in ('Precision@20', 'Recall@20', 'HitRate@20', 'MRR', 'NDCG@20', 'Hits'):
            marker = ' ✓' if d >= 0 else ''
        print(f'  {label:<22} {ov:>10.4f} {rv:>10.4f} {d:>+10.4f} {pct:>+8.1f}%{marker}')

    # Corpus block
    print(f'  {"-- corpus --":<22}')
    for label, ok, rk in corpus_keys:
        ov = s.get(ok)
        rv = s.get(rk)
        if ov is None or rv is None:
            continue
        d = rv - ov
        pct = (d / ov * 100.0) if ov else float('inf')
        marker = ''
        if label == 'Gini':
            marker = ' ✓' if d < 0 else ''  # lower Gini = better diversity
        elif label in ('UniqueRecs', 'Coverage'):
            marker = ' ✓' if d >= 0 else ''
        print(f'  {label:<22} {ov:>10.4f} {rv:>10.4f} {d:>+10.4f} {pct:>+8.1f}%{marker}')
