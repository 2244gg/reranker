import json
raw = json.load(open('results/gpt-4.1-mini/lastfm1k/raw/recommendation_results_full.json', 'r', encoding='utf-8'))
rer = json.load(open('results/gpt-4.1-mini/lastfm1k/reranked/recommendation_results_full.json', 'r', encoding='utf-8'))

uid = 'user_000004'
print('user_info:', raw[uid]['user_info'])
print()

scenarios = ['neutral', 'gender', 'age_group']
for sc in scenarios:
    print(f'=== scenario={sc} ===')
    rn = raw[uid]['results'][sc]
    rrn = rer[uid]['results'][sc]
    print('raw keys:', list(rn.keys()))
    print('rer keys:', list(rrn.keys()))

    # original list
    orig_list = rn.get('recommendations') or rn.get('top_recommendations') or rn.get('parsed_titles')
    print(f'orig (top 10):', orig_list[:10] if orig_list else '(none)')

    # reranked list
    rer_list = rrn.get('reranked_recommendations') or rrn.get('recommendations') or rrn.get('top_recommendations')
    print(f'rer  (top 10):', rer_list[:10] if rer_list else '(none)')

    # popularity column used + B values
    print('column used:', rrn.get('popularity_column_used'))
    scores = rrn.get('rerank_scores') or []
    if scores:
        bs = [round(s.get('B_i_norm', 0), 4) for s in scores[:10]]
        ps = [round(s.get('P_i_norm', 0), 4) for s in scores[:10]]
        ss = [round(s.get('S_i', 0), 4) for s in scores[:10]]
        print('B (first10 by S):', bs)
        print('P (first10 by S):', ps)
        print('S (first10 by S):', ss)
    print('missing_titles count:', len(rrn.get('missing_titles', [])))
    print('alpha used:', rrn.get('alpha_used'))
    print()
