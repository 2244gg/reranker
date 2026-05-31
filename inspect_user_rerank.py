"""Inspect one user's rerank: compare orig top-20 vs reranked top-20, with B values."""
import json

raw = json.load(open('results/gpt-4.1-mini/lastfm1k/raw/recommendation_results_full.json', encoding='utf-8'))
rer = json.load(open('results/gpt-4.1-mini/lastfm1k/reranked/recommendation_results_full.json', encoding='utf-8'))

# pick a user with KNOWN demographics so column lookups work
target_uid = None
for uid, blk in raw.items():
    info = blk['user_info']
    if info.get('gender') in ('male', 'female') and info.get('age_group') in ('young', 'middle-aged'):
        target_uid = uid
        break

uid = target_uid
print(f'=== user {uid} ===')
print('demographics:', raw[uid]['user_info'])
print()

for sc in ['neutral', 'gender', 'age_group', 'gender+age_group']:
    print(f'--- scenario={sc} ---')
    rn = raw[uid]['results'][sc]
    rrn = rer[uid]['results'][sc]
    orig = rn['recommendations']
    rer_list = rrn['reranked_recommendations']
    pop_col = rrn.get('popularity_column_used')
    alpha = rrn.get('dynamic_alpha')
    print(f'pop_col={pop_col}  alpha={alpha}')
    scores = {s['title']: s for s in rrn.get('rerank_scores', [])}
    print(f'  rank   ORIG title (B)                                      |  RER title (B)')
    for i in range(20):
        o = orig[i] if i < len(orig) else ''
        r = rer_list[i] if i < len(rer_list) else ''
        ob = scores.get(o, {}).get('B_i_norm', None)
        rb = scores.get(r, {}).get('B_i_norm', None)
        ob_s = f'{ob:.3f}' if ob is not None else '-'
        rb_s = f'{rb:.3f}' if rb is not None else '-'
        print(f'  {i+1:>3}  {o[:48]:<48}({ob_s})  |  {r[:48]:<48}({rb_s})')
    # also show items dropped
    orig_set = set(orig[:20])
    rer_set = set(rer_list[:20])
    dropped = [t for t in orig[:20] if t not in rer_set]
    added = [t for t in rer_list if t not in orig_set]
    print(f'  dropped from top20 ({len(dropped)}):', dropped[:5])
    print(f'  added to top20  ({len(added)}):', added[:5])
    print()
