"""Time PopularityTable + ItemScoreTable construction with new title_key indexing."""
import time
import pandas as pd

print("[t0] start", flush=True)
t0 = time.time()
df = pd.read_csv("data/processed/lastfm1k_track/items_with_popularity.csv")
print(f"[t1] CSV loaded in {time.time()-t0:.1f}s, shape={df.shape}", flush=True)

# Simulate v2 _apply_b_transform
t1 = time.time()
df2 = df.copy()
meta = {"item_id", "title"}
slice_cols = [c for c in df2.columns if c not in meta]
for c in slice_cols:
    col = df2[c].astype(float)
    mask_pos = col > 0
    ranked = col[mask_pos].rank(method="average", pct=True)
    new = pd.Series(0.0, index=col.index)
    new[mask_pos] = ranked
    df2[c] = new
print(f"[t2] rank transform in {time.time()-t1:.1f}s", flush=True)

# PopularityTable construction
t2 = time.time()
from src.experiment.runner import PopularityTable
pt = PopularityTable(df2)
print(f"[t3] PopularityTable build in {time.time()-t2:.1f}s, rows={len(pt._df)}", flush=True)

# ItemScoreTable construction
t3 = time.time()
from src.popularity.item_scoring import ItemScoreTable
st = ItemScoreTable.from_popularity(pt._df)
print(f"[t4] ItemScoreTable build in {time.time()-t3:.1f}s", flush=True)

# Sample lookups
t4 = time.time()
hits_lookup = 0
for title in ["Yesterday - The Beatles", "Bohemian Rhapsody (Remastered) - Queen", "Creep - Radiohead"] * 100:
    pt.lookup(title, "neutral_All")
    hits_lookup += 1
print(f"[t5] {hits_lookup} lookups in {time.time()-t4:.3f}s", flush=True)

print(f"[total] {time.time()-t0:.1f}s elapsed", flush=True)
