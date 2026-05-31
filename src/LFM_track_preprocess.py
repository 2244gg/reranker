"""LastFM-1K 单曲流行度预处理脚本。

设计完全对照 ``src/ML_preprocess.py``，把数据集换成 LastFM-1K，并把
物品粒度从“艺人”切换为“单曲”——每行 = (Artist, Track) 组合。

公式
------
对每个 (单曲, 用户分组) 数对：

    raw(item, group) = sum_{u in group} 1[positive_feedback(u, item)]

其中 ``positive_feedback(u, item)`` 等价于 ML_preprocess.py 里的
``Rating > 3``，在 LFM-1K 的实现是：

    aggregate_playcount(u, item) >= 2  AND  playcount > user_median_playcount(u)

原因：单曲粒度 ML-1M 的“高分”概念失效（无评分），用 “播放次数显著
高于该用户的中位曲目” 作为代理，并强制 ≥2 排除一次性试听噪声。

归一化（同 ML_preprocess.py）：

    B(item, group) = raw(item, group) / max_item raw(item, group)

最热门那首在该列上 = 1.0；最稀有 = 0.0。

输出列命名（与 ML_preprocess.py 完全一致）：

    neutral_All
    gender_M, gender_F
    age_Youth, age_Middle, age_Senior
    cross_M_Youth, cross_F_Youth, cross_M_Middle, cross_F_Middle,
    cross_M_Senior, cross_F_Senior

性别/年龄未知的用户被排除在所有分组之外（与 ML_preprocess.py 行为一致：
那里 ``map_age_to_group`` 返回 ``Unknown`` 时也不在任何分组里）。

用法
------
    python -m src.LFM_track_preprocess \
        --raw-dir data/raw/lastfm-1k/lastfm-dataset-1K \
        --output  data/processed/lastfm1k_track/items_with_popularity.csv \
        --min-track-listeners 3
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


PROFILE_FILE = "userid-profile.tsv"
EVENTS_FILE = "userid-timestamp-artid-artname-traid-traname.tsv"
EVENTS_CHUNK_SIZE = 1_000_000

TRACK_KEY_SEP = " ||| "        # 内部唯一键（保持与 lastfm1k.py 一致）
TRACK_TITLE_SEP = " - "        # LLM 与 popularity 表的展示分隔符


# ---------------------------------------------------------------------------
# 1. 加载用户画像（对应 ML_preprocess.py::load_ml1m_data）
# ---------------------------------------------------------------------------

def load_lfm1k_users(profile_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        profile_path,
        sep="\t",
        header=None,
        names=["UserID", "Gender", "Age", "Country", "Signup"],
        engine="python",
        on_bad_lines="skip",
    )
    df["UserID"] = df["UserID"].astype(str)
    return df


# ---------------------------------------------------------------------------
# 2. 用户分组（对应 ML_preprocess.py::create_user_groups）
# ---------------------------------------------------------------------------

def map_age_to_group(age) -> str:
    """LFM-1K 2-bucket binning (data-driven).

    Profile has 285 users with valid age (mean 25, std 7); >=35 long tail
    is just 4 users. Two-bucket split keeps both columns ~120-160 users:
        <25     -> Youth
        25..100 -> Middle (absorbs what would have been Senior)
    """
    try:
        a = float(age)
    except (TypeError, ValueError):
        return "Unknown"
    if pd.isna(a) or a <= 0 or a > 100:
        return "Unknown"
    if a < 25:
        return "Youth"
    return "Middle"


def normalize_gender(g) -> str:
    if not isinstance(g, str):
        return "U"
    g = g.strip().lower()
    if g in ("m", "male"):
        return "M"
    if g in ("f", "female"):
        return "F"
    return "U"


def create_user_groups(users: pd.DataFrame) -> Tuple[Dict[str, Dict[str, Set[str]]], pd.DataFrame]:
    users = users.copy()
    users["GenderCode"] = users["Gender"].apply(normalize_gender)
    users["AgeGroup"] = users["Age"].apply(map_age_to_group)

    groups: Dict[str, Dict[str, Set[str]]] = {
        "gender": {
            "M": set(users.loc[users["GenderCode"] == "M", "UserID"]),
            "F": set(users.loc[users["GenderCode"] == "F", "UserID"]),
            "U": set(users.loc[users["GenderCode"] == "U", "UserID"]),
        },
        "age": {
            "Youth":   set(users.loc[users["AgeGroup"] == "Youth",   "UserID"]),
            "Middle":  set(users.loc[users["AgeGroup"] == "Middle",  "UserID"]),
            # Unknown is the LARGEST age cohort in LFM-1K (~708 users).
            # Treat it as a real slice rather than discarding it.
            "Unknown": set(users.loc[users["AgeGroup"] == "Unknown", "UserID"]),
        },
        "cross": {
            "M_Youth":   set(users.loc[(users["GenderCode"] == "M") & (users["AgeGroup"] == "Youth"),   "UserID"]),
            "F_Youth":   set(users.loc[(users["GenderCode"] == "F") & (users["AgeGroup"] == "Youth"),   "UserID"]),
            "U_Youth":   set(users.loc[(users["GenderCode"] == "U") & (users["AgeGroup"] == "Youth"),   "UserID"]),
            "M_Middle":  set(users.loc[(users["GenderCode"] == "M") & (users["AgeGroup"] == "Middle"),  "UserID"]),
            "F_Middle":  set(users.loc[(users["GenderCode"] == "F") & (users["AgeGroup"] == "Middle"),  "UserID"]),
            "U_Middle":  set(users.loc[(users["GenderCode"] == "U") & (users["AgeGroup"] == "Middle"),  "UserID"]),
            "M_Unknown": set(users.loc[(users["GenderCode"] == "M") & (users["AgeGroup"] == "Unknown"), "UserID"]),
            "F_Unknown": set(users.loc[(users["GenderCode"] == "F") & (users["AgeGroup"] == "Unknown"), "UserID"]),
            "U_Unknown": set(users.loc[(users["GenderCode"] == "U") & (users["AgeGroup"] == "Unknown"), "UserID"]),
        },
        "neutral": {
            "All": set(users["UserID"]),
        },
    }
    return groups, users


def user_to_group_keys(
    user_groups: Dict[str, Dict[str, Set[str]]],
) -> Dict[str, List[str]]:
    """反向索引：user_id -> 列名列表（避免每条事件双重 for 嵌套）。"""
    idx: Dict[str, List[str]] = defaultdict(list)
    for group_type, subgroups in user_groups.items():
        for sub_name, members in subgroups.items():
            col = f"{group_type}_{sub_name}"
            for uid in members:
                idx[uid].append(col)
    return idx


# ---------------------------------------------------------------------------
# 3. 流式聚合 (user, track) playcount
# ---------------------------------------------------------------------------

def aggregate_user_track_plays(
    events_path: Path,
    valid_user_ids: Set[str],
    chunksize: int = EVENTS_CHUNK_SIZE,
) -> Dict[str, Counter]:
    """流式扫 ``events_path``，返回 ``per_user[uid]`` = ``Counter({track_key: playcount})``。

    track_key = ``"Artist ||| Track"``。
    """
    if not events_path.exists():
        raise FileNotFoundError(f"events file not found: {events_path}")

    per_user: Dict[str, Counter] = defaultdict(Counter)

    # 列：0=userid 1=ts 2=artid 3=artname 4=traid 5=traname
    reader = pd.read_csv(
        events_path,
        sep="\t",
        header=None,
        usecols=[0, 3, 5],
        names=["user_id", "artist_name", "track_name"],
        chunksize=chunksize,
        on_bad_lines="skip",
        dtype=str,
        engine="c",
        quoting=3,  # QUOTE_NONE
    )
    rows_seen = 0
    for ci, chunk in enumerate(reader):
        chunk = chunk[chunk["user_id"].isin(valid_user_ids)]
        if chunk.empty:
            continue
        chunk = chunk.dropna(subset=["artist_name", "track_name"])
        chunk = chunk[chunk["track_name"].astype(str).str.len() > 0]
        chunk = chunk[chunk["artist_name"].astype(str).str.len() > 0]
        if chunk.empty:
            continue
        for uid, an, tn in zip(
            chunk["user_id"].values,
            chunk["artist_name"].values,
            chunk["track_name"].values,
        ):
            key = f"{an}{TRACK_KEY_SEP}{tn}"
            per_user[str(uid)][key] += 1
        rows_seen += len(chunk)
        if (ci + 1) % 5 == 0:
            print(f"  ... chunk {ci+1}, valid rows so far: {rows_seen:,}")
    print(f"  events aggregation done. valid rows: {rows_seen:,}, "
          f"users with plays: {len(per_user):,}")
    return per_user


# ---------------------------------------------------------------------------
# 4. 阳性反馈过滤 + 计算流行度
#    (对应 ML_preprocess.py::calculate_popularity 的 Rating>3 过滤段)
# ---------------------------------------------------------------------------

def calculate_popularity(
    per_user: Dict[str, Counter],
    user_groups: Dict[str, Dict[str, Set[str]]],
    min_track_listeners: int = 1,
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, str]]:
    """返回 ``(popularity[item][group_col] = count, item_titles[item] = display_title)``。

    阳性反馈：``playcount >= 2 AND playcount > user_median_playcount``。
    与 ML_preprocess.py 的 ``Rating > 3`` 等价位置——筛掉低强度交互。
    """
    user_to_cols = user_to_group_keys(user_groups)

    # (a) 单曲 listener 计数（用于冷曲过滤）
    listener_count: Counter = Counter()
    for counter in per_user.values():
        listener_count.update(counter.keys())
    if min_track_listeners > 1:
        keep = {k for k, n in listener_count.items() if n >= min_track_listeners}
        print(f"  cold-track filter: keep {len(keep):,} / {len(listener_count):,} "
              f"tracks (min_listeners={min_track_listeners})")
    else:
        keep = None  # 不过滤

    # (b) 阳性反馈累加。
    popularity: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    item_titles: Dict[str, str] = {}

    for uid, counter in per_user.items():
        cols = user_to_cols.get(uid)
        if not cols:
            continue  # 用户性别/年龄全 unknown → 不归入任何分组
        if not counter:
            continue
        user_median = float(median(counter.values()))
        for track_key, pc in counter.items():
            if keep is not None and track_key not in keep:
                continue
            # ML_preprocess.py 的 Rating > 3 类比：高强度交互
            if pc < 2 or pc <= user_median:
                continue
            # 展示标题：Track - Artist
            if track_key not in item_titles:
                an, _, tn = track_key.partition(TRACK_KEY_SEP)
                item_titles[track_key] = f"{tn}{TRACK_TITLE_SEP}{an}"
            for col in cols:
                popularity[track_key][col] += 1

    print(f"  positive-feedback items: {len(popularity):,}")
    return popularity, item_titles


# ---------------------------------------------------------------------------
# 5. 归一化（与 ML_preprocess.py::normalize_popularity 完全一致）
# ---------------------------------------------------------------------------

def normalize_popularity(
    popularity: Dict[str, Dict[str, int]],
    group_columns: List[str],
) -> Dict[str, Dict[str, float]]:
    max_counts = {col: 0 for col in group_columns}
    for item, row in popularity.items():
        for col, c in row.items():
            if c > max_counts.get(col, 0):
                max_counts[col] = c

    norm: Dict[str, Dict[str, float]] = {}
    for item, row in popularity.items():
        norm[item] = {}
        for col in group_columns:
            mx = max_counts.get(col, 0)
            if mx > 0:
                norm[item][col] = row.get(col, 0) / mx
            else:
                norm[item][col] = 0.0
    return norm, max_counts


# ---------------------------------------------------------------------------
# 6. 装配 DataFrame（对应 add_popularity_to_items）
# ---------------------------------------------------------------------------

# 列顺序：与 ML_preprocess.py 一致；左两列改为 item_id/title，与
# src.popularity.item_scoring._META_COLS 兼容，便于 PopularityTable / lookup
# 直接复用。
GROUP_ORDER: List[str] = [
    "neutral_All",
    "gender_M", "gender_F", "gender_U",
    "age_Youth", "age_Middle", "age_Unknown",
    "cross_M_Youth", "cross_F_Youth", "cross_U_Youth",
    "cross_M_Middle", "cross_F_Middle", "cross_U_Middle",
    "cross_M_Unknown", "cross_F_Unknown", "cross_U_Unknown",
]


def assemble_dataframe(
    normalized: Dict[str, Dict[str, float]],
    item_titles: Dict[str, str],
) -> pd.DataFrame:
    rows = []
    for item_key in sorted(normalized.keys()):
        row = {
            "item_id": item_key,
            "title": item_titles.get(item_key, item_key),
        }
        for col in GROUP_ORDER:
            row[col] = float(normalized[item_key].get(col, 0.0))
        rows.append(row)
    return pd.DataFrame(rows, columns=["item_id", "title"] + GROUP_ORDER)


# ---------------------------------------------------------------------------
# 7. main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        default="data/raw/lastfm-1k/lastfm-dataset-1K",
        help="包含 userid-profile.tsv 与 events 文件的目录",
    )
    parser.add_argument(
        "--output",
        default="data/processed/lastfm1k_track/items_with_popularity.csv",
    )
    parser.add_argument(
        "--min-track-listeners",
        type=int,
        default=3,
        help="冷曲过滤：只保留被至少 N 个不同用户听过的单曲（含未阳性反馈的也算 listener）",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=EVENTS_CHUNK_SIZE,
        help="事件文件流式读取的每块行数，内存吃紧调小",
    )
    args = parser.parse_args()

    raw = Path(args.raw_dir)
    profile_path = raw / PROFILE_FILE
    events_path = raw / EVENTS_FILE
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== LFM-1K 单曲流行度预处理 ===")
    print(f"raw_dir   = {raw}")
    print(f"output    = {output_path}")

    print("\n[1/6] 加载用户画像...")
    users = load_lfm1k_users(profile_path)
    print(f"  users in profile: {len(users):,}")

    print("\n[2/6] 创建用户分组...")
    user_groups, users = create_user_groups(users)
    for g, sub in user_groups.items():
        for s, m in sub.items():
            print(f"  {g}_{s}: {len(m):,} users")

    print("\n[3/6] 流式聚合 (用户, 单曲) 播放次数 (耗时 5-15 分钟)...")
    valid_users = set(users["UserID"])
    per_user = aggregate_user_track_plays(
        events_path, valid_users, chunksize=args.chunksize,
    )

    print("\n[4/6] 计算阳性反馈分组流行度...")
    popularity, item_titles = calculate_popularity(
        per_user, user_groups, min_track_listeners=args.min_track_listeners,
    )

    print("\n[5/6] 归一化 (raw / column_max)...")
    normalized, max_counts = normalize_popularity(popularity, GROUP_ORDER)
    for col in GROUP_ORDER:
        print(f"  max[{col}] = {max_counts.get(col, 0)}")

    print("\n[6/6] 写出 CSV...")
    df = assemble_dataframe(normalized, item_titles)
    df.to_csv(output_path, index=False)
    print(f"  rows  = {len(df):,}")
    print(f"  cols  = {list(df.columns)}")
    print(f"  saved → {output_path}")

    # 头几行抽样
    print("\n抽样前 5 行:")
    print(df[["item_id", "title", "neutral_All", "gender_M", "gender_F"]].head().to_string(index=False))

    # 统计
    print("\n各分组流行度统计 (非零):")
    for col in GROUP_ORDER:
        nz = df[col][df[col] > 0]
        if len(nz) == 0:
            print(f"  {col}: 全 0")
            continue
        print(f"  {col}: nonzero={len(nz):,}, mean={nz.mean():.4f}, "
              f"max={nz.max():.4f}, min={nz.min():.4f}")


if __name__ == "__main__":
    sys.exit(main())
