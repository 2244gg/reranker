# Datasets

Raw data files are NOT committed (per `.gitignore`). Each dataset must be
obtained from its official source and placed at the path indicated below.

After downloading, run `python scripts/download_datasets.py --dataset <name>`
to verify expected paths exist.

---

## ML-1M

- **Path**: `ml-1m/ml-1m/{users.dat,movies.dat,ratings.dat,README}` (already shipped)
- **Source**: <https://files.grouplens.org/datasets/movielens/ml-1m.zip>
- **License**: Free for research use under GroupLens terms
  (<https://files.grouplens.org/datasets/movielens/ml-1m-README.txt>)
- **Citation**:
  > F. Maxwell Harper and Joseph A. Konstan. 2015.
  > The MovieLens Datasets: History and Context.
  > ACM Transactions on Interactive Intelligent Systems 5, 4 (2015), 1-19.
- **Demographics exposed**: gender, age_group, occupation_group (3 dims)

## BookCrossing

- **Path**: `data/raw/bookcrossing/{BX-Users.csv,BX-Books.csv,BX-Book-Ratings.csv}`
- **Source**: <http://www2.informatik.uni-freiburg.de/~cziegler/BX/>
  (download `BX-CSV-Dump.zip`)
- **License**: Free for research use, contact author for commercial use.
- **Citation**:
  > Cai-Nicolas Ziegler, Sean M. McNee, Joseph A. Konstan, Georg Lausen. 2005.
  > Improving Recommendation Lists Through Topic Diversification.
  > In Proceedings of the 14th International World Wide Web Conference (WWW '05).
- **Demographics exposed**: age_group, country (2 dims)
- **Encoding**: Latin-1, `;` separator. Some rows include unescaped quotes;
  the adapter calls `pandas.read_csv(... on_bad_lines="skip")`.

## LFM-2b

- **Path**: `data/raw/lfm-2b/{users.tsv,user_artist_playcount.tsv,artists.tsv}`
- **Source**: <http://www.cp.jku.at/datasets/LFM-2b/>
  (apply for access; research-only EULA)
- **License**: Research-only with attribution requirement.
- **Citation**:
  > Markus Schedl, Stefan Brost, Cynthia C. S. Liem, Peter Knees, Elias Pampalk,
  > Vito Walter Anelli, Tommaso Di Noia, Elisabeth Lex. 2022.
  > LFM-2b: A Dataset of Enriched Music Listening Events for Recommender
  > Systems Research and Fairness Analysis. CHIIR '22.
- **Demographics exposed**: gender, age_group, country (3 dims)

## Tenrec

- **Path**: `data/raw/tenrec/{user_features.csv,video_impressions.csv,items.csv?}`
- **Source**: <https://github.com/yuangh-x/2022-NIPS-Tenrec>
- **License**: Apache 2.0 (per the repository) for the released splits.
- **Citation**:
  > Guanghu Yuan, Fajie Yuan, Yudong Li, Beibei Kong, Shujie Li, Lei Chen,
  > Min Yang, Chenyun Yu, Bo Hu, Zang Li, Yu Xu, Xiaohu Qie. 2022.
  > Tenrec: A Large-scale Multipurpose Benchmark Dataset for Recommender Systems.
  > NeurIPS 2022 Datasets and Benchmarks Track.
- **Demographics exposed**: gender, age_group (2 dims, Chinese-domain validation)

---

## Demographic Schema Summary

| Dataset       | Domain | Demographics                              | Dims |
|---------------|--------|-------------------------------------------|------|
| ML-1M         | movie  | gender, age_group, occupation_group       | 3    |
| BookCrossing  | book   | age_group, country                        | 2    |
| LFM-2b        | music  | gender, age_group, country                | 3    |
| Tenrec        | video  | gender, age_group                         | 2    |

Each dataset satisfies the >= 2-dimension requirement for intersectional
bias analysis. ML-1M and LFM-2b additionally enable 3-way intersectional
ablations (`gender x age x occupation` and `gender x age x country`).

## Reproducibility

- All preprocessing seeds default to `42` and are overridable via experiment configs.
- Top-K country lists are computed deterministically from the user table at
  load time; no remote calls, no time-dependent state.
- Adapter outputs unify into `UserRecord` (`src/datasets/base.py`); downstream
  code never reads raw files directly.
