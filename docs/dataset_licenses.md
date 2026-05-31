# Dataset Licenses & Citations

This project does **not** redistribute raw data. Each dataset is governed by
its original license; users must download data themselves and comply with
the terms below. ``data/README.md`` tracks the on-disk layout; this file is
the authoritative license + citation reference.

---

## ML-1M (in-repo)

- **License**: Free for research use under the GroupLens / MovieLens terms
  (`ml-1m/ml-1m/README` shipped with the dataset).
- **Source**: <https://files.grouplens.org/datasets/movielens/ml-1m.zip>
- **Citation**:

  ```bibtex
  @article{Harper2015MovieLens,
    author = {F. Maxwell Harper and Joseph A. Konstan},
    title = {The MovieLens Datasets: History and Context},
    journal = {ACM Transactions on Interactive Intelligent Systems},
    volume = {5},
    number = {4},
    pages = {1--19},
    year = {2015}
  }
  ```

## BookCrossing

- **License**: Free for research use, contact author for commercial use
  (Cai-Nicolas Ziegler).
- **Source (current)**: archived at the Wayback Machine
  ``https://web.archive.org/web/2020*/http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip``
- **Citation**:

  ```bibtex
  @inproceedings{Ziegler2005BookCrossing,
    author = {Cai-Nicolas Ziegler and Sean M. McNee and Joseph A. Konstan and
              Georg Lausen},
    title = {Improving Recommendation Lists Through Topic Diversification},
    booktitle = {WWW},
    year = {2005}
  }
  ```

## LFM-2b

- **License**: Research-only EULA. Apply for access at the JKU Linz
  Computational Perception group page; data may not be redistributed.
- **Source**: <http://www.cp.jku.at/datasets/LFM-2b/>
- **Citation**:

  ```bibtex
  @inproceedings{Schedl2022LFM2b,
    author = {Markus Schedl and Stefan Brost and Cynthia C. S. Liem and
              Peter Knees and Elias Pampalk and Vito Walter Anelli and
              Tommaso Di Noia and Elisabeth Lex},
    title = {LFM-2b: A Dataset of Enriched Music Listening Events for
             Recommender Systems Research and Fairness Analysis},
    booktitle = {CHIIR},
    year = {2022}
  }
  ```

## Tenrec

- **License**: CC BY-NC 4.0 International (per the official repository).
- **Source**:
  - GitHub: <https://github.com/yuangh-x/2022-NIPS-Tenrec>
  - Dataset landing page (registration required):
    <https://static.qblv.qq.com/qblv/h5/algo-frontend/tenrec_dataset.html>
- **Citation**:

  ```bibtex
  @inproceedings{Yuan2022Tenrec,
    author = {Guanghu Yuan and Fajie Yuan and Yudong Li and Beibei Kong and
              Shujie Li and Lei Chen and Min Yang and Chenyun Yu and Bo Hu and
              Zang Li and Yu Xu and Xiaohu Qie},
    title = {Tenrec: A Large-scale Multipurpose Benchmark Dataset for
             Recommender Systems},
    booktitle = {NeurIPS Datasets and Benchmarks Track},
    year = {2022}
  }
  ```

---

## Compliance checklist

If you use this framework in a paper or product:

- [ ] Cite each dataset whose results you report.
- [ ] Cite this repository if it accelerated your research.
- [ ] Honor the redistribution terms of LFM-2b (research-only, no
      redistribution) and Tenrec (CC BY-NC, non-commercial).
- [ ] Do not commit raw data into derivative repositories; ``.gitignore``
      ships with this project to prevent accidental commits.
