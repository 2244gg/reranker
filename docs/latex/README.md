# LaTeX 论文源码——使用说明

ACM SIG 投稿格式（acmart `sigconf` 文档类）。适用于 RecSys / SIGIR / WWW / KDD / CIKM。

## 文件结构

```
docs/latex/
├── paper.tex                # 主文件，含文档类、宏包、metadata、\input 调用
├── paper.bib                # BibTeX 文献库（21 篇）
├── sec_intro.tex            # 1. Introduction
├── sec_related.tex          # 2. Related Work
├── sec_problem.tex          # 3. Problem Formulation
├── sec_method.tex           # 4. Methodology
├── sec_setup.tex            # 5. Experimental Setup
├── sec_results_ml1m.tex     # 6. Results on MovieLens-1M
├── sec_results_lfm1k.tex    # 7. Results on LastFM-1K
├── sec_ablation.tex         # 8. Ablation
├── sec_discussion.tex       # 9. Discussion
└── sec_conclusion.tex       # 10. Conclusion + Acknowledgments
```

主文件 `paper.tex` 通过 `\input{}` 调用各章节，方便分工修改与版本管理。

## 路径 1：本地编译（最快）

需要 LaTeX 发行版 + acmart 模板。Windows 推荐 **MiKTeX**。

```bat
:: 1. 安装 MiKTeX (https://miktex.org/download) 一次性
::    勾选 "automatic package install" 自动拉取 acmart

:: 2. 打开命令行进入目录
cd e:\Reranker\reranker\docs\latex

:: 3. 第一次编译（4 步：tex → bib → tex → tex 解决交叉引用）
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

编译产物：`paper.pdf`（投稿用）。

之后只改章节内容只需 `pdflatex paper.tex` 一次即可。

## 路径 2：Overleaf 在线编译（推荐给导师协作）

Overleaf 是在线 LaTeX 协作平台，**无需本地装任何东西**。

```
1. 注册 https://www.overleaf.com  免费账号
2. New Project → Upload Project
3. 把整个 docs/latex/ 文件夹打成 zip 上传
4. 主文件选 paper.tex
5. 自动编译，PDF 实时预览
6. 分享链接给导师，导师可直接在线编辑评论
```

ACM 模板已被 Overleaf 内置支持，无需手动装包。

## 路径 3：把 ACM 官方 acmart.cls 直接放入项目

如果机器上没装 acmart：

```
https://github.com/borisveytsman/acmart  下载 acmart.cls
放到 docs/latex/ 同目录
```

之后 `\documentclass[sigconf,review,anonymous]{acmart}` 就能找到。

## 当前状态

| 项 | 状态 |
|---|---|
| 文档类 | acmart `sigconf,review,anonymous`（双盲审版） |
| 章节 | 10 章全部就位 |
| 公式 | 16+ 个数学公式（与 `docs/formulas_reference.md` 对齐） |
| 表格 | 7 张表（数据、超参、ML-1M 主结果、LFM-1K 主结果、LFM-1K 每用户覆盖、消融 B、消融 K'） |
| 参考文献 | 21 篇（Dwork、Abdollahpouri、Bao TALLRec、Hua Up5、Hu SIGIR2024 等） |
| 占位符 | §6.4 ML-1M 跨切片差异、§7.5 LFM-1K 统计歧视差异、§8.3 dynamic-α 消融、§8.4 Pareto 散点 |

## 投稿前需要做的事

1. **替换 anonymous → 真实作者信息**（`paper.tex` 中 `\author`、`\affiliation`、`\email`）
2. **删除 `review,anonymous` 选项**（投终稿）：
   ```latex
   \documentclass[sigconf]{acmart}
   ```
3. **替换 dummy ACM metadata**（DOI、ISBN、conference name）
4. **填实验占位符**（参见上表"占位符"行）
5. **加图**：Pareto 散点 + 柱状图，建议用 matplotlib 生成 PNG，`\includegraphics{}` 引入
6. **拼写检查**: `chktex paper.tex` 或 Overleaf 内置 grammar 工具

## 投稿目标会议（按方向匹配度排序）

| 会议 | 截稿月 | 匹配度 |
|---|---|---|
| **RecSys** (ACM Conference on Recommender Systems) | 5 月 | 🟢 完美对口 |
| **SIGIR** | 1 月 | 🟢 完美对口 |
| **WWW (TheWebConf)** | 10 月 | 🟢 完美对口 |
| **CIKM** | 5 月 | 🟢 对口 |
| **WSDM** | 8 月 | 🟢 对口 |
| **KDD** | 2 月 | 🟡 需要更强机器学习 framing |
| **AAAI** | 8 月 | 🟡 fairness 角度可投 |

短论文（4-page short paper）也都在这些会议里有 track，门槛比 full paper 低，可作为先期 publication。

## 中文版本

中文学位论文一般用学校自己的 LaTeX 模板，把 paper 内容塞到对应章节即可。或用 pandoc：

```bat
pandoc docs\paper_draft.md -o paper_zh_cn.docx --reference-doc=学校模板.docx
```
