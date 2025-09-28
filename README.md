# Article → FOS Topic Linking (CiNii)

**Scope.** This repository **evaluates retrieval methods** (semantic kNN vs. MinHash/LSH) for linking CiNii articles to **OECD FOS 2007** topics, and **proposes** a simple pipeline design. It does **not** ship a production linker.

* **Semantic** kNN over sentence embeddings (FAISS **HNSW**)
* **Lexical** set similarity with **MinHash + LSH/LSH Forest**
* Optional **LLM baseline** (GPT-5-mini) for direct classification

## Highlights

* CiNii-based English benchmark: `data/cinii_topics_benchmark/article_labels.csv`
* Topic scheme (mini-ontology): **OECD FOS 2007** (`data/cinii_topics_benchmark/oecd_fos_2007.ttl`)
* Methods: `all-MiniLM-L6-v2` + FAISS HNSW; MinHash (num_perm=128) + LSH Forest
* Metrics: IR (P@k, R@k, AP@k, nDCG@k) and classification (Accuracy, Macro P/R/F1)
* **Proposed** hybrid pipeline (design only): LSH for candidate recall → HNSW for semantic re-ranking → optional LLM triage

For full methodology, dataset details, and results, see the **report**. This repo provides code and minimal steps to reproduce the experiments.

---

## Setup (with `uv`)

Requires Python ≥ 3.11 (project used 3.13), Git, and `uv`.

```bash
git clone <this-repo>
cd <this-repo>
uv sync   # creates .venv and installs all dependencies from pyproject/lock
```

Run scripts with `uv run` (or use the notebook `main.ipynb` for the experiments).

---

## Labeling TUI (optional)

A small TUI assists manual labeling:

```bash
uv run src/create_labels_cli.py
```
<img width="1627" height="648" alt="image" src="https://github.com/user-attachments/assets/19622c51-be0e-47e9-a6cd-23cd7c13f317" />

Controls:

* `h` / `m` / `l` → set confidence (high/medium/low)
* Arrow keys or `j`/`k` → navigate
* `Enter` → confirm label
* `?` → help pop-up with label description
