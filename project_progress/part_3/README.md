# Part 3: Ranking and Filtering

## Contents
- `indexing_p3.py`: builds the inverted index with the statistics needed for ranking, performs conjunctive (AND) search, and ranks results with TF-IDF + cosine, BM25 (k1 = 1.5, b = 0.75), and a custom score (BM25 + rating + discount).
- `word2vec_ranking.py`: trains a Word2Vec model, represents each text as the average of its word vectors, and ranks the AND search results by cosine similarity.
- `word2vec_evaluation.py`: evaluates the Word2Vec ranking on labelled queries with the metrics from Part 2.
- `grid_search.py`: selects the Word2Vec parameters (`vector_size`, `window`, `sg`) on the validation queries.
- `main.py`: runs Part 3.1 (the three ranking methods) and Part 3.2 (grid search, evaluation, and top 20 Word2Vec results) for the test queries.
- `../../config.yaml`: defines the data paths, indexed fields, queries, top K, and Word2Vec parameters.


## Reproduce the results
Python version used for verification: **3.13**.

Install the project dependencies from the repository root if they are not installed yet:

```bash
pip install -r requirements.txt
```

The processed dataset `data/processed_fashion_products.json` (generated in Part 1) and the label files `data/validation_labels.csv` and `data/validation2_labels.csv` must be in the `data/` directory.

From the repository root, run the Part 3 pipeline:

```bash
python project_progress/part_3/main.py
```
