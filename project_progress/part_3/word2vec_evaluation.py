import numpy as np
import pandas as pd

# our evaluation metrics of part 2
from project_progress.part_2.evaluation import evaluate_query, mean_average_precision, mean_reciprocal_rank
from word2vec_ranking import search_word2vec


def load_labelled_queries(queries, labels_path):
    """
    Returns a list of (query_text, labels_of_that_query) for the queries of one csv.
    The query_id of the csv is the position of the query in the list (starting at 1).
    """
   
    all_labels = pd.read_csv(labels_path)
    labelled = []
    for q_idx in range(len(queries)):
        labels = all_labels[all_labels['query_id'] == q_idx + 1]
        labelled.append((queries[q_idx], labels))
    return labelled


def judged_relevance(ranked_results, labels):
    """
    Same protocol as in part 2: we keep only the products that have a label,
    in the order given by the ranking, and return their labels.
    """
    rel_map = dict(zip(labels['pid'], labels['labels']))
    relevance = []
    for score, pid in ranked_results:
        if pid in rel_map:
            relevance.append(rel_map[pid])
    total_relevant = int((labels['labels'] == 1).sum())
    return relevance, total_relevant


def metrics_per_query(labelled_queries, preprocessor, index, model, doc_vectors, k=10):
    """
    Metrics of every query (with our evaluate_query of part 2).
    Returns a list of (query_text, metrics).
    """
    results = []
    for query, labels in labelled_queries:
        # we rank all the AND results (not only the top 20) so every judged product gets a position
        ranked = search_word2vec(query, preprocessor, index, model, doc_vectors, top_k=len(doc_vectors))
        relevance, total_relevant = judged_relevance(ranked, labels)
        results.append((query, evaluate_query(relevance, total_relevant, k=k)))
    return results


def evaluate_word2vec(labelled_queries, preprocessor, index, model, doc_vectors, k=10):
    """
    Mean of every metric over the queries (for MAP and MRR we use our functions of evaluation.py).
    """
    per_query = metrics_per_query(labelled_queries, preprocessor, index, model, doc_vectors, k=k)

    result = {}
    for name in ['P@K', 'R@K', 'F1@K', 'NDCG@K']:
        values = []
        for query, m in per_query:
            values.append(m[name])
        result[name] = round(float(np.mean(values)), 3)

    ap_list = []
    rr_list = []
    for query, m in per_query:
        ap_list.append(m['AP@K'])
        rr_list.append(m['RR@K'])
    result['MAP'] = round(mean_average_precision(ap_list), 3)
    result['MRR'] = round(mean_reciprocal_rank(rr_list), 3)
    return result


def best_combination(results):
    """
    Sorts the results of the grid search by NDCG@K (and MAP if there is a tie)
    and returns the table sorted and the best combination.
    """
    table = pd.DataFrame(results)
    table = table.sort_values(['NDCG@K', 'MAP'], ascending=False).reset_index(drop=True)
    best = table.iloc[0].to_dict()
    return table, best