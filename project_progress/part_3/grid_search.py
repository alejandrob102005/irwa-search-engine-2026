from sklearn.model_selection import ParameterGrid

from word2vec_ranking import train_word2vec, build_doc_vectors
from word2vec_evaluation import evaluate_word2vec



def grid_search(docs, text_fields, param_grid, labelled_queries, preprocessor, index,
                min_count=2, epochs=10, seed=42, k=10):
    results = []
    for params in ParameterGrid(param_grid):
        model = train_word2vec(docs, text_fields,
                               vector_size=params['vector_size'],
                               window=params['window'],
                               sg=params['sg'],
                               min_count=min_count, epochs=epochs, seed=seed)
        doc_vectors = build_doc_vectors(docs, text_fields, model)

        metrics = evaluate_word2vec(labelled_queries, preprocessor, index, model, doc_vectors, k=k)

        row = dict(params)
        row.update(metrics)
        results.append(row)
        print(row)

    return results
