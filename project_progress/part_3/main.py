import sys
import json
from pathlib import Path
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from project_progress.part_1.preprocess import TextPreprocessor
from project_progress.part_2.indexing import build_inverted_index
from word2vec_ranking import train_word2vec, build_doc_vectors, search_word2vec
from word2vec_evaluation import load_labelled_queries, best_combination, metrics_per_query, evaluate_word2vec
from grid_search import grid_search
from indexing_p3 import (
    build_p3_index, 
    conjunctive_search, 
    rank_documents_tfidf, 
    rank_documents_bm25, 
    rank_documents_custom
)

def main():
    with open(ROOT_DIR / 'config.yaml', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    data_dir = ROOT_DIR / config['paths']['data_dir']
    text_fields = config['indexing']['text_fields']
    top_k = config['search']['top_k']
    w2v = config['word2vec']

    with open(data_dir / config['paths']['processed_dataset'], encoding='utf-8') as f:
        docs = json.load(f)

    preprocessor = TextPreprocessor()

    
    
    #3.1
    print("\n PART 3.1 \n")
    index_p3, tf, tf_raw, df, idf, doc_lengths, avgdl, product_id_info_p3, N = build_p3_index(docs, text_fields)
    
    print('Number of documents:', len(docs))
    print('Number of terms in the index:', len(index_p3))

    p3_test_queries = config['search']['test_queries']
    for query in p3_test_queries:
        query_terms = preprocessor.preprocess_text(query)
        matching_docs = conjunctive_search(query_terms, index_p3)
        
        print(f'\nQuery: "{query}"')
        print(f'Documents containing all the terms (AND): {len(matching_docs)}')

        # TF-IDF + Cosine Ranking
        ranked_tfidf = rank_documents_tfidf(query_terms, matching_docs, index_p3, tf, idf)
        print("Top TF-IDF Results:")
        for position in range(min(top_k, len(ranked_tfidf))):
            score, pid = ranked_tfidf[position]
            title = product_id_info_p3[pid]['title']
            print(f"     {position + 1}. Score: {score:.3f} | ID: {pid} | {title}")

        # BM25 Ranking
        ranked_bm25 = rank_documents_bm25(query_terms, matching_docs, index_p3, tf_raw, df, N, doc_lengths, avgdl)
        print("Top BM25 Results:")
        for position in range(min(top_k, len(ranked_bm25))):
            score, pid = ranked_bm25[position]
            title = product_id_info_p3[pid]['title']
            print(f"     {position + 1}. Score: {score:.3f} | ID: {pid} | {title}")

        # 3. Custom Score Ranking
        ranked_custom = rank_documents_custom(query_terms, matching_docs, index_p3, tf_raw, df, N, doc_lengths, avgdl, product_id_info_p3)
        print("Top Custom Score Results:")
        for position in range(min(top_k, len(ranked_custom))):
            score, pid = ranked_custom[position]
            title = product_id_info_p3[pid]['title']
            print(f"     {position + 1}. Score: {score:.3f} | ID: {pid} | {title}")
    
    
    
    #3.2
    print("\n PART 3.2 \n")
    index, product_id_info = build_inverted_index(docs, text_fields)

    
    validation_queries = load_labelled_queries(config['evaluation']['validation_queries'],
                                               data_dir / config['paths']['validation_labels'])
    
    test_queries = load_labelled_queries(config['search']['test_queries'],
                                         data_dir / config['paths']['validation2_labels'])

    
    results = grid_search(docs, text_fields, w2v['param_grid'], validation_queries, preprocessor, index,
                          min_count=w2v['min_count'], epochs=w2v['epochs'], seed=w2v['seed'])
    table, best = best_combination(results)
   
    print('Best combination:', best)

    #best model, evaluated with our test queries 
    model = train_word2vec(docs, text_fields, vector_size=int(best['vector_size']),
                           window=int(best['window']), sg=int(best['sg']),
                           min_count=w2v['min_count'], epochs=w2v['epochs'], seed=w2v['seed'])
    doc_vectors = build_doc_vectors(docs, text_fields, model)

    print('\nEvaluation of the best model with our test queries')
    for query, metrics in metrics_per_query(test_queries, preprocessor, index, model, doc_vectors):
        print(query, metrics)
    print('Mean:', evaluate_word2vec(test_queries, preprocessor, index, model, doc_vectors))

    # top 20 of our test queries 
    for query in config['search']['test_queries']:
        ranked = search_word2vec(query, preprocessor, index, model, doc_vectors, top_k=w2v['top_k'])
        print('\nQuery:', query)
        for position in range(len(ranked)):
            score = ranked[position][0]
            pid = ranked[position][1]
            print(position + 1, round(score, 3), pid, product_id_info[pid]['title'])


if __name__ == '__main__':
    main()