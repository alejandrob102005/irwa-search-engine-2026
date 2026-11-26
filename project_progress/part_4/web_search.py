"""
4.1 - Search engine used by the web application adapting previous algorithms 
"""
import json
import math
import sys
import time
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from project_progress.part_1.preprocess import TextPreprocessor, load_and_preprocess_dataset
from project_progress.part_3.indexing_p3 import build_p3_index, conjunctive_search, rank_documents_tfidf


# ranking methods that the user can choose in the web page
RANKING_METHODS = {
    'custom': 'Custom (BM25 + rating + discount)',
    'bm25': 'BM25',
    'tfidf': 'TF-IDF + cosine',
}


class SearchResult:
    """One product of the results page."""

    def __init__(self, rank, score, doc, details_url):
        self.rank = rank                # position in the ranking (1, 2, 3...)
        self.score = score              # score given by the ranking method
        self.doc = doc                  # Document of the corpus (title, prices, rating, url...)
        self.details_url = details_url  # link to document details page
        # the RAG generator reads .pid and .title of each result
        self.pid = doc.pid
        self.title = doc.title


class SearchOutput:
    """Everything the results page needs to know about one search."""

    def __init__(self, query, method):
        self.query = query
        self.method = method
        self.results = []          # top_k SearchResult objects
        self.total_found = 0       # matching products before top_k
        self.match_mode = 'and'    # 'and' or 'or' (fallback)
        self.unknown_terms = []    # query terms not in any product
        self.elapsed_ms = 0.0

#Search algorithms:

def rank_documents_bm25_fast(query_terms, docs, tf_by_doc, df, N, doc_lengths, avgdl, k1=1.5, b=0.75):
    """
    Same BM25 as rank_documents_bm25 but tf_by_doc[term][pid] gives tf directly.
    In Part 3 we used index[term].index(pid), slower.
    """
    doc_scores = []
    for product_id in docs:
        score = 0.0
        for term in query_terms:
            f_qi_d = tf_by_doc.get(term, {}).get(product_id)
            if not f_qi_d:
                continue
            doc_len = doc_lengths.get(product_id, avgdl)
            idf_val = math.log2(N / df[term])
            numerator = (k1 + 1) * f_qi_d
            denominator = k1 * ((1 - b) + b * (doc_len / avgdl)) + f_qi_d
            score += idf_val * (numerator / denominator)
        doc_scores.append([score, product_id])

    doc_scores.sort(key=lambda x: x[0], reverse=True)
    return doc_scores


def rank_documents_custom_fast(query_terms, docs, tf_by_doc, df, N, doc_lengths, avgdl, product_id_info, alpha=1.0, beta=0.5):
    """ Same custom score as Part 3 with fast bm25"""
    bm25_scores = rank_documents_bm25_fast(query_terms, docs, tf_by_doc, df, N, doc_lengths, avgdl)

    custom_scores = []
    #BM25 + alpha * rating/5 + beta * discount/100
    for score_bm25, product_id in bm25_scores:
        doc_info = product_id_info.get(product_id, {})
        rating = doc_info.get('average_rating_value') or 0.0
        discount = doc_info.get('discount_percent') or 0.0
        norm_rating = rating / 5.0 if rating <= 5.0 else 1.0
        norm_discount = discount / 100.0 if discount <= 100.0 else 1.0
        custom_scores.append([score_bm25 + alpha * norm_rating + beta * norm_discount, product_id])

    custom_scores.sort(key=lambda x: x[0], reverse=True)
    return custom_scores


class WebSearchEngine:
    """Search engine of the web application."""

    def __init__(self, config_path=ROOT_DIR / 'config.yaml'):
        with open(config_path, encoding='utf-8') as f:
            config = yaml.safe_load(f)

        data_dir = ROOT_DIR / config['paths']['data_dir']
        processed_path = data_dir / config['paths']['processed_dataset']
        text_fields = config['indexing']['text_fields']

        self.preprocessor = TextPreprocessor()

        if processed_path.exists():
            with open(processed_path, encoding='utf-8') as f:
                docs = json.load(f)
        else:
            print('Processed dataset not found, preprocessing the raw dataset (Part 1)')
            docs = load_and_preprocess_dataset(data_dir / config['paths']['raw_dataset'], processed_path,
                                               config['preprocessing']['output_fields'])

        (self.index, self.tf, self.tf_raw, self.df, self.idf, 
        self.doc_lengths, self.avgdl, self.product_id_info, self.N) = build_p3_index(docs, text_fields)

        # extra for BM25 (built once at start)
        self.tf_by_doc = {}
        for term in self.index:
            self.tf_by_doc[term] = dict(zip(self.index[term], self.tf_raw[term]))

        print(f'Part 4 search engine ready: {self.N} documents, {len(self.index)} terms')

    #search function
    def search(self, query, corpus, search_id=None, method='custom', top_k=20, only_available=False):
        """ General search function of the engine."""
        start = time.perf_counter()
        if method not in RANKING_METHODS:
            method = 'custom'
        output = SearchOutput(query=query, method=method)

        query_terms = self.preprocessor.preprocess_text(query)
        output.unknown_terms = [t for t in dict.fromkeys(query_terms) if t not in self.index]

        #AND search
        matching_docs = conjunctive_search(query_terms, self.index)

        # if no product has all the terms -> OR
        if not matching_docs:
            known_terms = [t for t in query_terms if t in self.index]
            query_terms = known_terms
            matching_docs = set()
            for term in known_terms:
                matching_docs.update(self.index[term])
            if matching_docs:
                output.match_mode = 'or'

        # optional filter
        if only_available:
            matching_docs = {pid for pid in matching_docs if self.product_id_info[pid].get('is_available', True)}

        # ranking
        if method == 'tfidf':
            ranked = rank_documents_tfidf(query_terms, matching_docs, self.index, self.tf, self.idf)
        elif method == 'bm25':
            ranked = rank_documents_bm25_fast(query_terms, matching_docs, self.tf_by_doc, self.df, self.N,
                                              self.doc_lengths, self.avgdl)
        else:
            ranked = rank_documents_custom_fast(query_terms, matching_docs, self.tf_by_doc, self.df, self.N,
                                                self.doc_lengths, self.avgdl, self.product_id_info)

        # top_k results 
        output.total_found = len(ranked)
        for position, (score, pid) in enumerate(ranked[:top_k], start=1):
            if pid not in corpus:
                continue
            details_url = f'/doc_details?pid={pid}&search_id={search_id}&rank={position}'
            output.results.append(SearchResult(rank=position, score=float(score), doc=corpus[pid],
                                               details_url=details_url))

        output.elapsed_ms = (time.perf_counter() - start) * 1000
        return output
