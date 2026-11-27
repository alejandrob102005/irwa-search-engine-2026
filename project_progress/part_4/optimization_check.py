
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from project_progress.part_3.indexing_p3 import conjunctive_search, rank_documents_bm25, rank_documents_custom
from project_progress.part_4.web_search import (WebSearchEngine, rank_documents_bm25_fast,
                                                rank_documents_custom_fast)
from myapp.search.load_corpus import load_corpus

QUERIES = ['women cotton printed kurta', 'men black leather jacket', 'men tshirt', 'women']


def timed(function, *args):
    start = time.perf_counter()
    result = function(*args)
    return result, (time.perf_counter() - start) * 1000


def main():
    engine = WebSearchEngine()
    e = engine

    print('\nQuery | matching docs | method | Part 3 (ms) | Part 4 (ms) | same ranking')
    for query in QUERIES:
        terms = e.preprocessor.preprocess_text(query)
        docs = conjunctive_search(terms, e.index)

        old, t_old = timed(rank_documents_bm25, terms, docs, e.index, e.tf_raw, e.df, e.N, e.doc_lengths, e.avgdl)
        new, t_new = timed(rank_documents_bm25_fast, terms, docs, e.tf_by_doc, e.df, e.N, e.doc_lengths, e.avgdl)
        same = [pid for _, pid in old] == [pid for _, pid in new]
        print(f'{query} | {len(docs)} | BM25 | {t_old:.1f} | {t_new:.1f} | {same}')

        old, t_old = timed(rank_documents_custom, terms, docs, e.index, e.tf_raw, e.df, e.N, e.doc_lengths,
                           e.avgdl, e.product_id_info)
        new, t_new = timed(rank_documents_custom_fast, terms, docs, e.tf_by_doc, e.df, e.N, e.doc_lengths,
                           e.avgdl, e.product_id_info)
        same = [pid for _, pid in old] == [pid for _, pid in new]
        print(f'{query} | {len(docs)} | Custom | {t_old:.1f} | {t_new:.1f} | {same}')

    # AND -> OR fallback, with the same corpus as the web app
    corpus = load_corpus(ROOT_DIR / 'data' / 'fashion_products_dataset.json')
    query = 'men leather saree'
    output = engine.search(query, corpus, top_k=5)
    print(f'\nFallback example: "{query}" -> mode={output.match_mode}, '
          f'found={output.total_found}, unknown terms={output.unknown_terms}')
    for r in output.results:
        print(f'  {r.rank}. {r.score:.3f} {r.pid} {r.title}')


if __name__ == '__main__':
    main()
