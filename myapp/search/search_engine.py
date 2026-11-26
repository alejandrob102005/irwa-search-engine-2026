import random
import numpy as np

from myapp.search.objects import Document
from project_progress.part_4.web_search import WebSearchEngine


def dummy_search(corpus: dict, search_id, num_results=20):
    """
    Just a demo method, that returns random <num_results> documents from the corpus
    :param corpus: the documents corpus
    :param search_id: the search id
    :param num_results: number of documents to return
    :return: a list of random documents from the corpus
    """
    res = []
    doc_ids = list(corpus.keys())
    docs_to_return = np.random.choice(doc_ids, size=num_results, replace=False)
    for doc_id in docs_to_return:
        doc = corpus[doc_id]
        res.append(Document(pid=doc.pid, title=doc.title, description=doc.description,
                            url="doc_details?pid={}&search_id={}&param2=2".format(doc.pid, search_id), ranking=random.random()))
    return res


class SearchEngine:
    """Class that implements the search engine logic (Part 4: project_progress/part_4/web_search.py)"""

    def __init__(self):
        # the index of Part 3 is built only once, when the web app starts
        self.engine = WebSearchEngine()

    def search(self, search_query, search_id, corpus, method='custom', top_k=20, only_available=False):
        """
        :return: SearchOutput with the ranked results (list of SearchResult) and information
                 about the search (number of products found, AND/OR mode, time...)
        """
        print("Search query:", search_query)
        return self.engine.search(search_query, corpus, search_id=search_id, method=method,
                                  top_k=top_k, only_available=only_available)
