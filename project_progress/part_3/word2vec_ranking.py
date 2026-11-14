import numpy as np
from gensim.models import Word2Vec

from project_progress.part_2.indexing import get_doc_terms, conjunctive_search


def train_word2vec(docs, text_fields, vector_size=100, window=5, sg=1, min_count=2, epochs=10, seed=42):
    
    sentences = []
    for doc in docs:
        terms = get_doc_terms(doc, text_fields)
        if len(terms) > 0:
            sentences.append(terms)

    model = Word2Vec(sentences=sentences,
                     vector_size=vector_size,
                     window=window,
                     min_count=min_count,
                     sg=sg,
                     epochs=epochs,
                     seed=seed,
                     workers=-1)
    return model

def text_vector(terms, model):
    
    vectors = []
    for term in terms:
        if term in model.wv:
            vectors.append(model.wv[term])

    if len(vectors) == 0:
        return None

    
    total = np.zeros(model.vector_size)
    for v in vectors:
        total += v
    return total / len(vectors)


def build_doc_vectors(docs, text_fields, model):
    
    doc_vectors = {}
    for doc in docs:
        terms = get_doc_terms(doc, text_fields)
        vector = text_vector(terms, model)
        if vector is not None:
            doc_vectors[doc['pid']] = vector
    return doc_vectors


def cosine_similarity(v1, v2):
    
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def rank_word2vec(query_terms, docs, model, doc_vectors):
    
    query_vector = text_vector(query_terms, model)
    # if no word of the query is in the vocabulary we can't rank
    if query_vector is None:
        return []

    doc_scores = []
    for product_id in docs:
        if product_id not in doc_vectors:
            continue
        score = cosine_similarity(query_vector, doc_vectors[product_id])
        doc_scores.append([score, product_id])

    
    doc_scores.sort(reverse=True)
    return doc_scores


def search_word2vec(query, preprocessor, index, model, doc_vectors, top_k=20):
   
    query_terms = preprocessor.preprocess_text(query)
    docs = conjunctive_search(query_terms, index)
    ranked_docs = rank_word2vec(query_terms, docs, model, doc_vectors)
    return ranked_docs[:top_k]
