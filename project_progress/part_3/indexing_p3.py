import math
from collections import defaultdict, Counter
import numpy as np


def get_doc_terms(doc, text_fields):
    terms = []
    for field in text_fields:
        tokens = doc.get(field)
        if not tokens:
            continue
        for token in tokens:
            terms.append(token)
    return terms


def build_p3_index(docs, text_fields):
    index = defaultdict(list)
    tf = defaultdict(list)
    tf_raw = defaultdict(list)
    df = defaultdict(int)
    idf = {}
    product_id_info = {}
    doc_lengths = {}
    
    N = len(docs)
    total_terms = 0

    for doc in docs:
        product_id = doc['pid']
        product_id_info[product_id] = doc
        terms = get_doc_terms(doc, text_fields)
        
        doc_len = len(terms)
        doc_lengths[product_id] = doc_len
        total_terms += doc_len

        term_counts = Counter(terms)

        tf_sublinear = {}
        for term, count in term_counts.items():
            tf_sublinear[term] = 1 + math.log2(count)

        norm = math.sqrt(sum(w ** 2 for w in tf_sublinear.values())) if tf_sublinear else 1.0

        for term, count in term_counts.items():
            index[term].append(product_id)
            tf_raw[term].append(count)
            sub_w = tf_sublinear[term]
            tf[term].append(np.round(sub_w / norm, 4) if norm > 0 else 0.0)
            df[term] += 1

    for term in df:
        idf[term] = np.round(math.log2(N / df[term]), 4)

    avgdl = total_terms / N if N > 0 else 0.0

    return index, tf, tf_raw, df, idf, doc_lengths, avgdl, product_id_info, N


def conjunctive_search(query_terms, index):
    if len(query_terms) == 0:
        return set()
    
    result = None
    for term in query_terms:
        if term not in index:
            return set()
        term_docs = set(index[term])
        
        if result is None:
            result = term_docs
        else:
            result = result.intersection(term_docs)
            
        if len(result) == 0:
            break
            
    return result


def rank_documents_tfidf(query_terms, docs, index, tf, idf):
    unique_terms = []
    for term in query_terms:
        if term not in unique_terms:
            unique_terms.append(term)

    query_counts = Counter(query_terms)
    query_weights = {term: 1 + math.log2(count) for term, count in query_counts.items()}
    query_norm = math.sqrt(sum(w ** 2 for w in query_weights.values())) if query_weights else 1.0

    query_vector = [0.0] * len(unique_terms)
    doc_vectors = defaultdict(lambda: [0.0] * len(unique_terms))

    for i in range(len(unique_terms)):
        term = unique_terms[i]
        if term not in index:
            continue
        query_vector[i] = (query_weights[term] / query_norm) * idf[term] if query_norm > 0 else 0.0

        postings = index[term]
        for doc_position in range(len(postings)):
            product_id = postings[doc_position]
            if product_id in docs:
                doc_vectors[product_id][i] = tf[term][doc_position] * idf[term]

    doc_scores = []
    for product_id in doc_vectors:
        score = np.dot(doc_vectors[product_id], query_vector)
        doc_scores.append([score, product_id])

    doc_scores.sort(key=lambda x: x[0], reverse=True)
    return doc_scores


def rank_documents_bm25(query_terms, docs, index, tf_raw, df, N, doc_lengths, avgdl, k1=1.5, b=0.75):
    doc_scores = []

    for product_id in docs:
        score = 0.0
        for term in query_terms:
            if term not in index or product_id not in index[term]:
                continue

            doc_pos = index[term].index(product_id)
            f_qi_d = tf_raw[term][doc_pos]

            doc_len = doc_lengths.get(product_id, avgdl)

            n_i = df[term]
            idf_val = math.log2(N / n_i) if n_i > 0 else 0.0

            numerator = (k1 + 1) * f_qi_d
            denominator = k1 * ((1 - b) + b * (doc_len / avgdl)) + f_qi_d

            score += idf_val * (numerator / denominator)

        doc_scores.append([score, product_id])

    doc_scores.sort(key=lambda x: x[0], reverse=True)
    return doc_scores


def rank_documents_custom(query_terms, docs, index, tf_raw, df, N, doc_lengths, avgdl, product_id_info, alpha=1.0, beta=0.5):
    bm25_scores = rank_documents_bm25(query_terms, docs, index, tf_raw, df, N, doc_lengths, avgdl)
    
    custom_scores = []
    for score_bm25, product_id in bm25_scores:
        doc_info = product_id_info.get(product_id, {})
        
        rating = doc_info.get('average_rating_value') or 0.0
        discount = doc_info.get('discount_percent') or 0.0
        
        norm_rating = rating / 5.0 if rating <= 5.0 else 1.0
        norm_discount = discount / 100.0 if discount <= 100.0 else 1.0

        final_score = score_bm25 + (alpha * norm_rating) + (beta * norm_discount)
        custom_scores.append([final_score, product_id])
        
    custom_scores.sort(key=lambda x: x[0], reverse=True)
    return custom_scores