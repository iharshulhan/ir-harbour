# -*- coding: utf-8 -*-
"""This module performs search on index"""
import os
import time

import sklearn.externals.joblib as joblib
from search_backend.db.schemas import *
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords
from collections import defaultdict
from search_backend.process_documents import index_files
import search_backend.read_files as read_files
import search_backend.text_rank as text_rank
from operator import attrgetter
from joblib import Parallel, delayed
import math
import statistics
import numpy as np

tokenizer = RegexpTokenizer(r'\w+')

dir = os.path.dirname(__file__)
model = joblib.load((dir + '/RandomForest90'))


def tf(num):
    return 1 + math.log(1 + num)


def idf(num_docs, term_num):
    return math.log(num_docs / (1 + term_num))


def vector_space_model(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


@db_session
def features_for_doc(tokens, doc):
    start_glob = time.time()
    start = time.time()
    from search_backend.db.schemas import Document, DocumentPosition, Index
    covered_query_term_number = 0
    sum_idf = 0
    tfs = []
    tfs_stream = []
    tf_idfs = []
    stream_length = doc.len
    tfs_all_tokens_query = np.zeros(Index.select().count())
    tfs_all_tokens_doc = np.zeros(Index.select().count())
    zero_id = select(min(i.id) for i in Index)[:][0]
    doc_len = Document.select().count()
    end = time.time()
    # print('Init locals: ', (end - start))

    start = time.time()
    for token in tokens:
        len_doc = len(Index.get(key=token).documents.filter(lambda doc_position: doc_position.document == doc))
        covered_query_term_number += len_doc > 0
        tfs_all_tokens_query[Index.get(key=token).id - zero_id] = tf(len_doc)
        current_idf = idf(doc_len, count(i.documents.document for i in Index if i.key == token))
        sum_idf += current_idf
        tfs.append(tf(len_doc))
        tfs_stream.append(len_doc / stream_length)
        tf_idfs.append(current_idf * tf(len_doc))

    end = time.time()
    # print('Time elapsed on normal features: ', (end - start))

    start = time.time()
    covered_query_term_ratio = covered_query_term_number / len(tokens)

    if len(tokens) == 1:
        features = [covered_query_term_number, covered_query_term_ratio, stream_length, sum_idf, sum(tfs), min(tfs),
                    max(tfs), statistics.mean(tfs), 0, sum(tfs_stream), min(tfs_stream),
                    max(tfs_stream), statistics.mean(tfs_stream), 0, sum(tf_idfs),
                    min(tf_idfs), max(tf_idfs), statistics.mean(tf_idfs), 0]
    else:
        features = [covered_query_term_number, covered_query_term_ratio, stream_length, sum_idf, sum(tfs), min(tfs),
                    max(tfs), statistics.mean(tfs), statistics.variance(tfs), sum(tfs_stream), min(tfs_stream),
                    max(tfs_stream), statistics.mean(tfs_stream), statistics.variance(tfs_stream), sum(tf_idfs),
                    min(tf_idfs), max(tf_idfs), statistics.mean(tf_idfs), statistics.variance(tf_idfs)]

    end = time.time()
    # print('Statistics and copying: ', (end - start))

    start = time.time()
    v1 = np.array(tfs_all_tokens_query)
    length_arr = select((dp.index, count(dp.position)) for dp in DocumentPosition if dp.document == doc)
    for index in length_arr:
        len_doc = index[1]
        tfs_all_tokens_doc[index[0].id - zero_id] = tf(len_doc)
    v2 = np.array(tfs_all_tokens_doc)

    features.append(1)  # Boolean model
    features.append(vector_space_model(v1, v2))
    db.merge_local_stats()
    end = time.time()
    # print('Time elapsed on vector space model: ', (end - start))

    end = time.time()
    # print('Building features: ', (end - start_glob))
    return features


@db_session
def check_doc(token, doc):
    from search_backend.db.schemas import Index
    try:
        len_doc = count(Index.get(key=token).documents
                        .filter(lambda doc_position: doc_position.document.id == doc))
    except:
        len_doc = -1
    db.merge_local_stats()
    return doc, len_doc


@db_session
def find_exact_phrase(doc, tokens, distance, Index, result_docs_position, model, Document, db):
    token_num = 0
    token_index = []
    for i in range(len(tokens)):
        token_index.append(0)

    while 1 > 0:
        current_index = lambda: token_index[token_num]

        def docs(token_number):
            return Index.get(key=tokens[token_number]).documents \
                       .filter(lambda doc_position: doc_position.document.id == doc)[:]

        if token_num == len(tokens) - 1:
            doc_positions = docs(0)
            result_docs_position[doc] = doc_positions[token_index[0]].position
            if doc not in result_docs:
                prob = model.predict_proba([features_for_doc(tokens, Document.get(id=doc))])
                result_docs[doc] = prob[0][1] - prob[0][0]
            token_index[token_num] += 1

            if len(docs(token_num)) <= current_index():
                break
            token_num -= 1
            continue

        distance_current_next = docs(token_num + 1)[token_index[token_num + 1]].position - \
                                docs(token_num)[current_index()].position - len(tokens[token_num])
        if -20 < distance_current_next < distance + 20:
            token_num += 1
        else:
            if distance_current_next < -20:
                token_index[token_num + 1] += 1
                if len(docs(token_num + 1)) <= token_index[token_num + 1]:
                    break
            else:
                token_index[token_num] += 1
                if len(docs(token_num)) <= current_index():
                    break
                if token_num != 0:
                    token_num -= 1
    db.merge_local_stats()


@db_session
def predict_rank(tokens, k, v, model):
    from search_backend.db.schemas import Document
    if v == len(tokens):
        document = Document.get(id=k)
        start = time.time()
        features = features_for_doc(tokens, document)
        end = time.time()
        print('Build features', (end - start))
        start = time.time()
        prob = model.predict_proba([features])
        end = time.time()
        print('Model predict: ', (end - start))
        return k, prob[0][1]


class Search:
    def __init__(self):
        self.stopwords = set(stopwords.words('english'))
        self.stemmer = EnglishStemmer()

    @db_session
    def search(self, query_string):
        result_list = []
        doc = query_string
        tokens = []
        for start, end in tokenizer.span_tokenize(doc):
            token = doc[start:end].lower()
            if token in self.stopwords:
                continue
            tokens.append(self.stemmer.stem(token))
        if len(tokens) == 0:
            return ([{
                'title': 'No meaningful words provided',
                'snippet': '',
                'href': 'http://www.example.com'
            }])

        start = time.time()
        # docs, positions = self.query_phrase(tokens, 1)
        docs = self.query_and(tokens)
        end = time.time()
        print('Time elapsed: ', (end - start))
        if len(docs) > 0:
            for key, value in sorted(docs.items(), key=lambda kv: (-kv[1], kv[0])):
                document = Document.get(id=key)
                summary = document.snippet
                result_list.append({
                    'title': document.location,
                    'snippet':  # open(document.location).read()[positions[key]:positions[key] + 150] +
                        '<br\><h3>Summary</h3>' + summary,
                    'href': 'http://www.example.com'
                })
        else:
            result_list.append({
                'title': 'Nothing was found :(',
                'snippet': '',
                'href': 'http://www.example.com'
            })

        return result_list

    @db_session
    def query_and(self, tokens):
        start = time.time()
        docs = defaultdict(int)
        docs_num = defaultdict(int)
        result_docs = defaultdict(int)
        min_token = select((i.key, count(i.documents)) for i in Index
                           if i.key in tokens).order_by(2)[:][0][0]

        for doc in Index.get(key=min_token).documents:
            docs[doc.document.id] = 0
        for token in tokens:
            len1 = len(docs)
            lens = Parallel(n_jobs=min(1, len1))(
                delayed(check_doc)(token, doc) for doc in docs)
            for x in lens:
                if x[1] > 0:
                    docs[x[0]] = docs[x[0]] + 1
                    docs_num[x[0]] = docs_num[x[0]] + int(x[1])
        end = time.time()
        print('Time elapsed Query and: ', (end - start))
        len1 = len(docs)
        start = time.time()
        ranks = Parallel(n_jobs=min(1, len1))(
            delayed(predict_rank)(tokens, k, v, model) for (k, v) in docs.items())
        for rank in ranks:
            if rank is not None:
                result_docs[rank[0]] = rank[1]
        query_stats = sorted(db.global_stats.values(),
                             reverse=True, key=attrgetter('sum_time'))
        end = time.time()
        print('Time elapsed Ranking: ', (end - start))
        for qs in query_stats:
            print(qs.sum_time, qs.db_count, qs.sql)
        return result_docs

    @db_session
    def query_phrase(self, tokens, distance):
        global result_docs, result_docs_position
        docs = self.query_and(tokens)
        result_docs = defaultdict(int)
        result_docs_position = defaultdict(int)
        Parallel(n_jobs=min(8, len(docs)))(delayed(find_exact_phrase)(tokens, doc, distance, Index,
                                                                      result_docs_position, self.model, Document, db)
                                           for doc in docs)
        query_stats = sorted(db.global_stats.values(),
                             reverse=True, key=attrgetter('sum_time'))
        for qs in query_stats:
            print(qs.sum_time, qs.db_count, qs.sql)
        return result_docs, result_docs_position
