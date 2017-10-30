# -*- coding: utf-8 -*-
"""This module performs search on index"""
import time

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

tokenizer = RegexpTokenizer(r'\w+')

docs = defaultdict(int)
docs_num = defaultdict(int)


def tf(num):
    return 1 + math.log(num)


def idf(num_docs, term_num):
    return math.log(num_docs / (1 + term_num))


@db_session
def features_for_doc(tokens, doc):
    covered_query_term_number = 0
    sum_idf = 0
    tfs = []
    tfs_stream = []
    tf_idfs = []
    stream_length = 10  # doc.length
    for token in tokens:
        len_doc = len(Index.get(key=token).documents.filter(lambda doc_position: doc_position.document == doc))
        covered_query_term_number += len_doc > 0
        current_idf = idf(len(Document.select()), count(i.documents.document for i in Index if i.key == token))
        sum_idf += current_idf
        tfs.append(tf(len_doc))
        tfs_stream.append(len_doc / stream_length)
        tf_idfs.append(current_idf * tf(len_doc))

    covered_query_term_ratio = covered_query_term_number / len(tokens)

    features = [covered_query_term_number, covered_query_term_ratio, stream_length, sum_idf, sum(tfs), min(tfs),
                max(tfs), statistics.mean(tfs), statistics.variance(tfs), sum(tfs_stream), min(tfs_stream),
                max(tfs_stream), statistics.mean(tfs_stream), statistics.variance(tfs_stream), sum(tf_idfs),
                min(tf_idfs), max(tf_idfs), statistics.mean(tf_idfs), statistics.variance(tf_idfs)]

    print(features)

    # query_stats = sorted(db.local_stats.values(),
    #                      reverse=True, key=attrgetter('sum_time'))
    # for qs in query_stats:
    #     print(qs.sum_time, qs.db_count, qs.sql)


@db_session
def test_start():
    features_for_doc(['python', 'good'], Index.get(key='good').documents.page(1, pagesize=1)[0].document)


test_start()


@db_session
def check_doc(token, doc):
    len_doc = count(Index.get(key=token).documents
                    .filter(lambda doc_position: doc_position.document.id == doc))
    if len_doc > 0:
        docs[doc] = docs[doc] + 1
        docs_num[doc] = docs_num[doc] + int(len_doc)
    db.merge_local_stats()


result_docs = defaultdict(int)
result_docs_position = defaultdict(int)


@db_session
def find_exact_phrase(doc, tokens, distance):
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
            if doc in result_docs:
                result_docs[doc] += 1
            else:
                result_docs[doc] = 1
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
        docs, positions = self.query_phrase(tokens, 1)
        end = time.time()
        print('Time elapse: ', (end - start))
        if len(docs) > 0:
            for key, value in sorted(docs.items(), key=lambda kv: (-kv[1], kv[0])):
                document = Document.get(id=key)
                summary = document.snippet
                result_list.append({
                    'title': document.location,
                    'snippet': open(document.location).read()[positions[key]:positions[key] + 150] +
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

    @staticmethod
    @db_session
    def query_and(tokens):
        global docs, docs_num
        docs = defaultdict(int)
        docs_num = defaultdict(int)
        result_docs = defaultdict(int)
        min_token = select((i.key, count(i.documents)) for i in Index
                           if i.key in tokens).order_by(2)[:][0][0]

        for doc in Index.get(key=min_token).documents:
            docs[doc.document.id] = 0
        for token in tokens:
            Parallel(n_jobs=8, backend="threading")(
                delayed(check_doc)(token=token, doc=doc, index=Index) for doc in docs)
        for (k, v) in docs.items():
            if v == len(tokens):
                result_docs[k] = docs_num[k]
        return result_docs

    @db_session
    def query_phrase(self, tokens, distance):
        global result_docs, result_docs_position
        docs = self.query_and(tokens)
        result_docs = defaultdict(int)
        result_docs_position = defaultdict(int)
        Parallel(n_jobs=8, backend="threading")(delayed(find_exact_phrase)(tokens=tokens, doc=doc,
                                                                           distance=distance, index=Index)
                                                for doc in docs)
        query_stats = sorted(db.global_stats.values(),
                             reverse=True, key=attrgetter('sum_time'))
        for qs in query_stats:
            print(qs.sum_time, qs.db_count, qs.sql)
        return result_docs, result_docs_position
