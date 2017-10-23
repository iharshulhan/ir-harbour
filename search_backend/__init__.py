# -*- coding: utf-8 -*-
"""This module performs search on index"""
from search_backend.db.schemas import *
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords
from collections import defaultdict
from search_backend.process_documents import index_files
import search_backend.read_files as read_files
import search_backend.text_rank as text_rank
from operator import attrgetter

tokenizer = RegexpTokenizer(r'\w+')


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
        docs, positions = self.query_phrase(tokens, 1)

        if len(docs) > 0:
            for key, value in sorted(docs.items(), key=lambda kv: (-kv[1], kv[0])):
                document = Document.get(id=key)
                # summary = text_rank.summarize(document.location)
                result_list.append({
                    'title': document.location,
                    'snippet': '',
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
        docs = defaultdict(int)
        min_token = select((i.key, count(i.documents)) for i in Index
                           if i.key in tokens).order_by(2)[:][0][0]
        for doc in Index.get(key=min_token).documents:
            docs[doc.document.id] = 0
        for token in tokens:
            new_docs = docs.copy()
            for doc in docs:
                len_doc = count(Index.get(key=token).documents
                                .filter(lambda doc_position: doc_position.document.id == doc))
                if len_doc == 0:
                    new_docs.pop(doc)
                else:
                    new_docs[doc] = new_docs[doc] + len_doc
            docs = new_docs.copy()
        return docs

    @db_session
    def query_phrase(self, tokens, distance):
        docs = self.query_and(tokens)
        result_docs = defaultdict(int)
        result_docs_position = defaultdict(int)
        for doc in docs:
            token_num = 0
            token_index = []
            for i in range(len(tokens)):
                token_index.append(0)

            while 1 > 0:
                current_index = lambda: token_index[token_num]

                def docs(token_number):
                    return Index.get(key=tokens[token_number]).documents \
                               .filter(lambda doc_position: doc_position.document.id == doc) \
                               .order_by(DocumentPosition.position)[:]

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
        query_stats = sorted(db.local_stats.values(),
                             reverse=True, key=attrgetter('sum_time'))
        for qs in query_stats:
            print(qs.sum_time, qs.db_count, qs.sql)
        return result_docs, result_docs_position
