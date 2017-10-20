# -*- coding: utf-8 -*-
from search_backend.process_documents import index_files
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords
from collections import defaultdict
import search_backend.read_files as read_files
import nltk
nltk.download('stopwords')

tokenizer = RegexpTokenizer(r'\w+')


class Search:

    def __init__(self):
        self.stopwords = set(stopwords.words('english'))
        self.stemmer = EnglishStemmer()
        self.index = index_files()

    def search(self, query_string):
        result_list = []
        doc = query_string
        tokens = []
        for start, end in tokenizer.span_tokenize(doc):
            token = doc[start:end].lower()
            if token in self.stopwords:
                continue
            tokens.append(self.stemmer.stem(token))
        docs, positions = self.query_phrase(tokens, 1)
        documents = read_files.crawl_files()
        if len(docs) > 0:
            for key, value in sorted(docs.items(), key=lambda kv: (-kv[1], kv[0])):
                result_list.append({
                    'title': documents[key]['name'],
                    'snippet': documents[key]['doc'][positions[key]:positions[key] + 150],
                    'href': 'http://www.example.com'
                })
        else:
            result_list.append({
                'title': 'Nothing was found :(',
                'snippet': '',
                'href': 'http://www.example.com'
            })

        return result_list

    def query_and(self, tokens):
        docs = defaultdict(int)
        for doc in self.index[tokens[0]]:
            docs[doc] = 0
        for token in tokens:
            new_docs = docs.copy()
            for doc in docs:
                if len(self.index[token][doc]) == 0:
                    new_docs.pop(doc)
                else:
                    new_docs[doc] = new_docs[doc] + len(self.index[token][doc])
            docs = new_docs.copy()
        return docs

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
                current_index = token_index[token_num]
                if token_num == len(tokens) - 1:
                    result_docs_position[doc] = self.index[tokens[0]][doc][token_index[0]]
                    if doc in result_docs:
                        result_docs[doc] += 1
                    else:
                        result_docs[doc] = 1
                    token_index[token_num] += 1
                    if len(self.index[tokens[token_num]][doc]) <= token_index[token_num]:
                        break
                    token_num -= 1
                    continue
                distance_current_next = self.index[tokens[token_num + 1]][doc][token_index[token_num + 1]] - \
                    self.index[tokens[token_num]][doc][current_index] - len(tokens[token_num])
                if -20 < distance_current_next < distance + 20:
                    token_num += 1
                else:
                    if distance_current_next < -20:
                        token_index[token_num + 1] += 1
                        if len(self.index[tokens[token_num + 1]][doc]) <= token_index[token_num + 1]:
                            break
                    else:
                        token_index[token_num] += 1
                        if len(self.index[tokens[token_num]][doc]) <= token_index[token_num]:
                            break
                        if token_num != 0:
                            token_num -= 1
        return result_docs, result_docs_position
