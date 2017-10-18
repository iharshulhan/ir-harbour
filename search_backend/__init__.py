# -*- coding: utf-8 -*-
from search_backend.process_documents import index_files
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords
from collections import defaultdict
import search_backend.read_files
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
        docs = self.query_or(tokens)
        documents = read_files.crawl_files()
        if len(docs) > 0:
            for key, value in sorted(docs.items(), key=lambda kv: (-kv[1], kv[0])):
                result_list.append({
                    'title': documents[key]['name'],
                    'snippet': documents[key]['doc'][0:150],
                    'href': 'http://www.example.com'
                })
        else:
            result_list.append({
                'title': 'Nothing was found :(',
                'snippet': '',
                'href': 'http://www.example.com'
            })

        return result_list

    def query_or(self, tokens):
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
