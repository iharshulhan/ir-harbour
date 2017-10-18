# -*- coding: utf-8 -*-

from process_documents import index_files
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords
from collections import defaultdict
import read_files
tokenizer = RegexpTokenizer(r'\w+')

stopwords = set(stopwords.words('english'))
stemmer = EnglishStemmer()
index = index_files()

def search(query_string):
    result_list = []

    doc = query_string
    tokens = []
    for start, end in tokenizer.span_tokenize(doc):
        token = doc[start:end].lower()
        if token in stopwords:
            continue
        tokens.append(stemmer.stem(token))
    docs = queryOr(tokens)
    documents = read_files.crawl_files()

    for key, value in sorted(docs.iteritems(), key=lambda (k, v): (v, k)):
        result_list.append({
            'title': documents[key]['name'],
            'snippet': documents[key]['doc'][0:150],
            'href': 'http://www.example.com'
        })

    return result_list


def queryOr(tokens):
    docs = defaultdict(int)
    for doc in index[tokens[0]]:
        docs[doc] = 0

    for token in tokens:
        new_docs = docs.copy()
        for doc in docs:
            if len(index[token][doc]) == 0:
                new_docs.pop(doc)
            else:
                new_docs[doc] = new_docs[doc] + len(index[token][doc])

        docs = new_docs.copy()
    return docs


print search('Python is garbage')