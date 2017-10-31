"""This module indexes files"""
import math
import os
from collections import defaultdict

from search_backend.db.schemas import *
import io
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords
from operator import attrgetter
import search_backend.read_files as read_files
import search_backend.text_rank as text_rank
import bz2
import xml.etree.ElementTree as ET

tokenizer = RegexpTokenizer(r'\w+')
stopwords = set(stopwords.words('english'))
stemmer = EnglishStemmer()


def tf(num):
    return 1 + math.log(num)


def idf(num_docs, term_num):
    return math.log(num_docs / (1 + term_num))


@db_session
def index_files():
    dir = os.path.dirname(__file__)
    f = bz2.open(dir + '/../enwiki-20171020-pages-articles1.xml-p10p30302.bz2')

    def readNextPage(file):
        page = ''
        for line in file:
            line = str(line, encoding='utf-8').strip()
            if line == '<page>':
                page = line
            elif line == '</page>':
                page += '\n' + line
                return ET.fromstring(page)
            elif page != '':
                page += '\n' + line

    while 1 > 0:
        file = readNextPage(f)
        if file.find('title').text is None:
            print('loh', file.find('revision').find('text').text)
            break
        doc_path = file.find('title').text
        doc = file.find('revision').find('text').text
        document_instance = Document.get(location=doc_path)
        print(doc_path)
        if document_instance is None:
            summary = ''
            try:
                summary = text_rank.summarize(doc)
            except ValueError:
                print("Oops!  Summary error")
            document_instance = Document(location=doc_path, snippet=summary)

            for start, end in tokenizer.span_tokenize(doc):
                token = doc[start:end].lower()
                if token in stopwords:
                    continue
                token = stemmer.stem(token)
                index_token = Index.get(key=token)
                if index_token is None:
                    index_token = Index(key=token)
                document_position = DocumentPosition(document=document_instance, position=start, index=index_token)
        if document_instance.len is None:
            num = 0
            dic = defaultdict(int)
            for start, end in tokenizer.span_tokenize(doc):
                token = doc[start:end].lower()
                if token in stopwords:
                    continue
                token = stemmer.stem(token)
                num += 1
                if token not in dic:
                    dic[token] = 0
                dic[token] += 1

            document_instance.len = num

        # query_stats = sorted(db.local_stats.values(),
        #                      reverse=True, key=attrgetter('sum_time'))
        # for qs in query_stats:
        #     print(qs.sum_time, qs.db_count, qs.sql)
        commit()


# index_files()
