"""This module indexes files"""
import math
import os

from search_backend.db.schemas import *
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords
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
            for start, end in tokenizer.span_tokenize(doc):
                token = doc[start:end].lower()
                if token in stopwords:
                    continue
                num += 1

            document_instance.len = num

        if document_instance.vector is None:
            length_arr = select((dp.index, count(dp.position)) for dp in DocumentPosition
                                if dp.document == document_instance)
            v2_2 = 0
            for index in length_arr:
                tf_doc = tf(index[1])
                v2_2 += tf_doc * tf_doc
            document_instance.vector = v2_2

        # Title index
        title_instance = Title.get(location=doc_path)
        if title_instance is None:
            title_instance = Title(location=doc_path)

            for start, end in tokenizer.span_tokenize(doc_path):
                token = doc_path[start:end].lower()
                if token in stopwords:
                    continue
                token = stemmer.stem(token)
                index_title_token = IndexTitle.get(key=token)
                if index_title_token is None:
                    index_title_token = IndexTitle(key=token)
                title_position = TitlePosition(title=title_instance, position=start, index=index_title_token)
        if title_instance.len is None:
            num = 0
            for start, end in tokenizer.span_tokenize(doc_path):
                token = doc_path[start:end].lower()
                if token in stopwords:
                    continue
                num += 1

            title_instance.len = num

        if title_instance.vector is None:
            length_arr = select((dp.index, count(dp.position)) for dp in TitlePosition
                                if dp.title == title_instance)
            v2_2 = 0
            for index in length_arr:
                tf_doc = tf(index[1])
                v2_2 += tf_doc * tf_doc
            title_instance.vector = v2_2

        # from pympler import summary, muppy
        # sum1 = summary.summarize(muppy.get_objects())
        # summary.print_(sum1)


        # query_stats = sorted(db.local_stats.values(),
        #                      reverse=True, key=attrgetter('sum_time'))
        # for qs in query_stats:
        #     print(qs.sum_time, qs.db_count, qs.sql)
        commit()


if __name__ == '__main__':
    index_files()
