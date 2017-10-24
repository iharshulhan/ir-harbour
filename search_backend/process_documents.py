"""This module indexes files"""
from search_backend.db.schemas import *
import io
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords
from operator import attrgetter
import search_backend.read_files as read_files
import search_backend.text_rank as text_rank


tokenizer = RegexpTokenizer(r'\w+')
stopwords = set(stopwords.words('english'))
stemmer = EnglishStemmer()


@db_session
def index_files():
    files = read_files.crawl_files()
    for id, file in enumerate(files):
        doc_path = file['name']
        doc = io.open(doc_path, encoding="utf-8").read()
        document_instance = Document.get(location=doc_path)
        if document_instance is None:
            document_instance = Document(location=doc_path, snippet=text_rank.summarize(doc_path))

            for start, end in tokenizer.span_tokenize(doc):
                token = doc[start:end].lower()
                if token in stopwords:
                    continue
                token = stemmer.stem(token)
                index_token = Index.get(key=token)
                if index_token is None:
                    index_token = Index(key=token)
                document_position = DocumentPosition(document=document_instance, position=start, index=index_token)

        query_stats = sorted(db.local_stats.values(),
                             reverse=True, key=attrgetter('sum_time'))
        for qs in query_stats:
            print(qs.sum_time, qs.db_count, qs.sql)


