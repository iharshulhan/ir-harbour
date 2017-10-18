import search_backend.read_files as read_files
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords
from collections import defaultdict

tokenizer = RegexpTokenizer(r'\w+')
stopwords = set(stopwords.words('english'))
stemmer = EnglishStemmer()


def index_files():
    files = read_files.crawl_files()
    index = defaultdict(defaultdict(list).copy)
    for id, file in enumerate(files):
        doc = file['doc']
        for start, end in tokenizer.span_tokenize(doc):
            token = doc[start:end].lower()
            if token in stopwords:
                continue
            token = stemmer.stem(token)
            index[token][id].append(start)
    return index
