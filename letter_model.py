# -*- coding: utf-8 -*-
import itertools
import search_backend.read_files as read_files
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict

tokenizer = RegexpTokenizer(r'\w+')

first = 'Spanish'
second = 'German'
spanish = defaultdict(float)
german = defaultdict(float)
number_of_grams = 3
y = 0.77


def read_file(file_name1):
    with open(file_name1) as auto:
        file_first = auto.read().decode('utf-8')
    return file_first


def calc_probability_letter(doc, dic, number):
    for start, end in tokenizer.span_tokenize(doc):
        for j in range(0, number):
            for i in range(start, end - number + j):
                dic[doc[i: i + number - j]] += 1
                if j == number - 1:
                    dic[u'$$wholeword$$'] += 1


def calc_probability_words(doc, dic, number):
    words = []
    for s, e in tokenizer.span_tokenize(doc):
        words.append(doc[s:e])
    for j in range(0, number):
        for i in range(len(words) - number + j):
            set_of_words = u''
            for k in range(i, i + number - j):
                set_of_words += words[k] + u' '
            dic[set_of_words] += 1
            if j == number - 1:
                dic[u'$$wholeword$$'] += 1


def build_model(dic1, dic2, file_name_1, file_name_2):
    file_first = read_file(file_name_1)
    file_second = read_file(file_name_2)
    calc_probability_letter(file_first, dic1, number_of_grams)
    calc_probability_letter(file_second, dic2, number_of_grams)


def build_model_words(dic1, dic2, file_name_1, file_name_2):
    file_first = read_file(file_name_1)
    file_second = read_file(file_name_2)
    calc_probability_words(file_first, dic1, number_of_grams)
    calc_probability_words(file_second, dic2, number_of_grams)


def word_prob(query, start, end, dic, number):
    prob_word = 1
    for i in range(start, end - number):
        total_prob = 0
        for j in range(0, number):
            if dic[query[i:i + number - j - 1]] == 0:
                total_prob += (y ** (j + 1) * (dic[query[i:i + number - j]] / (dic[u'$$wholeword$$'])))
            else:
                total_prob += (y**(j + 1) * (dic[query[i:i + number - j]] / (dic[query[i:i + number - j - 1]])))
        prob_word *= total_prob
    return prob_word


def words_prob(query, start, end, dic, number):
    prob_words = 1
    words = []
    for s, e in tokenizer.span_tokenize(query[start:end]):
        words.append(query[start + s:start + e])
    for i in range(len(words) - number):
        total_prob = 0
        for j in range(0, number):
            set_of_words = u''
            set_of_words_small = u''
            for k in range(i, i + number - j):
                set_of_words += words[k] + u' '
            for k in range(i, i + number - j - 1):
                set_of_words_small += words[k] + u' '
            if dic[set_of_words_small] == 0:
                total_prob += (y**(j + 1) * (dic[set_of_words] / (dic[u'$$wholeword$$'])))
            else:
                total_prob += (y**(j + 1) * (dic[set_of_words] / (dic[set_of_words_small])))
        if (total_prob == 0):
            print set_of_words, set_of_words_small, dic[set_of_words], dic[set_of_words_small]
        prob_words *= total_prob
    return prob_words


def predict_text(query, dic1, dic2):
    query = query
    prob1 = 1
    prob2 = 1
    for start, end in tokenizer.span_tokenize(query):
        prob1 *= word_prob(query, start, end, dic1, number_of_grams)
        prob2 *= word_prob(query, start, end, dic2, number_of_grams)

    print (prob1, prob2)
    if prob1 > prob2:
        print (first)
    else:
        print (second)


def predict_text_by_words(query, dic1, dic2):
    prob1 = 1
    prob2 = 1
    words = []
    for s, e in tokenizer.span_tokenize(query):
        words.append({'start': s, 'end': e})

    for i in (range(len(words) - 5)):
        prob1 *= words_prob(query, words[i]['start'], words[i + 5]['end'], dic1, number_of_grams)
        prob2 *= words_prob(query, words[i]['start'], words[i + 5]['end'], dic2, number_of_grams)

    print (prob1, prob2)
    if prob1 > prob2:
        print (first)
    else:
        print (second)


def correct_spelling(word, dic1, dic2):
    word = word.decode('utf-8')
    right_word = word
    max_prob = 0
    for p in itertools.permutations(word):
        perm = "".join(p)
        prob = max(word_prob(perm, 0, len(perm), dic1), word_prob(perm, 0, len(perm), dic2))
        if prob > max_prob:
            max_prob = prob
            right_word = perm
    return right_word

number_of_grams = 3
build_model(spanish, german, 'spanish_book.txt', 'german_book.txt')

predict_text(u'Jetzt liebe ich Gott: die Menschen liebe ich nicht', spanish, german)

spanish2 = defaultdict(float)
german2 = defaultdict(float)

first = 'First book'
second = 'Second book'
build_model_words(spanish2, german2, 'spanish_book.txt', 'spanish_book_2.txt')
predict_text_by_words(u'seria caer en la mayor de ellas: la y el buen sentido no deben', spanish2, german2)


