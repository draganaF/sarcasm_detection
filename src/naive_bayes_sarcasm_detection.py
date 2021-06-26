from __future__ import print_function

import re
import string
import math
import pandas as pd
import json
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize


def load_data():
    # TODO 1: ucitati podatke iz data/train.tsv datoteke
    # rezultat treba da budu dve liste, texts i sentiments
    texts, sentiments = [], []
    with open('../data/dataset_part1.json') as file:
        for line in file:
            json_text = json.loads(line)
            texts.append(json_text["headline"])
            sentiments.append(json_text["is_sarcastic"])

    return texts, sentiments


def preprocess(text):
    # TODO 2: implementirati preprocesiranje teksta
    # - izbacivanje znakova interpunkcije
    # - svodjenje celog teksta na mala slova
    # rezultat treba da bude preprocesiran tekst
    text = text.lower()

    punct = set(string.punctuation)
    stop = ['a', 'an', 'the', 'and', 'for', 'of']
    ps = PorterStemmer()
    text_returned = ''
    for t in text:
        tt = ''
        if t not in punct:
            tt += t
        text_returned += tt + ''
    r = ''

    for returned in text_returned.split(" "):

        if returned not in stop:
            returned = ps.stem(returned)
            r += returned + ' '

    return r


def tokenize(text):
    text = preprocess(text)
    # TODO 3: implementirati tokenizaciju teksta na reci
    # rezultat treba da bude lista reci koje se nalaze u datom tekstu
    words = text.split(' ')
    return words


def count_words(text):
    if isinstance(text, list):
        words = text
    else:
        words = tokenize(text)
    # TODO 4: implementirati prebrojavanje reci u datum tekstu
    # rezultat treba da bude mapa, ciji kljucevi su reci, a vrednosti broj ponavljanja te reci u datoj recenici
    words_count = {}
    for w in words:
        if words_count.keys().__contains__(w):
            words_count[w] = words_count[w] + 1
        else:
            words_count[w] = 1
    return words_count


def fit(texts, sentiments):
    # inicijalizacija struktura
    bag_of_words = {}  # bag-of-words za sve recenzije
    words_count = {'sarcastic': {},  # isto bag-of-words, ali posebno za pozivitne i negativne recenzije
                   'notsarcastic': {}}
    texts_count = {'sarcastic': 0.0,  # broj tekstova za pozivitne i negativne recenzije
                   'notsarcastic': 0.0}

    # TODO 5: proci kroz sve recenzije i sentimente i napuniti gore inicijalizovane strukture
    # bag-of-words je mapa svih reci i broja njihovih ponavljanja u celom korpusu recenzija
    for text, sentiment in zip(texts, sentiments):
        words_dic = count_words(text)
        for word, count in words_dic.items():
            if bag_of_words.__contains__(word):
                bag_of_words[word] += count
            else:
                bag_of_words[word] = count

            if sentiment == 1:
                key = "sarcastic"
            else:
                key = "notsarcastic"

            if words_count[key].__contains__(word):
                words_count[key][word] += count
            else:
                words_count[key][word] = count

            texts_count[key] += 1

    return bag_of_words, words_count, texts_count


def predict(text, bag_of_words, words_count, texts_count):
    words = tokenize(text)  # tokenizacija teksta

    # TODO 6: implementirati Naivni Bayes klasifikator za sentiment teksta (recenzije)
    # rezultat treba da bude mapa verovatnoca da je dati tekst klasifikovan kao pozitivnu i negativna recenzija

    score_pos, score_neg = 0.0, 0.0
    sum_sentiments = {'sarcastic': 0.0, 'notsarcastic': 0.0}

    sum_all_sentiments = float(sum(texts_count.values()))
    p_sentiments = {}
    for sentiment in texts_count.keys():
        p_sentiments[sentiment] = texts_count[sentiment] / sum_all_sentiments

    sum_all_words = float(sum(bag_of_words.values()))
    sum_words = {}
    for sentiment in texts_count.keys():
        sum_words[sentiment] = float(sum(words_count[sentiment].values()))

    for word in words:
        if word in bag_of_words:
            word_probability = bag_of_words[word] / sum_all_words

        for sentiment in texts_count.keys():
            if word in words_count[sentiment]:
                pp_word = words_count[sentiment][word] / sum_words[sentiment]
                sum_sentiments[sentiment] += math.log(pp_word / word_probability)

    score_pos = math.exp(sum_sentiments['sarcastic'] + math.log(p_sentiments['sarcastic']))
    score_neg = math.exp(sum_sentiments['notsarcastic'] + math.log(p_sentiments['notsarcastic']))
    return {'sarcastic': score_pos, 'notsarcastic': score_neg}

def test_function():
    # ucitavanje data seta
    texts, sentiments = load_data()

    # izracunavanje / prebrojavanje stvari potrebnih za primenu Naivnog Bayesa
    bag_of_words, words_count, texts_count = fit(texts, sentiments)

    good, bad = 0, 0

    # recenzija
    with open('../data/dataset_part2.json') as file:
        for line in file:
            json_text = json.loads(line)
            text = json_text["headline"]
            sentiment = json_text["is_sarcastic"]
            predictions = predict(text, bag_of_words, words_count, texts_count)

            if predictions['sarcastic'] > predictions['notsarcastic'] and sentiment == 1:
                good += 1
            elif predictions['sarcastic'] < predictions['notsarcastic'] and sentiment == 0:
                good += 1
            else:
                bad += 1
    print(good)
    print(bad)


if __name__ == '__main__':
    # ucitavanje data seta
    texts, sentiments = load_data()

    # izracunavanje / prebrojavanje stvari potrebnih za primenu Naivnog Bayesa
    bag_of_words, words_count, texts_count = fit(texts, sentiments)

    # recenzija
    text = "pope francis wearing sweater vestments he got for christmas"
    # klasifikovati sentiment recenzije koriscenjem Naivnog Bayes klasifikatora
    predictions = predict(text, bag_of_words, words_count, texts_count)

    test_function()

    print('-' * 30)
    print('Review: {0}'.format(text))
    print('Score(sarcastic): {0}'.format(predictions['sarcastic']))
    print('Score(notsarcastic): {0}'.format(predictions['notsarcastic']))
