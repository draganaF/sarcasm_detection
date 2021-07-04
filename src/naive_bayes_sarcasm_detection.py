from __future__ import print_function
import string
import math
import json
from datetime import time, datetime

from nltk.stem import PorterStemmer


def load_data():
    texts, sentiments = [], []
    with open('../data/dataset_part1.json') as file:
        for line in file:
            json_text = json.loads(line)
            texts.append(json_text["headline"])
            sentiments.append(json_text["is_sarcastic"])

    return texts, sentiments


def preprocess(text):
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
    words = text.split(' ')
    return words


def count_words(text):
    if isinstance(text, list):
        words = text
    else:
        words = tokenize(text)

    words_count = {}
    for w in words:
        if words_count.keys().__contains__(w):
            words_count[w] = words_count[w] + 1
        else:
            words_count[w] = 1
    return words_count


def fit(texts, sentiments):
    begin = datetime.now()
    bag_of_words = {}  # bag-of-words za sve recenzije
    words_count = {'sarcastic': {},  # isto bag-of-words, ali posebno za pozivitne i negativne recenzije
                   'notsarcastic': {}}
    texts_count = {'sarcastic': 0.0,  # broj tekstova za pozivitne i negativne recenzije
                   'notsarcastic': 0.0}

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

    end = datetime.now()
    print((end-begin).seconds)

    return bag_of_words, words_count, texts_count


def predict(text, bag_of_words, words_count, texts_count):
    words = tokenize(text)  # tokenizacija teksta

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

    sarcastic_good = 0
    sarcastic_bad = 0

    not_sarcastic_good = 0
    not_sarcastic_bad = 0

    # recenzija
    with open('../data/dataset_part2.json') as file:
        for line in file:
            json_text = json.loads(line)
            text = json_text["headline"]
            sentiment = json_text["is_sarcastic"]
            predictions = predict(text, bag_of_words, words_count, texts_count)

            if predictions['sarcastic'] > predictions['notsarcastic'] and sentiment == 1:
                sarcastic_good += 1
            elif predictions['sarcastic'] < predictions['notsarcastic'] and sentiment == 0:
                not_sarcastic_good += 1
            elif predictions['sarcastic'] > predictions['notsarcastic'] and sentiment == 0:
                sarcastic_bad += 1
            else:
                not_sarcastic_bad += 1

    print("Sarcastic accuracy: " + str(round(sarcastic_good * 100 / (sarcastic_good + sarcastic_bad), 3)))
    print(
        "Not sarcastic accuracy: " + str(round(not_sarcastic_good * 100 / (not_sarcastic_good + not_sarcastic_bad), 3)))


if __name__ == '__main__':
    # ucitavanje data seta
    texts, sentiments = load_data()

    test_function()